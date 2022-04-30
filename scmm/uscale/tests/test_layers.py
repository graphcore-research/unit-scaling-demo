from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pytest
import tensorflow as tf

from ...pedal import utility
from .. import layers
from .test_ops import assert_unit_scale


def test_initializers():
    assert_unit_scale(layers.initializers.uniform(100)((1000,)), 0.05)
    assert_unit_scale(layers.initializers.normal(200)((1000,)), 0.05)


def output_and_gradients(
    layer: Callable[..., tf.Tensor], input_shape: Tuple[int, ...], seed: int
) -> Dict[str, np.ndarray]:
    random = np.random.Generator(np.random.PCG64(seed))
    inputs = tf.constant(random.normal(size=input_shape).astype(np.float32))
    with tf.GradientTape() as tape:
        tape.watch(inputs)
        outputs = layer(inputs)
    grad_outputs = random.normal(size=outputs.shape).astype(np.float32)
    gradient_tensors = {f"grad_{k}": v for k, v in utility.named_weights(layer)}
    gradient_tensors["grad_inputs"] = inputs
    gradients = tape.gradient(outputs, gradient_tensors, grad_outputs)
    return dict(inputs=inputs, outputs=outputs, grad_outputs=grad_outputs, **gradients)


def test_layer_dense():
    layer = layers.Dense(200, seed=123)
    out = output_and_gradients(layer, (150, 100), seed=456)
    assert out["outputs"].shape == (150, 200)
    np.testing.assert_allclose(
        np.std(out["outputs"]) * np.std(out["grad_inputs"]), 1, atol=0.01
    )
    assert_unit_scale(out["grad_kernel"], 0.01)
    assert_unit_scale(out["grad_bias"], 0.01)


def test_layer_causalconv1d():
    # Choose kernel size << sequence length to reduce the effect of padding
    # (which we don't account for in variance preservation)
    layer = layers.CausalConv1D(filters=200, kernel_size=3, seed=321)
    out = output_and_gradients(layer, (30, 19, 100), seed=654)
    assert out["outputs"].shape == (30, 19, 200)
    np.testing.assert_allclose(
        np.std(out["outputs"]) * np.std(out["grad_inputs"]), 1, atol=0.1
    )
    assert_unit_scale(out["grad_kernel"], 0.05)
    assert_unit_scale(out["grad_bias"], 0.05)

    with pytest.raises(ValueError) as error:
        layers.CausalConv1D(filters=16, kernel_size=5, groups=3)
    assert "Filters (16)" in str(error)
    assert "groups (3)" in str(error)

    layer = layers.CausalConv1D(filters=15, kernel_size=5, groups=3)
    with pytest.raises(ValueError) as error:
        layer(tf.zeros((1, 10, 16)))
    assert "feature size (16)" in str(error)
    assert "groups (3)" in str(error)


def test_layer_embedding():
    random = np.random.Generator(np.random.PCG64(seed=200))
    layer = layers.Embedding(table_size=40, embeddings_size=50, seed=100)
    with tf.GradientTape() as tape:
        outputs = layer(random.integers(layer.table_size, size=(8, 16)))
    assert outputs.shape == (8, 16, 50)
    assert_unit_scale(outputs, tol=0.1)

    grad_embeddings = tape.gradient(
        outputs, layer.embeddings, random.normal(size=outputs.shape).astype(np.float32)
    )
    grad_embeddings = tf.math.unsorted_segment_sum(
        grad_embeddings.values, grad_embeddings.indices, grad_embeddings.shape[0]
    )
    assert_unit_scale(grad_embeddings, tol=0.1)


@pytest.mark.parametrize(
    ["norm_type", "alpha"], [("pre", 0.1), ("post", 0.1), (None, 0.1), (None, 0.9)]
)
def test_layer_residual(
    norm_type: Optional[str], alpha: float
):  # also tests LayerNormalization
    layer = layers.ResidualLayer(
        layers.Dense(250, seed=4832), norm_type=norm_type, alpha=alpha
    )
    out = output_and_gradients(layer, (29, 19, 250), seed=2393)
    assert_unit_scale(out["outputs"], tol=0.1)
    assert_unit_scale(out["grad_inputs"], tol=0.1)
    for name, variable in utility.named_weights(layer):
        if variable.trainable:
            assert_unit_scale(out[f"grad_{name}"], tol=0.1, err_msg=f"for grad_{name}")


def test_layer_ffn():
    random = np.random.Generator(np.random.PCG64(seed=200))
    layer = layers.FFNLayer(2, (48723, 7428))
    output = layer(random.normal(size=(5, 6, 7)))
    assert output.shape == (5, 6, 7)
    assert {k: v.shape for k, v in utility.named_weights(layer)} == {
        "up.kernel": (7, 14),
        "up.bias": (14,),
        "down.kernel": (14, 7),
        "down.bias": (7,),
    }
