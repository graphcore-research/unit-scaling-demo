import functools as ft
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import pytest
import tensorflow as tf
from tensorflow import keras

from .. import uscale
from ..pedal import utility

####################
# Utility


def test_activation_tracker_example():
    # Sync with docstring
    layer = keras.layers.Dense(10)
    layer.build((None, 20))

    tracker = uscale.ActivationTracker(layer)

    with tf.GradientTape() as tape:
        loss = tf.reduce_sum(layer(tf.zeros((3, 20))))
    grads_and_vars = zip(
        tape.gradient(loss, layer.trainable_variables), layer.trainable_variables
    )
    tracker.log_gradients(grads_and_vars)  # only needed for weight gradients

    # Checks
    (trace,) = tracker.trace
    assert trace.name == ""
    assert trace.layer is layer
    assert trace.activation.shape == (1, 3, 10)
    assert trace.gradient.shape == (1, 3, 10)  # type:ignore[union-attr]

    weights = {t.name: t for t in trace.weights}
    assert weights["kernel"].activation.shape == (20, 10)
    assert weights["kernel"].gradient.shape == (1, 20, 10)  # type:ignore[union-attr]
    assert weights["bias"].activation.shape == (10,)
    assert weights["bias"].gradient.shape == (1, 10)  # type:ignore[union-attr]


def test_activation_tracker_multiple_outputs():
    class Duplicate(keras.layers.Layer):
        def call(self, x: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
            return x, x

    layer = Duplicate()
    uscale.ActivationTracker(layer)
    with pytest.raises(ValueError) as error:
        layer(tf.ones(3))
    assert "tuple" in str(error)


def test_activation_tracker_embedding_gradient():
    layer = keras.layers.Embedding(5, 8)
    layer.build((None,))

    tracker = uscale.ActivationTracker(layer)
    with tf.GradientTape() as tape:
        loss = tf.reduce_sum(layer(tf.constant([0, 3, 3])))
    grads_and_vars = zip(
        tape.gradient(loss, layer.trainable_variables), layer.trainable_variables
    )
    tracker.log_gradients(grads_and_vars)

    embedding_trace = tracker.trace[0].weights[0]
    assert embedding_trace.name == "embeddings"
    assert embedding_trace.gradient.shape == (1, 5, 8)  # type:ignore[union-attr]
    np.testing.assert_equal(
        embedding_trace.gradient[0, :, 0],  # type:ignore[index]
        [1, 0, 0, 2, 0],
    )


def test_printing(capsys):
    with tf.GradientTape() as tape:
        x = tf.constant([0, 1, 0, 1], dtype=tf.float32)
        tape.watch(x)
        y = tf.reduce_sum(uscale.printing("x")(x) ** 2)
    tape.gradient(y, x)
    assert capsys.readouterr().err == "x forward std 0.5\nx backward std 1.0\n"


####################
# Ops


def test_scaling():
    with tf.GradientTape() as tape:
        x = tf.range(3, dtype=tf.float32)
        tape.watch(x)
        y = uscale.scaling(forward=2, backward=3)(x)
    grad_x = tape.gradient(y, x, tf.ones_like(y))
    np.testing.assert_allclose(y, [0, 2, 4])
    np.testing.assert_allclose(grad_x, [3, 3, 3])


def test_scaling_indexed_slices():
    with tf.GradientTape() as tape:
        table = tf.range(10, dtype=tf.float32)
        tape.watch(table)
        y = tf.gather(uscale.scaling(backward=5)(table), tf.constant([1, 1, 2]))
    grad_table = tape.gradient(y, table, tf.ones_like(y))
    np.testing.assert_allclose(y, [1, 1, 2])
    np.testing.assert_allclose(grad_table.indices, [1, 1, 2])
    np.testing.assert_allclose(grad_table.values, [5, 5, 5])


def assert_scaled_allclose(
    actual: np.ndarray, desired: np.ndarray, atol: float, err_msg: str = ""
) -> None:
    np.testing.assert_allclose(
        actual / np.std(actual), desired / np.std(desired), atol=atol, err_msg=err_msg
    )


def assert_unit_scale(value: np.ndarray, tol: float, err_msg: str = "") -> None:
    np.testing.assert_allclose(np.std(value), 1, atol=tol, err_msg=err_msg)


def check_op(
    scaled: Callable[..., tf.Tensor],
    reference: Callable[..., tf.Tensor],
    seed: int,
    args: Dict[str, Union[np.ndarray, Tuple[int, ...]]],
    extra_args: Optional[Dict[str, Any]] = None,
    shifted: bool = False,
) -> Dict[str, tf.Tensor]:
    # pylint:disable=too-many-locals
    random = np.random.Generator(np.random.PCG64(seed))

    inputs = {}
    for key, value in args.items():
        if isinstance(value, np.ndarray):
            inputs[key] = tf.constant(value)
        else:
            inputs[key] = tf.constant(random.normal(size=value).astype(np.float32))

    with tf.GradientTape() as tape:
        tape.watch(inputs.values())
        scaled_out = scaled(**inputs, **(extra_args or {}))
    output_grad = random.normal(size=scaled_out.shape).astype(np.float32)
    scaled_grad = tape.gradient(scaled_out, inputs, output_grad)
    with tf.GradientTape() as tape:
        tape.watch(inputs.values())
        reference_out = reference(**inputs, **(extra_args or {}))
    reference_grad = tape.gradient(reference_out, inputs, output_grad)

    assert_scaled_allclose(
        scaled_out - shifted * np.mean(scaled_out),
        reference_out - shifted * np.mean(reference_out),
        atol=1e-3,
    )
    for arg in scaled_grad:
        assert_scaled_allclose(
            scaled_grad[arg], reference_grad[arg], atol=1e-3, err_msg=f"for grad {arg}"
        )

    return dict(
        out=scaled_out,
        grad=scaled_grad,
        reference_out=reference_out,
        reference_grad=reference_grad,
    )


def test_op_add_bias():
    out = check_op(
        uscale.add_bias,
        lambda features, bias: features + bias,
        seed=842,
        args=dict(features=(1000,), bias=np.zeros(1000, dtype=np.float32)),
    )
    assert_unit_scale(out["grad"]["bias"], tol=0.05)


def test_op_multiply_scale():
    out = check_op(
        uscale.multiply_scale,
        lambda features, scale: features * scale,
        seed=2345,
        args=dict(features=(1000,), scale=np.ones(1000, dtype=np.float32)),
    )
    assert_unit_scale(out["grad"]["scale"], tol=0.05)


def test_op_pointwise():
    out = check_op(
        uscale.pointwise,
        lambda inputs, weights: tf.matmul(  # pylint:disable=unnecessary-lambda
            inputs, weights
        ),
        seed=300,
        args=dict(inputs=(3, 99, 150), weights=(150, 110)),
    )
    np.testing.assert_allclose(
        np.std(out["out"]) * np.std(out["grad"]["inputs"]), 1, atol=0.05
    )
    assert_unit_scale(out["grad"]["weights"], tol=0.05)


def test_op_conv1d():
    for padding in ["SAME", "VALID"]:
        # We're a bit sloppy about padding when using "SAME"
        out = check_op(
            uscale.conv1d,
            ft.partial(tf.nn.conv1d, stride=1),
            seed=400,
            args=dict(input=(50, 27, 96), filters=(5, 96, 128)),
            extra_args=dict(padding=padding),
        )
        np.testing.assert_allclose(
            np.std(out["out"]) * np.std(out["grad"]["input"]), 1, atol=0.1
        )
        assert_unit_scale(out["grad"]["filters"], tol=0.1)


def test_op_softmax_cross_entropy():
    batch_size, sequence_length, n_classes = (10, 20, 5)
    random = np.random.Generator(np.random.PCG64(seed=1000))
    scores = tf.constant(random.normal(size=(batch_size, sequence_length, n_classes)))
    ids = tf.constant(random.integers(n_classes, size=(batch_size, sequence_length)))
    with tf.GradientTape() as tape:
        tape.watch(scores)
        loss, n_ids = uscale.softmax_cross_entropy(scores, ids, tf.ones_like(ids))
    assert int(n_ids) == batch_size * sequence_length
    assert 0 < float(loss) < 2 * np.log(n_classes)
    grad_scores = tape.gradient(loss, scores)
    assert_unit_scale(grad_scores, 0.1)


####################
# Layers


def test_initializers():
    assert_unit_scale(uscale.initializers.uniform(100)((1000,)), 0.05)
    assert_unit_scale(uscale.initializers.normal(200)((1000,)), 0.05)


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
    layer = uscale.Dense(200, seed=123)
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
    layer = uscale.CausalConv1D(filters=200, kernel_size=3, seed=321)
    out = output_and_gradients(layer, (30, 19, 100), seed=654)
    assert out["outputs"].shape == (30, 19, 200)
    np.testing.assert_allclose(
        np.std(out["outputs"]) * np.std(out["grad_inputs"]), 1, atol=0.1
    )
    assert_unit_scale(out["grad_kernel"], 0.05)
    assert_unit_scale(out["grad_bias"], 0.05)

    with pytest.raises(ValueError) as error:
        uscale.CausalConv1D(filters=16, kernel_size=5, groups=3)
    assert "Filters (16)" in str(error)
    assert "groups (3)" in str(error)

    layer = uscale.CausalConv1D(filters=15, kernel_size=5, groups=3)
    with pytest.raises(ValueError) as error:
        layer(tf.zeros((1, 10, 16)))
    assert "feature size (16)" in str(error)
    assert "groups (3)" in str(error)


def test_layer_embedding():
    random = np.random.Generator(np.random.PCG64(seed=200))
    layer = uscale.Embedding(table_size=40, embeddings_size=50, seed=100)
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
    layer = uscale.ResidualLayer(
        uscale.Dense(250, seed=4832), norm_type=norm_type, alpha=alpha
    )
    out = output_and_gradients(layer, (29, 19, 250), seed=2393)
    assert_unit_scale(out["outputs"], tol=0.1)
    assert_unit_scale(out["grad_inputs"], tol=0.1)
    for name, variable in utility.named_weights(layer):
        if variable.trainable:
            assert_unit_scale(out[f"grad_{name}"], tol=0.1, err_msg=f"for grad_{name}")


def test_layer_ffn():
    random = np.random.Generator(np.random.PCG64(seed=200))
    layer = uscale.FFNLayer(2, (48723, 7428))
    output = layer(random.normal(size=(5, 6, 7)))
    assert output.shape == (5, 6, 7)
    assert {k: v.shape for k, v in utility.named_weights(layer)} == {
        "up.kernel": (7, 14),
        "up.bias": (14,),
        "down.kernel": (14, 7),
        "down.bias": (7,),
    }
