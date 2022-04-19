import functools as ft
from typing import Any, Callable, Dict, Optional, Tuple

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


####################
# Ops


def test_scaling():
    with tf.GradientTape() as tape:
        x = tf.range(3, dtype=tf.float32)
        tape.watch(x)
        y = uscale.scaling(x, forward=2, backward=3)
    grad_x = tape.gradient(y, x, tf.ones_like(y))
    np.testing.assert_allclose(y, [0, 2, 4])
    np.testing.assert_allclose(grad_x, [3, 3, 3])


def test_scaling_indexed_slices():
    with tf.GradientTape() as tape:
        table = tf.range(10, dtype=tf.float32)
        tape.watch(table)
        y = tf.gather(uscale.scaling(table, backward=5), tf.constant([1, 1, 2]))
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


def assert_all_unit_scale(values: Dict[str, np.ndarray], tol: float) -> None:
    for k in values:
        assert_unit_scale(values[k], tol=tol, err_msg=f"for {k}")


def check_op(
    scaled: Callable[..., tf.Tensor],
    reference: Callable[..., tf.Tensor],
    seed: int,
    args: Dict[str, Tuple[int, ...]],
    extra_args: Optional[Dict[str, Any]] = None,
) -> None:
    random = np.random.Generator(np.random.PCG64(seed))
    inputs = {
        k: tf.constant(random.normal(size=shape).astype(np.float32))
        for k, shape in args.items()
    }
    with tf.GradientTape() as tape:
        tape.watch(inputs.values())
        scaled_out = scaled(**inputs, **(extra_args or {}))
    output_grad = random.normal(size=scaled_out.shape).astype(np.float32)
    scaled_grad = tape.gradient(scaled_out, inputs, output_grad)
    with tf.GradientTape() as tape:
        tape.watch(inputs.values())
        reference_out = reference(**inputs, **(extra_args or {}))
    reference_grad = tape.gradient(reference_out, inputs, output_grad)

    assert_unit_scale(scaled_out, tol=0.1)
    for arg in scaled_grad:
        assert_unit_scale(scaled_grad[arg], tol=0.1, err_msg=f"for grad {arg}")

    assert_scaled_allclose(scaled_out, reference_out, atol=1e-3)
    for arg in scaled_grad:
        assert_scaled_allclose(
            scaled_grad[arg], reference_grad[arg], atol=1e-3, err_msg=f"for grad {arg}"
        )


def test_op_relu():
    check_op(uscale.relu, tf.nn.relu, seed=100, args=dict(features=(1000,)))


def test_op_matmul():
    check_op(uscale.matmul, tf.matmul, seed=200, args=dict(a=(100, 200), b=(200, 300)))
    check_op(
        uscale.matmul, tf.matmul, seed=300, args=dict(a=(3, 99, 150), b=(150, 130))
    )


def test_op_conv1d():
    for padding in ["SAME", "VALID"]:
        check_op(
            uscale.conv1d,
            ft.partial(tf.nn.conv1d, stride=1),
            seed=400,
            args=dict(input=(50, 17, 64), filters=(5, 64, 128)),
            extra_args=dict(padding=padding),
        )


def test_op_softmax_cross_entropy():
    random = np.random.Generator(np.random.PCG64(seed=1000))
    scores = tf.constant(random.normal(size=(10, 5)))
    ids = tf.constant(random.integers(scores.shape[1], size=scores.shape[0]))
    with tf.GradientTape() as tape:
        tape.watch(scores)
        loss, n_ids = uscale.softmax_cross_entropy(scores, ids, tf.ones_like(ids))
    assert int(n_ids) == scores.shape[0]
    assert 0 < float(loss) < 2 * np.log(5)
    grad_scores = tape.gradient(loss, scores)
    assert_unit_scale(grad_scores, 0.1)


####################
# Layers


def test_initializers():
    assert_unit_scale(uscale.Initializers.uniform(100)((1000,)), 0.05)
    assert_unit_scale(uscale.Initializers.normal(200)((1000,)), 0.05)


def output_and_gradients(
    layer: Callable[..., tf.Tensor], input_shape: Tuple[int, ...], seed: int
) -> Dict[str, np.ndarray]:
    random = np.random.Generator(np.random.PCG64(seed))
    inputs = tf.constant(random.normal(size=input_shape).astype(np.float32))
    with tf.GradientTape() as tape:
        tape.watch(inputs)
        outputs = layer(inputs)
    output_gradients = random.normal(size=outputs.shape).astype(np.float32)
    gradient_tensors = {f"grad_{k}": v for k, v in utility.named_weights(layer)}
    gradient_tensors["grad_inputs"] = inputs
    gradients = tape.gradient(outputs, gradient_tensors, output_gradients)
    return dict(
        inputs=inputs, outputs=outputs, output_gradients=output_gradients, **gradients
    )


def test_layer_dense():
    layer = uscale.Dense(200, seed=123)
    out = output_and_gradients(layer, (100, 150), seed=456)
    assert out["outputs"].shape == (100, 200)
    assert_all_unit_scale(out, tol=0.1)


def test_layer_causalconv1d():
    layer = uscale.CausalConv1D(filters=200, kernel_size=7, seed=321)
    out = output_and_gradients(layer, (30, 19, 100), seed=654)
    assert out["outputs"].shape == (30, 19, 200)
    assert_all_unit_scale(out, tol=0.1)

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
    assert_unit_scale(outputs, tol=0.1)

    grad_embeddings = tape.gradient(
        outputs, layer.embeddings, random.normal(size=outputs.shape).astype(np.float32)
    )
    assert_unit_scale(grad_embeddings, tol=0.1)
