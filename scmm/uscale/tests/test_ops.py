import functools
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import pytest
import tensorflow as tf

from ...tests import testing
from .. import ops


def test_scaling():
    with tf.GradientTape() as tape:
        x = tf.range(3, dtype=tf.float32)
        tape.watch(x)
        y = ops.scaling(forward=2, backward=3)(x)
    grad_x = tape.gradient(y, x, tf.ones_like(y))
    np.testing.assert_allclose(y, [0, 2, 4])
    np.testing.assert_allclose(grad_x, [3, 3, 3])


def test_scaling_indexed_slices():
    with tf.GradientTape() as tape:
        table = tf.range(10, dtype=tf.float32)
        tape.watch(table)
        y = tf.gather(ops.scaling(backward=5)(table), tf.constant([1, 1, 2]))
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


def test_add_bias():
    out = check_op(
        ops.add_bias,
        lambda features, bias: features + bias,
        seed=842,
        args=dict(features=(1000,), bias=np.zeros(1000, dtype=np.float32)),
    )
    testing.assert_unit_scale(out["grad"]["bias"], tol=0.05)


def test_multiply_scale():
    out = check_op(
        ops.multiply_scale,
        lambda features, scale: features * scale,
        seed=2345,
        args=dict(features=(1000,), scale=np.ones(1000, dtype=np.float32)),
    )
    testing.assert_unit_scale(out["grad"]["scale"], tol=0.05)


@pytest.mark.parametrize(
    "scale_for",
    ["forward", "backward", "both", "both_arithmetic", "both_min", "separate"],
)
def test_pointwise(scale_for):
    out = check_op(
        functools.partial(ops.pointwise, scale_for=scale_for),
        lambda inputs, weights: tf.matmul(  # pylint:disable=unnecessary-lambda
            inputs, weights
        ),
        seed=300,
        args=dict(inputs=(3, 99, 200), weights=(200, 110)),
    )
    testing.assert_unit_scale(out["grad"]["weights"], tol=0.05)
    std_out = np.std(out["out"])
    std_grad_inputs = np.std(out["grad"]["inputs"])
    if scale_for in {"forward", "separate"}:
        np.testing.assert_allclose(std_out, 1, atol=0.05)
    if scale_for in {"backward", "separate"}:
        np.testing.assert_allclose(std_grad_inputs, 1, atol=0.05)
    if scale_for == "both":
        np.testing.assert_allclose(std_out * std_grad_inputs, 1, atol=0.05)
    if scale_for == "both_min":
        print(std_out, std_grad_inputs)
        assert std_out < 1.05 and std_grad_inputs < 1.05
    if scale_for == "both_arithmetic":
        assert (std_out < 1) != (std_grad_inputs < 1), "signs should be swapped"


@pytest.mark.parametrize(
    ["padding", "stride"], [("SAME", 1), ("SAME", 2), ("VALID", 1), ("VALID", 2)]
)
def test_conv1d(padding, stride):
    # Note: we're a bit sloppy about padding when using "SAME"
    out = check_op(
        ops.conv1d,
        tf.nn.conv1d,
        seed=400,
        args=dict(input=(50, 27, 96), filters=(5, 96, 128)),
        extra_args=dict(stride=stride, padding=padding),
    )
    np.testing.assert_allclose(
        np.std(out["out"]) * np.std(out["grad"]["input"]), 1, atol=0.1
    )
    testing.assert_unit_scale(out["grad"]["filters"], tol=0.1)


@pytest.mark.parametrize(
    ["padding", "stride"], [("SAME", 1), ("SAME", 2), ("VALID", 1), ("VALID", 2)]
)
def test_conv2d(padding, stride):
    # Note: we're a bit sloppy about padding when using "SAME"
    out = check_op(
        ops.conv2d,
        tf.nn.conv2d,
        seed=500,
        args=dict(input=(2, 20, 30, 8), filters=(2, 3, 8, 12)),
        extra_args=dict(strides=stride, padding=padding),
    )
    np.testing.assert_allclose(
        np.std(out["out"]) * np.std(out["grad"]["input"]), 1, atol=0.1
    )
    testing.assert_unit_scale(out["grad"]["filters"], tol=0.1)


def test_batched_gather():
    tables = tf.reshape(tf.range(2 * 3 * 4), (2, 3, 4))
    indices = tf.constant([[0, 0, 3], [2, 2, 3]])
    np.testing.assert_equal(
        np.array(ops.batched_gather(tables, indices)),
        [[0 + 0, 4 + 0, 8 + 3], [12 + 2, 16 + 2, 20 + 3]],
    )


def test_softmax_cross_entropy():
    batch_size, sequence_length, n_classes = (10, 20, 5)
    random = np.random.Generator(np.random.PCG64(seed=1000))
    scores = tf.constant(random.normal(size=(batch_size, sequence_length, n_classes)))
    ids = tf.constant(random.integers(n_classes, size=(batch_size, sequence_length)))
    with tf.GradientTape() as tape:
        tape.watch(scores)
        loss, n_ids = ops.softmax_cross_entropy(scores, ids, tf.ones_like(ids))
    assert int(n_ids) == batch_size * sequence_length
    assert 0 < float(loss) < 2 * np.log(n_classes)
    grad_scores = tape.gradient(loss, scores)
    testing.assert_unit_scale(grad_scores, 0.1)
