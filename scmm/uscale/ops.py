"""Unit scaling - basic, portable operations."""

from typing import Callable, Optional, Tuple

import numpy as np
import tensorflow as tf


def scaling(
    forward: Optional[float] = None, backward: Optional[float] = None
) -> Callable[[tf.Tensor], tf.Tensor]:
    """Perform arbitary *seperate* scaling in the forward and backward passes.

    E.g.

        y = scaling(forward=2, backward=3)(x)

    """

    @tf.custom_gradient  # type:ignore[misc]
    def operation(input: tf.Tensor) -> tf.Tensor:  # pylint:disable=redefined-builtin
        def grad(upstream: tf.Tensor) -> tf.Tensor:
            grad_input = upstream
            if backward is not None:
                if isinstance(upstream, tf.IndexedSlices):
                    grad_input = tf.IndexedSlices(
                        values=upstream.values * backward,
                        indices=upstream.indices,
                        dense_shape=upstream.dense_shape,
                    )
                else:
                    grad_input = grad_input * backward
            return grad_input

        output = input
        if forward is not None:
            output = output * forward

        return output, grad

    return operation  # type:ignore[no-any-return]


def pointwise(
    inputs: tf.Tensor, weights: tf.Tensor, scale_for: str = "both"
) -> tf.Tensor:
    """A scaled pointwise transformation.

    inputs -- activations, will receive consistent gradients between forward & backward passes

    weights -- will receive scaled gradients

    scale_for -- "forward" | "backward" | "both" -- how should the forward/backward-inputs
                 pass scale be chosen?

                "forward" -- preserve variance in the forward pass
                "backward" -- preserve variance in the backward pass
                "both" -- trade off forward and backward pass variance
    """
    assert len(weights.shape) == 2, "pointwise requires 2D rhs `weights`"

    input_size, output_size = weights.shape
    # Note "both" is different from Glorot's sqrt(2 / (input_size + output_size)), as this
    # should preserves scale better after boom_down(boom_up(x))
    forward_scale = dict(
        forward=input_size**-0.5,
        backward=output_size**-0.5,
        both=(input_size * output_size) ** -0.25,
    )[scale_for]
    backward_scale = np.prod(inputs.shape[:-1]) ** -0.5

    return inputs @ scaling(forward=forward_scale, backward=backward_scale)(weights)


def conv1d(
    input: tf.Tensor,  # pylint:disable=redefined-builtin
    filters: tf.Tensor,
    padding: str,
) -> tf.Tensor:
    """Scaling version of tf.nn.conv1d."""
    # pylint:disable=too-many-locals
    *batch_shape, input_length, input_size = input.shape
    filter_width, filter_input_size, output_size = filters.shape

    output_length = dict(
        SAME=input_length,
        VALID=input_length + 1 - filter_width,
    )[padding]
    n_groups = input_size // filter_input_size
    batch_size = np.prod(batch_shape)

    forward_contraction = filter_width * input_size // n_groups
    backward_contraction = filter_width * output_size // n_groups
    forward_scale = (forward_contraction * backward_contraction) ** -0.25
    backward_scale = (output_length * batch_size) ** -0.5

    return tf.nn.conv1d(
        input,
        scaling(forward=forward_scale, backward=backward_scale)(filters),
        stride=1,
        padding=padding,
    )


def add_bias(features: tf.Tensor, bias: tf.Tensor) -> tf.Tensor:
    """Add a bias (which should be zero-initialized), with a scaled backward pass."""
    assert len(bias.shape) == 1, "bias should be 1D"
    batch_size = np.prod(features.shape[:-1])
    return features + scaling(backward=batch_size**-0.5)(bias)


def multiply_scale(features: tf.Tensor, scale: tf.Tensor) -> tf.Tensor:
    """Multiply by a scale tensor (which should be unit-initialized), with a scaled backward pass."""
    assert len(scale.shape) == 1, "scale should be 1D"
    batch_size = np.prod(features.shape[:-1])
    return features * scaling(backward=batch_size**-0.5)(scale)


def batched_gather(tables: tf.Tensor, indices: tf.Tensor) -> tf.Tensor:
    """Simulate tf.gather(tables, indices, batch_dims=indices.ndim).

    Better compilation on IPU vs `tf.gather(logp, ids, batch_dims=2)`
    """
    assert len(tables.shape) == len(indices.shape) + 1
    offsets = (
        np.arange(np.prod(indices.shape)).reshape(indices.shape) * tables.shape[-1]
    )
    values = tf.gather(tf.reshape(tables, (-1,)), tf.reshape(indices + offsets, (-1,)))
    return tf.reshape(values, indices.shape)


def softmax_cross_entropy(
    scores: tf.Tensor, ids: tf.Tensor, mask: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Compute masked softmax cross entropy loss.

    Note that we abandon unit scaling in the forward pass, since this is
    designed as a final loss term.

    returns -- (average_loss, n_items)
    """
    assert mask.shape == ids.shape, "mask should match target ids"
    logp = tf.nn.log_softmax(scores, axis=-1)
    # Better compilation on IPU vs `tf.gather(logp, ids, batch_dims=2)`
    target_logp = batched_gather(logp, ids)
    total_loss = tf.reduce_sum(tf.cast(mask, target_logp.dtype) * -target_logp)
    n_ids = tf.reduce_sum(tf.cast(mask, tf.int32))
    n_classes = scores.shape[-1]
    loss = scaling(backward=np.prod(mask.shape) * n_classes / np.sqrt(n_classes - 1))(
        total_loss / tf.cast(n_ids, total_loss.dtype)
    )
    return loss, n_ids
