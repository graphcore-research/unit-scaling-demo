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
    def operation(input: tf.Tensor) -> tf.Tensor:
        def grad(upstream: tf.Tensor) -> tf.Tensor:
            grad_input = upstream
            if backward is not None:
                if isinstance(grad_input, tf.IndexedSlices):
                    grad_input = tf.IndexedSlices(
                        values=grad_input.values
                        * tf.cast(backward, grad_input.values.dtype),
                        indices=upstream.indices,
                        dense_shape=upstream.dense_shape,
                    )
                else:
                    grad_input = grad_input * tf.cast(backward, grad_input.dtype)
            return grad_input

        output = input
        if forward is not None:
            output = output * tf.cast(forward, output.dtype)

        return output, grad

    return operation  # type:ignore[no-any-return]


def pointwise(
    inputs: tf.Tensor, weights: tf.Tensor, scale_for: str = "both"
) -> tf.Tensor:
    """A scaled pointwise transformation.

    inputs -- activations, will receive consistent gradients between forward & backward passes

    weights -- will receive scaled gradients

    scale_for -- how should the forward/backward-inputs pass scale be chosen?

                "forward" -- preserve variance in the forward pass
                "backward" -- preserve variance in the backward pass
                "both" -- trade off forward and backward pass variance
                "both_arithmetic" -- ditto, using arithmetic mean
                "both_min" - ditto, using the minimum scale between forward and backward
                "separate" -- separate scaling factors in the forward and backward-inputs passes
                              WARNING - when using skip connections, this may cause inconsistent
                              gradients.
    """
    assert len(weights.shape) == 2, "pointwise requires 2D rhs `weights`"

    input_size, output_size = weights.shape
    backward_weights_scale = np.prod(inputs.shape[:-1]) ** -0.5

    if scale_for == "separate":
        return scaling(forward=input_size**-0.5)(
            scaling(backward=output_size**-0.5)(inputs)
            @ scaling(backward=backward_weights_scale)(weights)
        )

    # Note "both" is different from Glorot's sqrt(2 / (input_size + output_size)), as this
    # should preserves scale better after boom_down(boom_up(x))
    forward_scale = dict(
        forward=input_size**-0.5,
        backward=output_size**-0.5,
        both=(input_size * output_size) ** -0.25,
        both_arithmetic=((input_size + output_size) / 2) ** -0.5,
        both_min=max(input_size, output_size) ** -0.5,
    )[scale_for]

    return inputs @ scaling(forward=forward_scale, backward=backward_weights_scale)(
        weights
    )


def conv1d(
    input: tf.Tensor, filters: tf.Tensor, stride: int = 1, padding: str = "SAME"
) -> tf.Tensor:
    """Scaling version of tf.nn.conv1d."""
    # pylint:disable=too-many-locals
    batch_size, input_length, channels_in = input.shape
    filter_width, filter_channels_in, channels_out = filters.shape

    # See https://www.tensorflow.org/api_docs/python/tf/nn#notes_on_padding_2
    output_length = dict(
        SAME=np.ceil(input_length / stride),
        VALID=np.ceil((input_length + 1 - filter_width) / stride),
    )[padding]
    n_groups = channels_in // filter_channels_in

    forward_contraction = filter_width * channels_in // n_groups
    backward_contraction = filter_width / stride * channels_out // n_groups
    forward_scale = (forward_contraction * backward_contraction) ** -0.25
    backward_scale = (output_length * batch_size) ** -0.5

    return tf.nn.conv1d(
        input,
        scaling(forward=forward_scale, backward=backward_scale)(filters),
        stride=stride,
        padding=padding,
    )


def conv2d(
    input: tf.Tensor, filters: tf.Tensor, strides: int = 1, padding: str = "SAME"
) -> tf.Tensor:
    """Scaling version of tf.nn.conv2d."""
    # pylint:disable=too-many-locals
    batch_size, height, width, channels_in = input.shape
    kernel_height, kernel_width, filter_channels_in, channels_out = filters.shape

    # See https://www.tensorflow.org/api_docs/python/tf/nn#notes_on_padding_2
    output_area = dict(
        SAME=np.ceil(height / strides) * np.ceil(width / strides),
        VALID=np.ceil((height + 1 - kernel_height) / strides)
        * np.ceil((width + 1 - kernel_width) / strides),
    )[padding]
    n_groups = channels_in // filter_channels_in

    forward_contraction = kernel_height * kernel_width * channels_in // n_groups
    backward_contraction = (
        (kernel_height / strides) * (kernel_width / strides) * channels_out // n_groups
    )
    forward_scale = (forward_contraction * backward_contraction) ** -0.25
    backward_scale = (output_area * batch_size) ** -0.5

    return tf.nn.conv2d(
        input,
        scaling(forward=forward_scale, backward=backward_scale)(filters),
        strides=strides,
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
    # Implemented here and in scmm.layers to avoid circular dependency issues
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

    returns -- (average_loss, n_items) -- average_loss always in fp32
    """
    assert mask.shape == ids.shape, "mask should match target ids"
    # Use float32 for local computation - keeping things simple
    logp = tf.nn.log_softmax(tf.cast(scores, tf.float32), axis=-1)
    target_logp = batched_gather(logp, ids)
    total_loss = tf.reduce_sum(tf.cast(mask, target_logp.dtype) * -target_logp)
    n_ids = tf.reduce_sum(tf.cast(mask, tf.int32))
    n_classes = scores.shape[-1]
    loss = scaling(backward=np.prod(mask.shape) * n_classes / np.sqrt(n_classes - 1))(
        total_loss / tf.cast(n_ids, total_loss.dtype)
    )
    return loss, n_ids
