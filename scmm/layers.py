"""General purpose layers and functions."""

from typing import Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras


def batched_gather(tables: tf.Tensor, indices: tf.Tensor) -> tf.Tensor:
    """Simulate tf.gather(tables, indices, batch_dims=indices.ndim)."""
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

    returns -- (average_loss, n_items)
    """
    logp = tf.nn.log_softmax(scores)
    # Better compilation on IPU vs `tf.gather(logp, ids, batch_dims=2)`
    target_logp = batched_gather(logp, ids)
    total_loss = tf.reduce_sum(tf.cast(mask, target_logp.dtype) * -target_logp)
    n_ids = tf.reduce_sum(tf.cast(mask, tf.int32))
    return total_loss / tf.cast(n_ids, total_loss.dtype), n_ids


class PreNormResidualLayer(keras.layers.Layer):  # type:ignore[misc]
    """A PreNorm residual layer (https://aclanthology.org/P18-1008/)."""

    def __init__(self, body: keras.layers.Layer):
        super().__init__()
        self.norm = keras.layers.LayerNormalization()
        self.body = body

    def build(self, input_shape: tf.TensorShape) -> None:
        super().build(input_shape)
        self.norm.build(input_shape)
        self.body.build(input_shape)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return x + self.body(self.norm(x))


class FFNLayer(keras.layers.Layer):  # type:ignore[misc]
    """A pointwise expansion FFN layer (a la Transformer, https://arxiv.org/abs/1706.03762)."""

    def __init__(self, multiple: float, seeds: Optional[Tuple[int, int]] = None):
        super().__init__()
        self.multiple = multiple
        self.seeds = seeds if seeds else (None, None)
        self.up: Optional[keras.layers.Layer] = None  # pylint:disable=invalid-name
        self.down: Optional[keras.layers.Layer] = None

    def build(self, input_shape: tf.TensorShape) -> None:
        super().build(input_shape)
        hidden_size = input_shape[-1]
        intermediate_size = int(self.multiple * hidden_size)
        self.up = keras.layers.Dense(
            intermediate_size,
            kernel_initializer=keras.initializers.GlorotUniform(seed=self.seeds[0]),
        )
        self.up.build((hidden_size,))
        self.down = keras.layers.Dense(
            hidden_size,
            kernel_initializer=keras.initializers.GlorotUniform(seed=self.seeds[1]),
        )
        self.down.build((intermediate_size,))

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return self.down(tf.nn.relu(self.up(x)))  # type:ignore[misc]


class PadAndShiftLayer(keras.layers.Layer):  # type:ignore[misc]
    """Shifts sequence features one place to the right with a trainable padding vector."""

    def __init__(self) -> None:
        super().__init__()
        self.padding: tf.Variable = None

    def build(self, input_shape: tf.TensorShape) -> None:
        super().build(input_shape)
        if len(input_shape) != 3:
            raise ValueError(
                f"Input should be 3D (batch, sequence, feature), actual shape {input_shape}"
            )
        self.padding = self.add_weight(
            name="padding",
            shape=input_shape[-1],
            dtype=self.dtype,
            initializer=keras.initializers.zeros,
        )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        pad = tf.tile(self.padding[tf.newaxis, tf.newaxis], [inputs.shape[0], 1, 1])
        return tf.concat([pad, inputs[:, :-1, :]], axis=1)
