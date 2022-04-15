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


class PreNormResidualLayer(keras.layers.Layer):  # type:ignore[misc]
    """A PreNorm residual layer (https://aclanthology.org/P18-1008/)."""

    def __init__(self, body: keras.layers.Layer):
        super().__init__()
        self.norm = keras.layers.LayerNormalization()
        self.body = body

    def build(self, input_shape: tf.TensorShape) -> None:
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
        if self.up is not None:
            return  # build() can be called multiple times
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
