"""Keras layers replacements with unit scaling."""

from typing import Optional

import numpy as np
import tensorflow as tf
from tensorflow import keras

from .. import layers
from . import ops


class initializers:  # pylint:disable=invalid-name
    """Unit-variance initializers."""

    @staticmethod
    def uniform(seed: Optional[int]) -> keras.initializers.Initializer:
        """Uniform distribution (symmetric about 0)."""
        return keras.initializers.RandomUniform(-np.sqrt(3), np.sqrt(3), seed=seed)

    @staticmethod
    def normal(seed: Optional[int]) -> keras.initializers.Initializer:
        """Standard normal distribution."""
        return keras.initializers.RandomNormal(stddev=1, seed=seed)


class Dense(keras.layers.Layer):  # type:ignore[misc]
    """A scaled (and more restrictive) version of keras.layers.Dense."""

    def __init__(
        self,
        units: int,
        activation: Optional[str] = None,
        scale_for: str = "both",
        seed: Optional[int] = None,
    ):
        super().__init__(self)
        self.units = units
        self.scale_for = scale_for
        self.kernel: tf.Variable = None
        self.kernel_initializer = initializers.uniform(seed)
        self.bias: tf.Variable = None
        self.bias_initializer = keras.initializers.zeros()
        self.activation = keras.activations.get(activation)

    def build(self, input_shape: tf.TensorShape) -> None:
        super().build(input_shape)
        self.kernel = self.add_weight(
            "kernel",
            shape=(input_shape[-1], self.units),
            initializer=self.kernel_initializer,
        )
        self.bias = self.add_weight(
            "bias",
            shape=self.units,
            initializer=self.bias_initializer,
        )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return self.activation(
            ops.add_bias(
                ops.pointwise(inputs, self.kernel, scale_for=self.scale_for), self.bias
            )
        )


class CausalConv1D(keras.layers.Layer):  # type:ignore[misc]
    """A scaled causal 1D convolution."""

    # pylint:disable=too-many-instance-attributes

    def __init__(
        self,
        filters: int,
        kernel_size: int,
        groups: Optional[int] = None,
        activation: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.groups = groups or 1
        if filters % self.groups != 0:
            raise ValueError(
                f"Filters ({filters}) must be evenly divisible by groups ({self.groups})"
            )
        self.kernel: tf.Variable = None
        self.kernel_initializer = initializers.uniform(seed)
        self.bias: tf.Variable = None
        self.bias_initializer = keras.initializers.zeros()
        self.activation = keras.activations.get(activation)

    def build(self, input_shape: tf.TensorShape) -> None:
        super().build(input_shape)
        input_features = input_shape[-1]
        if input_features % self.groups != 0:
            raise ValueError(
                f"Input feature size ({input_features}) must be evenly divisible"
                f" by groups ({self.groups})"
            )
        self.kernel = self.add_weight(
            "kernel",
            shape=(self.kernel_size, input_shape[-1] // self.groups, self.filters),
            initializer=self.kernel_initializer,
        )
        self.bias = self.add_weight(
            "bias", shape=self.filters, initializer=self.bias_initializer
        )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        padded = tf.pad(inputs, [(0, 0), (self.kernel_size - 1, 0), (0, 0)])
        return self.activation(
            ops.add_bias(ops.conv1d(padded, self.kernel, padding="VALID"), self.bias)
        )


class Embedding(keras.layers.Layer):  # type:ignore[misc]
    """A scaled variant of keras.layers.Embedding."""

    def __init__(
        self, table_size: int, embeddings_size: int, seed: Optional[int] = None
    ):
        super().__init__(self)
        self.table_size = table_size
        self.embeddings_size = embeddings_size
        self.embeddings: tf.Variable = None
        self.embeddings_initializer = keras.initializers.RandomUniform(
            -np.sqrt(3), np.sqrt(3), seed=seed
        )

    def build(self, input_shape: tf.TensorShape) -> None:
        super().build(input_shape)
        self.embeddings = self.add_weight(
            "embeddings",
            shape=(self.table_size, self.embeddings_size),
            initializer=self.embeddings_initializer,
        )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        # We don't need to worry about inputs scaling, as it is non-differentiable
        batch_size = np.prod(inputs.shape)
        return tf.gather(
            ops.scaling(backward=(self.table_size / batch_size) ** 0.5)(
                self.embeddings
            ),
            inputs,
        )


class LayerNormalization(keras.layers.Layer):  # type:ignore[misc]
    """A scaled variant of keras.layers.LayerNormalization."""

    def __init__(self, epsilon: float = 0.001):
        super().__init__()
        self.epsilon = epsilon
        self.beta: tf.Variable = None
        self.beta_initializer = keras.initializers.zeros()
        self.gamma: tf.Variable = None
        self.gamma_initializer = keras.initializers.ones()

    def build(self, input_shape: tf.TensorShape) -> None:
        super().build(input_shape)
        self.beta = self.add_weight(
            "beta", shape=input_shape[-1], initializer=self.beta_initializer
        )
        self.gamma = self.add_weight(
            "gamma", shape=input_shape[-1], initializer=self.gamma_initializer
        )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        assert inputs.dtype != tf.float16, "this implementation is not float16-safe"
        z = inputs - tf.reduce_mean(inputs, axis=-1, keepdims=True)
        normed = z / tf.sqrt(tf.reduce_mean(z**2, axis=-1, keepdims=True))
        return ops.add_bias(ops.multiply_scale(normed, self.gamma), self.beta)


class ResidualLayer(layers.ResidualLayer):
    """A scaled (interpolation) residual layer."""

    def __init__(
        self,
        body: keras.layers.Layer,
        norm_type: Optional[str],
        alpha: float,
    ):
        super().__init__(
            body, norm_type=norm_type, alpha=alpha, norm_cls=LayerNormalization
        )

    def call(self, x: tf.Tensor) -> tf.Tensor:
        assert (
            self.alpha is not None
        ), "cannot preserve variance with plain residual (please set 'alpha')"

        residual_scale = self.alpha**0.5
        branch = ops.scaling(backward=residual_scale)(x)
        if self.norm_type == "pre":
            branch = self.norm(branch)

        branch = self.body(branch)

        y = (1 - self.alpha) ** 0.5 * x + ops.scaling(forward=residual_scale)(branch)

        if self.norm_type == "post":
            y = self.norm(y)
        return y


class FFNLayer(layers.FFNLayer):
    """A scaled FFN layer."""

    def build(self, input_shape: tf.TensorShape) -> None:
        super().build(input_shape)
        hidden_size = input_shape[-1]
        intermediate_size = int(self.multiple * hidden_size)
        self.up = Dense(intermediate_size, seed=self.seeds[0])
        self.up.build(input_shape[:-1] + (hidden_size,))
        self.down = Dense(hidden_size, seed=self.seeds[1])
        self.down.build(input_shape[:-1] + (intermediate_size,))

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return self.down(keras.activations.relu(self.up(x)))  # type:ignore[misc]
