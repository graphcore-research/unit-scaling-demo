"""Keras layers replacements with unit scaling."""

from typing import Optional, Tuple

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
        dtype: tf.DType = tf.float32,
        seed: Optional[int] = None,
    ):
        super().__init__(dtype=dtype)
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
        dtype: tf.DType = tf.float32,
        seed: Optional[int] = None,
    ):
        super().__init__(dtype=dtype)
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
        self,
        table_size: int,
        embeddings_size: int,
        dtype: tf.DType = tf.float32,
        seed: Optional[int] = None,
    ):
        super().__init__(dtype=dtype)
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


class LayerNormalization(layers.LayerNormalization):
    """A scaled variant of keras.layers.LayerNormalization."""

    def __init__(self, epsilon: float = 0.001, dtype: tf.DType = tf.float32):
        super().__init__(epsilon=epsilon, dtype=dtype)
        # Overwritten from base
        self.beta_initializer = keras.initializers.zeros()
        self.gamma_initializer = keras.initializers.ones()

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return ops.add_bias(
            ops.multiply_scale(self._normalize(inputs), self.gamma), self.beta
        )


class ResidualLayer(layers.ResidualLayer):
    """A scaled (interpolation) residual layer."""

    def __init__(
        self,
        body: keras.layers.Layer,
        norm_type: Optional[str],
        alpha: float,
        dtype: tf.DType = tf.float32,
    ):
        super().__init__(
            body,
            norm_type=norm_type,
            alpha=alpha,
            dtype=dtype,
            norm_cls=LayerNormalization,
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
        self.up = Dense(intermediate_size, dtype=self.dtype, seed=self.seeds[0])
        self.up.build(input_shape[:-1] + (hidden_size,))
        self.down = Dense(hidden_size, dtype=self.dtype, seed=self.seeds[1])
        self.down.build(input_shape[:-1] + (intermediate_size,))

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return self.down(keras.activations.relu(self.up(x)))  # type:ignore[misc]


class MultiHeadAttention(keras.layers.Layer):  # type:ignore[misc]
    """Scaled multi-head self attention a la Transformer.

    With causal masking.

    With relative-positional embeddings a la Transformer XL.
    """

    # pylint:disable=too-many-instance-attributes
    # pylint:disable=R0801

    def __init__(
        self,
        heads: int,
        head_size: int,
        frequencies: int,
        max_period: int,
        dtype: tf.DType = tf.float32,
        seeds: Optional[Tuple[int, int, int]] = None,
    ):
        super().__init__(dtype=dtype)
        self.heads = heads
        self.head_size = head_size
        self.frequencies = frequencies
        self.max_period = max_period
        self.seeds = (None, None, None) if seeds is None else seeds
        self.qkv: tf.Variable = None
        self.q_bias: tf.Variable = None
        self.positional: tf.Variable = None
        self.out: keras.layers.Layer = None

    def build(self, input_shape: tf.TensorShape) -> None:
        super().build(input_shape)
        input_size = input_shape[-1]
        self.qkv = self.add_weight(
            name="qkv",
            shape=(input_size, 3, self.heads, self.head_size),
            initializer=initializers.uniform(self.seeds[0]),
        )
        self.q_bias = self.add_weight(
            name="q_bias",
            shape=(self.heads, self.head_size),
            initializer=keras.initializers.zeros(),
        )
        self.positional = self.add_weight(
            name="positional",
            shape=(self.frequencies, self.heads, self.head_size),
            initializer=initializers.uniform(self.seeds[1]),
        )
        self.out = Dense(input_size, dtype=self.dtype, seed=self.seeds[2])
        self.out.build(input_shape[:-1] + (self.heads * self.head_size,))

    def _positional_weights(self, query: tf.Tensor) -> tf.Tensor:
        sequence_length = query.shape[-2]
        sins = tf.constant(
            np.sqrt(2)
            * layers.sinusoid_embedding(
                sequence_length, self.frequencies, self.max_period
            ),
            dtype=query.dtype,
        )
        embeddings = tf.einsum(
            "sf,fnh->nsh",
            sins,
            ops.scaling(
                forward=self.frequencies**-0.5, backward=sequence_length**-0.5
            )(self.positional),
        )
        scores = tf.einsum("bnqh,nvh->bnqv", query, embeddings) * self.head_size**-0.5
        return layers.relative_causal_reshape(scores)

    def call(self, input: tf.Tensor) -> tf.Tensor:
        # pylint:disable=invalid-name
        batch_size, sequence_length, input_size = input.shape
        q, k, v = tf.unstack(
            tf.einsum(
                "bsx,xAnh -> Abnsh",
                input,
                ops.scaling(
                    forward=(3 * input_size * self.head_size * self.heads) ** -0.25,
                    backward=(batch_size * sequence_length) ** -0.5,
                )(self.qkv),
            )
        )
        q += ops.scaling(backward=(batch_size * sequence_length) ** -0.5)(
            self.q_bias[:, tf.newaxis, :]
        )
        a = tf.einsum("bnqh,bnkh->bnqk", q, k) * self.head_size**-0.5
        a += self._positional_weights(q)
        a = layers.causal_mask(a)
        a = tf.nn.softmax(a, axis=-1)
        o = tf.einsum("bnqk,bnkh->bqnh", a, v)
        return self.out(tf.reshape(o, o.shape[:-2] + (self.head_size * self.heads,)))


class RecurrentHighwayCell(keras.layers.Layer):  # type:ignore[misc]
    """Scaled recurrent highway cell from https://arxiv.org/abs/1607.03474."""

    # pylint:disable=R0801

    def __init__(
        self,
        hidden_size: int,
        rebias: float,
        dtype: tf.DType = tf.float32,
        seed: Optional[int] = None,
    ):
        super().__init__(name=type(self).__name__, dtype=dtype)
        self.hidden_size = hidden_size
        self.carry_rebias = rebias
        self.update_rebias = -rebias
        self.seed = seed
        self.gates: tf.Variable = None
        self.gates_bias: tf.Variable = None

    def build(self, input_shape: tf.TensorShape) -> None:
        super().build(input_shape)
        self.gates = self.add_weight(
            "gates",
            shape=(2, input_shape[-1] + self.hidden_size, self.hidden_size),
            initializer=initializers.uniform(seed=self.seed),
        )
        self.gates_bias = self.add_weight(
            "gates_bias",
            shape=(2, self.hidden_size),
            initializer=keras.initializers.zeros(),
        )

    def call(
        self, input: tf.Tensor, hidden: tf.Tensor, sequence_length: int
    ) -> tf.Tensor:
        batch_size = input.shape[0] * sequence_length
        gates_scale = (
            2 * (input.shape[1] + self.hidden_size) * self.hidden_size
        ) ** -0.25
        gate_outputs = tf.concat([input, hidden], axis=1) @ ops.scaling(
            forward=gates_scale, backward=batch_size**-0.5
        )(self.gates)
        gate_outputs += ops.scaling(backward=batch_size**-0.5)(
            self.gates_bias[:, tf.newaxis]
        )
        transform, update = tf.unstack(gate_outputs)
        update = tf.sigmoid(update + self.update_rebias)
        return (1 - update) * hidden + update * tf.tanh(transform)


class RNN(layers.RNN):
    """A scaled, basic unidirectional RNN."""

    def call(self, input: tf.Tensor) -> tf.Tensor:
        batch_size, sequence_length, _ = input.shape
        # Note: sbh = (sequence, batch, hidden)
        input_sbh = tf.transpose(input, (1, 0, 2))
        initial_hidden = tf.tile(
            ops.scaling(backward=batch_size**-0.5)(self.initial_hidden[tf.newaxis]),
            (batch_size, 1),
        )
        output_sbh = tf.scan(
            lambda hidden, input: self.cell(
                input, hidden, sequence_length=sequence_length
            ),
            input_sbh,
            initializer=initial_hidden,
        )
        return tf.transpose(output_sbh, (1, 0, 2))
