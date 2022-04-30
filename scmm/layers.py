"""General purpose layers and functions."""

from typing import Iterable, List, Optional, Tuple, Type

import numpy as np
import tensorflow as tf
from tensorflow import keras


def batched_gather(tables: tf.Tensor, indices: tf.Tensor) -> tf.Tensor:
    """Simulate tf.gather(tables, indices, batch_dims=indices.ndim).

    Better compilation on IPU vs `tf.gather(logp, ids, batch_dims=2)`
    """
    # pylint:disable=R0801
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


class ResidualLayer(keras.layers.Layer):  # type:ignore[misc]
    """A residual layer, supporting PreNorm, PostNorm, NoNorm & interpolation.

    norm_type -- None | "pre" | "post"

    alpha -- None | <float>  -- interpolation constant, higher to incorporate
                                more of the residual branch, lower to preserve
                                the skip connection.

        y = sqrt(1 - alpha) * x + sqrt(alpha) * f(x)
    """

    def __init__(
        self,
        body: keras.layers.Layer,
        norm_type: Optional[str],
        alpha: Optional[float],
        norm_cls: Type[keras.layers.Layer] = keras.layers.LayerNormalization,
    ):
        super().__init__()
        self.body = body
        self.norm_type = norm_type
        self.alpha_value = alpha
        self.alpha: tf.Variable = None
        assert norm_type in {None, "pre", "post"}, f"unexpected norm_type {norm_type}"
        self.norm_cls = norm_cls
        self.norm: keras.layers.Layer = None

    def build(self, input_shape: tf.TensorShape) -> None:
        super().build(input_shape)
        self.body.build(input_shape)
        if self.norm_type is not None:
            self.norm = self.norm_cls()
            self.norm.build(input_shape)
        if self.alpha_value is not None:
            # Turn alpha into a non-trainable variable, for sake of outlining
            self.alpha = self.add_weight(
                name="alpha",
                shape=(),
                initializer=keras.initializers.constant(self.alpha_value),
                trainable=False,
            )

    def call(self, x: tf.Tensor) -> tf.Tensor:
        branch = x
        if self.norm_type == "pre":
            branch = self.norm(branch)

        branch = self.body(branch)

        if self.alpha is not None:
            y = (1 - self.alpha) ** 0.5 * x + self.alpha**0.5 * branch
        else:
            y = x + branch

        if self.norm_type == "post":
            y = self.norm(y)
        return y


class FFNLayer(keras.layers.Layer):  # type:ignore[misc]
    """A pointwise expansion FFN layer (a la Transformer, https://arxiv.org/abs/1706.03762)."""

    def __init__(self, multiple: float, seeds: Optional[Tuple[int, int]] = None):
        super().__init__()
        self.multiple = multiple
        self.seeds = seeds or (None, None)
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
        self.up.build(input_shape[:-1] + (hidden_size,))
        self.down = keras.layers.Dense(
            hidden_size,
            kernel_initializer=keras.initializers.GlorotUniform(seed=self.seeds[1]),
        )
        self.down.build(input_shape[:-1] + (intermediate_size,))

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
            initializer=keras.initializers.zeros,
        )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        pad = tf.tile(self.padding[tf.newaxis, tf.newaxis], [inputs.shape[0], 1, 1])
        return tf.concat([pad, inputs[:, :-1, :]], axis=1)


class Isotropic(keras.layers.Layer):  # type:ignore[misc]
    """Like keras.models.Sequential, but isotropic & with friendly names for each layer."""

    def __init__(self, **layers: keras.layers.Layer):
        super().__init__()
        self._layers = layers
        for name, layer in layers.items():
            setattr(self, name, layer)

    def build(self, input_shape: tf.TensorShape) -> None:
        super().build(input_shape)
        for layer in self._layers.values():
            layer.build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        outputs = inputs
        for layer in self._layers.values():
            outputs = layer(outputs)
        return outputs


####################
# Attention


def sinusoid_embedding(
    sequence_length: int, frequencies: int, max_period: int
) -> np.ndarray:
    """Generate a family of sin/cos embeddings.

    See "Attention Is All You Need", Vaswani et al., section 3.5.

    sequence_length -- output dimension (number of indices)

    frequencies -- number of components to generate

    max_period -- the period (in indices) of the lowest frequency component

    returns -- array(sequence_length x frequencies)
    """
    index = np.arange(frequencies)
    frequency = np.pi * (2 / max_period) ** ((index // 2) / (frequencies // 2 - 1))
    phase = np.pi / 2 * (index % 2)
    time = np.arange(sequence_length)
    return np.sin(frequency * time[:, np.newaxis] + phase)


def relative_causal_reshape(scores: tf.Tensor) -> tf.Tensor:
    """Transform relative scores to an attention matrix.

    Fills the lower-left quadrant of the result with scores

        result[..., i, j] = scores[..., i, i - j]
    """
    sequence_length = scores.shape[-1]
    ndim = len(scores.shape)

    padded = tf.pad(scores[..., ::-1], [(0, 0)] * (ndim - 1) + [(0, sequence_length)])

    # A reshaping and slicing trick to move to relative positions
    tmp = tf.reshape(padded, padded.shape[:-2] + (2 * sequence_length**2,))
    tmp = tmp[..., :-sequence_length]
    tmp = tf.reshape(tmp, tmp.shape[:-1] + (sequence_length, 2 * sequence_length - 1))
    tmp = tmp[..., sequence_length - 1 :]

    return tmp


class MultiHeadAttention(keras.layers.Layer):  # type:ignore[misc]
    """Multi-head self attention a la Transformer.

    With causal masking.

    With relative-positional embeddings a la Transformer XL.
    """

    # pylint:disable=too-many-instance-attributes

    def __init__(
        self,
        heads: int,
        head_size: int,
        frequencies: int,
        max_period: int,
        seeds: Tuple[int, int, int],
    ):
        super().__init__()
        self.heads = heads
        self.head_size = head_size
        self.frequencies = frequencies
        self.max_period = max_period
        self.seeds = seeds
        self.qkv: tf.Variable = None
        self.q_bias: tf.Variable = None
        self.positional: tf.Variable = None
        self.out: keras.layers.Layer = None

    def build(self, input_shape: tf.TensorShape) -> None:
        super().build(input_shape)
        input_size = input_shape[-1]
        qkv_scale = np.sqrt(3) * input_size**-0.5
        self.qkv = self.add_weight(
            name="qkv",
            shape=(input_size, 3, self.heads, self.head_size),
            initializer=keras.initializers.random_uniform(
                -qkv_scale, qkv_scale, seed=self.seeds[0]
            ),
        )
        self.q_bias = self.add_weight(
            name="q_bias",
            shape=(self.heads, self.head_size),
            initializer=keras.initializers.zeros(),
        )
        positional_scale = np.sqrt(3) * self.frequencies**-0.5
        self.positional = self.add_weight(
            name="positional",
            shape=(self.frequencies, self.heads, self.head_size),
            initializer=keras.initializers.random_uniform(
                -positional_scale, positional_scale, seed=self.seeds[1]
            ),
        )
        self.out = keras.layers.Dense(
            input_size,
            kernel_initializer=keras.initializers.GlorotUniform(seed=self.seeds[2]),
        )
        self.out.build(input_shape[:-1] + (self.heads * self.head_size,))

    @staticmethod
    def _causal_mask(attention: tf.Tensor) -> tf.Tensor:
        sequence_length = attention.shape[-1]
        return tf.constant(
            np.triu(np.full((sequence_length, sequence_length), -1000), k=1),
            dtype=attention.dtype,
        )

    def _positional_mask(self, query: tf.Tensor) -> tf.Tensor:
        sequence_length = query.shape[-2]
        sins = tf.constant(
            sinusoid_embedding(sequence_length, self.frequencies, self.max_period),
            dtype=query.dtype,
        )
        embeddings = tf.einsum("sf,fnh->nsh", sins, self.positional)
        scores = tf.einsum("bnqh,nvh->bnqv", query, embeddings) * self.head_size**-0.5
        return relative_causal_reshape(scores)

    def call(self, input: tf.Tensor) -> tf.Tensor:
        # pylint:disable=invalid-name
        q, k, v = tf.unstack(tf.einsum("bsx,xAnh -> Abnsh", input, self.qkv))
        q += self.q_bias[:, tf.newaxis, :]
        a = tf.einsum("bnqh,bnkh->bnqk", q, k) * self.head_size**-0.5
        a += self._positional_mask(q)
        a += self._causal_mask(a)
        a = tf.nn.softmax(a, axis=-1)
        o = tf.einsum("bnqk,bnkh->bqnh", a, v)
        return self.out(tf.reshape(o, o.shape[:-2] + (self.head_size * self.heads,)))


####################
# Optimizers


class AdamW(keras.optimizers.Optimizer):  # type:ignore[misc]
    """AdamW (https://arxiv.org/abs/1711.05101)."""

    def __init__(
        self,
        learning_rate: float = 0.001,
        weight_decay: float = 0.004,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-7,
        name: str = "AdamW",
    ):
        super().__init__(name=name)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self._step_variable: tf.Variable = None

    @property
    def _step(self) -> tf.Variable:
        if self._step_variable is None:
            with tf.name_scope(self._name):
                self._step_variable = self.add_weight("step", (), dtype=tf.int32)
        return self._step_variable

    def _update(
        self, gradient: tf.Tensor, variable: tf.Variable, scale: tf.Tensor
    ) -> List[tf.Operation]:
        assert variable.dtype == tf.float32, "float16 AdamW not implemented"
        with tf.name_scope(self._name):
            m_prev = self.add_slot(variable, "adam_m")
            v_prev = self.add_slot(variable, "adam_v")

        if isinstance(gradient, tf.IndexedSlices):
            # Convert to dense gradient, which is probably fine
            gradient = tf.math.unsorted_segment_sum(
                gradient.values, gradient.indices, gradient.shape[0]
            )

        m_next = m_prev.assign(self.beta_1 * m_prev + (1 - self.beta_1) * gradient)
        v_next = v_prev.assign(self.beta_2 * v_prev + (1 - self.beta_2) * gradient**2)
        variable_update = variable.assign(
            variable
            - self.learning_rate * self.weight_decay * variable
            - scale * m_next / (tf.sqrt(v_next) + self.epsilon)
        )
        return [m_next, v_next, variable_update]

    def apply_gradients(
        self,
        grads_and_vars: Iterable[Tuple[tf.Tensor, tf.Variable]],
        name: Optional[str] = None,
    ) -> tf.Operation:
        step_prev = self._step
        step = step_prev.assign(step_prev + 1)
        scale = (
            self.learning_rate
            * tf.sqrt(1 - self.beta_2 ** tf.cast(step, tf.float32))
            / (1 - self.beta_1 ** tf.cast(step, tf.float32))
        )
        updates = [step]
        for grad, variable in grads_and_vars:
            updates.extend(self._update(grad, variable, scale=scale))
        return tf.group(*updates, name=name)
