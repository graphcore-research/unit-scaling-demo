"""General purpose layers and functions."""

from typing import Iterable, List, Optional, Tuple, Type

import numpy as np
import tensorflow as tf
from tensorflow import keras


def batched_gather(tables: tf.Tensor, indices: tf.Tensor) -> tf.Tensor:
    """Simulate tf.gather(tables, indices, batch_dims=indices.ndim).

    Better compilation on IPU vs `tf.gather(logp, ids, batch_dims=2)`
    """
    # Implemented here and in uscale.ops to avoid circular dependency issues
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

    returns -- (average_loss, n_items) -- average_loss always in fp32
    """
    assert mask.shape == ids.shape, "mask should match target ids"
    # Use float32 for local computation - keeping things simple
    logp = tf.nn.log_softmax(tf.cast(scores, tf.float32))
    target_logp = batched_gather(logp, ids)
    total_loss = tf.reduce_sum(tf.cast(mask, target_logp.dtype) * -target_logp)
    n_ids = tf.reduce_sum(tf.cast(mask, tf.int32))
    loss = total_loss / tf.cast(n_ids, total_loss.dtype)
    return loss, n_ids


class LayerNormalization(keras.layers.Layer):  # type:ignore[misc]
    """A FP16-safe variant of keras.layers.LayerNormalization."""

    def __init__(self, epsilon: float = 0.001, dtype: tf.DType = tf.float32):
        super().__init__(dtype=dtype)
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

    @staticmethod
    def _normalize(inputs: tf.Tensor) -> tf.Tensor:
        inputs_fp32 = tf.cast(inputs, tf.float32)
        z = inputs_fp32 - tf.reduce_mean(inputs_fp32, axis=-1, keepdims=True)
        normed = z / tf.sqrt(tf.reduce_mean(z**2, axis=-1, keepdims=True))
        return tf.cast(normed, inputs.dtype)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return self.gamma * self._normalize(inputs) + self.beta


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
        dtype: tf.DType = tf.float32,
        norm_cls: Type[keras.layers.Layer] = LayerNormalization,
    ):
        super().__init__(dtype=dtype)
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
            self.norm = self.norm_cls(dtype=self.dtype)
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

    def __init__(
        self,
        multiple: float,
        dtype: tf.DType = tf.float32,
        seeds: Optional[Tuple[int, int]] = None,
    ):
        super().__init__(dtype=dtype)
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
            dtype=self.dtype,
            kernel_initializer=keras.initializers.GlorotUniform(seed=self.seeds[0]),
        )
        self.up.build(input_shape[:-1] + (hidden_size,))
        self.down = keras.layers.Dense(
            hidden_size,
            dtype=self.dtype,
            kernel_initializer=keras.initializers.GlorotUniform(seed=self.seeds[1]),
        )
        self.down.build(input_shape[:-1] + (intermediate_size,))

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return self.down(tf.nn.relu(self.up(x)))  # type:ignore[misc]


class PadAndShiftLayer(keras.layers.Layer):  # type:ignore[misc]
    """Shifts sequence features one place to the right with a trainable padding vector."""

    def __init__(self, dtype: tf.DType = tf.float32) -> None:
        super().__init__(dtype=dtype)
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

    def __init__(self, dtype: tf.DType = tf.float32, **layers: keras.layers.Layer):
        super().__init__(dtype=dtype)
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


def causal_mask(attention: tf.Tensor, mask_value: float = -1000) -> tf.Tensor:
    """Apply a causal mask to an attention matrix of shape (*, L, L)."""
    sequence_length = attention.shape[-1]
    return attention + tf.constant(
        np.triu(np.full((sequence_length, sequence_length), mask_value), k=1),
        dtype=attention.dtype,
    )


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
            dtype=self.dtype,
            kernel_initializer=keras.initializers.GlorotUniform(seed=self.seeds[2]),
        )
        self.out.build(input_shape[:-1] + (self.heads * self.head_size,))

    def _positional_weights(self, query: tf.Tensor) -> tf.Tensor:
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
        a += self._positional_weights(q)
        a = causal_mask(a)
        a = tf.nn.softmax(a, axis=-1)
        o = tf.einsum("bnqk,bnkh->bqnh", a, v)
        return self.out(tf.reshape(o, o.shape[:-2] + (self.head_size * self.heads,)))


####################
# RNN


class RecurrentHighwayCell(keras.layers.Layer):  # type:ignore[misc]
    """A recurrent highway cell from https://arxiv.org/abs/1607.03474."""

    def __init__(self, hidden_size: int, rebias: float, tied_gates: bool):
        super().__init__(name=type(self).__name__)
        self.hidden_size = hidden_size
        self.carry_rebias = rebias
        self.update_rebias = -rebias
        self.tied_gates = tied_gates
        self.gates: tf.Variable = None
        self.gates_bias: tf.Variable = None

    def build(self, input_shape: tf.TensorShape) -> None:
        super().build(input_shape)
        input_size = input_shape[-1]
        scale = (3 / (input_size + self.hidden_size)) ** 0.5
        n_gates = 2 + (not self.tied_gates)
        self.gates = self.add_weight(
            "gates",
            shape=(n_gates, input_size + self.hidden_size, self.hidden_size),
            initializer=keras.initializers.random_uniform(-scale, scale),
        )
        self.gates_bias = self.add_weight(
            "gates_bias",
            shape=(n_gates, self.hidden_size),
            initializer=keras.initializers.zeros(),
        )

    def call(self, input: tf.Tensor, hidden: tf.Tensor) -> tf.Tensor:
        gates = (
            tf.concat([input, hidden], axis=1) @ self.gates
            + self.gates_bias[:, tf.newaxis]
        )
        if self.tied_gates:
            transform, update = tf.unstack(gates)
            update = tf.sigmoid(update + self.update_rebias)
            return (1 - update) * hidden + update * tf.tanh(transform)

        transform, update, carry = tf.unstack(gates)
        carry = tf.sigmoid(carry + self.carry_rebias)
        update = tf.sigmoid(update + self.update_rebias)
        return carry * hidden + update * tf.tanh(transform)


class RNN(keras.layers.Layer):  # type:ignore[misc]
    """A basic, unidirectional RNN.

    Expects inputs of shape (batch, sequence, feature), and produces outputs of shape
    (batch, sequence, hidden).
    """

    def __init__(self, cell: keras.layers.Layer):
        super().__init__(name=type(self).__name__)
        self.cell = cell
        self.initial_hidden: tf.Variable = None

    def build(self, input_shape: tf.TensorShape) -> None:
        super().build(input_shape)
        self.cell.build(tf.TensorShape([input_shape[0], input_shape[2]]))
        self.initial_hidden = self.add_weight(
            "initial_hidden",
            shape=(self.cell.hidden_size,),
            initializer=keras.initializers.zeros(),
        )

    def call(self, input: tf.Tensor) -> tf.Tensor:
        input_sbh = tf.transpose(input, (1, 0, 2))
        output_sbh = tf.scan(
            lambda hidden, input: self.cell(input, hidden),
            input_sbh,
            initializer=tf.tile(self.initial_hidden[tf.newaxis], (input.shape[0], 1)),
        )
        return tf.transpose(output_sbh, (1, 0, 2))


####################
# Optimizers


class _Optimizer(keras.optimizers.Optimizer):  # type:ignore[misc]
    """A small extension of the keras base optimizer."""

    def _add_slot_with_dtype(
        self, variable: tf.Variable, name: str, dtype: tf.DType
    ) -> tf.Variable:
        # pylint:disable=protected-access
        key = variable._shared_name if variable._in_graph_mode else variable._unique_id
        result = self._slots.setdefault(key, {}).get(name)
        if result is None:
            result = tf.Variable(
                tf.zeros(variable.shape, dtype=dtype),
                name=f"{key}/{name}",
                trainable=False,
            )
            self._slots[key][name] = result
        return result


class SgdM(_Optimizer):
    """SGD with momentum and loss scaling support."""

    def __init__(
        self,
        learning_rate: float = 0.01,
        loss_scale: float = 1,
        momentum: float = 0,
        name: str = "SGD",
    ):
        super().__init__(name=name)
        self.learning_rate = learning_rate
        self.loss_scale = loss_scale
        self.momentum = momentum

    def _update(self, gradient: tf.Tensor, variable: tf.Variable) -> List[tf.Operation]:
        if isinstance(gradient, tf.IndexedSlices):
            # Convert to dense gradient, which is probably fine
            gradient = tf.math.unsorted_segment_sum(
                gradient.values, gradient.indices, gradient.shape[0]
            )

        with tf.name_scope(self._name):
            momentum_prev = self._add_slot_with_dtype(
                variable, "momentum", dtype=variable.dtype
            )

        # This FP32 dance probably isn't of much importance/help here
        momentum_next = self.momentum * tf.cast(momentum_prev, tf.float32) + tf.cast(
            gradient, tf.float32
        )
        step_size = self.learning_rate / self.loss_scale
        variable_next = tf.cast(variable, tf.float32) - step_size * momentum_next
        return [
            variable.assign(tf.cast(variable_next, variable.dtype)),
            momentum_prev.assign(tf.cast(momentum_next, momentum_prev.dtype)),
        ]

    def apply_gradients(
        self,
        grads_and_vars: Iterable[Tuple[tf.Tensor, tf.Variable]],
        name: Optional[str] = None,
    ) -> tf.Operation:
        return tf.group(
            *(self._update(grad, variable) for grad, variable in grads_and_vars),
            name=name,
        )


class AdamW(_Optimizer):
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
        if isinstance(gradient, tf.IndexedSlices):
            # Convert to dense gradient, which is probably fine
            gradient = tf.math.unsorted_segment_sum(
                gradient.values, gradient.indices, gradient.shape[0]
            )

        with tf.name_scope(self._name):
            m_prev = self._add_slot_with_dtype(variable, "adam_m", dtype=variable.dtype)
            v_prev = self._add_slot_with_dtype(variable, "adam_v", dtype=tf.float32)

        gradient_fp32 = tf.cast(gradient, tf.float32)
        m_next = (
            self.beta_1 * tf.cast(m_prev, tf.float32)
            + (1 - self.beta_1) * gradient_fp32
        )
        v_next = (
            self.beta_2 * tf.cast(v_prev, tf.float32)
            + (1 - self.beta_2) * gradient_fp32**2
        )
        variable_fp32 = tf.cast(variable, tf.float32)
        variable_next = (
            variable_fp32
            - self.learning_rate * self.weight_decay * variable_fp32
            - scale * m_next / (tf.sqrt(v_next) + self.epsilon)
        )
        return [
            variable.assign(tf.cast(variable_next, variable.dtype)),
            m_prev.assign(tf.cast(m_next, m_prev.dtype)),
            v_prev.assign(tf.cast(v_next, v_prev.dtype)),
        ]

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
        return tf.group(
            step,
            *(
                self._update(grad, variable, scale=scale)
                for grad, variable in grads_and_vars
            ),
            name=name,
        )
