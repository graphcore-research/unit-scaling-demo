"""General purpose layers and functions."""

from typing import Iterable, List, Optional, Tuple, Type

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
        self.alpha = alpha
        assert norm_type in {None, "pre", "post"}, f"unexpected norm_type {norm_type}"
        self.norm_cls = norm_cls
        self.norm: keras.layers.Layer = None

    def build(self, input_shape: tf.TensorShape) -> None:
        super().build(input_shape)
        self.body.build(input_shape)
        if self.norm_type is not None:
            self.norm = self.norm_cls()
            self.norm.build(input_shape)

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
            dtype=self.dtype,
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
