"""Core model definitions."""

from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras

from . import layers
from .pedal import utility


@dataclass
class Settings:
    """Model configuration."""

    seed: int
    vocab_size: int
    hidden_size: int
    depth: int
    kernel_size: int


@dataclass
class SimpleConv(Settings):
    """A stack of causual convolutions with relu nonlinearity."""

    kind: str = "simple_conv"


@dataclass
class ResidualConv(Settings):
    """A prenorm stack of causal grouped convolutions and pointwise FFNs."""

    group_size: int
    ffn_multiple: float
    kind: str = "residual_conv"


def _built(
    layer: keras.layers.Layer, shape: Tuple[Optional[int], ...]
) -> keras.layers.Layer:
    """Build a layer and return it."""
    layer.build(shape)
    return layer


class _SimpleConvLayer(keras.layers.Conv1D):  # type:ignore[misc]
    def __init__(self, settings: SimpleConv, seeds: Iterator[int]):
        super().__init__(
            settings.hidden_size,
            kernel_size=settings.kernel_size,
            padding="causal",
            kernel_initializer=keras.initializers.GlorotUniform(seed=next(seeds)),
        )

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return tf.nn.relu(super().call(x))


class _ResidualConvLayer(keras.layers.Layer):  # type:ignore[misc]
    def __init__(self, settings: ResidualConv, seeds: Iterator[int]):
        super().__init__()
        self.conv = layers.PreNormResidualLayer(
            keras.layers.Conv1D(
                settings.hidden_size,
                kernel_size=settings.kernel_size,
                groups=settings.hidden_size // settings.group_size,
                padding="causal",
                kernel_initializer=keras.initializers.GlorotUniform(seed=next(seeds)),
            )
        )
        self.ffn = layers.PreNormResidualLayer(
            layers.FFNLayer(settings.ffn_multiple, seeds=(next(seeds), next(seeds)))
        )

    def build(self, input_shape: tf.TensorShape) -> None:
        self.conv.build(input_shape)
        self.ffn.build(input_shape)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = self.conv(x)
        return self.ffn(x)


def _create_trunk(settings: Settings, seeds: Iterator[int]) -> keras.layers.Layer:
    if isinstance(settings, SimpleConv):
        return _SimpleConvLayer(settings, seeds)
    if isinstance(settings, ResidualConv):
        return _ResidualConvLayer(settings, seeds)
    assert False, f"Unexpected models settings type {type(settings)}"


class Model(keras.layers.Layer):  # type:ignore[misc]
    """Base language model."""

    def __init__(self, settings: Settings):
        super().__init__()
        seeds = iter(utility.split_seed(settings.seed, 1000))  # plenty of seeds
        shape = (None, None, settings.hidden_size)
        self.embed = _built(
            keras.layers.Embedding(
                settings.vocab_size,
                settings.hidden_size,
                embeddings_initializer=keras.initializers.RandomUniform(
                    seed=next(seeds)
                ),
            ),
            (),
        )
        self.trunk = [
            _built(_create_trunk(settings, seeds), shape) for _ in range(settings.depth)
        ]
        self.norm = _built(keras.layers.LayerNormalization(), shape)
        self.predict = _built(
            keras.layers.Dense(
                settings.vocab_size,
                kernel_initializer=keras.initializers.GlorotUniform(seed=next(seeds)),
            ),
            shape,
        )
        self.predict_padding = self.add_weight(
            name="predict_padding",
            shape=settings.vocab_size,
            dtype=self.dtype,
            initializer=keras.initializers.zeros,
        )

    def weight_stats(self) -> Dict[str, Any]:
        """Stats regarding weights in the model."""
        shapes = {k: tuple(v.shape) for k, v in utility.named_weights(self)}
        return dict(
            n_weights=sum(np.prod(v) for v in shapes.values()),
            n_weights_no_embedding=sum(
                np.prod(v)
                for k, v in shapes.items()
                if k not in {"embed.embeddings", "predict.kernel"}
            ),
            weight_shapes=shapes,
        )

    def save(self) -> Dict[str, np.ndarray]:
        """Save model weights to a dictionary of numpy arrays."""
        return {k: np.array(v) for k, v in utility.named_weights(self)}

    def load(self, weights: Dict[str, np.ndarray]) -> None:
        """Load model weights from a dictionary of numpy arrays."""
        variables = dict(utility.named_weights(self))
        if variables.keys() != weights.keys():
            raise ValueError(
                "Load does not set all weights"
                f", extra: {weights.keys() - variables.keys()}"
                f", missing: {variables.keys() - weights.keys()}"
            )
        for k in weights:
            variables[k].assign(weights[k])

    def _shift_predictions(self, predictions: tf.Tensor) -> tf.Tensor:
        pad = tf.tile(
            self.predict_padding[tf.newaxis, tf.newaxis], [predictions.shape[0], 1, 1]
        )
        return tf.concat([pad, predictions[:, :-1, :]], axis=1)

    @staticmethod
    def _total_loss(scores: tf.Tensor, tokens: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
        logp = tf.nn.log_softmax(scores)
        # Better compilation on IPU vs `tf.gather(logp, tokens, batch_dims=2)`
        target_logp = layers.batched_gather(logp, tokens)
        return tf.reduce_sum(tf.cast(mask, target_logp.dtype) * -target_logp)

    def run(self, tokens: tf.Tensor, mask: tf.Tensor) -> Dict[str, tf.Tensor]:
        """Run the language model for cross entropy loss."""
        hiddens = self.embed(tokens)
        for layer in self.trunk:
            hiddens = layer(hiddens)
        scores = self._shift_predictions(self.predict(self.norm(hiddens)))
        loss = self._total_loss(scores, tokens, mask)
        n_tokens = tf.reduce_sum(tf.cast(mask, tf.int32))
        return dict(loss=loss / tf.cast(n_tokens, loss.dtype), n_tokens=n_tokens)
