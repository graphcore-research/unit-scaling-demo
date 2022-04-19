"""Core model definitions."""

from dataclasses import dataclass
from typing import Any, Dict, Iterator

import numpy as np
import tensorflow as tf
from tensorflow import keras

from . import layers, uscale
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
        super().build(input_shape)
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
        self.settings = settings
        seeds = iter(utility.split_seed(settings.seed, 1000))  # plenty of seeds

        self.embed = keras.layers.Embedding(
            settings.vocab_size,
            settings.hidden_size,
            embeddings_initializer=uscale.Initializers.uniform(next(seeds)),
        )
        self.trunk = [_create_trunk(settings, seeds) for _ in range(settings.depth)]
        self.norm = keras.layers.LayerNormalization()
        self.predict = keras.layers.Dense(
            settings.vocab_size,
            kernel_initializer=keras.initializers.GlorotUniform(seed=next(seeds)),
        )
        self.predict_padding = layers.PadAndShiftLayer()

        # Our base model is always pre-built
        self.build((None, None))
        for _, layer in utility.named_layers(self):
            assert layer.built

    def build(self, input_shape: tf.TensorShape) -> None:
        super().build(input_shape)

        self.embed.build(input_shape)
        hidden_shape = tuple(input_shape) + (self.settings.hidden_size,)
        for layer in self.trunk:
            layer.build(hidden_shape)
        self.norm.build(hidden_shape)
        self.predict.build(hidden_shape)
        self.predict_padding.build(tuple(input_shape) + (self.settings.vocab_size,))

    def run(self, tokens: tf.Tensor, mask: tf.Tensor) -> Dict[str, tf.Tensor]:
        """Run the language model for cross entropy loss."""
        hiddens = self.embed(tokens)
        for layer in self.trunk:
            hiddens = layer(hiddens)
        scores = self.predict_padding(self.predict(self.norm(hiddens)))
        loss, n_tokens = layers.softmax_cross_entropy(scores, tokens, mask)
        return dict(loss=loss, n_tokens=n_tokens)

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
