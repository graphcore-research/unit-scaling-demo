"""Core model definitions."""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras

from . import layers, uscale
from .pedal import utility


@dataclass
class Settings:
    """Model configuration."""

    unit_scale: bool
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


class _ModelFactory:  # pylint:disable=missing-function-docstring
    def __init__(self, settings: Settings, seeds: Iterator[int]):
        self.settings = settings
        self.seeds = seeds

    def kernel_initializer(self) -> keras.initializers.Initializer:
        return keras.initializers.GlorotUniform(seed=next(self.seeds))

    def embed(self) -> keras.layers.Layer:
        return layers.Embedding(
            self.settings.vocab_size, self.settings.hidden_size, seed=next(self.seeds)
        )

    def trunk_layer(self) -> keras.layers.Layer:
        if isinstance(self.settings, SimpleConv):
            if self.settings.unit_scale:
                return uscale.CausalConv1D(
                    self.settings.hidden_size,
                    kernel_size=self.settings.kernel_size,
                    activation="relu",
                    seed=next(self.seeds),
                )
            return keras.layers.Conv1D(
                self.settings.hidden_size,
                kernel_size=self.settings.kernel_size,
                activation="relu",
                padding="causal",
                kernel_initializer=self.kernel_initializer(),
            )

        if isinstance(self.settings, ResidualConv):
            assert (
                not self.settings.unit_scale
            ), "unit_scale unimplemented for ResidualConv"
            return layers.Isotropic(
                conv=layers.PreNormResidualLayer(
                    keras.layers.Conv1D(
                        self.settings.hidden_size,
                        kernel_size=self.settings.kernel_size,
                        groups=self.settings.hidden_size // self.settings.group_size,
                        padding="causal",
                        kernel_initializer=self.kernel_initializer(),
                    )
                ),
                ffn=layers.PreNormResidualLayer(
                    layers.FFNLayer(
                        self.settings.ffn_multiple,
                        seeds=(next(self.seeds), next(self.seeds)),
                    )
                ),
            )
        assert False, f"Unexpected model type {type(self.settings)}"

    def trunk(self) -> List[keras.layers.Layer]:
        return [self.trunk_layer() for _ in range(self.settings.depth)]

    @staticmethod
    def norm() -> keras.layers.Layer:
        return keras.layers.LayerNormalization()

    def predict(self) -> keras.layers.Layer:
        if self.settings.unit_scale:
            return uscale.Dense(self.settings.vocab_size, seed=next(self.seeds))
        return keras.layers.Dense(
            self.settings.vocab_size, kernel_initializer=self.kernel_initializer()
        )

    @staticmethod
    def predict_padding() -> keras.layers.Layer:
        return layers.PadAndShiftLayer()

    def loss(
        self,
    ) -> Callable[[tf.Tensor, tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]:
        return (
            uscale.softmax_cross_entropy
            if self.settings.unit_scale
            else layers.softmax_cross_entropy
        )


class Model(keras.layers.Layer):  # type:ignore[misc]
    """Base language model."""

    def __init__(self, settings: Settings):
        super().__init__()
        self.settings = settings

        factory = _ModelFactory(
            settings, iter(utility.split_seed(settings.seed, 1000))  # plenty of seeds
        )
        self.embed = factory.embed()
        self.trunk = factory.trunk()
        self.norm = factory.norm()
        self.predict = factory.predict()
        self.predict_padding = factory.predict_padding()
        self.loss = factory.loss()

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
        loss, n_tokens = self.loss(scores, tokens, mask)
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
