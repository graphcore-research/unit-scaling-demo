"""Core model definitions."""

import itertools as it
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from tensorflow import keras

from . import layers, uscale
from .pedal import utility, xpu


@dataclass
class Residual:
    """Residual settings."""

    norm: Optional[str]  # None | "pre" | "post"
    alpha: Union[None, str, float]  # None | "mean" | <float>


@dataclass
class Conv:
    """Convolution (sequence mixing) settings."""

    kernel_size: int
    groups: int
    kind: str = "conv"


@dataclass
class Attention:
    """Attention (sequence mixing) settings."""

    heads: int
    head_size: int
    frequencies: int
    max_period: int
    kind: str = "attention"


@dataclass
class RNN:
    """Recurrence (sequence mixing) settings."""

    rebias: float
    kind: str = "rnn"


@dataclass
class FFN:
    """FFN (token mixing) settings."""

    multiple: float
    kind: str = "ffn"


@dataclass
class Settings:
    """Model configuration."""

    vocab_size: int
    hidden_size: int
    depth: int
    residual: Optional[Residual]
    sequence: Union[Conv, Attention]
    token: Optional[FFN]
    unit_scale: Optional[str]
    dtype: str
    seed: int


class _ModelFactory:  # pylint:disable=missing-function-docstring
    """Builds the various kinds of model from settings."""

    def __init__(self, settings: Settings, seeds: Iterator[int]):
        self.settings = settings
        self.dtype = tf.as_dtype(settings.dtype)
        self.seeds = seeds
        assert settings.unit_scale in {None, "0.2"}

    def kernel_initializer(self) -> keras.initializers.Initializer:
        assert not self.settings.unit_scale, "unit scale shouldn't use Glorot"
        return keras.initializers.GlorotUniform(seed=next(self.seeds))

    def embed(self) -> keras.layers.Layer:
        if self.settings.unit_scale:
            return uscale.layers.Embedding(
                self.settings.vocab_size,
                self.settings.hidden_size,
                dtype=self.dtype,
                seed=next(self.seeds),
            )
        # Unit variance embeddings make sense in any case
        return keras.layers.Embedding(
            self.settings.vocab_size,
            self.settings.hidden_size,
            dtype=self.dtype,
            embeddings_initializer=keras.initializers.RandomUniform(
                -np.sqrt(3), np.sqrt(3), seed=next(self.seeds)
            ),
        )

    def conv(self, settings: Conv) -> keras.layers.Layer:
        if self.settings.unit_scale:
            return uscale.layers.CausalConv1D(
                self.settings.hidden_size,
                kernel_size=settings.kernel_size,
                groups=settings.groups,
                activation="relu",
                dtype=self.dtype,
                seed=next(self.seeds),
            )
        return keras.layers.Conv1D(
            self.settings.hidden_size,
            kernel_size=settings.kernel_size,
            groups=settings.groups,
            activation="relu",
            padding="causal",
            dtype=self.dtype,
            kernel_initializer=self.kernel_initializer(),
        )

    def attention(self, settings: Attention) -> keras.layers.Layer:
        cls = (
            uscale.layers.MultiHeadAttention
            if self.settings.unit_scale
            else layers.MultiHeadAttention
        )
        return cls(
            heads=settings.heads,
            head_size=settings.head_size,
            frequencies=settings.frequencies,
            max_period=settings.max_period,
            dtype=self.dtype,
            seeds=(next(self.seeds), next(self.seeds), next(self.seeds)),
        )

    def rnn(self, settings: RNN) -> keras.layers.Layer:
        (cls, cell_cls) = (
            (uscale.layers.RNN, uscale.layers.RecurrentHighwayCell)
            if self.settings.unit_scale
            else (layers.RNN, layers.RecurrentHighwayCell)
        )
        return cls(
            cell_cls(
                hidden_size=self.settings.hidden_size,
                rebias=settings.rebias,
                dtype=self.dtype,
                seed=next(self.seeds),
            )
        )

    def sequence_layer(self) -> keras.layers.Layer:
        if isinstance(self.settings.sequence, Conv):
            return self.conv(self.settings.sequence)
        if isinstance(self.settings.sequence, Attention):
            return self.attention(self.settings.sequence)
        if isinstance(self.settings.sequence, RNN):
            return self.rnn(self.settings.sequence)
        assert False, f"unexpected sequence settings {self.settings.sequence}"

    def token_layer(self) -> keras.layers.Layer:
        assert self.settings.token is not None
        cls = uscale.layers.FFNLayer if self.settings.unit_scale else layers.FFNLayer
        return cls(
            self.settings.token.multiple,
            dtype=self.dtype,
            seeds=(next(self.seeds), next(self.seeds)),
        )

    def residual(self, body: keras.layers.Layer, index: int) -> keras.layers.Layer:
        if self.settings.residual is None:
            return body

        if self.settings.residual.alpha is None:
            alpha = None
        elif self.settings.residual.alpha == "mean":
            alpha = 1 / (1 + index)
        elif isinstance(self.settings.residual.alpha, (float, int)):
            alpha = self.settings.residual.alpha
        else:
            assert False, f"unexpected residual.alpha {self.settings.residual.alpha}"

        layer_cls = (
            uscale.layers.ResidualLayer
            if self.settings.unit_scale
            else layers.ResidualLayer
        )
        return layer_cls(
            body, norm_type=self.settings.residual.norm, alpha=alpha, dtype=self.dtype
        )

    def trunk_layer(self, index: Iterator[int]) -> keras.layers.Layer:
        # Relying heavily on dict ordering...
        parts = dict(sequence=self.residual(self.sequence_layer(), next(index)))
        if self.settings.token:
            parts["token"] = self.residual(self.token_layer(), next(index))
        return layers.Isotropic(dtype=self.dtype, **parts)

    def trunk(self) -> List[keras.layers.Layer]:
        index = iter(it.count())
        return [self.trunk_layer(index) for _ in range(self.settings.depth)]

    def norm(self) -> keras.layers.Layer:
        return (
            uscale.layers.LayerNormalization(dtype=self.dtype)
            if self.settings.unit_scale
            else layers.LayerNormalization(dtype=self.dtype)
        )

    def predict(self) -> keras.layers.Layer:
        if self.settings.unit_scale:
            return uscale.layers.Dense(
                self.settings.vocab_size,
                scale_for="both_min",
                dtype=self.dtype,
                seed=next(self.seeds),
            )
        return keras.layers.Dense(
            self.settings.vocab_size,
            dtype=self.dtype,
            kernel_initializer=self.kernel_initializer(),
        )

    def predict_padding(self) -> keras.layers.Layer:
        return layers.PadAndShiftLayer(dtype=self.dtype)

    def loss(
        self,
    ) -> Callable[[tf.Tensor, tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]:
        if self.settings.unit_scale:
            return uscale.ops.softmax_cross_entropy
        return layers.softmax_cross_entropy


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
        for name, layer in utility.named_layers(self):
            assert layer.built, f"layer {name} ({layer}) was not built"

        for layer in self.trunk:
            xpu.current_context().outline(layer)

    def build(self, input_shape: tf.TensorShape) -> None:
        super().build(input_shape)

        self.embed.build(input_shape)
        hidden_shape = tuple(input_shape) + (self.settings.hidden_size,)
        for layer in self.trunk:
            layer.build(hidden_shape)
        self.norm.build(hidden_shape)
        self.predict.build(hidden_shape)
        prediction_shape = tuple(input_shape) + (self.settings.vocab_size,)
        self.predict_padding.build(prediction_shape)

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
                "Load does not set correct weights"
                f", extra: {weights.keys() - variables.keys()}"
                f", missing: {variables.keys() - weights.keys()}"
            )
        for k in weights:
            variables[k].assign(weights[k])
