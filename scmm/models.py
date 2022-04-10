"""Core model definitions."""

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras

from .pedal import utility


@dataclass
class Settings:
    """Model configuration."""

    vocab_size: int
    hidden_size: int
    depth: int
    kernel_size: int
    seed: int
    kind: str = "simple_conv"


def _built(layer: keras.layers.Layer, shape: Tuple[int, ...]) -> keras.layers.Layer:
    """Build a layer and return it."""
    layer.build(shape)
    return layer


class Model(keras.layers.Layer):  # type:ignore[misc]
    """Base language model."""

    def __init__(self, settings: Settings):
        super().__init__()
        seeds = iter(utility.split_seed(settings.seed, 1000))  # plenty of seeds
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
        self.convs = [
            _built(
                keras.layers.Conv1D(
                    settings.hidden_size,
                    kernel_size=settings.kernel_size,
                    padding="causal",
                    kernel_initializer=keras.initializers.GlorotUniform(
                        seed=next(seeds)
                    ),
                ),
                (settings.hidden_size,),
            )
            for _ in range(settings.depth)
        ]
        self.predict = _built(
            keras.layers.Dense(
                settings.vocab_size,
                kernel_initializer=keras.initializers.GlorotUniform(seed=next(seeds)),
            ),
            shape=(settings.hidden_size,),
        )
        self.predict_padding = self.add_weight(
            name="predict_padding",
            shape=settings.vocab_size,
            dtype=self.dtype,
            initializer=keras.initializers.zeros,
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
        target_logp = -tf.gather(logp, tf.cast(tokens, tf.int32), batch_dims=2)
        return tf.reduce_sum(mask * target_logp)

    def run(self, tokens: tf.Tensor, mask: tf.Tensor) -> Dict[str, tf.Tensor]:
        """Run the language model for cross entropy loss."""
        hiddens = self.embed(tokens)
        for conv in self.convs:
            hiddens = tf.nn.relu(conv(hiddens))
        scores = self._shift_predictions(self.predict(hiddens))
        loss = self._total_loss(scores, tokens, mask)
        n_tokens = tf.reduce_sum(tf.cast(mask, tf.int32))
        return dict(loss=loss / tf.cast(n_tokens, loss.dtype), n_tokens=n_tokens)
