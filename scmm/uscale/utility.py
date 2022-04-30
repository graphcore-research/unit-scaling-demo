"""Utilities to support unit scaling development."""

import collections
import functools
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras

from ..pedal import utility


class ActivationTracker:
    """Track activations (and gradients) for layers in a model.

    Note that this only works in eager mode, and is designed for offline
    use.

        layer = keras.layers.Dense(10)
        layer.build((None, 20))

        tracker = ActivationTracker(layer)
        with tf.GradientTape() as tape:
            loss = tf.reduce_sum(layer(tf.zeros((3, 20))))
        grads_and_vars = zip(
            tape.gradient(loss, layer.trainable_variables),
            layer.trainable_variables)
        tracker.log_gradients(grads_and_vars)  # for weight gradients only

        print({t.name: np.std(t.gradient) for t in tracker.trace})
    """

    @dataclass
    class Trace:
        """Forward and backward pass tensors from a single edge in the graph."""

        name: str
        activation: np.ndarray
        gradient: Optional[np.ndarray]

    @dataclass
    class LayerTrace(Trace):
        """Forward and backward pass information for a layer (with one output)."""

        layer: keras.layers.Layer
        weights: Tuple["ActivationTracker.Trace", ...]

    def __init__(self, *layers_to_track: keras.layers.Layer):
        self._layers: Dict[str, keras.layers.Layer] = {}
        self._variable_to_weight_name: Dict[tf.Variable, Tuple[str, str]] = {}
        self._weights: Dict[str, Dict[str, np.ndarray]] = collections.defaultdict(dict)
        self._weight_gradients: Dict[
            str, Dict[str, List[np.ndarray]]
        ] = collections.defaultdict(lambda: collections.defaultdict(list))
        self._activations: Dict[str, List[np.ndarray]] = collections.defaultdict(list)
        self._gradients: Dict[str, List[np.ndarray]] = collections.defaultdict(list)
        for layer in layers_to_track:
            self.track(layer)

    def _track_layer(self, name: str, layer: keras.layers.Layer) -> None:
        self._layers[name] = layer

        for weight_name, weight in utility.named_weights(layer, recursive=False):
            self._weights[name][weight_name] = weight.numpy()
            self._variable_to_weight_name[weight.ref()] = (name, weight_name)

        @tf.custom_gradient  # type:ignore[misc]
        def identity_log(x: tf.Tensor) -> tf.Tensor:
            self._activations[name].append(x.numpy())

            def grad(upstream: tf.Tensor) -> tf.Tensor:
                self._gradients[name].append(upstream.numpy())
                return upstream

            return x, grad

        original_call = layer.call

        @functools.wraps(original_call)
        def wrapper(*args: Any, **kwargs: Any) -> tf.Tensor:
            output = original_call(*args, **kwargs)
            if not isinstance(output, tf.Tensor):
                raise ValueError(
                    "Expected a layer to output a single tensor, actual output"
                    f" {type(output)} from layer {layer}"
                )
            return identity_log(output)

        layer.call = wrapper

    def track(self, layer: keras.layers.Layer) -> None:
        """Start track this layer's output and any (recursive) sublayers."""
        for name, sublayer in utility.named_layers(layer):
            self._track_layer(name, sublayer)

    def log_gradients(
        self, grads_and_vars: Iterable[Tuple[tf.Tensor, tf.Variable]]
    ) -> None:
        """Log weight gradients (optional call)."""
        for grad, variable in grads_and_vars:
            if isinstance(grad, tf.IndexedSlices):
                grad = tf.math.unsorted_segment_sum(
                    grad.values, grad.indices, grad.shape[0]
                )
            layer_name, weight_name = self._variable_to_weight_name[variable.ref()]
            self._weight_gradients[layer_name][weight_name].append(grad.numpy())

    @staticmethod
    def _stack_optional(items: Sequence[np.ndarray]) -> Optional[np.ndarray]:
        return np.stack(items) if items else None

    @property
    def trace(self) -> Tuple[LayerTrace, ...]:
        """Get activation and gradient traces for each layer (ordered by forward pass)."""
        return tuple(
            self.LayerTrace(
                name=layer_name,
                activation=np.stack(self._activations[layer_name]),
                gradient=self._stack_optional(self._gradients[layer_name]),
                layer=self._layers[layer_name],
                weights=tuple(
                    self.Trace(
                        name=weight_name,
                        activation=weight,
                        gradient=self._stack_optional(
                            self._weight_gradients[layer_name][weight_name]
                        ),
                    )
                    for weight_name, weight in self._weights[layer_name].items()
                ),
            )
            for layer_name in self._activations  # forward pass ordering
        )


def printing(
    name: str, summary: Callable[[tf.Tensor], Any] = np.std
) -> Callable[[tf.Tensor], tf.Tensor]:
    """Utility for printing forward/backward pass statistics.

    E.g.
        x = printing("x")(x)
    """

    @tf.custom_gradient  # type:ignore[misc]
    def operation(x: tf.Tensor) -> tf.Tensor:
        print(f"{name} forward {summary.__name__}", summary(x), file=sys.stderr)

        def grad(upstream: tf.Tensor) -> tf.Tensor:
            print(
                f"{name} backward {summary.__name__}",
                summary(upstream),
                file=sys.stderr,
            )
            return upstream

        return x, grad

    return operation  # type:ignore[no-any-return]
