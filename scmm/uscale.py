"""Core components for unit-scaling activations and gradients."""

import collections
import functools
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras

from .pedal import utility

###############################################################################
# Utility


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


###############################################################################
# Ops


@tf.custom_gradient  # type:ignore[misc]
def scaling(
    input: tf.Tensor,  # pylint:disable=redefined-builtin
    *,
    forward: Optional[float] = None,
    backward: Optional[float] = None,
) -> tf.Tensor:
    """Perform arbitary *seperate* scaling in the forward and backward passes."""

    def grad(upstream: tf.Tensor) -> tf.Tensor:
        grad_input = upstream
        if backward is not None:
            grad_input *= backward
        return grad_input

    output = input
    if forward is not None:
        output *= forward

    return output, grad


def matmul(a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
    """Scaling version of tf.matmul (where b must be 2D)."""
    assert len(b.shape) == 2, "matmul requires 2D rhs (argument `b`)"

    input_size, output_size = b.shape
    batch_size = np.prod(a.shape[:-1])

    a = scaling(a, backward=output_size**-0.5)
    b = scaling(b, backward=batch_size**-0.5)
    return scaling(a @ b, forward=input_size**-0.5)


def conv1d(
    input: tf.Tensor,  # pylint:disable=redefined-builtin
    filters: tf.Tensor,
    padding: str,
) -> tf.Tensor:
    """Scaling version of tf.nn.conv1d."""
    *batch_shape, input_length, input_size = input.shape
    filter_width, filter_input_size, output_size = filters.shape

    output_length = dict(
        SAME=input_length,
        VALID=input_length + 1 - filter_width,
    )[padding]
    n_groups = input_size // filter_input_size
    batch_size = np.prod(batch_shape)

    input = scaling(
        input,
        backward=(filter_width * output_length / input_length * output_size // n_groups)
        ** -0.5,
    )
    filters = scaling(filters, backward=(output_length * batch_size) ** -0.5)
    output = tf.nn.conv1d(input, filters, stride=1, padding=padding)
    return scaling(output, forward=(filter_width * input_size // n_groups) ** -0.5)


def relu(features: tf.Tensor) -> tf.Tensor:
    """A scaled ReLU nonlinearity."""
    return scaling(
        tf.nn.relu(features),
        forward=np.sqrt(2) / np.sqrt(1 - 1 / np.pi),
        backward=np.sqrt(2),
    )


###############################################################################
# Layers


class Initializers:
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

    def __init__(self, units: int, seed: Optional[int] = None):
        super().__init__(self)
        self.units = units
        self.kernel: tf.Variable = None
        self.kernel_initializer = Initializers.uniform(seed)
        self.bias: tf.Variable = None
        self.bias_initializer = keras.initializers.zeros()

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
        return matmul(inputs, self.kernel) + scaling(
            self.bias, backward=np.prod(inputs.shape[:-1]) ** -0.5
        )
