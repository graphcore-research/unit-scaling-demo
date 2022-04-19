"""Core components for unit-scaling activations and gradients."""

import collections
import functools
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras

from . import layers
from .pedal import utility

####################
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


####################
# Ops


def scaling(
    forward: Optional[float] = None, backward: Optional[float] = None
) -> Callable[[tf.Tensor], tf.Tensor]:
    """Perform arbitary *seperate* scaling in the forward and backward passes.

    E.g.

        y = scaling(forward=2, backward=3)(x)

    """

    @tf.custom_gradient  # type:ignore[misc]
    def operation(input: tf.Tensor) -> tf.Tensor:  # pylint:disable=redefined-builtin
        def grad(upstream: tf.Tensor) -> tf.Tensor:
            grad_input = upstream
            if backward is not None:
                if isinstance(upstream, tf.IndexedSlices):
                    grad_input = tf.IndexedSlices(
                        values=upstream.values * backward,
                        indices=upstream.indices,
                        dense_shape=upstream.dense_shape,
                    )
                else:
                    grad_input *= backward
            return grad_input

        output = input
        if forward is not None:
            output *= forward

        return output, grad

    return operation  # type:ignore[no-any-return]


def matmul(a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
    """Scaling version of tf.matmul (where b must be 2D)."""
    assert len(b.shape) == 2, "matmul requires 2D rhs (argument `b`)"

    input_size, output_size = b.shape
    batch_size = np.prod(a.shape[:-1])

    a = scaling(backward=output_size**-0.5)(a)
    b = scaling(backward=batch_size**-0.5)(b)
    return scaling(forward=input_size**-0.5)(a @ b)


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
        backward=(filter_width * output_length / input_length * output_size // n_groups)
        ** -0.5
    )(input)
    filters = scaling(backward=(output_length * batch_size) ** -0.5)(filters)
    output = tf.nn.conv1d(input, filters, stride=1, padding=padding)
    return scaling(forward=(filter_width * input_size // n_groups) ** -0.5)(output)


def relu(features: tf.Tensor) -> tf.Tensor:
    """A scaled ReLU nonlinearity, shifted to have zero mean for unit normal inputs."""
    return scaling(forward=np.sqrt(2 / (1 - 1 / np.pi)), backward=np.sqrt(2))(
        tf.nn.relu(features) - 1 / np.sqrt(2 * np.pi)
    )


def add_bias(features: tf.Tensor, bias: tf.Tensor) -> tf.Tensor:
    """Add a bias (which should be zero-initialized), with a scaled backward pass."""
    assert len(bias.shape) == 1, "bias should be 1D"
    batch_size = np.prod(features.shape[:-1])
    return features + scaling(backward=batch_size**-0.5)(bias)


def softmax_cross_entropy(
    scores: tf.Tensor, ids: tf.Tensor, mask: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Compute masked softmax cross entropy loss.

    Note that we abandon unit scaling in the forward pass, since this is
    designed as a final loss term.

    returns -- (average_loss, n_items)
    """
    logp = tf.nn.log_softmax(scores)
    # Better compilation on IPU vs `tf.gather(logp, ids, batch_dims=2)`
    target_logp = layers.batched_gather(logp, ids)
    total_loss = tf.reduce_sum(tf.cast(mask, target_logp.dtype) * -target_logp)
    n_ids = tf.reduce_sum(tf.cast(mask, tf.int32))
    n_classes = scores.shape[1]
    loss = scaling(backward=np.prod(mask.shape) * n_classes / np.sqrt(n_classes - 1))(
        total_loss / tf.cast(n_ids, total_loss.dtype)
    )
    return loss, n_ids


####################
# Layers


class initializers:  # pylint:disable=invalid-name
    """Unit-variance initializers."""

    @staticmethod
    def uniform(seed: Optional[int]) -> keras.initializers.Initializer:
        """Uniform distribution (symmetric about 0)."""
        return keras.initializers.RandomUniform(-np.sqrt(3), np.sqrt(3), seed=seed)

    @staticmethod
    def normal(seed: Optional[int]) -> keras.initializers.Initializer:
        """Standard normal distribution."""
        return keras.initializers.RandomNormal(stddev=1, seed=seed)


class activations:  # pylint:disable=invalid-name,too-few-public-methods
    """Unit-variance activations."""

    linear = staticmethod(tf.identity)
    relu = staticmethod(relu)

    @classmethod
    def get(cls, name: Optional[str]) -> Callable[[tf.Tensor], tf.Tensor]:
        """Select an activation function by name (default: linear)."""
        return getattr(cls, name or "linear")  # type:ignore[no-any-return]


class Dense(keras.layers.Layer):  # type:ignore[misc]
    """A scaled (and more restrictive) version of keras.layers.Dense."""

    def __init__(
        self, units: int, activation: Optional[str] = None, seed: Optional[int] = None
    ):
        super().__init__(self)
        self.units = units
        self.kernel: tf.Variable = None
        self.kernel_initializer = initializers.uniform(seed)
        self.bias: tf.Variable = None
        self.bias_initializer = keras.initializers.zeros()
        self.activation = activations.get(activation)

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
        return self.activation(add_bias(matmul(inputs, self.kernel), self.bias))


class CausalConv1D(keras.layers.Layer):  # type:ignore[misc]
    """A scaled causal 1D convolution."""

    # pylint:disable=too-many-instance-attributes

    def __init__(
        self,
        filters: int,
        kernel_size: int,
        groups: Optional[int] = None,
        activation: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.groups = groups or 1
        if filters % self.groups != 0:
            raise ValueError(
                f"Filters ({filters}) must be evenly divisible by groups ({self.groups})"
            )
        self.kernel: tf.Variable = None
        self.kernel_initializer = initializers.uniform(seed)
        self.bias: tf.Variable = None
        self.bias_initializer = keras.initializers.zeros()
        self.activation = activations.get(activation)

    def build(self, input_shape: tf.TensorShape) -> None:
        super().build(input_shape)
        input_features = input_shape[-1]
        if input_features % self.groups != 0:
            raise ValueError(
                f"Input feature size ({input_features}) must be evenly divisible"
                f" by groups ({self.groups})"
            )
        self.kernel = self.add_weight(
            "kernel",
            shape=(self.kernel_size, input_shape[-1] // self.groups, self.filters),
            initializer=self.kernel_initializer,
        )
        self.bias = self.add_weight(
            "bias", shape=self.filters, initializer=self.bias_initializer
        )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        padded = tf.pad(
            inputs,
            [(0, 0), (self.kernel_size - 1, 0), (0, 0)],
        )
        length = inputs.shape[1]
        # Scaling here to account for zero-padding
        outputs = scaling(
            forward=(length / (length + (1 - self.kernel_size) / 2)) ** 0.5
        )(conv1d(padded, self.kernel, padding="VALID"))
        return self.activation(add_bias(outputs, self.bias))


class Embedding(layers.Embedding):
    """A scaled embedding layer."""

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        # We don't need to worry about inputs scaling, as it is non-differentiable
        batch_size = np.prod(inputs.shape)
        return layers.gather_dense_gradients(
            scaling(backward=(self.table_size / batch_size) ** 0.5)(self.embeddings),
            inputs,
        )
