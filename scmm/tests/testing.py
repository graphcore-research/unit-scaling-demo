from typing import Callable, Dict, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras

from ..pedal import utility


def assert_unit_scale(value: np.ndarray, tol: float, err_msg: str = "") -> None:
    """Check that a tensor has unit std."""
    np.testing.assert_allclose(np.std(value), 1, atol=tol, err_msg=err_msg)


def weight_shapes(layer: keras.layers.Layer) -> Dict[str, Tuple[int, ...]]:
    """A map of name to weight shape."""
    return {name: tuple(w.shape) for name, w in utility.named_weights(layer)}


def correlated_batch_random(
    random: np.random.RandomState, shape: Tuple[int, ...]
) -> np.ndarray:
    """Create a random tensor tiled over all 'batch' dimensions (all except last)."""
    # return random.normal(size=shape).astype(np.float32)
    return np.tile(random.normal(size=shape[-1]).astype(np.float32), shape[:-1] + (1,))


def output_and_gradients(
    layer: Callable[..., tf.Tensor], input_shape: Tuple[int, ...], seed: int
) -> Dict[str, np.ndarray]:
    """Randomly generate inputs and grads (unit norm), and return everything.

    Creates identical outputs and gradients across the batch axis, since
    realistic gradients are better modelled by perfect correlation than no
    correlation.
    """
    random = np.random.Generator(np.random.PCG64(seed))
    inputs = tf.constant(correlated_batch_random(random, input_shape))
    with tf.GradientTape() as tape:
        tape.watch(inputs)
        outputs = layer(inputs)
    grad_outputs = correlated_batch_random(random, outputs.shape)
    gradient_tensors = {f"grad_{k}": v for k, v in utility.named_weights(layer)}
    gradient_tensors["grad_inputs"] = inputs
    gradients = tape.gradient(outputs, gradient_tensors, grad_outputs)
    return dict(inputs=inputs, outputs=outputs, grad_outputs=grad_outputs, **gradients)
