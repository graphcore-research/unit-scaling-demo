from typing import Tuple

import numpy as np
import pytest
import tensorflow as tf
from tensorflow import keras

from .. import uscale


def test_activation_tracker_example():
    # Sync with docstring
    layer = keras.layers.Dense(10)
    layer.build((None, 20))

    tracker = uscale.ActivationTracker(layer)

    with tf.GradientTape() as tape:
        loss = tf.reduce_sum(layer(tf.zeros((3, 20))))
    grads_and_vars = zip(
        tape.gradient(loss, layer.trainable_variables), layer.trainable_variables
    )
    tracker.log_gradients(grads_and_vars)  # only needed for weight gradients

    print({t.name: np.std(t.gradient) for t in tracker.trace})

    # Checks
    (trace,) = tracker.trace
    assert trace.name == ""
    assert trace.layer is layer
    assert trace.activation.shape == (1, 3, 10)
    assert trace.gradient.shape == (1, 3, 10)  # type:ignore[union-attr]

    weights = {t.name: t for t in trace.weights}
    assert weights["kernel"].activation.shape == (20, 10)
    assert weights["kernel"].gradient.shape == (1, 20, 10)  # type:ignore[union-attr]
    assert weights["bias"].activation.shape == (10,)
    assert weights["bias"].gradient.shape == (1, 10)  # type:ignore[union-attr]


def test_activation_tracker_multiple_outputs():
    class Duplicate(keras.layers.Layer):
        def call(self, x: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
            return x, x

    layer = Duplicate()
    uscale.ActivationTracker(layer)
    with pytest.raises(ValueError) as error:
        layer(tf.ones(3))
    assert "tuple" in str(error)


def test_activation_tracker_embedding_gradient():
    layer = keras.layers.Embedding(5, 8)
    layer.build((None,))

    tracker = uscale.ActivationTracker(layer)
    with tf.GradientTape() as tape:
        loss = tf.reduce_sum(layer(tf.constant([0, 3, 3])))
    grads_and_vars = zip(
        tape.gradient(loss, layer.trainable_variables), layer.trainable_variables
    )
    tracker.log_gradients(grads_and_vars)

    embedding_trace = tracker.trace[0].weights[0]
    assert embedding_trace.name == "embeddings"
    assert embedding_trace.gradient.shape == (1, 5, 8)  # type:ignore[union-attr]
    np.testing.assert_equal(
        embedding_trace.gradient[0, :, 0],  # type:ignore[index]
        [1, 0, 0, 2, 0],
    )
