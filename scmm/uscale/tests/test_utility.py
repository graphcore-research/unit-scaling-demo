# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

from typing import Tuple

import numpy as np
import pytest
import tensorflow as tf
from tensorflow import keras

from .. import utility


def test_activation_tracker_example():
    # Sync with docstring
    layer = keras.layers.Dense(10)
    layer.build((None, 20))

    tracker = utility.ActivationTracker(layer)

    with tf.GradientTape() as tape:
        loss = tf.reduce_sum(layer(tf.zeros((3, 20))))
    grads_and_vars = zip(
        tape.gradient(loss, layer.trainable_variables), layer.trainable_variables
    )
    tracker.log_gradients(grads_and_vars)  # only needed for weight gradients

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
        # pylint:disable=too-few-public-methods
        def call(self, x: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
            return x, x

    layer = Duplicate()
    utility.ActivationTracker(layer)
    with pytest.raises(ValueError) as error:
        layer(tf.ones(3))
    assert "tuple" in str(error)


def test_activation_tracker_embedding_gradient():
    layer = keras.layers.Embedding(5, 8)
    layer.build((None,))

    tracker = utility.ActivationTracker(layer)
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


def test_printing(capsys):
    with tf.GradientTape() as tape:
        x = tf.constant([0, 1, 0, 1], dtype=tf.float32)
        tape.watch(x)
        y = tf.reduce_sum(utility.printing("x")(x) ** 2)
    tape.gradient(y, x)
    assert capsys.readouterr().err == "x forward std 0.5\nx backward std 1.0\n"
