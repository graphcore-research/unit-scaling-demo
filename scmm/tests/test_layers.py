import numpy as np
import tensorflow as tf
from tensorflow import keras

from .. import layers


def test_batched_gather():
    tables = tf.reshape(tf.range(2 * 3 * 4), (2, 3, 4))
    indices = tf.constant([[0, 0, 3], [2, 2, 3]])
    np.testing.assert_equal(
        np.array(layers.batched_gather(tables, indices)),
        [[0 + 0, 4 + 0, 8 + 3], [12 + 2, 16 + 2, 20 + 3]],
    )


def test_pre_norm_residual_layer():
    layer = layers.PreNormResidualLayer(keras.layers.Dense(7))
    layer.build((7,))
    assert layer.body.kernel.shape == (7, 7)
    assert layer(tf.ones((2, 3, 7))).shape == (2, 3, 7)


def test_ffn_layer():
    layer = layers.FFNLayer(3, (100, 200))
    layer.build((7,))
    layer.build((7,))
    assert layer.up.kernel.shape == (7, 21)  # type:ignore[union-attr]
    assert layer.down.kernel.shape == (21, 7)  # type:ignore[union-attr]
    assert layer(tf.ones((2, 3, 7))).shape == (2, 3, 7)
