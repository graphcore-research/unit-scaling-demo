import numpy as np
import pytest
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
    layer.build((None, None, 7))
    assert layer.body.kernel.shape == (7, 7)
    assert layer(tf.ones((2, 3, 7))).shape == (2, 3, 7)


def test_ffn_layer():
    layer = layers.FFNLayer(3, (100, 200))
    layer.build((7,))
    assert layer.up.kernel.shape == (7, 21)  # type:ignore[union-attr]
    assert layer.down.kernel.shape == (21, 7)  # type:ignore[union-attr]
    assert layer(tf.ones((2, 3, 7))).shape == (2, 3, 7)


def test_softmax_cross_entropy():
    loss, n_ids = layers.softmax_cross_entropy(
        tf.ones((2, 3, 20)),
        tf.constant([[0, 9, 19], [2, 2, 2]]),
        tf.constant([[True, True, True], [True, False, False]]),
    )
    assert int(n_ids) == 4
    np.testing.assert_allclose(float(loss), np.log(20))


def test_pad_and_shift_layer():
    layer = layers.PadAndShiftLayer()
    layer.build((None, None, 11))
    assert layer.padding.shape == (11,)
    output = layer(tf.ones((3, 5, 11)))
    assert output.shape == (3, 5, 11)
    np.testing.assert_allclose(output[:, 0, :], 0)
    np.testing.assert_allclose(output[:, 1:, :], 1)

    with pytest.raises(ValueError):
        layers.PadAndShiftLayer().build((None, 11))
