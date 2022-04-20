from typing import List

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


def test_isotropic():
    layer = layers.Isotropic(
        dense=keras.layers.Dense(15), norm=keras.layers.LayerNormalization()
    )
    layer.build((None, 15))
    assert layer.dense.built
    assert layer.norm.built
    random = np.random.Generator(np.random.PCG64(seed=500))
    assert layer(random.normal(size=(7, 15))).shape == (7, 15)


def test_adamw():
    random = np.random.Generator(np.random.PCG64(12345))
    xs = random.normal(size=(1000, 20))
    ys = xs @ random.normal(size=(20, 10))

    reference_loss: List[float] = []
    custom_loss: List[float] = []
    decay_loss: List[float] = []
    for optimizer, losses in [
        (keras.optimizers.Adam(0.1), reference_loss),
        (layers.AdamW(0.1, weight_decay=0), custom_loss),
        (layers.AdamW(0.1, weight_decay=0.1), decay_loss),
    ]:
        model = keras.layers.Dense(
            10, kernel_initializer=keras.initializers.GlorotUniform(seed=67890)
        )
        for _ in range(10):
            with tf.GradientTape() as tape:
                loss = keras.losses.mse(ys.flatten(), tf.reshape(model(xs), -1))
            optimizer.minimize(loss, model.trainable_variables, tape=tape)
            losses.append(float(loss))
    np.testing.assert_allclose(custom_loss, reference_loss, atol=1e-4)
    assert (
        1e-3 + reference_loss[-1] < decay_loss[-1]
    ), "decayed should be slightly worse"


def test_adamw_indexed_slices():
    optimizer = layers.AdamW(0.1)
    table = tf.Variable(np.zeros((5, 3), dtype=np.float32))
    indices = tf.constant([0, 2, 2, 4])
    optimizer.minimize(
        lambda: tf.reduce_sum(tf.gather(table, indices)), var_list=[table]
    )
    np.testing.assert_equal(table[:, 0].numpy() < 0, [True, False, True, False, True])
