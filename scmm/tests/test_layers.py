from typing import List, Optional

import numpy as np
import pytest
import tensorflow as tf
from tensorflow import keras

from .. import layers
from . import testing


def test_batched_gather():
    tables = tf.reshape(tf.range(2 * 3 * 4), (2, 3, 4))
    indices = tf.constant([[0, 0, 3], [2, 2, 3]])
    np.testing.assert_equal(
        np.array(layers.batched_gather(tables, indices)),
        [[0 + 0, 4 + 0, 8 + 3], [12 + 2, 16 + 2, 20 + 3]],
    )


# Also tests layers.LayerNormalization
@pytest.mark.parametrize(
    ["norm_type", "alpha"], [(None, 0.5), ("pre", None), ("post", 0.1)]
)
def test_residual_layer(norm_type: Optional[str], alpha: Optional[float]):
    layer = layers.ResidualLayer(
        keras.layers.Dense(7), norm_type=norm_type, alpha=alpha
    )
    layer.build((None, None, 7))
    assert layer.body.kernel.shape == (7, 7)
    assert layer(tf.ones((2, 3, 7))).shape == (2, 3, 7)


def test_ffn_layer():
    layer = layers.FFNLayer(3, seeds=(100, 200))
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


####################
# Attention


def test_sinusoid_embedding():
    embedding = layers.sinusoid_embedding(
        sequence_length=32, frequencies=6, max_period=16
    )
    assert embedding.shape == (32, 6)
    np.testing.assert_allclose(embedding[:, 0], 0, atol=1e-6)
    np.testing.assert_allclose(
        embedding[:, 1], np.cos(np.pi * np.arange(32)), atol=1e-6
    )
    np.testing.assert_allclose(
        embedding[:, -2], np.sin(2 * np.pi / 16 * np.arange(32)), atol=1e-6
    )
    np.testing.assert_allclose(
        embedding[:, -1], np.cos(2 * np.pi / 16 * np.arange(32)), atol=1e-6
    )


def test_relative_causal_reshape():
    scores = tf.constant(
        [
            [1, 2, 3, 4],
            [11, 12, 13, 14],
            [21, 22, 23, 24],
            [31, 32, 33, 34],
        ]
    )
    attention = layers.relative_causal_reshape(scores)
    np.testing.assert_equal(
        attention.numpy(),
        [
            [1, 0, 0, 0],
            [12, 11, 0, 0],
            [23, 22, 21, 0],
            [34, 33, 32, 31],
        ],
    )


def test_multi_head_attention():
    random = np.random.Generator(np.random.PCG64(387232))
    layer = layers.MultiHeadAttention(
        heads=5, head_size=4, frequencies=13, max_period=16, seeds=(287, 918, 734)
    )
    inputs = random.normal(size=(11, 7, 8))
    result = layer(inputs)

    # Hard to check too much here - just make sure it's broadly sensible
    assert result.shape == inputs.shape
    np.testing.assert_array_less(np.std(result, axis=-1), 10)
    assert testing.weight_shapes(layer) == {
        "qkv": (8, 3, 5, 4),
        "q_bias": (5, 4),
        "positional": (13, 5, 4),
        "out.kernel": (5 * 4, 8),
        "out.bias": (8,),
    }

    # Check out the causal masking, by perturbing inputs[i, i+1]
    # in which case inputs[i, :i+1] should be unchanged
    perturbed = layer(inputs + np.eye(11, 7, k=1)[..., np.newaxis])
    mask = np.tril(np.ones((11, 7)), k=0)[..., np.newaxis]
    np.testing.assert_allclose(perturbed * mask, result * mask)
    assert not np.allclose(perturbed, result)


####################
# RNN


def test_recurrent_highway_cell():
    random = np.random.Generator(np.random.PCG64(387232))
    layer = layers.RecurrentHighwayCell(
        hidden_size=16, rebias=1, seed=random.integers(10000)
    )
    layer.build((3, 8))
    assert testing.weight_shapes(layer) == dict(gates=(2, 24, 16), gates_bias=(2, 16))

    output = layer(random.random(size=(3, 8)), random.random(size=(3, 16)))
    assert output.shape == (3, 16)
    assert np.std(output) < 10


def test_rnn():
    layer = layers.RNN(layers.RecurrentHighwayCell(hidden_size=20, rebias=0, seed=439))
    out = testing.output_and_gradients(layer, (3, 5, 8), seed=1341)
    assert out["outputs"].shape == (3, 5, 20)


####################
# Optimizers


def _train_sample_model(optimizer: keras.optimizers.Optimizer) -> List[float]:
    random = np.random.Generator(np.random.PCG64(983532))
    xs = random.normal(size=(1000, 20))
    ys = xs @ random.normal(size=(20, 10))
    model = keras.layers.Dense(
        10, kernel_initializer=keras.initializers.GlorotUniform(seed=87321)
    )
    losses = []
    for _ in range(10):
        with tf.GradientTape() as tape:
            loss = keras.losses.mse(ys.flatten(), tf.reshape(model(xs), -1))
        optimizer.minimize(loss, model.trainable_variables, tape=tape)
        losses.append(float(loss))
    return losses


def test_sgdm():
    np.testing.assert_allclose(
        _train_sample_model(layers.SgdM(0.1, momentum=0.9)),
        _train_sample_model(keras.optimizers.SGD(0.1, momentum=0.9)),
        atol=1e-4,
    )


def test_adamw():
    reference_loss = _train_sample_model(keras.optimizers.Adam(0.1))
    custom_loss = _train_sample_model(layers.AdamW(0.1, weight_decay=0))
    decay_loss = _train_sample_model(layers.AdamW(0.1, weight_decay=0.1))

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


def test_optimiser_decay_and_vector_learning_rate():
    for optimiser, learning_rate in [(layers.SgdM, 1.0), (layers.AdamW, 0.1)]:
        log = _train_sample_model(
            optimiser(
                learning_rate, learning_rate_decay=0.1, scale_vector_learning_rate=True
            )
        )
        assert log[-1] < log[0] / 2, log
