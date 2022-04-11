import dataclasses

import numpy as np
import pytest
import tensorflow as tf

from .. import models


def test_batched_gather():
    tables = tf.reshape(tf.range(2 * 3 * 4), (2, 3, 4))
    indices = tf.constant([[0, 0, 3], [2, 2, 3]])
    np.testing.assert_equal(
        np.array(models.batched_gather(tables, indices)),
        [[0 + 0, 4 + 0, 8 + 3], [12 + 2, 16 + 2, 20 + 3]],
    )


SETTINGS = models.Settings(
    vocab_size=100,
    hidden_size=8,
    depth=2,
    kernel_size=5,
    seed=100,
)


def test_model():
    batch_sequences = 3
    sequence_length = 12
    random = np.random.Generator(np.random.PCG64(200))
    tokens = random.integers(
        SETTINGS.vocab_size, size=(batch_sequences, sequence_length)
    )
    mask = random.random(size=(batch_sequences, sequence_length)) < 0.9

    model = models.Model(SETTINGS)
    result = model.run(tokens=tokens, mask=mask)
    assert 0 < float(result["loss"]) < 1.5 * np.log(SETTINGS.vocab_size)
    assert int(result["n_tokens"]) == np.sum(mask)

    # Same seed - expect same weights & results
    model2 = models.Model(SETTINGS)
    result2 = model2.run(tokens=tokens, mask=mask)
    np.testing.assert_allclose(float(result2["loss"]), float(result["loss"]))
    np.testing.assert_equal(model2.save(), model.save())


def test_model_load_save():
    # Change seed, expect different weights
    base = models.Model(SETTINGS)
    other = models.Model(dataclasses.replace(SETTINGS, seed=SETTINGS.seed + 1))
    base_weights = base.save()
    other_weights = other.save()
    assert any(np.any(other_weights[k] != base_weights[k]) for k in base_weights)

    with pytest.raises(ValueError) as error:
        other.load(dict(**base_weights, non_existent_weight=np.zeros(2)))
    assert "non_existent_weight" in str(error)

    other.load(base_weights)
    np.testing.assert_equal(other.save(), base_weights)
