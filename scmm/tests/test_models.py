import dataclasses

import numpy as np
import pytest

from .. import models

SIMPLE_SETTINGS = models.SimpleConv(
    unit_scale=False,
    seed=100,
    vocab_size=100,
    hidden_size=8,
    depth=2,
    kernel_size=5,
)
SIMPLE_UNIT_SCALE_SETTINGS = dataclasses.replace(SIMPLE_SETTINGS, unit_scale=True)
RESIDUAL_SETTINGS = models.ResidualConv(
    unit_scale=False,
    seed=100,
    vocab_size=100,
    hidden_size=8,
    depth=2,
    kernel_size=5,
    group_size=4,
    ffn_multiple=1.5,
)


@pytest.mark.parametrize(
    "settings", [SIMPLE_SETTINGS, SIMPLE_UNIT_SCALE_SETTINGS, RESIDUAL_SETTINGS]
)
def test_model(settings: models.Settings):
    batch_sequences = 3
    sequence_length = 12
    random = np.random.Generator(np.random.PCG64(200))
    tokens = random.integers(
        settings.vocab_size, size=(batch_sequences, sequence_length)
    )
    mask = random.random(size=(batch_sequences, sequence_length)) < 0.9

    model = models.Model(settings)
    result = model.run(tokens=tokens, mask=mask)
    assert 0 < float(result["loss"]) < 1.5 * np.log(settings.vocab_size)
    assert int(result["n_tokens"]) == np.sum(mask)

    # Same seed - expect same weights & results
    model2 = models.Model(settings)
    result2 = model2.run(tokens=tokens, mask=mask)
    np.testing.assert_allclose(float(result2["loss"]), float(result["loss"]))
    np.testing.assert_equal(model2.save(), model.save())


def test_model_load_save():
    # Change seed, expect different weights
    base = models.Model(SIMPLE_SETTINGS)
    other = models.Model(
        dataclasses.replace(SIMPLE_SETTINGS, seed=SIMPLE_SETTINGS.seed + 1)
    )
    base_weights = base.save()
    other_weights = other.save()
    assert any(np.any(other_weights[k] != base_weights[k]) for k in base_weights)

    with pytest.raises(ValueError) as error:
        other.load(dict(**base_weights, non_existent_weight=np.zeros(2)))
    assert "non_existent_weight" in str(error)

    other.load(base_weights)
    np.testing.assert_equal(other.save(), base_weights)
