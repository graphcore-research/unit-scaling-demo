import dataclasses

import numpy as np
import pytest

from .. import models
from ..pedal import xpu

SETTINGS = models.Settings(
    vocab_size=100,
    hidden_size=8,
    depth=2,
    residual=None,
    sequence=models.Conv(kernel_size=5, groups=1),
    token=None,
    dtype="float32",
    seed=100,
)


@pytest.fixture
def cpu_context():
    with xpu.context(xpu.CpuSettings(compile=False)) as context:
        yield context


MODEL_SETTINGS = [
    SETTINGS,
    dataclasses.replace(
        SETTINGS,
        residual=models.Residual(norm=None, alpha=None),
        token=models.FFN(multiple=1.5),
    ),
    dataclasses.replace(
        SETTINGS,
        residual=models.Residual(norm="pre", alpha="mean"),
        sequence=models.Attention(heads=2, head_size=4, frequencies=16, max_period=16),
    ),
    dataclasses.replace(
        SETTINGS,
        residual=models.Residual(norm="post", alpha=0.5),
        sequence=models.RNN(rebias=1),
    ),
]


@pytest.mark.parametrize(
    "settings,unit_scale",
    [
        (dataclasses.replace(settings, dtype=dtype), unit_scale)
        for settings in MODEL_SETTINGS
        for dtype in ["float32", "float16"]
        for unit_scale in [None, "0.3"]
    ],
    ids=repr,
)
def test_model(cpu_context: xpu.Context, settings: models.Settings, unit_scale: bool):
    if unit_scale and settings.residual is not None and settings.residual.alpha is None:
        pytest.skip("unsupported combination")

    batch_sequences = 3
    sequence_length = 12
    random = np.random.Generator(np.random.PCG64(200))
    tokens = random.integers(
        settings.vocab_size, size=(batch_sequences, sequence_length)
    )
    mask = random.random(size=(batch_sequences, sequence_length)) < 0.9

    model = models.Model(settings, unit_scale=unit_scale)
    result = model.run(tokens=tokens, mask=mask)
    assert 0 < float(result["loss"]) < 1.5 * np.log(settings.vocab_size)
    assert int(result["n_tokens"]) == np.sum(mask)

    # Same seed - expect same weights & results
    model2 = models.Model(settings, unit_scale=unit_scale)
    result2 = model2.run(tokens=tokens, mask=mask)
    np.testing.assert_allclose(float(result2["loss"]), float(result["loss"]))
    np.testing.assert_equal(model2.save(), model.save())


def test_model_load_save(cpu_context: xpu.Context):
    # Change seed, expect different weights
    base = models.Model(SETTINGS, unit_scale=False)
    other = models.Model(
        dataclasses.replace(SETTINGS, seed=SETTINGS.seed + 1), unit_scale=False
    )
    base_weights = base.save()
    other_weights = other.save()
    assert any(np.any(other_weights[k] != base_weights[k]) for k in base_weights)

    with pytest.raises(ValueError) as error:
        other.load(dict(**base_weights, non_existent_weight=np.zeros(2)))
    assert "non_existent_weight" in str(error)

    other.load(base_weights)
    np.testing.assert_equal(other.save(), base_weights)
