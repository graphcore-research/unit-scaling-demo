import numpy as np
import pytest

from .. import datasets, models, training
from ..pedal import xpu


@pytest.mark.parametrize(
    "optimiser", [training.AdamW(0.1), training.SgdM(0.1, 0.9)], ids=lambda s: s.kind
)
def test_training(optimiser: training.Optimiser):
    data_sequence = np.arange(100) % 3
    data = datasets.Data(("a", "b", "c"), dict(train=data_sequence))
    with xpu.context(xpu.CpuSettings()) as context:
        model = models.Model(
            models.SimpleConv(
                unit_scale=False,
                seed=100,
                vocab_size=len(data.vocab),
                hidden_size=32,
                depth=2,
                kernel_size=2,
                group_size=32,
            )
        )
        settings = training.Settings(
            datasets.BatchSettings(2, 8, 2, loop_seed=200),
            steps=10,
            valid_interval=None,
            optimiser=optimiser,
        )
        log = list(training.train(model, data, context, settings))
        assert 0.5 * np.log(3) < log[0]["loss"]
        assert log[-1]["loss"] < 0.01 * np.log(3)
