# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import numpy as np
import pytest

from .. import datasets, models, training
from ..pedal import xpu


@pytest.mark.parametrize(
    "optimiser",
    [
        training.AdamW(0.1, learning_rate_decay=0.0),
        training.SgdM(0.1, learning_rate_decay=0.0, momentum=0.9),
    ],
    ids=lambda s: s.kind,
)
def test_training(optimiser: training.Optimiser):
    data_sequence = np.arange(100) % 3
    data = datasets.Data(
        ("a", "b", "c"),
        dict(train=data_sequence, valid=data_sequence, test=data_sequence),
    )
    with xpu.context(xpu.CpuSettings()) as context:
        model = models.Model(
            models.Settings(
                vocab_size=len(data.vocab),
                hidden_size=32,
                depth=2,
                residual=None,
                sequence=models.Conv(2, groups=1),
                token=None,
                dtype="float32",
                seed=100,
            ),
            unit_scale=False,
        )
        settings = training.Settings(
            datasets.BatchSettings(2, 8, 2, loop_seed=200),
            steps=10,
            valid_interval=None,
            optimiser=optimiser,
            loss_scale=1e3,
        )
        log = list(training.train(model, data, context, settings, unit_scale=False))
        train_log = [line for line in log if line["kind"] == "train_step"]
        assert 0.5 * np.log(3) < train_log[0]["loss"]
        assert train_log[-1]["loss"] < 0.01 * np.log(3)
