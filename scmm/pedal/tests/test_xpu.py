# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

from typing import Dict, List

import numpy as np
import pytest
import tensorflow as tf
from tensorflow import keras

from .. import xpu

SETTINGS: List[xpu.Settings] = [
    xpu.CpuSettings(compile=False),
    xpu.CpuSettings(compile=True),
]
if xpu.IPU:
    SETTINGS.extend(
        [
            xpu.IpuSettings(iterations_per_loop=1),
            xpu.IpuSettings(
                iterations_per_loop=4,
                available_memory_proportion=0.2,
                stochastic_rounding=True,
            ),
        ]
    )


@pytest.mark.parametrize("settings", SETTINGS)
def test_context(settings: xpu.Settings):
    traces = 0

    def model(x: tf.Tensor) -> Dict[str, tf.Tensor]:
        nonlocal traces
        traces += 1
        return dict(y=2 * x)

    data = (dict(x=np.array(x, dtype=np.float32)) for x in range(5))

    with xpu.context(settings) as context:
        results = list(context.loop(model, data))
        np.testing.assert_equal(results, [dict(y=2 * x) for x in range(5)])
        if not (isinstance(settings, xpu.CpuSettings) and not settings.compile):
            assert traces == 1


@pytest.mark.parametrize(
    "settings",
    filter(
        None,
        [
            xpu.CpuSettings(compile=False),
            xpu.IpuSettings(iterations_per_loop=1) if xpu.IPU else None,
        ],
    ),
)
def test_outline(settings: xpu.Settings):
    with xpu.context(settings) as context:
        assert xpu.current_context() is context
        layer1 = keras.layers.Dense(10)
        layer2 = keras.layers.Dense(10)
        context.outline(layer1)
        context.outline(layer2)

        results = list(
            context.loop(
                lambda x: dict(y=layer2(layer1(x))),
                [dict(x=np.ones((1, 10), dtype=np.float32))],
            )
        )
        assert results[0]["y"].shape == (1, 10)
