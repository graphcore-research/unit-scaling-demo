from typing import Dict, List

import numpy as np
import pytest
import tensorflow as tf

from .. import xpu

SETTINGS: List[xpu.Settings] = [
    xpu.CpuSettings(compile=False),
    xpu.CpuSettings(compile=True),
]
if xpu.IPU:
    SETTINGS.extend(
        [
            xpu.IpuSettings(iterations_per_loop=1),
            xpu.IpuSettings(iterations_per_loop=4, available_memory_proportion=0.2),
        ]
    )


@pytest.mark.parametrize("settings", SETTINGS)
def test_context(settings):
    traces = 0

    def model(x: tf.Tensor) -> Dict[str, tf.Tensor]:
        nonlocal traces
        traces += 1
        return dict(y=2 * x)

    data = (dict(x=np.array(x, dtype=np.float32)) for x in range(5))

    with xpu.context(settings) as context:
        results = list(context.loop(model, data))
        np.testing.assert_equal(results, [dict(y=2 * x) for x in range(5)])
        if settings.type == "ipu" or settings.compile:
            assert traces == 1
