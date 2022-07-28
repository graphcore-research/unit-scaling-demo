"""Custom ops for FP8 quantisation (without actually using FP8 for compute)."""

from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf
from tensorflow.python import ipu  # pylint:disable=no-name-in-module


def quantise_fp8(x: tf.Tensor, bias: int, format: str) -> tf.Tensor:
    """Quantise to FP8 & back again.

    x -- tf.Tensor -- float32 or float16

    bias -- int -- log scale, for example bias=0, range=(-240, 240),
                   bias=1, range=(-480, 480)

    format -- str -- either "1.4.3" or "1.5.2" (sign.exponent.mantissa bits)

    returns -- tf.Tensor -- same shape & dtype as `x`
    """

    build_dir = Path(__file__).parent / "build"
    (output,) = ipu.custom_ops.precompiled_user_op(
        [x],
        library_path=str(build_dir / "libquantisefp8.so"),
        attributes=f"{format} {bias}",
        outs=dict(output_types=[x.dtype], output_shapes=[x.shape]),
        name="quantise_fp8",
    )
    return output


@pytest.fixture(name="strategy", scope="module")
def _strategy() -> ipu.ipu_strategy.IPUStrategy:
    config = ipu.config.IPUConfig()
    config.auto_select_ipus = 1
    ipu.config.configure_ipu_system(config)
    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
        yield strategy


@pytest.mark.parametrize(
    "format,bias,dtype",
    [
        ("1.4.3", 0, np.float16),
        ("1.4.3", 1, np.float16),
        ("1.5.2", 0, np.float16),
        ("1.4.3", 0, np.float32),
    ],
)
def test_quantise_fp8(
    strategy: ipu.ipu_strategy.IPUStrategy, format: str, bias: int, dtype: np.dtype
) -> None:
    # pylint:disable=missing-function-docstring
    x = np.arange(-256, 256, 0.25, dtype=dtype)
    y = strategy.run(
        tf.function(lambda x: quantise_fp8(x, bias=bias, format=format)), [x]
    ).numpy()
    assert not np.any(np.isnan(y))
    assert len(set(y)) <= 256
