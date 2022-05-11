import contextlib
import datetime
import json
import multiprocessing
from pathlib import Path
from typing import Dict

import numpy as np
import pytest
from tensorflow import keras

from .. import utility


def test_split_seed():
    seeds = utility.split_seed(123456789, 3)
    assert len(seeds) == 3
    assert len(set(seeds)) == 3


def test_remove_keys():
    assert utility.remove_keys(dict(a=1, b=2, c=3), "b", "d") == dict(a=1, c=3)


def test_to_jsonable():
    t0 = datetime.datetime.now()
    data = json.dumps(
        dict(time=t0, array=np.ones(2), path=Path("fake/path")),
        default=utility.to_jsonable,
    )
    assert json.loads(data) == dict(time=t0.isoformat(), array=[1, 1], path="fake/path")

    with pytest.raises(TypeError) as exc:
        json.dumps(dict(obj=object()), default=utility.to_jsonable)
    assert "object" in str(exc)


def test_logging():
    history = []

    @contextlib.contextmanager
    def logger():
        history.append("pre")
        yield history.append
        history.append("post")

    with utility.logging(
        logger(), lambda line: history.append(f"lambda {line}")
    ) as log:
        log(123)
        log(456)

    assert history == ["pre", 123, "lambda 123", 456, "lambda 456", "post"]


def test_named_layers_and_weights():
    class TestModel(keras.layers.Layer):
        def __init__(self):
            super().__init__()
            self.projection = keras.layers.Dense(20)
            self.projection.build((10,))
            self.transforms = [
                keras.layers.LayerNormalization(),
                keras.layers.Dense(20, use_bias=False),
            ]
            for transform in self.transforms:
                transform.build((20,))
            self.final_bias = self.add_weight(
                name="final_bias", shape=(7,), initializer="zeros"
            )

    model = TestModel()

    assert {k: type(v).__name__ for k, v in utility.named_layers(model)} == {
        "": "TestModel",
        "projection": "Dense",
        "transforms.0": "LayerNormalization",
        "transforms.1": "Dense",
    }

    assert {k: tuple(v.shape) for k, v in utility.named_weights(model)} == {
        "projection.kernel": (10, 20),
        "projection.bias": (20,),
        "transforms.0.beta": (20,),
        "transforms.0.gamma": (20,),
        "transforms.1.kernel": (20, 20),
        "final_bias": (7,),
    }
    assert dict(utility.named_weights(model, recursive=False)).keys() == {"final_bias"}


def _stub(x: int) -> Dict[str, int]:
    if x % 2 == 0:
        raise ValueError("x is even")
    return dict(y=5 * x)


def test_run_in_subprocess():
    assert utility.run_in_subprocess(_stub, x=3) == dict(y=15)
    with pytest.raises(multiprocessing.ProcessError):
        utility.run_in_subprocess(_stub, x=4)
