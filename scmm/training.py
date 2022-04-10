"""Top-line training logic."""

import dataclasses
import datetime
import itertools as it
from dataclasses import dataclass
from typing import Any, Dict, Iterable

import numpy as np
import tensorflow as tf
from tensorflow import keras

from . import datasets, models


@dataclass
class Settings:
    """Training settings."""

    batch: datasets.BatchSettings
    steps: int
    valid_interval: int
    learning_rate: float
    beta_1: float = 0.9
    beta_2: float = 0.999
    optimiser: str = "adam"


def evaluate(
    model: models.Model, batches: Iterable[datasets.Batch]
) -> Dict[str, float]:
    """Evaluate a model."""

    total_tokens, loss = 0, 0.0
    for batch in batches:
        result = model.run(**batch)
        n_tokens = int(result["n_tokens"])
        total_tokens += n_tokens
        loss += float(result["loss"]) * n_tokens
    return dict(loss=loss / total_tokens, n_tokens=total_tokens)


def train(
    model: models.Model, data: datasets.Data, settings: Settings
) -> Iterable[Dict[str, Any]]:
    """Train a model."""

    assert (
        settings.batch.loop_seed is not None
    ), "please specify a seed for training batches"
    assert settings.optimiser == "adam"
    optimiser = keras.optimizers.Adam(
        learning_rate=settings.learning_rate,
        beta_1=settings.beta_1,
        beta_2=settings.beta_2,
    )

    def _log(kind: str, step: int, data: Dict[str, Any]) -> Dict[str, Any]:
        return dict(kind=kind, step=step, time=datetime.datetime.now(), **data)

    def _validate(step: int) -> Iterable[Dict[str, Any]]:
        valid_batches = data.batches(
            "valid", dataclasses.replace(settings.batch, loop_seed=None)
        )
        yield _log(
            "eval_valid",
            step,
            {f"valid_{k}": v for k, v in evaluate(model, valid_batches).items()},
        )
        train_batches = it.islice(
            data.batches("train", settings.batch),
            1 + data.parts["valid"].size // settings.batch.target_tokens,
        )
        yield _log(
            "eval_train",
            step,
            {f"train_{k}": v for k, v in evaluate(model, train_batches).items()},
        )

    def _training_step(**batch: tf.Tensor) -> Dict[str, tf.Tensor]:
        with tf.GradientTape() as tape:
            result = model.run(**batch)
        gradients = tape.gradient(result["loss"], model.trainable_variables)
        optimiser.apply_gradients(zip(gradients, model.trainable_variables))
        return result

    step = 0
    for step, batch in enumerate(
        it.islice(data.batches("train", settings.batch), settings.steps)
    ):
        if step % settings.valid_interval == 0:
            yield from _validate(step)
        log = {k: np.array(v).tolist() for k, v in _training_step(**batch).items()}
        yield _log("train_step", step, log)
    yield from _validate(step + 1)
