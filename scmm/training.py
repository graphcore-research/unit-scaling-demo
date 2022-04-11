"""Top-line training logic."""

import dataclasses
import datetime
import itertools as it
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

import tensorflow as tf
from tensorflow import keras

from . import datasets, models
from .pedal import xpu


@dataclass
class Settings:
    """Training settings."""

    batch: datasets.BatchSettings
    steps: int
    valid_interval: Optional[int]
    learning_rate: float
    beta_1: float = 0.9
    beta_2: float = 0.999
    optimiser: str = "adam"


def eval_summary(results: Iterable[datasets.Batch]) -> Dict[str, float]:
    """Summarise evaluation results."""

    total_tokens, loss = 0, 0.0
    for result in results:
        n_tokens = int(result["n_tokens"])
        total_tokens += n_tokens
        loss += float(result["loss"]) * n_tokens
    return dict(loss=loss / total_tokens, n_tokens=total_tokens)


def train(
    model: models.Model, data: datasets.Data, context: xpu.Context, settings: Settings
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
        for part in ["valid", "train"]:
            batch_settings = settings.batch
            if part == "valid":
                batch_settings = dataclasses.replace(batch_settings, loop_seed=None)
            batches = data.batches(part, batch_settings)
            if part == "train":
                batches = it.islice(
                    batches,
                    1 + data.parts["valid"].size // settings.batch.target_tokens,
                )
            results = eval_summary(context.loop(model.run, batches))
            results = {f"{part}_{k}": v for k, v in results.items()}
            yield _log(f"eval_{part}", step, results)

    def _training_step(**batch: tf.Tensor) -> Dict[str, tf.Tensor]:
        with tf.GradientTape() as tape:
            result = model.run(**batch)
        gradients = tape.gradient(result["loss"], model.trainable_variables)
        optimiser.apply_gradients(zip(gradients, model.trainable_variables))
        return result

    train_steps = iter(
        context.loop(_training_step, data.batches("train", settings.batch))
    )
    for step in it.count():
        if step >= settings.steps:
            if settings.valid_interval is not None:
                yield from _validate(step)
            break
        if settings.valid_interval is not None and step % settings.valid_interval == 0:
            yield from _validate(step)
        results = next(train_steps)  # pylint:disable=stop-iteration-return
        yield _log("train_step", step, {k: v.tolist() for k, v in results.items()})
