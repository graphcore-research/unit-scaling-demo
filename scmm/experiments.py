"""Top-level experiment running."""

import contextlib
import copy
import dataclasses
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, Optional

import numpy as np
import wandb

from . import datasets, models, training
from .pedal import utility, xpu


@contextlib.contextmanager
def log_wandb() -> Generator[utility.Logger, None, None]:
    """Log to weights & biases."""

    def _log(item: Dict[str, Any]) -> None:
        if item["kind"] == "settings":
            Path("out").mkdir(exist_ok=True, parents=True)
            wandb.init(
                config=utility.remove_keys(item, "kind"),
                dir="out",
                project="scaled-matmuls",
            )
        elif item["kind"] == "stats":
            wandb.run.summary.update(  # type:ignore[union-attr]
                utility.remove_keys(item, "kind")
            )
        else:
            wandb.log(utility.remove_keys(item, "step"), step=item["step"])

    yield _log

    # Otherwise we hang (when started in a subprocess from a sweep)
    wandb.finish()


@contextlib.contextmanager
def log_jsonl(path: Path) -> Generator[utility.Logger, None, None]:
    """Log to file."""
    path.parent.mkdir(exist_ok=True, parents=True)
    with path.open("w") as f:
        yield lambda item: print(json.dumps(item, default=utility.to_jsonable), file=f)


def log_checkpoint(path: Path, model: models.Model) -> utility.Logger:
    """Save model checkpoints whenever validation runs."""

    def log(item: Dict[str, Any]) -> None:
        if item["kind"] == "eval_valid":
            np.savez(path, step=item["step"], **model.save())

    return log


def log_stderr(item: Dict[str, Any]) -> None:
    """Log to terminal."""
    print(
        str(item) + " " * 20,
        file=sys.stderr,
        end="\r" if item["kind"] == "train_step" else "\n",
    )


@dataclass
class DataSettings:
    """Dataset settings."""

    path: Path
    kind: str = "wikitext-103-raw"


@dataclass
class OutputSettings:
    """Output control settings."""

    wandb: bool
    log: Optional[Path]
    checkpoint: Optional[Path]


@dataclass
class Settings:
    """Top-level settings."""

    data: DataSettings
    model: models.Settings
    training: training.Settings
    target: xpu.Settings
    output: OutputSettings
    metadata: Dict[str, Any]
    seed: int

    def set_defaults(self, data: datasets.Data) -> None:
        """Fill in all optional fields."""
        # Seeds
        if self.seed is None:
            self.seed = int(np.random.SeedSequence().generate_state(1)[0])
        model_seed, batching_seed = utility.split_seed(self.seed, 2)
        if self.model.seed is None:
            self.model.seed = model_seed
        if self.training.batch.loop_seed is None:
            self.training.batch.loop_seed = batching_seed

        # Model
        if self.model.vocab_size is None:
            self.model.vocab_size = len(data.vocab)

        # Metadata
        if "SSUB_UID" in os.environ:
            self.metadata.setdefault("ssub_id", os.environ["SSUB_UID"])
        if "SLURM_JOB_ID" in os.environ:
            self.metadata.setdefault("slurm_job_id", os.environ["SLURM_JOB_ID"])


def _loggers(settings: Settings, model: models.Model) -> Iterable[utility.Logger]:
    yield log_stderr
    if settings.output.wandb:
        yield log_wandb()
    if settings.output.log:
        yield log_jsonl(settings.output.log)
    if settings.output.checkpoint:
        yield log_checkpoint(settings.output.checkpoint, model)


def _settings_line(settings: Settings) -> Dict[str, Any]:
    return dict(
        kind="settings",
        **utility.remove_keys(dataclasses.asdict(settings), "output"),
    )


def run(settings: Settings) -> None:
    """Run an experiment, logging results as requested."""
    data = datasets.load_character(
        settings.data.path, train="train.txt", valid="valid.txt", test="test.txt"
    )
    settings = copy.deepcopy(settings)
    settings.set_defaults(data)

    with xpu.context(settings.target) as context:
        model = models.Model(settings.model)
        with utility.logging(*_loggers(settings, model)) as log:
            log(_settings_line(settings))
            log(dict(kind="stats", **model.weight_stats()))
            for item in training.train(model, data, context, settings.training):
                log(item)
