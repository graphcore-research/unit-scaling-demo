"""Top-level experiment running."""

import contextlib
import copy
import dataclasses
import itertools as it
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, Optional, Tuple

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
                reinit=True,
            )
        elif item["kind"] == "stats":
            wandb.run.summary.update(  # type:ignore[union-attr]
                utility.remove_keys(item, "kind")
            )
        elif item["kind"] == "train_step":
            pass  # skip training steps (too large)
        else:
            wandb.log(item, step=item["step"])

    try:
        yield _log
    except Exception as exc:
        wandb.run.summary.update(dict(error=repr(exc)))  # type:ignore[union-attr]
        wandb.finish(1)
        raise
    else:
        # Always call finish(), otherwise we hang (when started in a subprocess from a sweep)
        wandb.finish(0)


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
    if item["kind"] == "train_step":
        return
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

    stderr: bool
    wandb: bool
    log: Optional[Path]
    checkpoint: Optional[Path]


@dataclass
class Settings:
    """Top-level settings."""

    data: DataSettings
    model: models.Settings
    training: training.Settings
    unit_scale: Optional[str]
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


def _loggers(settings: OutputSettings, model: models.Model) -> Iterable[utility.Logger]:
    if settings.stderr:
        yield log_stderr
    if settings.wandb:
        yield log_wandb()
    if settings.log:
        yield log_jsonl(settings.log)
    if settings.checkpoint:
        yield log_checkpoint(settings.checkpoint, model)


def _settings_line(settings: Settings) -> Dict[str, Any]:
    return dict(
        kind="settings",
        **utility.remove_keys(dataclasses.asdict(settings), "output"),
    )


def run(settings: Settings) -> Dict[str, Any]:
    """Run an experiment, logging results as requested."""
    data = datasets.load_character(
        settings.data.path, train="train.txt", valid="valid.txt", test="test.txt"
    )
    settings = copy.deepcopy(settings)
    settings.set_defaults(data)
    assert settings.unit_scale in {None, "0.3"}

    last_eval_valid: Optional[Dict[str, Any]] = None
    with xpu.context(settings.target) as context:
        model = models.Model(settings.model, unit_scale=bool(settings.unit_scale))
        with utility.logging(*_loggers(settings.output, model)) as log:
            log(_settings_line(settings))
            log(dict(kind="stats", **model.weight_stats()))
            for item in training.train(
                model, data, context, settings.training, bool(settings.unit_scale)
            ):
                log(item)
                if item["kind"] == "eval_valid":
                    last_eval_valid = item
    assert last_eval_valid is not None
    return last_eval_valid


####################
# LR sweep


@dataclass
class LrSweep:
    """Learning rate sweep settings."""

    base: Settings
    step: float
    threshold: float


def find_learning_rate(
    settings: LrSweep, run_in_subprocess: bool = True
) -> Tuple[Settings, float]:
    """Perform a LR sweep, starting from `base`.

    Tries: initial, initial * step, initial * step^2, ...

    Until the validation loss is more than `best_loss + threshold`.

    Run in subprocess (by default) for sake of isolation & memory use.
    """
    best_loss, best_settings = None, None
    for n in it.count():
        test_settings = copy.deepcopy(settings.base)
        test_settings.training.optimiser.learning_rate *= settings.step**n
        # Run in subprocess for sake of isolation & memory use
        loss = (
            utility.run_in_subprocess(run, settings=test_settings)
            if run_in_subprocess
            else run(settings=test_settings)
        )["valid_loss"]
        print(
            f"LR {test_settings.training.optimiser.learning_rate} -> {loss}",
            file=sys.stderr,
        )
        if best_loss is None or loss < best_loss:
            best_loss = loss
            best_settings = test_settings
        if np.isnan(loss) or best_loss + settings.threshold < loss:
            return best_settings, best_loss
    assert False, "unreachable code (infinite loop)"
