import collections
import contextlib
import json
import os
import unittest.mock as um
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

from .. import datasets, experiments, models, training
from ..pedal import xpu


def test_log_wandb_finish_on_error():
    with pytest.raises(ValueError), um.patch("wandb.init"), um.patch(
        "wandb.finish"
    ) as wandb_finish:
        with experiments.log_wandb():
            raise ValueError
    wandb_finish.assert_called_once_with(1)


def _test_settings(path: Path) -> experiments.Settings:
    return experiments.Settings(
        data=experiments.DataSettings(Path(__file__).parent / "data"),
        model=models.Settings(
            hidden_size=64,
            depth=1,
            residual=None,
            sequence=models.Conv(kernel_size=5, groups=1),
            token=None,
            unit_scale="0.2",
            dtype="float32",
            vocab_size=None,  # type:ignore[arg-type]
            seed=None,  # type:ignore[arg-type]
        ),
        training=training.Settings(
            batch=datasets.BatchSettings(
                sequences=10, sequence_length=32, overlap_length=8, loop_seed=None
            ),
            steps=100,
            valid_interval=50,
            optimiser=training.AdamW(learning_rate=0.05),
            loss_scale=1,
        ),
        target=xpu.CpuSettings(compile=False),
        output=experiments.OutputSettings(
            wandb=True,
            stderr=True,
            log=path / "log.jsonl",
            checkpoint=path / "model.npz",
        ),
        metadata=dict(experiment="testxp"),
        seed=None,  # type:ignore[arg-type]
    )


def test_run_experiment(tmp_path: Path):  # pylint:disable=too-many-locals

    with contextlib.ExitStack() as stack:
        wandb_init = stack.enter_context(um.patch("wandb.init"))
        wandb_log = stack.enter_context(um.patch("wandb.log"))
        stack.enter_context(um.patch("wandb.run"))
        wandb_finish = stack.enter_context(um.patch("wandb.finish"))
        stack.enter_context(
            um.patch.dict(
                os.environ, {"SSUB_UID": "ssub123", "SLURM_JOB_ID": "slurm123"}
            )
        )

        experiments.run(_test_settings(tmp_path))

    wandb_init.assert_called_once()
    wandb_init_args = wandb_init.call_args[1]
    assert wandb_init_args.get("project") == "scaled-matmuls"
    assert wandb_init_args["config"]["metadata"]["ssub_id"] == "ssub123"
    assert wandb_log.call_count == 2 * 3
    wandb_finish.assert_called_once()

    # Checkpoint
    checkpoint = np.load(tmp_path / "model.npz")
    assert checkpoint["step"] == 100
    assert len(checkpoint) >= 2

    # Log
    log_by_kind = collections.defaultdict(list)
    with (tmp_path / "log.jsonl").open() as file_:
        for line in file_:
            item = json.loads(line)
            log_by_kind[item["kind"]].append(item)

    (log_settings,) = log_by_kind["settings"]
    assert log_settings["metadata"]["experiment"] == "testxp"
    assert isinstance(log_settings["model"]["seed"], int)

    (log_stats,) = log_by_kind["stats"]
    assert 5 * 64 * 64 < log_stats["n_weights"]
    assert len(log_stats["weight_shapes"])

    assert len(log_by_kind["train_step"]) == 100
    first_step, *_, last_step = log_by_kind["train_step"]
    assert first_step["step"] == 0
    assert last_step["step"] == 99
    assert last_step["loss"] < first_step["loss"]

    assert [x["valid_n_tokens"] for x in log_by_kind["eval_valid"]] == 3 * [7665]
    assert log_by_kind["eval_valid"][-1]["valid_loss"] < 2.5

    assert len([x["train_n_tokens"] for x in log_by_kind["eval_train"]]) == 3
    assert log_by_kind["eval_train"][-1]["train_loss"] < 2.5


def test_find_learning_rate():
    called_with_lr = []

    def _fake_run(settings: experiments.Settings) -> Dict[str, Any]:
        learning_rate = settings.training.optimiser.learning_rate
        called_with_lr.append(learning_rate)
        return dict(valid_loss=(15 - learning_rate) ** 2)

    base = _test_settings(Path("fake"))
    base.training.optimiser.learning_rate = 3
    with um.patch("scmm.experiments.run", new_callable=lambda: _fake_run):
        # LR = 3, 6, 12, 24, 48, ...
        # Note - disable run_in_subprocess, otherwise our um.patch() wouldn't work
        best, best_loss = experiments.find_learning_rate(
            experiments.LrSweep(base, step=2, threshold=30), run_in_subprocess=False
        )

    np.testing.assert_allclose(best.training.optimiser.learning_rate, 12)
    np.testing.assert_allclose(best_loss, 9)
    np.testing.assert_allclose(called_with_lr, [3, 6, 12, 24])
