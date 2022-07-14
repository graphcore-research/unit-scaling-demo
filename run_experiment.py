"""Run a single experiment."""

import dataclasses
import json
import os
from pathlib import Path

import scmm as S

# pylint:disable=invalid-name

# ssub -t mk2 -n 1 -- python run_experiment.py
if __name__ == "__main__":

    out, profile, sweep = None, None, False
    # profile = Path("out/profiles/dev")
    # out = Path("out/training/dev")
    # sweep = True

    settings = S.experiments.Settings(
        # data=S.experiments.DataSettings(Path("scmm/tests/data"), kind="test"),
        data=S.experiments.DataSettings(
            Path("/home/research-datasets/wikitext103_raw")
        ),
        model=S.models.Settings(
            hidden_size=128,
            depth=8,
            residual=S.models.Residual(norm="pre", alpha="mean"),
            sequence=S.models.Conv(kernel_size=7, groups=8),
            token=S.models.FFN(multiple=4),
            dtype="float32",
            vocab_size=None,  # type:ignore[arg-type]
            seed=None,  # type:ignore[arg-type]
        ),
        training=S.training.Settings(
            batch=S.datasets.BatchSettings(
                sequences=8, sequence_length=256, overlap_length=32, loop_seed=None
            ),
            steps=int(2**20),
            valid_interval=int(2**14),
            optimiser=S.training.AdamW(
                learning_rate=2**-6, learning_rate_decay=2**-16
            ),
            loss_scale=1,
        ),
        unit_scale="0.3",
        target=S.pedal.xpu.IpuSettings(iterations_per_loop=int(2**10)),
        output=S.experiments.OutputSettings(
            stderr=False,
            wandb=True,
            log=out and out / "log.jsonl",
            checkpoint=out and out / "model.npz",
        ),
        metadata=dict(experiment="dev"),
        seed=None,  # type:ignore[arg-type]
    )

    ####################
    # Common

    if profile:
        profile.mkdir(parents=True, exist_ok=True)
        os.environ["POPLAR_ENGINE_OPTIONS"] = json.dumps(
            {
                "autoReport.all": True,
                "autoReport.outputArchive": False,
                "autoReport.directory": str(profile),
                "debug.allowOutOfMemory": True,
                "profiler.replicaToProfile": 0,
            }
        )
        os.environ["PVTI_OPTIONS"] = json.dumps(
            dict(enable=True, directory=str(profile))
        )
        settings = dataclasses.replace(
            settings,
            # Switch out the data to avoid a large delay "loading"
            data=S.experiments.DataSettings(Path("scmm/tests/data")),
            model=dataclasses.replace(settings.model, vocab_size=5008),
            training=dataclasses.replace(
                settings.training, steps=2, valid_interval=None
            ),
            target=dataclasses.replace(settings.target, iterations_per_loop=int(2)),
            output=S.experiments.OutputSettings(
                stderr=True, wandb=False, log=profile / "log.jsonl", checkpoint=None
            ),
        )
    else:
        os.environ["TF_POPLAR_FLAGS"] = (
            "--show_progress_bar=false"
            f" --executable_cache_path=/a/scratch/{os.environ['USER']}_research/tmp/cache"
        )

    if sweep:
        sweep_settings = S.experiments.LrSweep(settings, step=4, threshold=0.05, reps=1)
        S.experiments.find_learning_rate(settings=sweep_settings)
    else:
        # Run in subprocess so that the PVTI options "take"
        S.pedal.utility.run_in_subprocess(S.experiments.run, settings=settings)
