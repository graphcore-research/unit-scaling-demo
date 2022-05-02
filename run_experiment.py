"""Run a single experiment."""

import dataclasses
import json
import os
from pathlib import Path

import scmm as S

out, profile = None, None
# profile = Path("out/profiles/dev")
# out = Path("out/dev")

# ssub -t mk2 -n 1 -- python run_experiment.py
settings = S.experiments.Settings(
    # data=S.experiments.DataSettings(Path("scmm/tests/data"), kind="test"),
    data=S.experiments.DataSettings(Path("/home/research-datasets/wikitext103_raw")),
    model=S.models.Settings(
        hidden_size=128,
        depth=8,
        residual=S.models.Residual(norm="pre", alpha="mean"),
        sequence=S.models.Conv(kernel_size=7, groups=8),
        token=S.models.FFN(multiple=4),
        unit_scale="0.2",
        dtype="float32",
        vocab_size=None,  # type:ignore[arg-type]
        seed=None,  # type:ignore[arg-type]
    ),
    training=S.training.Settings(
        batch=S.datasets.BatchSettings(
            sequences=8, sequence_length=256, overlap_length=32, loop_seed=None
        ),
        steps=int(1e5),
        valid_interval=int(1e4),
        optimiser=S.training.AdamW(learning_rate=2**-6),
        loss_scale=1,
    ),
    # target=S.pedal.xpu.CpuSettings(),
    target=S.pedal.xpu.IpuSettings(iterations_per_loop=int(1e3)),
    output=S.experiments.OutputSettings(
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
    settings = dataclasses.replace(
        settings,
        # Switch out the data to avoid a large delay "loading"
        data=S.experiments.DataSettings(Path("scmm/tests/data")),
        model=dataclasses.replace(settings.model, vocab_size=5008),
        training=dataclasses.replace(settings.training, steps=2, valid_interval=None),
        target=dataclasses.replace(settings.target, iterations_per_loop=int(2)),
        output=S.experiments.OutputSettings(
            wandb=False, log=profile / "log.jsonl", checkpoint=None
        ),
    )
else:
    os.environ[
        "TF_POPLAR_FLAGS"
    ] = f"--executable_cache_path=/a/scratch/{os.environ['USER']}_research/tmp/cache"

S.experiments.run(settings)
