import os
from pathlib import Path
import json
import dataclasses

import scmm as S

out, profile = None, None
# profile = Path("out/profiles/dev")
# out = Path("out/dev")

# ssub -t mk2 -n 1 -- python run_experiment.py
settings = S.experiments.Settings(
    # data=S.experiments.DataSettings(Path("scmm/tests/data"), kind="test"),
    data=S.experiments.DataSettings(Path("/home/research-datasets/wikitext103_raw")),
    model=S.models.ResidualConv(
        vocab_size=None,
        seed=None,
        unit_scale=True,
        hidden_size=128,
        depth=8,
        kernel_size=7,
        ffn_multiple=4,
        group_size=16,
        residual_alpha="mean",
    ),
    training=S.training.Settings(
        batch=S.datasets.BatchSettings(
            sequences=8, sequence_length=256, overlap_length=32, loop_seed=None
        ),
        steps=int(1e6),
        valid_interval=int(1e4),
        learning_rate=2**-6,
        weight_decay=0,
    ),
    # target=S.pedal.xpu.CpuSettings(),
    target=S.pedal.xpu.IpuSettings(iterations_per_loop=int(1e3)),
    output=S.experiments.OutputSettings(
        wandb=True, log=out and out / "log.jsonl", checkpoint=out and out / "model.npz"
    ),
    metadata=dict(experiment="dev"),
    seed=None,
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
        training=dataclasses.replace(settings.training, steps=1, valid_interval=None),
        target=dataclasses.replace(settings.target, iterations_per_loop=int(1)),
        output=S.experiments.OutputSettings(
            wandb=False, log=profile / "log.jsonl", checkpoint=None
        ),
    )
else:
    os.environ[
        "TF_POPLAR_FLAGS"
    ] = f"--executable_cache_path=/a/scratch/{os.environ['USER']}_research/tmp/cache"

S.experiments.run(settings)
