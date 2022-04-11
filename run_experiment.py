import os
from pathlib import Path
import json
import dataclasses

import scmm as S

# profile = Path("out/profile")
profile = None


settings = S.experiments.Settings(
    data=S.experiments.DataSettings(Path("data/wikitext103_raw")),
    model=S.models.Settings(
        hidden_size=128, depth=1, kernel_size=5, vocab_size=None, seed=None
    ),
    training=S.training.Settings(
        batch=S.datasets.BatchSettings(
            sequences=4, sequence_length=128, overlap_length=16, loop_seed=None
        ),
        steps=int(2e3),
        valid_interval=int(1e3),
        learning_rate=1e-2,
    ),
    # target=S.pedal.xpu.CpuSettings(compile=False),
    target=S.pedal.xpu.IpuSettings(iterations_per_loop=100),
    output=S.experiments.OutputSettings(
        wandb=False, log=Path("out/dev.jsonl"), checkpoint=None
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
        output=S.experiments.OutputSettings(wandb=False, log=None, checkpoint=None),
    )

S.experiments.run(settings)
