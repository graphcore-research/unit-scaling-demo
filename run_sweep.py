# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

"""Run a multi-axis hyperparameter sweep."""

import copy
import dataclasses
import itertools as it
import multiprocessing
import multiprocessing.pool
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import scmm as S

# pylint:disable=redefined-outer-name


class Sweeper:
    """Utility for sweeping multiple settings axes."""

    def __init__(
        self,
        settings: Union[S.experiments.Settings, S.experiments.LrSweep],
        n_workers: int,
        reps: int,
    ):
        self.n_workers = n_workers
        self.reps = reps
        if isinstance(settings, S.experiments.LrSweep):
            self.base_settings = settings.base
            self.lr_settings: Optional[S.experiments.LrSweep] = settings
        else:
            self.base_settings = settings
            self.lr_settings = None
        self.axes: List[List[Dict[str, Any]]] = []

    def add(self, values: Iterable[Dict[str, Any]]) -> None:
        """Add an independent axis to the sweep."""
        self.axes.append(list(values))

    @staticmethod
    def _recursive_assign(
        settings: S.experiments.Settings, path: str, value: Any
    ) -> None:
        # Nested lookup
        *prefix, last = path.split(".")
        node = settings
        for key in prefix:
            node = getattr(node, key)

        # Dataclass type checking
        expected_type: Any = {f.name: f.type for f in dataclasses.fields(node)}.get(
            last
        )
        if expected_type is float:
            expected_type = (int, float)
        if getattr(expected_type, "__origin__", None) is Union:
            expected_type = expected_type.__args__
        if not isinstance(value, expected_type):
            raise ValueError(
                f"Expected {path} to be {expected_type}, actual {value} (type {type(value)})"
            )

        setattr(node, last, value)

    @property
    def configs(self) -> Iterable[Union[S.experiments.Settings, S.experiments.LrSweep]]:
        """Iterate through all settings configurations included in the sweep."""
        for overrides in it.product(*self.axes):
            settings = copy.deepcopy(self.base_settings)
            for override in overrides:
                for path, value in override.items():
                    self._recursive_assign(settings, path, value)
            if self.lr_settings is not None:
                yield dataclasses.replace(self.lr_settings, base=settings)
            else:
                yield settings

    def run(self) -> None:
        """Run a parallel sweep."""
        # os.environ["TMPDIR"] = "/localdata/tmp"
        os.environ["TF_POPLAR_FLAGS"] = (
            "--show_progress_bar=false"
            f" --executable_cache_path=/a/scratch/{os.environ['USER']}_research/tmp/cache/sweep"
        )
        with multiprocessing.pool.ThreadPool(self.n_workers) as pool:
            for _ in range(self.reps):
                for setting in self.configs:
                    target = (
                        S.experiments.run
                        if isinstance(setting, S.experiments.Settings)
                        else S.experiments.find_learning_rate
                    )
                    pool.apply_async(
                        S.pedal.utility.run_in_subprocess,
                        kwds=dict(command=target, settings=setting),
                    )
            pool.close()
            pool.join()


# Run sweep
#   ssub -n 16 -p ipu-large -- python run_sweep.py

if __name__ == "__main__":
    settings = S.experiments.Settings(
        data=S.experiments.DataSettings(
            Path("/home/research-datasets/wikitext103_raw")
        ),
        model=S.models.Settings(
            hidden_size=128,
            depth=8,
            residual=S.models.Residual(norm="pre", alpha="mean"),
            sequence=S.models.Attention(
                heads=2, head_size=64, frequencies=128, max_period=1024
            ),
            token=S.models.FFN(multiple=4),
            dtype="float32",
            vocab_size=None,  # type:ignore[arg-type]
            seed=None,  # type:ignore[arg-type]
        ),
        training=S.training.Settings(
            batch=S.datasets.BatchSettings(
                sequences=8,
                sequence_length=256,
                overlap_length=32,
                loop_seed=None,
            ),
            steps=int(2**19),
            valid_interval=int(2**14),
            optimiser=S.training.AdamW(
                learning_rate=2**-14,
                learning_rate_decay=2**-16,
            ),
            loss_scale=1,
        ),
        unit_scale=None,
        target=S.pedal.xpu.IpuSettings(
            iterations_per_loop=int(2**10),
            stochastic_rounding=True,
        ),
        output=S.experiments.OutputSettings(
            wandb=True, stderr=False, log=None, checkpoint=None
        ),
        seed=None,  # type:ignore[arg-type]
        metadata=dict(experiment="20230115_large_p0"),
    )

    sweeper = Sweeper(
        S.experiments.LrSweep(settings, step=2, threshold=0.1, reps=3),
        n_workers=16,
        reps=1,
    )

    def _all_settings() -> Iterable[Dict[str, Any]]:
        # pylint:disable=too-many-nested-blocks
        attention = S.models.Attention(
            heads=2, head_size=64, frequencies=128, max_period=1024
        )
        conv = S.models.Conv(kernel_size=7, groups=8)
        rnn = S.models.RNN(rebias=1)
        for sequence in [rnn, conv, attention]:
            for norm in ["pre", "post"]:
                for dtype in ["float16", "float32"]:
                    for unit_scale in [None, "0.4"]:
                        for loss_scale in [1, 2048]:
                            sequence_kind = sequence.kind  # type:ignore[attr-defined]
                            if (
                                unit_scale or dtype == "float32"
                            ) and loss_scale != 1:  # unnecessary
                                continue

                            if norm == "post" and sequence_kind != "attention":
                                continue  # only run post-norm for attention

                            yield {
                                "model.residual.norm": norm,
                                "model.depth": 2 if sequence_kind == "rnn" else 8,
                                "model.sequence": sequence,
                                "model.dtype": dtype,
                                "unit_scale": unit_scale,
                                "training.loss_scale": loss_scale,
                                "training.optimiser.learning_rate": (
                                    2**-14 if unit_scale is None else 2**-8
                                ),
                            }

    sweeper.add(_all_settings())

    # This also runs basic checks on `configs` (e.g. in --dry-run)
    print(
        f"Sweeping {sum(1 for _ in sweeper.configs)} settings, {sweeper.reps} reps,"
        f" as {sweeper.base_settings.metadata['experiment']!r}",
        file=sys.stderr,
    )
    if not set(sys.argv) & {"-d", "--dry-run", "--dryrun"}:
        # subprocess.check_call(["ulimit", "-u", "16384"], shell=True)
        sweeper.run()
