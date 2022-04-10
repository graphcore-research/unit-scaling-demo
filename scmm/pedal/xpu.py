"""General utilities to unify CPU/IPU programming."""

import functools as ft
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Callable, Dict, Iterable, Iterator, Optional, Type, Union

import numpy as np
import tensorflow as tf

try:
    from tensorflow.python import ipu

    IPU = True
except ImportError:  # pragma: no cover
    IPU = False


Function = Callable[..., Any]
Operation = Callable[..., Dict[str, tf.Tensor]]
Batch = Dict[str, np.ndarray]
FunctionCache = Callable[[Any], Callable[[Function], Function]]


def _make_cache(**function_args: Any) -> FunctionCache:
    """Make a decorator that calls tf.function, with a user-keyed cache.

    E.g.

        cache = make_cache(experimental_compile=True)

        body = ...

        @cache(key=("model", body))
        def model(x: tf.Tensor) -> tf.Tensor:
            return 2 * body(x)
    """
    _cache: Dict[Any, Function] = {}

    def wrap(key: Any) -> Callable[[Function], Function]:
        def wrapper(fn: Operation) -> Operation:
            if key not in _cache:
                _cache[key] = tf.function(**function_args)(fn)
            return _cache[key]

        return wrapper

    return wrap


@dataclass
class CpuSettings:
    """CPU-specific settings."""

    compile: bool

    type: str = "cpu"


@dataclass
class IpuSettings:
    """IPU-specific settings."""

    iterations_per_loop: int

    type: str = "ipu"


Settings = Union[CpuSettings, IpuSettings]


class Context:
    """Manages target setup and a cache for compiled functions."""

    def __init__(
        self,
        strategy: tf.distribute.Strategy,
        runner: Callable[..., Iterable[Batch]],
        compile: bool,  # pylint:disable=redefined-builtin
    ):
        self.strategy = strategy
        self._runner = runner
        self._scope = self.strategy.scope()
        self._cache = (
            _make_cache(experimental_compile=True)
            if compile
            else (lambda key: lambda fn: fn)
        )

    def __enter__(self) -> "Context":
        self._scope.__enter__()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self._scope.__exit__(exc_type, exc_val, exc_tb)

    def loop(self, operation: Operation, inputs: Iterable[Batch]) -> Iterable[Batch]:
        """Stream inputs into an operation and return all outputs.

        operation -- callable as `result = operation(**input)`,
                        where `result` is a `dict`
        """
        return self._runner(
            operation, inputs, strategy=self.strategy, cache=self._cache
        )


def context(settings: Settings) -> Context:
    """Create an execution context with the given settings."""
    if isinstance(settings, CpuSettings):
        return Context(
            tf.distribute.OneDeviceStrategy(""),
            loop_cpu,
            compile=settings.compile,
        )
    if isinstance(settings, IpuSettings):
        if not IPU:  # pragma: no cover
            raise ValueError(
                "Cannot create IPU context - tensorflow.python.ipu could not be imported"
            )
        return _create_ipu_context(settings)
    assert False, f"Unexpected Context settings type {settings}"


def loop_cpu(
    operation: Operation,
    inputs: Iterable[Batch],
    strategy: tf.distribute.Strategy,
    cache: FunctionCache,
) -> Iterable[Batch]:
    """Stream inputs into an operation and return all outputs.

    operation -- callable as `result = operation(**input)`,
                    where `result` is a `dict`
    """
    fn = cache(key=operation)(operation)  # type:ignore[call-arg]
    for input_ in inputs:
        yield {k: np.array(v) for k, v in strategy.run(fn, kwargs=input_).items()}


if IPU:

    def _create_ipu_context(settings: IpuSettings) -> Context:
        config = ipu.config.IPUConfig()
        config.auto_select_ipus = 1
        ipu.utils.configure_ipu_system(config)
        return Context(
            ipu.ipu_strategy.IPUStrategy(),
            ft.partial(loop_ipu, iterations_per_loop=settings.iterations_per_loop),
            compile=True,
        )

    def _padded_dataset(inputs: Iterable[Batch]) -> tf.data.Dataset:
        iterator = iter(inputs)
        head = next(iterator)

        def generator() -> Iterable[Dict[str, np.ndarray]]:  # pragma: no cover
            yield dict(**head, _pad=np.array(False))
            for item in iterator:
                yield dict(**item, _pad=np.array(False))
            while True:  # padding
                yield dict(**head, _pad=np.array(True))

        signature = {
            k: tf.TensorSpec(shape=v.shape, dtype=v.dtype) for k, v in head.items()
        }
        signature["_pad"] = tf.TensorSpec(shape=(), dtype=np.bool)
        return tf.data.Dataset.from_generator(generator, output_signature=signature)

    def loop_ipu(
        operation: Operation,
        inputs: Iterable[Batch],
        strategy: tf.distribute.Strategy,
        cache: FunctionCache,
        iterations_per_loop: int,
    ) -> Iterable[Dict[str, np.ndarray]]:
        """Stream inputs into an operation and return all outputs.

        operation -- callable as `result = operation(**input)`,
                     where `result` is a `dict`
        """

        @cache(  # type:ignore[call-arg]
            key=("loop_ipu", operation, iterations_per_loop)
        )
        def _loop(
            iterator: Iterator[Dict[str, tf.Tensor]],
            outfeed: ipu.ipu_outfeed_queue.IPUOutfeedQueue,
        ) -> None:  # pragma: no cover
            for _ in tf.range(iterations_per_loop):
                batch = next(iterator)
                pad = batch.pop("_pad")
                results = operation(**batch)
                results["_pad"] = pad
                outfeed.enqueue(results)

        iterator = iter(_padded_dataset(inputs))
        outfeed = ipu.ipu_outfeed_queue.IPUOutfeedQueue()
        while True:
            strategy.run(_loop, (iterator, outfeed))
            for item in outfeed:
                if item.pop("_pad"):
                    # Prevent: Error occurred when finalizing GeneratorDataset iterator
                    del iterator
                    del outfeed
                    return
                yield {k: np.array(v) for k, v in item.items()}
