"""General standalone utilities."""

import contextlib
import datetime
from pathlib import Path
from typing import (
    Any,
    Callable,
    ContextManager,
    Dict,
    Generator,
    Iterable,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np
import tensorflow as tf
from tensorflow import keras


def split_seed(seed: int, n: int) -> Tuple[int, ...]:
    """Split a random seed into n seeds.

    Note that the original seed should not be used after calling this.
    """
    return tuple(
        int(seq.generate_state(1)[0]) for seq in np.random.SeedSequence(seed).spawn(n)
    )


T = TypeVar("T")


def remove_keys(dict_: Dict[str, T], *keys: str) -> Dict[str, T]:
    """Return a new dictionary with specific keys removed."""
    return {k: v for k, v in dict_.items() if k not in keys}


def to_jsonable(obj: Any) -> Any:
    """A decent default=? function for json.dump."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.ndarray, np.number)):
        return obj.tolist()
    if isinstance(obj, datetime.date):  # datetime.datetime is a subclass
        return obj.isoformat()
    raise TypeError(f"Type '{type(obj).__name__}' is not JSON-serialisable")


Logger = Callable[..., None]


@contextlib.contextmanager
def logging(
    *loggers: Union[ContextManager[Logger], Logger]
) -> Generator[Logger, None, None]:
    """A context manager that delegates logging calls to multiple "loggers".

    Arguments are either:
     - Callable actions
     - Context managers that return callable actions

    For example:

        @contextlib.contextmanager
        def log_to_file(path: Path) -> None:
            with path.open("w") as f:
                yield lambda item: print(item, file=f)

        with logging(print, log_to_file(Path("log.txt"))) as log:
            log("item one")
            log("item two")
    """
    with contextlib.ExitStack() as stack:
        functions = [
            stack.enter_context(logger)  # type:ignore[arg-type]
            if hasattr(logger, "__enter__")
            else logger
            for logger in loggers
        ]

        def apply(*args: Any, **kwargs: Any) -> None:
            for fn in functions:
                fn(*args, **kwargs)  # type:ignore[operator]

        yield apply


def named_weights(
    layer: Union[keras.layers.Layer, Sequence[keras.layers.Layer]],
    prefix: Tuple[str, ...] = (),
) -> Iterable[Tuple[str, tf.Variable]]:
    """Walk a layer, recursively trying to find weight variables."""
    if isinstance(layer, (list, tuple)):
        for n, child in enumerate(layer):
            yield from named_weights(child, prefix + (str(n),))
    if isinstance(layer, keras.layers.Layer):
        for attr, child in vars(layer).items():
            if attr.startswith("_"):
                continue
            if isinstance(child, (list, tuple, keras.layers.Layer)):
                yield from named_weights(child, prefix + (attr,))
            if isinstance(child, tf.Variable):
                yield (".".join(prefix + (attr,)), child)
