# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

"""Data loading and batching."""

import itertools as it
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

Batch = Dict[str, np.ndarray]
Vocab = Tuple[str, ...]


def to_ids(data: str, vocab: Vocab, dtype: np.dtype) -> np.ndarray:
    """Use a (complete) vocabulary to map characters onto term IDs."""
    assert len(vocab) < np.iinfo(dtype).max
    ch_to_idx = {ch: idx for idx, ch in enumerate(vocab)}
    return np.array([ch_to_idx[ch] for ch in data], dtype=dtype)


def to_str(terms: np.ndarray, vocab: Vocab) -> str:
    """Map term IDs back to characters."""
    return "".join(vocab[idx] for idx in terms)


@dataclass
class BatchSettings:
    """Settings for a stream of batches."""

    sequences: int
    sequence_length: int
    overlap_length: int
    loop_seed: Optional[int]

    @property
    def shape(self) -> Tuple[int, int]:
        """The 2D shape of a batch."""
        return (self.sequences, self.sequence_length)

    @property
    def target_length(self) -> int:
        """The maximum number of target tokens per sequence."""
        return self.sequence_length - self.overlap_length

    @property
    def target_tokens(self) -> int:
        """The maximum number of target tokens."""
        return self.sequences * self.target_length


@dataclass
class Data:
    """A dataset that can generate batches."""

    vocab: Vocab
    parts: Dict[str, np.ndarray]

    def batches(self, part: str, settings: BatchSettings) -> Iterable[Batch]:
        """Batch with overlapping sequences.

        Note - if `loop_seed` is non-None, generates an infinite stream of batches, sampled
        with replacement.
        """

        data = self.parts[part]
        batch_tokens = []
        batch_mask = []
        idxs = np.arange(settings.sequence_length)
        if settings.loop_seed is None:
            starts: Iterable[int] = range(0, len(data), settings.target_length)
        else:
            random = np.random.Generator(np.random.PCG64(settings.loop_seed))
            starts = (
                random.integers(len(data) - settings.target_length) for _ in it.count()
            )

        for start in starts:
            begin = max(0, start - settings.overlap_length)
            sequence = data[begin : start + settings.target_length]
            # "token padding"
            npad = settings.sequence_length - len(sequence)
            sequence = np.pad(sequence, ((0, npad),))
            mask = ((start - begin) <= idxs) & (idxs < (len(sequence) - npad))
            batch_tokens.append(sequence.astype(np.int32))
            batch_mask.append(mask.astype(np.int32))
            if settings.sequences <= len(batch_tokens):
                yield dict(tokens=np.stack(batch_tokens), mask=np.stack(batch_mask))
                batch_tokens.clear()
                batch_mask.clear()

        # Incomplete final batch - needs "sequence padding"
        if batch_tokens:
            npad = settings.sequences - len(batch_tokens)
            batch_tokens.extend(npad * [batch_tokens[-1]])
            batch_mask.extend(npad * [np.zeros_like(batch_mask[-1])])
            yield dict(tokens=np.stack(batch_tokens), mask=np.stack(batch_mask))


def load_character(root: Path, **parts: str) -> Data:
    """Load a character-based dataset.

    e.g. given parts=dict(train="train.txt", valid="valid.txt")

    root/
        vocab.json
        train.txt
        valid.txt
    """
    vocab = tuple(json.loads((root / "vocab.json").read_text()))
    return Data(
        vocab=vocab,
        parts={
            name: to_ids((root / path).read_text("utf8"), vocab, np.uint16)
            for name, path in parts.items()
        },
    )
