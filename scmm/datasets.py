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
class Data:
    """A dataset that can generate batches."""

    vocab: Vocab
    parts: Dict[str, np.ndarray]

    def train_batches(
        self, batch_sequences: int, sequence_length: int, overlap_length: int, seed: int
    ) -> Iterable[Batch]:
        """An infinite generator of shuffled batches for training."""
        return _batches(
            self.parts["train"],
            batch_sequences=batch_sequences,
            sequence_length=sequence_length,
            overlap_length=overlap_length,
            loop_with_seed=seed,
        )

    def eval_batches(
        self, part: str, batch_sequences: int, sequence_length: int, overlap_length: int
    ) -> Iterable[Batch]:
        """A finite generator of unshuffled batches for validation/test."""
        return _batches(
            self.parts[part],
            batch_sequences=batch_sequences,
            sequence_length=sequence_length,
            overlap_length=overlap_length,
            loop_with_seed=None,
        )


def load_char(root: Path, **parts: str) -> Data:
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


def _batches(
    data: np.ndarray,
    batch_sequences: int,
    sequence_length: int,
    overlap_length: int,
    loop_with_seed: Optional[int],
) -> Iterable[Batch]:
    """Batch with overlapping sequences."""
    batch_tokens = []
    batch_mask = []
    idxs = np.arange(sequence_length)
    if loop_with_seed is None:
        starts: Iterable[int] = range(0, len(data), sequence_length - overlap_length)
    else:
        random = np.random.Generator(np.random.PCG64(loop_with_seed))
        starts = (random.integers(sequence_length - overlap_length) for _ in it.count())

    for start in starts:
        begin = max(0, start - overlap_length)
        sequence = data[begin : start + sequence_length - overlap_length]
        # "token padding"
        npad = sequence_length - len(sequence)
        sequence = np.pad(sequence, ((0, npad),))
        mask = ((start - begin) <= idxs) & (idxs < (len(sequence) - npad))
        batch_tokens.append(sequence)
        batch_mask.append(mask)
        if batch_sequences <= len(batch_tokens):
            yield dict(tokens=np.stack(batch_tokens), mask=np.stack(batch_mask))
            batch_tokens.clear()
            batch_mask.clear()

    # Incomplete final batch - needs "sequence padding"
    if batch_tokens:
        npad = batch_sequences - len(batch_tokens)
        batch_tokens.extend(npad * [batch_tokens[-1]])
        batch_mask.extend(npad * [np.zeros_like(batch_mask[-1])])
        yield dict(tokens=np.stack(batch_tokens), mask=np.stack(batch_mask))
