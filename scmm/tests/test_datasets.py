# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import itertools as it
from pathlib import Path

import numpy as np

from .. import datasets


def test_to_ids_to_str():
    vocab = tuple(" abcd")
    original = "bad cad"
    ids = datasets.to_ids(original, vocab, dtype=np.uint16)
    np.testing.assert_equal(ids, [2, 1, 4, 0, 3, 1, 4])
    assert datasets.to_str(ids, vocab) == original


def test_data():
    vocab = tuple(" abcd")
    parts = dict(
        train="a bad cad abba a bad dad",
        valid="a bb ccc dddd",
    )
    data = datasets.Data(
        vocab=vocab,
        parts={k: datasets.to_ids(v, vocab, np.uint16) for k, v in parts.items()},
    )
    S = datasets.BatchSettings
    for settings in [
        S(sequences=3, sequence_length=6, overlap_length=2, loop_seed=None),
        S(sequences=3, sequence_length=6, overlap_length=2, loop_seed=100),
        S(sequences=1, sequence_length=8, overlap_length=0, loop_seed=None),
        S(sequences=1, sequence_length=8, overlap_length=0, loop_seed=200),
    ]:
        for part, text in parts.items():
            max_batch = 25
            batches = list(it.islice(data.batches(part, settings), max_batch))
            for batch in batches:
                assert batch["tokens"].shape == settings.shape
                assert batch["tokens"].dtype == np.int32
                assert batch["mask"].shape == settings.shape
                assert batch["mask"].dtype == np.int32
                assert np.sum(batch["mask"]) >= (
                    1 if settings.loop_seed is None else settings.target_tokens
                )

            if settings.loop_seed is None:
                flat_tokens = np.ravel([b["tokens"] for b in batches])
                flat_mask = np.ravel([b["mask"] for b in batches]).astype(np.bool)
                assert datasets.to_str(flat_tokens[flat_mask], vocab) == text
            else:
                assert len(batches) == max_batch


def test_load():
    # Path("data/vocab.json").write_text(json.dumps(sorted({
    #     ch for path in Path("data").glob("*.txt") for ch in path.read_text("utf8")
    # })))
    folder = Path(__file__).parent / "data"
    data = datasets.load_character(folder, train="train.txt", valid="valid.txt")

    assert datasets.to_str(data.parts["valid"], data.vocab) == (
        folder / "valid.txt"
    ).read_text("utf8")
    assert len(data.parts["train"]) > len(data.parts["valid"])
