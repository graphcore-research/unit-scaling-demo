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
    for args in [
        dict(batch_sequences=3, sequence_length=6, overlap_length=2),
        dict(batch_sequences=1, sequence_length=8, overlap_length=0),
    ]:
        batch_shape = (args["batch_sequences"], args["sequence_length"])
        train_batches = list(
            it.islice(data.train_batches(**args, seed=sum(args.values())), 100)
        )
        for batch in train_batches:
            assert batch["tokens"].shape == batch_shape
            assert batch["tokens"].dtype == np.uint16
            assert batch["mask"].shape == batch_shape
            assert batch["mask"].dtype == np.bool
            assert (
                np.sum(batch["mask"])
                >= int(args["sequence_length"] - args["overlap_length"])
                * args["batch_sequences"]
            )

        for part, text in parts.items():
            eval_batches = list(data.eval_batches(part, **args))
            for batch in eval_batches:
                assert batch["tokens"].shape == batch_shape
                assert batch["tokens"].dtype == np.uint16
                assert batch["mask"].shape == batch_shape
                assert batch["mask"].dtype == np.bool
                assert np.sum(batch["mask"]) > 0
            flat_tokens = np.ravel([b["tokens"] for b in eval_batches])
            flat_mask = np.ravel([b["mask"] for b in eval_batches])
            assert datasets.to_str(flat_tokens[flat_mask], vocab) == text


def test_load():
    # Path("data/vocab.json").write_text(json.dumps(sorted({
    #     ch for path in Path("data").glob("*.txt") for ch in path.read_text("utf8")
    # })))
    folder = Path(__file__).parent / "data"
    data = datasets.load_char(folder, train="train.txt", valid="valid.txt")

    assert datasets.to_str(data.parts["valid"], data.vocab) == (
        folder / "valid.txt"
    ).read_text("utf8")
    assert len(data.parts["train"]) > len(data.parts["valid"])
