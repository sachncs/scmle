"""Tests for dataset ingestion."""

import pandas as pd
import pytest

from mml.contracts import errors
from mml.data import readers


@pytest.fixture
def frame():
    return pd.DataFrame(
        {"a": [1, 2, 3], "b": ["x", "y", "z"], "y": [0, 1, 0]}
    )


def test_csv_and_parquet_roundtrip(tmp_path, frame):
    frame.to_csv(tmp_path / "d.csv", index=False)
    frame.to_parquet(tmp_path / "d.parquet")
    assert readers.read_table(tmp_path / "d.csv").shape == (3, 3)
    assert readers.read_table(tmp_path / "d.parquet").shape == (3, 3)


def test_missing_file(tmp_path):
    with pytest.raises(errors.DataError, match="not found"):
        readers.read_table(tmp_path / "nope.csv")


def test_unsupported_suffix(tmp_path):
    (tmp_path / "d.xlsx").write_text("x")
    with pytest.raises(errors.DataError, match="unsupported"):
        readers.read_table(tmp_path / "d.xlsx")


def test_empty_file(tmp_path):
    (tmp_path / "e.csv").write_text("a,b\n")
    with pytest.raises(errors.DataError, match="no rows"):
        readers.read_table(tmp_path / "e.csv")


def test_duplicate_columns(tmp_path):
    class DuplicatingReader(readers.TableReader):
        suffixes = (".csv",)

        def load(self, path):
            return pd.DataFrame([[1, 2]], columns=["a", "a"])

    (tmp_path / "d.csv").write_text("a\n1\n")
    loader = readers.DatasetLoader((DuplicatingReader(),))
    with pytest.raises(errors.DataError, match="duplicate"):
        loader.load(tmp_path / "d.csv")


def test_custom_reader_is_dispatched_by_suffix(tmp_path):
    class TsvReader(readers.TableReader):
        suffixes = (".tsv",)

        def load(self, path):
            return pd.read_csv(path, sep="\t")

    (tmp_path / "d.tsv").write_text("a\tb\n1\t2\n")
    loader = readers.DatasetLoader((TsvReader(),))
    assert loader.load(tmp_path / "d.tsv").shape == (1, 2)
    assert loader.supported_suffixes() == [".tsv"]


def test_manifest_deterministic_and_input_immutable(frame):
    before = frame.copy()
    first = readers.build_manifest(frame, "y")
    second = readers.build_manifest(frame.copy(), "y")
    assert first == second
    pd.testing.assert_frame_equal(frame, before)
    changed = frame.assign(a=[9, 2, 3])
    changed_manifest = readers.build_manifest(changed, "y")
    assert changed_manifest.data_hash != first.data_hash
    assert changed_manifest.schema_hash == first.schema_hash


def test_manifest_requires_target(frame):
    with pytest.raises(errors.DataError, match="target"):
        readers.build_manifest(frame, "nope")
