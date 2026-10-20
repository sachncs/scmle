"""Tests for dataset ingestion."""

import pandas as pd
import pytest

from scml.contracts import errors
from scml.data.readers import tabular


@pytest.fixture
def frame():
    return pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"], "y": [0, 1, 0]})


def test_csv_and_parquet_roundtrip(tmp_path, frame):
    frame.to_csv(tmp_path / "d.csv", index=False)
    frame.to_parquet(tmp_path / "d.parquet")
    assert tabular.read_table(tmp_path / "d.csv").shape == (3, 3)
    assert tabular.read_table(tmp_path / "d.parquet").shape == (3, 3)


@pytest.mark.parametrize("suffix", [".tsv", ".tab"])
def test_delimited_text_roundtrip(tmp_path, frame, suffix):
    frame.to_csv(tmp_path / f"d{suffix}", sep="\t", index=False)
    pd.testing.assert_frame_equal(
        tabular.read_table(tmp_path / f"d{suffix}"), frame
    )


def test_json_and_json_lines_roundtrip(tmp_path, frame):
    frame.to_json(tmp_path / "records.json", orient="records")
    frame.to_json(tmp_path / "records.jsonl", orient="records", lines=True)
    pd.testing.assert_frame_equal(
        tabular.read_table(tmp_path / "records.json"), frame
    )
    pd.testing.assert_frame_equal(
        tabular.read_table(tmp_path / "records.jsonl"), frame
    )


def test_compressed_text_and_json_lines_roundtrip(tmp_path, frame):
    csv_path = tmp_path / "records.csv.gz"
    jsonl_path = tmp_path / "records.jsonl.gz"
    frame.to_csv(csv_path, index=False, compression="gzip")
    frame.to_json(jsonl_path, orient="records", lines=True, compression="gzip")
    pd.testing.assert_frame_equal(tabular.read_table(csv_path), frame)
    pd.testing.assert_frame_equal(tabular.read_table(jsonl_path), frame)


def test_missing_file(tmp_path):
    with pytest.raises(errors.DataError, match="not found"):
        tabular.read_table(tmp_path / "nope.csv")


def test_unsupported_suffix(tmp_path):
    (tmp_path / "d.xml").write_text("x")
    with pytest.raises(errors.DataError, match="unsupported"):
        tabular.read_table(tmp_path / "d.xml")


def test_excel_is_registered_as_an_optional_format():
    assert ".xlsx" in tabular.DatasetLoader().supported_suffixes()


def test_excel_roundtrip_when_optional_dependency_is_installed(tmp_path, frame):
    openpyxl = pytest.importorskip("openpyxl")
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.append(list(frame.columns))
    for row in frame.itertuples(index=False, name=None):
        sheet.append(list(row))
    path = tmp_path / "records.xlsx"
    workbook.save(path)
    pd.testing.assert_frame_equal(tabular.read_table(path), frame)


def test_empty_file(tmp_path):
    (tmp_path / "e.csv").write_text("a,b\n")
    with pytest.raises(errors.DataError, match="no rows"):
        tabular.read_table(tmp_path / "e.csv")


def test_duplicate_columns(tmp_path):
    class DuplicatingReader(tabular.TableReader):
        suffixes = (".csv",)

        def load(self, path):
            return pd.DataFrame([[1, 2]], columns=["a", "a"])

    (tmp_path / "d.csv").write_text("a\n1\n")
    loader = tabular.DatasetLoader((DuplicatingReader(),))
    with pytest.raises(errors.DataError, match="duplicate"):
        loader.load(tmp_path / "d.csv")


def test_custom_reader_is_dispatched_by_suffix(tmp_path):
    class TsvReader(tabular.TableReader):
        suffixes = (".tsv",)

        def load(self, path):
            return pd.read_csv(path, sep="\t")

    (tmp_path / "d.tsv").write_text("a\tb\n1\t2\n")
    loader = tabular.DatasetLoader((TsvReader(),))
    assert loader.load(tmp_path / "d.tsv").shape == (1, 2)
    assert loader.supported_suffixes() == [".tsv"]


def test_manifest_deterministic_and_input_immutable(frame):
    before = frame.copy()
    first = tabular.build_manifest(frame, "y")
    second = tabular.build_manifest(frame.copy(), "y")
    assert first == second
    pd.testing.assert_frame_equal(frame, before)
    changed = frame.assign(a=[9, 2, 3])
    changed_manifest = tabular.build_manifest(changed, "y")
    assert changed_manifest.data_hash != first.data_hash
    assert changed_manifest.schema_hash == first.schema_hash


def test_manifest_populates_deterministic_snapshot_and_lineage(frame, tmp_path):
    path = tmp_path / "records.csv"
    frame.to_csv(path, index=False)
    first = tabular.build_manifest(frame, "y", path)
    second = tabular.build_manifest(frame.copy(), "y", path)

    assert first.snapshot_id == second.snapshot_id
    assert first.snapshot_id.startswith("snap_")
    assert len(first.snapshot_id) == len("snap_") + 64
    assert first.source.acquisition_method == "local-file"
    assert first.source.owner == ""
    assert first.source.license == ""
    assert first.lineage.source_snapshot == first.snapshot_id
    assert first.lineage.object_hashes["tabular-frame"] == first.data_hash
    assert first.lineage.object_hashes["raw-source"] == (
        tabular.raw_source_hash(path)
    )


def test_raw_source_hash_changes_snapshot_when_bytes_change(frame, tmp_path):
    path = tmp_path / "records.csv"
    frame.to_csv(path, index=False)
    first = tabular.build_manifest(frame, "y", path)

    path.write_bytes(path.read_bytes() + b"\n")
    second = tabular.build_manifest(frame, "y", path)

    assert first.data_hash == second.data_hash
    assert (
        first.lineage.object_hashes["raw-source"]
        != (second.lineage.object_hashes["raw-source"])
    )
    assert first.snapshot_id != second.snapshot_id


def test_snapshot_identity_changes_for_content_schema_and_uri(frame):
    baseline = tabular.build_manifest(frame, "y", "s3://bucket/records")
    changed_content = tabular.build_manifest(
        frame.assign(a=[9, 2, 3]), "y", "s3://bucket/records"
    )
    changed_schema = tabular.build_manifest(
        frame.astype({"a": "float64"}), "y", "s3://bucket/records"
    )
    changed_uri = tabular.build_manifest(
        frame, "y", "s3://bucket/other-records"
    )

    assert changed_content.snapshot_id != baseline.snapshot_id
    assert changed_schema.snapshot_id != baseline.snapshot_id
    assert changed_uri.snapshot_id != baseline.snapshot_id


def test_uri_semantics_are_explicit_without_ownership_or_license_claims(frame):
    manifest = tabular.build_manifest(frame, "y", "s3://bucket/records")

    assert manifest.source.acquisition_method == "s3-uri"
    assert manifest.source.owner == ""
    assert manifest.source.license == ""
    assert tabular.uri_semantics("<in-memory>") == "memory"
    assert tabular.canonical_uri("<memory>") == "memory://"


def test_manifest_requires_target(frame):
    with pytest.raises(errors.DataError, match="target"):
        tabular.build_manifest(frame, "nope")


def test_replace_infinities_reports_counts_and_leaves_input_untouched():
    frame = pd.DataFrame(
        {
            "a": [1.0, float("inf"), 3.0],
            "b": [float("-inf"), 2.0, float("inf")],
            "s": ["x", "y", "z"],
        }
    )
    before = frame.copy()
    cleaned, counts = tabular.replace_infinities(frame)
    assert counts == {"a": 1, "b": 2}
    assert cleaned["a"].isna().sum() == 1 and cleaned["b"].isna().sum() == 2
    pd.testing.assert_frame_equal(frame, before)


def test_replace_infinities_returns_same_frame_when_clean():
    frame = pd.DataFrame({"a": [1.0, 2.0], "s": ["x", "y"]})
    cleaned, counts = tabular.replace_infinities(frame)
    assert counts == {} and cleaned is frame
