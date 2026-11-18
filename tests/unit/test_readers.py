"""Tests for dataset ingestion."""

import pandas
import pytest

from scmle.contracts import errors
from scmle.data.readers import tabular


@pytest.fixture
def frame():
    return pandas.DataFrame(
        {"a": [1, 2, 3], "b": ["x", "y", "z"], "y": [0, 1, 0]}
    )


def test_csv_and_parquet_roundtrip(tmp_path, frame):
    frame.to_csv(tmp_path / "d.csv", index=False)
    frame.to_parquet(tmp_path / "d.parquet")
    assert tabular.read_table(tmp_path / "d.csv").shape == (3, 3)
    assert tabular.read_table(tmp_path / "d.parquet").shape == (3, 3)


@pytest.mark.parametrize("suffix", [".tsv", ".tab"])
def test_delimited_text_roundtrip(tmp_path, frame, suffix):
    frame.to_csv(tmp_path / f"d{suffix}", sep="\t", index=False)
    pandas.testing.assert_frame_equal(
        tabular.read_table(tmp_path / f"d{suffix}"), frame
    )


def test_json_and_json_lines_roundtrip(tmp_path, frame):
    frame.to_json(tmp_path / "records.json", orient="records")
    frame.to_json(tmp_path / "records.jsonl", orient="records", lines=True)
    pandas.testing.assert_frame_equal(
        tabular.read_table(tmp_path / "records.json"), frame
    )
    pandas.testing.assert_frame_equal(
        tabular.read_table(tmp_path / "records.jsonl"), frame
    )


def test_compressed_text_and_json_lines_roundtrip(tmp_path, frame):
    csv_path = tmp_path / "records.csv.gz"
    jsonl_path = tmp_path / "records.jsonl.gz"
    frame.to_csv(csv_path, index=False, compression="gzip")
    frame.to_json(jsonl_path, orient="records", lines=True, compression="gzip")
    pandas.testing.assert_frame_equal(tabular.read_table(csv_path), frame)
    pandas.testing.assert_frame_equal(tabular.read_table(jsonl_path), frame)


def test_missing_file(tmp_path):
    with pytest.raises(errors.DataError, match="not found"):
        tabular.read_table(tmp_path / "nope.csv")


def test_unsupported_suffix(tmp_path):
    (tmp_path / "d.xml").write_text("x")
    with pytest.raises(errors.DataError, match="unsupported"):
        tabular.read_table(tmp_path / "d.xml")


def test_excel_is_registered_as_an_optional_format():
    assert ".xlsx" in tabular.Loader().extensions()


def test_excel_roundtrip_when_optional_dependency_is_installed(tmp_path, frame):
    openpyxl = pytest.importorskip("openpyxl")
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.append(list(frame.columns))
    for row in frame.itertuples(index=False, name=None):
        sheet.append(list(row))
    path = tmp_path / "records.xlsx"
    workbook.save(path)
    pandas.testing.assert_frame_equal(tabular.read_table(path), frame)


def test_empty_file(tmp_path):
    (tmp_path / "e.csv").write_text("a,b\n")
    with pytest.raises(errors.DataError, match="no rows"):
        tabular.read_table(tmp_path / "e.csv")


def test_duplicate_columns(tmp_path):
    class DuplicatingReader(tabular.TableReader):
        suffixes = (".csv",)

        def load(self, path):
            return pandas.DataFrame([[1, 2]], columns=["a", "a"])

    (tmp_path / "d.csv").write_text("a\n1\n")
    loader = tabular.Loader((DuplicatingReader(),))
    with pytest.raises(errors.DataError, match="duplicate"):
        loader.load(tmp_path / "d.csv")


def test_custom_reader_is_dispatched_by_suffix(tmp_path):
    class Tsv(tabular.TableReader):
        suffixes = (".tsv",)

        def load(self, path):
            return pandas.read_csv(path, sep="\t")

    (tmp_path / "d.tsv").write_text("a\tb\n1\t2\n")
    loader = tabular.Loader((Tsv(),))
    assert loader.load(tmp_path / "d.tsv").shape == (1, 2)
    assert loader.extensions() == [".tsv"]


def test_manifest_deterministic_and_input_immutable(frame):
    before = frame.copy()
    first = tabular.manifested(frame, "y")
    second = tabular.manifested(frame.copy(), "y")
    assert first == second
    pandas.testing.assert_frame_equal(frame, before)
    changed = frame.assign(a=[9, 2, 3])
    changed_manifest = tabular.manifested(changed, "y")
    assert changed_manifest.digests["data"] != first.digests["data"]
    assert changed_manifest.digests["schema"] == first.digests["schema"]


def test_manifest_populates_deterministic_snapshot_and_lineage(frame, tmp_path):
    path = tmp_path / "records.csv"
    frame.to_csv(path, index=False)
    first = tabular.manifested(frame, "y", path)
    second = tabular.manifested(frame.copy(), "y", path)

    assert first.snapshot == second.snapshot
    assert first.snapshot.startswith("snap_")
    assert len(first.snapshot) == len("snap_") + 64
    assert first.source.acquisition == "local-file"
    assert first.source.owner == ""
    assert first.source.license == ""
    assert first.lineage.snapshot == first.snapshot
    assert first.lineage.digests["tabular-frame"] == first.digests["data"]
    assert first.lineage.digests["raw-source"] == (
        tabular.raw_source_hash(path)
    )


def test_raw_source_hash_changes_snapshot_when_bytes_change(frame, tmp_path):
    path = tmp_path / "records.csv"
    frame.to_csv(path, index=False)
    first = tabular.manifested(frame, "y", path)

    path.write_bytes(path.read_bytes() + b"\n")
    second = tabular.manifested(frame, "y", path)

    assert first.digests["data"] == second.digests["data"]
    assert (
        first.lineage.digests["raw-source"]
        != (second.lineage.digests["raw-source"])
    )
    assert first.snapshot != second.snapshot


def test_snapshot_identity_changes_for_content_schema_and_uri(frame):
    baseline = tabular.manifested(frame, "y", "s3://bucket/records")
    changed_content = tabular.manifested(
        frame.assign(a=[9, 2, 3]), "y", "s3://bucket/records"
    )
    changed_schema = tabular.manifested(
        frame.astype({"a": "float64"}), "y", "s3://bucket/records"
    )
    changed_uri = tabular.manifested(frame, "y", "s3://bucket/other-records")

    assert changed_content.snapshot != baseline.snapshot
    assert changed_schema.snapshot != baseline.snapshot
    assert changed_uri.snapshot != baseline.snapshot


def test_uri_semantics_are_explicit_without_ownership_or_license_claims(frame):
    manifest = tabular.manifested(frame, "y", "s3://bucket/records")

    assert manifest.source.acquisition == "s3-uri"
    assert manifest.source.owner == ""
    assert manifest.source.license == ""
    assert tabular.semantic("<in-memory>") == "memory"
    assert tabular.canonicalize("<memory>") == "memory://"


def test_manifest_requires_target(frame):
    with pytest.raises(errors.DataError, match="target"):
        tabular.manifested(frame, "nope")


def test_replace_infinities_reports_counts_and_leaves_input_untouched():
    frame = pandas.DataFrame(
        {
            "a": [1.0, float("inf"), 3.0],
            "b": [float("-inf"), 2.0, float("inf")],
            "s": ["x", "y", "z"],
        }
    )
    before = frame.copy()
    cleaned, counts = tabular.sanitize(frame)
    assert counts == {"a": 1, "b": 2}
    assert cleaned["a"].isna().sum() == 1 and cleaned["b"].isna().sum() == 2
    pandas.testing.assert_frame_equal(frame, before)


def test_replace_infinities_returns_same_frame_when_clean():
    frame = pandas.DataFrame({"a": [1.0, 2.0], "s": ["x", "y"]})
    cleaned, counts = tabular.sanitize(frame)
    assert counts == {} and cleaned is frame
