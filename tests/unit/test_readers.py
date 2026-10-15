import pandas as pd
import pytest

from magicml.contracts.errors import DataError
from magicml.data.readers import build_manifest, read_table


@pytest.fixture
def df():
    return pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"], "y": [0, 1, 0]})


def test_csv_and_parquet_roundtrip(tmp_path, df):
    df.to_csv(tmp_path / "d.csv", index=False)
    df.to_parquet(tmp_path / "d.parquet")
    assert read_table(tmp_path / "d.csv").shape == (3, 3)
    assert read_table(tmp_path / "d.parquet").shape == (3, 3)


def test_errors(tmp_path, df):
    with pytest.raises(DataError, match="not found"):
        read_table(tmp_path / "nope.csv")
    (tmp_path / "d.xlsx").write_text("x")
    with pytest.raises(DataError, match="unsupported"):
        read_table(tmp_path / "d.xlsx")
    (tmp_path / "e.csv").write_text("a,b\n")
    with pytest.raises(DataError, match="no rows"):
        read_table(tmp_path / "e.csv")


def test_manifest_is_deterministic_and_input_immutable(df):
    before = df.copy()
    m1, m2 = build_manifest(df, "y"), build_manifest(df.copy(), "y")
    assert m1 == m2
    pd.testing.assert_frame_equal(df, before)
    changed = df.assign(a=[9, 2, 3])
    assert build_manifest(changed, "y").data_hash != m1.data_hash
    assert build_manifest(changed, "y").schema_hash == m1.schema_hash


def test_missing_target(df):
    with pytest.raises(DataError, match="target"):
        build_manifest(df, "nope")
