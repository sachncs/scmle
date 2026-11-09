"""Arrow-compatible interchange: Arrow, Polars and Feather in; Arrow out."""

import numpy
import pandas
import pyarrow
import pytest
import support

from scmle.contracts import errors
from scmle.data.readers import interchange
from scmle.data.readers import tabular


def run(data, tmp_path, **kwargs):
    return support.make_project(
        data, "y", time_budget=300, output_dir=tmp_path, **kwargs
    ).run()


def test_arrow_table_is_a_first_class_input(binary_frame, tmp_path):
    table = pyarrow.Table.from_pandas(binary_frame, preserve_index=False)
    result = run(table, tmp_path)
    assert result.evaluation["test_metrics"]["roc_auc"] > 0.8
    features = binary_frame.drop(columns="y")
    arrow_features = pyarrow.Table.from_pandas(features, preserve_index=False)
    numpy.testing.assert_array_equal(
        result.predict(arrow_features), result.predict(features)
    )


def test_polars_frames_and_lazyframes_are_accepted(binary_frame, tmp_path):
    polars = pytest.importorskip("polars")
    eager = polars.from_pandas(binary_frame)
    result = run(eager, tmp_path / "eager")
    lazy = run(eager.lazy(), tmp_path / "lazy")
    assert (
        result.evaluation["split_fingerprint"]
        == lazy.evaluation["split_fingerprint"]
    )
    features = polars.from_pandas(binary_frame.drop(columns="y"))
    assert len(result.predict(features)) == len(binary_frame)


def test_results_export_to_arrow(binary_frame, tmp_path):
    result = run(binary_frame, tmp_path)
    table = result.leaderboard_arrow()
    assert isinstance(table, pyarrow.Table) and table.num_rows == len(
        result.trials
    )
    assert "score_mean" in table.column_names


def test_feather_files_are_read_and_scored(binary_frame, tmp_path):
    path = tmp_path / "d.feather"
    binary_frame.to_feather(path)
    assert tabular.read_table(path).shape == binary_frame.shape
    result = run(binary_frame, tmp_path / "r")
    new = tmp_path / "new.arrow"
    binary_frame.drop(columns="y").to_feather(new)
    assert len(result.predict(new)) == len(binary_frame)


def test_to_pandas_copies_and_rejects_non_tables():
    frame = pandas.DataFrame({"a": [1, 2]})
    copy = interchange.to_pandas(frame)
    copy.loc[0, "a"] = 99
    assert frame.loc[0, "a"] == 1
    with pytest.raises(errors.DataError):
        interchange.to_pandas(42)
    assert not interchange.is_tabular("some/path.csv")
    assert interchange.is_tabular(pyarrow.table({"a": [1]}))
