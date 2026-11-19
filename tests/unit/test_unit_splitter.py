"""Behavioural tests for the split strategy registry and the splitter."""

import numpy
import pandas
import pytest

from scmle.contracts import errors
from scmle.contracts import tasks
from scmle.data.splitting.splitters import partitions
from scmle.data.splitting.splitters import spatial
from scmle.data.splitting.splitters import splits
from scmle.data.splitting.splitters import splitter
from scmle.data.splitting.splitters import temporal


def spec(strategy="auto", **data):
    return tasks.Config(
        data=tasks.Dataset(uri="m", target="y", **data),
        validation=tasks.Validation(strategy=strategy, seed=3),
    )


def frame(rows=200, seed=0):
    rng = numpy.random.default_rng(seed)
    return pandas.DataFrame(
        {
            "x": rng.normal(size=rows),
            "g": rng.integers(0, 20, rows),
            "t": pandas.date_range("2024-01-01", periods=rows, freq="h"),
            "y": (rng.random(rows) < 0.3).astype(int),
        }
    )


def test_the_registry_names_every_strategy():
    assert {
        "random": splits.Random,
        "stratified": splits.Stratified,
        "stratified_events": splits.Events,
        "multilabel_stratified": splits.Multilabel,
        "treatment_stratified": splits.Treatment,
        "preset": splits.Preset,
        "group": splits.Group,
        "temporal": temporal.Chronology,
        "rolling": temporal.Rolling,
        "grouped_temporal": temporal.Succession,
        "spatial_block": spatial.Tiling,
    } == splitter.STRATEGIES
    assert splitter.TASK_STRATEGIES == {
        tasks.Task.SURVIVAL: "stratified_events",
        tasks.Task.RANKING: "group",
        tasks.Task.MULTILABEL: "multilabel_stratified",
        tasks.Task.UPLIFT: "treatment_stratified",
    }


@pytest.mark.parametrize(
    ("data", "task", "expected"),
    [
        (
            {"time_column": "t", "group_column": "g"},
            tasks.Task.BINARY,
            "temporal",
        ),
        ({"group_column": "g"}, tasks.Task.SURVIVAL, "group"),
        ({}, tasks.Task.SURVIVAL, "stratified_events"),
        ({}, tasks.Task.RANKING, "group"),
        ({}, tasks.Task.BINARY, "stratified"),
        ({}, tasks.Task.MULTICLASS, "stratified"),
        ({}, tasks.Task.REGRESSION, "random"),
    ],
)
def test_scheme_chooses_by_data_structure(data, task, expected):
    assert splitter.scheme(spec(**data), task) == expected


def test_scheme_respects_an_explicit_strategy():
    assert splitter.scheme(
        spec("random", time_column="t"), tasks.Task.BINARY
    ) == ("random")
    built = splitter.Splitter(
        spec("group", group_column="g"), tasks.Task.BINARY
    )
    assert built.strategy_name == "group"
    assert built.task is tasks.Task.BINARY


def test_feature_columns_drop_every_role():
    data = pandas.DataFrame(
        columns=["x", "y", "g", "t", "w", "i", "e", "u", "z"]
    )
    built = splitter.Splitter(
        spec(
            group_column="g",
            time_column="t",
            weight_column="w",
            id_columns=["i"],
            excluded_columns=["e"],
            unavailable_at_prediction=["u"],
        ),
        tasks.Task.REGRESSION,
    )
    assert built.feature_columns(data) == ["x", "z"]


def test_split_is_disjoint_and_records_its_manifest():
    data = frame()
    result = splitter.Splitter(spec(), tasks.Task.BINARY).split(data)
    manifest = result.manifest
    assert manifest.strategy == "stratified"
    assert manifest.seed == 3
    assert not set(manifest.train_ids) & set(manifest.test_ids)
    assert len(manifest.train_ids) + len(manifest.test_ids) == 200
    assert len(manifest.test_ids) == 40
    assert len(result.dev_features) == len(manifest.train_ids)
    assert "y" not in result.dev_features.columns
    assert (
        result.dev_target.tolist()
        == data["y"].iloc[list(manifest.train_ids)].tolist()
    )
    assert list(result.dev_features.index) == list(range(160))
    assert len(manifest.folds) == 5


def test_split_uses_a_preset_side_column():
    data = frame(10)
    data["split_side"] = [0] * 8 + [1] * 2
    result = splitter.Splitter(spec(), tasks.Task.REGRESSION).split(data)
    assert result.manifest.strategy == "preset"
    assert result.manifest.test_ids == (8, 9)
    # two test rows cannot score both classes of a binary task
    with pytest.raises(errors.BlockedError, match="test rows"):
        splitter.Splitter(spec(), tasks.Task.BINARY).split(data)


def test_disclose_counts_test_rows_that_repeat_development():
    data = pandas.DataFrame(
        {"x": [1.0, 1.0, 2.0, 3.0, 4.0, 5.0], "y": [0, 0, 1, 0, 1, 0]}
    )
    built = splitter.Splitter(spec(), tasks.Task.BINARY)
    part = partitions.Partition(
        numpy.array([0, 2, 3, 4]), numpy.array([1, 5]), [], ["n"]
    )
    told = built.disclose(data, part)
    assert told.dev_ids.tolist() == [0, 2, 3, 4]
    assert told.test_ids.tolist() == [1, 5]
    assert told.notes[0] == "n"
    assert told.notes[1].startswith(
        "1 of 2 test rows repeat a development record"
    )
    unique = pandas.DataFrame({"x": [1.0, 2.0], "y": [0, 1]})
    plain = partitions.Partition(numpy.array([0]), numpy.array([1]), [], [])
    assert built.disclose(unique, plain) is plain


def test_duplicates_are_ignored_by_identity_columns():
    data = pandas.DataFrame({"i": [1, 2], "x": [1.0, 1.0], "y": [0, 1]})
    built = splitter.Splitter(spec(id_columns=["i"]), tasks.Task.BINARY)
    part = partitions.Partition(numpy.array([0]), numpy.array([1]), [], [])
    assert built.disclose(data, part) is not part
    keyed = splitter.Splitter(spec(), tasks.Task.BINARY)
    assert keyed.disclose(data, part) is part
