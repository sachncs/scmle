"""Tests for split strategies, manifest and the locked test set."""

import numpy as np
import pandas as pd
import pytest

from mml.contracts import errors
from mml.contracts import tasks
from mml.data import splitting

BINARY = tasks.TaskType.BINARY


def make_spec(strategy="auto", **data_overrides):
    return tasks.ProjectSpec(
        data=tasks.DataSpec(uri="m", target="y", **data_overrides),
        validation=tasks.ValidationSpec(strategy=strategy),
    )


def split(frame, spec=None, task=BINARY):
    return splitting.SplitManager(spec or make_spec(), task).split(frame)


@pytest.fixture
def frame():
    rng = np.random.default_rng(0)
    rows = 400
    return pd.DataFrame(
        {
            "x": rng.normal(size=rows),
            "g": rng.integers(0, 40, rows),
            "t": pd.date_range("2024-01-01", periods=rows, freq="h"),
            "y": (rng.random(rows) < 0.2).astype(int),
        }
    )


def test_stratified_preserves_ratio_and_is_disjoint(frame):
    result = split(frame)
    manifest = result.manifest
    assert manifest.strategy == "stratified"
    assert set(manifest.train_ids).isdisjoint(manifest.test_ids)
    assert len(manifest.train_ids) + len(manifest.test_ids) == len(frame)
    assert abs(result.dev_target.mean() - frame.y.mean()) < 0.03
    for train, val in manifest.folds:
        assert set(train).isdisjoint(val)
        assert abs(result.dev_target.iloc[list(val)].mean() - frame.y.mean()) < 0.08
    assert "y" not in result.dev_features.columns


def test_group_split_has_no_group_overlap(frame):
    result = split(frame, make_spec(group_column="g"))
    manifest = result.manifest
    assert manifest.strategy == "group"
    train_groups = set(frame.g.iloc[list(manifest.train_ids)])
    test_groups = set(frame.g.iloc[list(manifest.test_ids)])
    assert train_groups.isdisjoint(test_groups)
    dev_groups = frame.g.iloc[list(manifest.train_ids)].to_numpy()
    for train, val in manifest.folds:
        assert set(dev_groups[list(train)]).isdisjoint(dev_groups[list(val)])


def test_temporal_split_respects_order(frame):
    shuffled = frame.sample(frac=1, random_state=1).reset_index(drop=True)
    manifest = split(shuffled, make_spec(time_column="t")).manifest
    times = shuffled.t
    assert manifest.strategy == "temporal"
    assert times.iloc[list(manifest.train_ids)].max() <= times.iloc[
        list(manifest.test_ids)
    ].min()
    dev_times = times.iloc[list(manifest.train_ids)].to_numpy()
    for train, val in manifest.folds:
        assert dev_times[list(train)].max() <= dev_times[list(val)].min()


def test_temporal_with_unparseable_time_blocks(frame):
    broken = frame.assign(t=frame.t.astype(str))
    broken.loc[3, "t"] = "not a date"
    with pytest.raises(errors.BlockedError, match="ambiguous"):
        split(broken, make_spec(time_column="t"))


def test_group_split_blocks_with_few_groups(frame):
    few = frame.assign(g=frame.index % 3)
    with pytest.raises(errors.BlockedError, match="groups"):
        split(few, make_spec(group_column="g"))


def test_manifest_is_deterministic(frame):
    first = split(frame).manifest
    second = split(frame.copy()).manifest
    assert first.fingerprint == second.fingerprint
    assert first.to_dict()["n_test"] == len(first.test_ids)


def test_tiny_minority_class_blocks(frame):
    rare = frame.assign(y=0)
    rare.loc[0, "y"] = 1
    with pytest.raises(errors.BlockedError):
        split(rare)


def test_folds_shrink_to_minority_size(frame):
    thin = frame.assign(y=0)
    thin.loc[:3, "y"] = 1
    manifest = split(thin).manifest
    assert len(manifest.folds) < 5
    assert any("folds reduced" in note for note in manifest.notes)


def test_regression_defaults_to_random_and_rejects_stratified(frame):
    regression = frame.assign(y=frame.x * 2)
    result = split(regression, task=tasks.TaskType.REGRESSION)
    assert result.manifest.strategy == "random"
    with pytest.raises(errors.BlockedError):
        splitting.StratifiedSplit(make_spec(), tasks.TaskType.REGRESSION)


def test_strategy_registry_is_polymorphic():
    assert set(splitting.STRATEGIES) == {
        "random",
        "stratified",
        "group",
        "temporal",
    }
    assert issubclass(splitting.StratifiedSplit, splitting.RandomSplit)


def test_locked_test_set_opens_once(frame):
    locked = split(frame).locked_test
    assert locked.opened is False and "features" not in repr(locked)
    with pytest.raises(errors.LeakageError, match="invalid token"):
        locked.open("guess")
    features, target = locked.open(locked.issue_token())
    assert len(features) == len(locked) == len(target)
    with pytest.raises(errors.LeakageError, match="already used"):
        locked.open(locked.issue_token())
