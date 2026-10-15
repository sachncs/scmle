"""Tests for task resolution and dataset profiling."""

import numpy as np
import pandas as pd
import pytest

from mml.contracts import errors
from mml.contracts import tasks
from mml.data import profiling
from mml.data import readers


def make_spec(task="auto", **data_overrides):
    data = tasks.DataSpec(uri="m", target="y", **data_overrides)
    return tasks.ProjectSpec(data=data, task=task)


def profile(frame, task="auto", **data_overrides):
    spec = make_spec(task, **data_overrides)
    resolved = profiling.TaskResolver(spec).resolve(frame["y"])
    manifest = readers.build_manifest(frame, "y")
    result = profiling.DatasetProfiler(spec).profile(frame, manifest, resolved)
    return resolved, result


@pytest.fixture
def classification_frame():
    rng = np.random.default_rng(0)
    rows = 300
    return pd.DataFrame(
        {
            "num": rng.normal(size=rows),
            "cat": rng.choice(list("abc"), rows),
            "const": 1,
            "uid": [f"u{i}" for i in range(rows)],
            "y": rng.integers(0, 2, rows),
        }
    )


def resolve(frame, task="auto"):
    return profiling.TaskResolver(make_spec(task)).resolve(frame["y"])


def test_task_resolution(classification_frame):
    frame = classification_frame
    assert resolve(frame) is tasks.TaskType.BINARY
    assert resolve(frame, "classification") is tasks.TaskType.BINARY
    multi = frame.assign(y=np.arange(len(frame)) % 4)
    assert resolve(multi, "classification") is tasks.TaskType.MULTICLASS
    continuous = frame.assign(y=np.random.default_rng(1).normal(size=len(frame)))
    assert resolve(continuous) is tasks.TaskType.REGRESSION
    strings = frame.assign(y=np.where(frame.y == 1, "yes", "no"))
    assert resolve(strings) is tasks.TaskType.BINARY


def test_ambiguous_integer_target_blocks(classification_frame):
    ambiguous = classification_frame.assign(
        y=np.arange(len(classification_frame)) % 5
    )
    with pytest.raises(errors.BlockedError, match="ambiguous"):
        resolve(ambiguous)


def test_explicit_requests_must_match_data(classification_frame):
    with pytest.raises(errors.BlockedError):
        resolve(classification_frame, "multiclass_classification")
    two_class = classification_frame.assign(y=np.arange(300) % 3)
    with pytest.raises(errors.BlockedError):
        resolve(two_class, "binary_classification")
    text = classification_frame.assign(y=np.where(np.arange(300) % 2, "a", "b"))
    with pytest.raises(errors.BlockedError):
        resolve(text, "regression")
    with pytest.raises(errors.BlockedError):
        resolve(classification_frame.assign(y=1), "classification")


def test_feature_plan_drops_constant_and_id_like(classification_frame):
    _, result = profile(classification_frame)
    plan = result.feature_plan
    assert plan.numeric == ("num",) and plan.categorical == ("cat",)
    assert plan.dropped["const"] == "constant"
    assert "id-like" in plan.dropped["uid"]


def test_roles_are_excluded_from_features(classification_frame):
    frame = classification_frame.assign(grp=np.arange(300) % 30, rid=range(300))
    _, result = profile(frame, group_column="grp", id_columns=["rid"])
    dropped = result.feature_plan.dropped
    assert "splitting" in dropped["grp"] and dropped["rid"] == "id column"
    assert "grp" not in result.feature_plan.features


def test_missing_configured_column(classification_frame):
    with pytest.raises(errors.DataError, match="not found"):
        profile(classification_frame, excluded_columns=["ghost"])


def test_leaky_feature_flagged(classification_frame):
    leaky = classification_frame.assign(leak=classification_frame["y"] * 10 + 1)
    _, result = profile(leaky)
    assert any("'leak'" in finding for finding in result.risk_findings)
    assert not any("'num'" in finding for finding in result.risk_findings)
    assert result.column_statistics["leak"]["target_association"] >= 0.98


def test_missing_target_blocks(classification_frame):
    broken = classification_frame.copy()
    broken.loc[0, "y"] = np.nan
    with pytest.raises(errors.BlockedError, match="missing"):
        profile(broken)


def test_no_usable_features_blocks():
    frame = pd.DataFrame({"c": 1, "y": [0, 1] * 20})
    with pytest.raises(errors.BlockedError, match="no usable"):
        profile(frame)


def test_imbalance_and_duplicates(classification_frame):
    frame = classification_frame
    skewed = pd.concat([frame[frame.y == 0], frame[frame.y == 1].head(5)])
    _, result = profile(skewed)
    assert any("imbalance" in r for r in result.risk_findings)
    duplicated = pd.concat([frame, frame.head(5)])
    _, result = profile(duplicated)
    assert any("duplicate" in q for q in result.quality_findings)


def test_regression_target_summary():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"x": rng.normal(size=100)})
    frame["y"] = frame.x * 2
    task, result = profile(frame)
    assert task is tasks.TaskType.REGRESSION
    assert {"mean", "std", "skew"} <= set(result.target_summary)
    assert any("'x'" in r for r in result.risk_findings)


def test_profile_serializes(classification_frame):
    _, result = profile(classification_frame)
    payload = result.to_dict()
    assert payload["feature_plan"]["numeric"] == ["num"]
