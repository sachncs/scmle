"""Tests for task resolution and dataset profiling."""

import numpy as np
import pandas as pd
import pytest

from mml.contracts import errors
from mml.contracts import tasks
from mml.data.profiling import profiler
from mml.data.readers import tabular
from mml.data.validation import target as target_module


def make_spec(task="auto", **data_overrides):
    data = tasks.DataSpec(uri="m", target="y", **data_overrides)
    return tasks.ProjectSpec(data=data, task=task)


def profile(frame, task="auto", **data_overrides):
    spec = make_spec(task, **data_overrides)
    resolved = target_module.TaskResolver(spec).resolve(frame["y"])
    manifest = tabular.build_manifest(frame, "y")
    result = profiler.DatasetProfiler(spec).profile(frame, manifest, resolved)
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
    return target_module.TaskResolver(make_spec(task)).resolve(frame["y"])


def test_task_resolution(classification_frame):
    frame = classification_frame
    assert resolve(frame) is tasks.TaskType.BINARY
    assert resolve(frame, "classification") is tasks.TaskType.BINARY
    multi = frame.assign(y=np.arange(len(frame)) % 4)
    assert resolve(multi, "classification") is tasks.TaskType.MULTICLASS
    continuous = frame.assign(
        y=np.random.default_rng(1).normal(size=len(frame))
    )
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
    result = profile(classification_frame)[1]
    plan = result.feature_plan
    assert plan.numeric == ("num",) and plan.categorical == ("cat",)
    assert plan.dropped["const"] == "constant"
    assert "id-like" in plan.dropped["uid"]


def test_roles_are_excluded_from_features(classification_frame):
    frame = classification_frame.assign(grp=np.arange(300) % 30, rid=range(300))
    result = profile(frame, group_column="grp", id_columns=["rid"])[1]
    dropped = result.feature_plan.dropped
    assert "splitting" in dropped["grp"] and dropped["rid"] == "id column"
    assert "grp" not in result.feature_plan.features


def test_missing_configured_column(classification_frame):
    with pytest.raises(errors.DataError, match="not found"):
        profile(classification_frame, excluded_columns=["ghost"])


def test_leaky_feature_flagged(classification_frame):
    leaky = classification_frame.assign(leak=classification_frame["y"] * 10 + 1)
    result = profile(leaky)[1]
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
    result = profile(skewed)[1]
    assert any("imbalance" in r for r in result.risk_findings)
    duplicated = pd.concat([frame, frame.head(5)])
    result = profile(duplicated)[1]
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
    result = profile(classification_frame)[1]
    payload = result.to_dict()
    assert payload["feature_plan"]["numeric"] == ["num"]


def test_outcome_dependent_recording_is_reported_as_possible_leakage():
    rng = np.random.default_rng(4)
    rows = 400
    frame = pd.DataFrame({"signal": rng.normal(size=rows)})
    frame["y"] = rng.integers(0, 2, rows)
    frame["discharge_code"] = np.where(
        frame.y == 1, rng.integers(1, 5, rows), np.nan
    )
    result = profile(frame)[1]
    assert any(
        "'discharge_code'" in finding and "recorded" in finding
        for finding in result.risk_findings
    )
    assert (
        result.column_statistics["discharge_code"]["missingness_association"]
        >= 0.9
    )
    assert result.column_statistics["signal"]["missingness_association"] == 0.0


def test_features_unavailable_at_prediction_time_are_excluded_with_a_reason(
    classification_frame,
):
    frame = classification_frame.assign(refund_issued=lambda f: f["y"] * 3.0)
    result = profile(frame, unavailable_at_prediction=["refund_issued"])[1]
    reason = result.feature_plan.dropped["refund_issued"]
    assert "not available at prediction time" in reason
    assert "refund_issued" not in result.feature_plan.features


def test_target_cannot_be_declared_unavailable():
    with pytest.raises(Exception, match="unavailable"):
        make_spec(unavailable_at_prediction=["y"])
