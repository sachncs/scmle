"""Tests for task resolution and dataset profiling."""

import numpy
import pandas
import pytest

from scmle.contracts import errors
from scmle.contracts import tasks
from scmle.data.profiling import profiler
from scmle.data.readers import tabular
import scmle.data.validation.target


def make_spec(task="auto", **data_overrides):
    data = tasks.Dataset(uri="m", target="y", **data_overrides)
    return tasks.Config(data=data, task=task)


def profile(frame, task="auto", **data_overrides):
    spec = make_spec(task, **data_overrides)
    resolved = scmle.data.validation.target.TaskResolver(spec).resolve(
        frame["y"]
    )
    manifest = tabular.build_manifest(frame, "y")
    result = profiler.Profiler(spec).profile(frame, manifest, resolved)
    return resolved, result


@pytest.fixture
def classification_frame():
    rng = numpy.random.default_rng(0)
    rows = 300
    return pandas.DataFrame(
        {
            "num": rng.normal(size=rows),
            "cat": rng.choice(list("abc"), rows),
            "const": 1,
            "uid": [f"u{i}" for i in range(rows)],
            "y": rng.integers(0, 2, rows),
        }
    )


def resolve(frame, task="auto"):
    return scmle.data.validation.target.TaskResolver(make_spec(task)).resolve(
        frame["y"]
    )


def test_task_resolution(classification_frame):
    frame = classification_frame
    assert resolve(frame) is tasks.Task.BINARY
    assert resolve(frame, "classification") is tasks.Task.BINARY
    multi = frame.assign(y=numpy.arange(len(frame)) % 4)
    assert resolve(multi, "classification") is tasks.Task.MULTICLASS
    continuous = frame.assign(
        y=numpy.random.default_rng(1).normal(size=len(frame))
    )
    assert resolve(continuous) is tasks.Task.REGRESSION
    strings = frame.assign(y=numpy.where(frame.y == 1, "yes", "no"))
    assert resolve(strings) is tasks.Task.BINARY


def test_ambiguous_integer_target_blocks(classification_frame):
    ambiguous = classification_frame.assign(
        y=numpy.arange(len(classification_frame)) % 5
    )
    with pytest.raises(errors.BlockedError, match="ambiguous"):
        resolve(ambiguous)


def test_explicit_requests_must_match_data(classification_frame):
    with pytest.raises(errors.BlockedError):
        resolve(classification_frame, "multiclass_classification")
    two_class = classification_frame.assign(y=numpy.arange(300) % 3)
    with pytest.raises(errors.BlockedError):
        resolve(two_class, "binary_classification")
    text = classification_frame.assign(
        y=numpy.where(numpy.arange(300) % 2, "a", "b")
    )
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
    frame = classification_frame.assign(
        grp=numpy.arange(300) % 30, rid=range(300)
    )
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
    screen = result.leakage
    assert screen["suspects"] == ["leak"]
    assert screen["blocking"] == ["leak"]
    assert screen["rule"] == "benjamini-yekutieli"
    (found,) = [f for f in screen["findings"] if f["suspect"]]
    assert found["adjusted_p"] <= screen["alpha"]
    assert found["lower_bound"] > screen["ceilings"]["skill"]
    assert result.blocking_leaks == ("leak",)
    assert "leakage" in result.to_dict()


def test_clean_table_has_an_empty_screen_result(classification_frame):
    result = profile(classification_frame)[1]
    assert result.leakage["suspects"] == []
    assert result.leakage["blocking"] == []
    assert result.blocking_leaks == ()
    assert not result.risk_findings


def test_a_strong_legitimate_predictor_is_not_a_leakage_risk():
    rng = numpy.random.default_rng(3)
    rows = 600
    y = rng.integers(0, 2, rows)
    frame = pandas.DataFrame(
        {"strong": y + rng.normal(scale=0.8, size=rows), "y": y}
    )
    result = profile(frame)[1]
    assert not any("leakage" in r for r in result.risk_findings)
    (finding,) = result.leakage["findings"]
    assert finding["significant"] and not finding["suspect"]


def test_the_screen_uses_groups_and_the_time_order(classification_frame):
    frame = classification_frame.assign(
        leak=classification_frame["y"] + 0.0,
        grp=numpy.arange(300) % 30,
        when=numpy.arange(300)[::-1],
    )
    grouped = profile(frame, group_column="grp")[1]
    timed = profile(frame, time_column="when")[1]
    assert grouped.leakage["suspects"] == ["leak"]
    assert timed.leakage["suspects"] == ["leak"]


def test_the_screen_is_skipped_for_a_non_numeric_non_class_target():
    frame = pandas.DataFrame({"a": range(120), "y": ["x"] * 120})
    spec = make_spec("auto")
    report = profiler.Profiler(spec).leakage_screen(
        frame, ["a"], tasks.Task.MULTILABEL
    )
    assert report is None


def test_missing_target_blocks(classification_frame):
    broken = classification_frame.copy()
    broken.loc[0, "y"] = numpy.nan
    with pytest.raises(errors.BlockedError, match="missing"):
        profile(broken)


def test_no_usable_features_blocks():
    frame = pandas.DataFrame({"c": 1, "y": [0, 1] * 20})
    with pytest.raises(errors.BlockedError, match="no usable"):
        profile(frame)


def test_imbalance_and_duplicates(classification_frame):
    frame = classification_frame
    skewed = pandas.concat([frame[frame.y == 0], frame[frame.y == 1].head(5)])
    result = profile(skewed)[1]
    assert any("imbalance" in r for r in result.risk_findings)
    duplicated = pandas.concat([frame, frame.head(5)])
    result = profile(duplicated)[1]
    assert any("duplicate" in q for q in result.quality_findings)


def test_regression_target_summary():
    rng = numpy.random.default_rng(0)
    frame = pandas.DataFrame({"x": rng.normal(size=100)})
    frame["y"] = frame.x * 2
    task, result = profile(frame)
    assert task is tasks.Task.REGRESSION
    assert {"mean", "std", "skew"} <= set(result.target_summary)
    assert any("'x'" in r for r in result.risk_findings)


def test_profile_serializes(classification_frame):
    result = profile(classification_frame)[1]
    payload = result.to_dict()
    assert payload["feature_plan"]["numeric"] == ["num"]


def test_outcome_dependent_recording_is_reported_as_possible_leakage():
    rng = numpy.random.default_rng(4)
    rows = 400
    frame = pandas.DataFrame({"signal": rng.normal(size=rows)})
    frame["y"] = rng.integers(0, 2, rows)
    frame["discharge_code"] = numpy.where(
        frame.y == 1, rng.integers(1, 5, rows), numpy.nan
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


def test_datetime_columns_become_calendar_and_trend_features():
    rng = numpy.random.default_rng(0)
    frame = pandas.DataFrame(
        {
            "signed_up": pandas.date_range(
                "2022-01-01", periods=400, freq="40h"
            ),
            "constant_day": pandas.Timestamp("2024-03-04"),
            "x": rng.normal(size=400),
            "y": rng.integers(0, 2, 400),
        }
    )
    spec = tasks.Config(data=tasks.Dataset(uri="m", target="y"))
    manifest = tabular.build_manifest(frame, "y")

    profile = profiler.Profiler(spec).profile(
        frame, manifest, tasks.Task.BINARY
    )

    plan = profile.feature_plan
    names = {n.name for n in plan.derived}
    assert {
        "signed_up__year",
        "signed_up__month",
        "signed_up__dayofweek",
        "signed_up__hour",
        "signed_up__timestamp",
    } <= names
    assert not any(
        n.startswith("constant_day__") and "year" in n for n in names
    )
    assert "signed_up" in plan.extra_inputs and "signed_up" not in plan.features
    assert set(names) <= set(plan.numeric)
