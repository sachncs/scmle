"""End-to-end runs over every task type and split strategy."""

import dataclasses
import json
import pathlib

import numpy as np
import pytest
import support

import mml
from mml.contracts import errors


def read_kinds(run_dir):
    lines = (pathlib.Path(run_dir) / "events.jsonl").read_text().splitlines()
    return [json.loads(line)["kind"] for line in lines]


def run(frame, tmp_path, **kwargs):
    project = support.make_project(
        frame, "y", time_budget=120, output_dir=tmp_path, **kwargs
    )
    return project.run()


def test_binary_end_to_end_artifacts_and_inference(binary_frame, tmp_path):
    result = run(binary_frame, tmp_path)
    evaluation = result.evaluation
    assert evaluation["test_metrics"]["roc_auc"] > 0.8
    assert evaluation["selected_cv_score"] > evaluation["baseline_cv_score"]
    assert evaluation["test_set_opened"] is True and evaluation["n_test"] == 100
    expected = [
        "config.json",
        "environment.lock",
        "dataset_manifest.json",
        "dataset_profile.json",
        "validation_manifest.json",
        "execution_plan.json",
        "events.jsonl",
        "leaderboard.parquet",
        "evaluation.json",
        "report.html",
        "model/pipeline.joblib",
        "model/schema.json",
        "model/model_manifest.json",
        "diagnostics/decision_log.json",
        "trials/trial_001.json",
    ]
    for relative in expected:
        assert (result.run_dir / relative).exists(), relative
    states = [k for k in read_kinds(result.run_dir) if k.startswith("state.")]
    assert states == [
        f"state.{name}"
        for name in (
            "CREATED",
            "VALIDATING",
            "PROFILING",
            "PLANNING",
            "TRAINING",
            "SELECTING",
            "TESTING",
            "PACKAGING",
            "COMPLETED",
        )
    ]
    assert result.leaderboard().iloc[0]["selected"]
    assert "trial.ok" in read_kinds(result.run_dir)

    exported = result.save(tmp_path / "exported")
    loaded = mml.ModelBundle.load(exported)
    features = binary_frame.drop(columns="y")
    np.testing.assert_allclose(
        result.predict_proba(features), loaded.predict_proba(features)
    )
    assert result.predict(features.head(7)).shape == (7,)
    assert "run_" in result.summary()


def test_multiclass_with_string_labels(multiclass_frame, tmp_path):
    result = run(multiclass_frame, tmp_path, task="classification")
    assert result.evaluation["task"] == "multiclass_classification"
    features = multiclass_frame.drop(columns="y")
    assert set(result.predict(features)) <= {"lo", "mid", "hi"}
    assert result.predict_proba(features).shape[1] == 3


def test_regression_is_ranked_by_lower_rmse(regression_frame, tmp_path):
    result = run(regression_frame, tmp_path, metric="rmse")
    assert result.evaluation["test_metrics"]["r2"] > 0.9
    assert (
        result.evaluation["selected_cv_score"]
        < result.evaluation["baseline_cv_score"]
    )
    board = result.leaderboard()
    succeeded = board[board.status == "ok"]
    assert board.iloc[0]["score_mean"] == succeeded["score_mean"].min()


def test_csv_input_and_plan_without_side_effects(binary_frame, tmp_path):
    binary_frame.to_csv(tmp_path / "d.csv", index=False)
    project = support.make_project(
        tmp_path / "d.csv", "y", output_dir=tmp_path / "runs"
    )
    plan = project.plan()
    assert not (tmp_path / "runs").exists()
    assert project.profile is not None and "dummy" in plan.candidate_models
    assert project.run(plan).plan.plan_id == plan.plan_id


def test_approved_plan_restricts_models(binary_frame, tmp_path):
    project = support.make_project(binary_frame, "y", output_dir=tmp_path)
    plan = dataclasses.replace(
        project.plan(), candidate_models=("dummy", "logistic_regression")
    )
    result = project.run(plan)
    assert {t.model_name for t in result.trials} == {
        "dummy",
        "logistic_regression",
    }


def test_group_and_temporal_runs(binary_frame, tmp_path):
    frame = binary_frame.assign(
        g=np.arange(len(binary_frame)) % 50, t=np.arange(len(binary_frame))
    )
    assert (
        run(frame, tmp_path, group_column="g").evaluation["split_strategy"]
        == "group"
    )
    assert (
        run(frame, tmp_path, time_column="t").evaluation["split_strategy"]
        == "temporal"
    )


def test_blocked_run_records_terminal_state(binary_frame, tmp_path):
    ambiguous = binary_frame.assign(y=np.arange(len(binary_frame)) % 5)
    with pytest.raises(errors.BlockedError) as caught:
        run(ambiguous, tmp_path)
    assert read_kinds(caught.value.details["run_dir"])[-1] == "state.BLOCKED"


def test_unknown_metric_fails_fast_and_is_recorded(binary_frame, tmp_path):
    with pytest.raises(errors.ConfigError) as caught:
        run(binary_frame, tmp_path, metric="nope")
    assert read_kinds(caught.value.details["run_dir"])[-1] == "state.FAILED"


def test_reruns_are_isolated_and_reproducible(binary_frame, tmp_path):
    first, second = run(binary_frame, tmp_path), run(binary_frame, tmp_path)
    assert first.run_dir != second.run_dir
    assert (
        first.evaluation["split_fingerprint"]
        == second.evaluation["split_fingerprint"]
    )
    assert first.evaluation["selected_cv_score"] == pytest.approx(
        second.evaluation["selected_cv_score"]
    )
