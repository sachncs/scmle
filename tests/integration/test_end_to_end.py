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
    loaded = mml.ModelBundle.load(exported, trusted=True)
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


@pytest.mark.parametrize("metric", ["pr_auc", "f1", "roc_auc"])
def test_string_labelled_binary_target_supports_every_binary_metric(
    binary_frame, tmp_path, metric
):
    labelled = binary_frame.assign(
        y=binary_frame.y.map({0: "false", 1: "true"})
    )
    result = run(labelled, tmp_path, metric=metric)
    assert result.evaluation["metric"] == metric
    assert not np.isnan(result.evaluation["selected_cv_score"])
    assert not any(
        np.isnan(v) for v in result.evaluation["test_metrics"].values()
    )
    assert set(result.predict(labelled.drop(columns="y"))) <= {"false", "true"}


def test_diagnostics_are_written_and_exposed(binary_frame, tmp_path):
    result = run(binary_frame, tmp_path)
    names = set(result.evaluation["diagnostics"])
    assert {
        "classification_report",
        "calibration",
        "permutation_importance",
    } <= names
    for name in names:
        assert (result.run_dir / "diagnostics" / f"{name}.json").exists()
    importance = result.feature_importance()
    assert list(importance.columns) == ["feature", "importance", "std"]
    assert importance.iloc[0]["feature"] in {"a", "b", "c"}
    html = (result.run_dir / "report.html").read_text()
    assert "Feature importance" in html and "Brier score" in html


def test_regression_diagnostics_include_residuals(regression_frame, tmp_path):
    result = run(regression_frame, tmp_path, task="regression")
    assert "residuals" in result.diagnostics
    assert result.feature_importance().iloc[0]["feature"] in {"a", "b"}


def test_exported_bundle_can_be_signed_and_loaded_without_trust(
    binary_frame, tmp_path
):
    result = run(binary_frame, tmp_path)
    exported = result.save(tmp_path / "signed", signing_key="shared-secret")
    loaded = mml.ModelBundle.load(exported, signing_key="shared-secret")
    features = binary_frame.drop(columns="y")
    np.testing.assert_array_equal(
        loaded.predict(features), result.predict(features)
    )
    with pytest.raises(Exception, match="unauthenticated"):
        mml.ModelBundle.load(exported)


def test_confidence_intervals_follow_the_data_structure(binary_frame, tmp_path):
    frame = binary_frame.assign(
        g=np.arange(len(binary_frame)) % 40, t=np.arange(len(binary_frame))
    )
    plain = run(frame, tmp_path / "p").diagnostics["metric_uncertainty"]
    grouped = run(frame, tmp_path / "g", group_column="g").diagnostics[
        "metric_uncertainty"
    ]
    temporal = run(frame, tmp_path / "t", time_column="t").diagnostics[
        "metric_uncertainty"
    ]
    assert plain["resampling_unit"] == "rows"
    assert (
        grouped["resampling_unit"] == "groups"
        and grouped["independent_units"] < 40
    )
    assert temporal["resampling_unit"] == "blocks"
    low, high = plain["ci_95"]
    assert low <= plain["estimate"] <= high


def test_time_series_contracts_end_to_end(binary_frame, tmp_path):
    frame = binary_frame.assign(
        t=np.arange(len(binary_frame)), outcome_code=binary_frame.y * 9.0
    )
    project = support.make_project(
        frame,
        "y",
        time_budget=300,
        time_column="t",
        unavailable_at_prediction=["outcome_code"],
        output_dir=tmp_path,
    )
    result = project.run()
    assert "outcome_code" not in result.bundle.features
    assert (
        "not available" in result.profile.feature_plan.dropped["outcome_code"]
    )
    assert result.evaluation["split_strategy"] == "temporal"


def test_a_modified_plan_changes_validation_budget_and_metric(
    binary_frame, tmp_path
):
    project = support.make_project(binary_frame, "y", output_dir=tmp_path)
    plan = project.plan()
    modified = dataclasses.replace(
        plan,
        validation_strategy="random",
        metric="pr_auc",
        optimization_budget_seconds=250,
        candidate_models=("dummy", "logistic_regression"),
    )
    result = project.run(modified)
    assert result.evaluation["split_strategy"] == "random"
    assert result.evaluation["metric"] == "pr_auc"
    saved = json.loads((result.run_dir / "config.json").read_text())
    assert saved["validation"]["strategy"] == "random"
    assert saved["resources"]["time_budget_seconds"] == 250
    assert saved["metric"] == "pr_auc"
    assert {t.model_name for t in result.trials} == {
        "dummy",
        "logistic_regression",
    }


def test_a_plan_for_another_task_or_an_impossible_strategy_is_refused(
    binary_frame, tmp_path
):
    import pydantic

    project = support.make_project(binary_frame, "y", output_dir=tmp_path)
    plan = project.plan()
    wrong_task = dataclasses.replace(plan, task="regression")
    with pytest.raises(errors.ConfigError, match="approved plan is for"):
        project.run(wrong_task)
    impossible = dataclasses.replace(plan, validation_strategy="group")
    with pytest.raises(pydantic.ValidationError, match="group_column"):
        support.make_project(binary_frame, "y", output_dir=tmp_path / "x").run(
            impossible
        )


def test_a_run_explains_individual_predictions_against_a_background(
    binary_frame, tmp_path
):
    result = support.make_project(
        binary_frame, "y", time_budget=300, output_dir=tmp_path
    ).run()
    rows = binary_frame.drop(columns="y").head(3)
    background = binary_frame.drop(columns="y").iloc[100:160]

    table = result.explain(rows, background)

    assert list(table.columns) == ["prediction", "a", "b", "c"]
    assert len(table) == 3 and table["prediction"].between(0, 1).all()
    assert table[["a", "b", "c"]].abs().to_numpy().sum() > 0
    assert "partial_dependence" in result.diagnostics


def test_planning_writes_recommendations_the_project_can_return(
    binary_frame, tmp_path
):
    project = support.make_project(
        binary_frame.head(150), "y", time_budget=300, output_dir=tmp_path
    )

    advice = project.recommendations()
    result = project.run()

    assert any(item.topic == "small data" for item in advice)
    written = json.loads(
        (result.run_dir / "diagnostics" / "recommendations.json").read_text()
    )
    assert [item["topic"] for item in written] == [i.topic for i in advice]


def test_a_pytorch_model_family_trains_packages_and_predicts(
    binary_frame, tmp_path
):
    pytest.importorskip("torch")
    project = support.make_project(
        binary_frame, "y", time_budget=600, output_dir=tmp_path
    )
    plan = project.plan()
    only_torch = dataclasses.replace(
        plan, candidate_models=("dummy", "torch_residual")
    )

    result = project.run(only_torch)

    assert result.selected_model == "torch_residual"
    features = binary_frame.drop(columns="y").head(20)
    result.save(tmp_path / "net")
    reloaded = mml.load_model(tmp_path / "net", trusted=True)
    np.testing.assert_array_equal(
        reloaded.predict(features), result.predict(features)
    )
    assert reloaded.predict_proba(features).shape == (20, 2)
