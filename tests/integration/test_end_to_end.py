"""End-to-end runs over every task type and split strategy."""

import dataclasses
import json
import pathlib

import numpy as np
import pandas as pd
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


def test_fit_health_is_recorded_for_the_selected_model_and_each_trial(
    binary_frame, tmp_path
):
    result = support.make_project(
        binary_frame, "y", time_budget=300, output_dir=tmp_path
    ).run()

    health = result.evaluation["fit_health"]
    board = result.leaderboard()

    assert health["verdict"] in {"healthy", "overfit", "underfit", "unstable"}
    assert set(board["fit_health"]) <= {
        "",
        "healthy",
        "overfit",
        "underfit",
        "unstable",
    }
    assert "train_scores" not in board.columns
    assert all(len(t.train_scores) == len(t.fold_scores) for t in result.trials)


def test_reliability_checks_run_and_reach_the_report(binary_frame, tmp_path):
    result = support.make_project(
        binary_frame, "y", time_budget=300, output_dir=tmp_path
    ).run()

    assert {"slice_performance", "robustness", "adversarial_validation"} <= set(
        result.diagnostics
    )
    assert result.diagnostics["adversarial_validation"]["available"] is True
    page = (result.run_dir / "report.html").read_text()
    assert "Reliability checks" in page and "Fit health" in page
    assert "Development versus test rows" in page


def test_regression_bundles_return_calibrated_intervals(tmp_path):
    rng = np.random.default_rng(11)
    frame = pd.DataFrame({"x": rng.normal(size=900)})
    frame["y"] = 4 * frame.x + rng.normal(0, 1.0, 900)
    result = support.make_project(
        frame, "y", time_budget=300, output_dir=tmp_path
    ).run()

    new = pd.DataFrame({"x": rng.normal(size=3000)})
    truth = 4 * new.x + rng.normal(0, 1.0, 3000)
    narrow = result.bundle.predict_interval(new, coverage=0.5)
    wide = result.bundle.predict_interval(new, coverage=0.95)

    covered_wide = ((truth >= wide["lower"]) & (truth <= wide["upper"])).mean()
    assert covered_wide > 0.9
    assert (wide["upper"] - wide["lower"]).mean() > (
        narrow["upper"] - narrow["lower"]
    ).mean()
    with pytest.raises(errors.ArtifactError, match="classification"):
        result.bundle.predict_set(new)


def test_classification_bundles_return_prediction_sets_that_survive_saving(
    binary_frame, tmp_path
):
    result = support.make_project(
        binary_frame, "y", time_budget=300, output_dir=tmp_path
    ).run()
    result.save(tmp_path / "model")
    reloaded = mml.load_model(tmp_path / "model", trusted=True)
    rows = binary_frame.drop(columns="y")

    sets = reloaded.predict_set(rows, coverage=0.9)
    covered = np.mean(
        [label in s for label, s in zip(binary_frame["y"], sets, strict=True)]
    )

    assert covered > 0.85
    assert sets == result.bundle.predict_set(rows, coverage=0.9)
    with pytest.raises(errors.ArtifactError, match="regression"):
        reloaded.predict_interval(rows)


def series_frame(rows=600, seed=8):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "when": pd.date_range("2024-01-01", periods=rows, freq="h"),
            "demand": rng.normal(size=rows).cumsum(),
        }
    )
    # `report` describes the previous 3 rows' outcome and becomes known
    # only 3 rows later; using it undelayed would leak the future.
    frame["y"] = (frame["demand"].diff().shift(-1) > 0).astype(int)
    frame["report"] = frame["y"]
    return frame.dropna().reset_index(drop=True)


def test_rolling_validation_runs_end_to_end_with_a_time_aware_report(tmp_path):
    frame = series_frame()
    result = support.make_project(
        frame,
        "y",
        time_budget=300,
        output_dir=tmp_path,
        time_column="when",
        validation="rolling",
        folds=3,
        horizon_rows=60,
        window_rows=200,
        excluded_columns=["report"],
    ).run()

    manifest = json.loads(
        (result.run_dir / "validation_manifest.json").read_text()
    )
    assert manifest["strategy"] == "rolling" and len(manifest["folds"]) == 3
    assert result.evaluation["cv_confidence"]["note"]


def test_a_delayed_column_is_used_only_through_its_lagged_copy(tmp_path):
    frame = series_frame()
    project = support.make_project(
        frame,
        "y",
        time_budget=300,
        output_dir=tmp_path,
        time_column="when",
        feature_delays={"report": 3},
    )
    result = project.run()

    features = result.profile.feature_plan
    assert "report" not in features.numeric
    assert "report" in project.spec.data.excluded_columns
    assert any(
        n.name == "report__available" for n in project.spec.feature_graph.nodes
    )
    assert "report__available" in result.bundle.features or (
        "report__available" in result.bundle.schema.series_columns
    )


def test_datetime_features_train_and_predict_from_raw_timestamps(tmp_path):
    rng = np.random.default_rng(14)
    rows = 700
    frame = pd.DataFrame(
        {
            "seen": pd.date_range("2023-01-01", periods=rows, freq="7h"),
            "x": rng.normal(size=rows),
        }
    )
    frame["y"] = (
        frame["seen"].dt.dayofweek.isin([5, 6]).astype(int)
        + (rng.random(rows) < 0.05)
    ).clip(0, 1)
    result = support.make_project(
        frame, "y", time_budget=300, output_dir=tmp_path
    ).run()

    assert result.evaluation["test_metrics"]["roc_auc"] > 0.9
    as_text = (
        frame.drop(columns="y")
        .head(12)
        .assign(seen=lambda d: d["seen"].astype(str))
    )
    np.testing.assert_array_equal(
        result.predict(as_text),
        result.predict(frame.drop(columns="y").head(12)),
    )


def test_free_text_is_encoded_and_scored_from_raw_strings(tmp_path):
    rng = np.random.default_rng(31)
    good = [
        "great product works well",
        "love it fantastic quality",
        "excellent value recommend",
    ]
    bad = [
        "terrible broke after day",
        "awful waste of money",
        "poor quality disappointed refund",
    ]
    rows = 600
    label = rng.integers(0, 2, rows)
    frame = pd.DataFrame(
        {
            "review": [
                f"{rng.choice(good if y else bad)} number {i} thanks"
                for i, y in enumerate(label)
            ],
            "price": rng.normal(size=rows),
            "y": label,
        }
    )
    result = support.make_project(
        frame, "y", time_budget=300, output_dir=tmp_path
    ).run()

    assert "review" in result.profile.feature_plan.text
    assert result.evaluation["test_metrics"]["roc_auc"] > 0.95
    fresh = pd.DataFrame(
        {
            "review": ["great product recommend", "awful waste refund"],
            "price": [0.0, 0.0],
        }
    )
    assert list(result.predict(fresh)) == [1, 0]


def test_the_fairness_report_names_the_sensitive_columns_and_reaches_the_report(
    binary_frame, tmp_path
):
    result = support.make_project(
        binary_frame,
        "y",
        time_budget=300,
        output_dir=tmp_path,
        sensitive_columns=["c"],
    ).run()

    report = result.diagnostics["fairness"]
    assert report["available"] is True and "c" in report["columns"]
    groups = set(report["columns"]["c"]["groups"])
    assert groups and groups <= {"x", "y", "z"}
    assert (result.run_dir / "diagnostics" / "fairness.json").exists()
