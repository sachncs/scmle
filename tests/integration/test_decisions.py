"""Threshold optimization and probability calibration, end to end."""

import numpy
import support

import scmle.artifacts.bundle
from scmle.contracts import tasks

NO_SEARCH = tasks.Search(top_models=0)


def run(frame, tmp_path, decision=None, strategies=None, **kwargs):
    project = support.make_project(
        frame,
        "y",
        time_budget=600,
        output_dir=tmp_path,
        search=NO_SEARCH,
        strategies=strategies or tasks.Strategies(top_models=0),
        decision=decision,
        **kwargs,
    )
    return project.run()


def test_threshold_is_optimized_on_validation_and_frozen_into_the_bundle(
    imbalanced_frame, tmp_path
):
    decision = tasks.Threshold(threshold_objective="f1")
    result = run(imbalanced_frame, tmp_path, decision, metric="roc_auc")
    chosen = result.evaluation["decision"]
    assert chosen["objective"] == "f1" and 0.01 <= chosen["threshold"] <= 0.99
    assert result.bundle.info.decision_threshold == chosen["threshold"]
    features = imbalanced_frame.drop(columns="y")
    probability = result.predict_proba(features)[:, 1]
    expected = (probability >= chosen["threshold"]).astype(int)
    numpy.testing.assert_array_equal(result.predict(features), expected)
    assert result.evaluation["test_set_opened"] is True


def test_threshold_choice_never_uses_the_test_rows(imbalanced_frame, tmp_path):
    decision = tasks.Threshold(threshold_objective="balanced_accuracy")
    result = run(imbalanced_frame, tmp_path, decision)
    chosen = result.evaluation["decision"]
    assert chosen["validation_rows"] <= result.evaluation["n_dev"]
    reloaded = scmle.artifacts.bundle.Bundle.load(
        result.save(tmp_path / "bundle"), trusted=True
    )
    assert reloaded.info.decision_threshold == chosen["threshold"]


def test_cost_objective_responds_to_costs(imbalanced_frame, tmp_path):
    cheap = tasks.Threshold(
        threshold_objective="cost",
        false_positive_cost=10.0,
        false_negative_cost=1.0,
    )
    costly = tasks.Threshold(
        threshold_objective="cost",
        false_positive_cost=1.0,
        false_negative_cost=10.0,
    )
    high = run(imbalanced_frame, tmp_path / "a", cheap).evaluation["decision"][
        "threshold"
    ]
    low = run(imbalanced_frame, tmp_path / "b", costly).evaluation["decision"][
        "threshold"
    ]
    assert low < high


def test_disabled_and_non_binary_leave_decisions_alone(
    multiclass_frame, binary_frame, tmp_path
):
    assert run(binary_frame, tmp_path / "a").evaluation["decision"] is None
    decision = tasks.Threshold(threshold_objective="f1")
    multi = run(
        multiclass_frame, tmp_path / "b", decision, task="classification"
    )
    assert multi.evaluation["decision"] is None
    assert multi.bundle.info.decision_threshold is None


def test_string_labels_with_thresholds(binary_frame, tmp_path):
    labelled = binary_frame.assign(
        y=binary_frame.y.map({0: "stay", 1: "churn"})
    )
    decision = tasks.Threshold(threshold_objective="f1")
    result = run(labelled, tmp_path, decision, metric="pr_auc")
    assert set(result.predict(labelled.drop(columns="y"))) <= {"stay", "churn"}
    assert result.evaluation["decision"]["threshold"] > 0


def test_calibration_is_tried_for_uncalibrated_models_and_judged(
    imbalanced_frame, tmp_path
):
    strategies = tasks.Strategies(names=["calibrated"], top_models=2)
    result = run(
        imbalanced_frame, tmp_path, strategies=strategies, metric="log_loss"
    )
    calibrated = [t for t in result.trials if t.strategy == "calibrated"]
    assert calibrated and all(
        t.model_name != "logistic_regression" for t in calibrated
    )
    log = (result.run_dir / "diagnostics" / "decision_log.json").read_text()
    assert "calibrated@" in log


def test_calibration_improves_log_loss_for_forests(imbalanced_frame, tmp_path):
    strategies = tasks.Strategies(names=["calibrated"], top_models=3)
    result = run(
        imbalanced_frame, tmp_path, strategies=strategies, metric="log_loss"
    )
    pairs = {
        (t.model_name, t.strategy): t.score_mean
        for t in result.trials
        if t.status == "ok" and not t.params
    }
    gains = [
        pairs[(model, "default")] - pairs[(model, "calibrated")]
        for model, strategy in pairs
        if strategy == "calibrated"
        and model in {"random_forest", "extra_trees"}
    ]
    if gains:
        assert max(gains) > 0  # lower log loss after calibration
