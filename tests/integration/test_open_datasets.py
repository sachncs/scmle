"""End-to-end runs on real open datasets (scikit-learn bundled and OpenML)."""

import pytest
import support

from benchmarks import datasets


def run_dataset(dataset, tmp_path):
    frame = dataset.load()
    project = support.make_project(
        frame,
        dataset.target,
        task=dataset.task,
        metric=dataset.metric,
        time_budget=300,
        output_dir=tmp_path,
    )
    return frame, project.run()


def by_name(name):
    return next(d for d in datasets.ALL if d.name == name)


def test_breast_cancer_beats_baseline_and_generalizes(tmp_path):
    result = run_dataset(by_name("breast_cancer"), tmp_path)[1]
    evaluation = result.evaluation
    assert evaluation["task"] == "binary_classification"
    assert evaluation["test_metrics"]["roc_auc"] > 0.97
    assert evaluation["baseline_cv_score"] == pytest.approx(0.5)
    assert (
        abs(
            evaluation["test_metrics"]["roc_auc"]
            - evaluation["selected_cv_score"]
        )
        < 0.05
    )


def test_wine_multiclass(tmp_path):
    result = run_dataset(by_name("wine"), tmp_path)[1]
    assert result.evaluation["task"] == "multiclass_classification"
    assert result.evaluation["test_metrics"]["accuracy"] > 0.9
    assert result.evaluation["test_metrics"]["log_loss"] < 0.4


def test_digits_many_classes(tmp_path):
    frame, result = run_dataset(by_name("digits"), tmp_path)
    assert result.evaluation["test_metrics"]["accuracy"] > 0.95
    assert result.predict_proba(frame.drop(columns="target")).shape[1] == 10


def test_diabetes_regression_beats_mean_baseline(tmp_path):
    result = run_dataset(by_name("diabetes_progression"), tmp_path)[1]
    evaluation = result.evaluation
    assert evaluation["task"] == "regression"
    assert (
        evaluation["selected_cv_score"] < 0.85 * evaluation["baseline_cv_score"]
    )
    assert evaluation["test_metrics"]["r2"] > 0.3


def test_openml_credit_german_with_mixed_types(tmp_path):
    dataset = by_name("credit-g")
    try:
        dataset.load()
    except Exception as error:  # noqa: BLE001 -- no network or OpenML down: any failure means skip
        pytest.skip(f"OpenML unavailable: {error}")
    frame, result = run_dataset(dataset, tmp_path)
    assert result.evaluation["test_metrics"]["roc_auc"] > 0.7
    assert result.profile.feature_plan.categorical
    assert set(result.predict(frame.drop(columns="class"))) <= {"good", "bad"}


def test_openml_titanic_leakage_is_flagged(tmp_path):
    dataset = by_name("titanic")
    try:
        frame = dataset.load()
    except Exception as error:  # noqa: BLE001 -- no network or OpenML down: any failure means skip
        pytest.skip(f"OpenML unavailable: {error}")
    project = support.make_project(
        frame,
        "survived",
        task="classification",
        time_budget=120,
        output_dir=tmp_path,
    )
    project.plan()
    findings = " ".join(project.profile.risk_findings)
    assert "'boat'" in findings and "'body'" in findings
