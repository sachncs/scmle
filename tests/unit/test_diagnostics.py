"""Tests for post-selection diagnostics."""

import numpy as np
import pandas as pd

from mml.contracts import data as data_contracts
from mml.contracts import tasks
from mml.data import preprocessing
from mml.evaluation import diagnostics
from mml.evaluation import metrics
from mml.models import registry as registry_module

REGISTRY = registry_module.ModelRegistry.with_builtins()
PLAN = data_contracts.FeaturePlan(("signal", "noise"), ("group",))


def make_inputs(task, seed=0, rows=600):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "signal": rng.normal(size=rows),
            "noise": rng.normal(size=rows),
            "group": rng.choice(["a", "b", "c"], rows),
        }
    )
    if task is tasks.TaskType.REGRESSION:
        target = pd.Series(
            3 * frame.signal + (frame.group == "a") + rng.normal(0, 0.2, rows)
        )
        adapter = REGISTRY.get("ridge")
    else:
        labels = (frame.signal + rng.normal(0, 0.5, rows) > 0).astype(int)
        if task is tasks.TaskType.MULTICLASS:
            labels = labels + (frame.signal > 1).astype(int)
        target = pd.Series(labels)
        adapter = REGISTRY.get("logistic_regression")
    split = rows // 2
    model = preprocessing.build_pipeline(PLAN, adapter, task, 0).fit(
        frame.iloc[:split], target.iloc[:split]
    )
    metric = metrics.CATALOG.resolve(None, task)
    return diagnostics.DiagnosticInput(
        model, frame.iloc[split:], target.iloc[split:], task, metric, PLAN, 0
    )


def test_importance_ranks_the_signal_feature_first():
    report = diagnostics.PermutationImportance().compute(
        make_inputs(tasks.TaskType.BINARY)
    )
    ranked = [row["feature"] for row in report["features"]]
    assert ranked[0] == "signal" and ranked[-1] in {"noise", "group"}
    assert report["features"][0]["importance"] > 0.1


def test_classification_report_matches_sklearn():
    inputs = make_inputs(tasks.TaskType.MULTICLASS)
    report = diagnostics.ClassificationReport().compute(inputs)
    matrix = np.array(report["confusion_matrix"])
    assert matrix.sum() == len(inputs.target)
    assert sum(c["support"] for c in report["per_class"].values()) == len(
        inputs.target
    )
    assert report["labels"] == ["0", "1", "2"]


def test_calibration_report_is_consistent():
    report = diagnostics.CalibrationReport().compute(
        make_inputs(tasks.TaskType.BINARY)
    )
    assert 0.0 <= report["brier_score"] <= 0.5
    assert 0.0 <= report["expected_calibration_error"] < 0.2
    assert sum(b["count"] for b in report["curve"]) == 300


def test_threshold_report_trades_precision_for_recall():
    rows = diagnostics.ThresholdReport().compute(
        make_inputs(tasks.TaskType.BINARY)
    )["thresholds"]
    assert rows[0]["recall"] >= rows[-1]["recall"]
    assert rows[0]["precision"] <= rows[-1]["precision"]


def test_residual_report_segments_and_bias():
    report = diagnostics.ResidualReport().compute(
        make_inputs(tasks.TaskType.REGRESSION)
    )
    assert abs(report["bias"]) < 0.3 and report["mae"] > 0
    assert set(report["segments"]["group"]) == {"a", "b", "c"}
    assert (
        report["abs_error_quantiles"]["0.99"]
        >= report["abs_error_quantiles"]["0.5"]
    )


def test_suite_selects_by_task_and_isolates_failures():
    binary = diagnostics.run_suite(make_inputs(tasks.TaskType.BINARY))
    assert set(binary) == {
        "classification_report",
        "calibration",
        "threshold_analysis",
        "permutation_importance",
    }
    regression = diagnostics.run_suite(make_inputs(tasks.TaskType.REGRESSION))
    assert set(regression) == {"residuals", "permutation_importance"}

    class Broken(diagnostics.Diagnostic):
        name = "broken"

        def compute(self, inputs):
            raise RuntimeError("boom")

    reports = diagnostics.run_suite(
        make_inputs(tasks.TaskType.BINARY),
        (Broken(), diagnostics.CalibrationReport()),
    )
    assert reports["broken"] == {"error": "RuntimeError: boom"}
    assert "brier_score" in reports["calibration"]


def test_positive_label_comes_from_the_fitted_classes():
    inputs = make_inputs(tasks.TaskType.BINARY)
    assert inputs.positive_label == 1
    assert make_inputs(tasks.TaskType.REGRESSION).positive_label is None
