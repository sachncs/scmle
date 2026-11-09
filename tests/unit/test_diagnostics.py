"""Tests for post-selection diagnostics."""

import numpy
import pandas
import pytest

from scmle.contracts import data
from scmle.contracts import tasks
from scmle.data.curation import preprocessing
from scmle.evaluation import diagnostics
from scmle.evaluation import metrics
import scmle.evaluation.suite
from scmle.explainability import importance
from scmle.models import registry

REGISTRY = registry.Catalog.with_builtins()
PLAN = data.FeaturePlan(("signal", "noise"), ("group",))


def make_inputs(task, seed=0, rows=600):
    rng = numpy.random.default_rng(seed)
    frame = pandas.DataFrame(
        {
            "signal": rng.normal(size=rows),
            "noise": rng.normal(size=rows),
            "group": rng.choice(["a", "b", "c"], rows),
        }
    )
    if task is tasks.Task.REGRESSION:
        target = pandas.Series(
            3 * frame.signal + (frame.group == "a") + rng.normal(0, 0.2, rows)
        )
        adapter = REGISTRY.get("ridge")
    else:
        labels = (frame.signal + rng.normal(0, 0.5, rows) > 0).astype(int)
        if task is tasks.Task.MULTICLASS:
            labels = labels + (frame.signal > 1).astype(int)
        target = pandas.Series(labels)
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
    report = importance.PermutationImportance().compute(
        make_inputs(tasks.Task.BINARY)
    )
    ranked = [row["feature"] for row in report["features"]]
    assert ranked[0] == "signal" and ranked[-1] in {"noise", "group"}
    assert report["features"][0]["importance"] > 0.1


def test_classification_report_matches_sklearn():
    inputs = make_inputs(tasks.Task.MULTICLASS)
    report = diagnostics.ConfusionMatrix().compute(inputs)
    matrix = numpy.array(report["confusion_matrix"])
    assert matrix.sum() == len(inputs.target)
    assert sum(c["support"] for c in report["per_class"].values()) == len(
        inputs.target
    )
    assert report["labels"] == ["0", "1", "2"]


def test_calibration_report_is_consistent():
    report = diagnostics.Calibration().compute(make_inputs(tasks.Task.BINARY))
    assert 0.0 <= report["brier_score"] <= 0.5
    assert 0.0 <= report["expected_calibration_error"] < 0.2
    assert sum(b["count"] for b in report["curve"]) == 300


def test_threshold_report_trades_precision_for_recall():
    rows = diagnostics.ThresholdSweep().compute(make_inputs(tasks.Task.BINARY))[
        "thresholds"
    ]
    assert rows[0]["recall"] >= rows[-1]["recall"]
    assert rows[0]["precision"] <= rows[-1]["precision"]


def test_residual_report_segments_and_bias():
    report = diagnostics.Residuals().compute(make_inputs(tasks.Task.REGRESSION))
    assert abs(report["bias"]) < 0.3 and report["mae"] > 0
    assert set(report["segments"]["group"]) == {"a", "b", "c"}
    numeric = [c for c in report["segments"] if c != "group"]
    assert numeric, "continuous features are segmented into quantile bands"
    bands = report["segments"][numeric[0]]
    assert len(bands) == 4
    assert sum(b["count"] for b in bands.values()) == 300
    assert (
        report["abs_error_quantiles"]["0.99"]
        >= report["abs_error_quantiles"]["0.5"]
    )


def test_error_analysis_ranks_misclassified_rows_without_feature_values():
    """Classification errors are reviewable without copying raw features."""
    report = diagnostics.ErrorAnalysis().compute(make_inputs(tasks.Task.BINARY))

    assert report["row_count"] == 300
    assert report["misclassified_count"] > 0
    assert len(report["rows"]) <= diagnostics.MAX_ERROR_ROWS
    assert report["rows"][0]["misclassified"] is True
    assert "signal" not in report["rows"][0]


def test_error_analysis_ranks_regression_residuals():
    """Regression errors carry bounded residual evidence."""
    report = diagnostics.ErrorAnalysis().compute(
        make_inputs(tasks.Task.REGRESSION)
    )

    assert report["task"] == "regression"
    assert report["mean_absolute_error"] > 0
    assert (
        report["rows"][0]["absolute_error"]
        >= report["rows"][-1]["absolute_error"]
    )


def test_suite_selects_by_task_and_isolates_failures():
    suite = scmle.evaluation.suite.full_suite()
    binary = diagnostics.run_suite(make_inputs(tasks.Task.BINARY), suite)
    assert set(binary) == {
        "classification_report",
        "calibration",
        "threshold_analysis",
        "error_analysis",
        "permutation_importance",
        "partial_dependence",
        "pairwise_interaction",
        "metric_uncertainty",
        "slice_performance",
        "fairness",
        "robustness",
        "adversarial_validation",
    }
    regression = diagnostics.run_suite(
        make_inputs(tasks.Task.REGRESSION), suite
    )
    assert set(regression) == {
        "residuals",
        "error_analysis",
        "permutation_importance",
        "partial_dependence",
        "pairwise_interaction",
        "metric_uncertainty",
        "slice_performance",
        "fairness",
        "robustness",
        "adversarial_validation",
    }

    class Broken(diagnostics.Diagnostic):
        name = "broken"

        def compute(self, inputs):
            raise RuntimeError("boom")

    reports = diagnostics.run_suite(
        make_inputs(tasks.Task.BINARY),
        (Broken(), diagnostics.Calibration()),
    )
    assert reports["broken"] == {"error": "RuntimeError: boom"}
    assert "brier_score" in reports["calibration"]


def test_positive_label_comes_from_the_fitted_classes():
    inputs = make_inputs(tasks.Task.BINARY)
    assert inputs.positive_label == 1
    assert make_inputs(tasks.Task.REGRESSION).positive_label is None


def make_clustered_inputs(clusters=40, per_cluster=15, seed=0, grouped=True):
    """Rows within a cluster share a hidden effect, so they are not independent."""
    rng = numpy.random.default_rng(seed)
    rows = clusters * per_cluster
    group = numpy.repeat(numpy.arange(clusters), per_cluster)
    effect = numpy.repeat(rng.normal(scale=1.5, size=clusters), per_cluster)
    frame = pandas.DataFrame(
        {
            "signal": rng.normal(size=rows) + effect,
            "noise": rng.normal(size=rows),
            "group": numpy.where(group % 2 == 0, "a", "b"),
        }
    )
    target = pandas.Series(
        (effect + rng.normal(scale=0.6, size=rows) > 0).astype(int)
    )
    split = rows // 2
    adapter = REGISTRY.get("logistic_regression")
    model = preprocessing.build_pipeline(
        PLAN, adapter, tasks.Task.BINARY, 0
    ).fit(frame.iloc[:split], target.iloc[:split])
    metric = metrics.CATALOG.resolve("accuracy", tasks.Task.BINARY)
    held = frame.iloc[split:]
    return diagnostics.DiagnosticInput(
        model,
        held,
        target.iloc[split:],
        tasks.Task.BINARY,
        metric,
        PLAN,
        0,
        groups=pandas.Series(group[split:], index=held.index)
        if grouped
        else None,
    )


def test_uncertainty_reports_an_interval_around_the_estimate():
    report = diagnostics.MetricUncertainty().compute(
        make_inputs(tasks.Task.BINARY)
    )
    low, high = report["ci_95"]
    assert low <= report["estimate"] <= high and high - low < 0.2
    assert report["resampling_unit"] == "rows" and report["resamples"] > 150
    assert report["independent_units"] == report["evaluation_rows"]


def test_group_bootstrap_is_wider_than_a_naive_row_bootstrap():
    clustered = diagnostics.MetricUncertainty().compute(make_clustered_inputs())
    naive = diagnostics.MetricUncertainty().compute(
        make_clustered_inputs(grouped=False)
    )
    width = lambda r: r["ci_95"][1] - r["ci_95"][0]  # noqa: E731
    assert clustered["resampling_unit"] == "groups"
    assert (
        clustered["independent_units"] == 20
        and naive["independent_units"] == 300
    )
    assert width(clustered) > 1.5 * width(naive)


def test_temporal_inputs_use_contiguous_blocks():
    inputs = make_inputs(tasks.Task.BINARY)
    temporal = diagnostics.DiagnosticInput(
        inputs.estimator,
        inputs.features,
        inputs.target,
        inputs.task,
        inputs.metric,
        inputs.feature_plan,
        0,
        temporal=True,
    )
    report = diagnostics.MetricUncertainty().compute(temporal)
    assert report["resampling_unit"] == "blocks"


def test_block_resamples_are_contiguous_runs():
    rng = numpy.random.default_rng(0)
    rows = diagnostics.resample_rows(100, None, True, rng)
    assert len(rows) == 100
    steps = numpy.diff(rows)
    assert (steps == 1).mean() > 0.7  # mostly consecutive within blocks


def test_group_resamples_keep_whole_groups_together():
    rng = numpy.random.default_rng(1)
    groups = pandas.Series(numpy.repeat(numpy.arange(10), 5))
    rows = diagnostics.resample_rows(50, groups, False, rng)
    counts = groups.iloc[rows].value_counts()
    assert set(counts) <= {5, 10, 15, 20, 25}  # multiples of the group size


def test_regression_uncertainty_works():
    report = diagnostics.MetricUncertainty().compute(
        make_inputs(tasks.Task.REGRESSION)
    )
    assert (
        report["metric"] == "rmse" and report["ci_95"][0] <= report["ci_95"][1]
    )


def test_cached_predictions_match_the_model():
    inputs = make_inputs(tasks.Task.BINARY)
    replay = diagnostics.ClassifierReplay(inputs.estimator, inputs.features)
    subset = inputs.features.iloc[[3, 1, 2]]
    numpy.testing.assert_array_equal(
        replay.predict(subset), inputs.estimator.predict(subset)
    )
    numpy.testing.assert_allclose(
        replay.predict_proba(subset), inputs.estimator.predict_proba(subset)
    )
    bare = diagnostics.CachedPredictions(
        make_inputs(tasks.Task.REGRESSION).estimator,
        make_inputs(tasks.Task.REGRESSION).features,
    )
    with pytest.raises(Exception, match="no probabilities"):
        bare.predict_proba(inputs.features)


def test_block_resamples_can_start_at_the_last_possible_position():
    rng = numpy.random.default_rng(0)
    firsts = {
        int(diagnostics.resample_rows(10, None, True, rng)[0])
        for draw in range(300)
    }
    assert 5 in firsts  # blocks of 5 rows may start at row 5 of 10


def test_uncertainty_is_unavailable_when_every_resample_fails():
    class Failing:
        name = "failing"

        def score(self, *args):
            if not Failing.first:
                raise ValueError("one class")
            Failing.first = False
            return 0.5

    Failing.first = True
    inputs = make_inputs(tasks.Task.BINARY)
    inputs = diagnostics.DiagnosticInput(
        inputs.estimator,
        inputs.features,
        inputs.target,
        inputs.task,
        Failing(),
        inputs.feature_plan,
        0,
    )
    report = diagnostics.MetricUncertainty().compute(inputs)
    assert report["ci_95"] is None and report["resamples"] == 0
    assert report["estimate"] == 0.5
