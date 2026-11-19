"""Tests for post-selection diagnostics."""

import dataclasses

import numpy
import pandas
import pytest

from scmle.contracts import data
from scmle.contracts import tasks
from scmle.evaluation import bootstrap
from scmle.evaluation import diagnostics
from scmle.evaluation import metrics
from scmle.explainability import importance
from scmle.models import registry
import scmle.pipeline.suite
from scmle.preparation import preprocessing

REGISTRY = registry.Catalog.stock()
PLAN = data.Blueprint(("signal", "noise"), ("group",))


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
    model = preprocessing.instantiate(PLAN, adapter, task, 0).fit(
        frame.iloc[:split], target.iloc[:split]
    )
    metric = metrics.CATALOG.resolve(None, task)
    return diagnostics.DiagnosticInput(
        model, frame.iloc[split:], target.iloc[split:], task, metric, PLAN, 0
    )


def test_importance_ranks_the_signal_feature_first():
    report = importance.Importance().compute(make_inputs(tasks.Task.BINARY))
    ranked = [row["feature"] for row in report["features"]]
    assert ranked[0] == "signal" and ranked[-1] in {"noise", "group"}
    assert report["features"][0]["importance"] > 0.1


def test_classification_report_matches_sklearn():
    inputs = make_inputs(tasks.Task.MULTICLASS)
    report = diagnostics.Contingency().compute(inputs)
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
    rows = diagnostics.Sweep().compute(make_inputs(tasks.Task.BINARY))[
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
    report = diagnostics.Errors().compute(make_inputs(tasks.Task.BINARY))

    assert report["row_count"] == 300
    assert report["misclassified_count"] > 0
    assert len(report["rows"]) <= diagnostics.MAX_ERROR_ROWS
    assert report["rows"][0]["misclassified"] is True
    assert "signal" not in report["rows"][0]


def test_error_analysis_ranks_regression_residuals():
    """Regression errors carry bounded residual evidence."""
    report = diagnostics.Errors().compute(make_inputs(tasks.Task.REGRESSION))

    assert report["task"] == "regression"
    assert report["mean_absolute_error"] > 0
    assert (
        report["rows"][0]["absolute_error"]
        >= report["rows"][-1]["absolute_error"]
    )


def test_suite_selects_by_task_and_isolates_failures():
    suite = scmle.pipeline.suite.lineup()
    binary = scmle.pipeline.suite.run_suite(
        make_inputs(tasks.Task.BINARY), suite
    )
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
    regression = scmle.pipeline.suite.run_suite(
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

    reports = scmle.pipeline.suite.run_suite(
        make_inputs(tasks.Task.BINARY),
        (Broken(), diagnostics.Calibration()),
    )
    assert reports["broken"]["error"] == "RuntimeError: boom"
    # ERR-19: the report keeps the traceback, not only the one-line text.
    assert reports["broken"]["failure"]["kind"] == "RuntimeError"
    assert "boom" in reports["broken"]["failure"]["trace"]
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
    model = preprocessing.instantiate(PLAN, adapter, tasks.Task.BINARY, 0).fit(
        frame.iloc[:split], target.iloc[:split]
    )
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
    # Enforces docs/ARCHITECTURE.md:128.
    report = bootstrap.Uncertainty().compute(make_inputs(tasks.Task.BINARY))
    low, high = report["ci_95"]
    assert low <= report["estimate"] <= high and high - low < 0.2
    assert report["resampling_unit"] == "rows" and report["resamples"] > 150
    assert report["independent_units"] == report["evaluation_rows"]


def test_group_bootstrap_is_wider_than_a_naive_row_bootstrap():
    # Enforces docs/ARCHITECTURE.md:128.
    clustered = bootstrap.Uncertainty().compute(make_clustered_inputs())
    naive = bootstrap.Uncertainty().compute(
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
    # Enforces docs/ARCHITECTURE.md:128.
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
    report = bootstrap.Uncertainty().compute(temporal)
    assert report["resampling_unit"] == "blocks"


def test_block_resamples_are_contiguous_runs():
    rng = numpy.random.default_rng(0)
    rows = bootstrap.redraw(100, None, True, rng)
    assert len(rows) == 100
    steps = numpy.diff(rows)
    assert (steps == 1).mean() > 0.7  # mostly consecutive within blocks


def test_group_resamples_keep_whole_groups_together():
    rng = numpy.random.default_rng(1)
    groups = pandas.Series(numpy.repeat(numpy.arange(10), 5))
    rows = bootstrap.redraw(50, groups, False, rng)
    counts = groups.iloc[rows].value_counts()
    assert set(counts) <= {5, 10, 15, 20, 25}  # multiples of the group size


def test_regression_uncertainty_works():
    report = bootstrap.Uncertainty().compute(make_inputs(tasks.Task.REGRESSION))
    assert (
        report["metric"] == "rmse" and report["ci_95"][0] <= report["ci_95"][1]
    )


def test_cached_predictions_match_the_model():
    inputs = make_inputs(tasks.Task.BINARY)
    replay = bootstrap.Replay(inputs.estimator, inputs.features)
    subset = inputs.features.iloc[[3, 1, 2]]
    numpy.testing.assert_array_equal(
        replay.predict(subset), inputs.estimator.predict(subset)
    )
    numpy.testing.assert_allclose(
        replay.predict_proba(subset), inputs.estimator.predict_proba(subset)
    )
    bare = bootstrap.Cache(
        make_inputs(tasks.Task.REGRESSION).estimator,
        make_inputs(tasks.Task.REGRESSION).features,
    )
    with pytest.raises(Exception, match="no probabilities"):
        bare.predict_proba(inputs.features)


def test_block_resamples_can_start_at_the_last_possible_position():
    rng = numpy.random.default_rng(0)
    firsts = {
        int(bootstrap.redraw(10, None, True, rng)[0]) for draw in range(300)
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
    report = bootstrap.Uncertainty().compute(inputs)
    assert report["ci_95"] is None and report["resamples"] == 0
    assert report["estimate"] == 0.5


def test_small_evaluation_sets_get_more_resamples_than_large_ones():
    diagnostic = bootstrap.Uncertainty()

    assert diagnostic.replicates(200) == diagnostic.resamples
    assert diagnostic.replicates(100_000) == diagnostic.minimum_resamples
    assert diagnostic.replicates(10**9) == diagnostic.minimum_resamples
    assert diagnostic.replicates(4000) == 1000


class Clock:
    """A monotonic clock the test moves by hand."""

    def __init__(self, now=1000.0):
        self.now = now

    def __call__(self):
        return self.now


def test_run_suite_ends_by_the_run_deadline_not_its_own_allowance():
    seen = []

    class Probe(diagnostics.Diagnostic):
        name = "probe"

        def compute(self, inputs):
            seen.append(inputs.deadline)
            return {}

    finish = diagnostics.clock() + 5.0
    scmle.pipeline.suite.run_suite(
        make_inputs(tasks.Task.BINARY), (Probe(),), 100.0, deadline=finish
    )

    assert seen and seen[0] <= finish


def test_run_suite_records_every_diagnostic_skipped_after_the_deadline():
    reports = scmle.pipeline.suite.run_suite(
        make_inputs(tasks.Task.BINARY),
        scmle.pipeline.suite.lineup(),
        deadline=diagnostics.clock() - 1.0,
    )

    assert reports and all(
        body["skipped"] and body["available"] is False and body["reason"]
        for body in reports.values()
    )


def test_run_suite_skips_what_cannot_fit_and_runs_what_can():
    class Slow(diagnostics.Diagnostic):
        name = "slow"
        fixed = 1000.0

        def compute(self, inputs):
            raise AssertionError("must not start")

    reports = scmle.pipeline.suite.run_suite(
        make_inputs(tasks.Task.BINARY),
        (Slow(), diagnostics.Calibration()),
        deadline=diagnostics.clock() + 30.0,
    )

    assert reports["slow"]["skipped"]
    assert "needs" in reports["slow"]["reason"]
    assert "brier_score" in reports["calibration"]


def test_uncertainty_sizes_its_resamples_to_the_time_left(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(diagnostics, "clock", clock)
    base = make_inputs(tasks.Task.BINARY)
    diagnostic = bootstrap.Uncertainty()
    each = diagnostic.cost(base)
    inputs = dataclasses.replace(base, deadline=clock.now + 500.5 * each)

    report = diagnostic.compute(inputs)

    assert report["planned"] == 500
    assert report["resamples"] + report["dropped"] == 500
    assert report["ci_95"] is not None


def test_uncertainty_skips_rather_than_report_too_few_resamples(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(diagnostics, "clock", clock)
    base = make_inputs(tasks.Task.BINARY)
    diagnostic = bootstrap.Uncertainty()
    room = (diagnostic.minimum_resamples - 1) * diagnostic.cost(base)
    inputs = dataclasses.replace(base, deadline=clock.now + room)

    report = diagnostic.compute(inputs)

    assert report["available"] is False and report["skipped"]
    assert report["ci_95"] is None and "minimum" in report["reason"]
    assert report["estimate"] > 0.5


def test_uncertainty_stops_at_the_deadline_mid_loop(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(diagnostics, "clock", clock)
    base = make_inputs(tasks.Task.BINARY)
    calls = []
    score = base.metric.score

    class Ticking:
        name = "ticking"  # not studentized: every resample calls `score`

        def score(self, *args):
            calls.append(1)
            if len(calls) > 50:
                clock.now += 1e6
            return score(*args)

    inputs = dataclasses.replace(
        base, metric=Ticking(), deadline=clock.now + 1e5
    )

    report = bootstrap.Uncertainty().compute(inputs)

    assert report["skipped"] and report["ci_95"] is None
    assert report["resamples"] < bootstrap.Uncertainty.minimum_resamples


def test_uncertainty_cost_grows_with_rows_and_classes():
    diagnostic = bootstrap.Uncertainty()
    small = make_inputs(tasks.Task.BINARY, rows=200)
    large = make_inputs(tasks.Task.BINARY, rows=4000)
    many = make_inputs(tasks.Task.MULTICLASS, rows=4000)
    rows = make_inputs(tasks.Task.REGRESSION, rows=4000)

    assert diagnostic.cost(small) < diagnostic.cost(large)
    assert diagnostic.cost(large) < diagnostic.cost(many)
    assert diagnostic.cost(rows) < diagnostic.cost(large)


def test_calibration_is_skipped_once_its_deadline_has_passed():
    inputs = dataclasses.replace(
        make_inputs(tasks.Task.BINARY), deadline=diagnostics.clock() - 1.0
    )

    for diagnostic in (diagnostics.Calibration(), diagnostics.Reliability()):
        report = diagnostic.compute(inputs)
        assert report["skipped"] and report["reason"]
