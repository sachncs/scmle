"""Tests for the fairness report."""

import dataclasses

import numpy
import pandas
from sklearn import base

from scmle.contracts import data
from scmle.contracts import tasks
from scmle.evaluation import diagnostics
from scmle.evaluation import fairness
from scmle.evaluation import metrics

PLAN = data.FeaturePlan(("score",), ("group",))


class Rule(base.ClassifierMixin, base.BaseEstimator):
    """Approves everyone in group a, only high scorers elsewhere."""

    classes_ = numpy.array([0, 1])

    def fit(self, features, target):
        return self

    def predict(self, features):
        favoured = features["group"].to_numpy() == "a"
        return numpy.where(
            favoured | (features["score"].to_numpy() > 1.0), 1, 0
        )


def make_inputs(biased=True, rows=1200):
    rng = numpy.random.default_rng(0)
    frame = pandas.DataFrame(
        {
            "score": rng.normal(size=rows),
            "group": rng.choice(["a", "b"], rows),
        }
    )
    target = pandas.Series(
        (frame.score + rng.normal(0, 0.5, rows) > 0.5).astype(int)
    )
    estimator = Rule()
    if not biased:
        estimator.predict = lambda f: (f["score"].to_numpy() > 0.5).astype(int)
    return diagnostics.DiagnosticInput(
        estimator,
        frame,
        target,
        tasks.Task.BINARY,
        metrics.CATALOG.resolve("roc_auc", tasks.Task.BINARY),
        PLAN,
        0,
        sensitive_columns=("group",),
    )


def test_a_rule_that_favours_one_group_is_flagged_with_its_disparities():
    report = fairness.Fairness().compute(make_inputs())

    body = report["columns"]["group"]
    assert report["flagged"] == ["group"]
    assert body["demographic_parity_difference"] > 0.4
    assert body["disparate_impact_ratio"] < 0.8
    assert body["groups"]["a"]["selection_rate"] == 1.0
    assert any("selection rates differ" in f for f in body["flags"])


def test_group_blind_decisions_are_not_flagged():
    report = fairness.Fairness().compute(make_inputs(biased=False))

    assert report["flagged"] == []
    assert report["columns"]["group"]["demographic_parity_difference"] < 0.1


def test_without_sensitive_columns_the_report_says_so():
    inputs = dataclasses.replace(make_inputs(), sensitive_columns=())

    assert fairness.Fairness().compute(inputs) == {
        "available": False,
        "reason": "no sensitive columns",
    }


def test_numeric_sensitive_columns_are_banded_and_small_groups_skipped():
    inputs = make_inputs()
    inputs = dataclasses.replace(inputs, sensitive_columns=("score",))

    report = fairness.Fairness().compute(inputs)

    assert len(report["columns"]["score"]["groups"]) == 4
    tiny = dataclasses.replace(
        make_inputs(rows=40), sensitive_columns=("group",)
    )
    body = fairness.Fairness().compute(tiny)["columns"]["group"]
    assert all(
        g["rows"] >= fairness.MIN_GROUP_ROWS for g in body["groups"].values()
    )
    assert body["power_warning"] and body["excluded_groups"]


def test_intersectional_groups_are_reported_and_bounded():
    inputs = dataclasses.replace(
        make_inputs(), sensitive_columns=("group", "score")
    )

    report = fairness.Fairness().compute(inputs)

    body = report["columns"]["intersectional"]
    assert len(body["groups"]) >= 2
    assert all(
        group["rows"] >= fairness.MIN_GROUP_ROWS
        for group in body["groups"].values()
    )
    assert all(" | " in group for group in body["groups"])


def test_regression_reports_bias_per_group():
    rng = numpy.random.default_rng(3)
    rows = 800
    frame = pandas.DataFrame(
        {"x": rng.normal(size=rows), "group": rng.choice(["a", "b"], rows)}
    )
    truth = pandas.Series(frame.x * 2 + (frame.group == "b") * 3)

    class Ignorant(base.RegressorMixin, base.BaseEstimator):
        def fit(self, features, target):
            return self

        def predict(self, features):
            return features["x"].to_numpy() * 2

    inputs = diagnostics.DiagnosticInput(
        Ignorant(),
        frame,
        truth,
        tasks.Task.REGRESSION,
        metrics.CATALOG.resolve("rmse", tasks.Task.REGRESSION),
        data.FeaturePlan(("x",), ("group",)),
        0,
        sensitive_columns=("group",),
    )

    body = fairness.Fairness().compute(inputs)["columns"]["group"]

    assert (
        body["groups"]["b"]["bias"] < -2.5
        and abs(body["groups"]["a"]["bias"]) < 0.3
    )
    assert body["flags"] and body["bias_gap"] > 0.5


def test_regression_intersectional_groups_use_the_same_small_cell_guard():
    rng = numpy.random.default_rng(4)
    rows = 800
    frame = pandas.DataFrame(
        {"x": rng.normal(size=rows), "group": rng.choice(["a", "b"], rows)}
    )
    truth = pandas.Series(frame.x * 2 + (frame.group == "b") * 3)

    class Ignorant(base.RegressorMixin, base.BaseEstimator):
        def fit(self, features, target):
            return self

        def predict(self, features):
            return features["x"].to_numpy() * 2

    inputs = diagnostics.DiagnosticInput(
        Ignorant(),
        frame,
        truth,
        tasks.Task.REGRESSION,
        metrics.CATALOG.resolve("rmse", tasks.Task.REGRESSION),
        data.FeaturePlan(("x",), ("group",)),
        0,
        sensitive_columns=("group", "x"),
    )

    body = fairness.Fairness().compute(inputs)["columns"]["intersectional"]

    assert body["groups"]
    assert all(
        group["rows"] >= fairness.MIN_GROUP_ROWS
        for group in body["groups"].values()
    )
