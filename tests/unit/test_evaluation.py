"""Tests for the metric catalog and selection policy."""

import numpy as np
import pandas as pd
import pytest
from sklearn import dummy

from mml.contracts import errors
from mml.contracts import experiments
from mml.contracts import tasks
from mml.evaluation import metrics
from mml.evaluation import selection
from mml.models import registry as registry_module

CATALOG = metrics.CATALOG
TrialResult = experiments.TrialResult
Status = experiments.TrialStatus


def test_defaults_and_multiclass_alias():
    assert CATALOG.resolve(None, tasks.TaskType.BINARY).name == "roc_auc"
    assert CATALOG.resolve(None, tasks.TaskType.REGRESSION).name == "rmse"
    assert CATALOG.resolve("roc_auc", tasks.TaskType.MULTICLASS).name == (
        "roc_auc_ovr"
    )


def test_unknown_and_incompatible_metrics():
    with pytest.raises(errors.ConfigError, match="unknown"):
        CATALOG.resolve("nope", tasks.TaskType.BINARY)
    with pytest.raises(errors.ConfigError, match="not defined"):
        CATALOG.resolve("pr_auc", tasks.TaskType.MULTICLASS)


def test_orientation_natural_units():
    assert CATALOG.metrics["rmse"].to_natural(-2.0) == 2.0
    assert CATALOG.metrics["roc_auc"].to_natural(0.8) == 0.8


def test_evaluate_classification_and_regression():
    features = pd.DataFrame({"a": np.arange(40.0)})
    labels = pd.Series([0, 1] * 20)
    classifier = dummy.DummyClassifier(strategy="prior").fit(features, labels)
    scores = CATALOG.evaluate(
        classifier, features, labels, tasks.TaskType.BINARY
    )
    assert scores["roc_auc"] == 0.5
    assert scores["log_loss"] == pytest.approx(0.6931, abs=1e-3)
    values = pd.Series(np.arange(40.0))
    regressor = dummy.DummyRegressor().fit(features, values)
    scores = CATALOG.evaluate(
        regressor, features, values, tasks.TaskType.REGRESSION
    )
    assert scores["rmse"] > 0 and scores["mae"] > 0
    assert scores["r2"] == pytest.approx(0.0, abs=1e-9)


def test_evaluate_reports_nan_for_undefined_metric():
    features = pd.DataFrame({"a": np.arange(40.0)})
    labels = pd.Series([0, 1] * 20)
    classifier = dummy.DummyClassifier(strategy="prior").fit(features, labels)
    single_class = pd.Series([1] * 40)
    scores = CATALOG.evaluate(
        classifier, features, single_class, tasks.TaskType.BINARY
    )
    assert np.isnan(scores["roc_auc"])


def make_trial(name, score, status=Status.OK):
    return TrialResult(f"t_{name}", name, status, "m", score_mean=score)


@pytest.fixture(scope="module")
def registry():
    return registry_module.ModelRegistry.with_builtins()


def test_best_mean_respects_orientation(registry):
    policy = selection.BestMeanPolicy()
    trials = [
        make_trial("decision_tree", 0.7),
        make_trial("random_forest", 0.9),
    ]
    higher = CATALOG.metrics["roc_auc"]
    lower = CATALOG.metrics["rmse"]
    assert policy.select(trials, higher, registry).model_name == "random_forest"
    assert policy.select(trials, lower, registry).model_name == "decision_tree"


def test_ties_prefer_cheaper_model_and_baseline_excluded(registry):
    policy = selection.BestMeanPolicy()
    trials = [
        make_trial("random_forest", 0.8),
        make_trial("decision_tree", 0.8),
        make_trial("dummy", 0.99),
    ]
    winner = policy.select(trials, CATALOG.metrics["roc_auc"], registry)
    assert winner.model_name == "decision_tree"


def test_baseline_selected_only_when_alone(registry):
    policy = selection.BestMeanPolicy()
    trials = [make_trial("dummy", 0.5), make_trial("ridge", 0.0, Status.FAILED)]
    winner = policy.select(trials, CATALOG.metrics["roc_auc"], registry)
    assert winner.model_name == "dummy"


def test_all_failed_raises(registry):
    policy = selection.BestMeanPolicy()
    trials = [make_trial("ridge", 0.0, Status.FAILED)]
    with pytest.raises(errors.MagicMLError, match="all trials failed"):
        policy.select(trials, CATALOG.metrics["rmse"], registry)


def string_problem():
    rng = np.random.default_rng(0)
    features = pd.DataFrame({"a": rng.normal(size=200)})
    labels = pd.Series(
        np.where(features.a + rng.normal(size=200) > 0, "true", "false")
    )
    return features, labels


@pytest.mark.parametrize("name", ["pr_auc", "f1", "roc_auc"])
def test_binary_metrics_work_with_string_labels(name):
    from sklearn import linear_model

    features, labels = string_problem()
    model = linear_model.LogisticRegression().fit(features, labels)
    metric = CATALOG.metrics[name]
    positive = metrics.positive_label(labels, tasks.TaskType.BINARY)
    assert positive == "true"
    score = metric.score(model, features, labels, positive)
    assert 0.6 < score <= 1.0


def test_pr_auc_is_oriented_to_the_real_positive_class():
    from sklearn import linear_model

    features, labels = string_problem()
    model = linear_model.LogisticRegression().fit(features, labels)
    metric = CATALOG.metrics["pr_auc"]
    as_true = metric.score(model, features, labels, "true")
    as_false = metric.score(model, features, labels, "false")
    assert as_true != pytest.approx(as_false)


def test_evaluate_reports_f1_for_string_labels_instead_of_nan():
    from sklearn import linear_model

    features, labels = string_problem()
    model = linear_model.LogisticRegression().fit(features, labels)
    scores = CATALOG.evaluate(model, features, labels, tasks.TaskType.BINARY)
    assert not any(np.isnan(value) for value in scores.values()), scores


def test_positive_label_is_none_outside_binary_tasks():
    labels = pd.Series([0, 1, 2])
    assert metrics.positive_label(labels, tasks.TaskType.MULTICLASS) is None
    assert (
        metrics.positive_label(pd.Series([3, 7, 3]), tasks.TaskType.BINARY) == 7
    )
