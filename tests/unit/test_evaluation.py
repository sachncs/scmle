"""Tests for the metric catalog and selection policy."""

import numpy
import pandas
import pytest
from sklearn import dummy

from scmle.contracts import errors
from scmle.contracts import experiments
from scmle.contracts import tasks
from scmle.evaluation import metrics
from scmle.evaluation import selection
from scmle.models import registry as registry_module

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
    features = pandas.DataFrame({"a": numpy.arange(40.0)})
    labels = pandas.Series([0, 1] * 20)
    classifier = dummy.DummyClassifier(strategy="prior").fit(features, labels)
    scores = CATALOG.evaluate(
        classifier, features, labels, tasks.TaskType.BINARY
    )
    assert scores["roc_auc"] == 0.5
    assert scores["log_loss"] == pytest.approx(0.6931, abs=1e-3)
    values = pandas.Series(numpy.arange(40.0))
    regressor = dummy.DummyRegressor().fit(features, values)
    scores = CATALOG.evaluate(
        regressor, features, values, tasks.TaskType.REGRESSION
    )
    assert scores["rmse"] > 0 and scores["mae"] > 0
    assert scores["r2"] == pytest.approx(0.0, abs=1e-9)


def test_evaluate_reports_nan_for_undefined_metric():
    features = pandas.DataFrame({"a": numpy.arange(40.0)})
    labels = pandas.Series([0, 1] * 20)
    classifier = dummy.DummyClassifier(strategy="prior").fit(features, labels)
    single_class = pandas.Series([1] * 40)
    scores = CATALOG.evaluate(
        classifier, features, single_class, tasks.TaskType.BINARY
    )
    assert numpy.isnan(scores["roc_auc"])


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
    with pytest.raises(errors.ScmleError, match="all trials failed"):
        policy.select(trials, CATALOG.metrics["rmse"], registry)


def string_problem():
    rng = numpy.random.default_rng(0)
    features = pandas.DataFrame({"a": rng.normal(size=200)})
    labels = pandas.Series(
        numpy.where(features.a + rng.normal(size=200) > 0, "true", "false")
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
    assert not any(numpy.isnan(value) for value in scores.values()), scores


def test_positive_label_is_none_outside_binary_tasks():
    labels = pandas.Series([0, 1, 2])
    assert metrics.positive_label(labels, tasks.TaskType.MULTICLASS) is None
    assert (
        metrics.positive_label(pandas.Series([3, 7, 3]), tasks.TaskType.BINARY)
        == 7
    )


def scored_trial(name, score, std, folds=5):
    return TrialResult(
        f"t_{name}",
        name,
        Status.OK,
        "m",
        score_mean=score,
        score_std=std,
        fold_scores=[score] * folds,
    )


def test_tolerance_turns_small_gaps_into_ties_resolved_by_simplicity(registry):
    trials = [
        scored_trial("random_forest", 0.801, 0.01),
        scored_trial("decision_tree", 0.800, 0.01),
    ]
    auc = CATALOG.metrics["roc_auc"]
    strict = selection.BestMeanPolicy().select(trials, auc, registry)
    assert strict.model_name == "random_forest"
    loose = selection.BestMeanPolicy(
        tasks.SelectionSpec(tie_tolerance=0.005)
    ).select(trials, auc, registry)
    assert loose.model_name == "decision_tree"  # cheaper wins the tie


def test_one_standard_error_rule_treats_noise_as_a_tie(registry):
    trials = [
        scored_trial("random_forest", 0.810, 0.05),
        scored_trial("decision_tree", 0.790, 0.05),
    ]
    auc = CATALOG.metrics["roc_auc"]
    rule = tasks.SelectionSpec(uncertainty="one_standard_error")
    policy = selection.BestMeanPolicy(rule)
    assert policy.select(trials, auc, registry).model_name == "decision_tree"
    assert "one standard error" in policy.description


def test_lower_variance_tie_breaker_prefers_the_steadier_model(registry):
    trials = [
        scored_trial("decision_tree", 0.800, 0.08),
        scored_trial("random_forest", 0.800, 0.01),
    ]
    auc = CATALOG.metrics["roc_auc"]
    policy = selection.BestMeanPolicy(
        tasks.SelectionSpec(tie_breaker="lower_variance")
    )
    assert policy.select(trials, auc, registry).model_name == "random_forest"
    default = selection.BestMeanPolicy().select(trials, auc, registry)
    assert default.model_name == "decision_tree"


def test_maximize_sets_the_metric_and_conflicts_are_rejected():
    spec = tasks.ProjectSpec(
        data=tasks.DataSpec(uri="x", target="y"),
        selection=tasks.SelectionSpec(maximize="pr_auc"),
    )
    assert spec.primary_metric == "pr_auc"
    with pytest.raises(ValueError, match="conflicts with metric"):
        tasks.ProjectSpec(
            data=tasks.DataSpec(uri="x", target="y"),
            metric="roc_auc",
            selection=tasks.SelectionSpec(maximize="pr_auc"),
        )


def test_latency_safety_factor_scales_the_measured_latency():
    from scmle.evaluation import resources

    usage = resources.ResourceUsage(1.0, 10.0, 30, "test machine")
    plain = resources.ResourceLimits(max_p95_latency_ms=15.0)
    assert plain.violations(usage) == []
    slower = resources.ResourceLimits(
        max_p95_latency_ms=15.0, latency_safety_factor=2.0
    )
    assert "safety factor 2" in slower.violations(usage)[0]


def folds_trial(name, scores):
    return TrialResult(
        f"t_{name}",
        name,
        Status.OK,
        "m",
        score_mean=float(numpy.mean(scores)),
        score_std=float(numpy.std(scores)),
        fold_scores=scores,
    )


def test_bayesian_selection_prefers_the_simpler_model_when_gain_is_noise(
    registry,
):
    policy = selection.BestMeanPolicy(
        tasks.SelectionSpec(uncertainty="bayesian")
    )
    noisy_winner = folds_trial("random_forest", [0.80, 0.86, 0.78, 0.85, 0.79])
    simpler = folds_trial("decision_tree", [0.83, 0.82, 0.82, 0.82, 0.83])

    winner = policy.select(
        [noisy_winner, simpler], CATALOG.metrics["roc_auc"], registry
    )

    assert noisy_winner.score_mean > simpler.score_mean - 0.01
    assert winner.model_name == "decision_tree"


def test_bayesian_selection_keeps_a_credibly_better_model(registry):
    policy = selection.BestMeanPolicy(
        tasks.SelectionSpec(uncertainty="bayesian")
    )
    clear_winner = folds_trial("random_forest", [0.90, 0.91, 0.90, 0.92, 0.91])
    simpler = folds_trial("decision_tree", [0.80, 0.81, 0.80, 0.82, 0.81])

    winner = policy.select(
        [clear_winner, simpler], CATALOG.metrics["roc_auc"], registry
    )

    assert winner.model_name == "random_forest"
    assert "probability" in policy.description


def test_a_rope_makes_negligible_gains_ties(registry):
    policy = selection.BestMeanPolicy(
        tasks.SelectionSpec(uncertainty="bayesian", rope=0.02)
    )
    slightly_better = folds_trial(
        "random_forest", [0.851, 0.852, 0.851, 0.852, 0.851]
    )
    simpler = folds_trial("decision_tree", [0.84, 0.84, 0.84, 0.84, 0.84])

    winner = policy.select(
        [slightly_better, simpler], CATALOG.metrics["roc_auc"], registry
    )

    assert winner.model_name == "decision_tree"


def test_a_lower_is_better_positive_label_metric_scores_sign_adjusted():
    from sklearn import metrics as sk_metrics

    from scmle.contracts import metrics as metric_contracts

    metric = metric_contracts.PositiveLabelMetric(
        "false_f1",
        "f1",
        False,
        frozenset({tasks.TaskType.BINARY}),
        sk_metrics.f1_score,
        ("predict",),
    )
    target = pandas.Series(["a", "b", "a", "b"])
    features = pandas.DataFrame({"x": [0.0, 1.0, 0.0, 1.0]})
    fitted = dummy.DummyClassifier(strategy="most_frequent").fit(
        features, target
    )

    scorer = metric.scorer("a")

    assert scorer(fitted, features, target) <= 0
    assert metric.score(fitted, features, target, "a") >= 0


def test_positive_label_needs_two_distinct_labels():
    with pytest.raises(ValueError, match="two distinct labels"):
        metrics.positive_label(pandas.Series([1, 1, 1]), tasks.TaskType.BINARY)
