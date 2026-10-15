"""Tests for data strategies."""

import numpy as np
import pandas as pd
import pytest

from mml.contracts import errors
from mml.contracts import tasks
from mml.data import profiling
from mml.data import readers
from mml.data import strategies
from mml.models import registry as registry_module
from mml.models import resampling

REGISTRY = registry_module.ModelRegistry.with_builtins()
BINARY = tasks.TaskType.BINARY


def profile_of(frame, task=BINARY):
    spec = tasks.ProjectSpec(data=tasks.DataSpec(uri="m", target="y"))
    manifest = readers.build_manifest(frame, "y")
    return profiling.DatasetProfiler(spec).profile(frame, manifest, task)


@pytest.fixture
def skewed_imbalanced():
    rng = np.random.default_rng(0)
    rows = 400
    frame = pd.DataFrame(
        {
            "spend": rng.exponential(5.0, rows),
            "age": rng.normal(40, 10, rows),
            "tenure": rng.exponential(2.0, rows),
        }
    )
    frame["y"] = (rng.random(rows) < 0.1).astype(int)
    return frame


@pytest.fixture
def balanced():
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({"a": rng.normal(size=300), "b": rng.normal(size=300)})
    frame["y"] = rng.integers(0, 2, 300)
    return frame


def reason(name, profile, model, task=BINARY):
    return strategies.get_strategy(name).applicability(
        profile, task, REGISTRY.get(model)
    )


def test_imbalance_strategies_apply_only_to_skewed_classification(
    skewed_imbalanced, balanced
):
    skewed, even = profile_of(skewed_imbalanced), profile_of(balanced)
    for name in ("class_weight", "random_oversample", "random_undersample"):
        assert reason(name, skewed, "logistic_regression") is None
        assert "balanced enough" in reason(name, even, "logistic_regression")
    regression = skewed_imbalanced.assign(y=skewed_imbalanced.spend * 2)
    assert "classification" in reason(
        "random_oversample",
        profile_of(regression, tasks.TaskType.REGRESSION),
        "ridge",
        tasks.TaskType.REGRESSION,
    )


def test_class_weight_requires_estimator_support(skewed_imbalanced):
    from sklearn import dummy

    from mml.contracts import models

    class NoWeights(models.ModelAdapter):
        name = "no_weights"
        family = "test"
        capabilities = models.Capabilities(frozenset({BINARY}))

        def build(self, task, seed, params=None):
            return dummy.DummyClassifier(strategy="prior")

    profile = profile_of(skewed_imbalanced)
    assert reason("class_weight", profile, "random_forest") is None
    strategy = strategies.get_strategy("class_weight")
    found = strategy.applicability(profile, BINARY, NoWeights())
    assert found == "no_weights has no class_weight parameter"


def test_baseline_is_never_transformed(skewed_imbalanced):
    profile = profile_of(skewed_imbalanced)
    for name in strategies.EXPERIMENTAL:
        assert "baseline" in reason(name, profile, "dummy")
    assert reason("default", profile, "dummy") is None


def test_feature_strategies_follow_model_family(skewed_imbalanced):
    profile = profile_of(skewed_imbalanced)
    assert reason("log_skewed", profile, "logistic_regression") is None
    assert "tree splits" in reason("log_skewed", profile, "random_forest")
    assert reason("interactions", profile, "logistic_regression") is None
    assert "non-linear" in reason("interactions", profile, "random_forest")


def test_log_skewed_needs_a_skewed_feature(balanced):
    assert "no right-skewed" in reason(
        "log_skewed", profile_of(balanced), "logistic_regression"
    )


def test_interaction_feature_count_limits():
    rng = np.random.default_rng(2)
    wide = pd.DataFrame({f"f{i}": rng.normal(size=100) for i in range(14)})
    wide["y"] = rng.integers(0, 2, 100)
    assert "outside" in reason(
        "interactions", profile_of(wide), "logistic_regression"
    )
    narrow = wide[["f0", "y"]]
    assert "outside" in reason(
        "interactions", profile_of(narrow), "logistic_regression"
    )


def test_builds_produce_fittable_candidates(skewed_imbalanced):
    profile = profile_of(skewed_imbalanced)
    plan = profile.feature_plan
    features = skewed_imbalanced.drop(columns="y")
    target = skewed_imbalanced["y"]
    for name in strategies.STRATEGIES:
        adapter = REGISTRY.get("logistic_regression")
        candidate = strategies.get_strategy(name).build(
            plan, adapter, BINARY, 0
        )
        candidate.fit(features, target)
        assert candidate.predict_proba(features).shape == (400, 2), name


def test_resampling_strategy_wraps_the_pipeline(skewed_imbalanced):
    plan = profile_of(skewed_imbalanced).feature_plan
    candidate = strategies.get_strategy("random_undersample").build(
        plan, REGISTRY.get("random_forest"), BINARY, 3
    )
    assert isinstance(candidate, resampling.ResamplingClassifier)
    assert candidate.random_state == 3


def test_class_weight_is_injected_over_other_params(skewed_imbalanced):
    plan = profile_of(skewed_imbalanced).feature_plan
    candidate = strategies.get_strategy("class_weight").build(
        plan, REGISTRY.get("logistic_regression"), BINARY, 0, {"C": 0.5}
    )
    model = candidate.named_steps["model"]
    assert model.class_weight == "balanced" and model.C == 0.5


def test_registry_and_unknown_strategy():
    assert strategies.EXPERIMENTAL == (
        "class_weight",
        "random_oversample",
        "random_undersample",
        "calibrated",
        "log_skewed",
        "interactions",
    )
    with pytest.raises(errors.ConfigError, match="unknown strategy"):
        strategies.get_strategy("nope")
