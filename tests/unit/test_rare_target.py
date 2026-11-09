"""Regression has its own rare-target policy; classification resampling is not
reused for it."""

import numpy
import pandas
import pytest

from scmle.contracts import data
from scmle.contracts import tasks
import scmle.models.registry
from scmle.sampling import resampling
from scmle.sampling import strategies

REGRESSION = tasks.TaskType.REGRESSION


@pytest.fixture
def registry():
    return scmle.models.registry.ModelRegistry.with_builtins()


def long_tailed(rows=800, seed=0):
    rng = numpy.random.default_rng(seed)
    x = rng.normal(size=rows)
    target = numpy.exp(1.2 * x + rng.normal(scale=0.3, size=rows))
    return pandas.DataFrame({"x": x}), pandas.Series(target)


def profile_with(skew):
    return data.DatasetProfile(
        manifest=None,
        column_statistics={},
        quality_findings=(),
        risk_findings=(),
        target_summary={"task": "regression", "skew": skew},
        feature_plan=data.FeaturePlan(("x",), ()),
    )


def test_weights_favour_sparse_regions_of_the_target_range():
    target = long_tailed()[1]
    model = resampling.DensityWeightedRegressor(estimator=None)
    weights = model.weights(target.to_numpy())
    assert weights.mean() == pytest.approx(1.0)
    tail = target.to_numpy() > numpy.quantile(target, 0.97)
    assert weights[tail].mean() > 3 * weights[~tail].mean()
    assert weights.max() <= weights.min() * model.max_weight * (1 + 1e-9)


def test_strategy_applies_only_to_long_tailed_regression(registry):
    strategy = strategies.get_strategy("rare_target_weighting")
    forest = registry.get("random_forest")
    assert strategy.applicability(profile_with(3.0), REGRESSION, forest) is None
    assert "not long-tailed" in strategy.applicability(
        profile_with(0.2), REGRESSION, forest
    )
    assert "regression only" in strategy.applicability(
        profile_with(3.0), tasks.TaskType.BINARY, forest
    )
    knn = registry.get("knn")
    assert "sample weights" in strategy.applicability(
        profile_with(3.0), REGRESSION, knn
    )


def test_classification_resampling_strategies_never_touch_regression(registry):
    forest = registry.get("random_forest")
    for name in ("random_oversample", "smote", "class_weight"):
        reason = strategies.get_strategy(name).applicability(
            profile_with(3.0), REGRESSION, forest
        )
        assert reason is not None


def test_weighted_pipeline_fits_and_predicts(registry):
    features, target = long_tailed()
    plan = data.FeaturePlan(("x",), ())
    candidate = strategies.get_strategy("rare_target_weighting").build(
        plan, registry.get("random_forest"), REGRESSION, 0
    )
    candidate.fit(features, target)
    predictions = candidate.predict(features)
    assert predictions.shape == (len(features),)
    assert numpy.corrcoef(predictions, target)[0, 1] > 0.8
