"""Tests for the bagging adapters."""

import numpy
import pytest
from sklearn import ensemble

from scmle.contracts import tasks
from scmle.models import conformance
from scmle.models.sklearn import bagging


@pytest.mark.parametrize("adapter_class", bagging.BAGGING_ADAPTERS)
def test_bagging_adapters_pass_the_shared_conformance_suite(adapter_class):
    conformance.check_conformance(adapter_class())


def test_the_bagger_wraps_the_base_model_and_applies_both_parameter_sets():
    adapter = bagging.BaggedKnn()

    estimator = adapter.build(
        tasks.Task.BINARY,
        0,
        {"n_estimators": 12, "max_samples": 0.7, "n_neighbors": 9},
    )

    assert isinstance(estimator, ensemble.BaggingClassifier)
    assert estimator.n_estimators == 12 and estimator.max_samples == 0.7
    assert estimator.estimator.n_neighbors == 9


def test_regression_uses_the_regression_bagger_and_trees_gain_from_bagging():
    rng = numpy.random.default_rng(0)
    features = rng.normal(size=(400, 3))
    target = features[:, 0] * 2 + rng.normal(0, 1.0, 400)
    adapter = bagging.BaggedTree()
    single = adapter.base.build(tasks.Task.REGRESSION, 0).fit(
        features[:300], target[:300]
    )
    bagged = adapter.build(tasks.Task.REGRESSION, 0).fit(
        features[:300], target[:300]
    )

    single_error = numpy.mean(
        (single.predict(features[300:]) - target[300:]) ** 2
    )
    bagged_error = numpy.mean(
        (bagged.predict(features[300:]) - target[300:]) ** 2
    )

    assert isinstance(bagged, ensemble.BaggingRegressor)
    assert bagged_error < single_error


def test_capabilities_follow_the_base_model_with_higher_cost():
    adapter = bagging.BaggedLogistic()

    assert adapter.capabilities.tasks == adapter.base.capabilities.tasks
    assert adapter.capabilities.cost == adapter.base.capabilities.cost + 1
    assert not adapter.capabilities.sparse_input
