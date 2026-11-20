"""Ordinal adapters wrap regressors with learned thresholds."""

import numpy
from sklearn import ensemble
from sklearn import linear_model

from scmle.contracts import tasks
from scmle.models.ordinal import adapters
from scmle.models.ordinal import estimators


def test_the_ordinal_constants_are_fixed():
    assert {tasks.Task.ORDINAL} == adapters.ORDINAL
    assert adapters.LEVELS == "levels"
    assert dict(adapters.Ordinal.params) == {"levels": 3}
    assert adapters.ORDINAL_ADAPTERS == (
        adapters.ORDINAL_RIDGE,
        adapters.ORDINAL_FOREST,
        adapters.ORDINAL_BOOSTING,
    )


def test_an_ordinal_adapter_wraps_a_regressor_with_levels():
    built = adapters.ORDINAL_FOREST.build(
        tasks.Task.ORDINAL, 7, {"levels": 5, "n_estimators": 60}
    )
    assert isinstance(built, estimators.Threshold)
    assert built.levels == 5
    assert built.random_state == 7
    assert isinstance(built.regressor, ensemble.RandomForestRegressor)
    assert built.regressor.n_estimators == 60
    assert built.regressor.random_state == 7


def test_an_ordinal_adapter_without_levels_infers_them():
    built = adapters.ORDINAL_RIDGE.build(tasks.Task.ORDINAL, 1)
    assert built.levels is None
    assert isinstance(built.regressor, linear_model.Ridge)
    params = {"levels": 4, "alpha": 2.0}
    adapters.ORDINAL_RIDGE.build(tasks.Task.ORDINAL, 1, params)
    assert params == {"levels": 4, "alpha": 2.0}


def test_the_ordinal_adapters_declare_their_capabilities():
    assert adapters.ORDINAL_RIDGE.name == "ordinal_ridge"
    assert adapters.ORDINAL_RIDGE.family == "linear"
    assert adapters.ORDINAL_RIDGE.capabilities.scaling is True
    assert adapters.ORDINAL_RIDGE.capabilities.cost == 2
    assert adapters.ORDINAL_FOREST.capabilities.factor == 6.0
    assert adapters.ORDINAL_BOOSTING.capabilities.missing is True
    assert (
        adapters.ORDINAL_BOOSTING.regressor
        is ensemble.HistGradientBoostingRegressor
    )
    for adapter in adapters.ORDINAL_ADAPTERS:
        assert adapter.capabilities.tasks == {tasks.Task.ORDINAL}
        assert adapter.capabilities.probabilities is True
    names = list(adapters.ORDINAL_FOREST.space.names())
    assert names == ["n_estimators", "min_samples_leaf"]


def test_an_ordinal_adapter_fits_ordered_levels():
    rng = numpy.random.default_rng(0)
    x = rng.normal(size=(150, 2))
    y = numpy.digitize(x[:, 0], [-0.5, 0.5])
    model = adapters.ORDINAL_RIDGE.build(
        tasks.Task.ORDINAL, 0, {"levels": 3}
    ).fit(x, y)
    probabilities = model.predict_proba(x)
    assert probabilities.shape == (150, 3)
    numpy.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
