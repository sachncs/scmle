"""Multi-label and multi-output estimators and their adapters."""

import numpy
import pytest
from sklearn import dummy
from sklearn import ensemble
from sklearn import linear_model

from scmle.contracts import tasks
from scmle.models.multitarget import adapters
from scmle.models.multitarget import estimators


def labels(rows=120, seed=0):
    rng = numpy.random.default_rng(seed)
    x = rng.normal(size=(rows, 2))
    y = numpy.column_stack(
        [(x[:, 0] > 0).astype(int), (x[:, 1] > 0).astype(int)]
    )
    return x, y


def test_the_distinct_value_floor_is_two():
    assert estimators.MIN_DISTINCT_VALUES == 2


def test_a_per_target_fit_clones_one_model_per_column():
    x, y = labels()
    model = estimators.PerTarget(
        linear_model.LogisticRegression(), classify=True
    ).fit(x, y)
    assert len(model.estimators_) == 2
    assert model.estimators_[0] is not model.estimator
    direct = linear_model.LogisticRegression().fit(x, y[:, 1])
    numpy.testing.assert_allclose(
        model.predict_proba(x)[:, 1], direct.predict_proba(x)[:, 1]
    )
    assert model.predict(x).shape == (120, 2)
    assert (model.predict(x) == y).mean() > 0.9


def test_a_constant_label_becomes_a_prior_classifier():
    x, y = labels()
    y[:, 1] = 0
    model = estimators.PerTarget(
        linear_model.LogisticRegression(), classify=True
    ).fit(x, y)
    assert isinstance(model.estimators_[1], dummy.DummyClassifier)
    assert model.predict_proba(x)[:, 1].tolist() == [0.0] * 120
    y[:, 1] = 1
    ones = estimators.PerTarget(
        linear_model.LogisticRegression(), classify=True
    ).fit(x, y)
    assert ones.predict_proba(x)[:, 1].tolist() == [1.0] * 120


def test_a_constant_regression_target_keeps_the_estimator():
    x, ignored = labels()
    y = numpy.column_stack([x[:, 0] * 2, numpy.ones(120)])
    model = estimators.PerTarget(linear_model.Ridge(alpha=1e-6)).fit(x, y)
    assert isinstance(model.estimators_[1], linear_model.Ridge)
    numpy.testing.assert_allclose(model.predict(x)[:, 1], 1.0)
    numpy.testing.assert_allclose(
        model.predict(x)[:, 0], x[:, 0] * 2, atol=1e-4
    )


def test_an_ensemble_models_all_targets_jointly():
    x, y = labels()
    model = estimators.forest(True, 3)
    assert isinstance(model.estimator, ensemble.RandomForestClassifier)
    assert model.estimator.n_estimators == 150
    assert model.estimator.random_state == 3
    assert model.classify is True
    model.fit(x, y)
    probabilities = model.predict_proba(x)
    assert probabilities.shape == (120, 2)
    found = model.estimator_.predict_proba(x)
    numpy.testing.assert_allclose(probabilities[:, 0], found[0][:, 1])
    assert model.predict(x).shape == (120, 2)
    regressor = estimators.forest(False, 1)
    assert isinstance(regressor.estimator, ensemble.RandomForestRegressor)
    assert regressor.classify is False


def test_an_ensemble_gives_zero_probability_without_a_positive_class():
    x, y = labels()
    y[:, 0] = 0
    model = estimators.Ensemble(
        ensemble.RandomForestClassifier(n_estimators=5, random_state=0), True
    ).fit(x, y)
    assert model.predict_proba(x)[:, 0].tolist() == [0.0] * 120


def test_the_task_sets_are_fixed():
    assert {
        tasks.Task.MULTILABEL,
        tasks.Task.MULTIOUTPUT,
        tasks.Task.FORECAST,
    } == adapters.MULTI
    assert {
        tasks.Task.MULTIOUTPUT,
        tasks.Task.FORECAST,
    } == adapters.REGRESSION_LIKE


def test_a_logistic_adapter_builds_one_seeded_regression_per_label():
    built = adapters.Logistic().build(tasks.Task.MULTILABEL, 4, {"C": 0.5})
    assert isinstance(built, estimators.PerTarget)
    assert built.classify is True
    assert built.estimator.C == 0.5
    assert built.estimator.max_iter == 1000
    assert built.estimator.random_state == 4
    assert adapters.Logistic.name == "multilabel_logistic"
    assert adapters.Logistic.capabilities.tasks == {tasks.Task.MULTILABEL}


def test_a_ridge_adapter_builds_one_regression_per_quantity():
    built = adapters.MultioutputRidge().build(
        tasks.Task.MULTIOUTPUT, 4, {"alpha": 3.0}
    )
    assert built.classify is False
    assert built.estimator.alpha == 3.0
    assert (
        adapters.MultioutputRidge.capabilities.tasks == adapters.REGRESSION_LIKE
    )
    assert adapters.MultioutputRidge.capabilities.cost == 1


@pytest.mark.parametrize(
    ("task", "kind", "classify"),
    [
        (tasks.Task.MULTILABEL, ensemble.HistGradientBoostingClassifier, True),
        (tasks.Task.MULTIOUTPUT, ensemble.HistGradientBoostingRegressor, False),
        (tasks.Task.FORECAST, ensemble.HistGradientBoostingRegressor, False),
    ],
)
def test_the_boosting_adapter_picks_the_class_by_task(task, kind, classify):
    built = adapters.MULTI_BOOSTING.build(task, 6, {"max_leaf_nodes": 10})
    assert isinstance(built.estimator, kind)
    assert built.estimator.random_state == 6
    assert built.estimator.max_leaf_nodes == 10
    assert built.classify is classify


def test_a_class_without_a_seed_is_built_without_one():
    adapter = adapters.Multitarget(
        name="m", classifier=dummy.DummyClassifier, regressor=linear_model.Ridge
    )
    built = adapter.build(tasks.Task.MULTIOUTPUT, 9, {"alpha": 2.0})
    assert isinstance(built.estimator, linear_model.Ridge)
    assert built.estimator.alpha == 2.0
    assert adapter.name == "m"
    assert adapter.classifier is dummy.DummyClassifier


def test_a_forest_adapter_ignores_params_and_follows_the_task():
    built = adapters.Forest().build(tasks.Task.MULTILABEL, 2, {"ignored": 1})
    assert isinstance(built.estimator, ensemble.RandomForestClassifier)
    assert built.estimator.random_state == 2
    other = adapters.Forest().build(tasks.Task.FORECAST, 2)
    assert isinstance(other.estimator, ensemble.RandomForestRegressor)
    assert adapters.Forest.capabilities.memory_factor == 6.0


def test_the_adapters_are_registered_in_order():
    assert [type(a) for a in adapters.MULTITARGET_ADAPTERS] == [
        adapters.Logistic,
        adapters.MultioutputRidge,
        adapters.Multitarget,
        adapters.Forest,
    ]
    assert adapters.MULTITARGET_ADAPTERS[2] is adapters.MULTI_BOOSTING
    assert adapters.MULTI_BOOSTING.capabilities.handles_missing is True
    assert adapters.MULTI_BOOSTING.search_space.names() == (
        "learning_rate",
        "max_leaf_nodes",
    )
