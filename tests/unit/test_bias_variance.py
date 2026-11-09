"""Bootstrap bias and variance of known model families."""

import numpy
import pandas
import pytest
from sklearn import dummy
from sklearn import linear_model
from sklearn import tree

from scmle.evaluation import bias_variance


def curve(rows=400, seed=0, noise=0.3):
    rng = numpy.random.default_rng(seed)
    x = rng.uniform(-3, 3, rows)
    return pandas.DataFrame({"x": x}), pandas.Series(
        numpy.sin(2 * x) + rng.normal(0, noise, rows)
    )


def test_a_straight_line_on_a_curve_is_bias_dominated():
    features, target = curve()

    found = bias_variance.decompose(
        linear_model.LinearRegression, features, target, False, 25
    )

    assert found.verdict == "bias-dominated"
    assert found.bias > 5 * found.variance
    assert found.noise == pytest.approx(0.09, abs=0.08)  # Noise is 0.3**2.


def test_deeper_trees_have_more_variance():
    features, target = curve(noise=1.0)

    shallow = bias_variance.decompose(
        lambda: tree.DecisionTreeRegressor(max_depth=3),
        features,
        target,
        False,
        25,
    )
    deep = bias_variance.decompose(
        tree.DecisionTreeRegressor, features, target, False, 25
    )

    assert deep.variance > 2 * shallow.variance
    assert deep.noise == pytest.approx(1.0, abs=0.35)


def test_the_parts_roughly_add_up_to_the_error_of_single_fits():
    features, target = curve(noise=0.6)

    found = bias_variance.decompose(
        lambda: tree.DecisionTreeRegressor(max_depth=4),
        features,
        target,
        False,
        30,
    )

    assert found.total == pytest.approx(
        found.bias_plus_noise + found.variance, rel=0.15
    )


def test_classification_uses_zero_one_error():
    rng = numpy.random.default_rng(1)
    features = pandas.DataFrame({"x": rng.normal(size=300)})
    target = pandas.Series(
        (features.x + rng.normal(0, 0.8, 300) > 0).astype(int)
    )

    stump = bias_variance.decompose(
        lambda: tree.DecisionTreeClassifier(max_depth=1),
        features,
        target,
        True,
        20,
    )
    prior = bias_variance.decompose(
        lambda: dummy.DummyClassifier(strategy="prior"),
        features,
        target,
        True,
        20,
    )

    assert stump.kind == "zero_one" and stump.bias < prior.bias
    assert 0.0 <= stump.variance <= 0.5


def test_stopping_early_returns_nothing():
    features, target = curve()

    assert (
        bias_variance.decompose(
            linear_model.LinearRegression,
            features,
            target,
            False,
            10,
            should_stop=lambda: True,
        )
        is None
    )
