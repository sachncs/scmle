"""Sensitivity's splits, seeds and noise draws are the ones it states."""

import math

import numpy
import pandas
from sklearn import dummy
from sklearn import tree

from scmle.evaluation import sensitivity

ROWS = 50


def data() -> tuple[pandas.DataFrame, pandas.Series]:
    rng = numpy.random.default_rng(0)
    features = pandas.DataFrame({"a": rng.normal(size=ROWS)})
    return features, pandas.Series(rng.normal(size=ROWS))


def rows(model, features, target) -> float:
    """Scores by which rows were held out, so a moved split shows."""
    return float(sum(target.index))


def shifted(model, features, target) -> float:
    """Scores by the features, so the noise draws show."""
    return float(features["a"].sum())


def test_reseeded_sets_the_random_state_it_is_given() -> None:
    assert sensitivity.reseeded(tree.DecisionTreeRegressor, 7).random_state == 7


def test_the_split_of_variability_is_fixed_by_the_seed() -> None:
    features, target = data()
    first = sensitivity.variability(
        dummy.DummyRegressor, features, target, rows, repeats=2, seed=3
    )
    second = sensitivity.variability(
        dummy.DummyRegressor, features, target, rows, repeats=2, seed=3
    )
    assert first[0].scores == second[0].scores


def test_zero_noise_is_scored_once_and_other_levels_repeat() -> None:
    features, target = data()
    calls = []

    def counted(model, features, target) -> float:
        calls.append(len(target))
        return 0.0

    sensitivity.jitter(
        None, features, target, counted, levels=(0.0, 0.5), repeats=3
    )
    assert len(calls) == 4


def test_analyze_holds_out_thirty_percent_and_reuses_its_noise_seed() -> None:
    features, target = data()
    lengths = []

    def counted(model, features, target) -> float:
        lengths.append(len(target))
        return shifted(model, features, target)

    found = sensitivity.analyze(
        dummy.DummyRegressor, features, target, counted, repeats=2, seed=4
    )
    assert lengths[-1] == math.ceil(0.3 * ROWS)
    again = sensitivity.analyze(
        dummy.DummyRegressor, features, target, shifted, repeats=2, seed=4
    )
    assert found.noise == again.noise
