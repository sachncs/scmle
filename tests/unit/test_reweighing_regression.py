"""Reweighing for a continuous outcome: the outcome is cut into bins."""

import numpy
import pandas
import pytest

from scmle.contracts import capabilities
from scmle.contracts import tasks
from scmle.evaluation import reweighing
from scmle.pipeline import kinds


def test_the_bin_count_follows_sturges_and_is_capped():
    assert reweighing.MAX_BINS == 10
    assert reweighing.bins(1) == 1
    assert reweighing.bins(2) == 2
    assert reweighing.bins(8) == 4
    assert reweighing.bins(9) == 5
    assert reweighing.bins(100) == 8
    assert reweighing.bins(512) == 10
    assert reweighing.bins(100000) == 10


def test_a_continuous_outcome_is_cut_into_equal_count_bins():
    values = pandas.Series(numpy.arange(100, dtype=float)[::-1])

    cut = reweighing.discretise(values)

    assert sorted(cut.value_counts().tolist()) == [12, 12, 12, 12] + [13] * 4
    assert cut.iloc[0] == 7 and cut.iloc[-1] == 0
    assert cut.index.equals(values.index)


def test_ties_never_split_a_value_across_bins_by_order_alone():
    values = pandas.Series([1.0] * 50 + [2.0] * 50)

    cut = reweighing.discretise(values)

    assert cut.groupby(values).nunique().tolist() == [1, 1]
    assert sorted(cut.unique().tolist()) == [0, 1]


def test_a_missing_outcome_is_refused():
    with pytest.raises(ValueError, match=r"^the outcome has missing values$"):
        reweighing.discretise(pandas.Series([1.0, numpy.nan, 2.0]))


def test_binned_weights_make_group_and_outcome_level_independent():
    rng = numpy.random.default_rng(4)
    groups = pandas.Series(rng.choice(["a", "b"], size=400))
    outcome = pandas.Series(rng.normal(size=400) + (groups == "a") * 1.5)

    levels = reweighing.discretise(outcome)
    weights = reweighing.weigh(groups, levels)

    frame = pandas.DataFrame({"g": groups, "y": levels, "w": weights})
    joint = frame.groupby(["g", "y"]).w.sum() / frame.w.sum()
    group = frame.groupby("g").w.sum() / frame.w.sum()
    level = frame.groupby("y").w.sum() / frame.w.sum()
    expected = [group[g] * level[y] for g, y in joint.index]
    assert joint.to_numpy() == pytest.approx(expected)
    assert weights.mean() == pytest.approx(1.0)


def test_regression_is_registered_with_mitigation():
    assert kinds.fairnesses(tasks.Task.REGRESSION) == (
        capabilities.Fairness.NONE,
        capabilities.Fairness.MITIGATION,
    )
    assert kinds.fairnesses(tasks.Task.ORDINAL) == (capabilities.Fairness.NONE,)
