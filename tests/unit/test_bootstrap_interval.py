"""Tests for the studentized (bootstrap-t) metric interval (claim C3)."""

import dataclasses

import numpy
import pandas
import sklearn.metrics

from scmle.contracts import tasks
from scmle.evaluation import bootstrap
from scmle.evaluation import metrics
from tests.unit import test_diagnostics


def pairwise(values, truth, units):
    """DeLong placements and the clustered variance, the slow way."""
    positives, negatives = values[truth], values[~truth]
    wins = (positives[:, None] > negatives[None, :]) + 0.5 * (
        positives[:, None] == negatives[None, :]
    )
    area = wins.mean()
    effects = numpy.zeros(len(values))
    effects[truth] = (wins.mean(1) - area) / len(positives)
    effects[~truth] = (wins.mean(0) - area) / len(negatives)
    labels = numpy.unique(units)
    sums = numpy.array([effects[units == label].sum() for label in labels])
    count = len(labels)
    return area, numpy.sqrt(count / (count - 1) * (sums**2).sum())


def test_rank_matches_roc_auc_and_the_clustered_delong_error():
    rng = numpy.random.default_rng(0)
    values = numpy.round(rng.normal(size=120), 1)  # ties on purpose
    truth = rng.uniform(size=120) < 1 / (1 + numpy.exp(-2 * values))
    units = numpy.repeat(numpy.arange(30), 4)

    area, error = bootstrap.studentize("roc_auc", values, truth, units)

    expected, spread = pairwise(values, truth, units)
    assert numpy.isclose(area, sklearn.metrics.roc_auc_score(truth, values))
    assert numpy.isclose(area, expected) and numpy.isclose(error, spread)


def test_square_matches_rmse_and_the_delta_method_error():
    rng = numpy.random.default_rng(1)
    values, truth = rng.normal(size=200), rng.normal(size=200)
    units = numpy.arange(200)

    rmse, error = bootstrap.studentize("rmse", values, truth, units)

    squares = (truth - values) ** 2
    expected = numpy.sqrt(squares.mean())
    spread = numpy.sqrt(200 / 199 * ((squares - squares.mean()) ** 2).sum())
    assert numpy.isclose(rmse, expected)
    assert numpy.isclose(error, spread / 200 / (2 * expected))


def test_studentize_declines_a_single_class_or_an_unknown_metric():
    values, units = numpy.arange(4.0), numpy.arange(4)
    assert (
        bootstrap.studentize("roc_auc", values, numpy.ones(4, bool), units)
        is None
    )
    assert bootstrap.studentize("mae", values, values, units) is None


def test_draw_labels_each_drawn_group_copy_as_its_own_unit():
    groups = pandas.Series(numpy.repeat(numpy.arange(10), 5))
    rows, units = bootstrap.draw(50, groups, False, numpy.random.default_rng(3))
    again = bootstrap.redraw(50, groups, False, numpy.random.default_rng(3))
    assert numpy.array_equal(rows, again)
    assert len(numpy.unique(units)) == 10
    for unit in numpy.unique(units):
        assert len(set(groups.iloc[rows[units == unit]])) == 1
        assert (units == unit).sum() == 5


def test_draw_makes_each_row_a_unit_without_groups():
    rows, units = bootstrap.draw(40, None, False, numpy.random.default_rng(4))
    assert len(rows) == 40 and numpy.array_equal(units, numpy.arange(40))


def manual(inputs, groups):
    """The bootstrap-t interval rebuilt from the module's own pieces."""
    values = inputs.estimator.predict_proba(inputs.features)[:, 1]
    truth = inputs.target.to_numpy() == 1
    units = numpy.arange(len(truth)) if groups is None else groups.to_numpy()
    centre, error = bootstrap.studentize("roc_auc", values, truth, units)
    rng = numpy.random.default_rng(inputs.seed)
    ratios = []
    count = bootstrap.Uncertainty().replicates(len(truth))
    for replicate in range(count):  # noqa: B007
        rows, drawn = bootstrap.draw(len(truth), groups, False, rng)
        found = bootstrap.studentize(
            "roc_auc", values[rows], truth[rows], drawn
        )
        if found is not None:
            ratios.append((found[0] - centre) / found[1])
    high, low = numpy.percentile(ratios, [97.5, 2.5])
    return [centre - high * error, centre - low * error]


def test_grouped_roc_auc_interval_is_the_bootstrap_t_interval():
    inputs = dataclasses.replace(
        test_diagnostics.make_clustered_inputs(),
        metric=metrics.CATALOG.resolve(None, tasks.Task.BINARY),
    )
    groups = inputs.groups.reset_index(drop=True)

    report = bootstrap.Uncertainty().compute(inputs)

    assert numpy.allclose(report["ci_95"], manual(inputs, groups))
