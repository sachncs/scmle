"""Validation folds are scored with the rows' own weights."""

import numpy
import pandas
import pytest
from sklearn import metrics

from scmle.contracts import data
from scmle.contracts import tasks
import scmle.contracts.metrics
from scmle.experimentation import trials


def payload(weighted):
    features = pandas.DataFrame(
        {"x": [1.0, 2.0, 3.0, 4.0], "w": [1, 2, 3, 4.0]}
    )
    return trials.Payload(
        job=trials.Job("t", "dummy"),
        adapter=None,
        strategy=None,
        feature_plan=data.Blueprint(("x",), (), weight_column="w"),
        task=tasks.Task.REGRESSION,
        seed=0,
        metric=None,
        positive_label=None,
        features=features,
        target=pandas.Series([1.0, 2.0, 3.0, 4.0]),
        folds=[],
        threadcap=None,
        weighted=weighted,
    )


def test_the_weights_of_the_scored_rows_are_passed_to_the_scorer():
    found = trials.weigh(payload(True), numpy.array([1, 3]))

    assert list(found["sample_weight"]) == [2.0, 4.0]


def test_unweighted_selection_passes_nothing():
    assert trials.weigh(payload(False), numpy.array([1, 3])) == {}


def test_a_weighted_score_equals_the_hand_computed_one():
    truth = numpy.array([1.0, 2.0, 3.0])
    guess = numpy.array([1.0, 2.0, 5.0])
    weights = numpy.array([1.0, 1.0, 2.0])

    value = metrics.mean_absolute_error(truth, guess, sample_weight=weights)

    assert value == pytest.approx((0 + 0 + 2 * 2) / 4)


def plain_error(truth, guess):
    return float(numpy.abs(truth - guess).mean())


def weighted_error(truth, guess, sample_weight=None):
    return float(numpy.average(numpy.abs(truth - guess), weights=sample_weight))


def test_metrics_report_whether_their_function_takes_weights():
    every = frozenset(tasks.STANDARD_TASKS)
    plain = scmle.contracts.metrics.Formula("a", "", False, every, plain_error)
    weighted = scmle.contracts.metrics.Formula(
        "b", "", False, every, weighted_error
    )

    assert not plain.weightable()
    assert weighted.weightable()
