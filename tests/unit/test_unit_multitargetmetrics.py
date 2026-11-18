"""Multi-target metrics score every target column at once."""

import numpy
import pandas
import pytest

from scmle.contracts import tasks
from scmle.evaluation import multitarget

TRUTH = numpy.array([[1, 0], [1, 1], [0, 0], [0, 1]])
GUESS = numpy.array([[1, 0], [0, 1], [0, 1], [0, 1]])


def test_a_macro_f1_averages_the_per_label_f1():
    # label 0: tp=1 fp=0 fn=1 -> 2/3; label 1: tp=2 fp=1 fn=0 -> 0.8
    assert multitarget.f1_macro_labels(TRUTH, GUESS) == pytest.approx(
        (2 / 3 + 0.8) / 2
    )


def test_a_micro_f1_pools_every_decision():
    # tp=3 fp=1 fn=1 -> 6/8
    assert multitarget.f1_micro_labels(TRUTH, GUESS) == pytest.approx(0.75)


def test_an_f1_with_no_positives_counts_as_zero():
    empty = numpy.zeros((2, 2), dtype=int)
    assert multitarget.f1_macro_labels(empty, empty) == 0.0
    assert multitarget.f1_micro_labels(empty, empty) == 0.0


def test_a_hamming_loss_is_the_share_of_wrong_decisions():
    assert multitarget.hamming(TRUTH, GUESS) == 0.25


def test_a_subset_accuracy_needs_every_label_right():
    assert multitarget.subset_accuracy(TRUTH, GUESS) == 0.5
    assert multitarget.subset_accuracy(TRUTH, TRUTH) == 1.0
    assert isinstance(multitarget.subset_accuracy(TRUTH, GUESS), float)


def test_an_r2_mean_averages_columns_uniformly():
    truth = numpy.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    guess = numpy.array([[1.0, 2.0], [2.0, 4.0], [3.0, 2.0]])
    # column 1: ss_res=16, ss_tot=8 -> -1
    assert multitarget.r2_mean(truth, guess) == pytest.approx((1.0 - 1.0) / 2)


def test_a_scaled_mae_divides_by_the_population_spread():
    truth = numpy.array([[0.0, 5.0], [2.0, 5.0]])
    guess = numpy.array([[1.0, 6.0], [2.0, 3.0]])
    # column 0 std 1, column 1 constant -> 1
    assert multitarget.mae_mean_scaled(truth, guess) == pytest.approx(
        (1.0 + 0.0 + 1.0 + 2.0) / 4
    )


def test_the_function_table_names_every_metric():
    assert {
        "f1_macro_labels": multitarget.f1_macro_labels,
        "f1_micro_labels": multitarget.f1_micro_labels,
        "hamming": multitarget.hamming,
        "subset_accuracy": multitarget.subset_accuracy,
        "r2_mean": multitarget.r2_mean,
        "mae_scaled": multitarget.mae_mean_scaled,
    } == multitarget.FUNCTIONS


class Stub:
    target_columns = ("first", "second")

    def predict(self, features):
        return GUESS


def joint(kind, better):
    return multitarget.Joint(
        "name", "scorer", better, frozenset({tasks.Task.BINARY}), kind
    )


def test_a_joint_metric_defaults_to_macro_f1_and_is_unweighted():
    metric = multitarget.Joint("n", "s", True, frozenset())
    assert metric.kind == "f1_macro_labels"
    assert metric.weightable() is False


def test_a_joint_scorer_stacks_the_primary_and_other_targets():
    features = pandas.DataFrame({"second": TRUTH[:, 1], "x": [0, 0, 0, 0]})
    score = joint("hamming", True).scorer()
    assert score(Stub(), features, pandas.Series(TRUTH[:, 0])) == 0.25


def test_a_joint_scorer_negates_when_lower_is_better():
    features = pandas.DataFrame({"second": TRUTH[:, 1]})
    score = joint("hamming", False).scorer(positive="ignored")
    assert score(Stub(), features, TRUTH[:, 0], sample_weight=[9] * 4) == -0.25
