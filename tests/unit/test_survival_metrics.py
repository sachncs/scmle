"""Integrated Brier score and dynamic AUC against hand computations."""

import numpy
import pytest

from scmle.evaluation import survival


def test_the_brier_score_matches_a_hand_computation_without_censoring():
    duration = numpy.array([1.0, 2.0, 3.0])
    event = numpy.ones(3)
    times = numpy.array([1.5, 2.5])
    # Row 0 failed by both times, row 1 by 2.5, row 2 survived both.
    predicted = numpy.array([[0.2, 0.1], [0.9, 0.3], [0.8, 0.6]])

    scores = survival.brier_scores(duration, event, predicted, times)

    first = (0.2**2 + (1 - 0.9) ** 2 + (1 - 0.8) ** 2) / 3
    second = (0.1**2 + 0.3**2 + (1 - 0.6) ** 2) / 3
    assert scores == pytest.approx([first, second])


def test_integrating_two_points_is_their_mean():
    duration = numpy.array([1.0, 2.0, 3.0])
    event = numpy.ones(3)
    times = numpy.array([1.5, 2.5])
    predicted = numpy.array([[0.2, 0.1], [0.9, 0.3], [0.8, 0.6]])

    value = survival.integrated_brier(duration, event, predicted, times)

    scores = survival.brier_scores(duration, event, predicted, times)
    assert value == pytest.approx(scores.mean())


def test_dynamic_auc_is_one_for_a_perfect_ordering_and_half_for_ties():
    duration = numpy.array([1.0, 2.0, 3.0, 4.0])
    event = numpy.ones(4)
    times = numpy.array([2.5])

    assert survival.dynamic_auc(duration, event, -duration, times) == 1.0
    assert survival.dynamic_auc(duration, event, numpy.zeros(4), times) == 0.5


def test_a_better_model_has_a_lower_integrated_brier_score():
    rng = numpy.random.default_rng(0)
    rate = numpy.exp(rng.normal(size=600))
    happens = rng.exponential(1 / rate)
    stops = rng.exponential(2.0, 600)
    duration, event = (
        numpy.minimum(happens, stops),
        (happens <= stops).astype(int),
    )
    times = survival.time_grid(duration, event)

    informed = numpy.exp(-numpy.outer(rate, times))
    ignorant = numpy.tile(numpy.exp(-numpy.mean(rate) * times), (600, 1))

    assert survival.integrated_brier(
        duration, event, informed, times
    ) < survival.integrated_brier(duration, event, ignorant, times)


def test_heavy_censoring_does_not_break_the_weights():
    duration = numpy.array([1.0, 2.0, 3.0, 4.0, 5.0])
    event = numpy.array([1, 0, 0, 0, 0])

    probability = survival.censoring_probability(
        survival.censoring_curve(duration, event), duration
    )

    assert (probability >= survival.MIN_CENSORING_PROBABILITY).all()
    assert (probability <= 1.0).all()
