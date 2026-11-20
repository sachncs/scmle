"""Tests for the paired Bayesian comparison."""

import numpy
import pytest

from scmle.evaluation import comparison


def test_a_consistent_gain_is_decisive():
    base = numpy.array([0.80, 0.81, 0.79, 0.80, 0.82])
    better = base + numpy.array([0.020, 0.021, 0.019, 0.020, 0.022])

    result = comparison.pair(better, base)

    assert result.superior > 0.99
    assert result.difference == pytest.approx(0.0204, abs=1e-4)


def test_noise_around_zero_is_not_decisive():
    first = numpy.array([0.80, 0.84, 0.78, 0.83, 0.79])
    second = numpy.array([0.82, 0.79, 0.81, 0.80, 0.82])

    result = comparison.pair(first, second)

    assert 0.1 < result.superior < 0.9


def test_the_probabilities_form_a_distribution():
    rng = numpy.random.default_rng(0)
    first, second = rng.normal(0.8, 0.02, 10), rng.normal(0.79, 0.02, 10)

    result = comparison.pair(first, second, rope=0.005)

    total = result.superior + result.equivalent + result.inferior
    assert total == pytest.approx(1.0)
    assert result.equivalent > 0.0


def test_a_rope_turns_a_tiny_steady_gain_into_equivalence():
    base = numpy.array([0.80, 0.81, 0.79, 0.80, 0.82])
    tiny = base + 0.001

    result = comparison.pair(tiny, base, rope=0.01)

    assert result.equivalent > 0.95


def test_swapping_the_candidates_swaps_the_probabilities():
    rng = numpy.random.default_rng(1)
    first, second = rng.normal(0.8, 0.02, 8), rng.normal(0.78, 0.02, 8)

    forward = comparison.pair(first, second)
    backward = comparison.pair(second, first)

    assert forward.superior == pytest.approx(backward.inferior)


def test_identical_scores_are_equivalent_and_bad_input_is_rejected():
    same = [0.8, 0.8, 0.8, 0.8]
    assert comparison.pair(same, same).equivalent == 1.0
    with pytest.raises(ValueError, match="same folds"):
        comparison.pair([0.1, 0.2, 0.3], [0.1, 0.2])
    with pytest.raises(ValueError, match="at least"):
        comparison.pair([0.1, 0.2], [0.1, 0.3])
