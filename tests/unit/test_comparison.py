"""Tests for the paired Bayesian comparison."""

import numpy as np
import pytest

from scml.evaluation import comparison


def test_a_consistent_gain_is_decisive():
    base = np.array([0.80, 0.81, 0.79, 0.80, 0.82])
    better = base + np.array([0.020, 0.021, 0.019, 0.020, 0.022])

    result = comparison.compare_paired(better, base)

    assert result.probability_first_better > 0.99
    assert result.decisive(0.95) and result.mean_difference == pytest.approx(
        0.0204, abs=1e-4
    )


def test_noise_around_zero_is_not_decisive():
    first = np.array([0.80, 0.84, 0.78, 0.83, 0.79])
    second = np.array([0.82, 0.79, 0.81, 0.80, 0.82])

    result = comparison.compare_paired(first, second)

    assert not result.decisive(0.95)
    assert 0.1 < result.probability_first_better < 0.9


def test_the_probabilities_form_a_distribution():
    rng = np.random.default_rng(0)
    first, second = rng.normal(0.8, 0.02, 10), rng.normal(0.79, 0.02, 10)

    result = comparison.compare_paired(first, second, rope=0.005)

    total = (
        result.probability_first_better
        + result.probability_equivalent
        + result.probability_second_better
    )
    assert total == pytest.approx(1.0)
    assert result.probability_equivalent > 0.0


def test_a_rope_turns_a_tiny_steady_gain_into_equivalence():
    base = np.array([0.80, 0.81, 0.79, 0.80, 0.82])
    tiny = base + 0.001

    result = comparison.compare_paired(tiny, base, rope=0.01)

    assert result.probability_equivalent > 0.95


def test_swapping_the_candidates_swaps_the_probabilities():
    rng = np.random.default_rng(1)
    first, second = rng.normal(0.8, 0.02, 8), rng.normal(0.78, 0.02, 8)

    forward = comparison.compare_paired(first, second)
    backward = comparison.compare_paired(second, first)

    assert forward.probability_first_better == pytest.approx(
        backward.probability_second_better
    )


def test_identical_scores_are_equivalent_and_bad_input_is_rejected():
    same = [0.8, 0.8, 0.8, 0.8]
    assert comparison.compare_paired(same, same).probability_equivalent == 1.0
    with pytest.raises(ValueError, match="same folds"):
        comparison.compare_paired([0.1, 0.2, 0.3], [0.1, 0.2])
    with pytest.raises(ValueError, match="at least"):
        comparison.compare_paired([0.1, 0.2], [0.1, 0.3])
