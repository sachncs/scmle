"""Anytime-valid check that outcomes go missing alike in both arms."""

import numpy
import pytest

from scmle.decision import missing


def stream(seed, n, q, miss_control, miss_challenger):
    rng = numpy.random.default_rng(seed)
    arm = (rng.uniform(size=n) < q).astype(float)
    gone = rng.uniform(size=n) < numpy.where(
        arm == 1, miss_challenger, miss_control
    )
    return arm, ~gone


def test_equal_missingness_is_flagged_at_most_at_the_level_however_often_read():
    flags = [
        missing.differential_missingness(
            *stream(seed, 600, 0.5, 0.2, 0.2), assignment_prob=0.5, level=0.05
        )["differential"]
        for seed in range(600)
    ]
    # Ville: the probability of ever crossing is at most the level; the
    # observed rate over 600 streams stays within three standard errors
    assert numpy.mean(flags) <= 0.05 + 3 * numpy.sqrt(0.05 * 0.95 / 600)


def test_unequal_missingness_is_detected():
    found = [
        missing.differential_missingness(
            *stream(seed, 600, 0.5, 0.1, 0.4), assignment_prob=0.5, level=0.05
        )["differential"]
        for seed in range(100)
    ]
    assert numpy.mean(found) >= 0.95


def test_unequal_assignment_probability_is_respected():
    arm, observed = stream(1, 900, 0.2, 0.25, 0.25)
    result = missing.differential_missingness(
        arm, observed, assignment_prob=0.2, level=0.05
    )
    assert result["differential"] is False
    assert result["missing_units"] == int((~observed).sum())


def test_the_result_states_what_it_does_and_does_not_show():
    arm, observed = stream(2, 300, 0.5, 0.1, 0.5)
    result = missing.differential_missingness(arm, observed, 0.5, 0.05)
    assert result["status"] == "described"
    assert result["label"] == "missingness_not_handled"
    assert "log_bayes_factor" in result and "first_unit" in result
    assert "does not handle" in result["note"]


def test_nothing_missing_is_not_differential():
    arm = numpy.array([0.0, 1.0, 1.0, 0.0])
    result = missing.differential_missingness(
        arm, numpy.ones(4, dtype=bool), 0.5, 0.05
    )
    assert result["differential"] is False and result["missing_units"] == 0
    assert result["first_unit"] is None


def test_inputs_are_checked():
    with pytest.raises(ValueError):
        missing.differential_missingness([0, 1], [True], 0.5, 0.05)
    with pytest.raises(ValueError):
        missing.differential_missingness([0, 2], [True, True], 0.5, 0.05)
    with pytest.raises(ValueError):
        missing.differential_missingness([0, 1], [True, True], 1.5, 0.05)
    with pytest.raises(ValueError):
        missing.differential_missingness([0, 1], [True, True], 0.5, 1.5)
