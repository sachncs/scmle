"""Escalation edges: inclusive thresholds, open levels, numeric input."""

import numpy
import pytest

from scmle.contracts import errors
from scmle.governance import escalation


def test_a_score_at_the_threshold_is_answered() -> None:
    curve = escalation.sweep([1.0, 0.5], [0.0, 1.0], points=3)
    numpy.testing.assert_array_equal(curve.coverage, [0.5, 1.0, 1.0])


def test_a_level_of_one_is_refused() -> None:
    with pytest.raises(
        errors.InputError,
        match=r"^r_star and delta must be strictly between 0 and 1$",
    ):
        escalation.select([0.5, 0.6], [0, 1], 0.1, 1.0)


def test_confidences_given_as_text_are_read_as_numbers() -> None:
    found = escalation.adjudicate(["0.9", "0.2"], [0.9, 0.2], 0.5, 1.0, 2.0)
    assert (found.cut, found.disagreement, found.consistent) == (
        0.5,
        0.0,
        True,
    )
