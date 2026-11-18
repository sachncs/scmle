"""The missingness contrast's refusals, crossing unit and last reading."""

import numpy
import pytest

from scmle.contracts import errors
from scmle.decision import design
from scmle.decision import missing

LEVEL = 0.05


def expected(arm: list[int], prob: float) -> numpy.ndarray:
    """The log Bayes factor after each unit when every outcome is missing."""
    units = numpy.arange(1, len(arm) + 1)
    return design.srm_log_bayes_factor(numpy.cumsum(arm), units, prob)


def test_the_crossing_unit_is_the_first_one_beyond_the_bound() -> None:
    arm = [1] * 12 + [0, 1]
    found = missing.contrast(arm, [False] * len(arm), 0.5, LEVEL)
    trace = expected(arm, 0.5)
    crossed = numpy.flatnonzero(trace >= -numpy.log(LEVEL))
    assert found["first_unit"] == int(crossed[0]) + 1
    assert found["log_bayes_factor"] == pytest.approx(trace[-1])
    assert trace[-1] != pytest.approx(trace[-2])
    assert found["level"] == LEVEL


def test_no_missing_outcome_reads_a_zero_log_factor() -> None:
    found = missing.contrast([1, 0, 1], [True, True, True], 0.5, LEVEL)
    assert found["log_bayes_factor"] == 0.0
    assert found["first_unit"] is None


def test_unequal_shapes_are_refused() -> None:
    with pytest.raises(
        errors.InputError,
        match=r"^arm and observed must be one-dimensional and equal$",
    ):
        missing.contrast([1, 0], [True], 0.5, LEVEL)


def test_an_arm_other_than_zero_or_one_is_refused() -> None:
    with pytest.raises(errors.InputError, match=r"^arm must be 0 or 1$"):
        missing.contrast([2, 0], [True, False], 0.5, LEVEL)


@pytest.mark.parametrize(
    ("prob", "level"), [(0.0, LEVEL), (0.5, 0.0), (0.5, 1.0)]
)
def test_probabilities_on_the_boundary_are_refused(prob, level) -> None:
    with pytest.raises(
        errors.InputError,
        match=r"^assignment_prob and level must lie in \(0, 1\)$",
    ):
        missing.contrast([1, 0], [True, False], prob, level)
