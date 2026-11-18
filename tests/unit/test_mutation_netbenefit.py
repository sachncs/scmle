"""The net-benefit band's refusals and its longest positive run."""

import numpy
import pytest

from scmle.contracts import errors
from scmle.decision import netbenefit


def test_an_incumbent_of_another_length_is_refused_by_name() -> None:
    with pytest.raises(
        errors.InputError, match=r"^incumbent must have the length of y$"
    ):
        netbenefit.lowerband(
            [0, 1, 0, 1], [1, 1, 0, 0], (0.1, 0.3), incumbent=[1, 0]
        )


def test_the_longest_run_wins_over_an_earlier_shorter_one() -> None:
    grid = numpy.arange(6, dtype=float)
    positive = numpy.array([False, True, False, False, True, True])
    assert netbenefit.stretch(grid, positive) == (4.0, 5.0)


def test_a_second_look_is_refused_word_for_word() -> None:
    guard = netbenefit.FinalNetBenefit("abc")
    with pytest.raises(errors.InputError):
        guard.evaluate("abc", [0, 1], [0, 1], (0.0, 0.3))
    with pytest.raises(
        netbenefit.RepeatError,
        match=r"^the final table was already evaluated for this rule$",
    ):
        guard.evaluate("abc", [0, 1], [0, 1], (0.1, 0.3))
