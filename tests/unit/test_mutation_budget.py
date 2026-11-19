"""The decision budget's refusals, word for word, and its one-step horizon."""

from fractions import Fraction

import pytest

from scmle.assurance.ledger import budget
from scmle.contracts import errors


def fresh(horizon: int | None = None) -> budget.Decision:
    return budget.Decision(Fraction(1, 20), Fraction(1, 20), horizon)


def test_a_horizon_of_one_gives_the_first_experiment_everything() -> None:
    assert budget.gammas(1, 1) == Fraction(1)
    assert budget.gammas(2, 1) == Fraction(0)


def test_a_horizon_below_one_is_refused_by_name() -> None:
    with pytest.raises(
        errors.InputError, match=r"^horizon must be at least 1$"
    ):
        fresh(horizon=0)


def test_a_blank_experiment_id_is_refused_by_name() -> None:
    with pytest.raises(
        errors.InputError, match=r"^experiment_id must not be empty$"
    ):
        budget.enroll(fresh(), " ")


def test_a_used_up_horizon_is_refused_word_for_word() -> None:
    first, level = budget.enroll(fresh(horizon=1), "a")
    assert level == Fraction(1, 20)
    with pytest.raises(
        budget.ExhaustedError, match=r"^the experiment horizon is used up$"
    ):
        budget.enroll(first, "b")
