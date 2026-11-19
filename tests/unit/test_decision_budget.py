"""DecisionBudget: validation, assignment, rollback monitors, round trip."""

from fractions import Fraction

import pytest

import scmle.assurance.ledger.budget


def fresh(**overrides):
    values = {
        "promotion": Fraction(1, 50),
        "rollback": Fraction(1, 100),
    }
    values.update(overrides)
    return scmle.assurance.ledger.budget.Decision(**values)


def test_the_default_levels_follow_the_discount_exactly():
    budget = fresh()
    budget, first = scmle.assurance.ledger.budget.enroll(budget, "a")
    budget, second = scmle.assurance.ledger.budget.enroll(budget, "b")
    assert first == Fraction(1, 100)
    assert second == Fraction(1, 300)
    assert budget.spent == Fraction(4, 300)
    assert budget.remaining == Fraction(1, 50) - Fraction(4, 300)


def test_an_explicit_level_may_not_overspend():
    budget = fresh()
    budget, level = scmle.assurance.ledger.budget.enroll(
        budget, "a", Fraction(1, 100)
    )
    assert level == Fraction(1, 100)
    with pytest.raises(scmle.assurance.ledger.budget.ExhaustedError):
        scmle.assurance.ledger.budget.enroll(budget, "b", Fraction(1, 99))
    budget, ignored = scmle.assurance.ledger.budget.enroll(
        budget, "b", Fraction(1, 100)
    )
    assert budget.remaining == 0


def test_an_experiment_id_is_used_once_and_a_level_is_positive():
    budget, ignored = scmle.assurance.ledger.budget.enroll(fresh(), "a")
    with pytest.raises(ValueError, match="already has a level"):
        scmle.assurance.ledger.budget.enroll(budget, "a")
    with pytest.raises(ValueError, match="positive"):
        scmle.assurance.ledger.budget.enroll(budget, "b", 0)
    with pytest.raises(ValueError):
        scmle.assurance.ledger.budget.enroll(budget, " ")


def test_the_original_budget_is_not_modified():
    budget = fresh()
    scmle.assurance.ledger.budget.enroll(budget, "a")
    assert budget.assigned == ()


def test_the_rollback_level_is_per_champion_and_leaves_promotion_alone():
    budget = fresh()
    budget, level = scmle.assurance.ledger.budget.watch(budget, "v2")
    budget, again = scmle.assurance.ledger.budget.watch(budget, "v3")
    assert level == again == Fraction(1, 100)
    assert budget.remaining == Fraction(1, 50)
    with pytest.raises(ValueError, match="already monitored"):
        scmle.assurance.ledger.budget.watch(budget, "v2")


@pytest.mark.parametrize(
    "overrides",
    [
        {"promotion": Fraction(0)},
        {"promotion": Fraction(1)},
        {"rollback": Fraction(3, 2)},
        {"assigned": (("a", Fraction(1, 2)),)},
        {"assigned": (("a", Fraction(1, 100)), ("a", Fraction(1, 100)))},
        {"assigned": (("a", Fraction(-1, 100)),)},
        {"monitored": ("v", "v")},
        {"horizon": 0},
    ],
)
def test_invalid_budgets_are_refused(overrides):
    with pytest.raises(ValueError):
        fresh(**overrides)


def test_floats_are_read_as_written_and_records_reject_extras():
    budget = scmle.assurance.ledger.budget.Decision.create(0.02, 0.01)
    assert budget.promotion == Fraction(1, 50)
    record = budget.to_dict()
    assert scmle.assurance.ledger.budget.Decision.from_dict(record) == budget
    with pytest.raises(ValueError, match="unexpected"):
        scmle.assurance.ledger.budget.Decision.from_dict({**record, "extra": 1})


def test_discount_weights():
    assert scmle.assurance.ledger.budget.gammas(1, None) == Fraction(1, 2)
    assert scmle.assurance.ledger.budget.gammas(3, 4) == Fraction(1, 4)
    assert scmle.assurance.ledger.budget.gammas(5, 4) == 0
    with pytest.raises(ValueError):
        scmle.assurance.ledger.budget.gammas(0, None)
