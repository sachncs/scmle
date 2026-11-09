"""DecisionBudget: validation, assignment, rollback monitors, round trip."""

from fractions import Fraction

import pytest

import scmle.data.ledger.budget


def fresh(**overrides):
    values = {
        "alpha_promote": Fraction(1, 50),
        "alpha_rollback": Fraction(1, 100),
    }
    values.update(overrides)
    return scmle.data.ledger.budget.DecisionBudget(**values)


def test_the_default_levels_follow_the_discount_exactly():
    budget = fresh()
    budget, first = scmle.data.ledger.budget.assign_experiment(budget, "a")
    budget, second = scmle.data.ledger.budget.assign_experiment(budget, "b")
    assert first == Fraction(1, 100)
    assert second == Fraction(1, 300)
    assert budget.spent == Fraction(4, 300)
    assert budget.remaining == Fraction(1, 50) - Fraction(4, 300)


def test_an_explicit_level_may_not_overspend():
    budget = fresh()
    budget, level = scmle.data.ledger.budget.assign_experiment(
        budget, "a", Fraction(1, 100)
    )
    assert level == Fraction(1, 100)
    with pytest.raises(scmle.data.ledger.budget.BudgetExhaustedError):
        scmle.data.ledger.budget.assign_experiment(budget, "b", Fraction(1, 99))
    budget, ignored = scmle.data.ledger.budget.assign_experiment(
        budget, "b", Fraction(1, 100)
    )
    assert budget.remaining == 0


def test_an_experiment_id_is_used_once_and_a_level_is_positive():
    budget, ignored = scmle.data.ledger.budget.assign_experiment(fresh(), "a")
    with pytest.raises(ValueError, match="already has a level"):
        scmle.data.ledger.budget.assign_experiment(budget, "a")
    with pytest.raises(ValueError, match="positive"):
        scmle.data.ledger.budget.assign_experiment(budget, "b", 0)
    with pytest.raises(ValueError):
        scmle.data.ledger.budget.assign_experiment(budget, " ")


def test_the_original_budget_is_not_modified():
    budget = fresh()
    scmle.data.ledger.budget.assign_experiment(budget, "a")
    assert budget.assigned == ()


def test_the_rollback_level_is_per_champion_and_leaves_promotion_alone():
    budget = fresh()
    budget, level = scmle.data.ledger.budget.start_rollback_monitor(
        budget, "v2"
    )
    budget, again = scmle.data.ledger.budget.start_rollback_monitor(
        budget, "v3"
    )
    assert level == again == Fraction(1, 100)
    assert budget.remaining == Fraction(1, 50)
    with pytest.raises(ValueError, match="already monitored"):
        scmle.data.ledger.budget.start_rollback_monitor(budget, "v2")


@pytest.mark.parametrize(
    "overrides",
    [
        {"alpha_promote": Fraction(0)},
        {"alpha_promote": Fraction(1)},
        {"alpha_rollback": Fraction(3, 2)},
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
    budget = scmle.data.ledger.budget.DecisionBudget.create(0.02, 0.01)
    assert budget.alpha_promote == Fraction(1, 50)
    record = budget.to_dict()
    assert scmle.data.ledger.budget.DecisionBudget.from_dict(record) == budget
    with pytest.raises(ValueError, match="unexpected"):
        scmle.data.ledger.budget.DecisionBudget.from_dict(
            {**record, "extra": 1}
        )


def test_discount_weights():
    assert scmle.data.ledger.budget.discount_weight(1, None) == Fraction(1, 2)
    assert scmle.data.ledger.budget.discount_weight(3, 4) == Fraction(1, 4)
    assert scmle.data.ledger.budget.discount_weight(5, 4) == 0
    with pytest.raises(ValueError):
        scmle.data.ledger.budget.discount_weight(0, None)
