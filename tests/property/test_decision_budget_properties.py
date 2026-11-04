"""Properties of the decision budget (B5): exact levels, never overspent."""

from fractions import Fraction

from hypothesis import given
from hypothesis import HealthCheck
from hypothesis import settings
from hypothesis import strategies as st
import pytest

from scmle.data.ledger import budget as b

SETTINGS = settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
alphas = st.fractions(
    min_value=Fraction(1, 1000), max_value=Fraction(1, 2), max_denominator=1000
)
horizons = st.one_of(st.none(), st.integers(min_value=1, max_value=40))


@SETTINGS
@given(alphas, horizons, st.integers(min_value=0, max_value=60))
def test_assigned_levels_never_exceed_the_promotion_level(
    alpha, horizon, count
):
    budget = b.DecisionBudget(alpha, Fraction(1, 100), horizon)
    for index in range(count):
        try:
            budget, level = b.assign_experiment(budget, f"e{index}")
        except b.BudgetExhaustedError:
            break
        assert level > 0
        assert budget.spent <= alpha
        assert budget.spent + budget.remaining == alpha


@SETTINGS
@given(alphas, st.integers(min_value=1, max_value=50))
def test_the_open_ended_discount_sums_to_a_closed_form(alpha, count):
    budget = b.DecisionBudget(alpha, Fraction(1, 100))
    for index in range(count):
        budget, ignored = b.assign_experiment(budget, f"e{index}")
    assert budget.spent == alpha * Fraction(count, count + 1)


@SETTINGS
@given(alphas, st.integers(min_value=1, max_value=30))
def test_a_round_trip_keeps_every_level_bit_for_bit(alpha, count):
    budget = b.DecisionBudget(alpha, Fraction(1, 100), horizon=30)
    for index in range(count):
        budget, ignored = b.assign_experiment(budget, f"e{index}")
    budget, ignored = b.start_rollback_monitor(budget, "champion")
    assert b.DecisionBudget.from_dict(budget.to_dict()) == budget


def test_a_horizon_budget_is_exhausted_after_horizon_experiments():
    budget = b.DecisionBudget(Fraction(1, 50), Fraction(1, 100), horizon=3)
    for index in range(3):
        budget, ignored = b.assign_experiment(budget, f"e{index}")
    assert budget.spent == Fraction(1, 50)
    with pytest.raises(b.BudgetExhaustedError):
        b.assign_experiment(budget, "e3")
