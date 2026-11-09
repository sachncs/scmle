"""Properties the error-budget algebra must keep for every input."""

from fractions import Fraction
import itertools

from hypothesis import given
from hypothesis import HealthCheck
from hypothesis import settings
from hypothesis import strategies
import pytest

from scmle.data.ledger import budget as b

SETTINGS = settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)

alphas = strategies.fractions(
    min_value=Fraction(1, 1000), max_value=Fraction(1, 2), max_denominator=1000
)
wealths = strategies.fractions(min_value=0, max_value=1, max_denominator=1000)
amounts = strategies.fractions(min_value=0, max_value=1, max_denominator=1000)
rules = strategies.sampled_from(list(b.MultiplicityRule))


def budget_from(alpha, wealth):
    return b.ErrorBudget(
        confirmation=b.ConfirmationBudget(alpha),
        exploration=b.ExplorationBudget(initial=wealth, wealth=wealth),
        permutations_b=b.MIN_PERMUTATIONS,
        detector_tolerances=(),
    )


@SETTINGS
@given(alphas, wealths, strategies.lists(amounts, max_size=12))
def test_wealth_never_goes_negative_and_spending_is_conserved(
    alpha, wealth, attempts
):
    budget = budget_from(alpha, wealth)
    accepted = Fraction(0)
    for amount in attempts:
        try:
            budget = b.spend(budget, amount)
        except b.BudgetExhaustedError:
            assert amount > budget.exploration.wealth
            continue
        accepted += amount
        assert budget.exploration.wealth >= 0
    assert budget.exploration.wealth + accepted == wealth
    assert budget.exploration.wealth <= budget.exploration.initial


@SETTINGS
@given(alphas, strategies.lists(amounts, min_size=1, max_size=5))
def test_spend_is_order_independent(alpha, parts):
    total = sum(parts, Fraction(0))
    if total > 1:
        return
    results = set()
    for order in itertools.permutations(parts):
        budget = budget_from(alpha, Fraction(1))
        for amount in order:
            budget = b.spend(budget, amount)
        results.add(budget)
    assert len(results) == 1


@SETTINGS
@given(alphas, wealths, amounts)
def test_exploration_spending_never_touches_the_confirmation_budget(
    alpha, wealth, amount
):
    budget = budget_from(alpha, wealth)
    before = [
        b.confirmation_alpha(budget, n, r)
        for n in (1, 5)
        for r in b.MultiplicityRule
    ]
    if amount <= wealth:
        budget = b.spend(budget, amount)
    after = [
        b.confirmation_alpha(budget, n, r)
        for n in (1, 5)
        for r in b.MultiplicityRule
    ]
    assert before == after
    assert budget.confirmation.alpha_confirm == alpha


@SETTINGS
@given(
    alphas,
    wealths,
    strategies.dictionaries(
        strategies.text(alphabet="abcdefgh", min_size=1, max_size=3),
        strategies.fractions(
            min_value=Fraction(1, 100), max_value=10, max_denominator=100
        ),
        min_size=1,
        max_size=6,
    ),
)
def test_split_shares_sum_exactly_and_stay_valid(alpha, wealth, weights):
    budget = budget_from(alpha, wealth)
    parts = b.split_budget(budget, weights)
    assert sum(p.confirmation.alpha_confirm for p in parts.values()) == alpha
    assert sum(p.exploration.initial for p in parts.values()) == wealth
    assert all(
        p.exploration.wealth <= p.exploration.initial for p in parts.values()
    )


@SETTINGS
@given(alphas, strategies.integers(min_value=1, max_value=500), rules)
def test_confirmation_alpha_is_bounded_and_decreasing_in_n(alpha, n, rule):
    budget = budget_from(alpha, Fraction(1, 10))
    here = b.confirmation_alpha(budget, n, rule)
    more = b.confirmation_alpha(budget, n + 1, rule)
    assert 0 < more <= here <= float(alpha)


@SETTINGS
@given(alphas, strategies.integers(min_value=1, max_value=500))
def test_bonferroni_levels_add_back_to_alpha(alpha, n):
    budget = budget_from(alpha, Fraction(1, 10))
    level = b.confirmation_alpha(budget, n, b.MultiplicityRule.BONFERRONI)
    assert level * n == pytest.approx(float(alpha), rel=1e-12)


@SETTINGS
@given(alphas, strategies.integers(min_value=1, max_value=500))
def test_benjamini_yekutieli_is_never_looser_than_benjamini_hochberg(alpha, n):
    budget = budget_from(alpha, Fraction(1, 10))
    bh = b.confirmation_alpha(budget, n, b.MultiplicityRule.BH)
    by = b.confirmation_alpha(budget, n, b.MultiplicityRule.BY)
    assert by <= bh


@SETTINGS
@given(alphas, wealths, strategies.lists(amounts, max_size=6))
def test_serialization_round_trips_exactly(alpha, wealth, attempts):
    budget = budget_from(alpha, wealth)
    for amount in attempts:
        if amount <= budget.exploration.wealth:
            budget = b.spend(budget, amount)
    assert b.ErrorBudget.from_dict(budget.to_dict()) == budget
