"""Error budget: exact boundaries, messages, signs and conservation.

Written against the mutmut survivors of `scmle/data/ledger/budget.py`; see
`MUTANTS_budget.md` for the one mutant that cannot be killed.
"""

from fractions import Fraction
import itertools
import math
import re

import pytest

from scmle.data.ledger import budget as b


def make_budget(**overrides):
    values = {"alpha_confirm": 0.05, "wealth_exploration": 0.1}
    values.update(overrides)
    return b.ErrorBudget.create(**values)


def message(text):
    return f"^{re.escape(text)}$"


# --- interval boundaries --------------------------------------------------


def test_exploration_budget_accepts_the_closed_unit_interval_ends():
    full = b.ExplorationBudget(initial=Fraction(1), wealth=Fraction(1))
    assert full.initial == 1
    empty = b.ExplorationBudget(initial=Fraction(0), wealth=Fraction(0))
    assert empty.wealth == 0


def test_exploration_budget_rejects_just_outside_the_unit_interval():
    with pytest.raises(
        ValueError, match=message("initial must be within [0, 1]")
    ):
        b.ExplorationBudget(Fraction(10**9 + 1, 10**9), Fraction(0))
    with pytest.raises(
        ValueError, match=message("initial must be within [0, 1]")
    ):
        b.ExplorationBudget(Fraction(-1, 10**9), Fraction(0))


def test_exploration_wealth_may_equal_initial_but_not_exceed_it_or_go_negative():
    assert b.ExplorationBudget(Fraction(1, 3), Fraction(1, 3)).wealth == (
        Fraction(1, 3)
    )
    text = "wealth must be between 0 and initial"
    with pytest.raises(ValueError, match=message(text)):
        b.ExplorationBudget(Fraction(1, 3), Fraction(1, 3) + Fraction(1, 10**9))
    with pytest.raises(ValueError, match=message(text)):
        b.ExplorationBudget(Fraction(1, 3), Fraction(-1, 10**9))


def test_confirmation_alpha_is_open_at_both_ends_and_tight_inside():
    text = "alpha_confirm must be strictly between 0 and 1"
    for bad in (
        Fraction(0),
        Fraction(1),
        Fraction(-1, 10**9),
        Fraction(10**9 + 1, 10**9),
    ):
        with pytest.raises(ValueError, match=message(text)):
            b.ConfirmationBudget(bad)
    assert b.ConfirmationBudget(Fraction(1, 10**9)).alpha_confirm > 0
    assert b.ConfirmationBudget(Fraction(10**9 - 1, 10**9)).alpha_confirm < 1


def test_create_accepts_wealth_of_exactly_one_and_zero():
    assert make_budget(wealth_exploration=1).exploration.wealth == 1
    assert make_budget(wealth_exploration=Fraction(1)).exploration.initial == 1
    assert make_budget(wealth_exploration=0).exploration.wealth == 0


def test_create_rejects_wealth_just_outside_with_the_exact_message():
    text = "wealth_exploration must be within [0, 1]"
    for bad in (Fraction(10**9 + 1, 10**9), Fraction(-1, 10**9)):
        with pytest.raises(ValueError, match=message(text)):
            make_budget(wealth_exploration=bad)


def test_create_names_the_offending_argument_for_non_finite_input():
    with pytest.raises(
        ValueError, match=r"^alpha_confirm must be a finite number$"
    ):
        make_budget(alpha_confirm=float("inf"))
    with pytest.raises(
        ValueError, match=r"^wealth_exploration must be a finite number$"
    ):
        make_budget(wealth_exploration=float("nan"))


# --- validation messages name the right field ----------------------------


def test_detector_tolerance_names_each_invalid_field():
    with pytest.raises(ValueError, match=r"^detector_id must not be empty$"):
        b.DetectorTolerance("  ", 0.1, 0.9)
    with pytest.raises(
        ValueError, match=r"^max_false_alarm_rate must be within"
    ):
        b.DetectorTolerance("d", 1.1, 0.9)
    with pytest.raises(
        ValueError, match=r"^min_recall_lower_bound must be within"
    ):
        b.DetectorTolerance("d", 0.1, -0.1)
    with pytest.raises(
        ValueError, match=r"^min_recall_lower_bound must be a finite"
    ):
        b.DetectorTolerance("d", 0.1, float("nan"))


def test_detector_tolerance_accepts_the_probability_ends():
    assert b.DetectorTolerance("d", 0.0, 1.0).min_recall_lower_bound == 1.0
    assert b.DetectorTolerance("d", 1.0, 0.0).max_false_alarm_rate == 1.0


def test_error_budget_validation_messages():
    with pytest.raises(
        ValueError, match=r"^permutations_b must be at least 20$"
    ):
        make_budget(permutations_b=b.MIN_PERMUTATIONS - 1)
    assert make_budget(permutations_b=b.MIN_PERMUTATIONS).permutations_b == 20
    good = make_budget()
    with pytest.raises(
        ValueError, match=r"^detector_tolerances must be a tuple$"
    ):
        b.ErrorBudget(
            good.confirmation,
            good.exploration,
            20,
            [b.DetectorTolerance("d", 0.1, 0.9)],  # type: ignore[arg-type]
        )
    tolerance = b.DetectorTolerance("d", 0.1, 0.9)
    with pytest.raises(
        ValueError, match=r"^duplicate detector_id in detector_tolerances$"
    ):
        make_budget(detector_tolerances=(tolerance, tolerance))


def test_from_dict_names_the_unparseable_field():
    for section, key in (
        ("confirmation", "alpha_confirm"),
        ("exploration", "initial"),
        ("exploration", "wealth"),
    ):
        record = make_budget().to_dict()
        record[section][key] = "oops"
        with pytest.raises(
            ValueError, match=f"^{key}: invalid fraction 'oops'$"
        ):
            b.ErrorBudget.from_dict(record)


# --- spend ----------------------------------------------------------------


def test_spending_zero_is_allowed_and_changes_nothing():
    budget = make_budget()
    assert b.spend(budget, 0) == budget
    assert b.spend(budget, 0.0) == budget


def test_spending_the_smallest_overdraft_is_refused():
    budget = make_budget()
    just_over = budget.exploration.wealth + Fraction(1, 10**12)
    with pytest.raises(b.BudgetExhaustedError):
        b.spend(budget, just_over)
    just_under = budget.exploration.wealth - Fraction(1, 10**12)
    assert b.spend(budget, just_under).exploration.wealth == Fraction(1, 10**12)


def test_spend_error_messages_name_the_amount_and_the_wealth():
    with pytest.raises(ValueError, match=r"^amount must be non-negative$"):
        b.spend(make_budget(), Fraction(-1, 10**9))
    with pytest.raises(ValueError, match=r"^amount must be a finite number$"):
        b.spend(make_budget(), float("inf"))
    with pytest.raises(
        b.BudgetExhaustedError,
        match=r"^amount 0.25 exceeds remaining wealth 0.1$",
    ):
        b.spend(make_budget(), 0.25)


def test_spend_keeps_the_initial_wealth_and_other_fields():
    budget = make_budget(
        permutations_b=77,
        detector_tolerances=(b.DetectorTolerance("d", 0.1, 0.9),),
    )
    spent = b.spend(budget, 0.04)
    assert spent.exploration.initial == budget.exploration.initial
    assert spent.permutations_b == 77
    assert spent.detector_tolerances == budget.detector_tolerances


def test_spend_conserves_wealth_and_is_order_independent():
    budget = make_budget(wealth_exploration=Fraction(1, 3))
    amounts = [Fraction(1, 10), Fraction(1, 7), Fraction(1, 50)]
    results = set()
    for order in itertools.permutations(amounts):
        current = budget
        for amount in order:
            current = b.spend(current, amount)
        results.add(current.exploration.wealth)
        assert (
            current.exploration.wealth + sum(amounts)
            == budget.exploration.wealth
        )
    assert results == {Fraction(1, 3) - sum(amounts)}


# --- split_budget ---------------------------------------------------------


def test_split_budget_error_messages():
    with pytest.raises(ValueError, match=r"^weights must not be empty$"):
        b.split_budget(make_budget(), {})
    with pytest.raises(ValueError, match=r"^family name must not be empty$"):
        b.split_budget(make_budget(), {" ": 1})
    with pytest.raises(
        ValueError, match=r"^weight\[x\] must be a finite number$"
    ):
        b.split_budget(make_budget(), {"x": float("nan")})
    for weight in (0, -1, Fraction(-1, 10**9)):
        with pytest.raises(ValueError, match=r"^weight\[x\] must be positive$"):
            b.split_budget(make_budget(), {"x": weight})


def test_split_budget_accepts_the_smallest_positive_weight():
    parts = b.split_budget(make_budget(), {"a": Fraction(1, 10**9), "b": 1})
    assert parts["a"].confirmation.alpha_confirm > 0


def test_split_budget_shares_are_exact_proportions_and_keep_order():
    budget = b.spend(make_budget(), Fraction(1, 100))
    parts = b.split_budget(budget, {"z": 1, "a": 3})
    assert list(parts) == ["z", "a"]
    assert parts["z"].confirmation.alpha_confirm == Fraction(1, 80)
    assert parts["a"].confirmation.alpha_confirm == Fraction(3, 80)
    assert parts["z"].exploration.initial == Fraction(1, 40)
    assert parts["a"].exploration.wealth == Fraction(3, 4) * Fraction(9, 100)
    for part in parts.values():
        assert part.exploration.wealth <= part.exploration.initial


def test_split_budget_does_not_depend_on_weight_scale_or_input_order():
    budget = make_budget()
    one = b.split_budget(budget, {"x": 1, "y": 2})
    scaled = b.split_budget(budget, {"y": 20, "x": 10})
    assert one["x"] == scaled["x"]
    assert one["y"] == scaled["y"]


# --- harmonic and confirmation_alpha -------------------------------------


def exact_harmonic(n):
    return math.fsum(1.0 / k for k in range(1, n + 1))


def test_harmonic_small_values_are_exact():
    assert b.harmonic(0) == 0.0
    assert b.harmonic(1) == 1.0
    assert b.harmonic(2) == 1.5


def test_harmonic_uses_the_exact_sum_at_the_limit():
    assert b.harmonic(b.EXACT_HARMONIC_LIMIT) == exact_harmonic(
        b.EXACT_HARMONIC_LIMIT
    )


def test_harmonic_asymptotic_series_matches_the_exact_sum_just_above():
    n = b.EXACT_HARMONIC_LIMIT + 1
    assert abs(b.harmonic(n) - exact_harmonic(n)) < 1e-13


def test_harmonic_asymptotic_series_has_the_right_correction_terms():
    n = 50_000
    value = b.harmonic(n)
    assert abs(value - exact_harmonic(n)) < 1e-13
    base = math.log(n) + b.EULER_GAMMA + 1 / (2 * n)
    assert value == pytest.approx(base - 1 / (12 * n * n), abs=1e-15, rel=0)


def test_confirmation_alpha_by_and_bonferroni_are_exact_at_rank_one():
    budget = make_budget()
    assert b.confirmation_alpha(budget, 1, b.MultiplicityRule.BY) == 0.05
    assert b.confirmation_alpha(budget, 1, b.MultiplicityRule.BH) == 0.05
    assert b.confirmation_alpha(budget, 2, b.MultiplicityRule.HOLM) == 0.025
    assert b.confirmation_alpha(
        budget, 2, b.MultiplicityRule.BY
    ) == pytest.approx(0.05 / 3)


def test_confirmation_alpha_never_exceeds_alpha_and_shrinks_with_survivors():
    budget = make_budget()
    for rule in b.MultiplicityRule:
        levels = [b.confirmation_alpha(budget, n, rule) for n in range(0, 12)]
        assert levels[0] == 0.05
        assert all(0 < x <= 0.05 for x in levels)
        assert levels == sorted(levels, reverse=True)


def test_confirmation_alpha_input_validation():
    budget = make_budget()
    with pytest.raises(ValueError, match=r"^n_survivors must be non-negative$"):
        b.confirmation_alpha(budget, -1, b.MultiplicityRule.BH)
    with pytest.raises(ValueError, match=r"^n_survivors must be an integer$"):
        b.confirmation_alpha(budget, 1.5, b.MultiplicityRule.BH)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"^rule must be a MultiplicityRule$"):
        b.confirmation_alpha(budget, 1, "bh")  # type: ignore[arg-type]
