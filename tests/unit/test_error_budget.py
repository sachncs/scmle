"""Error budget: validation, immutability, splitting and serialization."""

from fractions import Fraction
import json

import pytest

import scmle.data.ledger.budget


def make_budget(**overrides):
    values = {"alpha_confirm": 0.05, "wealth_exploration": 0.1}
    values.update(overrides)
    return scmle.data.ledger.budget.ErrorBudget.create(**values)


def test_create_reads_decimals_exactly_as_written():
    budget = make_budget()
    assert budget.confirmation.alpha_confirm == Fraction(1, 20)
    assert budget.exploration.wealth == Fraction(1, 10)
    assert budget.exploration.initial == Fraction(1, 10)
    assert budget.permutations_b == scmle.data.ledger.budget.MIN_PERMUTATIONS


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, float("nan"), True])
def test_alpha_confirm_must_lie_strictly_inside_the_unit_interval(alpha):
    with pytest.raises(ValueError, match="alpha_confirm"):
        make_budget(alpha_confirm=alpha)


@pytest.mark.parametrize("wealth", [-0.1, 1.5, float("inf")])
def test_wealth_must_lie_in_the_unit_interval(wealth):
    with pytest.raises(ValueError, match="wealth"):
        make_budget(wealth_exploration=wealth)


def test_zero_wealth_means_no_exploration_budget():
    budget = make_budget(wealth_exploration=0.0)
    assert scmle.data.ledger.budget.remaining(budget) == 0.0


def test_permutation_count_has_a_floor():
    with pytest.raises(ValueError, match="permutations_b"):
        make_budget(
            permutations_b=scmle.data.ledger.budget.MIN_PERMUTATIONS - 1
        )
    assert make_budget(permutations_b=100).permutations_b == 100


def test_spending_returns_a_new_budget_and_never_mutates():
    budget = make_budget()
    spent = scmle.data.ledger.budget.spend(budget, 0.04)
    assert scmle.data.ledger.budget.remaining(budget) == pytest.approx(0.1)
    assert scmle.data.ledger.budget.remaining(spent) == pytest.approx(0.06)
    assert spent.confirmation is budget.confirmation
    with pytest.raises(AttributeError):
        budget.permutations_b = 5  # type: ignore[misc]


def test_overspending_is_refused_and_leaves_the_budget_intact():
    budget = make_budget()
    with pytest.raises(
        scmle.data.ledger.budget.BudgetExhaustedError, match="exceeds"
    ):
        scmle.data.ledger.budget.spend(budget, 0.11)
    assert scmle.data.ledger.budget.remaining(budget) == pytest.approx(0.1)


def test_spending_everything_is_allowed_and_leaves_exactly_zero():
    spent = scmle.data.ledger.budget.spend(make_budget(), 0.1)
    assert spent.exploration.wealth == 0


@pytest.mark.parametrize("amount", [-0.01, float("nan"), True])
def test_spend_rejects_invalid_amounts(amount):
    with pytest.raises(ValueError, match="amount"):
        scmle.data.ledger.budget.spend(make_budget(), amount)


def test_exploration_and_confirmation_budgets_are_separate_types():
    budget = make_budget()
    assert not isinstance(budget.exploration, type(budget.confirmation))
    assert not isinstance(budget.confirmation, type(budget.exploration))
    assert not hasattr(budget, "total")
    assert not hasattr(budget.exploration, "__add__")
    assert not hasattr(budget.confirmation, "__add__")


def test_exploration_budget_rejects_inconsistent_state():
    with pytest.raises(ValueError, match="wealth"):
        scmle.data.ledger.budget.ExplorationBudget(
            initial=Fraction(1, 10), wealth=Fraction(1, 5)
        )
    with pytest.raises(ValueError, match="initial"):
        scmle.data.ledger.budget.ExplorationBudget(
            initial=Fraction(3, 2), wealth=Fraction(1, 2)
        )


def test_split_budget_shares_sum_exactly_to_the_totals():
    budget = scmle.data.ledger.budget.spend(make_budget(), 0.03)
    parts = scmle.data.ledger.budget.split_budget(
        budget, {"x": 1, "y": 2, "z": 0.5}
    )
    assert set(parts) == {"x", "y", "z"}
    assert sum(p.confirmation.alpha_confirm for p in parts.values()) == (
        budget.confirmation.alpha_confirm
    )
    assert sum(p.exploration.wealth for p in parts.values()) == (
        budget.exploration.wealth
    )
    assert sum(p.exploration.initial for p in parts.values()) == (
        budget.exploration.initial
    )
    assert all(
        p.permutations_b == budget.permutations_b for p in parts.values()
    )


@pytest.mark.parametrize(
    "weights", [{}, {"x": 0}, {"x": -1}, {"x": float("nan")}, {" ": 1}]
)
def test_split_budget_rejects_bad_weights(weights):
    with pytest.raises(ValueError):
        scmle.data.ledger.budget.split_budget(make_budget(), weights)


def test_confirmation_alpha_for_each_rule():
    budget = make_budget()
    assert scmle.data.ledger.budget.confirmation_alpha(
        budget, 10, scmle.data.ledger.budget.Correction.BONFERRONI
    ) == (pytest.approx(0.005))
    assert scmle.data.ledger.budget.confirmation_alpha(
        budget, 10, scmle.data.ledger.budget.Correction.HOLM
    ) == (pytest.approx(0.005))
    assert scmle.data.ledger.budget.confirmation_alpha(
        budget, 10, scmle.data.ledger.budget.Correction.BH
    ) == (pytest.approx(0.005))
    harmonic = sum(1 / k for k in range(1, 11))
    assert scmle.data.ledger.budget.confirmation_alpha(
        budget, 10, scmle.data.ledger.budget.Correction.BY
    ) == (pytest.approx(0.05 / (10 * harmonic)))


def test_confirmation_alpha_with_no_survivors_is_the_full_alpha():
    assert scmle.data.ledger.budget.confirmation_alpha(
        make_budget(), 0, scmle.data.ledger.budget.Correction.BH
    ) == pytest.approx(0.05)


def test_confirmation_alpha_rejects_negative_counts():
    with pytest.raises(ValueError, match="n_survivors"):
        scmle.data.ledger.budget.confirmation_alpha(
            make_budget(), -1, scmle.data.ledger.budget.Correction.BH
        )


def test_detector_tolerances_validate_and_round_trip():
    tolerance = scmle.data.ledger.budget.DetectorTolerance(
        "label_noise", 0.05, 0.8
    )
    budget = make_budget(detector_tolerances=(tolerance,))
    assert budget.detector_tolerances == (tolerance,)
    with pytest.raises(ValueError, match="max_false_alarm_rate"):
        scmle.data.ledger.budget.DetectorTolerance("x", 1.5, 0.8)
    with pytest.raises(ValueError, match="duplicate"):
        make_budget(detector_tolerances=(tolerance, tolerance))


def test_json_round_trip_is_exact():
    budget = scmle.data.ledger.budget.spend(
        make_budget(
            detector_tolerances=(
                scmle.data.ledger.budget.DetectorTolerance("d", 0.05, 0.8),
            ),
            permutations_b=50,
        ),
        0.0123,
    )
    record = json.loads(json.dumps(budget.to_dict()))
    assert scmle.data.ledger.budget.ErrorBudget.from_dict(record) == budget


def test_loading_rejects_unknown_fields_and_bad_fractions():
    record = make_budget().to_dict()
    with pytest.raises(ValueError, match="unexpected"):
        scmle.data.ledger.budget.ErrorBudget.from_dict({**record, "extra": 1})
    record["confirmation"]["alpha_confirm"] = "not-a-fraction"
    with pytest.raises(ValueError, match="alpha_confirm"):
        scmle.data.ledger.budget.ErrorBudget.from_dict(record)
