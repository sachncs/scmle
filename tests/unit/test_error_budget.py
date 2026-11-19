"""Error budget: validation, immutability, splitting and serialization."""

from fractions import Fraction
import json

import pytest

import scmle.assurance.ledger.budget


def make_budget(**overrides):
    values = {"alpha_confirm": 0.05, "wealth_exploration": 0.1}
    values.update(overrides)
    return scmle.assurance.ledger.budget.Alpha.create(**values)


def test_create_reads_decimals_exactly_as_written():
    budget = make_budget()
    assert budget.confirmation.alpha_confirm == Fraction(1, 20)
    assert budget.exploration.wealth == Fraction(1, 10)
    assert budget.exploration.initial == Fraction(1, 10)
    assert budget.permutations == scmle.assurance.ledger.budget.MIN_PERMUTATIONS


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
    assert scmle.assurance.ledger.budget.remaining(budget) == 0.0


def test_permutation_count_has_a_floor():
    with pytest.raises(ValueError, match="permutations"):
        make_budget(
            permutations=scmle.assurance.ledger.budget.MIN_PERMUTATIONS - 1
        )
    assert make_budget(permutations=100).permutations == 100


def test_spending_returns_a_new_budget_and_never_mutates():
    budget = make_budget()
    spent = scmle.assurance.ledger.budget.spend(budget, 0.04)
    assert scmle.assurance.ledger.budget.remaining(budget) == pytest.approx(0.1)
    assert scmle.assurance.ledger.budget.remaining(spent) == pytest.approx(0.06)
    assert spent.confirmation is budget.confirmation
    with pytest.raises(AttributeError):
        budget.permutations = 5  # type: ignore[misc]


def test_overspending_is_refused_and_leaves_the_budget_intact():
    budget = make_budget()
    with pytest.raises(
        scmle.assurance.ledger.budget.ExhaustedError, match="exceeds"
    ):
        scmle.assurance.ledger.budget.spend(budget, 0.11)
    assert scmle.assurance.ledger.budget.remaining(budget) == pytest.approx(0.1)


def test_spending_everything_is_allowed_and_leaves_exactly_zero():
    spent = scmle.assurance.ledger.budget.spend(make_budget(), 0.1)
    assert spent.exploration.wealth == 0


@pytest.mark.parametrize("amount", [-0.01, float("nan"), True])
def test_spend_rejects_invalid_amounts(amount):
    with pytest.raises(ValueError, match="amount"):
        scmle.assurance.ledger.budget.spend(make_budget(), amount)


def test_exploration_and_confirmation_budgets_are_separate_types():
    budget = make_budget()
    assert not isinstance(budget.exploration, type(budget.confirmation))
    assert not isinstance(budget.confirmation, type(budget.exploration))
    assert not hasattr(budget, "total")
    assert not hasattr(budget.exploration, "__add__")
    assert not hasattr(budget.confirmation, "__add__")


def test_exploration_budget_rejects_inconsistent_state():
    with pytest.raises(ValueError, match="wealth"):
        scmle.assurance.ledger.budget.Exploration(
            initial=Fraction(1, 10), wealth=Fraction(1, 5)
        )
    with pytest.raises(ValueError, match="initial"):
        scmle.assurance.ledger.budget.Exploration(
            initial=Fraction(3, 2), wealth=Fraction(1, 2)
        )


def test_split_budget_shares_sum_exactly_to_the_totals():
    budget = scmle.assurance.ledger.budget.spend(make_budget(), 0.03)
    parts = scmle.assurance.ledger.budget.apportion(
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
    assert all(p.permutations == budget.permutations for p in parts.values())


@pytest.mark.parametrize(
    "weights", [{}, {"x": 0}, {"x": -1}, {"x": float("nan")}, {" ": 1}]
)
def test_split_budget_rejects_bad_weights(weights):
    with pytest.raises(ValueError):
        scmle.assurance.ledger.budget.apportion(make_budget(), weights)


def test_confirmation_alpha_for_each_rule():
    budget = make_budget()
    assert scmle.assurance.ledger.budget.allot(
        budget, 10, scmle.assurance.ledger.budget.Correction.BONFERRONI
    ) == (pytest.approx(0.005))
    assert scmle.assurance.ledger.budget.allot(
        budget, 10, scmle.assurance.ledger.budget.Correction.HOLM
    ) == (pytest.approx(0.005))
    assert scmle.assurance.ledger.budget.allot(
        budget, 10, scmle.assurance.ledger.budget.Correction.BH
    ) == (pytest.approx(0.005))
    harmonic = sum(1 / k for k in range(1, 11))
    assert scmle.assurance.ledger.budget.allot(
        budget, 10, scmle.assurance.ledger.budget.Correction.BY
    ) == (pytest.approx(0.05 / (10 * harmonic)))


def test_confirmation_alpha_with_no_survivors_is_the_full_alpha():
    assert scmle.assurance.ledger.budget.allot(
        make_budget(), 0, scmle.assurance.ledger.budget.Correction.BH
    ) == pytest.approx(0.05)


def test_confirmation_alpha_rejects_negative_counts():
    with pytest.raises(ValueError, match="n_survivors"):
        scmle.assurance.ledger.budget.allot(
            make_budget(), -1, scmle.assurance.ledger.budget.Correction.BH
        )


def test_detector_tolerances_validate_and_round_trip():
    tolerance = scmle.assurance.ledger.budget.Tolerance(
        "label_noise", 0.05, 0.8
    )
    budget = make_budget(tolerances=(tolerance,))
    assert budget.tolerances == (tolerance,)
    with pytest.raises(ValueError, match="alarms"):
        scmle.assurance.ledger.budget.Tolerance("x", 1.5, 0.8)
    with pytest.raises(ValueError, match="duplicate"):
        make_budget(tolerances=(tolerance, tolerance))


def test_json_round_trip_is_exact():
    budget = scmle.assurance.ledger.budget.spend(
        make_budget(
            tolerances=(
                scmle.assurance.ledger.budget.Tolerance("d", 0.05, 0.8),
            ),
            permutations=50,
        ),
        0.0123,
    )
    record = json.loads(json.dumps(budget.to_dict()))
    assert scmle.assurance.ledger.budget.Alpha.from_dict(record) == budget


def test_loading_rejects_unknown_fields_and_bad_fractions():
    record = make_budget().to_dict()
    with pytest.raises(ValueError, match="unexpected"):
        scmle.assurance.ledger.budget.Alpha.from_dict({**record, "extra": 1})
    record["confirmation"]["alpha_confirm"] = "not-a-fraction"
    with pytest.raises(ValueError, match="alpha_confirm"):
        scmle.assurance.ledger.budget.Alpha.from_dict(record)
