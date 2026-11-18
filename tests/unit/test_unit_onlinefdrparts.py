"""Unit tests of the online FDR rules and their base, by full module name."""

from fractions import Fraction
import math

import pytest

from scmle.assurance.ledger.online_fdr import rules
from scmle.assurance.ledger.online_fdr import tester
from scmle.contracts import errors

ALPHA = Fraction(1, 10)
W0 = Fraction(1, 20)

# ---- tester ----------------------------------------------------------------


def test_the_rule_names_and_defaults():
    assert tester.RULE_NAMES == ("e-lond", "lord++", "saffron")
    assert tester.DEFAULT_KAPPA == 0.5
    assert Fraction(1, 2) == tester.DEFAULT_LAMBDA
    assert tester.P_FLOOR == 1e-300


def test_the_default_discount_sums_to_one():
    discount = tester.Discount()
    assert discount.weight(1) == Fraction(1, 2)
    assert discount.weight(3) == Fraction(1, 12)
    assert discount.cumulative(0) == 0
    assert discount.cumulative(3) == Fraction(3, 4)


def test_a_horizon_discount_is_flat_then_zero():
    discount = tester.Discount(4)
    assert discount.weight(4) == Fraction(1, 4)
    assert discount.weight(5) == 0
    assert discount.cumulative(2) == Fraction(1, 2)
    assert discount.cumulative(9) == 1


def test_a_discount_refuses_bad_indexes_and_horizons():
    with pytest.raises(errors.InputError, match="index must be at least 1"):
        tester.Discount().weight(0)
    with pytest.raises(errors.InputError, match="horizon must be at least 1"):
        tester.Discount(0)
    with pytest.raises(errors.InputError, match="count must be non-negative"):
        tester.Discount().cumulative(-1)
    assert tester.Discount(1).horizon == 1


def test_evidence_needs_a_value_and_checks_it():
    with pytest.raises(
        errors.InputError, match="needs a p-value or an e-value"
    ):
        tester.Evidence()
    with pytest.raises(errors.InputError, match="p must be within"):
        tester.Evidence(p=1.5)
    with pytest.raises(errors.InputError, match="e must be non-negative"):
        tester.Evidence(e=-0.1)
    with pytest.raises(errors.InputError, match="e must be a finite"):
        tester.Evidence(e=math.inf)
    assert tester.Evidence(e=0.0).e == 0.0
    assert tester.Evidence(p=0.0, e=1.0).p == 0.0


def test_calibrate_is_kappa_times_p_to_the_kappa_minus_one():
    assert tester.calibrate(0.25) == 1.0
    assert tester.calibrate(0.0) == 0.5 / math.sqrt(1e-300)
    assert tester.calibrate(0.25, 0.25) == pytest.approx(0.25 * 0.25**-0.75)
    with pytest.raises(errors.InputError, match="kappa must be strictly"):
        tester.calibrate(0.5, 1.0)
    with pytest.raises(errors.InputError, match="kappa must be strictly"):
        tester.calibrate(0.5, 0.0)
    with pytest.raises(errors.InputError, match="p must be within"):
        tester.calibrate(-0.1)


def test_rational_reads_levels_within_zero_one():
    assert tester.rational(0.1, "a") == Fraction(1, 10)
    assert tester.rational(1, "a") == 1
    assert tester.rational(0, "a") == 0
    with pytest.raises(errors.InputError, match="a must be within"):
        tester.rational(Fraction(11, 10), "a")


def test_the_base_class_earns_no_credit_and_starts_empty():
    rule = rules.Lond(ALPHA)
    assert rule.credit(rule.initial()) == 0
    assert rule.initial() == tester.State(0, (), (), Fraction(0))
    assert rule.discount == tester.Discount()


# ---- e-LOND ----------------------------------------------------------------


def test_lond_levels_and_wealth_follow_the_discount():
    rule = rules.Lond(ALPHA)
    assert rule.level(rule.initial()) == Fraction(1, 20)
    assert rule.wealth(rule.initial()) == ALPHA
    after = tester.State(tests=1, rejections=(1,))
    assert rule.level(after) == Fraction(1, 30)
    assert rule.wealth(after) == Fraction(1, 20)


def test_lond_rejects_exactly_at_one_over_level():
    rule = rules.Lond(ALPHA)
    hit = rule.step(rule.initial(), tester.Evidence(e=20.0))
    assert hit.rejected is True
    assert hit.candidate is False
    assert hit.e_value == 20.0
    assert hit.state.rejections == (1,)
    assert hit.wealth_before == ALPHA
    assert hit.wealth_after == Fraction(1, 20)
    miss = rule.step(rule.initial(), tester.Evidence(e=19.99))
    assert miss.rejected is False
    assert miss.state.rejections == ()


def test_lond_calibrates_a_p_value_only_without_an_e_value():
    rule = rules.Lond(ALPHA)
    step = rule.step(rule.initial(), tester.Evidence(p=1 / 1600))
    assert step.e_value == 20.0
    assert step.rejected is True
    both = rule.step(rule.initial(), tester.Evidence(p=1e-9, e=1.0))
    assert both.e_value == 1.0
    assert both.rejected is False


def test_lond_refuses_a_bad_kappa():
    with pytest.raises(errors.InputError, match="kappa must be strictly"):
        rules.Lond(ALPHA, kappa=1.0)
    assert rules.Lond(ALPHA, kappa=0.3).kappa == 0.3


# ---- LORD++ ----------------------------------------------------------------


def test_lord_defaults_w0_to_half_alpha_and_bounds_it():
    assert rules.Lord(ALPHA).w0 == W0
    assert rules.Lord(ALPHA, w0=ALPHA).w0 == ALPHA
    assert rules.Lord(ALPHA, w0=0).w0 == 0
    with pytest.raises(
        errors.InputError, match=r"w0 must be within \[0, alpha\]"
    ):
        rules.Lord(ALPHA, w0=Fraction(11, 100))


def test_lord_levels_reward_each_rejection():
    rule = rules.Lord(ALPHA)
    first = rule.step(rule.initial(), tester.Evidence(p=0.02))
    assert first.level == Fraction(1, 40)
    assert first.rejected is True
    assert first.e_value is None
    assert first.wealth_before == W0
    assert first.wealth_after == Fraction(3, 40)
    second = rule.step(first.state, tester.Evidence(p=0.0))
    assert second.level == Fraction(1, 30)
    third_level = rule.level(second.state)
    assert third_level == Fraction(1, 16)


def test_lord_credit_counts_rejections():
    rule = rules.Lord(ALPHA)
    assert rule.credit(tester.State()) == W0
    assert rule.credit(tester.State(rejections=(1,))) == ALPHA
    assert rule.credit(tester.State(rejections=(1, 2, 3))) == Fraction(3, 10)


def test_lord_needs_a_p_value():
    rule = rules.Lord(ALPHA)
    with pytest.raises(errors.InputError, match="lord\\+\\+ needs a p-value"):
        rule.step(rule.initial(), tester.Evidence(e=5.0))


def test_lord_does_not_reject_just_above_the_level():
    rule = rules.Lord(ALPHA)
    step = rule.step(rule.initial(), tester.Evidence(p=0.0251))
    assert step.rejected is False
    assert step.state.spent == Fraction(1, 40)


# ---- SAFFRON ---------------------------------------------------------------


def test_saffron_marks_candidates_that_cost_nothing():
    rule = rules.Saffron(ALPHA)
    assert rule.lam == Fraction(1, 2)
    assert rule.level(rule.initial()) == Fraction(1, 80)
    candidate = rule.step(rule.initial(), tester.Evidence(p=0.5))
    assert candidate.candidate is True
    assert candidate.rejected is False
    assert candidate.state.candidates == (1,)
    assert candidate.state.spent == 0
    assert candidate.wealth_after == Fraction(1, 40)
    assert rule.level(candidate.state) == Fraction(1, 80)
    plain = rule.step(candidate.state, tester.Evidence(p=0.9))
    assert plain.candidate is False
    assert plain.state.spent == Fraction(1, 80)


def test_saffron_rejection_is_also_a_candidate():
    rule = rules.Saffron(ALPHA)
    step = rule.step(rule.initial(), tester.Evidence(p=0.01))
    assert step.rejected is True
    assert step.candidate is True
    assert rule.level(step.state) == min(
        Fraction(1, 2),
        Fraction(1, 2) * (W0 * Fraction(1, 2) + W0 * Fraction(1, 2)),
    )


def test_saffron_level_is_capped_at_lambda():
    rule = rules.Saffron(1, w0=1, lam=Fraction(1, 100))
    assert rule.level(rule.initial()) == Fraction(1, 100)


def test_saffron_refuses_a_lambda_outside_zero_one():
    for lam in (0, 1):
        with pytest.raises(errors.InputError, match="lambda must be strictly"):
            rules.Saffron(ALPHA, lam=lam)


def test_every_rule_states_a_verified_guarantee():
    assert rules.Lond.guarantee.dependence == "arbitrary"
    assert rules.Lord.guarantee.dependence == "independent"
    assert rules.Saffron.guarantee.dependence == "independent"
    assert all(
        r.guarantee.verified for r in (rules.Lond, rules.Lord, rules.Saffron)
    )
    assert (rules.Lond.name, rules.Lord.name, rules.Saffron.name) == (
        "e-lond",
        "lord++",
        "saffron",
    )


# ---- settings and replay ---------------------------------------------------


def test_settings_build_each_rule_with_its_parameters():
    lond = rules.Settings("e-lond", ALPHA, horizon=5, kappa=0.3).build()
    assert isinstance(lond, rules.Lond)
    assert lond.discount == tester.Discount(5)
    assert lond.kappa == 0.3
    lord = rules.Settings("lord++", ALPHA, w0=Fraction(1, 40)).build()
    assert isinstance(lord, rules.Lord)
    assert lord.w0 == Fraction(1, 40)
    saffron = rules.Settings("saffron", ALPHA, lam=Fraction(1, 4)).build()
    assert isinstance(saffron, rules.Saffron)
    assert saffron.lam == Fraction(1, 4)


def test_settings_refuse_an_unknown_rule():
    with pytest.raises(errors.InputError, match="rule must be one of"):
        rules.Settings("bonferroni", ALPHA)


def test_settings_round_trip_through_a_record():
    config = rules.Settings("saffron", ALPHA, horizon=3, w0=Fraction(1, 30))
    record = config.to_dict()
    assert record == {
        "rule": "saffron",
        "alpha": "1/10",
        "horizon": 3,
        "w0": "1/30",
        "lambda": "1/2",
        "kappa": 0.5,
    }
    assert rules.Settings.from_dict(record) == config
    plain = rules.Settings("e-lond", ALPHA)
    assert plain.to_dict()["w0"] is None
    assert rules.Settings.from_dict(plain.to_dict()) == plain


def test_settings_refuse_a_record_with_wrong_fields():
    record = rules.Settings("e-lond", ALPHA).to_dict()
    with pytest.raises(errors.InputError, match="unexpected or missing"):
        rules.Settings.from_dict({**record, "extra": 1})
    del record["kappa"]
    with pytest.raises(errors.InputError, match="unexpected or missing"):
        rules.Settings.from_dict(record)


def test_replay_runs_the_rule_from_the_start():
    config = rules.Settings("lord++", ALPHA)
    steps = rules.replay(
        config, [tester.Evidence(p=0.0), tester.Evidence(p=1.0)]
    )
    assert [s.rejected for s in steps] == [True, False]
    assert steps[1].state.tests == 2
    assert steps[1].level == Fraction(1, 30)
    assert rules.replay(config, []) == []
