"""Online FDR rules: hand-worked levels, boundaries and bookkeeping.

The hand calculations follow the formulas of the cited papers (see the
module docstring of `online_fdr`). No R output exists here: R is not
installed, so agreement with `onlineFDR` is not claimed.
"""

from fractions import Fraction
import math

import pytest

from scmle.data.ledger import online_fdr

ALPHA = Fraction(1, 10)


def run(rule, items):
    state = rule.initial()
    steps = []
    for item in items:
        step = rule.step(state, item)
        steps.append(step)
        state = step.state
    return steps


def p(value):
    return online_fdr.Evidence(p=value)


def e(value):
    return online_fdr.Evidence(e=value)


def test_default_discount_sums_to_one_in_closed_form():
    discount = online_fdr.Discount()
    assert discount.weight(1) == Fraction(1, 2)
    assert discount.weight(3) == Fraction(1, 12)
    assert discount.cumulative(0) == 0
    assert discount.cumulative(3) == sum(discount.weight(j) for j in (1, 2, 3))
    assert discount.cumulative(10**6) < 1


def test_horizon_discount_is_uniform_then_zero():
    discount = online_fdr.Discount(horizon=4)
    assert [discount.weight(j) for j in (1, 4, 5)] == [
        Fraction(1, 4),
        Fraction(1, 4),
        0,
    ]
    assert discount.cumulative(2) == Fraction(1, 2)
    assert discount.cumulative(9) == 1
    with pytest.raises(ValueError):
        discount.weight(0)
    with pytest.raises(ValueError):
        online_fdr.Discount(horizon=0)


def test_calibrator_values_and_validity_bounds():
    assert online_fdr.calibrate(0.25) == 1.0
    assert online_fdr.calibrate(1.0) == 0.5
    assert online_fdr.calibrate(0.01, kappa=0.25) == pytest.approx(
        0.25 * 0.01**-0.75
    )
    assert online_fdr.calibrate(0.0) == 0.5 / math.sqrt(online_fdr.P_FLOOR)
    for bad in (0.0, 1.0, -0.5, float("nan")):
        with pytest.raises(ValueError, match="kappa"):
            online_fdr.calibrate(0.5, kappa=bad)
    with pytest.raises(ValueError, match="p"):
        online_fdr.calibrate(1.5)


def test_calibrated_values_have_unit_integral_numerically():
    # The calibrator integrates to one, so E f(U) = 1 for uniform U.
    n = 200_000
    midpoints = [(i + 0.5) / n for i in range(n)]
    mean = sum(online_fdr.calibrate(x, 0.5) for x in midpoints) / n
    assert mean == pytest.approx(1.0, abs=2e-3)


def test_elond_levels_by_hand():
    rule = online_fdr.ELond(ALPHA)
    steps = run(rule, [e(20.0), e(30.0), e(1.0)])
    # alpha * gamma_t * (R_{t-1} + 1) with gamma = 1/2, 1/6, 1/12.
    assert [s.level for s in steps] == [
        Fraction(1, 20),
        Fraction(1, 30),
        Fraction(1, 40),
    ]
    assert [s.rejected for s in steps] == [True, True, False]


def test_elond_boundary_is_inclusive_and_exact():
    rule = online_fdr.ELond(ALPHA)
    assert run(rule, [e(20.0)])[0].rejected is True
    assert run(rule, [e(math.nextafter(20.0, 0.0))])[0].rejected is False
    assert run(rule, [e(0.0)])[0].rejected is False


def test_elond_calibrates_a_p_value_when_given_no_e_value():
    rule = online_fdr.ELond(ALPHA)
    # e = 0.5 / sqrt(p) >= 20  <=>  p <= 1/1600
    assert run(rule, [p(1 / 1600 * 0.99)])[0].rejected is True
    assert run(rule, [p(1 / 1600 * 1.01)])[0].rejected is False
    step = run(rule, [p(0.25)])[0]
    assert step.e_value == 1.0


def test_elond_prefers_an_explicit_e_value_over_the_p_value():
    rule = online_fdr.ELond(ALPHA)
    both = online_fdr.Evidence(p=1e-12, e=1.0)
    step = run(rule, [both])[0]
    assert step.rejected is False and step.e_value == 1.0


def test_elond_wealth_is_the_unassigned_level():
    rule = online_fdr.ELond(ALPHA)
    steps = run(rule, [e(0.0)] * 3)
    assert steps[0].wealth_before == ALPHA
    assert [s.wealth_after for s in steps] == [
        ALPHA * Fraction(1, 2),
        ALPHA * Fraction(1, 3),
        ALPHA * Fraction(1, 4),
    ]


def test_elond_zero_alpha_never_rejects():
    steps = run(online_fdr.ELond(0), [e(1e300), e(1e300)])
    assert [s.rejected for s in steps] == [False, False]
    assert all(s.level == 0 for s in steps)


def test_lordpp_levels_and_wealth_by_hand():
    rule = online_fdr.LordPlusPlus(ALPHA)
    steps = run(rule, [p(0.0), p(0.0), p(0.9)])
    assert rule.w0 == Fraction(1, 20)
    assert [s.level for s in steps] == [
        Fraction(1, 40),
        Fraction(1, 30),
        Fraction(1, 16),
    ]
    assert [s.rejected for s in steps] == [True, True, False]
    assert steps[0].wealth_before == Fraction(1, 20)
    assert steps[0].wealth_after == Fraction(1, 10) - Fraction(1, 40)
    assert steps[1].wealth_after == Fraction(1, 5) - Fraction(1, 40) - Fraction(
        1, 30
    )


def test_lordpp_rejection_boundary_is_inclusive():
    rule = online_fdr.LordPlusPlus(
        Fraction(1, 4)
    )  # first level 1/16, exact in binary
    level = 1 / 16
    assert run(rule, [p(level)])[0].rejected is True
    assert run(rule, [p(math.nextafter(level, 1.0))])[0].rejected is False


def test_lordpp_needs_a_p_value():
    with pytest.raises(ValueError, match="needs a p-value"):
        online_fdr.LordPlusPlus(ALPHA).step(online_fdr.State(), e(5.0))


def test_lordpp_initial_wealth_is_validated():
    with pytest.raises(ValueError, match="w0"):
        online_fdr.LordPlusPlus(ALPHA, w0=Fraction(2, 10))
    with pytest.raises(ValueError, match="w0"):
        online_fdr.LordPlusPlus(ALPHA, w0=-1)
    assert online_fdr.LordPlusPlus(ALPHA, w0=0).w0 == 0
    assert online_fdr.LordPlusPlus(ALPHA, w0=ALPHA).w0 == ALPHA


def test_saffron_levels_by_hand():
    rule = online_fdr.Saffron(ALPHA)
    steps = run(rule, [p(0.0), p(0.9), p(0.3)])
    assert [s.level for s in steps] == [
        Fraction(1, 80),
        Fraction(1, 40),
        Fraction(1, 120),
    ]
    assert [s.candidate for s in steps] == [True, False, True]
    assert [s.rejected for s in steps] == [True, False, False]


def test_saffron_candidates_do_not_cost_wealth():
    rule = online_fdr.Saffron(ALPHA)
    steps = run(rule, [p(0.0), p(0.3), p(0.3)])
    # After the first rejection the level is 1/40; a candidate keeps it.
    assert steps[1].level == Fraction(1, 40)
    assert steps[2].level == Fraction(1, 40)
    assert steps[1].wealth_after == steps[1].wealth_before
    # A non-candidate at the same level does cost wealth.
    spent = run(rule, [p(0.0), p(0.9)])[1]
    assert spent.wealth_after == spent.wealth_before - Fraction(1, 40)


def test_saffron_candidate_boundary_is_inclusive_and_capped_by_lambda():
    rule = online_fdr.Saffron(ALPHA)
    assert run(rule, [p(0.5)])[0].candidate is True
    assert run(rule, [p(math.nextafter(0.5, 1.0))])[0].candidate is False
    big = online_fdr.Saffron(
        Fraction(9, 10), lam=Fraction(1, 100), w0=Fraction(9, 10)
    )
    assert big.level(big.initial()) == Fraction(1, 100)


def test_saffron_lambda_is_validated():
    for bad in (0, 1, Fraction(3, 2), -1):
        with pytest.raises(ValueError, match="lambda"):
            online_fdr.Saffron(ALPHA, lam=bad)


def test_alpha_is_validated():
    for bad in (-0.1, 1.1, float("nan"), True):
        with pytest.raises(ValueError, match="alpha"):
            online_fdr.ELond(bad)


def test_kappa_is_validated():
    for bad in (0.0, 1.0, float("nan")):
        with pytest.raises(ValueError, match="kappa"):
            online_fdr.ELond(ALPHA, kappa=bad)


def test_evidence_validation():
    with pytest.raises(ValueError, match="evidence"):
        online_fdr.Evidence()
    with pytest.raises(ValueError):
        online_fdr.Evidence(p=1.5)
    with pytest.raises(ValueError, match="non-negative"):
        online_fdr.Evidence(e=-1.0)


def test_guarantees_state_their_dependence_assumption():
    assert online_fdr.ELond.guarantee.dependence == "arbitrary"
    assert online_fdr.LordPlusPlus.guarantee.dependence == "independent"
    assert online_fdr.Saffron.guarantee.dependence == "independent"
    for cls in (online_fdr.ELond, online_fdr.LordPlusPlus, online_fdr.Saffron):
        assert cls.guarantee.verified
        assert cls.guarantee.citation
        assert cls.guarantee.assumptions


def test_config_round_trips_and_rejects_bad_records():
    config = online_fdr.RuleConfig(
        online_fdr.SAFFRON,
        ALPHA,
        horizon=7,
        w0=Fraction(1, 25),
        lam=Fraction(2, 5),
        kappa=0.3,
    )
    again = online_fdr.RuleConfig.from_dict(config.to_dict())
    assert again == config
    assert isinstance(again.build(), online_fdr.Saffron)
    record = config.to_dict()
    record["extra"] = 1
    with pytest.raises(ValueError, match="unexpected"):
        online_fdr.RuleConfig.from_dict(record)
    with pytest.raises(ValueError, match="rule"):
        online_fdr.RuleConfig("bonferroni", ALPHA)


def test_replay_matches_stepping():
    config = online_fdr.RuleConfig(online_fdr.LORD_PLUS_PLUS, ALPHA)
    items = [p(0.001), p(0.5), p(0.0001), p(0.2)]
    direct = run(config.build(), items)
    assert online_fdr.replay(config, items) == direct


def brute_lord_level(alpha, w0, gamma, rejected, t):
    """LORD++ level at time t (1-based) from the definition, eq. (5)."""
    times = [i + 1 for i, r in enumerate(rejected[: t - 1]) if r]
    total = w0 * gamma(t)
    for j, tau in enumerate(times):
        total += (alpha - w0 if j == 0 else alpha) * gamma(t - tau)
    return total


def brute_saffron_level(alpha, w0, lam, gamma, rejected, candidate, t):
    """SAFFRON level at time t from Section 2.3, written index by index."""
    taus = [i + 1 for i, r in enumerate(rejected[: t - 1]) if r]
    c0 = sum(candidate[: t - 1])
    total = w0 * gamma(t - c0)
    for j, tau in enumerate(taus):
        cj = sum(candidate[tau : t - 1])
        total += (alpha - w0 if j == 0 else alpha) * gamma(t - tau - cj)
    return min(lam, (1 - lam) * total)


def test_levels_match_a_brute_force_rederivation_on_fixed_sequences():
    gamma = online_fdr.Discount().weight
    sequences = [
        [0.0, 0.3, 0.9, 1e-9, 0.04, 0.6, 0.0, 0.2, 0.001, 0.77],
        [0.9] * 6 + [0.0, 0.0, 0.0] + [0.5, 0.4],
        [0.002, 0.003, 0.4, 0.45, 0.0, 0.7, 0.01, 0.0001, 0.3, 0.2, 0.0],
    ]
    for values in sequences:
        lord = run(online_fdr.LordPlusPlus(ALPHA), [p(v) for v in values])
        saff = run(online_fdr.Saffron(ALPHA), [p(v) for v in values])
        rejected, cands = [], []
        for t, step in enumerate(lord, start=1):
            expect = brute_lord_level(ALPHA, ALPHA / 2, gamma, rejected, t)
            assert step.level == expect
            rejected.append(step.rejected)
        rejected, cands = [], []
        for t, step in enumerate(saff, start=1):
            expect = brute_saffron_level(
                ALPHA, ALPHA / 2, Fraction(1, 2), gamma, rejected, cands, t
            )
            assert step.level == expect
            rejected.append(step.rejected)
            cands.append(step.candidate)


def test_elond_equals_lond_levels_of_the_paper():
    # alpha_t = alpha * gamma_t * (|R_{t-1}| + 1), Section 3.2.
    rule = online_fdr.ELond(ALPHA, online_fdr.Discount(horizon=5))
    steps = run(rule, [e(1e9), e(1e9), e(0.0), e(1e9), e(0.0), e(1e9)])
    expected = [ALPHA / 5 * k for k in (1, 2, 3, 3, 4, 4)]
    assert [s.level for s in steps[:5]] == expected[:5]
    # Beyond the horizon gamma is zero: nothing can be rejected.
    assert steps[5].level == 0 and steps[5].rejected is False


def test_a_float_p_value_is_compared_exactly():
    rule = online_fdr.LordPlusPlus(ALPHA)
    level = rule.level(rule.initial())
    assert level == Fraction(1, 40)
    # The double nearest 0.025 lies a hair above 1/40, so it is not rejected.
    assert Fraction(0.025) > level
    assert run(rule, [p(0.025)])[0].rejected is False


def test_wealth_is_never_negative_over_long_adversarial_runs():
    for make in (online_fdr.LordPlusPlus, online_fdr.Saffron):
        for pattern in ([0.0] * 30, [0.9] * 30, [0.0, 0.9] * 15, [0.3] * 30):
            steps = run(make(ALPHA), [p(v) for v in pattern])
            assert all(s.wealth_after >= 0 for s in steps)
