"""BH, BY, Holm and Bonferroni on the survivors: oracles and boundaries."""

from fractions import Fraction

import numpy
import pytest
from scipy import stats

from benchmarks.qualification import oracles
from scmle.data.ledger import budget
from scmle.data.ledger import multiplicity

R = budget.MultiplicityRule
F = Fraction


def test_bh_and_by_adjusted_values_match_the_frozen_scipy_oracle():
    for case in oracles.load_oracle("fdr_control").cases:
        rule = R(case["inputs"]["method"])
        out = multiplicity.adjust(case["inputs"]["pvalues"], rule, F(1, 20))
        oracles.assert_agrees(
            list(out.adjusted),
            case["expected"]["adjusted"],
            case["id"],
            rel_tol=1e-12,
        )


@pytest.mark.parametrize("seed", range(8))
def test_bh_by_agree_with_scipy_and_decisions_match_adjusted(seed):
    rng = numpy.random.default_rng(seed)
    n = int(rng.integers(1, 40))
    p = numpy.concatenate(
        [rng.uniform(size=n), rng.uniform(0, 0.01, size=n // 2)]
    )
    for rule, method in ((R.BH, "bh"), (R.BY, "by")):
        out = multiplicity.adjust(p.tolist(), rule, F(1, 20))
        reference = stats.false_discovery_control(p, method=method)
        numpy.testing.assert_allclose(out.adjusted, reference, rtol=1e-12)
        assert list(out.rejected) == list(reference <= 0.05)


def brute_holm(p, alpha):
    """Holm from the definition: reject i iff p_(j) <= alpha/(m-j+1) for all
    ranks j up to the rank of i."""
    order = numpy.argsort(p, kind="stable")
    sorted_p = numpy.asarray(p)[order]
    m_total = len(p)
    ok = [sorted_p[j] <= alpha / (m_total - j) for j in range(m_total)]
    reject_sorted = numpy.cumprod(ok).astype(bool)
    out = numpy.zeros(m_total, dtype=bool)
    out[order] = reject_sorted
    adjusted_sorted = numpy.minimum(
        1.0,
        numpy.maximum.accumulate((m_total - numpy.arange(m_total)) * sorted_p),
    )
    adjusted = numpy.zeros(m_total)
    adjusted[order] = adjusted_sorted
    return out, adjusted


@pytest.mark.parametrize("seed", range(8))
def test_holm_matches_a_brute_force_definition(seed):
    rng = numpy.random.default_rng(100 + seed)
    p = numpy.concatenate([rng.uniform(size=12), rng.uniform(0, 0.004, size=4)])
    out = multiplicity.adjust(p.tolist(), R.HOLM, F(1, 20))
    reject, adjusted = brute_holm(p, 0.05)
    assert list(out.rejected) == list(reject)
    numpy.testing.assert_allclose(out.adjusted, adjusted, rtol=1e-12)


def test_bonferroni_threshold_and_adjustment():
    out = multiplicity.adjust(
        [0.01, 0.0124, 0.0126, 0.5], R.BONFERRONI, F(1, 20)
    )
    assert list(out.rejected) == [True, True, False, False]  # 0.05 / 4
    assert out.adjusted == pytest.approx((0.04, 0.0496, 0.0504, 1.0))
    # Inclusive at an exactly representable threshold: 1/2 / 2 = 0.25.
    edge = multiplicity.adjust([0.25, 0.75], R.BONFERRONI, F(1, 2))
    assert edge.rejected == (True, False)


def test_exact_boundaries_are_inclusive_and_step_wise():
    half = F(1, 2)
    # m = 2: BH thresholds are 1/4 and 1/2.
    assert multiplicity.adjust([0.25, 0.5], R.BH, half).rejected == (True, True)
    assert multiplicity.adjust([0.25, 0.5000001], R.BH, half).rejected == (
        True,
        False,
    )
    assert multiplicity.adjust([0.2500001, 0.5], R.BH, half).rejected == (
        True,
        True,
    )
    assert multiplicity.adjust([0.2500001, 0.5000001], R.BH, half).rejected == (
        False,
        False,
    )
    # Holm thresholds are 1/4 then 1/2.
    assert multiplicity.adjust([0.25, 0.5], R.HOLM, half).rejected == (
        True,
        True,
    )
    assert multiplicity.adjust([0.2500001, 0.4], R.HOLM, half).rejected == (
        False,
        False,
    )
    assert multiplicity.adjust([0.25, 0.5000001], R.HOLM, half).rejected == (
        True,
        False,
    )


def test_bh_is_step_up_and_holm_is_step_down():
    # p_(2) passes its BH threshold although p_(1) fails its own.
    p = [0.03, 0.04]
    assert multiplicity.adjust(p, R.BH, F(1, 20)).rejected == (True, True)
    assert multiplicity.adjust(p, R.HOLM, F(1, 20)).rejected == (False, False)


def test_by_is_more_conservative_than_bh():
    p = [0.001, 0.01, 0.02, 0.04, 0.2]
    bh = multiplicity.adjust(p, R.BH, F(1, 20))
    by = multiplicity.adjust(p, R.BY, F(1, 20))
    assert sum(by.rejected) <= sum(bh.rejected)
    assert all(a >= x for a, x in zip(by.adjusted, bh.adjusted, strict=True))
    assert multiplicity.harmonic_exact(3) == F(11, 6)


def test_order_of_the_input_is_preserved_and_ties_agree():
    p = [0.9, 0.001, 0.04, 0.001, 0.04]
    out = multiplicity.adjust(p, R.BH, F(1, 20))
    assert out.rejected[1] == out.rejected[3]
    assert out.rejected[2] == out.rejected[4]
    assert out.rejected[0] is False


def test_empty_family_and_validation():
    assert multiplicity.adjust([], R.BY, F(1, 20)).rejected == ()
    with pytest.raises(ValueError, match="alpha"):
        multiplicity.adjust([0.1], R.BH, F(0))
    with pytest.raises(ValueError, match="alpha"):
        multiplicity.adjust([0.1], R.BH, F(1))
    with pytest.raises(ValueError, match="p-value"):
        multiplicity.adjust([1.1], R.BH, F(1, 20))
    with pytest.raises(ValueError, match="rule"):
        multiplicity.adjust([0.1], "bh", F(1, 20))  # type: ignore[arg-type]
    assert set(multiplicity.STATEMENTS) == set(R)


def test_a_p_value_of_one_is_never_rejected_and_zero_always_is():
    for rule in R:
        out = multiplicity.adjust([0.0, 1.0], rule, F(1, 20))
        assert out.rejected == (True, False)
        assert out.adjusted[1] == 1.0


def test_empirical_error_rates_under_the_global_null_and_with_signal():
    rng = numpy.random.default_rng(0)
    runs, family = 3000, 20
    any_reject = {R.HOLM: 0, R.BONFERRONI: 0}
    false_proportions = {R.BH: [], R.BY: []}
    for null in rng.uniform(size=(runs, family)):
        for rule in any_reject:
            any_reject[rule] += int(
                any(multiplicity.adjust(null, rule, F(1, 20)).rejected)
            )
        mixed = numpy.concatenate([null[:15], rng.uniform(0, 1e-4, size=5)])
        for rule, store in false_proportions.items():
            rejected = numpy.array(
                multiplicity.adjust(mixed, rule, F(1, 20)).rejected
            )
            total = rejected.sum()
            store.append(rejected[:15].sum() / total if total else 0.0)
    band = 2 * (0.05 * 0.95 / runs) ** 0.5
    for count in any_reject.values():
        assert count / runs <= 0.05 + band  # FWER under the global null
    for values in false_proportions.values():
        assert numpy.mean(values) <= 0.05 * 15 / 20 + band  # FDR <= alpha m0/m
