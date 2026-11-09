"""Family-wise error and false discovery rate of the repository's rules.

`scmle.data.ledger.multiplicity` (Holm, Bonferroni, Benjamini-Hochberg,
Benjamini-Yekutieli), `scmle.evaluation.coverage.holm_rejections` and the
online rules in `scmle.data.ledger.online_fdr` (e-LOND, LORD++, SAFFRON) are
run on p-values simulated from a known truth. Under the global null the
family-wise error and the false discovery rate both equal the probability of
any rejection, which must not exceed alpha = 0.05 by more than 3.5 binomial
standard errors. With some true effects the false discovery rate is the mean
false discovery proportion and is held to its bound with 3.5 standard errors
of that mean, estimated from the simulated proportions.
"""

from fractions import Fraction

import numpy
from scipy import stats
import support_statistics

from scmle.data.ledger import budget
from scmle.data.ledger import multiplicity
from scmle.data.ledger import online_fdr
from scmle.evaluation import coverage

ALPHA = 0.05
LEVEL = Fraction(1, 20)
FAMILY = 20
RUNS = 4000
FWER_RULES = (budget.Correction.HOLM, budget.Correction.BONFERRONI)
FDR_RULES = (budget.Correction.BH, budget.Correction.BY)


def correlated_p_values(rng, runs, rho, signals=0, strength=4.0):
    """One-sided p-values of equicorrelated z statistics; effects first."""
    common = rng.normal(size=(runs, 1))
    z = numpy.sqrt(rho) * common + numpy.sqrt(1.0 - rho) * rng.normal(
        size=(runs, FAMILY)
    )
    z[:, :signals] += strength
    return stats.norm.sf(z)


def false_discovery_bound(proportions, bound):
    """Asserts a mean false discovery proportion is within its bound."""
    mean = float(numpy.mean(proportions))
    error = float(numpy.std(proportions, ddof=1) / numpy.sqrt(len(proportions)))
    assert mean <= bound + support_statistics.Z_LIMIT * error, (mean, bound)
    return mean


def test_error_rates_under_the_global_null_with_independent_p_values():
    rng = numpy.random.default_rng(0)
    p_values = rng.uniform(size=(RUNS, FAMILY))
    for rule in (*FWER_RULES, *FDR_RULES):
        any_rejected = sum(
            any(multiplicity.adjust(row, rule, LEVEL).rejected)
            for row in p_values
        )
        support_statistics.assert_rate_at_most(
            any_rejected / RUNS, ALPHA, RUNS, f"{rule.value} global null"
        )


def test_error_rates_under_the_global_null_with_positive_dependence():
    rng = numpy.random.default_rng(1)
    for rho in (0.5, 0.9):
        p_values = correlated_p_values(rng, RUNS, rho)
        for rule in (*FWER_RULES, *FDR_RULES):
            any_rejected = sum(
                any(multiplicity.adjust(row, rule, LEVEL).rejected)
                for row in p_values
            )
            support_statistics.assert_rate_at_most(
                any_rejected / RUNS,
                ALPHA,
                RUNS,
                f"{rule.value} global null, rho={rho}",
            )


def test_false_discovery_rate_with_true_effects_is_below_alpha_times_m0_over_m():
    rng = numpy.random.default_rng(2)
    signals = 5
    bound = ALPHA * (FAMILY - signals) / FAMILY
    for rho in (0.0, 0.5):
        p_values = correlated_p_values(rng, RUNS, rho, signals=signals)
        for rule in (*FDR_RULES, *FWER_RULES):
            proportions = []
            familywise = 0
            for row in p_values:
                rejected = numpy.array(
                    multiplicity.adjust(row, rule, LEVEL).rejected
                )
                false = int(rejected[signals:].sum())
                proportions.append(false / max(int(rejected.sum()), 1))
                familywise += false > 0
            mean = false_discovery_bound(proportions, bound)
            print(f"MEASURED fdr {rule.value} rho={rho}: {mean:.4f} <= {bound}")
            if rule in FWER_RULES:
                support_statistics.assert_rate_at_most(
                    familywise / RUNS,
                    ALPHA,
                    RUNS,
                    f"{rule.value} family-wise error, rho={rho}",
                )


def test_holm_in_the_coverage_module_agrees_with_the_ledger_holm():
    rng = numpy.random.default_rng(3)
    for ignored in range(500):
        p_values = rng.beta(0.3, 1.0, size=FAMILY).tolist()
        expected = multiplicity.adjust(
            p_values, budget.Correction.HOLM, LEVEL
        ).rejected
        assert tuple(coverage.holm_rejections(p_values, ALPHA)) == expected


def test_the_coverage_module_holm_controls_the_family_wise_error():
    rng = numpy.random.default_rng(4)
    any_rejected = sum(
        any(coverage.holm_rejections(row.tolist(), ALPHA))
        for row in rng.uniform(size=(RUNS, FAMILY))
    )
    support_statistics.assert_rate_at_most(
        any_rejected / RUNS, ALPHA, RUNS, "coverage.holm_rejections"
    )


def online_run(rule, rng, signals):
    """One sequence of independent p-values; returns (false, rejected)."""
    length = 50
    alternative = rng.random(length) < signals
    p_values = numpy.where(
        alternative, rng.beta(0.05, 1.0, length), rng.random(length)
    )
    config = online_fdr.RuleConfig(rule, LEVEL)
    steps = online_fdr.replay(
        config, [online_fdr.Evidence(p=float(p)) for p in p_values]
    )
    rejected = numpy.array([step.rejected for step in steps])
    return int((rejected & ~alternative).sum()), int(rejected.sum())


def test_online_rules_control_the_error_rate_under_the_global_null():
    rng = numpy.random.default_rng(5)
    runs = 1500
    for rule in online_fdr.RULE_NAMES:
        any_false = sum(
            online_run(rule, rng, 0.0)[0] > 0 for ignored in range(runs)
        )
        support_statistics.assert_rate_at_most(
            any_false / runs, ALPHA, runs, f"{rule} global null"
        )


def test_online_rules_control_the_false_discovery_rate_with_true_effects():
    rng = numpy.random.default_rng(6)
    runs = 1500
    for rule in online_fdr.RULE_NAMES:
        proportions = []
        rejections = 0
        for ignored in range(runs):
            false, rejected = online_run(rule, rng, 0.2)
            proportions.append(false / max(rejected, 1))
            rejections += rejected
        mean = false_discovery_bound(proportions, ALPHA)
        print(f"MEASURED online fdr {rule}: {mean:.4f} <= {ALPHA}")
        assert rejections > 0, f"{rule} never rejected: no power to check"


def test_e_lond_controls_the_error_rate_under_dependent_p_values():
    # e-LOND claims validity under arbitrary dependence between the tests.
    rng = numpy.random.default_rng(7)
    runs, length = 1500, 50
    any_false = 0
    for ignored in range(runs):
        common = rng.normal()
        z = numpy.sqrt(0.7) * common + numpy.sqrt(0.3) * rng.normal(size=length)
        steps = online_fdr.replay(
            online_fdr.RuleConfig(online_fdr.ELOND, LEVEL),
            [online_fdr.Evidence(p=float(p)) for p in stats.norm.sf(z)],
        )
        any_false += any(step.rejected for step in steps)
    support_statistics.assert_rate_at_most(
        any_false / runs, ALPHA, runs, "e-lond, dependent global null"
    )
