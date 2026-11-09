"""The ledger checks of the harness: their helpers and their gates."""

import math

import numpy
import pytest
from scipy import stats

from benchmarks.qualification import __main__
from benchmarks.qualification import ledger_checks
from scmle.data.ledger import bivariate
from scmle.data.ledger import online_fdr


def test_the_vectorized_tester_matches_the_library_test():
    rng = numpy.random.default_rng(0)
    x = rng.normal(size=(80, 5))
    x[:, 4] = numpy.round(x[:, 4])  # ties are ranked by average rank
    y = rng.normal(size=80) + x[:, 0]
    fast = ledger_checks.RankTester(x).pvalues(y)
    slow = [bivariate.spearman(x[:, j], y).p for j in range(5)]
    numpy.testing.assert_allclose(fast, slow, rtol=1e-10)


def test_band_is_alpha_plus_two_binomial_standard_errors():
    assert ledger_checks.band(0.05, 2000) == pytest.approx(0.0597, abs=1e-4)
    assert ledger_checks.band(0.05, 100) > ledger_checks.band(0.05, 2000)


def test_fdp_is_zero_without_rejections_and_a_share_otherwise():
    planted = numpy.array([True, False, False, True])
    assert ledger_checks.fdp(numpy.zeros(4, bool), planted) == 0.0
    assert ledger_checks.fdp(
        numpy.array([True, True, True, False]), planted
    ) == pytest.approx(2 / 3)


def test_power_oracle_has_the_right_limits_and_monotonicity():
    level = 0.01
    assert ledger_checks.spearman_power_oracle(
        1e-9, 300, level
    ) == pytest.approx(level, abs=1e-6)
    assert ledger_checks.spearman_power_oracle(0.9, 300, level) > 0.999
    low = ledger_checks.spearman_power_oracle(0.15, 300, level)
    assert low < ledger_checks.spearman_power_oracle(0.3, 300, level)
    assert low < ledger_checks.spearman_power_oracle(0.15, 900, level)
    # Symmetric in the sign of the correlation.
    assert ledger_checks.spearman_power_oracle(
        -0.2, 300, level
    ) == pytest.approx(ledger_checks.spearman_power_oracle(0.2, 300, level))


def test_the_elond_threshold_is_where_the_calibrated_e_value_hits_1_over_level():
    level = 0.025
    p = ledger_checks.elond_threshold(level)
    assert p == pytest.approx((0.5 * level) ** 2)
    assert online_fdr.calibrate(p) == pytest.approx(1.0 / level)
    assert online_fdr.calibrate(p * 1.01) < 1.0 / level


def test_empirical_power_tracks_the_oracle_on_a_small_run():
    finding = ledger_checks.check_power(300, 1)[0]
    assert finding.ok, finding.detail
    assert finding.numbers["power"] > 0.8


def test_fdr_is_within_the_band_for_each_rule_on_a_small_run():
    findings = ledger_checks.check_fdr(300, 2)
    assert len(findings) == 12
    assert all(f.ok for f in findings), [f.detail for f in findings if not f.ok]
    elond = [f for f in findings if f.numbers["rule"] == "e-lond"]
    assert all(f.numbers["enforced"] for f in elond)
    dependent_lord = [
        f
        for f in findings
        if f.numbers["rule"] == "lord++" and "correlated" in f.numbers["world"]
    ]
    assert dependent_lord and not any(
        f.numbers["enforced"] for f in dependent_lord
    )


def test_the_uniformity_check_rejects_a_skewed_p_value_source():
    rng = numpy.random.default_rng(0)
    skewed = rng.uniform(size=ledger_checks.KS_DRAWS) ** 2
    assert stats.kstest(skewed, "uniform").pvalue < ledger_checks.KS_FLOOR
    assert ledger_checks.check_uniformity(3)[0].ok


def test_the_control_checks_show_global_permutation_is_anti_conservative():
    findings = {f.name: f for f in ledger_checks.check_controls(60, 4)}
    assert all(f.ok for f in findings.values()), [
        f.detail for f in findings.values() if not f.ok
    ]
    grouped = next(f for n, f in findings.items() if n.startswith("grouped"))
    assert grouped.numbers["global_rate"] > 0.5
    assert grouped.numbers["strata_rate"] <= grouped.numbers["bound"]


def test_the_cli_accepts_check_fdr_and_keeps_plain_check(capsys):
    assert math.isclose(ledger_checks.band(0.5, 1), 0.5 + 2 * 0.5)
    code = __main__.main(["--check", "fdr", "--runs", "40", "--repeats", "20"])
    out = capsys.readouterr().out
    assert '"findings"' in out
    assert code in (0, 1)
    assert __main__.main(["--check"]) in (0, 1)
