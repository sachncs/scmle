"""Falsification battery: Holm control, false alarms and replication."""

import numpy
import pytest

from scmle.data.identification import battery


def test_wald_tests_and_holm_rejections():
    tests = [
        battery.wald_test("a", "negative_outcome", 0.1, 1.0),
        battery.wald_test("b", "negative_exposure", 5.0, 1.0),
    ]
    report = battery.run_battery(tests, 0.05)
    assert report.rejected == ("b",) and not report.passed and report.ran
    clean = battery.run_battery(tests[:1], 0.05)
    assert clean.passed and clean.to_dict()["rejected"] == []


def test_an_empty_battery_is_not_a_pass():
    report = battery.run_battery([], 0.05)
    assert not report.ran and not report.passed


def test_the_false_alarm_rate_is_at_most_alpha():
    alpha, runs, flagged = 0.05, 4000, 0
    for seed in range(runs):
        draws = numpy.random.default_rng(seed).standard_normal(3)
        tests = [
            battery.wald_test(f"t{i}", "placebo", float(d), 1.0)
            for i, d in enumerate(draws)
        ]
        flagged += not battery.run_battery(tests, alpha).passed
    assert flagged / runs <= alpha + 3 * numpy.sqrt(alpha * (1 - alpha) / runs)


def test_replication_needs_agreement_in_sign_and_homogeneity():
    ok = battery.replicate({"a": (1.0, 0.2), "b": (1.2, 0.2)}, 1.1)
    assert ok.passed and ok.same_sign
    flipped = battery.replicate({"a": (1.0, 0.2), "b": (-0.5, 0.2)}, 0.3)
    assert not flipped.same_sign and not flipped.passed
    spread = battery.replicate({"a": (1.0, 0.05), "b": (2.0, 0.05)}, 1.5)
    assert spread.same_sign and spread.p_heterogeneity < 0.001
    assert not spread.passed
    assert spread.to_dict()["strata"]["a"] == [1.0, 0.05]


def test_bad_inputs_are_refused():
    with pytest.raises(ValueError, match="at least 2"):
        battery.replicate({"a": (1.0, 0.1)}, 1.0)
    with pytest.raises(ValueError, match="positive"):
        battery.replicate({"a": (1.0, 0.0), "b": (1.0, 0.1)}, 1.0)
    with pytest.raises(ValueError, match="positive standard error"):
        battery.wald_test("x", "placebo", 1.0, 0.0)
    with pytest.raises(ValueError, match="alpha"):
        battery.run_battery([], 1.5)
