"""Closed-form checks that pin the feasibility arithmetic and its messages."""

import math

import pytest
from scipy import stats

from scmle.data.charter import feasibility
from scmle.data.detectors import certificates


def message(call):
    with pytest.raises(ValueError) as caught:
        call()
    return str(caught.value)


def test_open_unit_bounds_and_types():
    assert feasibility.open_unit(0.5, "x") == 0.5
    assert message(lambda: feasibility.open_unit(0.0, "x")) == (
        "x must be strictly between 0.0 and 1"
    )
    assert message(lambda: feasibility.open_unit(1.0, "x")) == (
        "x must be strictly between 0.0 and 1"
    )
    assert feasibility.open_unit(0.0, "x", low=-1.0) == 0.0
    assert message(lambda: feasibility.open_unit(-1.0, "x", low=-1.0)) == (
        "x must be strictly between -1.0 and 1"
    )
    for bad in (True, "0.5", None, float("nan")):
        assert message(lambda bad=bad: feasibility.open_unit(bad, "x")) == (
            "x must be strictly between 0.0 and 1"
        )


def test_positive_delta_bounds_and_types():
    assert feasibility.positive_delta(2, "d") == 2.0
    for bad in (0, -1, True, "1", float("inf"), float("nan")):
        assert message(
            lambda bad=bad: feasibility.positive_delta(bad, "d")
        ) == ("d must be a finite number greater than 0")


def test_k_must_be_an_integer_of_at_least_one():
    assert feasibility.check_k(1) == 1
    for bad in (0, True, 1.5, "2"):
        assert message(lambda bad=bad: feasibility.check_k(bad)) == (
            "k_max must be an integer of at least 1"
        )


def test_z_sum_is_the_sum_of_two_normal_quantiles():
    expected = stats.norm.ppf(1 - 0.05 / 6) + stats.norm.ppf(0.8)
    assert feasibility.z_sum(0.05, 3, 0.8) == pytest.approx(expected, rel=1e-12)
    assert message(lambda: feasibility.z_sum(1.0, 3, 0.8)) == (
        "alpha must be strictly between 0.0 and 1"
    )
    assert message(lambda: feasibility.z_sum(0.05, 3, 1.0)) == (
        "power must be strictly between 0.0 and 1"
    )


def test_mean_formulas_agree_with_each_other_and_their_boundaries():
    z = feasibility.z_sum(0.05, 2, 0.8)
    assert feasibility.n_required_mean(
        0.3, alpha=0.05, k_max=2, power=0.8
    ) == math.ceil((z / 0.3) ** 2)
    assert feasibility.mde_mean(
        100, alpha=0.05, k_max=2, power=0.8
    ) == pytest.approx(z / 10)
    assert feasibility.mde_mean(
        1, alpha=0.05, k_max=2, power=0.8
    ) == pytest.approx(z)
    assert math.isnan(feasibility.mde_mean(0, alpha=0.05, k_max=2, power=0.8))
    assert message(
        lambda: feasibility.n_required_mean(0, alpha=0.05, k_max=2, power=0.8)
    ) == ("delta must be a finite number greater than 0")
    critical = stats.norm.ppf(1 - 0.05 / 4)
    assert feasibility.power_mean(
        0.3, 100, alpha=0.05, k_max=2
    ) == pytest.approx(stats.norm.cdf(3.0 - critical), rel=1e-12)
    assert feasibility.power_mean(0.3, 0, alpha=0.05, k_max=2) == pytest.approx(
        stats.norm.cdf(-critical), rel=1e-12
    )
    assert message(
        lambda: feasibility.power_mean(0, 5, alpha=0.05, k_max=2)
    ) == ("delta must be a finite number greater than 0")
    assert message(
        lambda: feasibility.power_mean(0.3, 5, alpha=2, k_max=2)
    ) == ("alpha must be strictly between 0.0 and 1")


def test_correlation_formulas_follow_fisher_z():
    z = feasibility.z_sum(0.05, 1, 0.8)
    assert feasibility.n_required_correlation(
        0.3, alpha=0.05, k_max=1, power=0.8
    ) == math.ceil((z / math.atanh(0.3)) ** 2 + 3)
    assert feasibility.mde_correlation(103, alpha=0.05, k_max=1, power=0.8) == (
        pytest.approx(math.tanh(z / 10))
    )
    assert feasibility.mde_correlation(4, alpha=0.05, k_max=1, power=0.8) == (
        pytest.approx(math.tanh(z))
    )
    assert math.isnan(
        feasibility.mde_correlation(3, alpha=0.05, k_max=1, power=0.8)
    )
    assert message(
        lambda: feasibility.n_required_correlation(
            1.0, alpha=0.05, k_max=1, power=0.8
        )
    ) == ("r must be strictly between 0.0 and 1")


def test_two_proportions_matches_the_pooled_variance_formula():
    p1, p2, alpha, k, power = 0.3, 0.5, 0.05, 2, 0.8
    a = stats.norm.ppf(1 - alpha / (2 * k))
    b = stats.norm.ppf(power)
    pooled = (p1 + p2) / 2
    root = a * math.sqrt(2 * pooled * (1 - pooled)) + b * math.sqrt(
        p1 * (1 - p1) + p2 * (1 - p2)
    )
    expected = math.ceil(root**2 / (p1 - p2) ** 2)
    assert (
        feasibility.n_required_two_proportions(
            p1, p2, alpha=alpha, k_max=k, power=power
        )
        == expected
    )
    for name, args in (
        ("p1", (1.0, 0.5)),
        ("p2", (0.3, 0.0)),
    ):
        assert message(
            lambda args=args: feasibility.n_required_two_proportions(
                *args, alpha=0.05, k_max=2, power=0.8
            )
        ) == (f"{name} must be strictly between 0.0 and 1")
    assert message(
        lambda: feasibility.n_required_two_proportions(
            0.3, 0.3, alpha=0.05, k_max=2, power=0.8
        )
    ) == ("p1 and p2 must differ")
    assert message(
        lambda: feasibility.n_required_two_proportions(
            0.3, 0.5, alpha=1.5, k_max=2, power=0.8
        )
    ) == ("alpha must be strictly between 0.0 and 1")
    assert message(
        lambda: feasibility.n_required_two_proportions(
            0.3, 0.5, alpha=0.05, k_max=2, power=1.5
        )
    ) == ("power must be strictly between 0.0 and 1")


@pytest.mark.parametrize("n", [2, 10, 250])
@pytest.mark.parametrize("confidence", [0.9, 0.95, 0.99])
def test_wilson_half_width_equals_half_the_wilson_interval(n, confidence):
    low, high = certificates.wilson_interval(n // 2, n, confidence)
    assert feasibility.wilson_half_width(0.5, n, confidence) == pytest.approx(
        (high - low) / 2, rel=1e-12
    )


def test_wilson_half_width_at_the_edges():
    assert math.isinf(feasibility.wilson_half_width(0.5, 0))
    low, high = certificates.wilson_interval(0, 1, 0.95)
    assert feasibility.wilson_half_width(0.0, 1) == pytest.approx(
        (high - low) / 2
    )
    z = stats.norm.ppf(0.975)
    expected = z * math.sqrt(0.25 / 4 + z * z / 64) / (1 + z * z / 4)
    assert feasibility.wilson_half_width(0.5, 4) == pytest.approx(
        expected, rel=1e-12
    )


def test_assess_boundaries_and_record():
    below = feasibility.assess(
        n_confirmation=feasibility.MIN_CONFIRMATION_ROWS - 1,
        min_effect=0.3,
        alpha=0.05,
        k_max=3,
        power=0.8,
    )
    at = feasibility.assess(
        n_confirmation=feasibility.MIN_CONFIRMATION_ROWS,
        min_effect=0.3,
        alpha=0.05,
        k_max=3,
        power=0.8,
    )
    assert below.verdict is feasibility.Verdict.INFEASIBLE
    assert at.verdict is not feasibility.Verdict.INFEASIBLE
    assert at.n_confirmation == feasibility.MIN_CONFIRMATION_ROWS
    assert at.power_at_min_effect == pytest.approx(
        feasibility.power_mean(
            0.3, feasibility.MIN_CONFIRMATION_ROWS, alpha=0.05, k_max=3
        )
    )
    assert at.n_required == feasibility.n_required_mean(
        0.3, alpha=0.05, k_max=3, power=0.8
    )


def test_a_wide_minority_interval_downgrades_only_at_a_strict_inequality():
    plenty = {
        "n_confirmation": 5000,
        "min_effect": 0.3,
        "alpha": 0.05,
        "k_max": 3,
        "power": 0.8,
    }
    width = feasibility.wilson_half_width(0.5, 40)
    exact = feasibility.assess(**plenty, n_minority=40, max_half_width=width)
    tighter = feasibility.assess(
        **plenty, n_minority=40, max_half_width=width / 2
    )
    assert exact.verdict is feasibility.Verdict.FEASIBLE
    assert tighter.verdict is feasibility.Verdict.DOWNGRADE
    assert (
        feasibility.assess(**plenty, n_minority=40).verdict
        is feasibility.Verdict.FEASIBLE
    )
    assert (
        feasibility.assess(**plenty, max_half_width=0.001).verdict
        is feasibility.Verdict.FEASIBLE
    )


def test_the_record_of_a_verdict_names_its_verdict_and_nulls_non_finite():
    report = feasibility.assess(
        n_confirmation=5,
        min_effect=0.3,
        alpha=0.05,
        k_max=3,
        power=0.8,
    )
    record = report.to_dict()
    assert type(record["verdict"]) is str
    assert record["verdict"] == "infeasible"
    assert "VERDICT" not in record
    assert record["n_confirmation"] == 5
    assert record["mde"] is None or math.isfinite(record["mde"])
    nan = feasibility.Feasibility(
        feasibility.Verdict.FEASIBLE, "s", 1, 1, math.nan, math.inf, "r"
    ).to_dict()
    assert nan["mde"] is None and nan["power_at_min_effect"] is None
