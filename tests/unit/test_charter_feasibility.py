"""Feasibility: closed-form power, minimum detectable effect and verdicts."""

import math

import pytest
from scipy import stats

import scmle.data.charter.feasibility


def two_sided_z(alpha, k):
    return stats.norm.ppf(1 - alpha / (2 * k))


def test_mean_effect_sample_size_matches_the_closed_form():
    expected = math.ceil(
        ((stats.norm.ppf(0.975) + stats.norm.ppf(0.8)) / 0.5) ** 2
    )
    assert (
        scmle.data.charter.feasibility.n_required_mean(
            0.5, alpha=0.05, k_max=1, power=0.8
        )
        == expected
    )
    assert expected == 32


def test_correlation_sample_size_matches_cohens_table_value():
    assert (
        scmle.data.charter.feasibility.n_required_correlation(
            0.3, alpha=0.05, k_max=1, power=0.8
        )
        == 85
    )


@pytest.mark.parametrize("k_max", [1, 3, 10, 50])
@pytest.mark.parametrize("delta", [0.1, 0.3, 0.8])
def test_mde_inverts_the_mean_sample_size_to_1e_9(k_max, delta):
    n = scmle.data.charter.feasibility.n_required_mean(
        delta, alpha=0.05, k_max=k_max, power=0.8
    )
    zsum = two_sided_z(0.05, k_max) + stats.norm.ppf(0.8)
    assert scmle.data.charter.feasibility.mde_mean(
        n, alpha=0.05, k_max=k_max, power=0.8
    ) == pytest.approx(zsum / math.sqrt(n), abs=1e-12)
    # One more row never raises the MDE; the required n is the smallest n
    # whose MDE is at most delta.
    assert (
        scmle.data.charter.feasibility.mde_mean(
            n, alpha=0.05, k_max=k_max, power=0.8
        )
        <= delta + 1e-12
    )
    assert (
        scmle.data.charter.feasibility.mde_mean(
            n - 1, alpha=0.05, k_max=k_max, power=0.8
        )
        > delta
    )


@pytest.mark.parametrize("r", [0.1, 0.3, 0.5])
def test_mde_correlation_is_the_inverse_of_the_sample_size(r):
    n = scmle.data.charter.feasibility.n_required_correlation(
        r, alpha=0.05, k_max=5, power=0.8
    )
    assert (
        scmle.data.charter.feasibility.mde_correlation(
            n, alpha=0.05, k_max=5, power=0.8
        )
        <= r + 1e-12
    )
    assert (
        scmle.data.charter.feasibility.mde_correlation(
            n - 1, alpha=0.05, k_max=5, power=0.8
        )
        > r
    )


def test_two_proportion_sample_size_matches_the_textbook_formula():
    p1, p2, alpha, power = 0.5, 0.6, 0.05, 0.8
    pbar = (p1 + p2) / 2
    numerator = (
        stats.norm.ppf(1 - alpha / 2) * math.sqrt(2 * pbar * (1 - pbar))
        + stats.norm.ppf(power) * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    ) ** 2
    expected = math.ceil(numerator / (p1 - p2) ** 2)
    assert (
        scmle.data.charter.feasibility.n_required_two_proportions(
            p1, p2, alpha=alpha, k_max=1, power=power
        )
        == expected
    )
    assert expected == 388


def test_power_at_the_required_size_meets_the_target():
    n = scmle.data.charter.feasibility.n_required_mean(
        0.4, alpha=0.05, k_max=4, power=0.8
    )
    achieved = scmle.data.charter.feasibility.power_mean(
        0.4, n, alpha=0.05, k_max=4
    )
    assert achieved >= 0.8
    assert (
        scmle.data.charter.feasibility.power_mean(
            0.4, n - 1, alpha=0.05, k_max=4
        )
        < 0.8 + 1e-3
    )


def test_bonferroni_at_k_max_is_more_demanding_than_a_single_test():
    single = scmle.data.charter.feasibility.n_required_mean(
        0.3, alpha=0.05, k_max=1, power=0.8
    )
    many = scmle.data.charter.feasibility.n_required_mean(
        0.3, alpha=0.05, k_max=20, power=0.8
    )
    assert many > single


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"delta": 0.0}, "delta"),
        ({"alpha": 1.0}, "alpha"),
        ({"k_max": 0}, "k_max"),
        ({"power": 1.0}, "power"),
    ],
)
def test_invalid_arguments_are_rejected(kwargs, match):
    base = {"delta": 0.3, "alpha": 0.05, "k_max": 3, "power": 0.8}
    base.update(kwargs)
    with pytest.raises(ValueError, match=match):
        scmle.data.charter.feasibility.n_required_mean(
            base["delta"],
            alpha=base["alpha"],
            k_max=base["k_max"],
            power=base["power"],
        )


def test_wilson_half_width_shrinks_with_the_minority_count():
    small = scmle.data.charter.feasibility.wilson_half_width(0.5, 20)
    large = scmle.data.charter.feasibility.wilson_half_width(0.5, 2000)
    assert 0 < large < small < 0.5


def feasibility(n_confirmation, **overrides):
    values = {
        "n_confirmation": n_confirmation,
        "min_effect": 0.3,
        "alpha": 0.05,
        "k_max": 5,
        "power": 0.8,
    }
    values.update(overrides)
    return scmle.data.charter.feasibility.assess(**values)


def test_verdict_is_feasible_when_the_set_is_large_enough():
    required = scmle.data.charter.feasibility.n_required_mean(
        0.3, alpha=0.05, k_max=5, power=0.8
    )
    report = feasibility(required)
    assert report.verdict is scmle.data.charter.feasibility.Verdict.FEASIBLE
    assert report.scope == "confirmatory"
    assert report.mde <= 0.3 + 1e-12


def test_verdict_downgrades_when_underpowered_but_testable():
    required = scmle.data.charter.feasibility.n_required_mean(
        0.3, alpha=0.05, k_max=5, power=0.8
    )
    report = feasibility(required - 1)
    assert report.verdict is scmle.data.charter.feasibility.Verdict.DOWNGRADE
    assert report.scope == "exploratory_unconfirmed"
    assert report.mde > 0.3


def test_verdict_is_infeasible_without_a_testable_confirmation_set():
    report = feasibility(
        scmle.data.charter.feasibility.MIN_CONFIRMATION_ROWS - 1
    )
    assert report.verdict is scmle.data.charter.feasibility.Verdict.INFEASIBLE
    assert report.scope == "exploratory_unconfirmed"
    assert math.isnan(report.mde) or report.mde > 0.3


def test_minority_count_can_make_a_large_set_underpowered():
    required = scmle.data.charter.feasibility.n_required_mean(
        0.3, alpha=0.05, k_max=5, power=0.8
    )
    report = feasibility(
        required * 10,
        n_minority=scmle.data.charter.feasibility.MIN_CONFIRMATION_ROWS,
        max_half_width=0.05,
    )
    assert report.verdict is scmle.data.charter.feasibility.Verdict.DOWNGRADE
    assert "minority" in report.reason


def test_report_is_json_serialisable():
    import json

    payload = feasibility(500).to_dict()
    assert json.loads(json.dumps(payload))["verdict"] in {
        "feasible",
        "downgrade",
        "infeasible",
    }
