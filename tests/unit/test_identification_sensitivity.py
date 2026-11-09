"""E-value and Cinelli-Hazlett bounds against worked examples."""

import math

import numpy
import pandas
import pytest

from scmle.data.identification import sensitivity


def test_evalue_matches_the_papers_worked_example():
    assert sensitivity.evalue(3.9) == pytest.approx(7.26, abs=0.005)
    assert sensitivity.evalue(1.0) == 1.0
    assert sensitivity.evalue(0.8) == pytest.approx(1.809, abs=0.001)
    assert sensitivity.evalue(0.8) == pytest.approx(sensitivity.evalue(1.25))


def test_evalue_of_the_interval_uses_the_limit_nearest_the_null():
    both = sensitivity.evalue_interval(1.8, 1.2, 2.7)
    assert both["point"] == pytest.approx(sensitivity.evalue(1.8))
    assert both["limit"] == pytest.approx(sensitivity.evalue(1.2))
    assert sensitivity.evalue_interval(0.5, 0.3, 0.8)["limit"] == pytest.approx(
        sensitivity.evalue(0.8)
    )
    assert sensitivity.evalue_interval(1.3, 0.9, 1.9)["limit"] == 1.0
    with pytest.raises(ValueError):
        sensitivity.evalue_interval(5.0, 1.0, 2.0)
    with pytest.raises(ValueError):
        sensitivity.evalue(0.0)


def test_smd_conversion():
    assert sensitivity.smd_to_rr(0.5) == pytest.approx(math.exp(0.455))


def test_robustness_values_reproduce_the_sensemakr_readme_example():
    t, dof = 4.18445, 783
    assert sensitivity.partial_r2(t, dof) == pytest.approx(0.02187, abs=2e-5)
    assert sensitivity.robustness_value(t, dof, 1.0) == pytest.approx(
        0.13878, abs=2e-4
    )
    assert sensitivity.robustness_value(t, dof, 1.0, 0.05) == pytest.approx(
        0.07626, abs=2e-4
    )
    assert sensitivity.robustness_value(0.5, 100, 1.0, 0.05) == 0.0


def synthetic(n: int, seed: int, a_d: float, a_y: float, tau: float = 1.0):
    """Treatment and outcome with an unobserved confounder `u` of strength."""
    rng = numpy.random.default_rng(seed)
    x = rng.standard_normal((n, 3))
    u = rng.standard_normal(n)
    d = x @ numpy.array([0.8, 0.3, 0.1]) + a_d * u + rng.standard_normal(n)
    y = (
        tau * d
        + x @ numpy.array([1.0, 0.5, 0.2])
        + a_y * u
        + rng.standard_normal(n)
    )
    frame = pandas.DataFrame(x, columns=["x1", "x2", "x3"])
    frame["d"], frame["y"], frame["u"] = d, y, u
    return frame


def test_the_bias_formula_is_the_exact_omitted_variable_bias():
    frame = synthetic(5000, 0, 0.7, 0.9)
    base = ("x1", "x2", "x3")
    short = sensitivity.ols_fit(frame, "d", "y", base)
    full = sensitivity.ols_fit(frame, "d", "y", (*base, "u"))
    r2dz, r2yz = full.benchmarks["u"]
    true_bias = short.estimate - full.estimate
    assert abs(true_bias) > 0.1
    assert abs(true_bias) == pytest.approx(
        sensitivity.bias_bound(short, r2yz, r2dz), rel=1e-9
    )
    se_adjusted = sensitivity.adjusted_interval(short, r2yz, r2dz)
    assert se_adjusted[0] < full.estimate < se_adjusted[1]


def test_the_bound_holds_when_the_confounder_is_planted_inside_the_benchmark():
    hits = 0
    runs = 80
    for seed in range(runs):
        frame = synthetic(1500, seed, 0.4, 0.4)
        report = sensitivity.assess(
            frame.drop(columns="u"), "d", "y", ("x1", "x2", "x3"), strength=1.0
        )
        hits += report.low <= 1.0 <= report.high
    assert hits >= 0.95 * runs


def test_assess_flags_a_fragile_estimate_and_a_robust_one():
    robust = sensitivity.assess(
        synthetic(2000, 1, 0.0, 0.0, tau=3.0).drop(columns="u"),
        "d",
        "y",
        ("x1", "x2", "x3"),
        "fixed",
        0.01,
    )
    fragile = sensitivity.assess(
        synthetic(2000, 2, 0.0, 0.0, tau=0.05).drop(columns="u"),
        "d",
        "y",
        ("x1", "x2", "x3"),
        "fixed",
        0.3,
    )
    assert robust.holds and not fragile.holds
    assert robust.rv > fragile.rv
    assert "fixed" in robust.benchmark
    assert (
        sensitivity.assess(
            synthetic(500, 3, 0, 0).drop(columns="u"), "d", "y", ("x1",)
        ).covariate
        == "x1"
    )
    with pytest.raises(ValueError, match="at least one covariate"):
        sensitivity.assess(synthetic(100, 4, 0, 0), "d", "y", ())


def test_benchmark_multiples_that_exceed_the_data_are_refused():
    with pytest.raises(ValueError, match="too large"):
        sensitivity.benchmark_strength(0.4, 0.2, 5.0, 1.0)
    r2yz, r2dz = sensitivity.benchmark_strength(0.1, 0.05, 1.0, 1.0)
    assert 0 < r2dz < 1 and r2yz > 0.05 / 0.95 - 1e-9
    with pytest.raises(ValueError):
        sensitivity.bias_bound(
            sensitivity.ols_fit(synthetic(100, 5, 0, 0), "d", "y", ("x1",)),
            1.2,
            0.1,
        )
