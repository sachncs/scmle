"""Regression discontinuity: recovery, coverage, diagnostics."""

import numpy
import pandas
import pytest
from scipy import stats

from scmle.causal.identification import rd


def world(
    n: int,
    seed: int,
    tau: float = 1.0,
    sort: float = 0.0,
    slope: float = 2.0,
    curve: float = 1.0,
) -> pandas.DataFrame:
    rng = numpy.random.default_rng(seed)
    x = rng.uniform(-1, 1, n)
    if sort:
        # units just below the cutoff move above it
        move = (x < 0) & (x > -sort) & (rng.uniform(size=n) < 0.7)
        x = numpy.where(move, -x, x)
    y = (
        slope * x
        + curve * x**2
        + tau * (x >= 0)
        + rng.normal(scale=0.5, size=n)
    )
    z = 0.5 * x + rng.normal(scale=0.3, size=n)
    return pandas.DataFrame({"x": x, "y": y, "z": z})


def test_the_jump_is_recovered_and_the_interval_covers():
    hits = 0
    for seed in range(60):
        out = rd.estimate(world(1500, seed), "x", "y", 0.0)
        hits += out.low <= 1.0 <= out.high
    assert hits >= 52


def test_no_effect_world_does_not_exclude_zero_often():
    rejects = sum(
        rd.estimate(world(1500, s, tau=0.0), "x", "y", 0.0).low > 0
        or rd.estimate(world(1500, s, tau=0.0), "x", "y", 0.0).high < 0
        for s in range(40)
    )
    assert rejects <= 6


def test_a_known_polynomial_is_fitted_exactly_without_noise():
    x = numpy.linspace(-1, 1, 400)
    y = 1.0 + 2.0 * x + 3.0 * x**2 + 5.0 * (x >= 0)
    out = rd.estimate(
        pandas.DataFrame({"x": x, "y": y}), "x", "y", 0.0, bandwidth=0.8
    )
    assert out.corrected == pytest.approx(5.0, abs=1e-8)
    assert out.estimate == pytest.approx(5.0, abs=0.2)


def test_sorting_at_the_cutoff_trips_the_density_check():
    clean = rd.estimate(world(3000, 1), "x", "y", 0.0)
    sorted_ = rd.estimate(world(3000, 1, sort=0.15), "x", "y", 0.0)
    assert clean.density > 0.01
    assert sorted_.density < 0.001


def test_placebo_cutoffs_and_balance_and_sensitivity_are_reported():
    out = rd.estimate(world(3000, 2), "x", "y", 0.0, covariates=("z",))
    assert set(out.placebos) == {"left_median", "right_median"}
    assert all(p["p"] > 0.001 for p in out.placebos.values())
    assert out.balance["z"]["p"] > 0.001
    assert set(out.sensitivity) == {"0.5", "0.75", "1.25", "1.5"}
    assert out.n_left > 0 and out.n_right > 0
    assert out.to_dict()["bandwidth"] == out.bandwidth


def test_small_or_missing_data_is_refused():
    frame = world(1000, 3)
    with pytest.raises(ValueError, match="each side"):
        rd.estimate(frame, "x", "y", 5.0)
    with pytest.raises(ValueError, match="missing"):
        rd.estimate(frame.assign(y=numpy.nan), "x", "y", 0.0)


def test_a_local_fit_uses_the_leverage_corrected_hc3_variance():
    rng = numpy.random.default_rng(4)
    x = rng.uniform(0.0, 1.0, 40)
    y = 1.0 + x + rng.normal(scale=0.3, size=40)
    fit = rd.polyfit(x, y, 0.8, 2)
    weight = numpy.clip(1.0 - x / 0.8, 0.0, None)
    keep = weight > 0
    design = numpy.vander(x[keep], 3, increasing=True)
    w = weight[keep]
    bread = numpy.linalg.inv(design.T @ (design * w[:, None]))
    coef = bread @ (design.T @ (w * y[keep]))
    resid = y[keep] - design @ coef
    leverage = w * numpy.einsum("ij,jk,ik->i", design, bread, design)
    score = design * (w * resid / (1.0 - leverage))[:, None]
    expected = (bread @ score.T @ score @ bread)[0, 0]
    assert fit.n == int(keep.sum())
    assert fit.intercept == pytest.approx(coef[0])
    assert fit.variance == pytest.approx(expected)
    # the intercept is sum(a_i y_i); its Satterthwaite degrees of freedom
    # are (sum a_i^2)^2 / sum a_i^4, fewer than the points it uses
    a = (bread @ (design * w[:, None]).T)[0]
    assert fit.dof == pytest.approx((a**2).sum() ** 2 / (a**4).sum())
    assert 1 < fit.dof < fit.n


def test_a_jump_is_referred_to_student_t_with_welch_degrees_of_freedom():
    rng = numpy.random.default_rng(5)
    x = rng.uniform(-1.0, 1.0, 300)
    y = x + rng.normal(scale=0.5, size=300)
    found = rd.discontinuity(x, y, 0.0, 0.4, 0.95)
    sides = [rd.polyfit(x[m], y[m], 0.4, 2) for m in (x < 0, x >= 0)]
    left, right = sides
    dof = (left.variance + right.variance) ** 2 / (
        left.variance**2 / left.dof + right.variance**2 / right.dof
    )
    statistic = found["bias_corrected"] / found["se"]
    assert found["dof"] == pytest.approx(dof)
    assert found["p"] == pytest.approx(2 * stats.t.sf(abs(statistic), dof))
    half = stats.t.ppf(0.975, dof) * found["se"]
    assert found["high"] - found["low"] == pytest.approx(2 * half)
    # heavier tails than the normal: a wider interval, a larger p-value
    assert half > stats.norm.ppf(0.975) * found["se"]
