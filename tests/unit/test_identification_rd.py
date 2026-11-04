"""Regression discontinuity: recovery, coverage, diagnostics."""

import numpy as np
import pandas as pd
import pytest

from scmle.data.identification import rd


def world(
    n: int,
    seed: int,
    tau: float = 1.0,
    sort: float = 0.0,
    slope: float = 2.0,
    curve: float = 1.0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, n)
    if sort:
        # units just below the cutoff move above it
        move = (x < 0) & (x > -sort) & (rng.uniform(size=n) < 0.7)
        x = np.where(move, -x, x)
    y = (
        slope * x
        + curve * x**2
        + tau * (x >= 0)
        + rng.normal(scale=0.5, size=n)
    )
    z = 0.5 * x + rng.normal(scale=0.3, size=n)
    return pd.DataFrame({"x": x, "y": y, "z": z})


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
    x = np.linspace(-1, 1, 400)
    y = 1.0 + 2.0 * x + 3.0 * x**2 + 5.0 * (x >= 0)
    out = rd.estimate(
        pd.DataFrame({"x": x, "y": y}), "x", "y", 0.0, bandwidth=0.8
    )
    assert out.bias_corrected == pytest.approx(5.0, abs=1e-8)
    assert out.estimate == pytest.approx(5.0, abs=0.2)


def test_sorting_at_the_cutoff_trips_the_density_check():
    clean = rd.estimate(world(3000, 1), "x", "y", 0.0)
    sorted_ = rd.estimate(world(3000, 1, sort=0.15), "x", "y", 0.0)
    assert clean.density_p > 0.01
    assert sorted_.density_p < 0.001


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
        rd.estimate(frame.assign(y=np.nan), "x", "y", 0.0)
