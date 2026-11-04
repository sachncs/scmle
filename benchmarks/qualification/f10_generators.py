"""Simulated worlds with a known causal effect for the F10 certificates.

Every world is synthetic, including its covariates: a corpus of real
covariates (the plan's "planted effect on real `X`") needs data this
repository does not ship, so the certificates say what they measured (these
worlds) and nothing about real tables. Each function takes a numpy
`Generator`, so a replicate is reproducible from its seed. Knobs:

* confounding strength, overlap (`p_min`), hidden confounders shared or not
  shared with a negative-control outcome (`backdoor`);
* instrument strength and a direct effect that breaks exclusion (`iv`);
* a differential linear trend that breaks parallel trends (`panel`);
* sorting across the cutoff (`discontinuity`);
* donors, periods and a level of noise (`synthetic`).
"""

import math

import numpy as np
import pandas as pd

COVARIATES = 5


def covariates(rng: np.random.Generator, n: int) -> np.ndarray:
    """Correlated standard-normal covariates (AR(1) correlation 0.3)."""
    scale = 0.3 ** np.abs(
        np.subtract.outer(range(COVARIATES), range(COVARIATES))
    )
    return rng.standard_normal((n, COVARIATES)) @ np.linalg.cholesky(scale).T


def outcome_surface(x: np.ndarray, nonlinear: bool) -> np.ndarray:
    """Baseline outcome as a function of the covariates."""
    if nonlinear:
        return np.sin(x[:, 0]) + 0.5 * x[:, 1] ** 2 + 0.5 * x[:, 2]
    return 1.0 * x[:, 0] - 0.5 * x[:, 1] + 0.3 * x[:, 2]


def backdoor(
    rng: np.random.Generator,
    n: int,
    tau: float = 2.0,
    p_min: float = 0.1,
    hidden_d: float = 0.0,
    hidden_y: float = 0.0,
    hidden_nco: float | None = None,
    nonlinear: bool = False,
) -> pd.DataFrame:
    """Observational cross-section with a constant effect `tau`.

    The propensity is `expit(s x0 + hidden_d u)` with `x0` clipped to
    [-2, 2] and `s = logit(1 - p_min) / 2`, so `p_min` is the smallest
    propensity when `hidden_d = 0`. `hidden_y` is the confounder's effect on
    the outcome, `hidden_nco` (default `hidden_y`) on the negative-control
    outcome, which has no treatment effect.
    """
    x = covariates(rng, n)
    u = rng.standard_normal(n)
    strength = math.log((1 - p_min) / p_min) / 2 if p_min < 0.5 else 0.0
    logit = strength * np.clip(x[:, 0], -2, 2) + 0.3 * x[:, 1] + hidden_d * u
    t = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(float)
    base = outcome_surface(x, nonlinear)
    y = tau * t + base + hidden_y * u + rng.standard_normal(n)
    shared = hidden_y if hidden_nco is None else hidden_nco
    nco = base + shared * u + rng.standard_normal(n)
    frame = pd.DataFrame(x, columns=[f"x{j}" for j in range(COVARIATES)])
    frame["t"], frame["y"], frame["nco"] = t, y, nco
    frame["region"] = np.where(rng.uniform(size=n) < 0.5, "n", "s")
    return frame


def randomized(
    rng: np.random.Generator, n: int, tau: float = 2.0
) -> pd.DataFrame:
    """A randomized experiment with covariate-driven outcomes."""
    x = covariates(rng, n)
    t = (rng.uniform(size=n) < 0.5).astype(float)
    y = tau * t + outcome_surface(x, False) + rng.standard_normal(n)
    frame = pd.DataFrame(x, columns=[f"x{j}" for j in range(COVARIATES)])
    frame["t"], frame["y"] = t, y
    return frame


def iv(
    rng: np.random.Generator,
    n: int,
    strength: float,
    beta: float = 2.0,
    direct: float = 0.0,
) -> pd.DataFrame:
    """Endogenous treatment with one instrument."""
    z = rng.standard_normal(n)
    u = rng.standard_normal(n)
    d = strength * z + u + rng.standard_normal(n)
    y = beta * d + direct * z + 1.5 * u + rng.standard_normal(n)
    return pd.DataFrame({"z": z, "d": d, "y": y})


def panel(
    rng: np.random.Generator,
    treated: int,
    never: int,
    periods: int = 6,
    adopt: int = 3,
    tau: float = 1.5,
    drift: float = 0.0,
    second_group: bool = False,
) -> pd.DataFrame:
    """Unit-by-period panel with a differential linear trend `drift`."""
    groups = [(float(adopt), treated)]
    if second_group:
        groups.append((float(adopt + 1), treated))
    groups.append((math.nan, never))
    rows = []
    uid = 0
    for g, count in groups:
        for _ in range(count):
            level = rng.normal()
            for t in range(periods):
                y = level + 0.3 * t + rng.normal(scale=0.7)
                nco = level + 0.3 * t + rng.normal(scale=0.7)
                if not math.isnan(g):
                    y += drift * t
                    if t >= g:
                        y += tau
                rows.append((uid, t, g, y, nco))
            uid += 1
    return pd.DataFrame(rows, columns=["id", "t", "g", "y", "nco"])


def discontinuity(
    rng: np.random.Generator,
    n: int,
    tau: float = 1.0,
    sorting: float = 0.0,
    jump: float = 0.0,
) -> pd.DataFrame:
    """Sharp RD world.

    `sorting` is the share of units in a window just below the cutoff that
    move above it (manipulation); `jump` is a discontinuity in a
    predetermined covariate `z` that should have none.
    """
    x = rng.uniform(-1, 1, n)
    if sorting:
        move = (x < 0) & (x > -0.2) & (rng.uniform(size=n) < sorting)
        x = np.where(move, -x, x)
    y = 2 * x + x**2 + tau * (x >= 0) + rng.normal(scale=0.5, size=n)
    z = 0.5 * x + jump * (x >= 0) + rng.normal(scale=0.3, size=n)
    return pd.DataFrame({"x": x, "y": y, "z": z})


def synthetic(
    rng: np.random.Generator,
    donors: int = 24,
    pre: int = 12,
    post: int = 4,
    tau: float = 0.0,
) -> pd.DataFrame:
    """Factor-model panel with one treated unit (unit 0)."""
    factors = rng.standard_normal((pre + post, 2)).cumsum(axis=0) * 0.5
    loadings = rng.uniform(0, 1, (donors + 1, 2))
    rows = []
    for u in range(donors + 1):
        for t in range(pre + post):
            y = factors[t] @ loadings[u] + rng.normal(scale=0.1)
            if u == 0 and t >= pre:
                y += tau
            rows.append((u, t, y))
    return pd.DataFrame(rows, columns=["u", "t", "y"])
