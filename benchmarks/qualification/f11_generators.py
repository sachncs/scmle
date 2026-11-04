"""Simulated worlds with known truth for the F11 gates (canon P1).

Every world has an analytic referent: the Bayes threshold, the binormal net
benefit, the value of a bandit policy, the true gap between groups, the true
mean difference of an experiment. Generators take a `numpy` generator so the
certifier controls the seeds; nothing here reads the repository's own
estimators.
"""

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats

Z = stats.norm


# --- thresholds ------------------------------------------------------------


def calibration_world(
    rng: np.random.Generator, n: int, skew: float
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """True risk `p ~ Beta(1, 6)`, labels `Bern(p)`, score `p ** skew`.

    `skew = 1` is calibrated; any other value is a monotone distortion, so
    the Bayes rule on the true risk stays optimal for the score.
    """
    p = rng.beta(1.0, 6.0, n)
    y = (rng.random(n) < p).astype(float)
    return p**skew, p, y


def beta_cost(
    p: NDArray[np.float64], acts: NDArray[np.float64], c_fp: float, c_fn: float
) -> float:
    """Exact expected cost per row of a rule, given the true risks `p`."""
    return float(np.mean(acts * (1 - p) * c_fp + (1 - acts) * p * c_fn))


# --- net benefit -----------------------------------------------------------


def binormal(
    rng: np.random.Generator,
    n: int,
    prevalence: float,
    delta: float,
    cluster_size: int = 1,
    cluster_sd: float = 0.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.intp]]:
    """Score `N(delta y + u_g, 1)` with a shared cluster shift `u_g`."""
    y = (rng.random(n) < prevalence).astype(float)
    groups = np.arange(n) // cluster_size
    count = int(np.max(groups)) + 1
    shift = rng.normal(0.0, cluster_sd, count)[groups]
    return rng.normal(delta * y + shift, 1.0), y, groups


def binormal_nb(
    prevalence: float,
    delta: float,
    cut: float,
    ts: NDArray[np.float64],
    total_sd: float = 1.0,
) -> NDArray[np.float64]:
    """True net benefit of `score >= cut` at each threshold `t`."""
    tpr = 1.0 - Z.cdf((cut - delta) / total_sd)
    fpr = 1.0 - Z.cdf(cut / total_sd)
    w = ts / (1.0 - ts)
    result: NDArray[np.float64] = (
        prevalence * tpr - (1.0 - prevalence) * fpr * w
    )
    return result


def binormal_differences(
    prevalence: float,
    delta: float,
    cut: float,
    ts: NDArray[np.float64],
    total_sd: float = 1.0,
) -> NDArray[np.float64]:
    """True `NB_rule - NB_none` and `NB_rule - NB_all`, shape `(2, grid)`."""
    rule = binormal_nb(prevalence, delta, cut, ts, total_sd)
    w = ts / (1.0 - ts)
    everyone = prevalence - (1.0 - prevalence) * w
    return np.stack([rule, rule - everyone])


# --- off-policy evaluation -------------------------------------------------

EFFECT_UP = 0.25
EFFECT_DOWN = -0.10
BASE_REWARD = 0.30


def bandit(
    rng: np.random.Generator,
    n: int,
    sharpness: float,
    floor: float = 0.05,
    blind_above: float | None = None,
) -> dict[str, Any]:
    """A logged binary-action bandit with a covariate-dependent propensity.

    `mu(1 | x) = clip(sigmoid(sharpness x0), floor, 1 - floor)`; with
    `blind_above` the logging policy never takes action 1 where `x0` exceeds
    it (a planted support violation for a target that treats `x0 > 0`).
    Rewards are Bernoulli with mean `0.30` for action 0 and `0.30 + 0.25`
    (`x0 > 0`) or `0.30 - 0.10` (otherwise) for action 1.
    """
    x = rng.normal(size=(n, 2))
    mu1 = np.clip(1 / (1 + np.exp(-sharpness * x[:, 0])), floor, 1 - floor)
    if blind_above is not None:
        mu1 = np.where(x[:, 0] > blind_above, 0.0, mu1)
    action = (rng.random(n) < mu1).astype(int)
    effect = np.where(x[:, 0] > 0, EFFECT_UP, EFFECT_DOWN)
    mean = BASE_REWARD + action * effect
    reward = (rng.random(n) < mean).astype(float)
    return {
        "x": x,
        "action": action,
        "reward": reward,
        "behaviour": np.column_stack([1 - mu1, mu1]),
    }


def target_policy(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Treat iff `x0 > 0`, as an `(n, 2)` probability matrix."""
    treat = (x[:, 0] > 0).astype(float)
    return np.column_stack([1 - treat, treat])


def target_value() -> float:
    """`V(pi) = 0.30 + 0.25 P(x0 > 0)`, exact."""
    return BASE_REWARD + EFFECT_UP * 0.5


def outcome_features(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Features for the outcome model, including the step."""
    return np.column_stack([x, (x[:, 0] > 0).astype(float)])


# --- fairness --------------------------------------------------------------


def fairness_world(
    rng: np.random.Generator,
    n: int,
    gap: float,
    cut: float = 0.0,
    tpr: float = 0.7,
    prevalence: float = 0.3,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.intp]]:
    """Two groups with true-positive rates `tpr` and `tpr - gap` at `cut`.

    Positives of group `g` have score `N(cut + z(tpr_g), 1)` so that
    `P(score >= cut | y = 1, g) = tpr_g` exactly; negatives are `N(cut - 1,
    1)` in both groups, so only the true-positive rate differs. Returns
    scores, labels and groups; the rule is `score >= cut`.
    """
    groups = rng.integers(0, 2, n)
    y = (rng.random(n) < prevalence).astype(float)
    rate = np.where(groups == 0, tpr, tpr - gap)
    centre = np.where(y == 1, cut + Z.ppf(rate), cut - 1.0)
    return rng.normal(centre, 1.0), y, groups


def threshold_world(
    rng: np.random.Generator, n: int, shift: float, prevalence: float = 0.3
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.intp]]:
    """Binormal groups; group 1's positives sit `shift` lower."""
    groups = rng.integers(0, 2, n)
    y = (rng.random(n) < prevalence).astype(float)
    centre = np.where(y == 1, 1.5 - shift * groups, -0.5)
    return rng.normal(centre, 1.0), y, groups


def threshold_oracle(shift: float, t: float, prevalence: float = 0.3) -> float:
    """Best net benefit of group cuts with equal true-positive rates.

    For a common rate `r` group `g` cuts at `mu_g + z(1 - r)`; the net
    benefit is maximised over a fine grid of `r` (the closed form of each
    value, a one-dimensional search for the maximum).
    """
    w = t / (1.0 - t)
    best = -math.inf
    for r in np.linspace(0.0005, 0.9995, 4000):
        value = 0.0
        for g in (0, 1):
            mu1 = 1.5 - shift * g
            cut = mu1 + Z.ppf(1.0 - r)
            fpr = 1.0 - Z.cdf(cut + 0.5)
            value += 0.5 * (prevalence * r - (1 - prevalence) * w * fpr)
        best = max(best, value)
    return best


def base_rate_world(
    rng: np.random.Generator, n: int, rates: tuple[float, float]
) -> tuple[NDArray[np.float64], NDArray[np.intp]]:
    """Labels and groups with the given base rates."""
    groups = rng.integers(0, 2, n)
    p = np.where(groups == 0, rates[0], rates[1])
    return (rng.random(n) < p).astype(float), groups


# --- experiments -----------------------------------------------------------


def experiment(
    rng: np.random.Generator,
    runs: int,
    n: int,
    q: float,
    control: float,
    effect: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Arms and binary outcomes for `runs` experiments of `n` units."""
    arm = (rng.random((runs, n)) < q).astype(float)
    y = (rng.random((runs, n)) < control + effect * arm).astype(float)
    return arm, y


def drifting_experiment(
    rng: np.random.Generator,
    runs: int,
    n: int,
    q: float,
    effect_start: float,
    effect_end: float,
    wander: float = 0.02,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """A shared control rate that random-walks and a drifting effect.

    The walk changes both arms alike (randomisation keeps the difference
    unbiased); the effect moves linearly from start to end.
    """
    arm = (rng.random((runs, n)) < q).astype(float)
    steps = rng.normal(0.0, wander / math.sqrt(n), (runs, n))
    base = np.clip(0.15 + np.cumsum(steps, axis=1), 0.05, 0.5)
    effect = np.linspace(effect_start, effect_end, n)
    y = (rng.random((runs, n)) < np.clip(base + effect * arm, 0, 1)).astype(
        float
    )
    return arm, y
