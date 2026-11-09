"""Simulated worlds with known truth for the decision-layer gates.

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

    Args:
        rng: Random generator.
        n: Number of rows.
        skew: Exponent applied to the true risk to give the score.

    Returns:
        The score, the true risk and the labels, each of shape (n,).
    """
    p = rng.beta(1.0, 6.0, n)
    y = (rng.random(n) < p).astype(float)
    return p**skew, p, y


def beta_cost(
    p: NDArray[np.float64], acts: NDArray[np.float64], c_fp: float, c_fn: float
) -> float:
    """Exact expected cost per row of a rule, given the true risks `p`.

    Args:
        p: True risk of each row.
        acts: Decision of the rule for each row (1 = act).
        c_fp: Cost of acting on a negative.
        c_fn: Cost of not acting on a positive.

    Returns:
        The mean over rows of the expected cost.
    """
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
    """Score `N(delta y + u_g, 1)` with a shared cluster shift `u_g`.

    Args:
        rng: Random generator.
        n: Number of rows.
        prevalence: Share of positives.
        delta: Mean shift of the score for positives.
        cluster_size: Rows per cluster.
        cluster_sd: Standard deviation of the cluster shift.

    Returns:
        The scores, the labels and the cluster index of each row.
    """
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
    """True net benefit of `score >= cut` at each threshold `t`.

    Args:
        prevalence: Share of positives.
        delta: Mean shift of the score for positives.
        cut: Score cut of the rule.
        ts: Threshold probabilities in (0, 1).
        total_sd: Total standard deviation of the score, including any
            cluster shift.

    Returns:
        The net benefit at each threshold.
    """
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
    """True `NB_rule - NB_none` and `NB_rule - NB_all`, shape `(2, grid)`.

    Args:
        prevalence: Share of positives.
        delta: Mean shift of the score for positives.
        cut: Score cut of the rule.
        ts: Threshold probabilities in (0, 1).
        total_sd: Standard deviation of the score.

    Returns:
        Row 0 is the rule's net benefit (treat-none has zero); row 1 is the
        rule's net benefit minus that of treating everyone.
    """
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

    Args:
        rng: Random generator.
        n: Number of logged rounds.
        sharpness: Slope of the logging propensity in `x0`.
        floor: Smallest propensity of either action.
        blind_above: Value of `x0` above which the logger never takes
            action 1, or None.

    Returns:
        The covariates `x` (n, 2), the `action`, the `reward` and the
        `behaviour` probabilities of both actions (n, 2).
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
    """Treat iff `x0 > 0`, as an `(n, 2)` probability matrix.

    Args:
        x: Covariates, first column `x0`.

    Returns:
        A matrix whose columns are the probabilities of action 0 and 1.
    """
    treat = (x[:, 0] > 0).astype(float)
    return np.column_stack([1 - treat, treat])


def target_value() -> float:
    """`V(pi) = 0.30 + 0.25 P(x0 > 0)`, exact.

    Returns:
        The true value of the target policy.
    """
    return BASE_REWARD + EFFECT_UP * 0.5


def outcome_features(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Features for the outcome model, including the step.

    Args:
        x: Covariates.

    Returns:
        The covariates with an extra 0/1 column for `x0 > 0`.
    """
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

    Args:
        rng: Random generator.
        n: Number of rows.
        gap: True-positive-rate gap between group 0 and group 1.
        cut: Score cut of the rule.
        tpr: True-positive rate of group 0.
        prevalence: Share of positives.

    Returns:
        The scores, the labels and the group (0 or 1) of each row.
    """
    groups = rng.integers(0, 2, n)
    y = (rng.random(n) < prevalence).astype(float)
    rate = np.where(groups == 0, tpr, tpr - gap)
    centre = np.where(y == 1, cut + Z.ppf(rate), cut - 1.0)
    return rng.normal(centre, 1.0), y, groups


def threshold_world(
    rng: np.random.Generator, n: int, shift: float, prevalence: float = 0.3
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.intp]]:
    """Binormal groups; group 1's positives sit `shift` lower.

    Args:
        rng: Random generator.
        n: Number of rows.
        shift: How much lower group 1's positive mean is.
        prevalence: Share of positives.

    Returns:
        The scores, the labels and the group (0 or 1) of each row.
    """
    groups = rng.integers(0, 2, n)
    y = (rng.random(n) < prevalence).astype(float)
    centre = np.where(y == 1, 1.5 - shift * groups, -0.5)
    return rng.normal(centre, 1.0), y, groups


def threshold_oracle(shift: float, t: float, prevalence: float = 0.3) -> float:
    """Best net benefit of group cuts with equal true-positive rates.

    For a common rate `r` group `g` cuts at `mu_g + z(1 - r)`; the net
    benefit is maximised over a fine grid of `r` (the closed form of each
    value, a one-dimensional search for the maximum).

    Args:
        shift: As in `threshold_world`.
        t: Threshold probability in (0, 1).
        prevalence: Share of positives.

    Returns:
        The largest net benefit over the grid.
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
    """Labels and groups with the given base rates.

    Args:
        rng: Random generator.
        n: Number of rows.
        rates: Positive rate of group 0 and of group 1.

    Returns:
        The labels and the group (0 or 1) of each row.
    """
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
    """Arms and binary outcomes for `runs` experiments of `n` units.

    Args:
        rng: Random generator.
        runs: Number of experiments.
        n: Units per experiment.
        q: Probability of assignment to the treated arm.
        control: Outcome rate in the control arm.
        effect: Added outcome rate in the treated arm.

    Returns:
        The arm and the outcome, each of shape (runs, n).
    """
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

    Args:
        rng: Random generator.
        runs: Number of experiments.
        n: Units per experiment, in order.
        q: Probability of assignment to the treated arm.
        effect_start: Effect at the first unit.
        effect_end: Effect at the last unit.
        wander: Scale of the control rate's random walk over the experiment.

    Returns:
        The arm and the outcome, each of shape (runs, n).
    """
    arm = (rng.random((runs, n)) < q).astype(float)
    steps = rng.normal(0.0, wander / math.sqrt(n), (runs, n))
    base = np.clip(0.15 + np.cumsum(steps, axis=1), 0.05, 0.5)
    effect = np.linspace(effect_start, effect_end, n)
    y = (rng.random((runs, n)) < np.clip(base + effect * arm, 0, 1)).astype(
        float
    )
    return arm, y
