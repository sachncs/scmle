"""Rollout design: sample size, expected sample number, assignment, SRM.

Written before any traffic. The fixed-horizon size comes from
the closed forms in `scmle.data.charter.feasibility`; the expected sample
number of the anytime-valid design is simulated with the engine in
`scmle.decision.anytime`, and the ratio between the two is reported rather
than hidden. Assignment is a salted hash of the unit id, so it is
reproducible from the log. The sample-ratio check has two forms: the exact
two-sided binomial test at a fixed count, and an anytime-valid Bayes factor
(a likelihood ratio martingale against a uniform prior on the allocation)
that may be read after every unit.
"""

import hashlib
import math
from typing import Any

import numpy
from numpy.typing import ArrayLike
from numpy.typing import NDArray
from scipy import special
from scipy import stats

from scmle.contracts import qualification
from scmle.core import randomness
from scmle.data.charter import decision
from scmle.data.charter import feasibility
from scmle.decision import anytime

HASH_BYTES = 8


def assign(
    unit_ids: ArrayLike, salt: str, assignment_prob: float
) -> NDArray[numpy.float64]:
    """1 for the challenger, 0 for the champion, by a salted hash of the id.

    The hash is SHA-256 over ``salt|id``; its first eight bytes, read as an
    integer over ``2**64``, are compared with `assignment_prob`.

    Args:
        unit_ids: Identifiers of the units; flattened to one dimension.
        salt: Text mixed into the hash; must not be blank.
        assignment_prob: Probability of the challenger, strictly between 0
            and 1.

    Returns:
        One value per unit, in order: 1.0 for the challenger, 0.0 for the
        champion.

    Raises:
        ValueError: If `assignment_prob` is not strictly between 0 and 1 or
            is not a finite number, or if `salt` is blank.
    """
    q = qualification.finite(assignment_prob, "assignment_prob")
    if not 0.0 < q < 1.0:
        raise ValueError("assignment_prob must be strictly between 0 and 1")
    if not salt.strip():
        raise ValueError("salt must not be blank")
    out = []
    for unit in numpy.asarray(unit_ids).ravel():
        digest = hashlib.sha256(f"{salt}|{unit}".encode()).digest()
        value = int.from_bytes(digest[:HASH_BYTES], "big") / 2.0 ** (
            8 * HASH_BYTES
        )
        out.append(1.0 if value < q else 0.0)
    return numpy.array(out)


def fixed_horizon_rows(
    p_control: float,
    delta: float,
    *,
    alpha: float,
    power: float,
    design_effect: float = 1.0,
) -> int:
    """Total units of a fixed-horizon two-arm test of a proportion.

    Uses the pooled-variance closed form per arm, doubled, and inflated by
    the design effect when the randomisation unit is a cluster.

    Args:
        p_control: Control proportion, in (0, 1).
        delta: Difference in proportion to detect; `p_control + delta` must
            lie in (0, 1) and differ from `p_control`.
        alpha: Two-sided significance level, in (0, 1).
        power: Power of the test, in (0, 1).
        design_effect: Variance inflation from cluster randomisation; at
            least 1.

    Returns:
        The total number of units over both arms.

    Raises:
        ValueError: If `design_effect` is below 1. Invalid proportions,
            level or power are rejected by
            `feasibility.n_required_two_proportions` with a `ValueError`.
    """
    if design_effect < 1.0:
        raise ValueError("design_effect must be at least 1")
    per_arm = feasibility.n_required_two_proportions(
        p_control, p_control + delta, alpha=alpha, k_max=1, power=power
    )
    return math.ceil(2 * per_arm * design_effect)


def srm_p_value(challenger: int, total: int, assignment_prob: float) -> float:
    """Exact two-sided binomial p-value of the assignment counts.

    Args:
        challenger: Number of units assigned to the challenger.
        total: Number of units assigned; at least 1.
        assignment_prob: The recorded probability of the challenger.

    Returns:
        The p-value of the null that each unit went to the challenger with
        probability `assignment_prob`.

    Raises:
        ValueError: If `challenger` lies outside [0, `total`]. A `total`
            that is not an integer of at least 1 is rejected by
            `qualification.count` with a `ValueError`.
    """
    qualification.count(total, "total", minimum=1)
    if not 0 <= challenger <= total:
        raise ValueError("challenger count must lie in [0, total]")
    return float(stats.binomtest(challenger, total, assignment_prob).pvalue)


def srm_log_bayes_factor(
    challenger: ArrayLike, total: ArrayLike, assignment_prob: float
) -> NDArray[numpy.float64]:
    """Log Bayes factor of a uniform-prior allocation against a fixed rate.

    The numerator is the likelihood of the observed assignments averaged
    over a uniform prior on the challenger probability; the denominator is
    their likelihood at `assignment_prob`. It is a likelihood-ratio martingale
    under correct assignment, so by Ville's inequality
    ``P(sup BF >= 1 / level) <= level`` however often it is read.

    Args:
        challenger: Count of units assigned to the challenger, scalar or
            array.
        total: Count of units assigned, same shape as `challenger`.
        assignment_prob: The recorded probability of the challenger, in
            (0, 1); a value outside (0, 1) makes `math.log` raise a
            `ValueError`.

    Returns:
        The log Bayes factor, one value per element of the broadcast
        counts.
    """
    k = numpy.asarray(challenger, dtype=float)
    n = numpy.asarray(total, dtype=float)
    q = assignment_prob
    result: NDArray[numpy.float64] = (
        special.betaln(k + 1.0, n - k + 1.0)
        - k * math.log(q)
        - (n - k) * math.log1p(-q)
    )
    return result


def srm_rejects(
    challenger: int, total: int, assignment_prob: float, level: float
) -> bool:
    """Anytime-valid sample-ratio check at the current counts.

    Args:
        challenger: Number of units assigned to the challenger.
        total: Number of units assigned.
        assignment_prob: The recorded probability of the challenger.
        level: Level of the check, in (0, 1).

    Returns:
        True if the log Bayes factor has reached `-log(level)`, that is the
        assignment counts are inconsistent with `assignment_prob`.
    """
    value = srm_log_bayes_factor(challenger, total, assignment_prob)
    return bool(value >= -math.log(level))


def simulate_stopping(
    plan: decision.RolloutPlan,
    p_control: float,
    effect: float,
    runs: int,
    seed: int,
    level: float | None = None,
) -> dict[str, Any]:
    """Expected sample number of the primary e-process by simulation.

    Binary outcomes with control rate `p_control` and a true mean difference
    `effect`; peeking after every unit up to `n_max`.

    A `runs` that is not an integer of at least 1 is rejected by
    `qualification.count` with a `ValueError`.

    Args:
        plan: The rollout plan giving the assignment probability, metric
            range, smallest worthwhile improvement, level and `n_max`.
        p_control: Control rate of the binary outcome.
        effect: True difference in mean outcome between challenger and
            control.
        runs: Number of simulated experiments.
        seed: Seed of the random generator.
        level: Level of the e-process crossing; the rollout plan's promote level
            when None.

    Returns:
        The share of runs that crossed `1 / level`, the mean and median
        number of units at the crossing over the runs that crossed (None
        if none crossed), `n_max` and `runs`.
    """
    qualification.count(runs, "runs", minimum=1)
    chosen = float(plan.alpha_promote) if level is None else level
    rng = randomness.component_rng(seed, "decision.design.esn", 0)
    q = plan.assignment_prob
    arm = (rng.random((runs, plan.n_max)) < q).astype(float)
    y = (rng.random((runs, plan.n_max)) < p_control + effect * arm).astype(
        float
    )
    scaled = plan.delta_min / (plan.high - plan.low)
    paths = anytime.centered_paths(arm, y, q, scaled)
    hit = paths >= -math.log(chosen)
    crossed = hit.any(axis=1)
    when = numpy.where(crossed, hit.argmax(axis=1) + 1, plan.n_max)
    return {
        "crossed_share": float(crossed.mean()),
        "mean_units": float(when[crossed].mean()) if crossed.any() else None,
        "median_units": (
            float(numpy.median(when[crossed])) if crossed.any() else None
        ),
        "n_max": plan.n_max,
        "runs": runs,
    }
