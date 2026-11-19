"""Two-sided intervals for a binomial proportion, as certificates judge them.

A certificate's policy names its method: the Wilson score interval, or the
exact Clopper-Pearson interval.
"""

import math

from scipy import stats

from scmle.contracts import errors
from scmle.contracts import qualification
from scmle.evaluation import coverage

# how a policy computes its intervals: Wilson score, or Clopper-Pearson
METHODS = ("wilson", "exact")


def bound(
    successes: int, n: int, confidence: float = qualification.DEFAULT_CONFIDENCE
) -> tuple[float, float]:
    """Two-sided Wilson score interval for a binomial proportion.

    Args:
        successes: Observed successes, between 0 and `n`.
        n: Trials, at least 1.
        confidence: Level of the interval, in (0, 1).

    Returns:
        (low, high), clipped to [0, 1].

    Raises:
        ValueError: On a count that is not an integer in range or a
            confidence outside (0, 1).
    """
    qualification.count(n, "n", minimum=1)
    qualification.count(successes, "successes")
    if successes > n:
        raise errors.InputError("successes must be between 0 and n")
    if not 0.0 < confidence < 1.0:
        raise errors.InputError("confidence must be within (0, 1)")
    z = float(stats.norm.ppf(1.0 - (1.0 - confidence) / 2.0))
    p = successes / n
    scale = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / scale
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / scale
    # The interval contains p exactly; clamp away rounding at the edges.
    return max(0.0, min(centre - half, p)), min(1.0, max(centre + half, p))


def interval(
    successes: int, n: int, confidence: float, method: str
) -> tuple[float, float]:
    """Two-sided interval for a binomial proportion by a named method.

    Args:
        successes: Observed successes, between 0 and `n`.
        n: Trials, at least 1.
        confidence: Level of the interval, in (0, 1); each bound is
            one-sided at `(1 - confidence) / 2`.
        method: `wilson` or `exact` (Clopper-Pearson).

    Returns:
        (low, high).

    Raises:
        ValueError: On invalid counts, confidence or method.
    """
    if method == "exact":
        qualification.count(n, "n", minimum=1)
        return coverage.clopper_pearson(successes, n, confidence)
    if method != "wilson":
        raise errors.InputError(f"method must be one of {METHODS}")
    return bound(successes, n, confidence)
