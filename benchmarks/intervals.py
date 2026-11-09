"""Summaries of repeated benchmark runs."""

from collections.abc import Sequence
import math

from scipy import stats

LOWER_IS_BETTER = frozenset({"log_loss", "rmse", "mae"})
CONFIDENCE = 0.95
# fewest seeds for which an interval carries information
MINIMUM_SEEDS = 2


def mean_interval(
    values: Sequence[float], confidence: float = CONFIDENCE
) -> tuple[float, float, float]:
    """Mean and Student-t confidence interval of repeated measurements.

    Args:
        values: One measurement per seed.
        confidence: Interval coverage.

    Returns:
        Mean, lower and upper bound. With a single value the interval is
        degenerate (the three numbers are equal) and carries no information.
    """
    count = len(values)
    mean = sum(values) / count
    if count < MINIMUM_SEEDS:
        return mean, mean, mean
    spread = math.sqrt(sum((v - mean) ** 2 for v in values) / (count - 1))
    margin = (
        stats.t.ppf(0.5 + confidence / 2, count - 1) * spread / math.sqrt(count)
    )
    return mean, mean - margin, mean + margin


def advantage(metric: str, score: float, reference: float) -> float:
    """How much better the score is than the reference (positive = better).

    Args:
        metric: Metric name; names in `LOWER_IS_BETTER` are minimised.
        score: Score of the model being judged.
        reference: Score of the reference it is compared with.

    Returns:
        The signed advantage, `reference - score` for a lower-is-better metric
        and `score - reference` otherwise.
    """
    if metric in LOWER_IS_BETTER:
        return reference - score
    return score - reference


def verdict(low: float, high: float, seeds: int) -> str:
    """`ahead`, `behind` or `tie` from the interval of the advantage.

    One seed has no interval, so the sign alone decides and the label says so.

    Args:
        low: Lower bound of the advantage interval.
        high: Upper bound of the advantage interval.
        seeds: Number of seeds behind the interval.

    Returns:
        `ahead` when the whole interval is positive, `behind` when it is all
        negative, otherwise `tie`. With fewer than two seeds the label is
        chosen by the sign of the midpoint and suffixed ` (1 seed)`.
    """
    if seeds < MINIMUM_SEEDS:
        mean = (low + high) / 2
        label = "ahead" if mean > 0 else "behind" if mean < 0 else "tie"
        return f"{label} (1 seed)"
    if low > 0:
        return "ahead"
    if high < 0:
        return "behind"
    return "tie"
