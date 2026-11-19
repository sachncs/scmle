"""A lateness bound estimated from the delays a log has shown.

An event's delay is how far its time lies behind the newest time seen
before it arrived (zero for an event in order). A store admits an event
whose delay is at most its lateness bound and treats the rest as late, so
the bound should cover nearly every delay. `bound` returns a
distribution-free upper confidence bound for the `coverage` quantile of the
delays: an order statistic of the observed delays chosen so that, whatever
the delay distribution, it lies at or above the true quantile with
probability at least `confidence` (the binomial argument for quantile
confidence bounds; Hahn and Meeker, Statistical Intervals, 2017, 5.2).
"""

from collections.abc import Iterable
import math

from scipy import stats

from scmle.data.events import event

# Share of events the bound must admit as on time (policy): at most one
# event in a hundred is late, so a window of `m` events loses or revises
# about `m / 100` of them in expectation. A stricter share costs history:
# `needed` grows as `1 / (1 - COVERAGE)` (299 events at 0.99, 2,995 at
# 0.999).
COVERAGE = 0.99
# Probability that the bound covers the `COVERAGE` quantile: the
# conventional one-sided 95% level of the project's other confidence bounds.
CONFIDENCE = 0.95


def delays(times: Iterable[int]) -> list[int]:
    """Measures each event's delay behind the newest time before it.

    Args:
        times: Event times in arrival order, in nanoseconds.

    Returns:
        One non-negative delay per event, in arrival order.
    """
    found: list[int] = []
    frontier: int | None = None
    for time in times:
        found.append(0 if frontier is None else max(frontier - time, 0))
        frontier = time if frontier is None else max(frontier, time)
    return found


def needed(coverage: float = COVERAGE, confidence: float = CONFIDENCE) -> int:
    """Fewest delays from which the bound can be computed.

    With `n` delays the largest one covers the `coverage` quantile with
    probability `1 - coverage**n`; this is the smallest `n` that reaches
    `confidence`.

    Args:
        coverage: Share of delays the bound must cover.
        confidence: Probability the bound covers it.

    Returns:
        The smallest sufficient number of delays.
    """
    return math.ceil(math.log1p(-confidence) / math.log(coverage))


def bound(
    times: Iterable[int],
    coverage: float = COVERAGE,
    confidence: float = CONFIDENCE,
) -> int:
    """Estimates a lateness bound from event times in arrival order.

    Fewer than `k` of `n` delays fall below the `coverage` quantile with
    probability `Binomial(n, coverage).cdf(k - 1)`, and exactly then the
    `k`-th smallest delay lies at or above it. The bound is that delay for
    the smallest `k` whose probability reaches `confidence`; `k <= n` once
    `n` reaches `needed`. Ties among delays only make the bound safer.

    Args:
        times: Event times in arrival order, in nanoseconds.
        coverage: Share of delays the bound must cover, in (0, 1).
        confidence: Probability the bound covers it, in (0, 1).

    Returns:
        The bound in nanoseconds.

    Raises:
        EventError: If `coverage` or `confidence` is outside (0, 1), or
            there are fewer delays than `needed` asks for.
    """
    if not (0 < coverage < 1 and 0 < confidence < 1):
        raise event.EventError(
            "coverage and confidence must lie strictly between 0 and 1"
        )
    ordered = sorted(delays(times))
    least = needed(coverage, confidence)
    if len(ordered) < least:
        raise event.EventError(
            f"a lateness bound covering {coverage:.0%} of delays at "
            f"{confidence:.0%} confidence needs at least {least} events, "
            f"got {len(ordered)}",
            {"events": len(ordered), "needed": least},
        )
    rank = int(stats.binom.ppf(confidence, len(ordered), coverage)) + 1
    return ordered[rank - 1]
