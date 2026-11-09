"""Feasibility of the baseline claim from the closed form.

The per-row gains `d_i(t)` measured on `X` have standard deviation `s_d`. The
smallest gain worth acting on, `delta_min`, gives the standardised effect
`delta = delta_min / s_d`, and `charter.feasibility` applies as is. For
clustered rows the required rows are multiplied by the design effect
`1 + (m - 1) rho` (`m` the mean cluster size, `rho` the intraclass
correlation from the one-way ANOVA estimator on `X`). Both are estimates, so
the verdict is computed at the ends of an interval for `s_d` (chi-square) and
reported `inconclusive` when they disagree.
"""

import dataclasses
import math
from typing import Any

import numpy
from numpy.typing import ArrayLike
from scipy import stats

from scmle.data.charter import feasibility
from scmle.data.framing import baseline

INCONCLUSIVE = "inconclusive"
CONFIDENCE = 0.95


def intraclass(d: ArrayLike, clusters: ArrayLike) -> tuple[float, float]:
    """Estimates the intraclass correlation by the one-way ANOVA estimator.

    A `d` that is not a non-empty 1-d array, or a `clusters` that is not 1-d,
    raises `ValueError`.

    Args:
        d: Per-row gains.
        clusters: Cluster label of each row.

    Returns:
        The intraclass correlation `rho`, clipped to [0, 1], and the mean
        cluster size. With fewer than two clusters, or no more rows than
        clusters, `rho` is 0.
    """
    values = baseline.vector(d, "d")
    codes, g = baseline.factorize(clusters)
    n = values.size
    if g < 2 or n <= g:
        return 0.0, n / max(g, 1)
    counts = numpy.bincount(codes, minlength=g).astype(float)
    means = numpy.bincount(codes, weights=values, minlength=g) / counts
    grand = float(values.mean())
    ssb = float(numpy.sum(counts * (means - grand) ** 2))
    ssw = float(numpy.sum((values - means[codes]) ** 2))
    msb, msw = ssb / (g - 1), ssw / (n - g)
    m0 = (n - float(numpy.sum(counts**2)) / n) / (g - 1)
    denominator = msb + (m0 - 1) * msw
    rho = (msb - msw) / denominator if denominator > 0 else 0.0
    return min(max(rho, 0.0), 1.0), n / g


@dataclasses.dataclass(frozen=True, slots=True)
class BaselineFeasibility:
    """The verdict for the baseline claim, with its uncertainty.

    Attributes:
        verdict: `feasible`, `downgrade`, `infeasible` or `inconclusive`.
        scope: `confirmatory` or `exploratory_unconfirmed`.
        s_d: Standard deviation of the gain on `X`.
        s_d_interval: Chi-square interval for `s_d`.
        rho: Estimated intraclass correlation.
        design_effect: `1 + (m - 1) rho`.
        n_required: Rows needed at the point estimate.
        n_effective: Confirmation rows divided by the design effect.
        reason: Why the verdict was reached.
    """

    verdict: str
    scope: str
    s_d: float
    s_d_interval: tuple[float, float]
    rho: float
    design_effect: float
    n_required: int
    n_effective: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        """Returns the verdict as a JSON-ready record.

        Returns:
            A mapping of every field to its value, with `s_d_interval` as a
            list.
        """
        record = dataclasses.asdict(self)
        record["s_d_interval"] = list(self.s_d_interval)
        return record


def verdict_at(
    s_d: float,
    *,
    delta_min: float,
    n_confirmation: float,
    alpha: float,
    k_max: int,
    power: float,
) -> feasibility.Feasibility:
    """Computes the closed-form verdict for a given `s_d` and effective rows.

    Args:
        s_d: Standard deviation of the per-row gain.
        delta_min: Smallest net-benefit gain worth acting on.
        n_confirmation: Effective confirmation rows (truncated to an int).
        alpha: Level of the claim.
        k_max: Claims carried into confirmation.
        power: Target power.

    Returns:
        The feasibility verdict for the standardised effect
        `delta_min / s_d`.
    """
    return feasibility.assess(
        n_confirmation=int(n_confirmation),
        min_effect=delta_min / s_d,
        alpha=alpha,
        k_max=k_max,
        power=power,
    )


def assess(
    d: ArrayLike,
    clusters: ArrayLike,
    *,
    delta_min: float,
    n_confirmation: int,
    alpha: float,
    k_max: int,
    power: float,
) -> BaselineFeasibility:
    """Estimates feasibility from the gains measured on `X`.

    Args:
        d: Per-row gains `d_i(t)` on `X` at the harder endpoint.
        clusters: Grouping unit of each `X` row.
        delta_min: Smallest net-benefit gain worth acting on.
        n_confirmation: Rows the confirmation set will hold.
        alpha: Level of the claim.
        k_max: Claims carried into confirmation.
        power: Target power.

    Returns:
        The verdict, computed at the point estimate of `s_d`. It is
        `inconclusive` when the verdict differs at either end of the
        chi-square interval for `s_d`.

    Raises:
        ValueError: If `delta_min` is not positive, or `d` has fewer than
            three values or zero standard deviation. A `d` that is not a
            non-empty 1-d array also raises it.
    """
    values = baseline.vector(d, "d")
    if delta_min <= 0:
        raise ValueError("delta_min must be greater than 0")
    n = values.size
    s_d = float(values.std(ddof=1))
    if n < 3 or s_d == 0.0:
        raise ValueError("s_d cannot be estimated from these gains")
    rho, m = intraclass(values, clusters)
    deff = 1.0 + (m - 1.0) * rho
    tail = (1.0 - CONFIDENCE) / 2.0
    low = s_d * math.sqrt((n - 1) / float(stats.chi2.ppf(1 - tail, n - 1)))
    high = s_d * math.sqrt((n - 1) / float(stats.chi2.ppf(tail, n - 1)))
    effective = n_confirmation / deff
    settings: dict[str, Any] = {
        "delta_min": delta_min,
        "n_confirmation": effective,
        "alpha": alpha,
        "k_max": k_max,
        "power": power,
    }
    point = verdict_at(s_d, **settings)
    ends = {
        verdict_at(low, **settings).verdict,
        verdict_at(high, **settings).verdict,
    }
    verdict = point.verdict.value
    scope = point.scope
    reason = point.reason
    if len(ends | {point.verdict}) > 1:
        verdict, scope = INCONCLUSIVE, feasibility.UNCONFIRMED
        reason = (
            "the verdict changes across the interval for s_d "
            f"[{low:.4f}, {high:.4f}]; collect more exploration rows or "
            "treat the claim as unconfirmed"
        )
    return BaselineFeasibility(
        verdict,
        scope,
        s_d,
        (low, high),
        rho,
        deff,
        point.n_required,
        effective,
        reason,
    )
