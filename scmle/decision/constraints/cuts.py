"""Group thresholds that meet a fairness constraint on development data.

`equalize` fits one cut per group so that the criterion's rate matches across
groups and the development gap stays within the bound; `frontier` sweeps the
bound and `toll` prices the constraint in net benefit.
"""

from collections.abc import Sequence
import dataclasses
from typing import Any

import numpy
from numpy.typing import ArrayLike
from numpy.typing import NDArray

from scmle.assurance.charter import decision
from scmle.assurance.framing import costs
from scmle.contracts import errors
from scmle.decision import netbenefit
from scmle.decision.constraints import rates
from scmle.runtime import randomness

GRID = 99


@dataclasses.dataclass(frozen=True, slots=True)
class Thresholds:
    """Group-specific cuts chosen on development data.

    Attributes:
        cuts: Cut on the score for each group label.
        target_rate: The common rate the cuts were matched to.
        net_benefit: Development net benefit of the rule at `t`.
        development_gap: Largest development gap in the criterion.
        constrained: False when the unconstrained common cut already met the
            development bound.
    """

    cuts: dict[str, float]
    target_rate: float | None
    net_benefit: float
    development_gap: float
    constrained: bool

    def act(
        self, scores: ArrayLike, groups: ArrayLike
    ) -> NDArray[numpy.float64]:
        """Applies each group's cut.

        Args:
            scores: Score per row.
            groups: Group label per row; each must be a key of `cuts` once
                converted to text.

        Returns:
            1.0 where the score reaches its group's cut, otherwise 0.0.
        """
        s = numpy.asarray(scores, dtype=float)
        g = numpy.asarray(groups)
        cut = numpy.array([self.cuts[str(x)] for x in g])
        return (s >= cut).astype(float)


def development_gap(
    y: NDArray[numpy.float64],
    acts: NDArray[numpy.float64],
    labels: NDArray[Any],
    criterion: str,
) -> float:
    """Computes the largest gap between groups in the criterion's rates.

    Args:
        y: Outcomes, 0 or 1.
        acts: Actions, 0 or 1, aligned with `y`.
        labels: Group label per row.
        criterion: The fairness criterion.

    Returns:
        The largest difference between the highest and lowest group rate
        over the criterion's rates; a group with an empty denominator counts
        as rate 0.
    """
    worst = 0.0
    for table in rates.tabulate(y, acts, labels, criterion).values():
        shares = [c / t if t > 0 else 0.0 for c, t in table.values()]
        worst = max(worst, max(shares) - min(shares))
    return worst


def align(
    s: NDArray[numpy.float64],
    outcomes: NDArray[numpy.float64],
    labels: NDArray[Any],
    names: list[str],
    rate: float,
    positive: bool,
) -> dict[str, float] | None:
    """Each group's score quantile that gives it a common rate.

    Args:
        s: Development scores.
        outcomes: Development outcomes, 0 or 1.
        labels: Group label per row.
        names: The groups, as text.
        rate: The common selection (or true-positive) rate.
        positive: Whether the rate is over positive outcomes only.

    Returns:
        The cut of each group, or None when a group has no row to cut.
    """
    cuts: dict[str, float] = {}
    for name in names:
        m = labels.astype(str) == name
        pool = s[m & (outcomes == 1)] if positive else s[m]
        if pool.size == 0:
            return None
        cuts[name] = float(numpy.quantile(pool, 1.0 - rate))
    return cuts


def sweep(
    rows: tuple[NDArray[numpy.float64], NDArray[numpy.float64], NDArray[Any]],
    names: list[str],
    criterion: str,
    t: float,
    bound: float,
) -> Thresholds | None:
    """The best group cuts over a grid of `GRID` common target rates.

    Args:
        rows: Development scores, outcomes and group labels.
        names: The groups, as text.
        criterion: The fairness criterion.
        t: The Bayes threshold, also the weight point of the net benefit.
        bound: The largest development gap allowed.

    Returns:
        The cuts with the best development net benefit among the target
        rates whose development gap meets `bound`, or None when none does.
    """
    s, outcomes, labels = rows
    positive_only = criterion in ("equal_opportunity", "equalized_odds")
    best: Thresholds | None = None
    for rate in numpy.linspace(0.01, 0.99, GRID):
        cuts = align(s, outcomes, labels, names, rate, positive_only)
        if cuts is None:
            continue
        acts = Thresholds(cuts, rate, 0.0, 0.0, constrained=True).act(s, labels)
        achieved = development_gap(outcomes, acts, labels, criterion)
        if achieved > bound:
            continue
        value = costs.net_benefit(outcomes, acts, t)
        if best is None or value > best.net_benefit:
            best = Thresholds(
                cuts, float(rate), value, achieved, constrained=True
            )
    return best


def equalize(
    scores: ArrayLike,
    y: ArrayLike,
    groups: ArrayLike,
    constraint: decision.Constraint,
    t: float,
    *,
    margin: float = 0.5,
) -> Thresholds:
    """Fits group cuts that maximise net benefit within `margin * max_gap`.

    The development bound is `margin * constraint.max_gap`. The common cut at
    the Bayes threshold `t` is used when its development gap already meets
    the bound. Otherwise a common target rate (selection rate, or
    true-positive rate for `equal_opportunity` and `equalized_odds`) is swept
    over a grid of 99 values from 0.01 to 0.99 and each group takes the score
    quantile that gives it that rate; the best net benefit among the sweep
    points whose development gap (over all rates of the criterion) meets the
    bound wins. Predictive parity and unreachable bounds raise.

    Args:
        scores: Development scores.
        y: Development outcomes, 0 or 1.
        groups: Group label per row.
        constraint: The declared criterion and gap bound.
        t: The Bayes threshold, also the weight point of the net benefit.
        margin: Fraction of `max_gap` that the development gap may use.

    Returns:
        The group cuts, keyed by the group label as text. When the common
        cut at `t` already meets the bound, every group gets `t`,
        `target_rate` is None and `constrained` is False.

    Raises:
        InfeasibleError: If no sweep point meets the bound.
        ValueError: If the criterion is predictive parity, which group
            thresholds cannot move. Invalid inputs are also rejected with a
            `ValueError` by `rates.prepared`.
    """
    s = numpy.asarray(scores, dtype=float)
    prepared_parts = rates.prepared(y, numpy.zeros_like(s), groups)
    outcomes, labels = prepared_parts[0], prepared_parts[2]
    if constraint.criterion == "predictive_parity":
        raise errors.InputError(
            "group thresholds cannot target predictive parity"
        )
    bound = margin * constraint.gap
    common = (s >= t).astype(float)
    names = sorted(str(g) for g in numpy.unique(labels))
    gap = development_gap(outcomes, common, labels, constraint.criterion)
    if gap <= bound:
        return Thresholds(
            dict.fromkeys(names, t),
            None,
            costs.net_benefit(outcomes, common, t),
            gap,
            constrained=False,
        )
    best = sweep((s, outcomes, labels), names, constraint.criterion, t, bound)
    if best is None:
        raise rates.InfeasibleError(
            "no group thresholds meet the development bound "
            f"{bound:.4f} for {constraint.criterion}"
        )
    return best


def frontier(
    scores: ArrayLike,
    y: ArrayLike,
    groups: ArrayLike,
    constraint: decision.Constraint,
    t: float,
    gaps: Sequence[float],
) -> list[dict[str, Any]]:
    """Fits group thresholds along a sweep of `max_gap` bounds.

    Args:
        scores: Development scores.
        y: Development outcomes, 0 or 1.
        groups: Group label per row.
        constraint: The declared criterion; its `max_gap` is replaced by
            each value of `gaps`.
        t: The Bayes threshold.
        gaps: The gap bounds to sweep.

    Returns:
        One record per bound, in the order of `gaps`, with `max_gap` and
        `feasible`; a feasible point also carries its development
        `net_benefit` and `development_gap`. The default `margin` of
        `equalize` is used. Errors other than infeasibility,
        including the refusal of predictive parity, propagate from
        `equalize`.
    """
    rows = []
    for gap in gaps:
        trial = dataclasses.replace(constraint, gap=gap)
        try:
            fitted = equalize(scores, y, groups, trial, t)
        except rates.InfeasibleError:
            # handled: recorded as infeasible row
            rows.append({"max_gap": gap, "feasible": False})
            continue
        rows.append(
            {
                "max_gap": gap,
                "feasible": True,
                "net_benefit": fitted.net_benefit,
                "development_gap": fitted.development_gap,
            }
        )
    return rows


def toll(
    y: ArrayLike,
    unconstrained: ArrayLike,
    constrained: ArrayLike,
    t: float,
    *,
    clusters: ArrayLike | None = None,
    replicates: int = 500,
    seed: int = 0,
) -> dict[str, float]:
    """Computes `NB(unconstrained) - NB(constrained)` with a 95% interval.

    Args:
        y: Outcomes, 0 or 1.
        unconstrained: Actions of the rule without the constraint.
        constrained: Actions of the constrained rule.
        t: Threshold probability of the net benefit.
        clusters: Cluster label per row, or None for independent rows.
        replicates: Number of bootstrap replicates.
        seed: Seed of the bootstrap stream.

    Returns:
        The `price` (net benefit given up, per row) and the `lower` and
        `upper` ends of its 95% percentile bootstrap interval (resampling
        clusters when `clusters` is given, otherwise rows). Invalid inputs
        are refused with a `ValueError` by `costs.pair`, `costs.odds` and
        `netbenefit.factorize`.
    """
    outcomes, free = costs.pair(y, unconstrained)
    bound = costs.pair(y, constrained)[1]
    w = costs.odds(t)
    stats_matrix = numpy.column_stack(
        [
            free * outcomes,
            free * (1 - outcomes),
            bound * outcomes,
            bound * (1 - outcomes),
        ]
    )

    def price(m: NDArray[numpy.float64]) -> NDArray[numpy.float64]:
        """Net benefit of the free rule minus that of the constrained rule.

        Args:
            m: Row means of the four statistics (free and constrained
                true and false positives) on the last axis.

        Returns:
            The difference, one value per leading index.
        """
        return (m[..., 0] - w * m[..., 1]) - (m[..., 2] - w * m[..., 3])

    rng = randomness.component_rng(seed, "decision.constraints.price", 0)
    boot = price(
        netbenefit.redraw(
            stats_matrix, netbenefit.factorize(clusters), replicates, rng
        )
    )
    return {
        "price": float(price(stats_matrix.mean(axis=0))),
        "lower": float(numpy.quantile(boot, 0.025)),
        "upper": float(numpy.quantile(boot, 0.975)),
    }
