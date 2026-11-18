"""The satisfaction test of a fairness constraint on the final table.

`satisfy` takes rows as independent and bounds every gap with Newcombe's
interval; `cluster_satisfaction_test` resamples whole clusters of rows.
Either is a claim tested once: the upper bound of the gap must be at most
`max_gap`, and a group too small for a rate's denominator makes the test
`not_assessable`.
"""

import dataclasses
import itertools
from typing import Any

import numpy
from numpy.typing import ArrayLike
from numpy.typing import NDArray

from scmle.assurance.charter import decision
from scmle.contracts import errors
from scmle.decision.constraints import rates
from scmle.runtime import randomness

SATISFIED = "satisfied"
NOT_SATISFIED = "not_satisfied"
NOT_ASSESSABLE = "not_assessable"
MIN_CLUSTERS = 20
MIN_GROUPS = 2


@dataclasses.dataclass(frozen=True, slots=True)
class Satisfaction:
    """Outcome of the satisfaction test on the final table.

    Attributes:
        status: `satisfied`, `not_satisfied` or `not_assessable`.
        criterion: The criterion tested.
        max_gap: The declared bound.
        point_gap: Largest observed gap (None when not assessable).
        upper_bound: Upper bound of the gap, or None.
        small_groups: Groups whose denominators were too small.
        alpha: Total level. The row-wise test splits it over group pairs
            and rates; the cluster test uses it for the bootstrap bound.
    """

    status: str
    criterion: str
    max_gap: float
    point_gap: float | None
    upper_bound: float | None
    small_groups: tuple[str, ...]
    alpha: float

    def to_dict(self) -> dict[str, Any]:
        """Returns the JSON-ready record.

        Returns:
            Every field of the test as a mapping.
        """
        return dataclasses.asdict(self)


def screen(
    tables: dict[str, dict[str, tuple[float, float]]],
    count: int,
    constraint: decision.Constraint,
    alpha: float,
    enough: bool = True,
) -> Satisfaction | None:
    """The `not_assessable` outcome, when the table cannot support a test.

    Args:
        tables: `rates.tabulate`'s counts per rate and group.
        count: The number of groups.
        constraint: The declared criterion, gap bound and smallest group.
        alpha: The level of the test.
        enough: Whether there are enough independent units otherwise.

    Returns:
        A `not_assessable` outcome naming the groups whose denominators are
        below `constraint.min_group`, when there are such groups, fewer than
        `MIN_GROUPS` groups or not `enough` units; None otherwise.
    """
    small = sorted(
        {
            str(g)
            for table in tables.values()
            for g, counts in table.items()
            if counts[1] < constraint.minimum
        }
    )
    if small or count < MIN_GROUPS or not enough:
        return Satisfaction(
            NOT_ASSESSABLE,
            constraint.criterion,
            constraint.gap,
            None,
            None,
            tuple(small),
            alpha,
        )
    return None


def conclude(
    constraint: decision.Constraint, point: float, upper: float, alpha: float
) -> Satisfaction:
    """The outcome of a test that ran: satisfied when `upper` is in bound.

    Args:
        constraint: The declared criterion and gap bound.
        point: The largest observed gap.
        upper: Its upper bound.
        alpha: The level of the test.

    Returns:
        `satisfied` when `upper` is at most the gap bound, else
        `not_satisfied`.
    """
    status = SATISFIED if upper <= constraint.gap else NOT_SATISFIED
    return Satisfaction(
        status,
        constraint.criterion,
        constraint.gap,
        point,
        upper,
        (),
        alpha,
    )


def satisfy(
    y: ArrayLike,
    acts: ArrayLike,
    groups: ArrayLike,
    constraint: decision.Constraint,
    alpha: float = 0.05,
) -> Satisfaction:
    """Tests whether the frozen rule's group gap is at most `max_gap`.

    The claim is `satisfied` only if the upper bound of every compared gap is
    at most `constraint.max_gap`. Rows are taken as independent.

    Args:
        y: Outcomes on the final table, 0 or 1.
        acts: The frozen rule's actions, 0 or 1, aligned with `y`.
        groups: Group label per row.
        constraint: The declared criterion, gap bound and smallest group.
        alpha: Total level, split over group pairs and rates.

    Returns:
        The test outcome. The status is `not_assessable` when a group is
        smaller than `constraint.min_group` for a rate's denominator or
        there are fewer than two groups; otherwise `satisfied` or
        `not_satisfied`, with the largest observed gap and its upper bound.

    Raises:
        ValueError: If `alpha` is not strictly between 0 and 1. Invalid
            inputs are also rejected with a `ValueError` by `rates.prepared`.
    """
    outcomes, actions, labels = rates.prepared(y, acts, groups)
    if not 0.0 < alpha < 1.0:
        raise errors.InputError("alpha must be strictly between 0 and 1")
    tables = rates.tabulate(outcomes, actions, labels, constraint.criterion)
    names = sorted(str(g) for g in numpy.unique(labels))
    refused = screen(tables, len(names), constraint, alpha)
    if refused is not None:
        return refused
    pairs = list(itertools.combinations(sorted(tables[next(iter(tables))]), 2))
    confidence = 1.0 - alpha / (len(pairs) * len(tables))
    point = 0.0
    upper = 0.0
    for table in tables.values():
        for a, b in pairs:
            lo, hi = rates.newcombe(*table[a], *table[b], confidence)
            point = max(
                point,
                abs(table[a][0] / table[a][1] - table[b][0] / table[b][1]),
            )
            upper = max(upper, abs(lo), abs(hi))
    return conclude(constraint, point, upper, alpha)


def aggregate(
    rows: tuple[NDArray[Any], NDArray[Any], NDArray[Any]],
    names: list[Any],
    criterion: str,
    inverse: NDArray[numpy.intp],
    count: int,
) -> NDArray[numpy.float64]:
    """Sums each rate's numerator and denominator per cluster and group.

    Args:
        rows: Outcomes, actions and group labels per row.
        names: The groups, sorted.
        criterion: The declared fairness criterion.
        inverse: The cluster index of each row.
        count: The number of clusters.

    Returns:
        Totals with axes `(cluster, rate, group, 2)`, the last axis holding
        numerator and denominator.
    """
    outcomes, actions, labels = rows
    columns = []
    for rate in rates.CRITERION_RATES[criterion]:
        numerator, denominator = rates.RATE_PARTS[rate](outcomes, actions)
        for name in names:
            inside = (labels == name).astype(float)
            for part in (numerator, denominator):
                sums = numpy.zeros(count)
                numpy.add.at(sums, inverse, part * inside)
                columns.append(sums)
    return numpy.column_stack(columns).reshape(count, -1, len(names), 2)


def spread(totals: NDArray[numpy.float64]) -> NDArray[numpy.float64]:
    """Computes the largest gap in any rate between groups.

    Args:
        totals: Numerators and denominators with last axes
            `(rate, group, 2)`.

    Returns:
        The largest difference between the highest and lowest group rate,
        over the rates, ignoring undefined rates; one value per leading
        index.
    """
    with numpy.errstate(divide="ignore", invalid="ignore"):
        shares = totals[..., 0] / totals[..., 1]
    widths = numpy.nanmax(shares, axis=-1) - numpy.nanmin(shares, axis=-1)
    return numpy.asarray(numpy.nanmax(widths, axis=-1))


def bound(
    matrix: NDArray[numpy.float64], alpha: float, draws: int, seed: int
) -> tuple[float, float]:
    """The largest gap and its cluster-bootstrap upper bound.

    Args:
        matrix: `aggregate`'s totals per cluster, rate, group and part.
        alpha: Level of the bound.
        draws: Number of cluster resamples.
        seed: Seed of the resampling stream.

    Returns:
        The observed largest gap and the larger of it and the `1 - alpha`
        quantile of the resampled largest gaps.
    """
    count = len(matrix)
    point = float(spread(matrix.sum(axis=0)))
    rng = randomness.component_rng(seed, "decision.constraints.cluster", 0)
    weights = rng.multinomial(
        count, numpy.full(count, 1.0 / count), size=draws
    ).astype(float)
    resampled = spread(numpy.einsum("dk,krgp->drgp", weights, matrix))
    return point, max(point, float(numpy.quantile(resampled, 1.0 - alpha)))


def cluster_satisfaction_test(
    y: ArrayLike,
    acts: ArrayLike,
    groups: ArrayLike,
    clusters: ArrayLike,
    constraint: decision.Constraint,
    alpha: float = 0.05,
    draws: int = 1000,
    seed: int = 0,
) -> Satisfaction:
    """Tests satisfaction by resampling whole clusters of rows.

    `satisfy` takes rows as independent, so rows that share a
    cluster (a branch, a household, a repeated customer) make its interval
    too narrow and can call a rule fair that is not. Here the clusters are
    resampled with replacement `draws` times, the largest absolute gap over
    group pairs and rates is recomputed on each resample, and the upper bound
    is the larger of the observed gap and the `1 - alpha` quantile of those
    resampled gaps. The claim is `satisfied` only if that bound is at most
    `constraint.max_gap`. A group whose denominator is below `min_group`
    rows is `not_assessable`, as in the row-wise test.

    The bound is a percentile bootstrap of the maximum gap: it is not exact,
    its false-satisfaction rate in a clustered world is checked by simulation
    in `tests/unit/test_decision_constraints.py`, and it makes no claim about
    a handful of clusters (fewer than 20 clusters are reported
    `not_assessable`).

    Args:
        y: Outcomes on the final table, 0 or 1.
        acts: The frozen rule's actions, 0 or 1, aligned with `y`.
        groups: Group label per row.
        clusters: Cluster label per row.
        constraint: The declared criterion, gap bound and smallest group.
        alpha: Level of the bound.
        draws: Number of cluster resamples.
        seed: Seed of the resampling stream.

    Returns:
        The test outcome, `not_assessable` when a group is too small, there
        are fewer than two groups or fewer than `MIN_CLUSTERS` clusters;
        otherwise `satisfied` or `not_satisfied` with the observed largest
        gap and its bootstrap upper bound.

    Raises:
        ValueError: If `clusters` does not have the length of `y`, `alpha`
            is not strictly between 0 and 1, or `draws` is below 1. Invalid
            inputs are also rejected with a `ValueError` by `rates.prepared`.
    """
    outcomes, actions, labels = rates.prepared(y, acts, groups)
    keys = numpy.asarray(clusters)
    if keys.shape != outcomes.shape:
        raise errors.InputError("clusters must have the length of y")
    if not 0.0 < alpha < 1.0:
        raise errors.InputError("alpha must be strictly between 0 and 1")
    if draws < 1:
        raise errors.InputError("draws must be positive")
    tables = rates.tabulate(outcomes, actions, labels, constraint.criterion)
    names = sorted(numpy.unique(labels).tolist())
    cluster_ids, inverse = numpy.unique(keys, return_inverse=True)
    refused = screen(
        tables, len(names), constraint, alpha, len(cluster_ids) >= MIN_CLUSTERS
    )
    if refused is not None:
        return refused
    matrix = aggregate(
        (outcomes, actions, labels),
        names,
        constraint.criterion,
        inverse,
        len(cluster_ids),
    )
    point, upper = bound(matrix, alpha, draws, seed)
    return conclude(constraint, point, upper, alpha)
