"""Fairness as explicit constraints on a frozen rule.

Default: no constraint. A `FairnessConstraint` in the charter bounds the gap
between the groups of one attribute in the declared criterion for the frozen
decision rule. The constraint is solved on development data (group thresholds
that match the criterion's rate across groups, picking the one that
maximises net benefit); satisfaction is a claim tested once on the final
table: the upper bound of the gap must be at most `max_gap`, a point estimate
below it is not satisfaction. The bound is the largest absolute difference
over group pairs from Newcombe's hybrid score interval (Wilson intervals for
each rate), each pair at level `alpha / pairs`. Rows are taken as
independent. Groups too small for the denominator of the rate are
`not_assessable`, never passing.

When two declared criteria cannot hold together for groups whose base rates
differ (the impossibility results of Kleinberg, Mullainathan and Raghavan,
2016 and Chouldechova, 2017, recalled and not re-read) the set is refused,
naming the pair, using the measured base rates.
"""

from collections.abc import Sequence
import dataclasses
import itertools
from typing import Any

import numpy
from numpy.typing import ArrayLike
from numpy.typing import NDArray
from scipy import stats

from scmle.core import randomness
from scmle.data.charter import decision
from scmle.data.framing import costs
from scmle.decision import netbenefit

SATISFIED = "satisfied"
NOT_SATISFIED = "not_satisfied"
NOT_ASSESSABLE = "not_assessable"
INCOMPATIBLE = (
    frozenset({"demographic_parity", "predictive_parity"}),
    frozenset({"demographic_parity", "equalized_odds"}),
    frozenset({"equalized_odds", "predictive_parity"}),
)
GRID = 99
MIN_CLUSTERS = 20
MIN_GROUPS = 2


class InfeasibleConstraintsError(ValueError):
    """The declared criteria cannot hold together for these base rates."""


def prepared(
    y: ArrayLike, acts: ArrayLike, groups: ArrayLike
) -> tuple[NDArray[numpy.float64], NDArray[numpy.float64], NDArray[Any]]:
    """Validates and returns outcomes, actions and group labels.

    Args:
        y: Outcomes, 0 or 1.
        acts: Actions, 0 or 1, aligned with `y`.
        groups: Group label per row.

    Returns:
        The outcomes and actions as float arrays, and the group labels.

    Raises:
        ValueError: If `groups` does not have the length of `y`. Invalid
            outcomes or actions are rejected with a `ValueError` by
            `costs.pair`.
    """
    outcomes, actions = costs.pair(y, acts)
    labels = numpy.asarray(groups)
    if labels.shape != outcomes.shape:
        raise ValueError("groups must have the length of y")
    return outcomes, actions, labels


def wilson(count: float, total: float, z: float) -> tuple[float, float]:
    """Computes the Wilson score interval for a proportion.

    Args:
        count: Number of successes.
        total: Number of trials; positive.
        z: Normal quantile of the interval.

    Returns:
        The lower and upper ends of the interval.
    """
    p = count / total
    denom = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * numpy.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    return float(centre - half / denom), float(centre + half / denom)


def newcombe(
    count_a: float,
    total_a: float,
    count_b: float,
    total_b: float,
    confidence: float,
) -> tuple[float, float]:
    """Computes Newcombe's hybrid score interval for `p_a - p_b`.

    Args:
        count_a: Successes in group a.
        total_a: Trials in group a; positive.
        count_b: Successes in group b.
        total_b: Trials in group b; positive.
        confidence: Two-sided confidence level of the interval.

    Returns:
        The lower and upper ends of the interval for the difference.
    """
    z = float(stats.norm.ppf(1 - (1 - confidence) / 2))
    pa, pb = count_a / total_a, count_b / total_b
    la, ua = wilson(count_a, total_a, z)
    lb, ub = wilson(count_b, total_b, z)
    d = pa - pb
    lower = d - float(numpy.sqrt((pa - la) ** 2 + (ub - pb) ** 2))
    upper = d + float(numpy.sqrt((ua - pa) ** 2 + (pb - lb) ** 2))
    return lower, upper


def rate_tables(
    y: NDArray[numpy.float64],
    acts: NDArray[numpy.float64],
    groups: NDArray[Any],
    criterion: str,
) -> dict[str, dict[Any, tuple[float, float]]]:
    """Counts `(numerator, denominator)` per group for each compared rate.

    Args:
        y: Outcomes, 0 or 1.
        acts: Actions, 0 or 1, aligned with `y`.
        groups: Group label per row.
        criterion: One of `demographic_parity`, `equal_opportunity`,
            `equalized_odds` and `predictive_parity`.

    Returns:
        For each rate the criterion compares (`selection`, `tpr`, `fpr` or
        `ppv`), a mapping from group label to the numerator and denominator
        of that rate. The denominator is the group's rows (`selection`),
        positives (`tpr`), negatives (`fpr`) or acted-on rows (`ppv`) and
        can be zero. A `criterion` other than those listed fails with a
        `KeyError`.
    """
    tables: dict[str, dict[Any, tuple[float, float]]] = {}
    wanted = {
        "demographic_parity": ("selection",),
        "equal_opportunity": ("tpr",),
        "equalized_odds": ("tpr", "fpr"),
        "predictive_parity": ("ppv",),
    }[criterion]
    for rate in wanted:
        table: dict[Any, tuple[float, float]] = {}
        for g in numpy.unique(groups):
            m = groups == g
            if rate == "selection":
                table[g] = (float(acts[m].sum()), float(m.sum()))
            elif rate == "tpr":
                table[g] = (float((acts[m] * y[m]).sum()), float(y[m].sum()))
            elif rate == "fpr":
                table[g] = (
                    float((acts[m] * (1 - y[m])).sum()),
                    float((1 - y[m]).sum()),
                )
            else:
                table[g] = (float((acts[m] * y[m]).sum()), float(acts[m].sum()))
        tables[rate] = table
    return tables


@dataclasses.dataclass(frozen=True, slots=True)
class SatisfactionTest:
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


def satisfaction_test(
    y: ArrayLike,
    acts: ArrayLike,
    groups: ArrayLike,
    constraint: decision.FairnessConstraint,
    alpha: float = 0.05,
) -> SatisfactionTest:
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
            inputs are also rejected with a `ValueError` by `prepared`.
    """
    outcomes, actions, labels = prepared(y, acts, groups)
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")
    tables = rate_tables(outcomes, actions, labels, constraint.criterion)
    names = sorted(str(g) for g in numpy.unique(labels))
    small = sorted(
        {
            str(g)
            for table in tables.values()
            for g, counts in table.items()
            if counts[1] < constraint.min_group
        }
    )
    if small or len(names) < MIN_GROUPS:
        return SatisfactionTest(
            NOT_ASSESSABLE,
            constraint.criterion,
            constraint.max_gap,
            None,
            None,
            tuple(small),
            alpha,
        )
    pairs = list(itertools.combinations(sorted(tables[next(iter(tables))]), 2))
    confidence = 1.0 - alpha / (len(pairs) * len(tables))
    point = 0.0
    upper = 0.0
    for table in tables.values():
        for a, b in pairs:
            lo, hi = newcombe(*table[a], *table[b], confidence)
            point = max(
                point,
                abs(table[a][0] / table[a][1] - table[b][0] / table[b][1]),
            )
            upper = max(upper, abs(lo), abs(hi))
    status = SATISFIED if upper <= constraint.max_gap else NOT_SATISFIED
    return SatisfactionTest(
        status,
        constraint.criterion,
        constraint.max_gap,
        point,
        upper,
        (),
        alpha,
    )


RATE_PARTS = {
    "selection": lambda y, a: (a, numpy.ones_like(a)),  # noqa: ARG005 -- table entries share the (y, a) call
    "tpr": lambda y, a: (a * y, y),
    "fpr": lambda y, a: (a * (1.0 - y), 1.0 - y),
    "ppv": lambda y, a: (a * y, a),
}
CRITERION_RATES = {
    "demographic_parity": ("selection",),
    "equal_opportunity": ("tpr",),
    "equalized_odds": ("tpr", "fpr"),
    "predictive_parity": ("ppv",),
}


def cluster_satisfaction_test(
    y: ArrayLike,
    acts: ArrayLike,
    groups: ArrayLike,
    clusters: ArrayLike,
    constraint: decision.FairnessConstraint,
    alpha: float = 0.05,
    draws: int = 1000,
    seed: int = 0,
) -> SatisfactionTest:
    """Tests satisfaction by resampling whole clusters of rows.

    `satisfaction_test` takes rows as independent, so rows that share a
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
            inputs are also rejected with a `ValueError` by `prepared`.
    """
    outcomes, actions, labels = prepared(y, acts, groups)
    keys = numpy.asarray(clusters)
    if keys.shape != outcomes.shape:
        raise ValueError("clusters must have the length of y")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")
    if draws < 1:
        raise ValueError("draws must be positive")
    tables = rate_tables(outcomes, actions, labels, constraint.criterion)
    names = sorted(numpy.unique(labels).tolist())
    small = sorted(
        {
            str(g)
            for table in tables.values()
            for g, counts in table.items()
            if counts[1] < constraint.min_group
        }
    )
    cluster_ids, inverse = numpy.unique(keys, return_inverse=True)
    if small or len(names) < MIN_GROUPS or len(cluster_ids) < MIN_CLUSTERS:
        return SatisfactionTest(
            NOT_ASSESSABLE,
            constraint.criterion,
            constraint.max_gap,
            None,
            None,
            tuple(small),
            alpha,
        )
    columns = []
    for rate in CRITERION_RATES[constraint.criterion]:
        numerator, denominator = RATE_PARTS[rate](outcomes, actions)
        for name in names:
            inside = (labels == name).astype(float)
            for part in (numerator, denominator):
                sums = numpy.zeros(len(cluster_ids))
                numpy.add.at(sums, inverse, part * inside)
                columns.append(sums)
    matrix = numpy.column_stack(columns).reshape(
        len(cluster_ids), -1, len(names), 2
    )

    def largest_gap(totals: NDArray[numpy.float64]) -> NDArray[numpy.float64]:
        """Computes the largest gap in any rate between groups.

        Args:
            totals: Numerators and denominators with last axes
                `(rate, group, 2)`.

        Returns:
            The largest difference between the highest and lowest group
            rate, over the rates, ignoring undefined rates; one value per
            leading index.
        """
        with numpy.errstate(divide="ignore", invalid="ignore"):
            rates = totals[..., 0] / totals[..., 1]
        spread = numpy.nanmax(rates, axis=-1) - numpy.nanmin(rates, axis=-1)
        return numpy.asarray(numpy.nanmax(spread, axis=-1))

    point = float(largest_gap(matrix.sum(axis=0)))
    rng = randomness.component_rng(seed, "decision.constraints.cluster", 0)
    weights = rng.multinomial(
        len(cluster_ids),
        numpy.full(len(cluster_ids), 1.0 / len(cluster_ids)),
        size=draws,
    ).astype(float)
    resampled = largest_gap(numpy.einsum("dk,krgp->drgp", weights, matrix))
    upper = max(point, float(numpy.quantile(resampled, 1.0 - alpha)))
    status = SATISFIED if upper <= constraint.max_gap else NOT_SATISFIED
    return SatisfactionTest(
        status,
        constraint.criterion,
        constraint.max_gap,
        point,
        upper,
        (),
        alpha,
    )


def base_rates(
    y: ArrayLike, groups: ArrayLike
) -> dict[str, tuple[float, float]]:
    """Counts the positive outcomes and rows of each group.

    Args:
        y: Outcomes, 0 or 1.
        groups: Group label per row.

    Returns:
        For each group label (as text), the number of positive outcomes and
        the number of rows.

    Raises:
        ValueError: If `groups` and `y` differ in length.
    """
    outcomes = costs.binary_vector(y, "y")
    labels = numpy.asarray(groups)
    if labels.shape != outcomes.shape:
        raise ValueError("groups and y must have the same length")
    return {
        str(g): (
            float(outcomes[labels == g].sum()),
            float((labels == g).sum()),
        )
        for g in numpy.unique(labels)
    }


def check_feasibility(
    constraints: Sequence[decision.FairnessConstraint],
    y: ArrayLike,
    groups: ArrayLike,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Checks that the criteria can hold together at the measured base rates.

    Raises when two incompatible criteria are declared and the base rates
    differ; otherwise reports the comparison.

    Args:
        constraints: The declared fairness constraints.
        y: Outcomes, 0 or 1.
        groups: Group label per row.
        alpha: Total level of the base-rate comparison, split over group
            pairs.

    Returns:
        A mapping with `base_rates` (the share of positives per group, by
        group label as text), `differ` (whether any pair of groups has base
        rates whose interval for the difference excludes zero), `conflicts`
        (the incompatible pairs of declared criteria, as lists) and
        `refused` (always False, since a refusal raises).

    Raises:
        InfeasibleConstraintsError: Naming the conflicting pair, when two
            declared criteria are incompatible and the groups' base rates
            differ (their interval for the difference excludes zero).
    """
    rates = base_rates(y, groups)
    names = sorted(rates)
    differ = False
    pairs = list(itertools.combinations(names, 2))
    for a, b in pairs:
        lo, hi = newcombe(
            *rates[a], *rates[b], 1.0 - alpha / max(len(pairs), 1)
        )
        if lo > 0.0 or hi < 0.0:
            differ = True
    criteria = [c.criterion for c in constraints]
    conflicts = [
        tuple(sorted(pair))
        for pair in itertools.combinations(sorted(set(criteria)), 2)
        if frozenset(pair) in INCOMPATIBLE
    ]
    if conflicts and differ:
        pair = conflicts[0]
        raise InfeasibleConstraintsError(
            f"{pair[0]} and {pair[1]} cannot both hold: the base rates "
            f"differ ({ {n: round(c / t, 4) for n, (c, t) in rates.items()} })"
        )
    return {
        "base_rates": {n: c / t for n, (c, t) in rates.items()},
        "differ": differ,
        "conflicts": [list(c) for c in conflicts],
        "refused": False,
    }


@dataclasses.dataclass(frozen=True, slots=True)
class GroupThresholds:
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
    for table in rate_tables(y, acts, labels, criterion).values():
        rates = [c / t if t > 0 else 0.0 for c, t in table.values()]
        worst = max(worst, max(rates) - min(rates))
    return worst


def fit_group_thresholds(
    scores: ArrayLike,
    y: ArrayLike,
    groups: ArrayLike,
    constraint: decision.FairnessConstraint,
    t: float,
    *,
    margin: float = 0.5,
) -> GroupThresholds:
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
        InfeasibleConstraintsError: If no sweep point meets the bound.
        ValueError: If the criterion is predictive parity, which group
            thresholds cannot move. Invalid inputs are also rejected with a
            `ValueError` by `prepared`.
    """
    s = numpy.asarray(scores, dtype=float)
    prepared_parts = prepared(y, numpy.zeros_like(s), groups)
    outcomes, labels = prepared_parts[0], prepared_parts[2]
    if constraint.criterion == "predictive_parity":
        raise ValueError("group thresholds cannot target predictive parity")
    bound = margin * constraint.max_gap
    common = (s >= t).astype(float)
    names = sorted(str(g) for g in numpy.unique(labels))
    gap = development_gap(outcomes, common, labels, constraint.criterion)
    if gap <= bound:
        return GroupThresholds(
            dict.fromkeys(names, t),
            None,
            costs.net_benefit(outcomes, common, t),
            gap,
            constrained=False,
        )
    positive_only = constraint.criterion in (
        "equal_opportunity",
        "equalized_odds",
    )
    best: GroupThresholds | None = None
    for rate in numpy.linspace(0.01, 0.99, GRID):
        cuts: dict[str, float] = {}
        for name in names:
            m = labels.astype(str) == name
            pool = s[m & (outcomes == 1)] if positive_only else s[m]
            if pool.size == 0:
                break
            cuts[name] = float(numpy.quantile(pool, 1.0 - rate))
        else:
            acts = GroupThresholds(cuts, rate, 0.0, 0.0, constrained=True).act(
                s, labels
            )
            achieved = development_gap(
                outcomes, acts, labels, constraint.criterion
            )
            if achieved > bound:
                continue
            value = costs.net_benefit(outcomes, acts, t)
            if best is None or value > best.net_benefit:
                best = GroupThresholds(
                    cuts, float(rate), value, achieved, constrained=True
                )
    if best is None:
        raise InfeasibleConstraintsError(
            "no group thresholds meet the development bound "
            f"{bound:.4f} for {constraint.criterion}"
        )
    return best


def frontier(
    scores: ArrayLike,
    y: ArrayLike,
    groups: ArrayLike,
    constraint: decision.FairnessConstraint,
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
        `fit_group_thresholds` is used. Errors other than infeasibility,
        including the refusal of predictive parity, propagate from
        `fit_group_thresholds`.
    """
    rows = []
    for gap in gaps:
        trial = dataclasses.replace(constraint, max_gap=gap)
        try:
            fitted = fit_group_thresholds(scores, y, groups, trial, t)
        except InfeasibleConstraintsError:
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


def price_of_constraint(
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
        netbenefit.resample_means(
            stats_matrix, netbenefit.factorize(clusters), replicates, rng
        )
    )
    return {
        "price": float(price(stats_matrix.mean(axis=0))),
        "lower": float(numpy.quantile(boot, 0.025)),
        "upper": float(numpy.quantile(boot, 0.975)),
    }
