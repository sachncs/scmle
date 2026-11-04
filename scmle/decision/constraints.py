"""Fairness as explicit constraints on a frozen rule (feature F11, F).

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

import numpy as np
from numpy.typing import ArrayLike
from numpy.typing import NDArray
from scipy import stats

from scmle.core import randomness
from scmle.data.charter import decision as charter_decision
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


class InfeasibleConstraintsError(ValueError):
    """The declared criteria cannot hold together for these base rates."""


def prepared(
    y: ArrayLike, acts: ArrayLike, groups: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[Any]]:
    """Validated outcomes, actions and group labels."""
    outcomes, actions = costs.pair(y, acts)
    labels = np.asarray(groups)
    if labels.shape != outcomes.shape:
        raise ValueError("groups must have the length of y")
    return outcomes, actions, labels


def wilson(count: float, total: float, z: float) -> tuple[float, float]:
    """Wilson score interval for a proportion."""
    p = count / total
    denom = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    return float(centre - half / denom), float(centre + half / denom)


def newcombe(
    count_a: float,
    total_a: float,
    count_b: float,
    total_b: float,
    confidence: float,
) -> tuple[float, float]:
    """Newcombe's hybrid score interval for `p_a - p_b`."""
    z = float(stats.norm.ppf(1 - (1 - confidence) / 2))
    pa, pb = count_a / total_a, count_b / total_b
    la, ua = wilson(count_a, total_a, z)
    lb, ub = wilson(count_b, total_b, z)
    d = pa - pb
    lower = d - float(np.sqrt((pa - la) ** 2 + (ub - pb) ** 2))
    upper = d + float(np.sqrt((ua - pa) ** 2 + (pb - lb) ** 2))
    return lower, upper


def rate_tables(
    y: NDArray[np.float64],
    acts: NDArray[np.float64],
    groups: NDArray[Any],
    criterion: str,
) -> dict[str, dict[Any, tuple[float, float]]]:
    """Per group `(count, total)` for each rate the criterion compares."""
    tables: dict[str, dict[Any, tuple[float, float]]] = {}
    wanted = {
        "demographic_parity": ("selection",),
        "equal_opportunity": ("tpr",),
        "equalized_odds": ("tpr", "fpr"),
        "predictive_parity": ("ppv",),
    }[criterion]
    for rate in wanted:
        table: dict[Any, tuple[float, float]] = {}
        for g in np.unique(groups):
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
        alpha: Total level split over pairs and rates.
    """

    status: str
    criterion: str
    max_gap: float
    point_gap: float | None
    upper_bound: float | None
    small_groups: tuple[str, ...]
    alpha: float

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready record."""
        return dataclasses.asdict(self)


def satisfaction_test(
    y: ArrayLike,
    acts: ArrayLike,
    groups: ArrayLike,
    constraint: charter_decision.FairnessConstraint,
    alpha: float = 0.05,
) -> SatisfactionTest:
    """Tests that the frozen rule's gap is at most `max_gap`, once.

    The claim is `satisfied` only if the upper bound of every compared gap is
    at most `constraint.max_gap`.
    """
    outcomes, actions, labels = prepared(y, acts, groups)
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")
    tables = rate_tables(outcomes, actions, labels, constraint.criterion)
    names = sorted(str(g) for g in np.unique(labels))
    small = sorted(
        {
            str(g)
            for table in tables.values()
            for g, (ignored, total) in table.items()
            if total < constraint.min_group
        }
    )
    if small or len(names) < 2:
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


def base_rates(
    y: ArrayLike, groups: ArrayLike
) -> dict[str, tuple[float, float]]:
    """Per group `(positives, rows)`."""
    outcomes = costs.binary_vector(y, "y")
    labels = np.asarray(groups)
    return {
        str(g): (
            float(outcomes[labels == g].sum()),
            float((labels == g).sum()),
        )
        for g in np.unique(labels)
    }


def check_feasibility(
    constraints: Sequence[charter_decision.FairnessConstraint],
    y: ArrayLike,
    groups: ArrayLike,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Refuses criteria that cannot hold together at the measured base rates.

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

    def act(self, scores: ArrayLike, groups: ArrayLike) -> NDArray[np.float64]:
        """The rule's 0/1 actions."""
        s = np.asarray(scores, dtype=float)
        g = np.asarray(groups)
        cut = np.array([self.cuts[str(x)] for x in g])
        return (s >= cut).astype(float)


def development_gap(
    y: NDArray[np.float64],
    acts: NDArray[np.float64],
    labels: NDArray[Any],
    criterion: str,
) -> float:
    """Largest gap between groups in the criterion's rates (point values)."""
    worst = 0.0
    for table in rate_tables(y, acts, labels, criterion).values():
        rates = [c / t if t > 0 else 0.0 for c, t in table.values()]
        worst = max(worst, max(rates) - min(rates))
    return worst


def fit_group_thresholds(
    scores: ArrayLike,
    y: ArrayLike,
    groups: ArrayLike,
    constraint: charter_decision.FairnessConstraint,
    t: float,
    *,
    margin: float = 0.5,
) -> GroupThresholds:
    """Group cuts that maximise net benefit within `margin * max_gap`.

    The common cut at the Bayes threshold `t` is used when it already meets
    the development bound. Otherwise a common target rate (selection rate or
    true-positive rate) is swept over a grid and each group takes the cut
    whose rate is closest to it; the best net benefit within the bound wins.
    Criteria that group thresholds cannot move (predictive parity) and
    unreachable bounds raise.

    Raises:
        InfeasibleConstraintsError: If no sweep point meets the bound.
        ValueError: If the criterion is not reachable by group thresholds.
    """
    s = np.asarray(scores, dtype=float)
    outcomes, ignored, labels = prepared(y, np.zeros_like(s), groups)
    if constraint.criterion == "predictive_parity":
        raise ValueError("group thresholds cannot target predictive parity")
    bound = margin * constraint.max_gap
    common = (s >= t).astype(float)
    names = sorted(str(g) for g in np.unique(labels))
    gap = development_gap(outcomes, common, labels, constraint.criterion)
    if gap <= bound:
        return GroupThresholds(
            dict.fromkeys(names, t),
            None,
            costs.net_benefit(outcomes, common, t),
            gap,
            False,
        )
    positive_only = constraint.criterion in (
        "equal_opportunity",
        "equalized_odds",
    )
    best: GroupThresholds | None = None
    for rate in np.linspace(0.01, 0.99, GRID):
        cuts: dict[str, float] = {}
        for name in names:
            m = labels.astype(str) == name
            pool = s[m & (outcomes == 1)] if positive_only else s[m]
            if pool.size == 0:
                break
            cuts[name] = float(np.quantile(pool, 1.0 - rate))
        else:
            acts = GroupThresholds(cuts, rate, 0.0, 0.0, True).act(s, labels)
            achieved = development_gap(
                outcomes, acts, labels, constraint.criterion
            )
            if achieved > bound:
                continue
            value = costs.net_benefit(outcomes, acts, t)
            if best is None or value > best.net_benefit:
                best = GroupThresholds(cuts, float(rate), value, achieved, True)
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
    constraint: charter_decision.FairnessConstraint,
    t: float,
    gaps: Sequence[float],
) -> list[dict[str, Any]]:
    """Net benefit along a sweep of `max_gap`; infeasible points say so."""
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
    """`NB(unconstrained) - NB(constrained)` with a 95% bootstrap interval."""
    outcomes, free = costs.pair(y, unconstrained)
    ignored, bound = costs.pair(y, constrained)
    w = costs.odds(t)
    stats_matrix = np.column_stack(
        [
            free * outcomes,
            free * (1 - outcomes),
            bound * outcomes,
            bound * (1 - outcomes),
        ]
    )

    def price(m: NDArray[np.float64]) -> NDArray[np.float64]:
        return (m[..., 0] - w * m[..., 1]) - (m[..., 2] - w * m[..., 3])

    rng = randomness.component_rng(seed, "decision.constraints.price", 0)
    boot = price(
        netbenefit.resample_means(
            stats_matrix, netbenefit.factorize(clusters), replicates, rng
        )
    )
    return {
        "price": float(price(stats_matrix.mean(axis=0))),
        "lower": float(np.quantile(boot, 0.025)),
        "upper": float(np.quantile(boot, 0.975)),
    }
