"""Group rates, their intervals and the feasibility of a set of criteria.

The rates each fairness criterion compares (`CRITERION_RATES`), their counts
per group (`tabulate`), the Wilson interval of one rate and Newcombe's hybrid
interval of a difference, and the refusal of incompatible criteria at the
measured base rates (`check_feasibility`).
"""

from collections.abc import Sequence
import itertools
from typing import Any

import numpy
from numpy.typing import ArrayLike
from numpy.typing import NDArray
from scipy import stats

from scmle.assurance.charter import decision
from scmle.assurance.framing import costs
from scmle.contracts import errors

INCOMPATIBLE = (
    frozenset({"demographic_parity", "predictive_parity"}),
    frozenset({"demographic_parity", "equalized_odds"}),
    frozenset({"equalized_odds", "predictive_parity"}),
)


class InfeasibleError(ValueError):
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
        raise errors.InputError("groups must have the length of y")
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


def tabulate(
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
    outcomes = costs.dichotomous(y, "y")
    labels = numpy.asarray(groups)
    if labels.shape != outcomes.shape:
        raise errors.InputError("groups and y must have the same length")
    return {
        str(g): (
            float(outcomes[labels == g].sum()),
            float((labels == g).sum()),
        )
        for g in numpy.unique(labels)
    }


def check_feasibility(
    constraints: Sequence[decision.Constraint],
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
        InfeasibleError: Naming the conflicting pair, when two
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
        raise InfeasibleError(
            f"{pair[0]} and {pair[1]} cannot both hold: the base rates "
            f"differ ({ {n: round(c / t, 4) for n, (c, t) in rates.items()} })"
        )
    return {
        "base_rates": {n: c / t for n, (c, t) in rates.items()},
        "differ": differ,
        "conflicts": [list(c) for c in conflicts],
        "refused": False,
    }
