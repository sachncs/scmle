"""Observed outcomes against the approved expectation.

The cost of one served case is the cell ``c[action][outcome]`` of the
chartered decision-cost model. The observed mean cost per case gets a normal
approximation interval; that interval, not the mean, is compared with the
range the stakeholder approved. The statement it returns says what the
interval does and does not show.
"""

from collections.abc import Mapping
import math
from typing import Any

Z95 = 1.959964
WITHIN = "within"
ABOVE = "above"
BELOW = "below"
INSUFFICIENT = "insufficient"
ABSENT = "absent"
UNCERTAINTY = (
    "The interval is a normal-approximation 95% interval for the mean cost "
    "per case over the {n} cases whose outcome had arrived. It is not a "
    "forecast and not a guarantee for later cases. A cost that is above the "
    "approved range is a finding to act on; one inside it does not show "
    "the model caused any business result, because no counterfactual was "
    "observed."
)


def optimal_threshold(costs: Mapping[str, float]) -> float:
    """Computes the cut on the probability that minimises expected cost.

    Args:
        costs: The cost model with ``c_00``, ``c_01``, ``c_10``, ``c_11`` (the
            first index is the action, the second the outcome).

    Returns:
        The probability at which acting and not acting cost the same.
    """
    false_positive = costs["c_10"] - costs["c_00"]
    false_negative = costs["c_01"] - costs["c_11"]
    return false_positive / (false_positive + false_negative)


def case_costs(
    costs: Mapping[str, float],
    threshold: float,
    probability: list[float],
    label: list[int],
) -> list[float]:
    """Computes the cost of each case when the rule acts at the threshold.

    Args:
        costs: The cost model with ``c_00``, ``c_01``, ``c_10``, ``c_11``.
        threshold: The rule acts when the probability is at least this.
        probability: Predicted probability of each case.
        label: Observed outcome (0 or 1) of each case.

    Returns:
        The cost of each case, in the order given.
    """
    found = []
    for p_value, outcome in zip(probability, label, strict=True):
        action = 1 if p_value >= threshold else 0
        found.append(float(costs[f"c_{action}{int(outcome)}"]))
    return found


def interval(values: list[float]) -> tuple[float, float, float]:
    """Computes the mean and a normal-approximation 95% interval.

    Args:
        values: Costs of the cases; at least one.

    Returns:
        The mean, the lower bound and the upper bound; both bounds equal the
        mean when there is a single value.
    """
    n = len(values)
    mean = math.fsum(values) / n
    if n < 2:
        return mean, mean, mean
    spread = math.fsum((v - mean) ** 2 for v in values) / (n - 1)
    half = Z95 * math.sqrt(spread / n)
    return mean, mean - half, mean + half


def track(
    feed: Mapping[str, Any] | None,
    costs: Mapping[str, float],
    expectation: Mapping[str, Any],
    minimum_cases: int,
) -> dict[str, Any]:
    """Compares the observed cost per case with the approved range.

    Args:
        feed: The retained ``observed_outcomes.json``, or None.
        costs: The chartered cost model.
        expectation: The stakeholder's approved range (``low``, ``high``).
        minimum_cases: Fewest cases before the comparison is made.

    Returns:
        A JSON-ready record; ``status`` is ``within``, ``above``, ``below``,
        ``insufficient`` or ``absent``. ``above`` means the interval lies wholly
        over the approved range, ``below`` wholly under it.
    """
    if not feed or not feed.get("batches"):
        return {"status": ABSENT, "cases": 0}
    threshold = float(feed["threshold"])
    per_batch = []
    every: list[float] = []
    for batch in feed["batches"]:
        found = case_costs(
            costs, threshold, batch["probability"], batch["label"]
        )
        every.extend(found)
        per_batch.append(
            {
                "batch": batch["batch"],
                "cases": len(found),
                "mean_cost": math.fsum(found) / len(found),
            }
        )
    mean, low, high = interval(every)
    status = WITHIN
    if len(every) < minimum_cases:
        status = INSUFFICIENT
    elif low > float(expectation["high"]):
        status = ABOVE
    elif high < float(expectation["low"]):
        status = BELOW
    return {
        "status": status,
        "cases": len(every),
        "threshold": threshold,
        "mean_cost": mean,
        "interval": [low, high],
        "level": 0.95,
        "approved_range": [expectation["low"], expectation["high"]],
        "batches": per_batch,
        "source": feed.get("source", ""),
        "uncertainty": UNCERTAINTY.format(n=len(every)),
    }
