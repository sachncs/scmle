"""Success criteria: did the model reach what the problem statement asked?

A project can state the scores it must reach on the locked rows (for example
ROC-AUC of at least 0.8, or RMSE of at most 12). After the one test release,
each criterion is compared with the achieved score in the metric's own
direction, and the outcome is reported beside the scores themselves.
"""

import dataclasses
from typing import Any

from scmle.contracts import errors
from scmle.evaluation import metrics


@dataclasses.dataclass(frozen=True)
class CriterionResult:
    """One success criterion against the achieved score.

    Attributes:
        metric: Metric name.
        required: The threshold from the problem statement.
        achieved: The locked-test score (NaN when it could not be computed).
        higher_is_better: The metric's direction.
        met: Whether the score satisfies the threshold.
    """

    metric: str
    required: float
    achieved: float
    higher_is_better: bool
    met: bool

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready record."""
        return dataclasses.asdict(self)


def check(
    criteria: dict[str, float],
    test_metrics: dict[str, float],
    catalog: metrics.MetricCatalog = metrics.CATALOG,
) -> list[CriterionResult]:
    """Compares each criterion with the achieved score.

    Raises:
        ConfigError: If a criterion names a metric that was not reported
            for this task.
    """
    results = []
    for name, required in criteria.items():
        if name not in test_metrics:
            raise errors.ConfigError(
                f"success criterion '{name}' is not a metric of this task; "
                f"reported: {sorted(test_metrics)}"
            )
        achieved = float(test_metrics[name])
        higher = catalog.metrics[name].greater_is_better
        met = achieved >= required if higher else achieved <= required
        results.append(
            CriterionResult(name, float(required), achieved, higher, bool(met))
        )
    return results
