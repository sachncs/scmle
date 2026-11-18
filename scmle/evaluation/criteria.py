"""Success criteria: did the model reach what the problem statement asked?

A project can state the scores it must reach on the locked rows (for example
ROC-AUC of at least 0.8, or RMSE of at most 12). After the one test release,
each criterion is compared with the achieved score in the metric's own
direction, and the outcome is reported beside the scores themselves.
"""

import dataclasses
from typing import Any
from typing import ClassVar

from scmle.contracts import errors
from scmle.contracts import records
from scmle.evaluation import metrics

HIGHER = "higher"
LOWER = "lower"


def orient(layout: records.Layout) -> records.Layout:
    """Writes the stored direction flag as the direction's name.

    Args:
        layout: The record with `direction` as a bool.

    Returns:
        The record with `direction` as `higher` or `lower`.
    """
    if isinstance(layout.get("direction"), bool):
        layout["direction"] = HIGHER if layout["direction"] else LOWER
    return layout


def flag(layout: records.Layout) -> records.Layout:
    """Undoes `orient`: writes the direction's name as a bool.

    Args:
        layout: The record with `direction` as `higher` or `lower`.

    Returns:
        The record with `direction` true for `higher`.
    """
    if isinstance(layout.get("direction"), str):
        layout["direction"] = layout["direction"] == HIGHER
    return layout


@dataclasses.dataclass(frozen=True)
class Criterion:
    """One success criterion against the achieved score.

    Attributes:
        metric: Metric name.
        required: The threshold from the problem statement.
        achieved: The locked-test score (NaN when it could not be computed).
        higher_is_better: The metric's direction.
        met: Whether the score satisfies the threshold.
        migrations: The written layout's version steps, oldest first;
            version 2 writes `higher_is_better` as `direction`, `higher` or
            `lower`.
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = (
        records.Migration(
            renames={"higher_is_better": "direction"},
            change=orient,
            revert=flag,
        ),
    )

    metric: str
    required: float
    achieved: float
    higher_is_better: bool
    met: bool

    def to_dict(self) -> dict[str, Any]:
        """Returns the JSON-ready record.

        Returns:
            A mapping of every field of the result, with the metric's
            direction as `higher` or `lower`.
        """
        return records.upgrade(
            self.migrations, dataclasses.asdict(self), records.ORIGIN
        )


def check(
    criteria: dict[str, float],
    test_metrics: dict[str, float],
    catalog: metrics.Repertoire = metrics.CATALOG,
) -> list[Criterion]:
    """Compares each criterion with the achieved score.

    Args:
        criteria: Required score per metric name.
        test_metrics: Achieved score per metric name.
        catalog: Metric catalog giving each metric's direction.

    Returns:
        One result per criterion, in the order of `criteria`. A criterion is
        met when the score is at least (higher-is-better metrics) or at most
        (otherwise) the required value.

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
            Criterion(name, float(required), achieved, higher, bool(met))
        )
    return results
