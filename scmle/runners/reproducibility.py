"""Reproducibility across places: tolerances declared before the run.

`Tolerances` fixes, before any run, how far each metric of a result may differ
between the local reference and another place. `compare` reports every
metric: within tolerance or not, the difference and the tolerance. A miss is
published in the report; nothing is retuned afterwards.
"""

from collections.abc import Mapping
import dataclasses
import math
from typing import Any

from scmle.contracts import errors


@dataclasses.dataclass(frozen=True, slots=True)
class Tolerances:
    """Allowed absolute differences, declared in advance.

    Attributes:
        absolute: Metric name to allowed absolute difference.
        default: Allowed difference for a metric not listed.
    """

    absolute: Mapping[str, float]
    default: float = 0.0

    def __post_init__(self) -> None:
        """Validates the declaration.

        Raises:
            ConfigError: If a tolerance is negative or not finite.
        """
        values = [*self.absolute.values(), self.default]
        if any(not math.isfinite(v) or v < 0 for v in values):
            raise errors.ConfigError("tolerances must be finite and >= 0")

    def for_metric(self, name: str) -> float:
        """The tolerance of one metric."""
        return self.absolute.get(name, self.default)


def compare(
    reference: Mapping[str, float],
    candidate: Mapping[str, float],
    tolerances: Tolerances,
    *,
    place: str,
) -> dict[str, Any]:
    """Compares a result with the local reference.

    Args:
        reference: Metrics from the local reference run.
        candidate: Metrics from the run elsewhere.
        tolerances: The tolerances declared before the runs.
        place: Name of the place (for example `local`, `cluster`).

    Returns:
        A report with one row per metric and an overall `within_tolerance`.
        A metric missing on either side is a miss.
    """
    rows = []
    for name in sorted(set(reference) | set(candidate)):
        tolerance = tolerances.for_metric(name)
        if name not in reference or name not in candidate:
            rows.append({"metric": name, "within": False, "reason": "missing"})
            continue
        difference = abs(float(reference[name]) - float(candidate[name]))
        rows.append(
            {
                "metric": name,
                "reference": reference[name],
                "candidate": candidate[name],
                "difference": difference,
                "tolerance": tolerance,
                "within": difference <= tolerance,
            }
        )
    return {
        "place": place,
        "within_tolerance": all(row["within"] for row in rows),
        "metrics": rows,
    }
