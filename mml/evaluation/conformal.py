"""Split-conformal uncertainty: prediction intervals and sets with coverage.

Given out-of-sample errors of a model on data it was not trained on, the
finite-sample quantile of those errors yields a margin such that a new
exchangeable row falls inside it with at least the requested probability.
Regression gets an interval around the prediction; classification gets the
set of labels whose probability is not too low.
"""

from collections.abc import Sequence
import dataclasses
import math
from typing import Any
from typing import Self

import numpy as np

MIN_CALIBRATION_ROWS = 20


def finite_sample_quantile(scores: np.ndarray, coverage: float) -> float:
    """Conformal quantile of nonconformity scores.

    Args:
        scores: Nonconformity scores on held-out rows.
        coverage: Desired coverage in (0, 1).

    Returns:
        The smallest score such that at least `coverage` of new rows are
        expected to score no higher. Infinity when there are too few scores
        for the requested coverage.
    """
    count = len(scores)
    rank = math.ceil((count + 1) * coverage)
    if rank > count:
        return math.inf
    return float(np.sort(scores)[rank - 1])


@dataclasses.dataclass(frozen=True)
class ConformalCalibrator:
    """Calibrated margin for intervals (regression) or sets (classification).

    Attributes:
        kind: `regression` or `classification`.
        coverage: Targeted coverage.
        margin: Absolute-error margin (regression) or the maximum
            nonconformity `1 - probability` allowed in a set (classification).
        rows: Number of calibration rows.
        classes: Class labels, in probability-column order (classification).
    """

    kind: str
    coverage: float
    margin: float
    rows: int
    classes: tuple[Any, ...] = ()

    @classmethod
    def for_regression(
        cls,
        observed: Sequence[float],
        predicted: Sequence[float],
        coverage: float = 0.9,
    ) -> Self:
        """Calibrates from out-of-sample predictions.

        Args:
            observed: True values of held-out rows.
            predicted: Out-of-sample predictions for the same rows.
            coverage: Desired coverage.

        Raises:
            ValueError: With too few rows or an invalid coverage.
        """
        errors = np.abs(np.asarray(observed) - np.asarray(predicted))
        return cls(
            "regression",
            coverage,
            finite_sample_quantile(cls.checked(errors, coverage), coverage),
            len(errors),
        )

    @classmethod
    def for_classification(
        cls,
        labels: Sequence[Any],
        probabilities: np.ndarray,
        classes: Sequence[Any],
        coverage: float = 0.9,
    ) -> Self:
        """Calibrates from out-of-sample class probabilities.

        Args:
            labels: True labels of held-out rows.
            probabilities: Out-of-sample probabilities, one column per class.
            classes: Class labels in column order.
            coverage: Desired coverage.

        Raises:
            ValueError: With too few rows, an invalid coverage, or labels
                outside `classes`.
        """
        position = {label: index for index, label in enumerate(classes)}
        if any(label not in position for label in labels):
            raise ValueError("labels outside the known classes")
        true_probability = np.asarray(probabilities)[
            np.arange(len(labels)), [position[label] for label in labels]
        ]
        scores = 1.0 - true_probability
        return cls(
            "classification",
            coverage,
            finite_sample_quantile(cls.checked(scores, coverage), coverage),
            len(scores),
            tuple(classes),
        )

    @staticmethod
    def checked(scores: np.ndarray, coverage: float) -> np.ndarray:
        """Validates calibration input."""
        if not 0.0 < coverage < 1.0:
            raise ValueError("coverage must lie strictly between 0 and 1")
        if len(scores) < MIN_CALIBRATION_ROWS:
            raise ValueError(
                f"need at least {MIN_CALIBRATION_ROWS} calibration rows"
            )
        return scores

    def interval(
        self, predictions: Sequence[float]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Prediction intervals for a regression model.

        Raises:
            ValueError: For a classification calibrator.
        """
        if self.kind != "regression":
            raise ValueError("intervals need a regression calibrator")
        values = np.asarray(predictions, dtype=float)
        return values - self.margin, values + self.margin

    def sets(self, probabilities: np.ndarray) -> list[list[Any]]:
        """Prediction sets for a classifier.

        Raises:
            ValueError: For a regression calibrator.
        """
        if self.kind != "classification":
            raise ValueError("sets need a classification calibrator")
        keep = 1.0 - np.asarray(probabilities) <= self.margin
        return [
            [
                label
                for label, flag in zip(self.classes, row, strict=True)
                if flag
            ]
            for row in keep
        ]

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-serializable mapping."""
        record = dataclasses.asdict(self)
        record["classes"] = list(self.classes)
        record["margin"] = None if math.isinf(self.margin) else self.margin
        return record

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> Self:
        """Rebuilds a calibrator from `to_dict` output."""
        margin = record["margin"]
        return cls(
            record["kind"],
            record["coverage"],
            math.inf if margin is None else margin,
            record["rows"],
            tuple(record["classes"]),
        )
