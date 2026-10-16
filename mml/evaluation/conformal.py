"""Split-conformal uncertainty: prediction intervals and sets with coverage.

Given out-of-sample errors of a model on data it was not trained on, the
finite-sample quantile of those errors yields a margin such that a new
exchangeable row falls inside it with at least the requested probability.
Regression gets an interval around the prediction; classification gets the
set of labels whose probability is not too low. The calibrator keeps the
scores themselves, so any coverage level can be asked for later.
"""

from collections.abc import Sequence
import dataclasses
import math
from typing import Any
from typing import Self

import numpy as np

MIN_CALIBRATION_ROWS = 20
DEFAULT_COVERAGE = 0.9


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
    """Calibration scores for intervals (regression) or sets (classification).

    Attributes:
        kind: `regression` or `classification`.
        scores: Nonconformity scores of held-out rows: absolute errors
            (regression) or one minus the probability of the true class
            (classification).
        classes: Class labels, in probability-column order (classification).
    """

    kind: str
    scores: tuple[float, ...]
    classes: tuple[Any, ...] = ()

    @classmethod
    def for_regression(
        cls, observed: Sequence[float], predicted: Sequence[float]
    ) -> Self:
        """Calibrates from out-of-sample predictions.

        Args:
            observed: True values of held-out rows.
            predicted: Out-of-sample predictions for the same rows.

        Raises:
            ValueError: With too few rows.
        """
        errors = np.abs(np.asarray(observed) - np.asarray(predicted))
        return cls("regression", cls.checked(errors))

    @classmethod
    def for_classification(
        cls,
        labels: Sequence[Any],
        probabilities: np.ndarray,
        classes: Sequence[Any],
    ) -> Self:
        """Calibrates from out-of-sample class probabilities.

        Args:
            labels: True labels of held-out rows.
            probabilities: Out-of-sample probabilities, one column per class.
            classes: Class labels in column order.

        Raises:
            ValueError: With too few rows or labels outside `classes`.
        """
        position = {label: index for index, label in enumerate(classes)}
        if any(label not in position for label in labels):
            raise ValueError("labels outside the known classes")
        true_probability = np.asarray(probabilities)[
            np.arange(len(labels)), [position[label] for label in labels]
        ]
        return cls(
            "classification",
            cls.checked(1.0 - true_probability),
            tuple(classes),
        )

    @staticmethod
    def checked(scores: np.ndarray) -> tuple[float, ...]:
        """Validates calibration input and freezes it."""
        if len(scores) < MIN_CALIBRATION_ROWS:
            raise ValueError(
                f"need at least {MIN_CALIBRATION_ROWS} calibration rows"
            )
        return tuple(float(s) for s in scores)

    def margin(self, coverage: float = DEFAULT_COVERAGE) -> float:
        """The margin guaranteeing `coverage` for exchangeable new rows.

        Raises:
            ValueError: If coverage is not strictly between 0 and 1.
        """
        if not 0.0 < coverage < 1.0:
            raise ValueError("coverage must lie strictly between 0 and 1")
        return finite_sample_quantile(np.asarray(self.scores), coverage)

    def interval(
        self,
        predictions: Sequence[float] | np.ndarray,
        coverage: float = DEFAULT_COVERAGE,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Prediction intervals for a regression model.

        Raises:
            ValueError: For a classification calibrator.
        """
        if self.kind != "regression":
            raise ValueError("intervals need a regression calibrator")
        values = np.asarray(predictions, dtype=float)
        margin = self.margin(coverage)
        return values - margin, values + margin

    def sets(
        self,
        probabilities: np.ndarray,
        coverage: float = DEFAULT_COVERAGE,
    ) -> list[list[Any]]:
        """Prediction sets for a classifier.

        Raises:
            ValueError: For a regression calibrator.
        """
        if self.kind != "classification":
            raise ValueError("sets need a classification calibrator")
        keep = 1.0 - np.asarray(probabilities) <= self.margin(coverage)
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
        return {
            "kind": self.kind,
            "scores": list(self.scores),
            "classes": list(self.classes),
        }

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> Self:
        """Rebuilds a calibrator from `to_dict` output."""
        return cls(
            record["kind"],
            tuple(record["scores"]),
            tuple(record["classes"]),
        )
