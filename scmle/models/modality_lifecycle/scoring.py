"""Decision scores the modality explanations compare before and after masks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy

if TYPE_CHECKING:
    from collections.abc import Iterable

    from sklearn.linear_model import LogisticRegression


def anchor(
    model: LogisticRegression,
    features: numpy.ndarray,
    decision: numpy.ndarray | None = None,
) -> tuple[int, float]:
    """Scores one unmasked row for the class the model predicts.

    Args:
        model: Fitted classifier.
        features: Matrix with the one row to score.
        decision: The model's decision values for `features`, when the
            caller already holds them; computed here otherwise.

    Returns:
        The index of the tracked class (0 for a binary model) and the row's
        decision score for that class.
    """
    if decision is None:
        decision = numpy.asarray(model.decision_function(features))
    if decision.ndim == 1:
        return 0, float(decision[0])
    predicted = model.predict(features)[0]
    selected_class = int(numpy.flatnonzero(model.classes_ == predicted)[0])
    return selected_class, float(decision[0, selected_class])


def score(
    model: LogisticRegression, features: numpy.ndarray, selected_class: int
) -> float:
    """Scores one masked row for the tracked class.

    Args:
        model: Fitted classifier.
        features: Matrix with the one row to score.
        selected_class: Index of the tracked class; ignored for a binary
            model, whose decision values have one column.

    Returns:
        The row's decision score for the tracked class.
    """
    decision = numpy.asarray(model.decision_function(features))
    if decision.ndim == 1:
        return float(decision[0])
    return float(decision[0, selected_class])


def rank(terms: Iterable[tuple[str, float]]) -> list[tuple[str, float]]:
    """Keeps the ten terms with the largest absolute score change.

    Args:
        terms: Name and score change of each term.

    Returns:
        At most ten terms, largest absolute change first and ties by name.
    """
    return sorted(terms, key=lambda item: (-abs(item[1]), item[0]))[:10]
