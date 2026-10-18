"""Counterfactual explanations: the smallest change that flips a prediction.

Starting from one row, the search changes one actionable column at a time to
the background value that moves the prediction most toward the wanted outcome,
stops as soon as the outcome is reached, then drops every change that turned
out to be unnecessary. Columns that are not listed as actionable are never
touched, so the answer stays something a person could actually do.
"""

from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
from typing import Any

import numpy as np
import pandas as pd

QUANTILE_STEPS = 21
MAX_CATEGORIES = 30
MIN_SCALE = 1e-9


@dataclasses.dataclass(frozen=True)
class Counterfactual:
    """The outcome of a counterfactual search.

    Attributes:
        found: Whether the wanted outcome was reached.
        original: The row as given (actionable and fixed columns).
        changed: Column to (old value, new value) for every change made.
        row: The row after the changes.
        distance: Total cost of the changes: each numeric change in units of
            the column's interquartile range, each category change as 1.
        prediction: The model's prediction for `row`.
    """

    found: bool
    original: Mapping[str, Any]
    changed: Mapping[str, tuple[Any, Any]]
    row: Mapping[str, Any]
    distance: float
    prediction: Any

    @property
    def n_changed(self) -> int:
        """How many columns changed."""
        return len(self.changed)


class CounterfactualSearch:
    """Greedy search over actionable columns with a pruning pass."""

    def __init__(
        self,
        progress: Callable[[pd.DataFrame], np.ndarray],
        reached: Callable[[pd.DataFrame], np.ndarray],
        predict: Callable[[pd.DataFrame], np.ndarray],
        background: pd.DataFrame,
        actionable: Sequence[str],
        max_changes: int = 3,
    ) -> None:
        """Binds the search to a model and to the values it may try.

        Args:
            progress: Score per row; larger means closer to the outcome.
            reached: Whether each row already has the wanted outcome.
            predict: Prediction per row (reported in the result).
            background: Typical rows; their values are the candidates.
            actionable: Columns the search may change.
            max_changes: Most columns it may change.
        """
        self.progress = progress
        self.reached = reached
        self.predict = predict
        self.actionable = list(actionable)
        self.max_changes = max_changes
        self.values = {
            c: self.candidate_values(background[c]) for c in actionable
        }
        self.scales = {c: self.scale_of(background[c]) for c in actionable}

    @staticmethod
    def candidate_values(column: pd.Series) -> list[Any]:
        """Values worth trying: quantiles for numbers, common categories."""
        present = column.dropna()
        if pd.api.types.is_numeric_dtype(present) and not (
            pd.api.types.is_bool_dtype(present)
        ):
            grid = np.quantile(present, np.linspace(0, 1, QUANTILE_STEPS))
            return sorted({float(v) for v in grid})
        return present.value_counts().head(MAX_CATEGORIES).index.tolist()

    @staticmethod
    def scale_of(column: pd.Series) -> float:
        """Interquartile range of a numeric column (1 for categories)."""
        present = column.dropna()
        if pd.api.types.is_numeric_dtype(present) and not (
            pd.api.types.is_bool_dtype(present)
        ):
            low, high = np.quantile(present, [0.25, 0.75])
            return float(max(high - low, present.std(), MIN_SCALE))
        return 1.0

    def cost(self, column: str, old: Any, new: Any) -> float:
        """Price of changing one column."""
        if isinstance(new, (int, float, np.number)) and not isinstance(
            new, bool
        ):
            return abs(float(new) - float(old)) / self.scales[column]
        return 1.0

    def distance(
        self, original: Mapping[str, Any], current: Mapping[str, Any]
    ) -> float:
        """Total cost of the differences between two rows."""
        return float(
            sum(
                self.cost(c, original[c], current[c])
                for c in self.actionable
                if current[c] != original[c]
            )
        )

    def candidates(
        self, current: dict[str, Any], locked: set[str]
    ) -> tuple[pd.DataFrame, list[tuple[str, Any]]]:
        """Every one-column change from `current`, as a batch of rows."""
        rows, moves = [], []
        for column in self.actionable:
            if column in locked:
                continue
            for value in self.values[column]:
                if value != current[column]:
                    rows.append({**current, column: value})
                    moves.append((column, value))
        return pd.DataFrame(rows), moves

    def search(self, row: Mapping[str, Any]) -> Counterfactual:
        """Finds a small change reaching the wanted outcome, if one exists."""
        original = dict(row)
        current = dict(row)
        changed: dict[str, Any] = {}
        while not self.reached(pd.DataFrame([current]))[0]:
            if len(changed) >= self.max_changes:
                break
            batch, moves = self.candidates(current, set(changed))
            if batch.empty:
                break
            scores = self.progress(batch)
            baseline = float(self.progress(pd.DataFrame([current]))[0])
            order = np.argsort(-scores, kind="stable")
            best = order[0]
            if scores[best] <= baseline:
                break
            column, value = moves[best]
            current[column] = value
            changed[column] = value
        found = bool(self.reached(pd.DataFrame([current]))[0])
        if found:
            current, changed = self.prune(original, current, changed)
        frame = pd.DataFrame([current])
        return Counterfactual(
            found=found,
            original=original,
            changed={c: (original[c], current[c]) for c in changed},
            row=current,
            distance=self.distance(original, current),
            prediction=self.predict(frame)[0],
        )

    def prune(
        self,
        original: dict[str, Any],
        current: dict[str, Any],
        changed: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reverts changes that were not needed, largest cost first."""
        kept = dict(changed)
        order = sorted(
            kept,
            key=lambda c: -self.cost(c, original[c], current[c]),
        )
        for column in order:
            trial = {**current, column: original[column]}
            if self.reached(pd.DataFrame([trial]))[0]:
                current = trial
                del kept[column]
        return current, kept
