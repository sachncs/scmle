"""Inner cross-validation that respects entity and time structure.

Calibration and stacking cross-fit models *inside* a training fold. A plain
k-fold there would put rows of one entity on both sides of an inner split, or
let later rows help predict earlier ones, so the splitter below follows the
same structure the outer validation does.
"""

from collections.abc import Iterator
from typing import Any

import numpy as np
import pandas as pd
from sklearn import model_selection

from scml.contracts import data as data_contracts


class StructuredKFold(model_selection.BaseCrossValidator):
    """K-fold over a training frame that keeps groups and time order intact.

    * With a group column, whole groups go to one inner fold.
    * Otherwise with a time column, rows are cut into contiguous blocks in
      time order (no shuffling), so an inner fold is a period.
    * Otherwise stratified shuffled k-fold for classification and plain
      shuffled k-fold for regression.

    The splitter reads the structure from the frame passed to `split`; if
    that frame lacks the columns (for example a bare matrix) it falls back to
    the unstructured behaviour.

    Attributes:
        n_splits: Number of inner folds.
        group_column: Entity column, if any.
        time_column: Ordering column, if any.
        classification: Whether to stratify by label.
        seed: Seed for shuffled folds.
    """

    def __init__(
        self,
        n_splits: int,
        group_column: str | None = None,
        time_column: str | None = None,
        classification: bool = True,
        seed: int = 0,
    ) -> None:
        """Initializes the splitter.

        Args:
            n_splits: Number of inner folds.
            group_column: Entity column, if any.
            time_column: Ordering column, if any.
            classification: Whether to stratify by label.
            seed: Seed for shuffled folds.
        """
        self.n_splits = n_splits
        self.group_column = group_column
        self.time_column = time_column
        self.classification = classification
        self.seed = seed

    def get_n_splits(
        self, features: Any = None, y: Any = None, groups: Any = None
    ) -> int:
        """Returns the number of inner folds."""
        return self.n_splits

    def split(
        self, features: Any, y: Any = None, groups: Any = None
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yields (train, validation) positions for each inner fold."""
        frame = features if isinstance(features, pd.DataFrame) else None
        if frame is not None and self.group_column in frame.columns:
            keys = frame[self.group_column].astype(str).to_numpy()
            yield from model_selection.GroupKFold(self.n_splits).split(
                frame, y, keys
            )
        elif frame is not None and self.time_column in frame.columns:
            times = pd.to_datetime(frame[self.time_column], errors="coerce")
            order = np.argsort(times.to_numpy(), kind="stable")
            blocks = np.array_split(order, self.n_splits)
            everyone = np.arange(len(frame))
            for block in blocks:
                yield np.setdiff1d(everyone, block), np.sort(block)
        elif self.classification and y is not None:
            yield from model_selection.StratifiedKFold(
                self.n_splits, shuffle=True, random_state=self.seed
            ).split(features, y)
        else:
            yield from model_selection.KFold(
                self.n_splits, shuffle=True, random_state=self.seed
            ).split(features)


def for_plan(
    plan: data_contracts.FeaturePlan,
    classification: bool,
    folds: int,
    seed: int,
) -> StructuredKFold:
    """Builds the inner splitter matching a feature plan's structure."""
    return StructuredKFold(
        folds, plan.group_column, plan.time_column, classification, seed
    )
