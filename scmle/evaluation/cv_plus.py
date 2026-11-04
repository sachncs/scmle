"""Stored fold models for CV+ intervals (regression).

`conformal.cv_plus_bounds` needs the K models that were each fit without one
held-out fold, the fold of every calibration row and the absolute error of
that row under the model that did not see it. `CvPlusEnsemble` keeps exactly
that, so a served bundle can answer with the CV+ interval of Barber, Candes,
Ramdas and Tibshirani (arXiv:1905.02928) instead of a margin pooled from
scores of different refits.

The rule needs every calibration row to be held out in exactly one fold.
`partition_folds` checks it and returns the fold index of each row; with
overlapping held-out sets (repeated or expanding-window folds) it returns
None and the caller keeps the pooled split-conformal margin. The size of the
stored models is bounded by `MAX_BYTES`: a larger ensemble is not kept and
the run says why, so a bundle does not grow by a factor of K without notice.
"""

from collections.abc import Sequence
import dataclasses
import io
from typing import Any

import joblib
import numpy as np
import pandas as pd

from scmle.evaluation import conformal

MAX_BYTES = 200_000_000
MIN_ROWS_FOR_SPLIT = 1000


def partition_folds(
    held_out: Sequence[Sequence[int]], rows: int
) -> np.ndarray | None:
    """Fold index of every row when the held-out sets partition the rows.

    Args:
        held_out: Positions held out by each fold, in fold order.
        rows: Number of development rows.

    Returns:
        An array with the fold of each held-out row, ordered by fold and then
        by position within the fold (the order calibration scores are
        stored), or None when a row is held out twice.
    """
    seen: set[int] = set()
    order: list[int] = []
    for index, positions in enumerate(held_out):
        for position in positions:
            if position in seen or not 0 <= position < rows:
                return None
            seen.add(position)
            order.append(index)
    return np.asarray(order, dtype=int)


def serialized_size(models: Sequence[Any]) -> int:
    """Bytes the fold models take when written with joblib."""
    buffer = io.BytesIO()
    joblib.dump(list(models), buffer)
    return buffer.getbuffer().nbytes


@dataclasses.dataclass(frozen=True)
class CvPlusEnsemble:
    """The K fold models with the residuals and folds of the calibration rows.

    Attributes:
        models: One fitted model per fold, each trained without that fold.
        fold_of_row: Fold index of each calibration row.
        residuals: Absolute out-of-fold error of each calibration row.
    """

    models: tuple[Any, ...]
    fold_of_row: np.ndarray
    residuals: np.ndarray

    def __post_init__(self) -> None:
        """Checks that every calibration row names a model."""
        if len(self.fold_of_row) != len(self.residuals):
            raise ValueError("fold_of_row and residuals differ in length")
        if len(self.residuals) < conformal.MIN_CALIBRATION_ROWS:
            raise ValueError(
                f"need at least {conformal.MIN_CALIBRATION_ROWS} "
                "calibration rows"
            )
        if len(self.fold_of_row) and (
            self.fold_of_row.min() < 0
            or self.fold_of_row.max() >= len(self.models)
        ):
            raise ValueError("fold_of_row names a fold without a model")

    def interval(
        self,
        features: pd.DataFrame,
        coverage: float = conformal.DEFAULT_COVERAGE,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Lower and upper CV+ limits at new rows.

        The guarantee of the paper is a coverage of at least `1 - 2a` for
        the nominal `1 - a`; the measured coverage on held-out data is what
        this package reports.
        """
        predictions = np.column_stack(
            [np.asarray(m.predict(features), dtype=float) for m in self.models]
        )
        return conformal.cv_plus_bounds(
            predictions, self.fold_of_row, self.residuals, coverage
        )
