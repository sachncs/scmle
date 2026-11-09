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
stored models is bounded by `MAX_BYTES` (a charter's `uncertainty` block may
set another cap): a larger ensemble is not kept and the run says why, so a
bundle does not grow by a factor of K without notice. From
`MIN_ROWS_FOR_SPLIT` calibration rows the pooled out-of-fold scores alone are
served and no model is kept (`plan_scores`).
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


POOLED = "pooled_out_of_fold"
FOLD_MODELS = "cv_plus"


@dataclasses.dataclass(frozen=True)
class ScorePlan:
    """How the calibration scores of a regression bundle are served.

    Attributes:
        mode: `pooled_out_of_fold` (one margin from the scores of all
            held-out rows, each scored by the model that did not see it) or
            `cv_plus` (the margin as before, and the K fold models kept for
            CV+ intervals).
        folds: Fold index of each calibration row when the models are kept.
        reason: Why the fold models are not kept, None when they are.
    """

    mode: str
    folds: np.ndarray | None
    reason: str | None


def plan_scores(
    held_out: Sequence[Sequence[int]],
    rows: int,
    observed: int,
    models: Sequence[Any],
    max_bytes: int = MAX_BYTES,
) -> ScorePlan:
    """Chooses between the pooled out-of-fold scores and the CV+ models.

    The pooled out-of-fold margin is always available. With `MIN_ROWS_FOR_SPLIT`
    or more calibration rows it is what the bundle serves and no fold model
    is kept. Below that, the K fold models are kept for CV+ when the held-out
    sets partition the rows and the models fit in `max_bytes`.

    Args:
        held_out: Positions held out by each fold, in fold order.
        rows: Number of development rows.
        observed: Number of calibration scores.
        models: The fold models.
        max_bytes: Largest serialized size of the models to keep.

    Returns:
        A plan with mode `cv_plus`, the fold of each calibration row and no
        reason when the fold models are kept; otherwise mode
        `pooled_out_of_fold`, no folds and the reason they are not kept.
    """
    folds = partition_folds(held_out, rows)
    reason = None
    if folds is None or len(folds) != observed:
        reason = "held-out folds overlap; no CV+ ensemble"
    elif observed >= MIN_ROWS_FOR_SPLIT:
        reason = (
            f"{observed} calibration rows; pooled margin is used "
            f"from {MIN_ROWS_FOR_SPLIT}"
        )
    elif serialized_size(models) > max_bytes:
        reason = f"fold models exceed {max_bytes} bytes"
    if reason is not None or folds is None:
        return ScorePlan(POOLED, None, reason)
    return ScorePlan(FOLD_MODELS, folds, None)


def serialized_size(models: Sequence[Any]) -> int:
    """Bytes the fold models take when written with joblib.

    Args:
        models: The fold models.

    Returns:
        The size in bytes of the serialized list of models.
    """
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
        """Checks that every calibration row names a model.

        Raises:
            ValueError: If `fold_of_row` and `residuals` differ in length,
                there are fewer than `conformal.MIN_CALIBRATION_ROWS`
                calibration rows, or a row names a fold that has no model.
        """
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

        Args:
            features: New rows to predict for.
            coverage: Nominal coverage of the interval.

        Returns:
            The lower and the upper limit, each with one value per row of
            `features`. Invalid input is rejected by
            `conformal.cv_plus_bounds`.
        """
        predictions = np.column_stack(
            [np.asarray(m.predict(features), dtype=float) for m in self.models]
        )
        return conformal.cv_plus_bounds(
            predictions, self.fold_of_row, self.residuals, coverage
        )
