"""Inner cross-validation that respects entity and time structure.

Calibration and stacking cross-fit models *inside* a training fold. A plain
k-fold there would put rows of one entity on both sides of an inner split, or
let later rows help predict earlier ones, so the splitter below follows the
same structure the outer validation does.
"""

from collections.abc import Iterator
from typing import Any

import numpy
import pandas
from sklearn import model_selection

from scmle.contracts import data


class StructuredKFold(model_selection.BaseCrossValidator):  # type: ignore[misc]  # scikit-learn ships no type information, so its base class is Any
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

    def get_n_splits(  # pyright: ignore[reportIncompatibleMethodOverride]  # the base names this parameter X; scikit-learn passes it positionally
        self,
        features: Any = None,  # noqa: ARG002 -- scikit-learn cross-validator get_n_splits signature
        y: Any = None,  # noqa: ARG002 -- scikit-learn cross-validator get_n_splits signature
        groups: Any = None,  # noqa: ARG002 -- scikit-learn cross-validator get_n_splits signature
    ) -> int:
        """Returns the number of inner folds.

        Args:
            features: Ignored; present for the scikit-learn interface.
            y: Ignored; present for the scikit-learn interface.
            groups: Ignored; present for the scikit-learn interface.

        Returns:
            The configured number of inner folds.
        """
        return self.n_splits

    def split(  # pyright: ignore[reportIncompatibleMethodOverride]  # the base names this parameter X; scikit-learn passes it positionally
        self,
        features: Any,
        y: Any = None,
        groups: Any = None,  # noqa: ARG002 -- scikit-learn cross-validator split signature
    ) -> Iterator[tuple[numpy.ndarray, numpy.ndarray]]:
        """Yields (train, validation) positions for each inner fold.

        Args:
            features: The training frame. A DataFrame is needed for the group
                and time structures; anything else gets the unstructured
                fallback.
            y: Labels, used to stratify when `classification` is set.
            groups: Ignored; the group column of `features` is used instead.

        Yields:
            A pair of integer arrays: the positions to fit on and the
            positions to validate on, one pair per inner fold.
        """
        frame = features if isinstance(features, pandas.DataFrame) else None
        if frame is not None and self.group_column in frame.columns:
            keys = frame[self.group_column].astype(str).to_numpy()
            yield from model_selection.GroupKFold(self.n_splits).split(
                frame, y, keys
            )
        elif frame is not None and self.time_column in frame.columns:
            times = pandas.to_datetime(frame[self.time_column], errors="coerce")
            order = numpy.argsort(times.to_numpy(), kind="stable")
            blocks = numpy.array_split(order, self.n_splits)
            everyone = numpy.arange(len(frame))
            for block in blocks:
                yield numpy.setdiff1d(everyone, block), numpy.sort(block)
        elif self.classification and y is not None:
            yield from model_selection.StratifiedKFold(
                self.n_splits, shuffle=True, random_state=self.seed
            ).split(features, y)
        else:
            yield from model_selection.KFold(
                self.n_splits, shuffle=True, random_state=self.seed
            ).split(features)


def for_plan(
    plan: data.FeaturePlan,
    classification: bool,
    folds: int,
    seed: int,
) -> StructuredKFold:
    """Builds the inner splitter matching a feature plan's structure.

    Args:
        plan: The feature plan whose group and time columns are used.
        classification: Whether to stratify by label.
        folds: Number of inner folds.
        seed: Seed for shuffled folds.

    Returns:
        A splitter using the group and time columns of the
        feature plan.
    """
    return StructuredKFold(
        folds, plan.group_column, plan.time_column, classification, seed
    )
