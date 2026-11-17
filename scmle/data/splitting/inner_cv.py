"""Inner cross-validation that respects entity and time structure.

Calibration and stacking cross-fit models *inside* a training fold. A plain
k-fold there would put rows of one entity on both sides of an inner split, or
let later rows help predict earlier ones, so the splitter below follows the
same structure the outer validation does.

Cross-fitted calibration also needs every class in every inner fold, which a
rare class on a small training slice cannot give; `Calibrated` lowers the
fold count to what the rarest class allows instead of failing.
"""

from collections.abc import Iterator
import copy
from typing import Any

import numpy
import pandas
from sklearn import calibration
from sklearn import model_selection

from scmle.contracts import data
from scmle.contracts import errors

# Cross-fitting needs one fold to fit on and another to calibrate on, so two
# folds is the fewest that still keeps every calibration row out of the fit.
MIN_FOLDS = 2
# scikit-learn's own default when `cv` is None.
DEFAULT_FOLDS = 5
# A classifier needs at least two classes to separate.
MIN_CLASSES = 2


class Splitter(model_selection.BaseCrossValidator):  # type: ignore[misc]  # scikit-learn ships no type information, so its base class is Any
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
    plan: data.Blueprint,
    classification: bool,
    folds: int,
    seed: int,
) -> Splitter:
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
    return Splitter(
        folds, plan.group_column, plan.time_column, classification, seed
    )


class Calibrated(calibration.CalibratedClassifierCV):  # type: ignore[misc]  # scikit-learn ships no type information, so its base class is Any
    """Cross-fitted calibration whose fold count follows the rarest class.

    scikit-learn refuses outright when a class has fewer examples than the
    configured folds, which happens on real tables with a rare class once a
    learning curve or a small fold slices the training rows. This lowers the
    fold count to the rarest class's count (at least `MIN_FOLDS`) and keeps
    the configured splitter's structure; the count used is recorded as
    `folds_`. A class with a single example cannot be in every fold, so then
    each fold keeps its own calibrated model (`ensemble=True`), a fold
    without the class giving it no probability, and the folds' average
    gives it the rest. The configured `cv` and `ensemble` are left as they
    were.

    Attributes:
        cv: The configured cross-validator, as scikit-learn takes it.
        ensemble: Whether each fold keeps its own calibrated model.
        folds_: The fold count the last fit used, None for given splits.
    """

    cv: Any
    ensemble: Any
    folds_: int | None

    def fit(  # pyright: ignore[reportIncompatibleMethodOverride]  # the base names these parameters X and y; scikit-learn passes them positionally
        self,
        features: Any,
        labels: Any,
        sample_weight: Any = None,
        **params: Any,
    ) -> "Calibrated":
        """Fits with as many folds as the rarest class allows.

        Args:
            features: Training features.
            labels: Training labels.
            sample_weight: Optional row weights.
            **params: Routed to the calibrated estimator's `fit`.

        Returns:
            This fitted calibrator.

        Raises:
            InputError: If the labels hold fewer than two classes.
        """
        counts = numpy.unique(numpy.asarray(labels), return_counts=True)[1]
        if len(counts) < MIN_CLASSES:
            raise errors.InputError(
                "calibration needs at least two classes in the training rows"
            )
        configured: Any = self.cv
        requested = DEFAULT_FOLDS if configured is None else configured
        if not isinstance(requested, int):
            requested = getattr(requested, "n_splits", None)
        rarest = int(numpy.min(counts))
        if requested is None or rarest >= requested:
            self.folds_ = requested
            super().fit(features, labels, sample_weight, **params)
            return self
        self.folds_ = max(MIN_FOLDS, rarest)
        splitter = shrink(configured, self.folds_)
        ensembled: Any = self.ensemble
        self.cv = list(splitter.split(features, labels))
        if rarest < MIN_FOLDS:
            self.ensemble = True
        try:
            super().fit(features, labels, sample_weight, **params)
        finally:
            self.cv = configured
            self.ensemble = ensembled
        return self


def shrink(cv: Any, folds: int) -> Any:
    """A copy of a cross-validator with fewer folds.

    Args:
        cv: An int, None, or a splitter with an `n_splits` attribute.
        folds: The fold count to use.

    Returns:
        Stratified k-fold (what scikit-learn uses for an int or None on
        labels) or a copy of `cv` with `n_splits` set to `folds`.
    """
    if cv is None or isinstance(cv, int):
        return model_selection.StratifiedKFold(folds)
    smaller = copy.copy(cv)
    smaller.n_splits = folds
    return smaller
