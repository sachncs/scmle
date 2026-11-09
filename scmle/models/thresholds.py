"""Decision-threshold optimization for binary classifiers.

The threshold is chosen on out-of-fold validation probabilities against an
explicit objective, then frozen into `ThresholdClassifier`; the test set is
never consulted.
"""

import abc
from collections.abc import Callable
from collections.abc import Sequence
import dataclasses
from typing import Any
from typing import ClassVar
from typing import Self

import numpy as np
import pandas as pd
from sklearn import base

GRID = tuple(round(0.01 * step, 2) for step in range(1, 100))


class ThresholdObjective(abc.ABC):
    """A score to maximize when choosing the decision threshold.

    Attributes:
        name: Name used in `DecisionSpec.threshold_objective`.
    """

    name: ClassVar[str]

    @abc.abstractmethod
    def score(self, actual: np.ndarray, flagged: np.ndarray) -> float:
        """Scores a set of positive decisions (larger is better).

        Args:
            actual: True where the real label is the positive class.
            flagged: True where the model predicts the positive class.

        Returns:
            The objective value for these decisions; larger is better.
        """


class F1Objective(ThresholdObjective):
    """Harmonic mean of precision and recall for the positive class."""

    name = "f1"

    def score(self, actual: np.ndarray, flagged: np.ndarray) -> float:
        """Scores decisions by precision and recall of the positive class.

        Args:
            actual: True where the real label is the positive class.
            flagged: True where the model predicts the positive class.

        Returns:
            2 * true positives / (actual positives + flagged positives), or
            0.0 if both counts are zero.
        """
        true_positive = float((actual & flagged).sum())
        denominator = float(actual.sum() + flagged.sum())
        return 2 * true_positive / denominator if denominator else 0.0


class AccuracyObjective(ThresholdObjective):
    """Plain accuracy."""

    name = "accuracy"

    def score(self, actual: np.ndarray, flagged: np.ndarray) -> float:
        """Scores decisions by the share that are correct.

        Args:
            actual: True where the real label is the positive class.
            flagged: True where the model predicts the positive class.

        Returns:
            The fraction of rows where `flagged` equals `actual`.
        """
        return float((actual == flagged).mean())


class BalancedAccuracyObjective(ThresholdObjective):
    """Mean of sensitivity and specificity."""

    name = "balanced_accuracy"

    def score(self, actual: np.ndarray, flagged: np.ndarray) -> float:
        """Scores decisions by the mean of recall and specificity.

        Args:
            actual: True where the real label is the positive class.
            flagged: True where the model predicts the positive class.

        Returns:
            (recall + specificity) / 2; a term is 0.0 if its class is absent.
        """
        positives, negatives = actual.sum(), (~actual).sum()
        recall = (actual & flagged).sum() / positives if positives else 0.0
        specificity = (
            (~actual & ~flagged).sum() / negatives if negatives else 0.0
        )
        return float((recall + specificity) / 2)


@dataclasses.dataclass(frozen=True)
class CostObjective(ThresholdObjective):
    """Negative expected cost of errors (explicit operational costs).

    Attributes:
        false_positive_cost: Cost of flagging a negative.
        false_negative_cost: Cost of missing a positive.
    """

    false_positive_cost: float = 1.0
    false_negative_cost: float = 1.0
    name: ClassVar[str] = "cost"

    def score(self, actual: np.ndarray, flagged: np.ndarray) -> float:
        """Scores decisions by their negative mean cost.

        Args:
            actual: True where the real label is the positive class.
            flagged: True where the model predicts the positive class.

        Returns:
            Minus the total error cost divided by the number of rows (at
            least 1); closer to zero is better.
        """
        false_positives = float((~actual & flagged).sum())
        false_negatives = float((actual & ~flagged).sum())
        total = self.false_positive_cost * false_positives
        total += self.false_negative_cost * false_negatives
        return -total / max(len(actual), 1)


OBJECTIVES: dict[str, Callable[..., ThresholdObjective]] = {
    "f1": F1Objective,
    "accuracy": AccuracyObjective,
    "balanced_accuracy": BalancedAccuracyObjective,
    "cost": CostObjective,
}


def optimize_threshold(
    probabilities: np.ndarray,
    actual: np.ndarray,
    objective: ThresholdObjective,
    grid: Sequence[float] = GRID,
) -> tuple[float, float]:
    """Finds the threshold that maximizes the objective.

    Ties go to the threshold closest to 0.5 for stability.

    Args:
        probabilities: Positive-class probabilities (out-of-fold).
        actual: True where the label is the positive class.
        objective: What to maximize.
        grid: Candidate thresholds.

    Returns:
        The best threshold and its objective score.
    """
    scored = [
        (objective.score(actual, probabilities >= threshold), threshold)
        for threshold in grid
    ]
    best = max(
        scored, key=lambda pair: (round(pair[0], 12), -abs(pair[1] - 0.5))
    )
    return best[1], best[0]


class ThresholdClassifier(base.ClassifierMixin, base.BaseEstimator):
    """Applies a fixed decision threshold to a probabilistic classifier.

    Attributes:
        estimator: Classifier with `predict_proba`.
        threshold: Probability at or above which the positive class is chosen.
        estimator_: Fitted classifier.
        classes_: The two class labels; the second one is positive.
    """

    def __init__(
        self, estimator: base.BaseEstimator, threshold: float = 0.5
    ) -> None:
        """Initializes the wrapper.

        Args:
            estimator: Binary classifier with `predict_proba`.
            threshold: Decision threshold for the positive class.
        """
        self.estimator = estimator
        self.threshold = threshold

    @classmethod
    def around_fitted(cls, fitted: Any, threshold: float) -> Self:
        """Wraps an already fitted classifier without refitting it.

        Args:
            fitted: Fitted binary classifier with `predict_proba` and
                `classes_`.
            threshold: Decision threshold for the positive class.

        Returns:
            A wrapper that uses `fitted` as its fitted classifier.
        """
        wrapper = cls(fitted, threshold)
        wrapper.estimator_ = fitted
        wrapper.classes_ = fitted.classes_
        return wrapper

    def fit(self, features: Any, target: Any) -> Self:
        """Fits a clone of the wrapped classifier.

        Args:
            features: Training features.
            target: Training labels.

        Returns:
            This wrapper, fitted.
        """
        self.estimator_ = base.clone(self.estimator).fit(features, target)
        self.classes_ = self.estimator_.classes_
        return self

    def predict_proba(self, features: Any) -> np.ndarray:
        """Predicts class probabilities with the wrapped classifier.

        Args:
            features: Rows to predict.

        Returns:
            Array with one row per input row and one column per class, in
            `classes_` order.
        """
        return self.estimator_.predict_proba(features)

    def predict(self, features: Any) -> np.ndarray:
        """Predicts the positive class where probability >= threshold.

        Args:
            features: Rows to predict.

        Returns:
            Array of class labels, one per row: the second class where its
            probability is at or above the threshold, else the first.
        """
        positive = self.predict_proba(features)[:, 1] >= self.threshold
        return np.where(positive, self.classes_[1], self.classes_[0])


def out_of_fold_probabilities(
    build: Callable[[], Any],
    features: pd.DataFrame,
    target: pd.Series,
    folds: Sequence[tuple[Sequence[int], Sequence[int]]],
) -> tuple[np.ndarray, np.ndarray]:
    """Predicts every validation row with a model that never saw it.

    Works for any fold scheme (including expanding-window temporal folds
    that do not cover every row).

    Args:
        build: Factory of an unfitted candidate.
        features: Development features.
        target: Development labels.
        folds: `(train positions, validation positions)` pairs.

    Returns:
        Positive-class probabilities and the matching row positions.
    """
    probabilities, positions = [], []
    for train, validation in folds:
        model = build().fit(
            features.iloc[np.asarray(train)], target.iloc[np.asarray(train)]
        )
        probabilities.append(
            model.predict_proba(features.iloc[np.asarray(validation)])[:, 1]
        )
        positions.append(np.asarray(validation))
    return np.concatenate(probabilities), np.concatenate(positions)


MIN_GROUP_ROWS = 30
NUMERIC_LEVELS = 12
NUMERIC_BANDS = 4


@dataclasses.dataclass(frozen=True)
class GroupAssigner:
    """Maps a sensitive column to group names, fixed at training time.

    Categories keep their own name; numbers with many distinct values are cut
    into bands at the quantiles seen in training, so the same value lands in
    the same group at inference.

    Attributes:
        edges: Interior band edges for a numeric column, else empty.
    """

    edges: tuple[float, ...] = ()

    @classmethod
    def fit(cls, values: pd.Series) -> Self:
        """Learns the grouping from training values.

        Args:
            values: Training values of the sensitive column.

        Returns:
            An assigner with quartile band edges if the column is numeric
            with more than 12 distinct values, otherwise one that keeps
            categories as they are.
        """
        numeric = pd.api.types.is_numeric_dtype(values) and (
            values.nunique() > NUMERIC_LEVELS
        )
        if not numeric:
            return cls()
        cuts = np.quantile(
            values.dropna(), np.linspace(0, 1, NUMERIC_BANDS + 1)[1:-1]
        )
        return cls(tuple(float(c) for c in np.unique(cuts)))

    def assign(self, values: pd.Series) -> pd.Series:
        """Names the group of each row.

        Args:
            values: Values of the sensitive column.

        Returns:
            Series with the index of `values` holding the group name per row
            ("band_<n>" for numeric bands, else the value as text); missing
            values stay missing.
        """
        if self.edges:
            bands = np.searchsorted(self.edges, values.to_numpy(float))
            named = pd.Series(
                [f"band_{band}" for band in bands], index=values.index
            )
            return named.where(values.notna())
        return values.astype(object).where(values.notna()).astype(str)


class GroupThresholdClassifier(base.ClassifierMixin, base.BaseEstimator):
    """Decides with a separate probability threshold for each group.

    The wrapped classifier is untouched; only the cut between the two classes
    moves by group. Without the sensitive column in the rows (or for a group
    never seen) the default threshold applies.

    Attributes:
        estimator: Fitted binary classifier with `predict_proba`.
        column: Sensitive column the thresholds are keyed by.
        assigner: How values of that column become group names.
        thresholds: Group name to threshold.
        default_threshold: Threshold when the group is unknown.
        classes_: The two class labels; the second one is positive.
    """

    def __init__(
        self,
        estimator: Any,
        column: str,
        assigner: GroupAssigner,
        thresholds: dict[str, float],
        default_threshold: float = 0.5,
    ) -> None:
        """Wraps an already fitted classifier.

        Args:
            estimator: Fitted binary classifier with `predict_proba`.
            column: Sensitive column the thresholds are keyed by.
            assigner: How values of that column become group names.
            thresholds: Group name to threshold.
            default_threshold: Threshold for unknown or missing groups.
        """
        self.estimator = estimator
        self.column = column
        self.assigner = assigner
        self.thresholds = thresholds
        self.default_threshold = default_threshold
        self.classes_ = estimator.classes_

    def fit(self, features: Any, target: Any) -> Self:
        """Refits the classifier; the group thresholds are kept as given.

        Args:
            features: Training features.
            target: Training labels.

        Returns:
            This wrapper, with a refitted clone of the classifier.
        """
        self.estimator = base.clone(self.estimator).fit(features, target)
        self.classes_ = self.estimator.classes_
        return self

    def predict_proba(self, features: Any) -> np.ndarray:
        """Predicts class probabilities with the wrapped classifier.

        Args:
            features: Rows to predict.

        Returns:
            Array with one row per input row and one column per class, in
            `classes_` order.
        """
        return self.estimator.predict_proba(features)

    def cuts(self, features: Any) -> np.ndarray:
        """Looks up the threshold that applies to each row.

        Args:
            features: Rows to predict, ideally with the sensitive column.

        Returns:
            Float array with one threshold per row: that of the row's group,
            or the default if the column is absent or the group is unknown.
        """
        frame = pd.DataFrame(features)
        if self.column not in frame.columns:
            return np.full(len(frame), self.default_threshold)
        groups = self.assigner.assign(frame[self.column])
        return (
            groups.map(self.thresholds)
            .fillna(self.default_threshold)
            .to_numpy(float)
        )

    def predict(self, features: Any) -> np.ndarray:
        """Predicts the positive class where probability reaches the threshold.

        Args:
            features: Rows to predict.

        Returns:
            Array of class labels, one per row: the second class where its
            probability is at or above the row's group threshold, else the
            first.
        """
        positive = self.predict_proba(features)[:, 1] >= self.cuts(features)
        return np.where(positive, self.classes_[1], self.classes_[0])
