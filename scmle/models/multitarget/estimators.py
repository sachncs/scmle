"""Estimators over a matrix of targets: one column per label or quantity."""

from typing import Any
from typing import Self

import numpy
from sklearn import base
from sklearn import dummy
from sklearn import ensemble

MIN_DISTINCT_VALUES = 2


class PerTarget(base.BaseEstimator):  # type: ignore[misc]  # scikit-learn ships no type information
    """Fits one clone of an estimator for each target column.

    Attributes:
        estimator: Unfitted single-target estimator.
        classify: Whether the targets are binary labels.
        estimators_: The fitted clones, one per column.
    """

    def __init__(
        self, estimator: base.BaseEstimator, classify: bool = False
    ) -> None:
        """Initializes the wrapper.

        Args:
            estimator: Unfitted estimator, cloned for fitting.
            classify: Whether the targets are binary labels.
        """
        self.estimator = estimator
        self.classify = classify

    def fit(self, features: Any, targets: Any) -> Self:
        """Fits each column; a column with one value becomes a constant.

        Args:
            features: Training matrix.
            targets: Target matrix of shape `(n_rows, n_targets)`.

        Returns:
            This fitted estimator.
        """
        matrix = numpy.asarray(targets)
        self.estimators_ = []
        for column in range(matrix.shape[1]):
            values = matrix[:, column]
            model = (
                dummy.DummyClassifier(strategy="prior")
                if self.classify
                and len(numpy.unique(values)) < MIN_DISTINCT_VALUES
                else base.clone(self.estimator)
            )
            self.estimators_.append(model.fit(features, values))
        return self

    def predict(self, features: Any) -> numpy.ndarray:
        """One prediction column per target.

        Args:
            features: Matrix to predict for.

        Returns:
            Array of shape `(n_rows, n_targets)`.
        """
        return numpy.column_stack(
            [numpy.asarray(m.predict(features)) for m in self.estimators_]
        )

    def predict_proba(self, features: Any) -> numpy.ndarray:
        """Probability of the positive label for each target column.

        Args:
            features: Matrix to predict for.

        Returns:
            Array of shape `(n_rows, n_targets)`; a target whose fitted classes
            do not include the label 1 gets probability zero.
        """
        columns = []
        for model in self.estimators_:
            probabilities = numpy.asarray(model.predict_proba(features))
            positive = (
                list(model.classes_).index(1) if 1 in model.classes_ else None
            )
            columns.append(
                probabilities[:, positive]
                if positive is not None
                else numpy.zeros(probabilities.shape[0])
            )
        return numpy.column_stack(columns)


class NativeMultiOutput(base.BaseEstimator):  # type: ignore[misc]  # scikit-learn ships no type information
    """A tree ensemble that models all targets jointly.

    Attributes:
        estimator: Unfitted estimator that accepts a target matrix.
        classify: Whether the targets are binary labels.
        estimator_: The fitted estimator.
    """

    def __init__(
        self, estimator: base.BaseEstimator, classify: bool = False
    ) -> None:
        """Initializes the wrapper.

        Args:
            estimator: Unfitted estimator that accepts a target matrix.
            classify: Whether the targets are binary labels.
        """
        self.estimator = estimator
        self.classify = classify

    def fit(self, features: Any, targets: Any) -> Self:
        """Fits one model on the whole target matrix.

        Args:
            features: Training matrix.
            targets: Target matrix of shape `(n_rows, n_targets)`.

        Returns:
            This fitted estimator.
        """
        self.estimator_ = base.clone(self.estimator).fit(
            features, numpy.asarray(targets)
        )
        return self

    def predict(self, features: Any) -> numpy.ndarray:
        """One prediction column per target.

        Args:
            features: Matrix to predict for.

        Returns:
            Array of shape `(n_rows, n_targets)`.
        """
        return numpy.asarray(self.estimator_.predict(features))

    def predict_proba(self, features: Any) -> numpy.ndarray:
        """Probability of the positive label for each target column.

        Args:
            features: Matrix to predict for.

        Returns:
            Array of shape `(n_rows, n_targets)`; a target whose fitted classes
            do not include the label 1 gets probability zero.
        """
        found = self.estimator_.predict_proba(features)
        return numpy.column_stack(
            [
                numpy.asarray(p)[:, list(c).index(1)]
                if 1 in c
                else numpy.zeros(numpy.asarray(p).shape[0])
                for p, c in zip(found, self.estimator_.classes_, strict=True)
            ]
        )


def forest(classify: bool, seed: int) -> NativeMultiOutput:
    """Random forest over all targets together.

    Args:
        classify: True for a classifier, False for a regressor.
        seed: Random seed of the forest.

    Returns:
        An unfitted forest wrapped to model every target jointly.
    """
    inner = (
        ensemble.RandomForestClassifier(n_estimators=150, random_state=seed)
        if classify
        else ensemble.RandomForestRegressor(n_estimators=150, random_state=seed)
    )
    return NativeMultiOutput(inner, classify)
