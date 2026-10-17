"""Estimators over a matrix of targets: one column per label or quantity."""

from typing import Any
from typing import Self

import numpy as np
from sklearn import base
from sklearn import dummy
from sklearn import ensemble


class PerTarget(base.BaseEstimator):
    """Fits one clone of an estimator for each target column.

    Attributes:
        estimator: Unfitted single-target estimator.
        classify: Whether the targets are binary labels.
        estimators_: The fitted clones, one per column.
    """

    def __init__(
        self, estimator: base.BaseEstimator, classify: bool = False
    ) -> None:
        """Initializes the wrapper."""
        self.estimator = estimator
        self.classify = classify

    def fit(self, features: Any, targets: Any) -> Self:
        """Fits each column; a column with one value becomes a constant."""
        matrix = np.asarray(targets)
        self.estimators_ = []
        for column in range(matrix.shape[1]):
            values = matrix[:, column]
            model = (
                dummy.DummyClassifier(strategy="prior")
                if self.classify and len(np.unique(values)) < 2
                else base.clone(self.estimator)
            )
            self.estimators_.append(model.fit(features, values))
        return self

    def predict(self, features: Any) -> np.ndarray:
        """One prediction column per target."""
        return np.column_stack(
            [np.asarray(m.predict(features)) for m in self.estimators_]
        )

    def predict_proba(self, features: Any) -> np.ndarray:
        """Probability of the positive label for each target column."""
        columns = []
        for model in self.estimators_:
            probabilities = np.asarray(model.predict_proba(features))
            positive = (
                list(model.classes_).index(1) if 1 in model.classes_ else None
            )
            columns.append(
                probabilities[:, positive]
                if positive is not None
                else np.zeros(probabilities.shape[0])
            )
        return np.column_stack(columns)


class NativeMultiOutput(base.BaseEstimator):
    """A tree ensemble that models all targets jointly.

    Attributes:
        estimator: Unfitted estimator that accepts a target matrix.
        classify: Whether the targets are binary labels.
        estimator_: The fitted estimator.
    """

    def __init__(
        self, estimator: base.BaseEstimator, classify: bool = False
    ) -> None:
        """Initializes the wrapper."""
        self.estimator = estimator
        self.classify = classify

    def fit(self, features: Any, targets: Any) -> Self:
        """Fits one model on the whole target matrix."""
        self.estimator_ = base.clone(self.estimator).fit(
            features, np.asarray(targets)
        )
        return self

    def predict(self, features: Any) -> np.ndarray:
        """One prediction column per target."""
        return np.asarray(self.estimator_.predict(features))

    def predict_proba(self, features: Any) -> np.ndarray:
        """Probability of the positive label for each target column."""
        found = self.estimator_.predict_proba(features)
        return np.column_stack(
            [
                np.asarray(p)[:, list(c).index(1)]
                if 1 in c
                else np.zeros(np.asarray(p).shape[0])
                for p, c in zip(found, self.estimator_.classes_, strict=True)
            ]
        )


def forest(classify: bool, seed: int) -> NativeMultiOutput:
    """Random forest over all targets together."""
    inner = (
        ensemble.RandomForestClassifier(n_estimators=150, random_state=seed)
        if classify
        else ensemble.RandomForestRegressor(n_estimators=150, random_state=seed)
    )
    return NativeMultiOutput(inner, classify)
