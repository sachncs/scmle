"""Per-row sample weights taken from a column of the training data."""

import inspect
from typing import Any
from typing import Self

import numpy as np
import pandas as pd
from sklearn import base
from sklearn import pipeline


class WeightedEstimator(base.BaseEstimator):
    """Fits a pipeline so each training row counts by its weight.

    The weight column travels with the features, is read at fit time and
    ignored at prediction time (so inference never needs it). Models whose
    `fit` accepts `sample_weight` receive the weights directly. For the
    others (deep networks, wrapped estimators) rows are drawn with
    probability proportional to weight, which gives the same expected
    training distribution.

    Attributes:
        estimator: Unfitted pipeline whose final step is named `model`.
        column: Name of the weight column in the training features.
        random_state: Seed for the weighted draw.
        estimator_: Fitted clone of `estimator`.
    """

    def __init__(
        self,
        estimator: base.BaseEstimator,
        column: str,
        random_state: int = 0,
    ) -> None:
        """Initializes the wrapper.

        Args:
            estimator: Pipeline to fit with weights.
            column: Weight column name.
            random_state: Seed for the weighted draw.
        """
        self.estimator = estimator
        self.column = column
        self.random_state = random_state

    @property
    def event_column(self) -> str:
        """The wrapped survival pipeline's event column, so scorers find it."""
        return self.estimator.event_column

    @property
    def query_column(self) -> str:
        """The wrapped ranking pipeline's query column."""
        return self.estimator.query_column

    def accepts_weights(self, candidate: base.BaseEstimator) -> bool:
        """Whether the pipeline's model step takes `sample_weight`."""
        if not isinstance(candidate, pipeline.Pipeline):
            return False
        model = candidate.named_steps.get("model")
        if model is None:
            return False
        return "sample_weight" in inspect.signature(model.fit).parameters

    def fit(self, features: Any, target: Any) -> Self:
        """Fits the pipeline weighted by the weight column."""
        frame = pd.DataFrame(features).reset_index(drop=True)
        labels = pd.Series(np.asarray(target))
        weights = frame[self.column].to_numpy(dtype=float)
        candidate = base.clone(self.estimator)
        if self.accepts_weights(candidate):
            candidate.fit(frame, labels, model__sample_weight=weights)
        else:
            generator = np.random.default_rng(self.random_state)
            drawn = generator.choice(
                len(frame), size=len(frame), p=weights / weights.sum()
            )
            candidate.fit(frame.iloc[drawn], labels.iloc[drawn])
        self.estimator_ = candidate
        return self

    def predict(self, features: Any) -> np.ndarray:
        """Predicts with the fitted pipeline."""
        return self.estimator_.predict(features)

    def predict_survival(self, features: Any, times: Any) -> np.ndarray:
        """Survival curves from the fitted survival pipeline."""
        return self.estimator_.predict_survival(features, times)


class WeightedClassifier(base.ClassifierMixin, WeightedEstimator):
    """Weighted fitting for classifiers."""

    def fit(self, features: Any, target: Any) -> Self:
        """Fits and exposes the class labels."""
        super().fit(features, target)
        self.classes_ = self.estimator_.classes_
        return self

    def predict_proba(self, features: Any) -> np.ndarray:
        """Returns class probabilities in `classes_` order."""
        return self.estimator_.predict_proba(features)


class WeightedRegressor(base.RegressorMixin, WeightedEstimator):
    """Weighted fitting for regressors."""


def weighted(
    estimator: base.BaseEstimator, column: str, seed: int
) -> WeightedEstimator:
    """Wraps a candidate so rows are weighted by `column`."""
    kind = (
        WeightedClassifier
        if base.is_classifier(estimator)
        else WeightedRegressor
    )
    return kind(estimator, column, seed)
