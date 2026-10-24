"""Uplift estimators: how much a treatment changes the outcome, per row.

Every estimator follows one contract: `fit(features, outcome, treatment)`
where `treatment` is 1 for treated rows and 0 for control rows, and
`predict` returns the estimated change in the outcome if the row were
treated (the conditional average treatment effect). The estimates assume
that, given the features, treatment was assigned independently of the
outcome (no unmeasured confounding), which a randomized experiment
guarantees and observational data do not.
"""

from typing import Any
from typing import Self

import numpy as np
from sklearn import base
from sklearn import ensemble

MIN_ARM_ROWS = 20


def outcome_model(classify: bool, seed: int) -> base.BaseEstimator:
    """A fresh outcome model: boosted trees for a label or a number."""
    if classify:
        return ensemble.HistGradientBoostingClassifier(random_state=seed)
    return ensemble.HistGradientBoostingRegressor(random_state=seed)


def expected(model: Any, features: Any, classify: bool) -> np.ndarray:
    """Expected outcome: probability of the positive label, or the value."""
    if not classify:
        return np.asarray(model.predict(features), dtype=float)
    probabilities = np.asarray(model.predict_proba(features))
    if probabilities.shape[1] == 1:
        return np.full(probabilities.shape[0], float(model.classes_[0] == 1))
    return probabilities[:, list(model.classes_).index(1)]


def split_arms(
    treatment: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Boolean masks of the treated and control rows."""
    flag = np.asarray(treatment).astype(int) == 1
    return flag, ~flag


class TwoModelLearner(base.BaseEstimator):
    """T-learner: one outcome model per arm; uplift is their difference.

    Attributes:
        classify: Whether the outcome is a 0/1 label.
        random_state: Seed.
    """

    def __init__(self, classify: bool = True, random_state: int = 0) -> None:
        """Stores the settings (scikit-learn convention)."""
        self.classify = classify
        self.random_state = random_state

    def fit(self, features: Any, outcome: Any, treatment: Any = None) -> Self:
        """Fits the treated and control outcome models."""
        treated, control = split_arms(
            np.zeros(len(outcome)) if treatment is None else treatment
        )
        values = np.asarray(outcome)
        self.treated_ = outcome_model(self.classify, self.random_state).fit(
            features[treated] if treated.any() else features,
            values[treated] if treated.any() else values,
        )
        self.control_ = outcome_model(self.classify, self.random_state).fit(
            features[control] if control.any() else features,
            values[control] if control.any() else values,
        )
        return self

    def predict(self, features: Any) -> np.ndarray:
        """Estimated effect of treating each row."""
        return expected(self.treated_, features, self.classify) - expected(
            self.control_, features, self.classify
        )


class SingleModelLearner(base.BaseEstimator):
    """S-learner: one outcome model with the treatment as a feature.

    Attributes:
        classify: Whether the outcome is a 0/1 label.
        random_state: Seed.
    """

    def __init__(self, classify: bool = True, random_state: int = 0) -> None:
        """Stores the settings."""
        self.classify = classify
        self.random_state = random_state

    @staticmethod
    def stacked(features: Any, flag: float) -> np.ndarray:
        """Features with a constant treatment column appended."""
        matrix = np.asarray(features)
        return np.column_stack([matrix, np.full(matrix.shape[0], flag)])

    def fit(self, features: Any, outcome: Any, treatment: Any = None) -> Self:
        """Fits one model on features plus the treatment indicator."""
        flag = np.zeros(len(outcome)) if treatment is None else treatment
        self.model_ = outcome_model(self.classify, self.random_state).fit(
            np.column_stack([np.asarray(features), np.asarray(flag)]),
            np.asarray(outcome),
        )
        return self

    def predict(self, features: Any) -> np.ndarray:
        """Prediction with treatment switched on minus switched off."""
        return expected(
            self.model_, self.stacked(features, 1.0), self.classify
        ) - expected(self.model_, self.stacked(features, 0.0), self.classify)


class CrossLearner(base.BaseEstimator):
    """X-learner: imputes each row's missing outcome, then models the gap.

    Outcome models per arm predict what each row would have had under the
    other arm; the imputed individual effects are regressed on the features
    separately in each arm, and the two effect models are blended by the
    propensity of treatment. It does well when the arms are unbalanced.

    Attributes:
        classify: Whether the outcome is a 0/1 label.
        random_state: Seed.
    """

    def __init__(self, classify: bool = True, random_state: int = 0) -> None:
        """Stores the settings."""
        self.classify = classify
        self.random_state = random_state

    def fit(self, features: Any, outcome: Any, treatment: Any = None) -> Self:
        """Fits outcome, effect and propensity models."""
        matrix = np.asarray(features)
        values = np.asarray(outcome, dtype=float)
        treated, control = split_arms(
            np.zeros(len(values)) if treatment is None else treatment
        )
        self.degenerate_ = not (treated.any() and control.any())
        if self.degenerate_:
            return self  # One arm only: no effect can be estimated.
        first = TwoModelLearner(self.classify, self.random_state).fit(
            matrix, values, treated.astype(int)
        )
        treated_gain = values[treated] - expected(
            first.control_, matrix[treated], self.classify
        )
        control_gain = (
            expected(first.treated_, matrix[control], self.classify)
            - values[control]
        )
        self.treated_effect_ = ensemble.HistGradientBoostingRegressor(
            random_state=self.random_state
        ).fit(matrix[treated], treated_gain)
        self.control_effect_ = ensemble.HistGradientBoostingRegressor(
            random_state=self.random_state
        ).fit(matrix[control], control_gain)
        self.share_ = float(treated.mean())
        return self

    def predict(self, features: Any) -> np.ndarray:
        """Blend of the two effect models, weighted by treatment share."""
        matrix = np.asarray(features)
        if self.degenerate_:
            return np.zeros(matrix.shape[0])
        return (1 - self.share_) * self.treated_effect_.predict(
            matrix
        ) + self.share_ * (self.control_effect_.predict(matrix))


class ZeroEffect(base.BaseEstimator):
    """The baseline: no row is expected to respond differently."""

    def fit(self, features: Any, outcome: Any, treatment: Any = None) -> Self:
        """Nothing to learn."""
        return self

    def predict(self, features: Any) -> np.ndarray:
        """The same (zero) effect for every row."""
        return np.zeros(features.shape[0])
