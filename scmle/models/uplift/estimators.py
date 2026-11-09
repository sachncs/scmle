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

import numpy
from sklearn import base
from sklearn import ensemble

MIN_ARM_ROWS = 20


def outcome_model(classify: bool, seed: int) -> base.BaseEstimator:
    """A fresh outcome model: boosted trees for a label or a number.

    Args:
        classify: True for a 0/1 label outcome, False for a numeric one.
        seed: Random seed of the model.

    Returns:
        An unfitted histogram gradient-boosting classifier or regressor.
    """
    if classify:
        return ensemble.HistGradientBoostingClassifier(random_state=seed)
    return ensemble.HistGradientBoostingRegressor(random_state=seed)


def expected(model: Any, features: Any, classify: bool) -> numpy.ndarray:
    """Expected outcome: probability of the positive label, or the value.

    Args:
        model: Fitted outcome model.
        features: Matrix to predict for.
        classify: Whether the outcome is a 0/1 label.

    Returns:
        Array of shape `(n_rows,)`: the probability of label 1 for a
        classifier (a constant 1.0 or 0.0 when the model saw one class only),
        or the predicted value for a regressor.
    """
    if not classify:
        return numpy.asarray(model.predict(features), dtype=float)
    probabilities = numpy.asarray(model.predict_proba(features))
    if probabilities.shape[1] == 1:
        return numpy.full(probabilities.shape[0], float(model.classes_[0] == 1))
    return probabilities[:, list(model.classes_).index(1)]


def split_arms(
    treatment: Any,
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Boolean masks of the treated and control rows.

    Args:
        treatment: Per-row treatment indicator, 1 for treated.

    Returns:
        The pair `(treated, control)` of boolean arrays, each of shape
        `(n_rows,)`; control is the complement of treated.
    """
    flag = numpy.asarray(treatment).astype(int) == 1
    return flag, ~flag


class TwoModelLearner(base.BaseEstimator):
    """T-learner: one outcome model per arm; uplift is their difference.

    Attributes:
        classify: Whether the outcome is a 0/1 label.
        random_state: Seed.
    """

    def __init__(self, classify: bool = True, random_state: int = 0) -> None:
        """Stores the settings (scikit-learn convention).

        Args:
            classify: Whether the outcome is a 0/1 label.
            random_state: Seed.
        """
        self.classify = classify
        self.random_state = random_state

    def fit(self, features: Any, outcome: Any, treatment: Any = None) -> Self:
        """Fits the treated and control outcome models.

        An arm with no rows is fitted on all rows instead.

        Args:
            features: Training matrix.
            outcome: Observed outcome per row.
            treatment: 1 for treated rows and 0 for control rows; None treats
                every row as control.

        Returns:
            This fitted learner.
        """
        treated, control = split_arms(
            numpy.zeros(len(outcome)) if treatment is None else treatment
        )
        values = numpy.asarray(outcome)
        self.treated_ = outcome_model(self.classify, self.random_state).fit(
            features[treated] if treated.any() else features,
            values[treated] if treated.any() else values,
        )
        self.control_ = outcome_model(self.classify, self.random_state).fit(
            features[control] if control.any() else features,
            values[control] if control.any() else values,
        )
        return self

    def predict(self, features: Any) -> numpy.ndarray:
        """Estimated effect of treating each row.

        Args:
            features: Matrix to predict for.

        Returns:
            Array of shape `(n_rows,)`: expected outcome under treatment minus
            expected outcome under control.
        """
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
        """Stores the settings.

        Args:
            classify: Whether the outcome is a 0/1 label.
            random_state: Seed.
        """
        self.classify = classify
        self.random_state = random_state

    @staticmethod
    def stacked(features: Any, flag: float) -> numpy.ndarray:
        """Features with a constant treatment column appended.

        Args:
            features: Feature matrix.
            flag: Value written in the appended treatment column.

        Returns:
            Array of shape `(n_rows, n_features + 1)`.
        """
        matrix = numpy.asarray(features)
        return numpy.column_stack([matrix, numpy.full(matrix.shape[0], flag)])

    def fit(self, features: Any, outcome: Any, treatment: Any = None) -> Self:
        """Fits one model on features plus the treatment indicator.

        Args:
            features: Training matrix.
            outcome: Observed outcome per row.
            treatment: 1 for treated rows and 0 for control rows; None treats
                every row as control.

        Returns:
            This fitted learner.
        """
        flag = numpy.zeros(len(outcome)) if treatment is None else treatment
        self.model_ = outcome_model(self.classify, self.random_state).fit(
            numpy.column_stack([numpy.asarray(features), numpy.asarray(flag)]),
            numpy.asarray(outcome),
        )
        return self

    def predict(self, features: Any) -> numpy.ndarray:
        """Prediction with treatment switched on minus switched off.

        Args:
            features: Matrix to predict for.

        Returns:
            Array of shape `(n_rows,)` of estimated effects.
        """
        return expected(
            self.model_, self.stacked(features, 1.0), self.classify
        ) - expected(self.model_, self.stacked(features, 0.0), self.classify)


class CrossLearner(base.BaseEstimator):
    """X-learner: imputes each row's missing outcome, then models the gap.

    Outcome models per arm predict what each row would have had under the
    other arm; the imputed individual effects are regressed on the features
    separately in each arm, and the two effect models are blended by the
    treated share of the training rows. It does well when the arms are
    unbalanced.

    Attributes:
        classify: Whether the outcome is a 0/1 label.
        random_state: Seed.
    """

    def __init__(self, classify: bool = True, random_state: int = 0) -> None:
        """Stores the settings.

        Args:
            classify: Whether the outcome is a 0/1 label.
            random_state: Seed.
        """
        self.classify = classify
        self.random_state = random_state

    def fit(self, features: Any, outcome: Any, treatment: Any = None) -> Self:
        """Fits the per-arm outcome models, then one effect model per arm.

        The treated share of the training rows is kept as the blending weight
        (no propensity model is fitted). If only one arm is present no effect
        can be estimated and `predict` returns zeros.

        Args:
            features: Training matrix.
            outcome: Observed outcome per row.
            treatment: 1 for treated rows and 0 for control rows; None treats
                every row as control.

        Returns:
            This fitted learner.
        """
        matrix = numpy.asarray(features)
        values = numpy.asarray(outcome, dtype=float)
        treated, control = split_arms(
            numpy.zeros(len(values)) if treatment is None else treatment
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

    def predict(self, features: Any) -> numpy.ndarray:
        """Blends the two effect models by the treated share.

        The treated-arm effect model is weighted by one minus the share and the
        control-arm effect model by the share.

        Args:
            features: Matrix to predict for.

        Returns:
            Array of shape `(n_rows,)` of estimated effects; all zeros when fit
            saw only one arm.
        """
        matrix = numpy.asarray(features)
        if self.degenerate_:
            return numpy.zeros(matrix.shape[0])
        return (1 - self.share_) * self.treated_effect_.predict(
            matrix
        ) + self.share_ * (self.control_effect_.predict(matrix))


class ZeroEffect(base.BaseEstimator):
    """The baseline: no row is expected to respond differently."""

    def fit(self, features: Any, outcome: Any, treatment: Any = None) -> Self:  # noqa: ARG002 -- fit signature of the uplift estimators
        """Nothing to learn.

        Args:
            features: Training matrix.
            outcome: Observed outcome per row.
            treatment: 1 for treated rows and 0 for control rows; None treats
                every row as control.

        Returns:
            This estimator, unchanged.
        """
        return self

    def predict(self, features: Any) -> numpy.ndarray:
        """The same (zero) effect for every row.

        Args:
            features: Matrix to predict for.

        Returns:
            Array of zeros of shape `(n_rows,)`.
        """
        return numpy.zeros(features.shape[0])
