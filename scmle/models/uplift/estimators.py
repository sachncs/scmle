"""Uplift estimators: how much a treatment changes the outcome, per row.

Every estimator follows one contract: `fit(features, outcome, treatment)`
where `treatment` is 1 for treated rows and 0 for control rows, and
`predict` returns the estimated change in the outcome if the row were
treated (the conditional average treatment effect). The estimates assume
that, given the features, treatment was assigned independently of the
outcome (no unmeasured confounding), which a randomized experiment
guarantees and observational data do not.
"""

from collections.abc import Callable
from collections.abc import Mapping
from typing import Any
from typing import Self

import numpy
from sklearn import base
from sklearn import dummy
from sklearn import ensemble
from sklearn import linear_model

from scmle.contracts import errors

MIN_ARM_ROWS = 20


def booster(
    classify: bool, seed: int
) -> (
    ensemble.HistGradientBoostingClassifier
    | ensemble.HistGradientBoostingRegressor
):
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


def linear(
    classify: bool, seed: int
) -> linear_model.LogisticRegression | linear_model.Ridge:
    """A fresh linear outcome model: logistic for a label, ridge otherwise.

    Args:
        classify: True for a 0/1 label outcome, False for a numeric one.
        seed: Random seed of the logistic solver.

    Returns:
        An unfitted logistic or ridge regression; neither accepts missing
        values, so the features must be imputed and scaled.
    """
    if classify:
        return linear_model.LogisticRegression(max_iter=1000, random_state=seed)
    return linear_model.Ridge()


KINDS: Mapping[str, Callable[[bool, int], Any]] = {
    "boosting": booster,
    "linear": linear,
}
"""Outcome-model families a learner can be built on, by name."""


def model(kind: str, classify: bool, seed: int) -> Any:
    """A fresh outcome model of the named family.

    Args:
        kind: A key of `KINDS`.
        classify: True for a 0/1 label outcome, False for a numeric one.
        seed: Random seed of the model.

    Returns:
        The unfitted model.

    Raises:
        ConfigError: If `kind` names no family.
    """
    if kind not in KINDS:
        raise errors.ConfigError(
            f"unknown outcome model kind {kind!r}; choose from {sorted(KINDS)}"
        )
    return KINDS[kind](classify, seed)


def train(
    kind: str, classify: bool, seed: int, features: Any, values: Any
) -> Any:
    """Fits an outcome model, or a constant when a label has one class.

    Args:
        kind: A key of `KINDS`.
        classify: Whether the outcome is a 0/1 label.
        seed: Random seed of the model.
        features: Training matrix.
        values: Outcome of each row.

    Returns:
        The fitted model; a prior classifier when the labels hold one class,
        which a logistic regression cannot fit.

    Raises:
        ConfigError: If `kind` names no family.
    """
    fresh = model(kind, classify, seed)
    if classify and len(numpy.unique(numpy.asarray(values))) < 2:  # noqa: PLR2004 -- two classes make a label
        fresh = dummy.DummyClassifier(strategy="prior")
    return fresh.fit(features, values)


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


def split(
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


class TLearner(base.BaseEstimator):  # type: ignore[misc]  # scikit-learn ships no type information
    """T-learner: one outcome model per arm; uplift is their difference.

    Attributes:
        classify: Whether the outcome is a 0/1 label.
        random_state: Seed.
        kind: Outcome-model family, a key of `KINDS`.
    """

    def __init__(
        self,
        classify: bool = True,
        random_state: int = 0,
        kind: str = "boosting",
    ) -> None:
        """Stores the settings (scikit-learn convention).

        Args:
            classify: Whether the outcome is a 0/1 label.
            random_state: Seed.
            kind: Outcome-model family, a key of `KINDS`.
        """
        self.classify = classify
        self.random_state = random_state
        self.kind = kind

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
        treated, control = split(
            numpy.zeros(len(outcome)) if treatment is None else treatment
        )
        values = numpy.asarray(outcome)
        self.treated_ = train(
            self.kind,
            self.classify,
            self.random_state,
            features[treated] if treated.any() else features,
            values[treated] if treated.any() else values,
        )
        self.control_ = train(
            self.kind,
            self.classify,
            self.random_state,
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
        effect: numpy.ndarray = expected(
            self.treated_, features, self.classify
        ) - expected(self.control_, features, self.classify)
        return effect


class SingleModelLearner(base.BaseEstimator):  # type: ignore[misc]  # scikit-learn ships no type information
    """S-learner: one outcome model with the treatment as a feature.

    With `kind="linear"` the treatment enters as one additive coefficient,
    so the effect is homogeneous: the same for every row on the link scale
    (and, for a ridge outcome, on the outcome scale).

    Attributes:
        classify: Whether the outcome is a 0/1 label.
        random_state: Seed.
        kind: Outcome-model family, a key of `KINDS`.
    """

    def __init__(
        self,
        classify: bool = True,
        random_state: int = 0,
        kind: str = "boosting",
    ) -> None:
        """Stores the settings.

        Args:
            classify: Whether the outcome is a 0/1 label.
            random_state: Seed.
            kind: Outcome-model family, a key of `KINDS`.
        """
        self.classify = classify
        self.random_state = random_state
        self.kind = kind

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
        self.model_ = train(
            self.kind,
            self.classify,
            self.random_state,
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
        effect: numpy.ndarray = expected(
            self.model_, self.stacked(features, 1.0), self.classify
        ) - expected(self.model_, self.stacked(features, 0.0), self.classify)
        return effect


class CrossLearner(base.BaseEstimator):  # type: ignore[misc]  # scikit-learn ships no type information
    """X-learner: imputes each row's missing outcome, then models the gap.

    Outcome models per arm predict what each row would have had under the
    other arm; the imputed individual effects are regressed on the features
    separately in each arm, and the two effect models are blended by the
    treated share of the training rows. It does well when the arms are
    unbalanced.

    Attributes:
        classify: Whether the outcome is a 0/1 label.
        random_state: Seed.
        kind: Outcome-model family, a key of `KINDS`.
    """

    def __init__(
        self,
        classify: bool = True,
        random_state: int = 0,
        kind: str = "boosting",
    ) -> None:
        """Stores the settings.

        Args:
            classify: Whether the outcome is a 0/1 label.
            random_state: Seed.
            kind: Outcome-model family, a key of `KINDS`.
        """
        self.classify = classify
        self.random_state = random_state
        self.kind = kind

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
        treated, control = split(
            numpy.zeros(len(values)) if treatment is None else treatment
        )
        self.degenerate_ = not (treated.any() and control.any())
        if self.degenerate_:
            return self  # One arm only: no effect can be estimated.
        first = TLearner(self.classify, self.random_state, self.kind).fit(
            matrix, values, treated.astype(int)
        )
        treated_gain = values[treated] - expected(
            first.control_, matrix[treated], self.classify
        )
        control_gain = (
            expected(first.treated_, matrix[control], self.classify)
            - values[control]
        )
        self.treated_effect_ = model(
            self.kind, classify=False, seed=self.random_state
        ).fit(matrix[treated], treated_gain)
        self.control_effect_ = model(
            self.kind, classify=False, seed=self.random_state
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
        effect: numpy.ndarray = (
            1 - self.share_
        ) * self.treated_effect_.predict(matrix) + self.share_ * (
            self.control_effect_.predict(matrix)
        )
        return effect


class Inert(base.BaseEstimator):  # type: ignore[misc]  # scikit-learn ships no type information
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
