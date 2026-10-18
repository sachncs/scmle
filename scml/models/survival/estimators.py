"""Survival estimators: they learn from a duration and an event indicator.

Every estimator here follows one contract: `fit(features, duration, event=...)`
where `event` is 1 when the event was observed and 0 when the row was censored
(the follow-up ended first), `predict` returns a risk score (larger means the
event is expected sooner) and `predict_survival` returns the probability of
still being event-free at each requested time.
"""

import abc
from typing import Any
from typing import Self

import numpy as np
from scipy import optimize
from sklearn import base
from sklearn import ensemble

DEFAULT_BINS = 12
MAX_EXPANDED_ROWS = 300_000
MIN_BIN_EVENTS = 1


def kaplan_meier(
    duration: np.ndarray, event: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Kaplan-Meier survival curve.

    Returns:
        Distinct event times and the survival probability just after each.
    """
    time = np.asarray(duration, dtype=float)
    seen = np.asarray(event, dtype=bool)
    moments = np.unique(time[seen])
    survival = np.empty(len(moments))
    level = 1.0
    for position, moment in enumerate(moments):
        at_risk = (time >= moment).sum()
        happened = (seen & (time == moment)).sum()
        level *= 1.0 - happened / at_risk
        survival[position] = level
    return moments, survival


def step_lookup(
    moments: np.ndarray, values: np.ndarray, times: np.ndarray, before: float
) -> np.ndarray:
    """Value of a right-continuous step function at each time."""
    position = np.searchsorted(moments, np.asarray(times, dtype=float), "right")
    padded = np.concatenate([[before], values])
    return padded[position]


class SurvivalEstimator(base.RegressorMixin, base.BaseEstimator, abc.ABC):
    """Base class: `predict` is a risk score, `predict_survival` a curve."""

    @abc.abstractmethod
    def fit(self, features: Any, duration: Any, event: Any = None) -> Self:
        """Fits on durations and event indicators."""

    @abc.abstractmethod
    def predict(self, features: Any) -> np.ndarray:
        """Risk score per row; larger means the event is expected sooner."""

    @abc.abstractmethod
    def predict_survival(self, features: Any, times: Any) -> np.ndarray:
        """Probability of no event yet, shape `(rows, len(times))`."""


class MarginalSurvival(SurvivalEstimator):
    """Ignores the features and predicts the population curve.

    Attributes:
        moments_: Event times of the Kaplan-Meier curve.
        curve_: Survival just after each event time.
    """

    def fit(self, features: Any, duration: Any, event: Any = None) -> Self:
        """Fits the Kaplan-Meier curve."""
        seen = np.ones(len(duration)) if event is None else np.asarray(event)
        self.moments_, self.curve_ = kaplan_meier(np.asarray(duration), seen)
        return self

    def predict(self, features: Any) -> np.ndarray:
        """Same risk for every row, so the ordering is no better than chance."""
        return np.zeros(features.shape[0])

    def predict_survival(self, features: Any, times: Any) -> np.ndarray:
        """The population curve repeated for each row."""
        curve = step_lookup(self.moments_, self.curve_, np.asarray(times), 1.0)
        return np.tile(curve, (features.shape[0], 1))


class CoxRegressor(SurvivalEstimator):
    """Cox proportional hazards with a ridge penalty.

    The log hazard is linear in the features. Coefficients maximise the
    penalised partial likelihood (ties handled by the Breslow convention);
    the baseline hazard follows from the Breslow estimator.

    Attributes:
        penalty: Ridge strength on the coefficients.
        coefficients_: Fitted coefficients.
        moments_: Distinct event times.
        baseline_: Cumulative baseline hazard at each event time.
    """

    def __init__(self, penalty: float = 1.0) -> None:
        """Initializes the model.

        Args:
            penalty: Ridge strength; larger shrinks coefficients harder.
        """
        self.penalty = penalty

    def fit(self, features: Any, duration: Any, event: Any = None) -> Self:
        """Maximises the penalised partial likelihood."""
        matrix = np.asarray(features, dtype=float)
        time = np.asarray(duration, dtype=float)
        seen = (
            np.ones(len(time), dtype=bool)
            if event is None
            else np.asarray(event, dtype=bool)
        )
        order = np.argsort(time, kind="stable")
        matrix, time, seen = matrix[order], time[order], seen[order]
        first = np.searchsorted(time, time, side="left")

        def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
            eta = matrix @ beta
            shift = eta.max()
            weight = np.exp(eta - shift)
            tail_weight = np.cumsum(weight[::-1])[::-1][first]
            tail_matrix = np.cumsum((matrix * weight[:, None])[::-1], axis=0)[
                ::-1
            ][first]
            log_tail = np.log(tail_weight) + shift
            value = -(eta[seen] - log_tail[seen]).sum()
            mean = tail_matrix[seen] / tail_weight[seen, None]
            gradient = -(matrix[seen] - mean).sum(axis=0)
            return (
                value + 0.5 * self.penalty * float(beta @ beta),
                gradient + self.penalty * beta,
            )

        fitted = optimize.minimize(
            objective,
            np.zeros(matrix.shape[1]),
            jac=True,
            method="L-BFGS-B",
        )
        self.coefficients_ = fitted.x
        eta = matrix @ self.coefficients_
        weight = np.exp(eta - eta.max())
        tail_weight = np.cumsum(weight[::-1])[::-1][first]
        self.moments_ = np.unique(time[seen])
        increments = []
        for moment in self.moments_:
            happened = (seen & (time == moment)).sum()
            at_risk = tail_weight[np.searchsorted(time, moment, "left")]
            increments.append(happened / at_risk)
        self.scale_ = float(eta.max())
        self.baseline_ = np.cumsum(increments)
        return self

    def predict(self, features: Any) -> np.ndarray:
        """Linear predictor (log relative hazard)."""
        return np.asarray(features, dtype=float) @ self.coefficients_

    def predict_survival(self, features: Any, times: Any) -> np.ndarray:
        """`exp(-H0(t) * exp(eta))` for each row and time."""
        cumulative = step_lookup(
            self.moments_, self.baseline_, np.asarray(times), 0.0
        )
        relative = np.exp(self.predict(features) - self.scale_)
        return np.exp(-np.outer(relative, cumulative))


class DiscreteTimeBoosting(SurvivalEstimator):
    """Boosted trees on a discrete-time hazard.

    Time is cut into intervals at quantiles of the observed durations. Each
    training row is expanded into one record per interval it was still
    at risk in, labelled 1 only in the interval where its event happened. A
    boosted classifier learns the hazard in an interval given the features and
    the interval, which captures non-linear effects and interactions that the
    Cox model cannot. Survival is the running product of one minus the hazard.

    Attributes:
        bins: Number of time intervals.
        learning_rate: Boosting learning rate.
        max_leaf_nodes: Largest tree.
        random_state: Seed.
        edges_: Upper edge of each interval.
        model_: Fitted hazard classifier.
    """

    def __init__(
        self,
        bins: int = DEFAULT_BINS,
        learning_rate: float = 0.1,
        max_leaf_nodes: int = 15,
        random_state: int = 0,
    ) -> None:
        """Initializes the model."""
        self.bins = bins
        self.learning_rate = learning_rate
        self.max_leaf_nodes = max_leaf_nodes
        self.random_state = random_state

    def interval_of(self, time: np.ndarray) -> np.ndarray:
        """Index of the interval each time falls in."""
        return np.minimum(
            np.searchsorted(self.edges_, time, side="left"),
            len(self.edges_) - 1,
        )

    def expand(
        self, matrix: np.ndarray, last: np.ndarray, hit: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """One record per row and interval the row was at risk in."""
        lengths = last + 1
        rows = np.repeat(np.arange(len(matrix)), lengths)
        offsets = np.arange(lengths.sum()) - np.repeat(
            np.cumsum(lengths) - lengths, lengths
        )
        labels = (hit[rows] & (offsets == last[rows])).astype(int)
        return np.column_stack([matrix[rows], offsets]), labels

    def fit(self, features: Any, duration: Any, event: Any = None) -> Self:
        """Builds the person-interval data and fits the hazard model."""
        matrix = np.asarray(features, dtype=float)
        time = np.asarray(duration, dtype=float)
        seen = (
            np.ones(len(time), dtype=bool)
            if event is None
            else np.asarray(event, dtype=bool)
        )
        cuts = np.unique(
            np.quantile(
                time[seen] if seen.any() else time,
                np.linspace(0, 1, self.bins + 1)[1:],
            )
        )
        self.edges_ = cuts
        last = self.interval_of(time)
        average = float((last + 1).mean())
        if len(matrix) * average > MAX_EXPANDED_ROWS:
            keep = np.random.default_rng(self.random_state).choice(
                len(matrix),
                max(int(MAX_EXPANDED_ROWS / average), 1),
                replace=False,
            )
            matrix, last, seen = matrix[keep], last[keep], seen[keep]
        expanded, labels = self.expand(matrix, last, seen)
        self.model_ = ensemble.HistGradientBoostingClassifier(
            learning_rate=self.learning_rate,
            max_leaf_nodes=self.max_leaf_nodes,
            random_state=self.random_state,
        ).fit(expanded, labels)
        return self

    def hazards(self, features: Any) -> np.ndarray:
        """Hazard in each interval, shape `(rows, intervals)`."""
        matrix = np.asarray(features, dtype=float)
        count = len(self.edges_)
        stacked = np.column_stack(
            [
                np.repeat(matrix, count, axis=0),
                np.tile(np.arange(count), len(matrix)),
            ]
        )
        if 1 not in self.model_.classes_:
            return np.zeros((len(matrix), count))
        return self.model_.predict_proba(stacked)[:, 1].reshape(
            len(matrix), count
        )

    def predict(self, features: Any) -> np.ndarray:
        """Cumulative hazard over all intervals as the risk score."""
        return self.hazards(features).sum(axis=1)

    def predict_survival(self, features: Any, times: Any) -> np.ndarray:
        """Running product of one minus the hazard, read at each time."""
        curve = np.cumprod(1.0 - self.hazards(features), axis=1)
        interval = np.searchsorted(
            self.edges_, np.asarray(times, dtype=float), side="right"
        )
        padded = np.column_stack([np.ones(len(curve)), curve])
        return padded[:, interval]
