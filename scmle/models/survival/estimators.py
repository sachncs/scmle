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

import numpy
from scipy import optimize
from sklearn import base
from sklearn import ensemble

DEFAULT_BINS = 12
MAX_EXPANDED_ROWS = 300_000
MIN_BIN_EVENTS = 1


def kaplan_meier(
    duration: numpy.ndarray, event: numpy.ndarray
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Kaplan-Meier survival curve.

    Args:
        duration: Follow-up time of each row.
        event: 1 where the event was observed, 0 where censored.

    Returns:
        Distinct event times and the survival probability just after each.
    """
    time = numpy.asarray(duration, dtype=float)
    seen = numpy.asarray(event, dtype=bool)
    moments = numpy.unique(time[seen])
    survival = numpy.empty(len(moments))
    level = 1.0
    for position, moment in enumerate(moments):
        at_risk = (time >= moment).sum()
        happened = (seen & (time == moment)).sum()
        level *= 1.0 - happened / at_risk
        survival[position] = level
    return moments, survival


def step_lookup(
    moments: numpy.ndarray,
    values: numpy.ndarray,
    times: numpy.ndarray,
    before: float,
) -> numpy.ndarray:
    """Value of a right-continuous step function at each time.

    Args:
        moments: Sorted jump times of the function.
        values: Function value just after each jump time.
        times: Times at which to evaluate the function.
        before: Value before the first jump time.

    Returns:
        Array with one value per entry of `times`.
    """
    position = numpy.searchsorted(
        moments, numpy.asarray(times, dtype=float), "right"
    )
    padded = numpy.concatenate([[before], values])
    return padded[position]


class SurvivalEstimator(base.RegressorMixin, base.BaseEstimator, abc.ABC):
    """Base class: `predict` is a risk score, `predict_survival` a curve."""

    @abc.abstractmethod
    def fit(self, features: Any, duration: Any, event: Any = None) -> Self:
        """Fits on durations and event indicators.

        Args:
            features: Covariate matrix.
            duration: Follow-up time of each row.
            event: 1 where the event was observed, 0 where censored; None means
                every row had the event.

        Returns:
            This fitted estimator.
        """

    @abc.abstractmethod
    def predict(self, features: Any) -> numpy.ndarray:
        """Risk score per row; larger means the event is expected sooner.

        Args:
            features: Covariate matrix.

        Returns:
            Array of shape `(n_rows,)` of risk scores.
        """

    @abc.abstractmethod
    def predict_survival(self, features: Any, times: Any) -> numpy.ndarray:
        """Probability of no event yet, shape `(rows, len(times))`.

        Args:
            features: Covariate matrix.
            times: Times at which to evaluate survival.

        Returns:
            Array of shape `(n_rows, len(times))` of survival probabilities.
        """


class MarginalSurvival(SurvivalEstimator):
    """Ignores the features and predicts the population curve.

    Attributes:
        moments_: Event times of the Kaplan-Meier curve.
        curve_: Survival just after each event time.
    """

    def fit(self, features: Any, duration: Any, event: Any = None) -> Self:
        """Fits the Kaplan-Meier curve.

        Args:
            features: Covariate matrix.
            duration: Follow-up time of each row.
            event: 1 where the event was observed, 0 where censored; None means
                every row had the event.

        Returns:
            This fitted estimator.
        """
        seen = (
            numpy.ones(len(duration)) if event is None else numpy.asarray(event)
        )
        self.moments_, self.curve_ = kaplan_meier(numpy.asarray(duration), seen)
        return self

    def predict(self, features: Any) -> numpy.ndarray:
        """Same risk for every row, so the ordering is no better than chance.

        Args:
            features: Matrix whose row count sets the output length.

        Returns:
            Array of zeros of shape `(n_rows,)`.
        """
        return numpy.zeros(features.shape[0])

    def predict_survival(self, features: Any, times: Any) -> numpy.ndarray:
        """The population curve repeated for each row.

        Args:
            features: Covariate matrix.
            times: Times at which to evaluate survival.

        Returns:
            Array of shape `(n_rows, len(times))` of survival probabilities.
        """
        curve = step_lookup(
            self.moments_, self.curve_, numpy.asarray(times), 1.0
        )
        return numpy.tile(curve, (features.shape[0], 1))


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
        """Maximises the penalised partial likelihood.

        Args:
            features: Covariate matrix.
            duration: Follow-up time of each row.
            event: 1 where the event was observed, 0 where censored; None means
                every row had the event.

        Returns:
            This fitted estimator.
        """
        matrix = numpy.asarray(features, dtype=float)
        time = numpy.asarray(duration, dtype=float)
        seen = (
            numpy.ones(len(time), dtype=bool)
            if event is None
            else numpy.asarray(event, dtype=bool)
        )
        order = numpy.argsort(time, kind="stable")
        matrix, time, seen = matrix[order], time[order], seen[order]
        first = numpy.searchsorted(time, time, side="left")

        def objective(beta: numpy.ndarray) -> tuple[float, numpy.ndarray]:
            """Penalised negative log partial likelihood and its gradient.

            Args:
                beta: Coefficient vector.

            Returns:
                The objective value and its gradient with respect to `beta`.
            """
            eta = matrix @ beta
            shift = eta.max()
            weight = numpy.exp(eta - shift)
            tail_weight = numpy.cumsum(weight[::-1])[::-1][first]
            tail_matrix = numpy.cumsum(
                (matrix * weight[:, None])[::-1], axis=0
            )[::-1][first]
            log_tail = numpy.log(tail_weight) + shift
            value = -(eta[seen] - log_tail[seen]).sum()
            mean = tail_matrix[seen] / tail_weight[seen, None]
            gradient = -(matrix[seen] - mean).sum(axis=0)
            return (
                value + 0.5 * self.penalty * float(beta @ beta),
                gradient + self.penalty * beta,
            )

        fitted = optimize.minimize(
            objective,
            numpy.zeros(matrix.shape[1]),
            jac=True,
            method="L-BFGS-B",
        )
        self.coefficients_ = fitted.x
        eta = matrix @ self.coefficients_
        weight = numpy.exp(eta - eta.max())
        tail_weight = numpy.cumsum(weight[::-1])[::-1][first]
        self.moments_ = numpy.unique(time[seen])
        increments = []
        for moment in self.moments_:
            happened = (seen & (time == moment)).sum()
            at_risk = tail_weight[numpy.searchsorted(time, moment, "left")]
            increments.append(happened / at_risk)
        self.scale_ = float(eta.max())
        self.baseline_ = numpy.cumsum(increments)
        return self

    def predict(self, features: Any) -> numpy.ndarray:
        """Linear predictor (log relative hazard).

        Args:
            features: Covariate matrix.

        Returns:
            Array of shape `(n_rows,)` of linear predictors.
        """
        return numpy.asarray(features, dtype=float) @ self.coefficients_

    def predict_survival(self, features: Any, times: Any) -> numpy.ndarray:
        """Survival `exp(-B(t) * exp(eta))` for each row and time.

        Here `B` is the cumulative baseline hazard (Breslow, a step function
        that is 0 before the first event time) and `eta` the linear predictor.
        Internally the stored baseline is multiplied by `exp(scale_)` and
        `exp(eta)` divided by it (`scale_` is the largest training `eta`), so
        the product is unchanged.

        Args:
            features: Covariate matrix.
            times: Times at which to evaluate survival.

        Returns:
            Array of shape `(n_rows, len(times))` of survival probabilities.
        """
        cumulative = step_lookup(
            self.moments_, self.baseline_, numpy.asarray(times), 0.0
        )
        relative = numpy.exp(self.predict(features) - self.scale_)
        return numpy.exp(-numpy.outer(relative, cumulative))


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
        """Initializes the model.

        Args:
            bins: Number of time intervals.
            learning_rate: Boosting learning rate.
            max_leaf_nodes: Largest tree.
            random_state: Seed.
        """
        self.bins = bins
        self.learning_rate = learning_rate
        self.max_leaf_nodes = max_leaf_nodes
        self.random_state = random_state

    def interval_of(self, time: numpy.ndarray) -> numpy.ndarray:
        """Index of the interval each time falls in.

        Args:
            time: Times to place into intervals.

        Returns:
            Integer array of interval indices, capped at the last interval.
        """
        return numpy.minimum(
            numpy.searchsorted(self.edges_, time, side="left"),
            len(self.edges_) - 1,
        )

    def expand(
        self, matrix: numpy.ndarray, last: numpy.ndarray, hit: numpy.ndarray
    ) -> tuple[numpy.ndarray, numpy.ndarray]:
        """One record per row and interval the row was at risk in.

        Args:
            matrix: Covariate matrix.
            last: Index of the last interval each row was observed in.
            hit: Whether each row had the event in its last interval.

        Returns:
            The expanded covariates with the interval index as the final column,
            and the 0/1 label, which is 1 only in the interval of an event.
        """
        lengths = last + 1
        rows = numpy.repeat(numpy.arange(len(matrix)), lengths)
        offsets = numpy.arange(lengths.sum()) - numpy.repeat(
            numpy.cumsum(lengths) - lengths, lengths
        )
        labels = (hit[rows] & (offsets == last[rows])).astype(int)
        return numpy.column_stack([matrix[rows], offsets]), labels

    def fit(self, features: Any, duration: Any, event: Any = None) -> Self:
        """Builds the person-interval data and fits the hazard model.

        When the expanded data would exceed the row cap, a random subset of
        rows is used.

        Args:
            features: Covariate matrix.
            duration: Follow-up time of each row.
            event: 1 where the event was observed, 0 where censored; None means
                every row had the event.

        Returns:
            This fitted estimator.
        """
        matrix = numpy.asarray(features, dtype=float)
        time = numpy.asarray(duration, dtype=float)
        seen = (
            numpy.ones(len(time), dtype=bool)
            if event is None
            else numpy.asarray(event, dtype=bool)
        )
        cuts = numpy.unique(
            numpy.quantile(
                time[seen] if seen.any() else time,
                numpy.linspace(0, 1, self.bins + 1)[1:],
            )
        )
        self.edges_ = cuts
        last = self.interval_of(time)
        average = float((last + 1).mean())
        if len(matrix) * average > MAX_EXPANDED_ROWS:
            keep = numpy.random.default_rng(self.random_state).choice(
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

    def hazards(self, features: Any) -> numpy.ndarray:
        """Hazard in each interval, shape `(rows, intervals)`.

        Args:
            features: Covariate matrix.

        Returns:
            Array of shape `(n_rows, n_intervals)` of hazards; zeros when
            training saw no event.
        """
        matrix = numpy.asarray(features, dtype=float)
        count = len(self.edges_)
        stacked = numpy.column_stack(
            [
                numpy.repeat(matrix, count, axis=0),
                numpy.tile(numpy.arange(count), len(matrix)),
            ]
        )
        if 1 not in self.model_.classes_:
            return numpy.zeros((len(matrix), count))
        return self.model_.predict_proba(stacked)[:, 1].reshape(
            len(matrix), count
        )

    def predict(self, features: Any) -> numpy.ndarray:
        """Cumulative hazard over all intervals as the risk score.

        Args:
            features: Covariate matrix.

        Returns:
            Array of shape `(n_rows,)` of summed hazards.
        """
        return self.hazards(features).sum(axis=1)

    def predict_survival(self, features: Any, times: Any) -> numpy.ndarray:
        """Running product of one minus the hazard, read at each time.

        Args:
            features: Covariate matrix.
            times: Times at which to evaluate survival.

        Returns:
            Array of shape `(n_rows, len(times))` of survival probabilities.
        """
        curve = numpy.cumprod(1.0 - self.hazards(features), axis=1)
        interval = numpy.searchsorted(
            self.edges_, numpy.asarray(times, dtype=float), side="right"
        )
        padded = numpy.column_stack([numpy.ones(len(curve)), curve])
        return padded[:, interval]
