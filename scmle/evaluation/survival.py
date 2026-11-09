"""Time-dependent survival metrics weighted for censoring.

Rows that were censored early tell us less about who survives past a given
time than rows that were followed longer. Each observed event is therefore
weighted by one over the probability of still being uncensored at its time,
which keeps the scores unbiased under random censoring.
"""

import dataclasses
from typing import Any

import numpy
import pandas

from scmle.contracts import metrics
from scmle.models.survival import estimators

GRID_POINTS = 10
MIN_TIMES = 2
LOWEST_QUANTILE = 0.1
HIGHEST_QUANTILE = 0.9
MIN_CENSORING_PROBABILITY = 0.05


def censoring_curve(
    duration: numpy.ndarray, event: numpy.ndarray
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Computes the Kaplan-Meier curve of the censoring times.

    Censoring is treated as the event, so the curve is the probability of
    remaining uncensored.

    Args:
        duration: Observed times.
        event: 1 where the event was observed, 0 where censored.

    Returns:
        The curve as the distinct censoring times and the estimated
        probability of remaining uncensored just after each.
    """
    return estimators.kaplan_meier(duration, 1 - numpy.asarray(event))


def censoring_probability(
    curve: tuple[numpy.ndarray, numpy.ndarray], times: numpy.ndarray
) -> numpy.ndarray:
    """Returns the chance of still being uncensored at each time, floored.

    Args:
        curve: The censoring curve from `censoring_curve`.
        times: Times to look up.

    Returns:
        The step-function value at each time (1.0 before the first step),
        at least `MIN_CENSORING_PROBABILITY`.
    """
    moments, values = curve
    found = estimators.step_lookup(moments, values, times, 1.0)
    return numpy.maximum(found, MIN_CENSORING_PROBABILITY)


def time_grid(duration: numpy.ndarray, event: numpy.ndarray) -> numpy.ndarray:
    """Returns evaluation times between the 10th and 90th event-time percentile.

    Args:
        duration: Observed times.
        event: 1 where the event was observed.

    Returns:
        The distinct values at `GRID_POINTS` evenly spaced quantiles of the
        event times (of all durations if no event was observed).
    """
    observed = duration[numpy.asarray(event, dtype=bool)]
    source = observed if observed.size else duration
    return numpy.unique(
        numpy.quantile(
            source,
            numpy.linspace(LOWEST_QUANTILE, HIGHEST_QUANTILE, GRID_POINTS),
        )
    )


def brier_scores(
    duration: numpy.ndarray,
    event: numpy.ndarray,
    survival: numpy.ndarray,
    times: numpy.ndarray,
) -> numpy.ndarray:
    """Computes the censoring-weighted squared error of survival at each time.

    Args:
        duration: Observed times.
        event: 1 where the event was observed.
        survival: Predicted probability of no event, `(rows, len(times))`.
        times: The times the columns of `survival` refer to.

    Returns:
        The Brier score at each time (lower is better).
    """
    seen = numpy.asarray(event, dtype=bool)
    curve = censoring_curve(duration, event)
    at_row = censoring_probability(curve, duration)
    at_time = censoring_probability(curve, times)
    scores = numpy.empty(len(times))
    for index, moment in enumerate(times):
        predicted = survival[:, index]
        failed = (duration <= moment) & seen
        survived = duration > moment
        scores[index] = numpy.mean(
            failed * predicted**2 / at_row
            + survived * (1.0 - predicted) ** 2 / at_time[index]
        )
    return scores


def integrated_brier(
    duration: numpy.ndarray,
    event: numpy.ndarray,
    survival: numpy.ndarray,
    times: numpy.ndarray,
) -> float:
    """Averages the Brier score over the time grid (trapezoid rule).

    Args:
        duration: Observed times.
        event: 1 where the event was observed.
        survival: Predicted probability of no event, `(rows, len(times))`.
        times: The times the columns of `survival` refer to, ascending.

    Returns:
        The integral of the Brier score divided by the time span; the single
        score when there is one time.
    """
    scores = brier_scores(duration, event, survival, times)
    if len(times) < MIN_TIMES:
        return float(scores[0])
    area = numpy.sum((scores[1:] + scores[:-1]) / 2 * numpy.diff(times))
    return float(area / (times[-1] - times[0]))


def dynamic_auc(
    duration: numpy.ndarray,
    event: numpy.ndarray,
    risk: numpy.ndarray,
    times: numpy.ndarray,
) -> float:
    """Computes the mean cumulative/dynamic AUC over the time grid.

    At each time, rows whose event happened by then are cases (weighted by
    the inverse censoring probability) and rows still event-free are
    controls; the AUC is the weighted share of case-control pairs the risk
    score orders correctly, with ties counting one half.

    Args:
        duration: Observed times.
        event: 1 where the event was observed.
        risk: Risk score of each row; higher means earlier event.
        times: Times at which to evaluate.

    Returns:
        The mean AUC over the times that have both a case and a control, or
        NaN if there are none.
    """
    seen = numpy.asarray(event, dtype=bool)
    weights = 1.0 / censoring_probability(
        censoring_curve(duration, event), duration
    )
    values = []
    for moment in times:
        cases = (duration <= moment) & seen
        controls = duration > moment
        if not cases.any() or not controls.any():
            continue
        wins = (risk[cases][:, None] > risk[controls][None, :]) + 0.5 * (
            risk[cases][:, None] == risk[controls][None, :]
        )
        pair_weight = weights[cases][:, None]
        values.append(
            float(
                (wins * pair_weight).sum()
                / (pair_weight.sum() * controls.sum())
            )
        )
    return float(numpy.mean(values)) if values else float("nan")


@dataclasses.dataclass(frozen=True, slots=True)
class SurvivalMetric(metrics.Metric):
    """A time-dependent survival metric read off a survival estimator.

    Attributes:
        kind: `ibs` (integrated Brier score) or `td_auc` (dynamic AUC).
    """

    kind: str = "ibs"

    def accepts_weights(self) -> bool:
        """Returns whether row weights are accepted.

        These scores weight by censoring, not by row weights.

        Returns:
            Always False.
        """
        return False

    def scorer(self, positive: Any = None) -> Any:  # noqa: ARG002 -- metric scorer interface
        """Builds a scorer called as `(estimator, features, durations)`.

        Args:
            positive: Unused; survival has no positive class.

        Returns:
            A function returning the score of the estimator on the rows,
            where higher is better: the negated integrated Brier score for
            `ibs`, the mean dynamic AUC otherwise. The event indicator is
            read from the estimator's `event_column`; sample weights are
            ignored.
        """
        kind = self.kind

        def score(
            estimator: Any,
            features: pandas.DataFrame,
            target: pandas.Series,
            sample_weight: Any = None,  # noqa: ARG001 -- scorer protocol passes sample_weight
        ) -> float:
            duration = numpy.asarray(target, dtype=float)
            event = features[estimator.event_column].to_numpy(dtype=float)
            times = time_grid(duration, event)
            if kind == "ibs":
                survival = numpy.asarray(
                    estimator.predict_survival(features, times)
                )
                return -integrated_brier(duration, event, survival, times)
            risk = numpy.asarray(estimator.predict(features), dtype=float)
            return dynamic_auc(duration, event, risk, times)

        return score
