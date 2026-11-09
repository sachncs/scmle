"""Concordance index for right-censored durations."""

import dataclasses
from typing import Any

import numpy
import pandas

from scmle.contracts import metrics

MAX_CONCORDANCE_ROWS = 4000


def concordance_index(
    duration: numpy.ndarray, event: numpy.ndarray, risk: numpy.ndarray
) -> float:
    """Harrell's concordance index.

    Among comparable pairs (the earlier row had the event and the other row
    has a strictly later time), the share where the earlier row got the
    higher risk; tied risks count half. 0.5 is chance
    and 1.0 is a perfect ordering.

    Args:
        duration: Observed times.
        event: 1 where the event was observed.
        risk: Predicted risk scores.

    Returns:
        The index, or NaN when no pair is comparable. Large inputs are
        evenly thinned to `MAX_CONCORDANCE_ROWS` rows.
    """
    time = numpy.asarray(duration, dtype=float)
    seen = numpy.asarray(event, dtype=bool)
    score = numpy.asarray(risk, dtype=float)
    if len(time) > MAX_CONCORDANCE_ROWS:
        keep = numpy.linspace(0, len(time) - 1, MAX_CONCORDANCE_ROWS).astype(
            int
        )
        time, seen, score = time[keep], seen[keep], score[keep]
    concordant = 0.0
    comparable = 0.0
    for index in numpy.flatnonzero(seen):
        later = time > time[index]
        count = int(later.sum())
        if count == 0:
            continue
        comparable += count
        concordant += float((score[index] > score[later]).sum())
        concordant += 0.5 * float((score[index] == score[later]).sum())
    return concordant / comparable if comparable else float("nan")


@dataclasses.dataclass(frozen=True, slots=True)
class ConcordanceMetric(metrics.Metric):
    """Harrell's concordance index as a catalog metric.

    The scorer reads the event indicator from the frame the estimator was
    given (`estimator.event_column`), so it needs no extra plumbing.
    """

    def accepts_weights(self) -> bool:
        """Reports whether the metric uses sample weights.

        Returns:
            Always False; the concordance index here is unweighted.
        """
        return False

    def scorer(self, positive: Any = None) -> Any:
        """Builds a scorer `(estimator, features, durations)`.

        Args:
            positive: Unused; present for the shared metric interface.

        Returns:
            A callable that scores a fitted estimator on features and observed
            durations, reading events from `estimator.event_column`. Its
            optional `sample_weight` argument is ignored.
        """

        def score(
            estimator: Any,
            features: pandas.DataFrame,
            target: pandas.Series,
            sample_weight: Any = None,
        ) -> float:
            return concordance_index(
                numpy.asarray(target, dtype=float),
                features[estimator.event_column].to_numpy(dtype=float),
                numpy.asarray(estimator.predict(features), dtype=float),
            )

        return score
