"""Calibration of ranking scores: their order and their relevance map.

A ranking score claims two things. Its order: within a query, a higher score
marks a more relevant row, which the mean NDCG gain over a random order of
the same rows tests, with a bootstrap interval over queries. Its scale: an
increasing logistic (Platt) map, learnt on other queries, turns it into a
calibrated probability of relevance, which the grouped chi-square test
checks. A map that decreases is refused, so a reversed scorer never passes
through a flipped Platt slope.
"""

import math
from typing import Any
from typing import Self

import numpy
from sklearn import linear_model
from sklearn import model_selection

from scmle.contracts import errors
from scmle.contracts import tasks
from scmle.evaluation.calibration import verdict
import scmle.evaluation.ranking

FOLDS = 5
"""Query folds of the cross-fitted Platt map: each map is learnt on 80% of
the queries, so its estimate is close to the full-data map."""


class Platt:
    """A logistic map from a score to a probability.

    Attributes:
        model: The fitted logistic regression, or None when one class only
            was seen.
        share: The positive share seen, the constant answer without a model.
        increasing: Whether the map rises with the score (fitted slope above
            zero); a constant or decreasing map is not.
    """

    def __init__(self) -> None:
        """Starts unfitted."""
        self.model: linear_model.LogisticRegression | None = None
        self.share = math.nan
        self.increasing = False

    def fit(self, scores: Any, relevant: Any) -> Self:
        """Learns the map.

        Args:
            scores: Score of each row.
            relevant: 1 where the row is relevant.

        Returns:
            This map.
        """
        points = verdict.vector(scores, "scores")
        marks = verdict.vector(relevant, "relevance", len(points))
        self.share = float(marks.mean())
        self.model = (
            linear_model.LogisticRegression(C=numpy.inf).fit(
                points[:, None], marks.astype(int)
            )
            if len(numpy.unique(marks)) > 1
            else None
        )
        self.increasing = bool(
            self.model is not None and self.model.coef_[0, 0] > 0
        )
        return self

    def predict(self, scores: Any) -> numpy.ndarray:
        """Probability of relevance of each score.

        Args:
            scores: Scores to map.

        Returns:
            The probabilities.
        """
        points = verdict.vector(scores, "scores")
        if self.model is None:
            return numpy.full(len(points), self.share)
        mapped: numpy.ndarray = self.model.predict_proba(points[:, None])[:, 1]
        return mapped


def chance(
    grades: numpy.ndarray, cutoff: int = scmle.evaluation.ranking.CUTOFF
) -> float:
    """Expected NDCG of one query's rows in a uniformly random order.

    Every position holds each row with equal probability, so the expected
    gain at a position is the mean grade, and the expected DCG is the mean
    grade times the sum of the discounts `1 / log2(i + 2)` of the first
    `cutoff` positions; NDCG divides by the DCG of the ideal order.

    Args:
        grades: Relevance grades of the query's rows, some above zero.
        cutoff: Positions counted.

    Returns:
        The expected NDCG.
    """
    shown = min(cutoff, len(grades))
    discount = 1.0 / numpy.log2(numpy.arange(shown) + 2.0)
    ideal = float(numpy.sort(grades)[::-1][:shown] @ discount)
    return float(grades.mean() * discount.sum() / ideal)


def gain(
    grades: numpy.ndarray,
    points: numpy.ndarray,
    labels: numpy.ndarray,
    draws: int = verdict.DRAWS,
    seed: int = 0,
) -> verdict.Interval:
    """Mean NDCG gain of the scores' order over a random order.

    Each scorable query (as `ranking.ndcgs` counts them: two rows or more,
    something relevant) gives NDCG minus its `chance`; queries are the
    resampling unit of the percentile interval, since rows of one query are
    not independent.

    Args:
        grades: Relevance grade of each row.
        points: Ranking score of each row.
        labels: Query label of each row.
        draws: Bootstrap resamples of the queries.
        seed: Seed of the resampling.

    Returns:
        The `gain` interval, null zero; NaN throughout when no query is
        scorable, which never supports the claim.
    """
    values = scmle.evaluation.ranking.ndcgs(grades, points, labels)[0]
    baseline = numpy.asarray(
        [
            chance(grades[rows])
            for rows in scmle.evaluation.ranking.group(labels)
            if len(rows) >= scmle.evaluation.ranking.MIN_DOCUMENTS
            and grades[rows].max() > 0
        ]
    )
    if not values.size:
        return verdict.Interval("gain", math.nan, math.nan, math.nan, 0.0)
    gains = values - baseline
    rng = numpy.random.default_rng(seed)
    means = gains[rng.integers(0, gains.size, (draws, gains.size))].mean(1)
    tail = (1 - verdict.CONFIDENCE) / 2
    lower, upper = numpy.quantile(means, (tail, 1 - tail))
    return verdict.Interval(
        "gain", float(gains.mean()), float(lower), float(upper), 0.0
    )


def crossfit(
    relevant: numpy.ndarray,
    points: numpy.ndarray,
    labels: numpy.ndarray,
    subject: str,
) -> verdict.Check:
    """Tests the relevance probability a cross-fitted Platt map gives.

    The map of each query fold is learnt on the other folds, so no query is
    scored by a map fitted on it.

    Args:
        relevant: 1 where the row is relevant.
        points: Ranking score of each row.
        labels: Query label of each row.
        subject: What the probability is of.

    Returns:
        The test; untested with fewer than two queries or when any fold's
        map does not increase, since the scores then do not mark relevance.
    """
    count = len(numpy.unique(labels))
    if count < 2:  # noqa: PLR2004 -- a map must be learnt on another query
        return verdict.Check(subject, ())
    mapped = numpy.empty(len(points))
    folds = model_selection.GroupKFold(min(FOLDS, count))
    for train, held in folds.split(points, relevant, labels):
        platt = Platt().fit(points[train], relevant[train])
        if not platt.increasing:
            return verdict.Check(subject, ())
        mapped[held] = platt.predict(points[held])
    return verdict.bernoulli(subject, mapped, relevant)


def ranking(
    relevance: Any,
    scores: Any,
    queries: Any,
    cutoff: float = 1.0,
    level: float = verdict.LEVEL,
    draws: int = verdict.DRAWS,
    seed: int = 0,
) -> verdict.Claim:
    """Tests the scores' order and the relevance their Platt map gives.

    The calibration check is `crossfit`'s; a decreasing map leaves it
    untested, which never supports the claim.

    Args:
        relevance: Relevance grade of each row, zero or more.
        scores: Ranking score of each row.
        queries: Query label of each row.
        cutoff: Least grade that counts as relevant.
        level: Family-wise size of the tests.
        draws: Bootstrap resamples of the gain interval.
        seed: Seed of the bootstrap.

    Returns:
        The ranking claim: one calibration check, untested with fewer than
        two queries or a decreasing map, and the NDCG gain interval.

    Raises:
        InputError: If the lengths differ, a grade is negative or `draws`
            is not positive.
    """
    grades = verdict.vector(relevance, "relevance")
    points = verdict.vector(scores, "scores", len(grades))
    labels = numpy.asarray(queries)
    if len(labels) != len(grades):
        raise errors.InputError("give one query label per row")
    if (grades < 0).any():
        raise errors.InputError("relevance grades must not be negative")
    if draws < 1:
        raise errors.InputError("draws must be positive")
    check = crossfit(
        (grades >= cutoff).astype(float),
        points,
        labels,
        f"relevance >= {cutoff:g}",
    )
    return verdict.Claim(
        tasks.Task.RANKING,
        verdict.STATEMENTS[tasks.Task.RANKING],
        (check,),
        (gain(grades, points, labels, draws, seed),),
        level,
    )
