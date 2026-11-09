"""Ranking metrics, averaged over queries."""

import dataclasses
from typing import Any

import numpy as np
import pandas as pd
from sklearn import metrics as sk_metrics

from scmle.contracts import metrics as metric_contracts

CUTOFF = 10
MIN_DOCUMENTS = 2


def query_groups(queries: Any) -> list[np.ndarray]:
    """Row positions of each query.

    Args:
        queries: Query label of each row.

    Returns:
        One array of row positions per distinct query (labels compared as
        text), in order of first appearance.
    """
    codes, uniques = pd.factorize(pd.Series(np.asarray(queries)).astype(str))
    return [np.flatnonzero(codes == code) for code in range(len(uniques))]


def mean_ndcg(
    relevance: np.ndarray,
    scores: np.ndarray,
    queries: Any,
    cutoff: int = CUTOFF,
) -> float:
    """Normalized discounted cumulative gain at `cutoff`, per-query mean.

    Queries with fewer than two documents, or with nothing relevant, cannot
    be ranked and are skipped.

    Args:
        relevance: Relevance grade of each document.
        scores: Predicted score of each document, aligned with `relevance`.
        queries: Query label of each document.
        cutoff: Number of top-ranked documents considered.

    Returns:
        The mean NDCG over the scorable queries, or NaN if there are none.
    """
    values = []
    for rows in query_groups(queries):
        truth = np.asarray(relevance, dtype=float)[rows]
        if len(rows) < MIN_DOCUMENTS or truth.max() <= 0:
            continue
        values.append(
            sk_metrics.ndcg_score(
                truth[None, :],
                np.asarray(scores, dtype=float)[rows][None, :],
                k=cutoff,
            )
        )
    return float(np.mean(values)) if values else float("nan")


def per_query_ndcg(
    relevance: np.ndarray,
    scores: np.ndarray,
    queries: Any,
    cutoff: int = CUTOFF,
) -> tuple[np.ndarray, int]:
    """NDCG of every scorable query and the number of excluded queries.

    A query with nothing relevant (or a single document) is excluded and
    counted, not scored as zero.

    Args:
        relevance: Relevance grade of each document.
        scores: Predicted score of each document, aligned with `relevance`.
        queries: Query label of each document.
        cutoff: Number of top-ranked documents considered.

    Returns:
        The NDCG of each scorable query, in query order, and the number of
        excluded queries.
    """
    values = []
    excluded = 0
    for rows in query_groups(queries):
        truth = np.asarray(relevance, dtype=float)[rows]
        if len(rows) < MIN_DOCUMENTS or truth.max() <= 0:
            excluded += 1
            continue
        values.append(
            sk_metrics.ndcg_score(
                truth[None, :],
                np.asarray(scores, dtype=float)[rows][None, :],
                k=cutoff,
            )
        )
    return np.asarray(values, dtype=float), excluded


def query_bootstrap_ndcg(
    relevance: np.ndarray,
    scores: np.ndarray,
    queries: Any,
    cutoff: int = CUTOFF,
    replicates: int = 1000,
    level: float = 0.95,
    seed: int = 0,
) -> dict[str, float]:
    """Mean NDCG with a percentile interval from resampling whole queries.

    Documents of one query are not independent, so queries are the
    resampling unit.

    Args:
        relevance: Relevance grade of each document.
        scores: Predicted score of each document, aligned with `relevance`.
        queries: Query label of each document.
        cutoff: Number of top-ranked documents considered.
        replicates: Number of bootstrap resamples of the queries.
        level: Confidence level of the interval.
        seed: Seed of the resampling.

    Returns:
        The `mean`, `lower` and `upper` limits (NaN when no query is
        scorable), the number of scored `queries` and the number
        `excluded` (queries with a single document or nothing relevant).
    """
    values, excluded = per_query_ndcg(relevance, scores, queries, cutoff)
    if values.size == 0:
        nan = float("nan")
        return {
            "mean": nan,
            "lower": nan,
            "upper": nan,
            "queries": 0,
            "excluded": excluded,
        }
    generator = np.random.default_rng(seed)
    draws = generator.integers(0, values.size, size=(replicates, values.size))
    means = values[draws].mean(axis=1)
    tail = (1.0 - level) / 2.0
    return {
        "mean": float(values.mean()),
        "lower": float(np.quantile(means, tail)),
        "upper": float(np.quantile(means, 1.0 - tail)),
        "queries": int(values.size),
        "excluded": excluded,
    }


def mean_average_precision(
    relevance: np.ndarray,
    scores: np.ndarray,
    queries: Any,
    cutoff: int = CUTOFF,
) -> float:
    """Average precision at `cutoff` (relevant means grade above zero).

    Args:
        relevance: Relevance grade of each document.
        scores: Predicted score of each document, aligned with `relevance`.
        queries: Query label of each document.
        cutoff: Number of top-ranked documents considered.

    Returns:
        The mean over queries with at least two documents and one relevant
        document, or NaN if there are none.
    """
    values = []
    for rows in query_groups(queries):
        truth = np.asarray(relevance, dtype=float)[rows] > 0
        if len(rows) < MIN_DOCUMENTS or not truth.any():
            continue
        order = np.argsort(
            -np.asarray(scores, dtype=float)[rows], kind="stable"
        )
        hits = truth[order][:cutoff]
        precision = np.cumsum(hits) / (np.arange(len(hits)) + 1)
        values.append(
            float((precision * hits).sum() / min(truth.sum(), cutoff))
        )
    return float(np.mean(values)) if values else float("nan")


def ranked_hits(
    relevance: np.ndarray, scores: np.ndarray, rows: np.ndarray
) -> np.ndarray:
    """Relevance (above zero) of one query's documents in score order.

    Args:
        relevance: Relevance grade of each document.
        scores: Predicted score of each document, aligned with `relevance`.
        rows: Row positions of the query's documents.

    Returns:
        A boolean array, highest score first (ties in row order): True where
        the grade is above zero.
    """
    truth = np.asarray(relevance, dtype=float)[rows] > 0
    order = np.argsort(-np.asarray(scores, dtype=float)[rows], kind="stable")
    return truth[order]


def mean_reciprocal_rank(
    relevance: np.ndarray, scores: np.ndarray, queries: Any
) -> float:
    """Average of one over the rank of the first relevant document.

    Args:
        relevance: Relevance grade of each document.
        scores: Predicted score of each document, aligned with `relevance`.
        queries: Query label of each document.

    Returns:
        The mean reciprocal rank over queries with at least two documents
        and one relevant document, or NaN if there are none.
    """
    values = []
    for rows in query_groups(queries):
        hits = ranked_hits(relevance, scores, rows)
        if len(rows) < MIN_DOCUMENTS or not hits.any():
            continue
        values.append(1.0 / (int(np.argmax(hits)) + 1))
    return float(np.mean(values)) if values else float("nan")


def precision_at(
    relevance: np.ndarray,
    scores: np.ndarray,
    queries: Any,
    cutoff: int = CUTOFF,
) -> float:
    """Share of the top `cutoff` documents that are relevant, per query.

    A query with fewer than `cutoff` documents is divided by its own document
    count.

    Args:
        relevance: Relevance grade of each document.
        scores: Predicted score of each document, aligned with `relevance`.
        queries: Query label of each document.
        cutoff: Number of top-ranked documents considered.

    Returns:
        The mean over queries with at least two documents and one relevant
        document, or NaN if there are none.
    """
    values = []
    for rows in query_groups(queries):
        hits = ranked_hits(relevance, scores, rows)
        if len(rows) < MIN_DOCUMENTS or not hits.any():
            continue
        values.append(float(hits[:cutoff].sum() / min(cutoff, len(hits))))
    return float(np.mean(values)) if values else float("nan")


def recall_at(
    relevance: np.ndarray,
    scores: np.ndarray,
    queries: Any,
    cutoff: int = CUTOFF,
) -> float:
    """Share of a query's relevant documents found in the top `cutoff`.

    Args:
        relevance: Relevance grade of each document.
        scores: Predicted score of each document, aligned with `relevance`.
        queries: Query label of each document.
        cutoff: Number of top-ranked documents considered.

    Returns:
        The mean over queries with at least two documents and one relevant
        document, or NaN if there are none.
    """
    values = []
    for rows in query_groups(queries):
        hits = ranked_hits(relevance, scores, rows)
        if len(rows) < MIN_DOCUMENTS or not hits.any():
            continue
        found = int(hits[:cutoff].sum())
        values.append(found / int(hits.sum()))
    return float(np.mean(values)) if values else float("nan")


FUNCTIONS = {
    "ndcg": mean_ndcg,
    "map": mean_average_precision,
    "mrr": mean_reciprocal_rank,
    "precision_at_10": precision_at,
    "recall_at_10": recall_at,
}


@dataclasses.dataclass(frozen=True, slots=True)
class RankingMetric(metric_contracts.Metric):
    """A ranking metric read off a ranking estimator.

    Attributes:
        kind: A key of `FUNCTIONS` (`ndcg`, `map`, `mrr`, `precision_at_10`,
            `recall_at_10`). The inherited `Metric` fields are documented on
            `Metric`.
    """

    kind: str = "ndcg"

    def accepts_weights(self) -> bool:
        """Returns whether the metric can use row weights.

        Queries are the unit of scoring, not weighted rows.

        Returns:
            Always False.
        """
        return False

    def scorer(self, positive: Any = None) -> Any:
        """Builds a scorer `(estimator, features, relevance)`.

        Args:
            positive: Unused; ranking has no positive class.

        Returns:
            A function that scores the estimator's predictions for the
            features against the relevance grades, grouping documents by the
            estimator's `query_column`. It ignores sample weights.
        """
        function = FUNCTIONS[self.kind]

        def score(
            estimator: Any,
            features: pd.DataFrame,
            target: pd.Series,
            sample_weight: Any = None,
        ) -> float:
            return function(
                np.asarray(target, dtype=float),
                np.asarray(estimator.predict(features), dtype=float),
                features[estimator.query_column],
            )

        return score
