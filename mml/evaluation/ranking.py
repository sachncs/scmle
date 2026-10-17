"""Ranking metrics, averaged over queries."""

import dataclasses
from typing import Any

import numpy as np
import pandas as pd
from sklearn import metrics as sk_metrics

from mml.contracts import metrics as metric_contracts

CUTOFF = 10
MIN_DOCUMENTS = 2


def query_groups(queries: Any) -> list[np.ndarray]:
    """Row positions of each query."""
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


def mean_average_precision(
    relevance: np.ndarray,
    scores: np.ndarray,
    queries: Any,
    cutoff: int = CUTOFF,
) -> float:
    """Average precision at `cutoff` (relevant means grade above zero)."""
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


@dataclasses.dataclass(frozen=True, slots=True)
class RankingMetric(metric_contracts.Metric):
    """A ranking metric read off a ranking estimator.

    Attributes:
        kind: `ndcg` or `map`.
    """

    kind: str = "ndcg"

    def accepts_weights(self) -> bool:
        """Queries are the unit of scoring, not weighted rows."""
        return False

    def scorer(self, positive: Any = None) -> Any:
        """Builds a scorer `(estimator, features, relevance)`."""
        function = mean_ndcg if self.kind == "ndcg" else mean_average_precision

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
