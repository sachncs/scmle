"""Finding groups in unlabeled rows, and judging whether they are real.

Several methods and cluster counts are tried on standardized features. Each
candidate is scored by the silhouette (cohesion against separation, higher is
better), Davies-Bouldin (lower is better) and Calinski-Harabasz (higher is
better) indices, and by stability: the same method is refit on bootstrap
resamples, every resample's labels are compared with the full-data labels on
the rows both saw, and the mean adjusted Rand index says whether the same
groups come back. Clusters that do not survive resampling are not findings.
When true groups are known, `external_scores` gives the adjusted Rand index
and normalized mutual information against them.
"""

import dataclasses
from typing import Any

import numpy as np
import pandas as pd
from sklearn import base
from sklearn import cluster
from sklearn import metrics
from sklearn import mixture
from sklearn import preprocessing

MIN_ROWS = 30
STABLE_ARI = 0.7
METHODS = ("kmeans", "gaussian_mixture", "agglomerative")


@dataclasses.dataclass(frozen=True)
class Candidate:
    """One method at one cluster count.

    Attributes:
        method: Method name.
        clusters: Number of clusters.
        silhouette: Mean silhouette in [-1, 1].
        davies_bouldin: Davies-Bouldin index (lower is better).
        calinski_harabasz: Calinski-Harabasz index (higher is better).
        stability: Mean adjusted Rand index against bootstrap refits.
    """

    method: str
    clusters: int
    silhouette: float
    davies_bouldin: float
    calinski_harabasz: float
    stability: float


@dataclasses.dataclass(frozen=True)
class ClusterReport:
    """The candidates, the choice and a profile of the chosen clusters.

    Attributes:
        candidates: Every method and count tried.
        best: The candidate with the highest silhouette among the stable
            ones (or among all, if none is stable).
        stable: Whether `best` reached the stability threshold.
        labels: Cluster of each input row for `best`.
        sizes: Rows per cluster.
        profile: Mean of each feature per cluster, in original units.
    """

    candidates: tuple[Candidate, ...]
    best: Candidate
    stable: bool
    labels: np.ndarray
    sizes: dict[int, int]
    profile: pd.DataFrame

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready record (without the per-row labels)."""
        return {
            "candidates": [dataclasses.asdict(c) for c in self.candidates],
            "best": dataclasses.asdict(self.best),
            "stable": self.stable,
            "sizes": {str(k): v for k, v in self.sizes.items()},
        }


def build(method: str, count: int, seed: int) -> base.BaseEstimator:
    """A clusterer of the named kind."""
    if method == "kmeans":
        return cluster.KMeans(count, n_init=5, random_state=seed)
    if method == "gaussian_mixture":
        return mixture.GaussianMixture(count, random_state=seed)
    if method == "agglomerative":
        return cluster.AgglomerativeClustering(count)
    raise ValueError(f"unknown method '{method}'; use one of {METHODS}")


def assign(method: str, count: int, seed: int, matrix: np.ndarray) -> Any:
    """Cluster labels of the rows."""
    return build(method, count, seed).fit_predict(matrix)


def stability(
    method: str,
    count: int,
    matrix: np.ndarray,
    full: np.ndarray,
    rounds: int,
    seed: int,
) -> float:
    """Mean adjusted Rand index of bootstrap refits against the full fit."""
    generator = np.random.default_rng(seed)
    scores = []
    for round_index in range(rounds):
        rows = np.unique(generator.choice(len(matrix), len(matrix)))
        labels = assign(method, count, seed + round_index + 1, matrix[rows])
        scores.append(metrics.adjusted_rand_score(full[rows], labels))
    return float(np.mean(scores))


def score_candidate(
    method: str,
    count: int,
    matrix: np.ndarray,
    rounds: int,
    seed: int,
) -> tuple[Candidate, np.ndarray]:
    """Fits and scores one method at one cluster count."""
    labels = assign(method, count, seed, matrix)
    if len(set(labels)) < 2:
        return Candidate(method, count, -1.0, float("inf"), 0.0, 0.0), labels
    return (
        Candidate(
            method,
            count,
            float(metrics.silhouette_score(matrix, labels)),
            float(metrics.davies_bouldin_score(matrix, labels)),
            float(metrics.calinski_harabasz_score(matrix, labels)),
            stability(method, count, matrix, labels, rounds, seed),
        ),
        labels,
    )


def find_clusters(
    features: pd.DataFrame,
    counts: range | tuple[int, ...] = range(2, 9),
    methods: tuple[str, ...] = METHODS,
    rounds: int = 10,
    seed: int = 0,
) -> ClusterReport:
    """Tries each method at each count and picks the best stable one.

    Args:
        features: Numeric feature frame.
        counts: Cluster counts to try (each at least 2).
        methods: Methods from `METHODS`.
        rounds: Bootstrap refits per candidate for the stability score.
        seed: Seed.

    Returns:
        The report.

    Raises:
        ValueError: With too few rows, a count below 2 or an unknown method.
    """
    if len(features) < MIN_ROWS:
        raise ValueError(f"clustering needs at least {MIN_ROWS} rows")
    if min(counts) < 2:
        raise ValueError("cluster counts start at 2")
    numeric = features.select_dtypes("number").dropna()
    matrix = preprocessing.StandardScaler().fit_transform(numeric)
    scored = [
        (score_candidate(method, count, matrix, rounds, seed), count)
        for method in methods
        for count in counts
    ]
    candidates = tuple(item[0][0] for item in scored)
    stable = [c for c in candidates if c.stability >= STABLE_ARI]
    pool = stable or list(candidates)
    best = max(pool, key=lambda c: c.silhouette)
    labels = next(item[0][1] for item in scored if item[0][0] is best)
    profile = numeric.assign(cluster=labels).groupby("cluster").mean()
    sizes = {
        int(str(k)): int(v) for k, v in pd.Series(labels).value_counts().items()
    }
    return ClusterReport(candidates, best, bool(stable), labels, sizes, profile)


def external_scores(truth: Any, labels: Any) -> dict[str, float]:
    """Agreement with known groups: adjusted Rand index and NMI."""
    return {
        "adjusted_rand": float(metrics.adjusted_rand_score(truth, labels)),
        "normalized_mutual_info": float(
            metrics.normalized_mutual_info_score(truth, labels)
        ),
    }
