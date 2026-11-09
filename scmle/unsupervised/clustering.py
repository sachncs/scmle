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

import numpy
import pandas
from sklearn import base
from sklearn import cluster
from sklearn import metrics
from sklearn import mixture
from sklearn import preprocessing

MIN_ROWS = 30
MIN_CLUSTER_COUNT = 2
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
class Clustering:
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
    labels: numpy.ndarray
    sizes: dict[int, int]
    profile: pandas.DataFrame

    def to_dict(self) -> dict[str, Any]:
        """Builds a JSON-ready record of the report.

        Returns:
            A dict with keys `candidates` (list of candidate dicts), `best` (the
            chosen candidate as a dict), `stable` and `sizes` (cluster sizes
            keyed by the cluster label as a string). The per-row labels and the
            profile are omitted.
        """
        return {
            "candidates": [dataclasses.asdict(c) for c in self.candidates],
            "best": dataclasses.asdict(self.best),
            "stable": self.stable,
            "sizes": {str(k): v for k, v in self.sizes.items()},
        }


def build(method: str, count: int, seed: int) -> base.BaseEstimator:
    """Creates a clusterer of the named kind.

    Args:
        method: One of `METHODS`.
        count: Number of clusters.
        seed: Random seed (not used by agglomerative clustering).

    Returns:
        An unfitted scikit-learn clusterer.

    Raises:
        ValueError: If `method` is not one of `METHODS`.
    """
    if method == "kmeans":
        return cluster.KMeans(count, n_init=5, random_state=seed)
    if method == "gaussian_mixture":
        return mixture.GaussianMixture(count, random_state=seed)
    if method == "agglomerative":
        return cluster.AgglomerativeClustering(count)
    raise ValueError(f"unknown method '{method}'; use one of {METHODS}")


def assign(method: str, count: int, seed: int, matrix: numpy.ndarray) -> Any:
    """Fits a clusterer and labels the rows.

    Args:
        method: One of `METHODS`.
        count: Number of clusters.
        seed: Random seed.
        matrix: Feature matrix, one row per observation.

    Returns:
        The cluster label of each row, in row order. An unknown `method` makes
        the underlying `build` raise ValueError.
    """
    return build(method, count, seed).fit_predict(matrix)


def stability(
    method: str,
    count: int,
    matrix: numpy.ndarray,
    full: numpy.ndarray,
    rounds: int,
    seed: int,
) -> float:
    """Computes the mean adjusted Rand index of bootstrap refits.

    Each round resamples rows with replacement, refits, and compares the new
    labels with the full-fit labels on the distinct rows drawn.

    Args:
        method: One of `METHODS`.
        count: Number of clusters.
        matrix: Feature matrix, one row per observation.
        full: Labels from fitting all rows.
        rounds: Number of bootstrap refits.
        seed: Seed for the resampling; each refit uses a different offset.

    Returns:
        The mean adjusted Rand index over the rounds (1.0 means identical
        groupings).
    """
    generator = numpy.random.default_rng(seed)
    scores = []
    for round_index in range(rounds):
        rows = numpy.unique(generator.choice(len(matrix), len(matrix)))
        labels = assign(method, count, seed + round_index + 1, matrix[rows])
        scores.append(metrics.adjusted_rand_score(full[rows], labels))
    return float(numpy.mean(scores))


def score_candidate(
    method: str,
    count: int,
    matrix: numpy.ndarray,
    rounds: int,
    seed: int,
) -> tuple[Candidate, numpy.ndarray]:
    """Fits and scores one method at one cluster count.

    Args:
        method: One of `METHODS`.
        count: Number of clusters.
        matrix: Feature matrix, one row per observation.
        rounds: Bootstrap refits for the stability score.
        seed: Random seed.

    Returns:
        A pair of the scored candidate and the full-fit labels. If the fit
        yields fewer than two distinct labels, the candidate carries the worst
        scores (silhouette -1, Davies-Bouldin infinity, Calinski-Harabasz 0,
        stability 0) and no indices are computed.
    """
    labels = assign(method, count, seed, matrix)
    if len(set(labels)) < MIN_CLUSTER_COUNT:
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
    features: pandas.DataFrame,
    counts: range | tuple[int, ...] = range(2, 9),
    methods: tuple[str, ...] = METHODS,
    rounds: int = 10,
    seed: int = 0,
) -> Clustering:
    """Tries each method at each count and picks the best stable one.

    Args:
        features: Numeric feature frame. Non-numeric columns are dropped, then
            rows with missing values; the rest is standardized.
        counts: Cluster counts to try (each at least 2).
        methods: Methods from `METHODS`.
        rounds: Bootstrap refits per candidate for the stability score.
        seed: Random seed.

    Returns:
        The report: every candidate, the best one (highest silhouette among
        candidates whose stability reaches `STABLE_ARI`, or among all if none
        does), its labels, sizes and per-cluster feature means.

    Raises:
        ValueError: If there are fewer than `MIN_ROWS` rows, any count is below
            2, or a method is unknown.
    """
    if len(features) < MIN_ROWS:
        raise ValueError(f"clustering needs at least {MIN_ROWS} rows")
    if min(counts) < MIN_CLUSTER_COUNT:
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
        int(str(k)): int(v)
        for k, v in pandas.Series(labels).value_counts().items()
    }
    return Clustering(candidates, best, bool(stable), labels, sizes, profile)


def external_scores(truth: Any, labels: Any) -> dict[str, float]:
    """Compares labels with known groups.

    Args:
        truth: The known group of each row.
        labels: The cluster label of each row, in the same order.

    Returns:
        A dict with `adjusted_rand` (adjusted Rand index) and
        `normalized_mutual_info` (normalized mutual information).
    """
    return {
        "adjusted_rand": float(metrics.adjusted_rand_score(truth, labels)),
        "normalized_mutual_info": float(
            metrics.normalized_mutual_info_score(truth, labels)
        ),
    }
