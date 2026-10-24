"""Clustering selection, stability and external validation."""

import numpy as np
import pandas as pd
import pytest
from sklearn import datasets

from scmle.unsupervised import clustering


def blobs(centers=3, rows=300, spread=0.6, seed=0):
    values, truth = datasets.make_blobs(
        rows, centers=centers, cluster_std=spread, random_state=seed
    )
    return pd.DataFrame(values, columns=["x", "y"]), truth


def test_three_separated_groups_are_found_and_judged_stable():
    features, truth = blobs()

    report = clustering.find_clusters(
        features, counts=range(2, 6), methods=("kmeans",), rounds=6
    )

    assert report.best.clusters == 3 and report.stable
    assert (
        clustering.external_scores(truth, report.labels)["adjusted_rand"] > 0.95
    )
    assert sum(report.sizes.values()) == 300
    assert list(report.profile.columns) == ["x", "y"]


def test_uniform_noise_has_no_stable_clusters():
    rng = np.random.default_rng(1)
    features = pd.DataFrame(rng.uniform(size=(400, 2)), columns=["x", "y"])

    report = clustering.find_clusters(
        features, counts=(3, 4), methods=("kmeans",), rounds=6
    )

    assert not report.stable or report.best.silhouette < 0.45


def test_methods_are_compared_and_serialized():
    features = blobs(2, 150)[0]

    report = clustering.find_clusters(features, counts=(2, 3), rounds=3)

    assert {c.method for c in report.candidates} == set(clustering.METHODS)
    assert set(report.to_dict()) == {"candidates", "best", "stable", "sizes"}


def test_bad_inputs_are_refused():
    features = blobs()[0]
    with pytest.raises(ValueError, match="at least"):
        clustering.find_clusters(features.head(10))
    with pytest.raises(ValueError, match="start at 2"):
        clustering.find_clusters(features, counts=(1, 2))
    with pytest.raises(ValueError, match="unknown method"):
        clustering.find_clusters(features, methods=("magic",))
