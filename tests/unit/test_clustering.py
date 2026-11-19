"""Clustering selection, stability and external validation."""

import numpy
import pandas
import pytest
from sklearn import datasets
from sklearn import metrics

from scmle.contracts import errors
from scmle.unsupervised import clustering


def blobs(centers=3, rows=300, spread=0.6, seed=0):
    values, truth = datasets.make_blobs(
        rows, centers=centers, cluster_std=spread, random_state=seed
    )
    return pandas.DataFrame(values, columns=["x", "y"]), truth


def test_three_separated_groups_are_found_and_judged_stable():
    features, truth = blobs()

    report = clustering.discover(
        features, counts=range(2, 6), methods=("kmeans",), rounds=6
    )

    assert report.best.clusters == 3 and report.stable
    assert clustering.compare(truth, report.labels)["adjusted_rand"] > 0.95
    assert sum(report.sizes.values()) == 300
    assert list(report.profile.columns) == ["x", "y"]


def test_uniform_noise_has_no_stable_clusters():
    rng = numpy.random.default_rng(1)
    features = pandas.DataFrame(rng.uniform(size=(400, 2)), columns=["x", "y"])

    report = clustering.discover(
        features, counts=(3, 4), methods=("kmeans",), rounds=6
    )

    assert not report.stable or report.best.silhouette < 0.45


def test_methods_are_compared_and_serialized():
    features = blobs(2, 150)[0]

    report = clustering.discover(features, counts=(2, 3), rounds=3)

    assert {c.method for c in report.candidates} == set(clustering.METHODS)
    assert set(report.to_dict()) == {"candidates", "best", "stable", "sizes"}


def test_bad_inputs_are_refused():
    features = blobs()[0]
    with pytest.raises(ValueError, match="at least"):
        clustering.discover(features.head(10))
    with pytest.raises(ValueError, match="start at 2"):
        clustering.discover(features, counts=(1, 2))
    with pytest.raises(ValueError, match="unknown method"):
        clustering.discover(features, methods=("magic",))


@pytest.mark.parametrize("method", clustering.METHODS)
def test_the_fitted_model_gives_every_training_row_its_label(method):
    features = blobs()[0]

    report = clustering.discover(
        features, counts=(3,), methods=(method,), rounds=2
    )

    assert report.model is not None
    numpy.testing.assert_array_equal(
        report.model.predict(features), report.labels
    )


def test_new_rows_go_to_the_nearest_group():
    features, truth = blobs()
    report = clustering.discover(
        features, counts=(3,), methods=("agglomerative",), rounds=2
    )
    centres = features.groupby(report.labels).mean()

    assigned = report.model.predict(centres)

    numpy.testing.assert_array_equal(assigned, centres.index.to_numpy())
    assert clustering.compare(truth, report.labels)["adjusted_rand"] > 0.95


def test_a_missing_value_is_imputed_when_assigning():
    features = blobs()[0]
    report = clustering.discover(
        features, counts=(3,), methods=("kmeans",), rounds=2
    )
    gap = features.head(1).assign(y=numpy.nan)

    assert report.model.predict(gap).shape == (1,)


def test_the_indices_score_a_grouping():
    features, truth = blobs()
    matrix = features.to_numpy()

    scores = clustering.score(matrix, truth)

    assert set(scores) == {
        "silhouette",
        "davies_bouldin",
        "calinski_harabasz",
    }
    assert scores["silhouette"] == metrics.silhouette_score(matrix, truth)
    assert clustering.score(matrix, numpy.zeros(300)) == {
        "silhouette": -1.0,
        "davies_bouldin": float("inf"),
        "calinski_harabasz": 0.0,
    }


def test_unusable_columns_and_counts_are_refused():
    features = blobs()[0]
    with pytest.raises(errors.InputError, match="no numeric"):
        clustering.discover(features.astype(str))
    holed = features.assign(x=numpy.nan)
    with pytest.raises(errors.InputError, match="complete rows"):
        clustering.discover(holed.assign(z=1.0))
    with pytest.raises(errors.InputError, match="fewer clusters"):
        clustering.discover(features.head(40), counts=(2, 40))
