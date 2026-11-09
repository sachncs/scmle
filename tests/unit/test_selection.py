"""Tests for automatic feature construction and selection."""

import numpy
import pandas

from scmle.data.curation import selection


def frame(rows=600, seed=0):
    rng = numpy.random.default_rng(seed)
    data = pandas.DataFrame(
        {
            "price": rng.uniform(1, 10, rows),
            "quantity": rng.uniform(1, 10, rows),
            "noise": rng.normal(size=rows),
        }
    )
    return data, (data.price / data.quantity > 1).astype(int)


def test_the_most_associated_columns_are_combined_and_original_kept():
    data, target = frame()

    constructor = selection.FeatureConstructor(
        ("price", "quantity", "noise"), top=2
    ).fit(data, target)
    built = constructor.transform(data)

    assert constructor.pairs_ == [("price", "quantity")]
    assert set(built.columns) >= {
        "price",
        "quantity",
        "noise",
        "auto__price__over__quantity",
        "auto__price__minus__quantity",
    }
    expected = data.price / data.quantity
    numpy.testing.assert_allclose(
        built["auto__price__over__quantity"], expected
    )


def test_division_by_zero_becomes_missing_not_infinite():
    data, target = frame()
    data.loc[0, "quantity"] = 0.0

    built = (
        selection.FeatureConstructor(("price", "quantity"), top=2)
        .fit(data, target)
        .transform(data)
    )

    assert numpy.isnan(built.loc[0, "auto__price__over__quantity"])
    assert not numpy.isinf(built.select_dtypes("number").to_numpy()).any()


def test_construction_learns_only_from_the_training_fold():
    data, target = frame()
    constructor = selection.FeatureConstructor(
        ("price", "quantity", "noise"), top=2
    ).fit(data.iloc[:300], target.iloc[:300])

    other = constructor.transform(data.iloc[300:])

    assert constructor.pairs_ == [("price", "quantity")]
    assert len(other) == 300


def test_names_follow_the_transformed_columns():
    data, target = frame()
    constructor = selection.FeatureConstructor(("price", "quantity")).fit(
        data, target
    )

    names = list(constructor.get_feature_names_out())

    assert names == list(constructor.transform(data).columns)


def test_the_sparse_selector_drops_noise_and_keeps_signal():
    rng = numpy.random.default_rng(1)
    signal = rng.normal(size=(800, 2))
    noise = rng.normal(size=(800, 6))
    matrix = numpy.hstack([signal, noise])
    target = (signal[:, 0] + signal[:, 1] > 0).astype(int)

    chooser = selection.SparseSelector(classification=True).fit(matrix, target)

    assert chooser.support_[:2].all() and chooser.support_[2:].sum() <= 2
    assert chooser.transform(matrix).shape[1] == chooser.support_.sum()


def test_regression_selection_and_the_never_empty_guarantee():
    rng = numpy.random.default_rng(2)
    matrix = rng.normal(size=(500, 5))
    target = 3 * matrix[:, 1] + rng.normal(0, 0.1, 500)

    kept = selection.SparseSelector(classification=False).fit(matrix, target)
    assert kept.support_[1]

    heavy = selection.SparseSelector(classification=False, alpha=1e6).fit(
        matrix, target
    )
    assert heavy.support_.sum() == 1 and heavy.support_[1]


def test_the_sparse_selector_handles_more_than_two_classes():
    """liblinear cannot fit multiclass, which failed whole trials."""
    rng = numpy.random.default_rng(3)
    matrix = rng.normal(size=(300, 6))
    target = numpy.digitize(matrix[:, 0] + 0.1 * matrix[:, 1], [-0.5, 0.5])

    chooser = selection.SparseSelector(classification=True).fit(matrix, target)

    assert set(numpy.unique(target)) == {0, 1, 2}
    assert chooser.support_[0]
    assert chooser.transform(matrix).shape[0] == 300
