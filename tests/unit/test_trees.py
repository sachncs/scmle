"""Classification trees that split the same on every CPU architecture."""

import hashlib
import pickle

import numpy
import pytest
from sklearn import base
from sklearn import ensemble
from sklearn import tree

from scmle.contracts import errors
from scmle.models.sklearn import trees

KINDS = [
    (trees.Tree, tree.DecisionTreeClassifier),
    (trees.Forest, ensemble.RandomForestClassifier),
    (trees.Extra, ensemble.ExtraTreesClassifier),
]


def problem(rows=400, seed=7):
    """The interaction task: XOR of two signs, three one-hot levels."""
    rng = numpy.random.default_rng(seed)
    first, second = rng.normal(size=rows), rng.normal(size=rows)
    level = rng.integers(0, 3, rows)
    features = numpy.column_stack(
        [first, second, level == 0, level == 1, level == 2]
    ).astype(float)
    flipped = rng.uniform(size=rows) < 0.1
    labels = numpy.where((first * second > 0) ^ flipped, "yes", "no")
    return features, labels


def digest(values):
    return hashlib.sha256(numpy.ascontiguousarray(values).tobytes()).hexdigest()


def test_an_exact_tie_goes_to_the_first_threshold():
    # [2, 0] | [4, 4] and [4, 1] | [2, 3] have the same Gini impurity (4).
    labels = numpy.array([0, 0, 1, 0, 0, 1, 0, 1, 0, 1])
    features = numpy.arange(10, dtype=float)[:, None]
    grown = trees.Tree(max_depth=1).fit(features, labels)
    assert grown.tree_.threshold[0] == 1.5


@pytest.mark.parametrize(("kind", "stock"), KINDS)
def test_the_fit_is_a_stock_classifier_of_the_same_splits(kind, stock):
    features, labels = problem()
    grown = kind(random_state=3).fit(features, labels)
    plain = stock(random_state=3).fit(features, labels)
    numpy.testing.assert_array_equal(grown.classes_, ["no", "yes"])
    assert grown.n_classes_ == 2
    assert grown.n_outputs_ == 1
    proba = grown.predict_proba(features)
    numpy.testing.assert_allclose(proba.sum(axis=1), 1.0)
    numpy.testing.assert_array_equal(
        grown.predict(features), grown.classes_[proba.argmax(axis=1)]
    )
    # Gini and the indicators' squared error rank splits alike; only exact
    # ties may go another way, so the two fits agree on almost every row.
    agree = numpy.mean(grown.predict(features) == plain.predict(features))
    assert agree > 0.95
    numpy.testing.assert_allclose(grown.feature_importances_.sum(), 1.0)


def test_node_impurity_is_the_gini_impurity():
    features, labels = problem()
    grown = trees.Tree(max_depth=2).fit(features, labels)
    plain = tree.DecisionTreeClassifier(max_depth=2).fit(features, labels)
    assert grown.tree_.impurity[0] == pytest.approx(plain.tree_.impurity[0])


@pytest.mark.parametrize(("kind", "stock"), KINDS)
def test_clone_pickle_and_stock_keep_the_predictions(kind, stock):
    features, labels = problem()
    grown = kind(random_state=0).fit(features, labels)
    twin = base.clone(grown)
    assert type(twin) is kind
    assert twin.get_params() == grown.get_params()
    restored = pickle.loads(pickle.dumps(grown))
    plain = trees.stock(grown)
    assert type(plain) is stock
    for other in (twin.fit(features, labels), restored, plain):
        numpy.testing.assert_array_equal(
            other.predict_proba(features), grown.predict_proba(features)
        )
    assert trees.stock(plain) is plain


def test_class_weights_and_row_weights_reach_the_split_search():
    features, labels = problem()
    weighted = trees.Forest(random_state=0, class_weight="balanced")
    plain = trees.Forest(random_state=0)
    rows = numpy.where(labels == "yes", 3.0, 1.0)
    left = weighted.fit(features, labels).predict_proba(features)
    right = plain.fit(features, labels, sample_weight=rows)
    assert not numpy.array_equal(
        left, plain.fit(features, labels).predict_proba(features)
    )
    assert not numpy.array_equal(right.predict_proba(features), left)


@pytest.mark.parametrize(
    "params",
    [
        {"oob_score": True},
        {"warm_start": True},
        {"class_weight": "balanced_subsample"},
    ],
)
def test_options_the_twin_cannot_reproduce_are_refused(params):
    features, labels = problem()
    with pytest.raises(errors.InputError):
        trees.Forest(**params).fit(features, labels)


def test_the_forest_is_pinned_bit_for_bit():
    # The same bytes on arm64 and x86-64: the stock classifier differs
    # between them on this problem (the C6 `interaction` task).
    features, labels = problem()
    grown = trees.Forest(random_state=7).fit(features, labels)
    assert digest(grown.predict_proba(features)) == PINNED


@pytest.mark.parametrize("kind", [trees.Tree, trees.Forest, trees.Extra])
def test_node_padding_is_zero_whatever_the_allocator_held(kind):
    # The seven padding bytes after `missing_go_to_left` are pickled, so
    # leftover allocator bytes there made bundles differ between processes.
    features, labels = problem()
    poisoned = numpy.full((8, 1 << 16), 0xAB, dtype=numpy.uint8)
    del poisoned
    grown = kind(random_state=7, max_depth=4).fit(features, labels)
    for member in getattr(grown, "estimators_", [grown]):
        nodes = member.tree_.__getstate__()["nodes"]
        start = nodes.dtype.fields["missing_go_to_left"][1] + 1
        raw = nodes.view(numpy.uint8).reshape(len(nodes), nodes.dtype.itemsize)
        assert not raw[:, start:].any()


# Measured on macOS arm64 and on x86-64 (Rosetta), scikit-learn 1.9.1; the
# stock forest gave 9b03b644... and 68bfd25e... there.
PINNED = "4e80f90bec57c25664d4b107e75778fee1a9d7d0ba2e9fbd9acb535f35e8314b"
