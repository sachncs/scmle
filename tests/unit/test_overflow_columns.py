"""The overflow block: chosen at fit time, cross-fitted, named after its columns."""

import pickle

import numpy
import pandas

from scmle.contracts import data
from scmle.preparation import preprocessing

PLAN = data.Blueprint(numeric=(), categorical=("code",), cardinal=("code",))
OPTIONS = data.Preprocessing(cardinality="ordinal")


def unique(rows=2000, seed=0):
    codes = pandas.DataFrame({"code": [f"c{i}" for i in range(rows)]})
    target = pandas.Series(numpy.random.default_rng(seed).integers(0, 2, rows))
    return codes, target


def test_the_overflow_encoding_is_cross_fitted():
    # Every level is seen once, so a plain target mean would be the row's
    # own label; the cross-fitted encoding of a training row never sees it.
    codes, target = unique()
    prep = preprocessing.build(PLAN, OPTIONS)
    matrix = prep.fit_transform(codes, target)
    names = list(prep.get_feature_names_out())
    assert names == ["code", "code__target"]
    assert abs(numpy.corrcoef(matrix[:, 1], target)[0, 1]) < 0.1


def test_the_wide_columns_are_chosen_on_the_fitted_rows():
    frame = pandas.DataFrame(
        {
            "a": list("abcd"),
            "b": list("aabb"),
            "c": [None, "x", "y", "z"],
            "d": list("abcd"),
        }
    )
    # A missing value is one more level once the missing token fills it;
    # a column outside the high-cardinality set is never chosen.
    choose = preprocessing.Overflow(("a", "b", "c"), levels=3)
    assert choose(frame) == ["a", "c"]
    assert preprocessing.Overflow(("a", "b", "c"), levels=4)(frame) == []


def test_a_fitted_overflow_survives_a_pickle():
    codes, target = unique(rows=600)
    prep = preprocessing.build(PLAN, OPTIONS).fit(codes, target)
    again = pickle.loads(pickle.dumps(prep))
    assert numpy.array_equal(again.transform(codes), prep.transform(codes))
