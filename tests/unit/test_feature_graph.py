"""Tests for the feature-transformation graph."""

import pickle

import numpy
import pandas
import pytest
from sklearn import base

from scmle.contracts import errors
from scmle.contracts import tasks
from scmle.features import feature_graph


def node(name, op, inputs, **params):
    return tasks.FeatureNode(
        name=name, op=op, inputs=list(inputs), params=params
    )


def data_spec(**kwargs):
    return tasks.DataSpec(uri="m", target="y", **kwargs)


def transform(nodes, frame, fit_frame=None):
    transformer = feature_graph.FeatureGraphTransformer(tuple(nodes))
    transformer.fit(frame if fit_frame is None else fit_frame)
    return transformer.transform(frame)


def test_rowwise_ops_compute_expected_values():
    frame = pandas.DataFrame({"a": [1.0, 4.0, 9.0], "b": [2.0, 0.0, 3.0]})
    result = transform(
        [
            node("sq", "sqrt", ["a"]),
            node("sum", "sum", ["a", "b"]),
            node("diff", "difference", ["a", "b"]),
            node("prod", "product", ["a", "b"]),
            node("ratio", "ratio", ["a", "b"]),
            node("l1p", "log1p", ["a"]),
            node("abs", "abs", ["b"]),
            node("sqr", "square", ["b"]),
            node("miss", "is_missing", ["a"]),
        ],
        frame,
    )
    assert result["sq"].tolist() == [1.0, 2.0, 3.0]
    assert result["sum"].tolist() == [3.0, 4.0, 12.0]
    assert result["diff"].tolist() == [-1.0, 4.0, 6.0]
    assert result["prod"].tolist() == [2.0, 0.0, 27.0]
    assert result["ratio"].iloc[0] == 0.5 and numpy.isnan(
        result["ratio"].iloc[1]
    )
    assert result["l1p"].iloc[0] == pytest.approx(numpy.log(2))
    assert (
        result["sqr"].tolist() == [4.0, 0.0, 9.0] and result["miss"].sum() == 0
    )


def test_fitted_ops_learn_from_the_training_fold_only():
    train = pandas.DataFrame({"x": numpy.arange(100.0)})
    later = pandas.DataFrame({"x": [1000.0, -1000.0, 50.0]})
    transformer = feature_graph.FeatureGraphTransformer(
        (
            node("clipped", "clip", ["x"], lower=0.1, upper=0.9),
            node("z", "standardize", ["x"]),
            node("bucket", "bin", ["x"], bins=4),
        )
    ).fit(train)
    low, high = numpy.quantile(numpy.arange(100.0), [0.1, 0.9])
    result = transformer.transform(later)
    assert result["clipped"].tolist() == [high, low, 50.0]
    assert result["z"].iloc[2] == pytest.approx(
        (50 - 49.5) / numpy.arange(100.0).std(ddof=1)
    )
    assert result["bucket"].tolist() == [
        3.0,
        0.0,
        2.0,
    ]  # edges from the training data
    assert transformer.states_["bucket"].tolist() == [24.75, 49.5, 74.25]


def test_bin_keeps_missing_values_missing():
    train = pandas.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    result = transform(
        [node("b", "bin", ["x"], bins=2)],
        pandas.DataFrame({"x": [numpy.nan, 1.0]}),
        train,
    )
    assert numpy.isnan(result["b"].iloc[0]) and result["b"].iloc[1] == 0.0


def test_datepart_extracts_components_and_tolerates_bad_dates():
    frame = pandas.DataFrame(
        {"when": ["2024-03-15 14:30:00", "2025-12-01 03:00:00", "garbage"]}
    )
    result = transform(
        [
            node("month", "datepart", ["when"], part="month"),
            node("hour", "datepart", ["when"], part="hour"),
            node("dow", "datepart", ["when"], part="dayofweek"),
        ],
        frame,
    )
    assert result["month"].iloc[:2].tolist() == [3.0, 12.0]
    assert result["hour"].iloc[:2].tolist() == [14.0, 3.0]
    assert result["dow"].iloc[0] == 4.0 and numpy.isnan(result["month"].iloc[2])


def test_nodes_can_depend_on_earlier_nodes():
    frame = pandas.DataFrame({"a": [1.0, 2.0, 3.0], "b": [3.0, 2.0, 1.0]})
    result = transform(
        [node("s", "sum", ["a", "b"]), node("big", "square", ["s"])], frame
    )
    assert result["big"].tolist() == [16.0, 16.0, 16.0]


def test_transformer_is_a_well_behaved_estimator():
    transformer = feature_graph.FeatureGraphTransformer(
        (node("l", "log1p", ["a"]),)
    )
    twin = base.clone(transformer)
    assert twin.nodes == transformer.nodes and not hasattr(twin, "states_")
    fitted = transformer.fit(pandas.DataFrame({"a": [1.0, 2.0]}))
    restored = pickle.loads(pickle.dumps(fitted))
    pandas.testing.assert_frame_equal(
        restored.transform(pandas.DataFrame({"a": [3.0]})),
        fitted.transform(pandas.DataFrame({"a": [3.0]})),
    )


def series_frame():
    return pandas.DataFrame(
        {
            "grp": ["a", "a", "a", "b", "b", "b"],
            "t": pandas.to_datetime(
                ["2024-01-03", "2024-01-01", "2024-01-02"] * 2
            ),
            "v": [30.0, 10.0, 20.0, 3.0, 1.0, 2.0],
        }
    )


def build_series(nodes, frame):
    return feature_graph.SeriesFeatureBuilder(nodes, "grp", "t").apply(frame)


def test_series_features_are_computed_per_group_in_time_order():
    result = build_series(
        [
            node("prev", "lag", ["v"], k=1),
            node("change", "diff", ["v"], k=1),
            node("avg2", "rolling_mean", ["v"], window=2),
            node("run", "expanding_mean", ["v"]),
        ],
        series_frame(),
    )
    ordered = result.sort_values(["grp", "t"])
    assert ordered["prev"].tolist()[:3] == [
        pytest.approx(numpy.nan, nan_ok=True),
        10.0,
        20.0,
    ]
    assert ordered["change"].tolist()[1:3] == [10.0, 10.0]
    assert ordered["avg2"].tolist()[:3] == [10.0, 15.0, 25.0]
    assert ordered["run"].tolist()[:3] == [10.0, 15.0, 20.0]
    assert ordered["prev"].tolist()[3:][1:] == [
        1.0,
        2.0,
    ]  # group b independent of a


def test_series_features_never_use_future_rows():
    frame = series_frame()
    result = build_series(
        [
            node("prev", "lag", ["v"], k=1),
            node("avg", "rolling_mean", ["v"], window=3),
        ],
        frame,
    )
    perturbed = frame.copy()
    perturbed.loc[perturbed.t == pandas.Timestamp("2024-01-03"), "v"] *= (
        1000  # the latest rows
    )
    after = build_series(
        [
            node("prev", "lag", ["v"], k=1),
            node("avg", "rolling_mean", ["v"], window=3),
        ],
        perturbed,
    )
    earlier = frame.t < pandas.Timestamp("2024-01-03")
    # Changing the newest value cannot alter any feature of earlier rows.
    pandas.testing.assert_frame_equal(result[earlier], after[earlier])


def test_series_output_keeps_the_original_row_order_and_index():
    frame = series_frame()
    frame.index = [10, 11, 12, 13, 14, 15]
    result = build_series([node("prev", "lag", ["v"])], frame)
    assert list(result.index) == [10, 11, 12, 13, 14, 15]
    assert (
        result.loc[10, "prev"] == 20.0
        or numpy.isnan(result.loc[10, "prev"]) is False
    )


def test_series_without_groups_forms_one_series():
    frame = pandas.DataFrame(
        {
            "t": pandas.date_range("2024-01-01", periods=4),
            "v": [1.0, 2.0, 4.0, 8.0],
        }
    )
    result = feature_graph.SeriesFeatureBuilder(
        [node("prev", "lag", ["v"])], None, "t"
    ).apply(frame)
    assert numpy.isnan(result["prev"].iloc[0]) and result["prev"].tolist()[
        1:
    ] == [
        1.0,
        2.0,
        4.0,
    ]


def test_series_requires_parseable_times_and_a_time_column():
    frame = series_frame().assign(t="not a date")
    with pytest.raises(errors.ConfigError, match="unparseable"):
        build_series([node("prev", "lag", ["v"])], frame)
    with pytest.raises(errors.DataError, match="time column"):
        build_series(
            [node("prev", "lag", ["v"])], series_frame().drop(columns="t")
        )
    assert build_series([], series_frame()) is not None


def test_validation_rejects_bad_graphs():
    available = {"a", "b", "t", "y"}

    def check(nodes, **data):
        feature_graph.validate_graph(
            tasks.FeatureGraphSpec(nodes=nodes), available, data_spec(**data)
        )

    check([node("s", "sum", ["a", "b"])])
    with pytest.raises(errors.ConfigError, match="unknown feature operation"):
        check([node("s", "frobnicate", ["a"])])
    with pytest.raises(errors.ConfigError, match=r"takes 2\.\.2 inputs"):
        check([node("s", "ratio", ["a"])])
    with pytest.raises(errors.ConfigError, match="unknown parameters"):
        check([node("s", "clip", ["a"], bogus=1)])
    with pytest.raises(errors.ConfigError, match="unknown columns"):
        check([node("s", "log1p", ["ghost"])])
    with pytest.raises(
        errors.ConfigError, match="clashes with an existing column"
    ):
        check([node("a", "log1p", ["b"])])
    with pytest.raises(errors.ConfigError, match="target"):
        check([node("leak", "lag", ["y"])], time_column="t")
    with pytest.raises(errors.ConfigError, match=r"needs data\.time_column"):
        check([node("prev", "lag", ["a"])])
    with pytest.raises(errors.ConfigError, match="lag k must be >= 1"):
        check([node("prev", "lag", ["a"], k=0)], time_column="t")
    with pytest.raises(errors.ConfigError, match="datepart part"):
        check([node("d", "datepart", ["t"], part="century")])
    with pytest.raises(errors.ConfigError, match="cannot consume row-wise"):
        check(
            [node("r", "log1p", ["a"]), node("prev", "lag", ["r"])],
            time_column="t",
        )


def test_spec_rejects_cycles_duplicates_and_forward_references():
    with pytest.raises(ValueError, match="unique"):
        tasks.FeatureGraphSpec(
            nodes=[node("a", "abs", ["x"]), node("a", "abs", ["x"])]
        )
    with pytest.raises(ValueError, match="defined later"):
        tasks.FeatureGraphSpec(
            nodes=[node("a", "abs", ["b"]), node("b", "abs", ["x"])]
        )
    with pytest.raises(ValueError, match="itself"):
        tasks.FeatureGraphSpec(nodes=[node("a", "abs", ["a"])])
    with pytest.raises(ValueError, match="no inputs"):
        tasks.FeatureGraphSpec(nodes=[node("a", "abs", [])])


def test_graph_is_versioned_and_fingerprinted():
    one = tasks.FeatureGraphSpec(nodes=[node("a", "log1p", ["x"])])
    same = tasks.FeatureGraphSpec(nodes=[node("a", "log1p", ["x"])])
    other = tasks.FeatureGraphSpec(nodes=[node("a", "sqrt", ["x"])])
    assert one.fingerprint() == same.fingerprint() != other.fingerprint()
    spec = tasks.ProjectSpec(data=data_spec(), feature_graph=one)
    plain = tasks.ProjectSpec(data=data_spec())
    assert spec.config_hash() != plain.config_hash()
    assert one.version == tasks.GRAPH_VERSION
