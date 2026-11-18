"""Behavioural tests for the feature graph operations, registry and history."""

import numpy
import pandas
import pytest

from scmle.contracts import errors
from scmle.contracts import tasks
from scmle.data.feature_graph import history
from scmle.data.feature_graph import operations
from scmle.data.feature_graph import registry
from scmle.data.feature_graph import sequences
from scmle.data.feature_graph import transformer


def node(name, op, inputs, **params):
    return tasks.Node(name=name, op=op, inputs=list(inputs), params=params)


def spec(**kwargs):
    return tasks.Dataset(uri="m", target="y", **kwargs)


def run(op, values, **params):
    operation = registry.lookup(op)
    inputs = [pandas.Series(v, dtype=float) for v in values]
    resolved = {**operation.defaults, **params}
    state = operation.fit(inputs, resolved)
    return operation.apply(inputs, resolved, state).tolist(), state


def test_the_registry_holds_every_operation_by_name():
    assert sorted(registry.OPS) == sorted(
        [
            "log1p",
            "sqrt",
            "square",
            "abs",
            "is_missing",
            "sum",
            "difference",
            "product",
            "ratio",
            "clip",
            "standardize",
            "bin",
            "datepart",
            "lag",
            "diff",
            "rolling_mean",
            "rolling_std",
            "expanding_mean",
        ]
    )
    assert isinstance(registry.lookup("lag"), sequences.Lag)
    with pytest.raises(errors.ConfigError, match="unknown feature operation"):
        registry.lookup("nope")


def test_validate_checks_arity_at_both_bounds():
    ratio = operations.Ratio()
    ratio.validate(node("r", "ratio", ["a", "b"]))
    with pytest.raises(errors.ConfigError, match=r"takes 2\.\.2 inputs, got 1"):
        ratio.validate(node("r", "ratio", ["a"]))
    with pytest.raises(errors.ConfigError, match=r"got 3"):
        ratio.validate(node("r", "ratio", ["a", "b", "c"]))
    total = operations.Sum()
    total.validate(node("s", "sum", [str(i) for i in range(64)]))
    with pytest.raises(errors.ConfigError, match=r"2\.\.64 inputs, got 65"):
        total.validate(node("s", "sum", [str(i) for i in range(65)]))
    assert operations.MAX_ARITY == 64


def test_validate_rejects_unknown_parameters():
    with pytest.raises(errors.ConfigError, match=r"unknown parameters \['z'\]"):
        operations.Clip().validate(node("c", "clip", ["a"], z=1))
    operations.Clip().validate(node("c", "clip", ["a"], lower=0.1))


def test_resolved_parameters_overlay_the_defaults():
    clip = operations.Clip()
    assert clip.resolved(node("c", "clip", ["a"], lower=0.2)) == {
        "lower": 0.2,
        "upper": 0.99,
    }
    assert operations.Log1p().fit([], {}) is None


def test_unary_operations_compute_exact_values():
    assert run("log1p", [[-5.0, 0.0, numpy.e - 1]])[0] == pytest.approx(
        [0.0, 0.0, 1.0]
    )
    assert run("sqrt", [[-4.0, 4.0, 9.0]])[0] == [0.0, 2.0, 3.0]
    assert run("square", [[-3.0, 2.0]])[0] == [9.0, 4.0]
    assert run("abs", [[-3.0, 2.0]])[0] == [3.0, 2.0]
    assert run("is_missing", [[numpy.nan, 1.0]])[0] == [1.0, 0.0]


def test_binary_operations_compute_exact_values():
    assert run("sum", [[1.0, 2.0], [10.0, 20.0], [100.0, 0.0]])[0] == [
        111.0,
        22.0,
    ]
    assert run("difference", [[5.0, 1.0], [2.0, 4.0]])[0] == [3.0, -3.0]
    assert run("product", [[2.0, 3.0], [4.0, 5.0], [2.0, 2.0]])[0] == [
        16.0,
        30.0,
    ]
    ratios = run("ratio", [[6.0, 1.0], [3.0, 0.0]])[0]
    assert ratios[0] == 2.0
    assert numpy.isnan(ratios[1])


def test_operations_turn_infinities_into_missing_values():
    squares = run("square", [[numpy.inf, 2.0]])[0]
    assert numpy.isnan(squares[0])
    assert squares[1] == 4.0


def test_clip_learns_quantile_bounds_on_the_training_data():
    clip = operations.Clip()
    params = {"lower": 0.0, "upper": 0.5}
    train = [pandas.Series([0.0, 10.0, 20.0])]
    state = clip.fit(train, params)
    assert state == (0.0, 10.0)
    assert isinstance(state[0], float)
    later = [pandas.Series([-5.0, 5.0, 50.0])]
    assert clip.apply(later, params, state).tolist() == [0.0, 5.0, 10.0]
    empty = clip.fit([pandas.Series([numpy.nan])], params)
    assert empty == (-numpy.inf, numpy.inf)


def test_standardize_uses_training_mean_and_spread():
    standard = operations.Standardize()
    state = standard.fit([pandas.Series([1.0, 3.0])], {})
    assert state == (2.0, pytest.approx(numpy.sqrt(2.0)))
    scores = standard.apply([pandas.Series([2.0, 4.0])], {}, (2.0, 2.0))
    assert scores.tolist() == [0.0, 1.0]
    assert standard.fit([pandas.Series([5.0])], {}) == (5.0, 1.0)
    assert standard.fit([pandas.Series([5.0, 5.0])], {}) == (5.0, 1.0)
    assert standard.fit([pandas.Series([numpy.nan])], {}) == (0.0, 1.0)


def test_bin_learns_interior_edges_and_codes_values():
    binner = operations.Bin()
    state = binner.fit([pandas.Series(numpy.arange(5.0))], {"bins": 2})
    assert state.tolist() == [2.0]
    codes = binner.apply(
        [pandas.Series([0.0, 2.0, 3.0, numpy.nan])], {}, state
    ).tolist()
    assert codes[:3] == [0.0, 1.0, 1.0]
    assert numpy.isnan(codes[3])
    edges = binner.fit([pandas.Series(numpy.arange(5.0))], {"bins": 4})
    assert edges.tolist() == [1.0, 2.0, 3.0]
    assert binner.fit([pandas.Series([1.0, 1.0])], {"bins": 4}).tolist() == [
        1.0
    ]
    assert binner.fit([pandas.Series([numpy.nan])], {"bins": 4}).size == 0


def test_calendar_extracts_parts_and_validates_the_part():
    calendar = operations.Calendar()
    stamps = [pandas.Series(["1970-01-03 04:05", "bad"])]
    assert calendar.apply(stamps, {"part": "day"}, None).tolist()[0] == 3.0
    assert calendar.apply(stamps, {"part": "hour"}, None).tolist()[0] == 4.0
    days = calendar.apply(stamps, {"part": "timestamp"}, None).tolist()
    assert days[0] == pytest.approx(2 + (4 * 60 + 5) / 1440)
    assert numpy.isnan(days[1])
    calendar.validate(node("d", "datepart", ["t"], part="year"))
    with pytest.raises(errors.ConfigError, match="part must be one of"):
        calendar.validate(node("d", "datepart", ["t"], part="week"))
    assert "timestamp" in calendar.parts
    assert len(calendar.parts) == 9
    assert pandas.Timestamp("1970-01-01") == operations.EPOCH


def test_ordered_operations_refuse_row_wise_use():
    with pytest.raises(errors.ConfigError, match="lag is an ordered"):
        sequences.Lag().apply([], {}, None)
    assert sequences.Lag.ordered
    assert not operations.Clip.ordered


@pytest.mark.parametrize("op", ["lag", "diff"])
def test_lag_and_diff_need_a_positive_distance(op):
    registry.lookup(op).validate(node("n", op, ["a"], k=1))
    with pytest.raises(errors.ConfigError, match=f"{op} k must be >= 1"):
        registry.lookup(op).validate(node("n", op, ["a"], k=0))


def test_sequential_operations_compute_within_groups():
    values = pandas.Series([1.0, 3.0, 6.0, 10.0, 20.0])
    grouped = values.groupby([0, 0, 0, 1, 1])
    lag = sequences.Lag().compute(grouped, {"k": 1}).tolist()
    assert numpy.isnan(lag[0]) and numpy.isnan(lag[3])
    assert lag[1:3] + lag[4:] == [1.0, 3.0, 10.0]
    diff = sequences.Diff().compute(grouped, {"k": 2}).tolist()
    assert diff[2] == 5.0
    assert numpy.isnan(diff[1]) and numpy.isnan(diff[4])
    rolling = sequences.Rolling().compute(grouped, {"window": 2}).tolist()
    assert rolling == [1.0, 2.0, 4.5, 10.0, 15.0]
    spread = sequences.Volatility().compute(grouped, {"window": 2}).tolist()
    assert numpy.isnan(spread[0]) and numpy.isnan(spread[3])
    assert spread[1] == pytest.approx(numpy.sqrt(2.0))
    expanding = sequences.ExpandingMean().compute(grouped, {}).tolist()
    assert expanding == [1.0, 2.0, 10 / 3, 10.0, 15.0]


def test_sequential_lookbacks():
    assert sequences.Lag().lookback({"k": 3}) == 3
    assert sequences.Diff().lookback({"k": 2}) == 2
    assert sequences.Rolling().lookback({"window": 4}) == 4
    assert sequences.Volatility().lookback({"window": 5}) == 5
    assert sequences.ExpandingMean().lookback({}) is None


def test_segregate_splits_ordered_from_row_wise_nodes():
    graph = tasks.Derivation(
        nodes=[
            node("l", "lag", ["a"]),
            node("s", "sqrt", ["a"]),
            node("r", "rolling_mean", ["a"]),
        ]
    )
    ordered, rowwise = registry.segregate(graph)
    assert [n.name for n in ordered] == ["l", "r"]
    assert [n.name for n in rowwise] == ["s"]


def test_validate_graph_accepts_a_sound_graph():
    graph = tasks.Derivation(
        nodes=[node("l", "lag", ["a"]), node("s", "sqrt", ["l"])]
    )
    registry.validate_graph(graph, {"a", "y"}, spec(time_column="t"))


@pytest.mark.parametrize(
    ("nodes", "data", "message"),
    [
        ([node("a", "sqrt", ["y"])], {}, "clashes with an existing column"),
        ([node("s", "sqrt", ["y"])], {}, "uses the target"),
        ([node("s", "sqrt", ["q"])], {}, r"unknown columns \['q'\]"),
        ([node("l", "lag", ["a"])], {}, "needs data.time_column"),
        (
            [node("s", "sqrt", ["a"]), node("l", "lag", ["s"])],
            {"time_column": "t"},
            "cannot consume row-wise",
        ),
        ([node("s", "sqrt", ["a", "a"])], {}, "takes 1..1 inputs"),
    ],
)
def test_validate_graph_rejects_unsound_graphs(nodes, data, message):
    graph = tasks.Derivation(nodes=nodes)
    with pytest.raises(errors.ConfigError, match=message):
        registry.validate_graph(graph, {"a", "y"}, spec(**data))


def test_roots_lists_raw_inputs_in_first_use_order():
    nodes = [node("p", "sum", ["b", "a"]), node("q", "sum", ["p", "a", "c"])]
    assert registry.roots(nodes, {"p", "q"}) == ["b", "a", "c"]


def test_the_transformer_learns_state_on_fit_and_reuses_it():
    nodes = (
        node("c", "clip", ["x"], lower=0.0, upper=0.5),
        node("s", "square", ["c"]),
    )
    fitted = transformer.Transformer(nodes).fit(
        pandas.DataFrame({"x": [0.0, 1.0, 2.0]})
    )
    assert fitted.states_ == {"c": (0.0, 1.0), "s": None}
    result = fitted.transform(pandas.DataFrame({"x": [5.0, -1.0]}))
    assert result["c"].tolist() == [1.0, 0.0]
    assert result["s"].tolist() == [1.0, 0.0]
    assert list(result.columns) == ["x", "c", "s"]
    assert transformer.Transformer().nodes == ()


def frame():
    return pandas.DataFrame(
        {
            "g": ["a", "b", "a", "b", "a"],
            "t": [3, 1, 1, 2, 2],
            "v": [30.0, 100.0, 10.0, 200.0, 20.0],
        }
    )


def test_temporal_compute_orders_rows_by_time_within_each_group():
    builder = history.Temporal([node("l", "lag", ["v"])], "g", "t")
    result = builder.compute(frame())
    lags = result["l"].tolist()
    assert lags[0] == 20.0
    assert lags[3] == 100.0
    assert lags[4] == 10.0
    assert numpy.isnan(lags[1]) and numpy.isnan(lags[2])


def test_temporal_without_a_group_treats_rows_as_one_series():
    builder = history.Temporal([node("l", "lag", ["v"])], None, "t")
    data = pandas.DataFrame({"t": [2, 1], "v": [5.0, 7.0]})
    lags = builder.apply(data)["l"].tolist()
    assert lags[0] == 7.0 and numpy.isnan(lags[1])
    assert builder.identify(data).tolist() == [0, 0]


def test_temporal_apply_without_nodes_returns_the_frame_itself():
    data = frame()
    assert history.Temporal([], "g", "t").apply(data) is data


def test_temporal_compute_needs_a_parseable_time_column():
    builder = history.Temporal([node("l", "lag", ["v"])], None, "t")
    with pytest.raises(errors.DataError, match="time column 't' is required"):
        builder.compute(pandas.DataFrame({"v": [1.0]}))
    with pytest.raises(errors.ConfigError, match="unparseable"):
        builder.compute(pandas.DataFrame({"t": ["x"], "v": [1.0]}))


def test_temporal_rejects_a_row_wise_node():
    builder = history.Temporal([node("s", "sqrt", ["v"])], None, "t")
    with pytest.raises(errors.ConfigError, match="sqrt is not an ordered op"):
        builder.compute(pandas.DataFrame({"t": [1], "v": [1.0]}))
    with pytest.raises(errors.ConfigError, match="not an ordered op"):
        builder.depth  # noqa: B018 -- the property raises


def test_temporal_depth_and_history_columns():
    nodes = [node("l", "lag", ["v"], k=2), node("r", "rolling_mean", ["l"])]
    builder = history.Temporal(nodes, "g", "t")
    assert builder.depth == 5
    assert builder.historical == ["g", "t", "v"]
    unlimited = history.Temporal([node("e", "expanding_mean", ["v"])], "", "t")
    assert unlimited.depth is None
    assert unlimited.historical == ["t", "v"]


def test_retain_keeps_the_latest_rows_per_entity():
    builder = history.Temporal([node("l", "lag", ["v"])], "g", "t")
    data = frame()
    data.loc[5] = ["a", "bad", 99.0]
    kept = builder.retain(data)
    assert kept.to_dict("list") == {
        "g": ["a", "b"],
        "t": [3, 2],
        "v": [30.0, 200.0],
    }
    deep = history.Temporal([node("l", "lag", ["v"], k=2)], "g", "t")
    assert deep.retain(frame())["v"].tolist() == [20.0, 30.0, 100.0, 200.0]
    unlimited = history.Temporal([node("e", "expanding_mean", ["v"])], "g", "t")
    assert len(unlimited.retain(frame())) == 5


def test_backfill_uses_only_history_earlier_than_the_batch():
    nodes = [node("l", "lag", ["v"])]
    past = pandas.DataFrame(
        {"g": ["a", "a", "c"], "t": [1, 5, 0], "v": [1.0, 5.0, 9.0]}
    )
    builder = history.Temporal(nodes, "g", "t", history=past)
    batch = pandas.DataFrame({"g": ["a", "a"], "t": [3, 4], "v": [3.0, 4.0]})
    result = builder.apply(batch)
    assert result["l"].tolist() == [1.0, 3.0]
    assert list(result.index) == [0, 1]
    with pytest.raises(errors.DataError, match="is required"):
        builder.backfill(batch.drop(columns="t"))
    with pytest.raises(errors.ConfigError, match="unparseable"):
        builder.backfill(batch.assign(t=["x", "y"]))


def test_empty_history_falls_back_to_the_batch_alone():
    nodes = [node("l", "lag", ["v"])]
    empty = pandas.DataFrame({"g": [], "t": [], "v": []})
    builder = history.Temporal(nodes, "g", "t", history=empty)
    batch = pandas.DataFrame({"g": ["a", "a"], "t": [3, 4], "v": [3.0, 4.0]})
    lags = builder.apply(batch)["l"].tolist()
    assert numpy.isnan(lags[0]) and lags[1] == 3.0
