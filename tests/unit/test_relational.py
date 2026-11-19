"""The relational engine: stores, declarations and point-in-time features."""

import math
import sqlite3

import numpy
import pandas
import pytest

from benchmarks.qualification import ingest_naive
from scmle.assurance.ingest import asof_view
from scmle.assurance.ingest import synthesize
from scmle.contracts import errors
from scmle.data import relational
from scmle.data.relational import engine
from scmle.data.relational import schema
from scmle.data.relational import stores

ORDERS = pandas.DataFrame(
    {
        "order_id": [1, 2, 3, 4],
        "customer_id": [1, 1, 1, 2],
        "created": [1.0, 5.0, 10.0, 3.0],
        "recorded": [1.0, 6.0, 12.0, numpy.nan],
        "amount": [10.0, 20.0, 30.0, 7.0],
        "channel": ["web", "shop", "web", "web"],
    }
)
CUSTOMERS = pandas.DataFrame(
    {
        "customer_id": [1, 1, 2],
        "since": [0.0, 8.0, 0.0],
        "segment": ["a", "b", "c"],
    }
)


def declared(delay=0.5):
    return schema.Schema(
        (
            schema.Table("orders", "created", "recorded", delay),
            schema.Table("customers", "since"),
        ),
        (
            schema.Link("orders", "customer_id", "customer_id"),
            schema.Link("customers", "customer_id", "customer_id", schema.ONE),
        ),
    )


def built(delay=0.5):
    return relational.Engine(
        declared(delay), {"orders": ORDERS, "customers": CUSTOMERS}
    )


def asked(*pairs):
    return pandas.DataFrame(
        {
            "customer_id": [p[0] for p in pairs],
            "moment": [float(p[1]) for p in pairs],
        }
    )


def test_a_row_recorded_after_the_cutoff_is_not_visible():
    aggregates = relational.grid(
        "orders", ["amount"], windows=(None,), primitives=engine.PRIMITIVES
    )
    frame = built().features(asked((1, 10)), "moment", aggregates)
    assert frame.iloc[0].to_dict() == {
        "customers__since": 8.0,
        "customers__segment": "b",
        "orders__count__*__all": 2.0,
        "orders__sum__amount__all": 30.0,
        "orders__mean__amount__all": 15.0,
        "orders__min__amount__all": 10.0,
        "orders__max__amount__all": 20.0,
        "orders__last__amount__all": 20.0,
        "orders__trend__amount__all": 2.5,
    }


def test_every_row_is_visible_once_recorded():
    aggregates = relational.grid("orders", ["amount"], windows=(None,))
    row = built().features(asked((1, 12)), "moment", aggregates).iloc[0]
    assert row["orders__count__*__all"] == 3.0
    assert row["orders__last__amount__all"] == 30.0
    assert row["orders__trend__amount__all"] == pytest.approx(810 / 366)


def test_a_window_keeps_events_after_its_open_edge():
    aggregates = (
        relational.Aggregate("orders", "count", None, 5.0),
        relational.Aggregate("orders", "count", None, 5.5),
        relational.Aggregate("orders", "trend", "amount", 9.5),
        relational.Aggregate("orders", "trend", "amount", 5.0),
    )
    row = built().features(asked((1, 10)), "moment", aggregates).iloc[0]
    assert row["orders__count__*__5d"] == 0.0
    assert row["orders__count__*__5.5d"] == 1.0
    assert row["orders__trend__amount__9.5d"] == 2.5
    assert math.isnan(row["orders__trend__amount__5d"])


def test_the_lag_moves_the_cutoff_back():
    count = (relational.Aggregate("orders", "count"),)
    late = built().features(asked((1, 13)), "moment", count, lag=1.0)
    assert late.iloc[0]["orders__count__*__all"] == 3.0
    early = built().features(asked((1, 13)), "moment", count, lag=1.5)
    assert early.iloc[0]["orders__count__*__all"] == 2.0


def test_the_delay_stands_in_for_a_missing_recorded_time():
    count = (relational.Aggregate("orders", "count"),)
    frame = built(delay=0.5).features(
        asked((2, 3.4), (2, 3.5)), "moment", count
    )
    assert frame["orders__count__*__all"].tolist() == [0.0, 1.0]


def test_targets_without_rows_get_zero_counts_and_missing_statistics():
    aggregates = relational.grid(
        "orders", ["amount"], primitives=("count", "sum", "mean", "last")
    )[:4]
    frame = built().features(asked((2, 2), (9, 50)), "moment", aggregates)
    assert frame["orders__count__*__7d"].tolist() == [0.0, 0.0]
    assert frame["orders__sum__amount__7d"].tolist() == [0.0, 0.0]
    assert frame["orders__mean__amount__7d"].isna().all()
    assert frame["orders__last__amount__7d"].isna().all()


def test_a_target_without_a_moment_is_refused():
    count = (relational.Aggregate("orders", "count"),)
    targets = pandas.DataFrame(
        {"customer_id": [1, 1, 2], "moment": [10.0, None, None]}
    )
    with pytest.raises(errors.InputError, match="'moment' has 2 missing"):
        built().features(targets, "moment", count)
    with pytest.raises(errors.InputError, match="'moment' has 2 missing"):
        built().features(targets, "moment")


def test_columns_agree_within_the_tolerance_and_on_missing_values():
    numbers = pandas.Series([1.0, numpy.nan, 3.0])
    assert engine.agree(numbers, numbers * (1 + 1e-10), 1e-8)
    assert not engine.agree(numbers, numbers * (1 + 1e-6), 1e-8)
    assert not engine.agree(numbers, numbers.fillna(0.0), 1e-8)
    words = pandas.Series(["a", None, "c"], dtype=object)
    assert engine.agree(words, words.copy(), 1e-8)
    assert not engine.agree(words, pandas.Series(["a", None, "d"]), 1e-8)
    assert not engine.agree(words, pandas.Series(["a", "b", "c"]), 1e-8)


def test_a_many_to_one_join_takes_the_version_visible_at_the_cutoff():
    frame = built().features(asked((1, 7.9), (1, 8), (2, -1)), "moment")
    assert frame["customers__segment"].tolist()[:2] == ["a", "b"]
    assert pandas.isna(frame["customers__segment"].iloc[2])
    assert frame.index.tolist() == [0, 1, 2]


def test_the_output_keeps_the_index_of_the_targets():
    targets = asked((1, 10), (2, 9)).set_axis([30, 10])
    count = (relational.Aggregate("orders", "count"),)
    frame = built().features(targets, "moment", count)
    assert frame.index.tolist() == [30, 10]
    assert frame["orders__count__*__all"].tolist() == [2.0, 1.0]


def test_joins_chain_through_an_earlier_join():
    lines = pandas.DataFrame({"order_id": [3, 4, 4], "units": [1, 2, 5]})
    chained = schema.Schema(
        (
            schema.Table("orders", "created", "recorded"),
            schema.Table("customers", "since"),
        ),
        (
            schema.Link("orders", "order_id", "order_id", schema.ONE),
            schema.Link(
                "customers",
                "customer_id",
                "orders__customer_id",
                schema.ONE,
            ),
        ),
    )
    found = relational.Engine(
        chained, {"orders": ORDERS, "customers": CUSTOMERS}
    ).features(lines.assign(moment=[12.0, 12.0, 2.0]), "moment")
    assert found["customers__segment"].tolist()[:2] == ["b", "c"]
    assert pandas.isna(found["customers__segment"].iloc[2])
    assert found["orders__amount"].tolist()[:2] == [30.0, 7.0]


def test_a_tie_goes_to_the_later_stored_row():
    rows = pandas.DataFrame(
        {"k": [1, 1, 1], "t": [2.0, 2.0, 1.0], "v": [5.0, 6.0, 9.0]}
    )
    layout = schema.Schema(
        (schema.Table("r", "t"),), (schema.Link("r", "k", "k"),)
    )
    last = (relational.Aggregate("r", "last", "v"),)
    frame = relational.Engine(layout, {"r": rows}).features(
        pandas.DataFrame({"k": [1], "m": [3.0]}), "m", last
    )
    assert frame.iloc[0, 0] == 6.0


def test_a_row_without_an_event_time_is_never_visible():
    rows = pandas.DataFrame({"k": [1, 1], "t": [numpy.nan, 1.0], "v": [1, 2]})
    layout = schema.Schema(
        (schema.Table("r", "t"),), (schema.Link("r", "k", "k"),)
    )
    count = (relational.Aggregate("r", "count"),)
    frame = relational.Engine(layout, {"r": rows}).features(
        pandas.DataFrame({"k": [1], "m": [1e9]}), "m", count
    )
    assert frame.iloc[0, 0] == 1.0


def test_text_and_datetime_times_agree():
    rows = pandas.DataFrame(
        {"k": [1, 1], "t": ["2024-01-01", "2024-01-03 12:00:00"], "v": [1, 2]}
    )
    layout = schema.Schema(
        (schema.Table("r", "t"),), (schema.Link("r", "k", "k"),)
    )
    count = (relational.Aggregate("r", "count"),)
    moments = pandas.to_datetime(["2024-01-03 11:59:59", "2024-01-03 12:00:00"])
    frame = relational.Engine(layout, {"r": rows}).features(
        pandas.DataFrame({"k": [1, 1], "m": moments}), "m", count
    )
    assert frame.iloc[:, 0].tolist() == [1.0, 2.0]


def test_unreadable_time_text_is_refused():
    with pytest.raises(errors.InputError, match="not a time"):
        engine.days(pandas.Series(["2024-01-01", "soon"], name="t"))


def test_a_declared_many_to_one_table_must_not_repeat_a_version():
    twice = pandas.concat([CUSTOMERS, CUSTOMERS.iloc[[0]]])
    with pytest.raises(errors.SchemaError, match="holds 1 rows"):
        relational.Engine(declared(), {"orders": ORDERS, "customers": twice})


def test_a_declared_column_must_exist():
    with pytest.raises(errors.SchemaError, match=r"\['recorded'\]"):
        relational.Engine(
            declared(),
            {"orders": ORDERS.drop(columns="recorded"), "customers": CUSTOMERS},
        )


def test_an_aggregate_needs_a_one_to_many_link():
    wrong = (relational.Aggregate("customers", "count"),)
    with pytest.raises(errors.ConfigError, match="one_to_many"):
        built().features(asked((1, 1)), "moment", wrong)


def test_a_statistic_of_text_is_refused():
    wrong = (relational.Aggregate("orders", "mean", "channel"),)
    with pytest.raises(errors.InputError, match="needs numbers"):
        built().features(asked((1, 20)), "moment", wrong)


def test_last_of_text_is_the_text():
    last = (relational.Aggregate("orders", "last", "channel"),)
    frame = built().features(asked((1, 20)), "moment", last)
    assert frame["orders__last__channel__all"].tolist() == ["web"]


def test_keys_of_different_kinds_are_refused():
    targets = asked((1, 1)).assign(customer_id=["1"])
    with pytest.raises(errors.InputError, match="matches"):
        built().features(targets, "moment")


def test_a_reserved_column_is_refused():
    rows = ORDERS.assign(**{engine.ROW: 0})
    with pytest.raises(errors.InputError, match="reserved"):
        relational.Engine(declared(), {"orders": rows, "customers": CUSTOMERS})


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (("orders", "median", "amount"), "unknown primitive"),
        (("orders", "mean"), "takes a value"),
        (("orders", "count", "amount"), "takes no value"),
        (("orders", "mean", "amount", 0.0), "not positive"),
    ],
)
def test_a_malformed_aggregate_is_refused(arguments, message):
    with pytest.raises(errors.ConfigError, match=message):
        relational.Aggregate(*arguments)


@pytest.mark.parametrize(
    ("document", "message"),
    [
        ({"tables": [{"name": "a"}, {"name": "a"}]}, "table names repeat"),
        ({"links": [{"table": "x", "key": "k", "target": "k"}]}, "undeclared"),
        ({"tables": [{"name": "a", "known": "r"}]}, "without an event"),
        ({"tables": [{"name": "a", "delay": -1}]}, "non-negative"),
        ({"tables": [{"name": "a", "delay": "1"}]}, "number"),
        ({"tables": [{"name": ""}]}, "non-empty"),
        ({"tables": {}}, "lists"),
        ({"tables": [{"name": "a", "event": 5}]}, "table event"),
        ({"tables": [{"name": "a", "event": "t", "known": 1}]}, "table known"),
        ({"tables": [{"name": "a", "event": ""}]}, "table event"),
        (
            {
                "tables": [{"name": "a"}],
                "links": [{"table": "a", "key": "k", "target": "k", "name": 2}],
            },
            "link name",
        ),
        (
            {
                "tables": [{"name": "a"}],
                "links": [
                    {"table": "a", "key": "k", "target": "k", "cardinality": 1}
                ],
            },
            "link cardinality",
        ),
        (
            {
                "tables": [{"name": "a"}],
                "links": [
                    {
                        "table": "a",
                        "key": "k",
                        "target": "k",
                        "cardinality": "1",
                    }
                ],
            },
            "cardinality",
        ),
    ],
)
def test_a_malformed_schema_is_refused(document, message):
    with pytest.raises(errors.ConfigError, match=message):
        relational.parse(document)


def test_a_schema_document_round_trips_its_declarations():
    found = relational.parse(
        {
            "tables": [{"name": "orders", "event": "created", "delay": 2}],
            "links": [
                {
                    "table": "orders",
                    "key": "customer_id",
                    "target": "id",
                    "cardinality": "many_to_one",
                    "name": "latest",
                }
            ],
        }
    )
    assert found == schema.Schema(
        (schema.Table("orders", "created", None, 2.0),),
        (schema.Link("orders", "customer_id", "id", schema.ONE, "latest"),),
    )


def synthetic(seed):
    rng = numpy.random.default_rng(seed)
    size = 400
    event = rng.integers(0, 60, size).astype(float)
    recorded = event + rng.choice([0.0, 0.5, 3.0, numpy.nan], size)
    rows = pandas.DataFrame(
        {
            "k": rng.integers(0, 12, size),
            "t": event,
            "r": recorded,
            "v": numpy.where(
                rng.random(size) < 0.1, numpy.nan, rng.normal(size=size)
            ),
        }
    )
    targets = pandas.DataFrame(
        {"k": rng.integers(0, 14, 60), "m": rng.integers(-2, 66, 60) * 1.0}
    )
    return rows, targets


NAIVE = {"trend": "slope"}


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("primitive", engine.PRIMITIVES)
@pytest.mark.parametrize("window", [None, 7.0, 30.0])
def test_every_primitive_matches_the_naive_reference_and_the_synthesizer(
    seed, primitive, window
):
    rows, targets = synthetic(seed)
    delay, lag = 1.0, 0.5
    value = None if primitive == "count" else "v"
    layout = schema.Schema(
        (schema.Table("r", "t", "r", delay),), (schema.Link("r", "k", "k"),)
    )
    aggregate = relational.Aggregate("r", primitive, value, window)
    got = (
        relational.Engine(layout, {"r": rows})
        .features(targets, "m", (aggregate,), lag)[aggregate.name]
        .to_numpy(dtype=float)
    )
    want = ingest_naive.aggregate_feature(
        targets, "k", "m", rows, "k", "t", "r", value,
        NAIVE.get(primitive, primitive), window, lag, delay,
    )  # fmt: skip
    numpy.testing.assert_allclose(got, want, rtol=1e-8, atol=1e-8)
    rule = asof_view.Timing(asof_view.TRANSACTION, "r", delay)
    source = synthesize.Source(asof_view.View(rows, "k", rule, "t", "r"), "k")
    feature = synthesize.Feature(
        "f", "aggregate", 0, NAIVE.get(primitive, primitive), value, window
    )
    other = synthesize.evaluate(feature, targets, "m", [source], [], lag)
    numpy.testing.assert_allclose(
        got, other.to_numpy(dtype=float), rtol=1e-8, atol=1e-8
    )


def database(path):
    with sqlite3.connect(path) as connection:
        ORDERS.to_sql("orders", connection, index=False)
        CUSTOMERS.to_sql("customers", connection, index=False)
    connection.close()
    return path


def folders(tmp_path):
    base = database(tmp_path / "w.sqlite")
    parquet = tmp_path / "parquet"
    (parquet / "orders").mkdir(parents=True)
    ORDERS.iloc[:2].to_parquet(parquet / "orders" / "part-0.parquet")
    ORDERS.iloc[2:].to_parquet(parquet / "orders" / "part-1.parquet")
    CUSTOMERS.to_parquet(parquet / "customers.parquet")
    csv = tmp_path / "csv"
    csv.mkdir()
    ORDERS.to_csv(csv / "orders.csv", index=False)
    CUSTOMERS.to_csv(csv / "customers.csv", index=False)
    return base, parquet, csv


def test_each_kind_of_store_is_recognised_and_reads_the_same_tables(tmp_path):
    found = [stores.locate(p) for p in folders(tmp_path)]
    assert [type(s) for s in found] == [
        stores.Database,
        stores.Parquet,
        stores.Csv,
    ]
    for store in found:
        assert store.names() == ("customers", "orders")
        pandas.testing.assert_frame_equal(
            store.read("orders"), ORDERS, check_dtype=False
        )


def test_each_store_gives_the_same_features(tmp_path):
    aggregates = relational.grid("orders", ["amount"], windows=(None, 7.0))
    targets = asked((1, 10), (1, 12), (2, 4), (3, 4))
    frames = [
        relational.Engine.open(declared(), stores.locate(p)).features(
            targets, "moment", aggregates
        )
        for p in folders(tmp_path)
    ]
    assert frames[0].shape == (4, 2 + len(aggregates))
    for frame in frames[1:]:
        pandas.testing.assert_frame_equal(frame, frames[0])


def test_a_folder_is_mirrored_into_sqlite_with_every_row(tmp_path):
    ignored, parquet, csv = folders(tmp_path)
    for folder in (parquet, csv):
        connect = stores.locate(folder).factory(
            tmp_path / "mirrors" / folder.name
        )
        connection = connect()
        try:
            count = connection.execute("SELECT COUNT(*) FROM orders").fetchone()
            with pytest.raises(sqlite3.OperationalError, match="readonly"):
                connection.execute("DELETE FROM orders")
        finally:
            connection.close()
        assert count == (4,)


def test_a_database_needs_no_mirror(tmp_path):
    base = database(tmp_path / "w.sqlite")
    connect = stores.Database(base).factory(tmp_path / "unused")
    connection = connect()
    try:
        assert connection.execute(
            "SELECT COUNT(*) FROM customers"
        ).fetchone() == (3,)
    finally:
        connection.close()
    assert not (tmp_path / "unused").exists()


def test_a_directory_named_with_the_format_suffix_holds_one_table(tmp_path):
    folder = tmp_path / "spark"
    (folder / "orders.parquet").mkdir(parents=True)
    ORDERS.iloc[:2].to_parquet(folder / "orders.parquet" / "part-0.parquet")
    ORDERS.iloc[2:].to_parquet(folder / "orders.parquet" / "part-1.parquet")
    CUSTOMERS.to_parquet(folder / "customers.parquet")
    store = stores.locate(folder)
    assert type(store) is stores.Parquet
    assert store.names() == ("customers", "orders")
    pandas.testing.assert_frame_equal(store.read("orders"), ORDERS)


def test_a_table_held_twice_is_refused(tmp_path):
    folder = tmp_path / "twice"
    (folder / "orders").mkdir(parents=True)
    ORDERS.to_parquet(folder / "orders" / "part-0.parquet")
    ORDERS.to_parquet(folder / "orders.parquet")
    assert stores.Parquet(folder).names() == ("orders",)
    with pytest.raises(errors.DataError, match="held 2 times"):
        stores.Parquet(folder).read("orders")


def test_parts_with_different_columns_are_refused(tmp_path):
    folder = tmp_path / "t"
    (folder / "orders").mkdir(parents=True)
    ORDERS.to_parquet(folder / "orders" / "a.parquet")
    ORDERS.drop(columns="amount").to_parquet(folder / "orders" / "b.parquet")
    with pytest.raises(errors.DataError, match="share their columns"):
        stores.Parquet(folder).read("orders")


def test_stores_refuse_what_they_cannot_read(tmp_path):
    base, parquet, csv = folders(tmp_path)
    with pytest.raises(errors.ConfigError, match="holds no table 'items'"):
        stores.locate(base).read("items")
    with pytest.raises(errors.ConfigError, match="unknown store kind"):
        stores.locate(base, "xml")
    with pytest.raises(errors.ConfigError, match="no table store"):
        stores.locate(tmp_path / "absent")
    with pytest.raises(errors.ConfigError, match="not a SQLite database"):
        stores.locate(csv / "orders.csv")
    with pytest.raises(errors.ConfigError, match="not a folder"):
        stores.Csv(base)
    ORDERS.to_csv(parquet / "extra.csv", index=False)
    with pytest.raises(errors.ConfigError, match="both"):
        stores.locate(parquet)
    (tmp_path / "empty").mkdir()
    with pytest.raises(errors.ConfigError, match="neither"):
        stores.locate(tmp_path / "empty")
    assert stores.locate(parquet, "csv").names() == ("extra",)
