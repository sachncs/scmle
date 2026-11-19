"""The ingest stage reads a SQLite file, a Parquet folder or a CSV set."""

import dataclasses
import json
import sqlite3

import numpy
import pandas
import pytest

from scmle.assurance.charter import ingest
from scmle.assurance.ingest import asof_view
from scmle.assurance.ingest import synthesize
from scmle.contracts import errors
from scmle.data.relational import engine
from scmle.data.relational import schema
from scmle.pipeline import ingest_stage

CUSTOMERS = 120
# The synthesizer's value primitives the engine also computes.
PRIMITIVES = ("sum", "mean", "min", "max", "slope", "last")


def tables():
    rng = numpy.random.default_rng(0)
    customers = pandas.DataFrame(
        {
            "customer_id": numpy.arange(CUSTOMERS),
            "segment": ["ab"[i % 2] for i in range(CUSTOMERS)],
        }
    )
    rows = []
    for customer in range(CUSTOMERS):
        for ignored in range(int(rng.integers(1, 9))):
            made = float(rng.integers(0, 100))
            rows.append(
                (
                    len(rows),
                    customer,
                    made,
                    made + float(rng.integers(0, 3)),
                    float(rng.integers(5, 100)),
                )
            )
    orders = pandas.DataFrame(
        rows,
        columns=["order_id", "customer_id", "created", "recorded", "amount"],
    )
    return {"customers": customers, "orders": orders}


def places(tmp_path):
    """The same tables as a database, a Parquet folder and a CSV set."""
    database = tmp_path / "w.sqlite"
    with sqlite3.connect(database) as connection:
        for name, frame in tables().items():
            frame.to_sql(name, connection, index=False)
    connection.close()
    found = {"database": str(database)}
    for kind in ("parquet", "csv"):
        folder = tmp_path / kind
        folder.mkdir()
        for name, frame in tables().items():
            path = folder / f"{name}.{kind}"
            if kind == "parquet":
                frame.to_parquet(path)
            else:
                frame.to_csv(path, index=False)
        found[kind] = str(folder)
    return found


def plan(place, kind, **orders):
    where = (
        {"database": place}
        if kind == "database"
        else {"store": {"kind": kind, "path": place}}
    )
    view = {
        "name": "orders",
        "table": "orders",
        "order_by": "order_id",
        "key": "customer_id",
        "target_key": "customer_id",
        "event_time": "created",
        "rule": {
            "mode": "transaction",
            "known_column": "recorded",
            "delta_rec_days": 0.0,
        },
        **orders,
    }
    segment = {
        "name": "customers",
        "table": "customers",
        "order_by": "customer_id",
        "key": "customer_id",
        "cardinality": "many_to_one",
        "permissive": True,
        "rule": {"mode": "non_versioned"},
    }
    return {
        **where,
        "key": "customer_id",
        "moment": "moment",
        "tables": ["customers", "orders"],
        "unit": "customer",
        "target_table": "customers",
        "views": [view, segment],
    }


def target():
    return pandas.DataFrame(
        {"customer_id": numpy.arange(CUSTOMERS), "moment": 60.0}
    )


def run(document, tmp_path, name):
    spec = ingest.Ingestion(features=40, windows=(30.0, 90.0), lag=1.0)
    return ingest_stage.run(document, spec, target(), tmp_path / name)


def test_every_store_gives_the_same_features(tmp_path):
    found = places(tmp_path)
    outputs = {
        kind: run(plan(place, kind), tmp_path, kind)
        for kind, place in found.items()
    }
    first = outputs["database"].features
    assert "orders__count__30d" in first.columns
    assert "customers__lookup__segment" in first.columns
    assert first["orders__count__30d"].sum() > 0
    for kind in ("parquet", "csv"):
        pandas.testing.assert_frame_equal(outputs[kind].features, first)
        assert outputs[kind].declared == outputs["database"].declared


def test_a_cardinality_sets_the_role(tmp_path):
    found = places(tmp_path)
    place = found["parquet"]
    parent = run(plan(place, "parquet", role="parent"), tmp_path, "parent")
    child = run(
        plan(place, "parquet", cardinality="one_to_many"), tmp_path, "child"
    )
    assert "orders__count__30d" not in parent.features.columns
    assert "orders__lookup__amount" in parent.features.columns
    assert "orders__count__30d" in child.features.columns
    assert "orders__lookup__amount" not in child.features.columns
    # The customers view states no role: its many_to_one cardinality alone
    # makes it a parent, so it gives a lookup and no aggregates.
    for output in (parent, child):
        columns = output.features.columns
        assert "customers__lookup__segment" in columns
        assert not any(c.startswith("customers__count") for c in columns)
    contradicted = plan(place, "parquet", cardinality="one_to_many")
    contradicted["views"][0]["role"] = "parent"
    with pytest.raises(errors.ConfigError, match="has role 'parent'"):
        run(contradicted, tmp_path, "contradicted")


def watched(monkeypatch):
    """Records each call of the engine's features."""
    calls = []
    original = engine.Engine.features

    def features(self, targets, moment, aggregates=(), lag=0.0):
        calls.append((len(targets), moment, len(aggregates), lag))
        return original(self, targets, moment, aggregates, lag)

    monkeypatch.setattr(engine.Engine, "features", features)
    return calls


def test_the_engine_recomputes_every_feature_it_can(tmp_path, monkeypatch):
    calls = watched(monkeypatch)
    found = places(tmp_path)
    output = run(plan(found["csv"], "csv"), tmp_path, "run")
    assert len(calls) == 1
    assert calls[0][:2] == (CUSTOMERS, "moment")
    assert calls[0][3] == 1.0
    report = output.summary["engine"]
    unchecked = report["unchecked"]
    assert report["checked"] == calls[0][2]
    assert report["checked"] + len(unchecked) == output.features.shape[1]
    checked = set(output.features.columns) - set(unchecked)
    shared = ("count", *(f"{p}__amount" for p in PRIMITIVES))
    assert checked == {"customers__lookup__segment"} | {
        f"orders__{s}__{w}" for s in shared for w in ("30d", "90d")
    }
    assert report["checked"] == 15
    assert unchecked == {
        "orders__time_since_last__30d": "the engine has no time_since_last",
        "orders__time_since_last__90d": "the engine has no time_since_last",
        "orders__std__amount__30d": "the engine has no std",
        "orders__std__amount__90d": "the engine has no std",
        "orders__nunique__amount__90d": "the engine has no nunique",
    }
    assert report["tolerance"] == ingest_stage.AGREEMENT


def test_each_view_timing_maps_to_its_engine_table():
    rows = tables()["orders"]
    rules = {
        "transaction": asof_view.Timing("transaction", "recorded", 0.5),
        "event_delay": asof_view.Timing("event_delay", None, 2.0),
        "non_versioned": asof_view.Timing("non_versioned"),
    }
    found = {
        (mode, event): ingest_stage.timing(
            asof_view.View(rows, "customer_id", rule, event, "o", True)
        )
        for mode, rule in rules.items()
        for event in ("created", None)
        if mode == "non_versioned" or event
    }
    assert found == {
        ("transaction", "created"): schema.Table(
            "o", "created", "recorded", 0.5
        ),
        ("event_delay", "created"): schema.Table("o", "created", None, 2.0),
        ("non_versioned", "created"): None,
        ("non_versioned", None): schema.Table("o"),
    }
    unread = asof_view.Timing("transaction", "recorded")
    view = asof_view.View(rows, "customer_id", unread, "created", "o")
    assert ingest_stage.timing(view) is None
    feature = synthesize.Feature("f", "aggregate", 0, "sum", "amount", 7.0)
    assert ingest_stage.reason(feature, [None]) == (
        "the engine has no time rule like its view's"
    )
    nested = dataclasses.replace(feature, kind="nested", nest=0, inner="sum")
    assert ingest_stage.reason(nested, [schema.Table("o")]) == (
        "the engine has no depth-2 nest"
    )
    assert ingest_stage.reason(feature, [schema.Table("o")]) is None


def test_a_feature_the_engine_disagrees_with_stops_the_run(
    tmp_path, monkeypatch
):
    original = engine.Engine.features

    def features(self, targets, moment, aggregates=(), lag=0.0):
        found = original(self, targets, moment, aggregates, lag)
        name = "orders__sum__amount__30d"
        found[name] = found[name] * (1.0 + 1e-6)
        return found

    monkeypatch.setattr(engine.Engine, "features", features)
    found = places(tmp_path)
    with pytest.raises(errors.DataError, match="orders__sum__amount__30d"):
        run(plan(found["database"], "database"), tmp_path, "run")


def test_a_run_table_without_a_moment_is_refused(tmp_path, monkeypatch):
    found = places(tmp_path)
    frame = target()
    frame.loc[3, "moment"] = numpy.nan
    spec = ingest.Ingestion(features=40, windows=(30.0, 90.0), lag=1.0)
    with pytest.raises(errors.InputError, match="1 missing moments"):
        ingest_stage.run(
            plan(found["database"], "database"), spec, frame, tmp_path / "r"
        )


def test_a_many_to_one_view_whose_key_repeats_stops_the_run(tmp_path):
    found = places(tmp_path)
    document = plan(found["csv"], "csv", cardinality="many_to_one")
    with pytest.raises(errors.SchemaError, match="'orders' is declared"):
        run(document, tmp_path, "run")


def test_a_cardinality_that_contradicts_the_role_is_refused(tmp_path):
    found = places(tmp_path)
    document = plan(
        found["database"], "database", cardinality="many_to_one", role="child"
    )
    with pytest.raises(errors.ConfigError, match="has role 'child'"):
        run(document, tmp_path, "run")
    document = plan(found["database"], "database", cardinality="one")
    with pytest.raises(errors.ConfigError, match="cardinality 'one'"):
        run(document, tmp_path, "run")


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"store": {"kind": "xml", "path": "."}}, "unknown store kind"),
        ({"store": {"kind": "csv"}}, "needs a `path`"),
        ({"store": {"path": "absent-folder"}}, "no table store"),
    ],
)
def test_a_malformed_store_is_refused(tmp_path, change, message):
    document = {**plan("unused", "database"), **change}
    del document["database"]
    with pytest.raises(errors.ConfigError, match=message):
        run(document, tmp_path, "run")


def test_a_plan_file_names_exactly_one_store(tmp_path):
    found = places(tmp_path)
    both = {**plan(found["database"], "database"), "store": {"path": "."}}
    for document, ok in ((both, False), (plan(found["csv"], "csv"), True)):
        path = tmp_path / "plan.json"
        path.write_text(json.dumps(document))
        if ok:
            assert ingest_stage.recipe(str(path))["store"]["kind"] == "csv"
            continue
        with pytest.raises(errors.ConfigError, match="one of `database`"):
            ingest_stage.recipe(str(path))
