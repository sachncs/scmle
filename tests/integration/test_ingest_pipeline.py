"""End to end: customers, orders, late events and a mutable dimension."""

import json
import sqlite3

import numpy as np
import pandas as pd

from benchmarks.qualification import ingest_worlds as worlds
from scmle import cli
from scmle.contracts import qualification
from scmle.data.admissibility import provenance
from scmle.data.ingest import asof_view
from scmle.data.ingest import budget as budgets
from scmle.data.ingest import plan as ingest_plan
from scmle.data.ingest import profile
from scmle.data.ingest import reconcile
from scmle.data.ingest import synthesize
from scmle.data.readers import connectors

SEED = 20240601


def build(path=None):
    rng = np.random.default_rng(SEED)
    db = worlds.new_database("e2e")
    customers = 40
    worlds.load(
        db,
        "customers",
        pd.DataFrame(
            {
                "customer_id": np.arange(customers),
                "segment": rng.choice(["a", "b", "c"], customers),
            }
        ),
    )
    orders = 300
    event = rng.uniform(0, 100, orders).round(1)
    worlds.load(
        db,
        "orders",
        pd.DataFrame(
            {
                "order_id": np.arange(orders),
                "customer_id": rng.integers(0, customers, orders),
                "created": event,
                "recorded": (event + rng.exponential(1.5, orders)).round(1),
                "amount": rng.integers(5, 200, orders).astype(float),
            }
        ),
    )
    items = 900
    ievent = rng.uniform(0, 100, items).round(1)
    late = rng.random(items) < 0.2
    worlds.load(
        db,
        "events",
        pd.DataFrame(
            {
                "order_id": rng.integers(0, orders, items),
                "created": ievent,
                "recorded": (
                    ievent + np.where(late, rng.uniform(10, 40, items), 0.2)
                ).round(1),
                "qty": rng.integers(1, 5, items).astype(float),
            }
        ),
    )
    return db


def frame_of(db, table, order):
    connector = connectors.DbApiConnector(
        worlds.URI, f"SELECT * FROM {table} ORDER BY {order}", db.factory
    )
    return profile.consistent_read(connector, 100).frame


def run_pipeline(db, tmp_path):
    plan = ingest_plan.IngestPlan(
        tmp_path / "ingest",
        db.factory,
        ingest_plan.IngestSettings(
            tables=("customers", "orders", "events"),
            unit="customer",
            target_table="customers",
        ),
        budgets.IngestBudget(),
    )
    summary = plan.run_all()
    note = provenance.Attestation(
        "A. Owner",
        "data owner",
        "recorded at commit",
        "log table",
        "2024-01-01",
        "scope",
        "2099-01-01",
    )
    rule = asof_view.TimeRule(asof_view.TRANSACTION, "recorded", 0.0, note)
    orders = asof_view.AsOfView(
        frame_of(db, "orders", "order_id"),
        "customer_id",
        rule,
        "created",
        "orders",
    )
    events = asof_view.AsOfView(
        frame_of(db, "events", "order_id, created"),
        "order_id",
        rule,
        "created",
        "events",
    )
    dim = asof_view.AsOfView(
        frame_of(db, "customers", "customer_id"),
        "customer_id",
        asof_view.TimeRule(asof_view.NON_VERSIONED),
        name="customers_now",
    )
    targets = pd.DataFrame(
        {"customer_id": np.arange(40), "moment": np.full(40, 70.0)}
    )
    source = synthesize.Source(orders, "customer_id")
    found = synthesize.synthesize(
        targets,
        "moment",
        [source, synthesize.Source(dim, "customer_id", synthesize.PARENT)],
        [synthesize.Nest(source, events, "order_id")],
        lag_days=1.0,
    )
    table = synthesize.feature_provenance(found, 1.0, [orders, events, dim])
    report = reconcile.reconcile_extract(
        db.source(),
        frame_of(db, "customers", "customer_id"),
        "customers",
    )
    return plan, summary, found, table, report


def test_pipeline_artifacts_tiers_and_reconciliation(tmp_path):
    db = build()
    plan, summary, found, table, report = run_pipeline(db, tmp_path)
    run_dir = tmp_path / "ingest"
    for name in (
        "ingest_catalog.json",
        "grain_report.json",
        "join_graph.json",
        "ingest_budget.json",
    ):
        assert (run_dir / name).exists()
    assert summary["verdict"] == "match"
    accepted = {
        (link.child, link.child_columns[0], link.parent)
        for link in plan.graph.accepted()
    }
    assert ("orders", "customer_id", "customers") in accepted
    assert ("events", "order_id", "orders") in accepted
    assert found.declared_family_size == len(found.frame.columns) > 20
    assert any(m["kind"] == "nested" and m["kept"] for m in found.manifest)
    tiers = table.tiers()
    kept = {m["name"]: m for m in found.kept()}
    for name, tier in tiers.items():
        if kept[name]["tables"] == ["customers_now"]:
            assert tier is qualification.Tier.UNVERIFIED
        else:
            assert tier is qualification.Tier.ATTESTED
    assert report.passed
    budget = json.loads((run_dir / "ingest_budget.json").read_text())
    assert budget["partial"] is False


def test_pipeline_is_reproducible(tmp_path):
    first = run_pipeline(build(), tmp_path / "a")[2]
    second = run_pipeline(build(), tmp_path / "b")[2]
    pd.testing.assert_frame_equal(first.frame, second.frame)
    assert first.to_dict() == second.to_dict()


def test_features_count_only_orders_recorded_before_the_cutoff(tmp_path):
    db = build()
    found = run_pipeline(db, tmp_path)[2]
    orders = frame_of(db, "orders", "order_id")
    known = orders.loc[
        (orders["recorded"] <= 69.0) & (orders["created"] <= 69.0)
    ]
    expected = (
        known.groupby("customer_id").size().reindex(range(40), fill_value=0)
    )
    assert (orders["recorded"] > 69.0).any()
    got = found.frame["orders__count__90d"].to_numpy()
    assert got.tolist() == expected.astype(float).tolist()
    by_event_time = orders.loc[orders["created"] <= 69.0]
    naive = (
        by_event_time.groupby("customer_id")
        .size()
        .reindex(range(40), fill_value=0)
    )
    assert naive.sum() > expected.sum()


def test_cli_ingest_runs_and_blocks_on_a_coarser_unit(tmp_path, capsys):
    path = tmp_path / "w.db"
    keeper = sqlite3.connect(path)
    pd.DataFrame({"customer_id": range(10), "signup_month": [1] * 10}).to_sql(
        "customers", keeper, index=False
    )
    pd.DataFrame(
        {"order_id": range(20), "customer_id": [k % 10 for k in range(20)]}
    ).to_sql("orders", keeper, index=False)
    keeper.commit()
    keeper.close()
    out = tmp_path / "out"
    code = cli.main(
        [
            "ingest",
            "--sqlite",
            str(path),
            "--tables",
            "customers,orders",
            "--unit",
            "customer",
            "--target-table",
            "customers",
            "--output",
            str(out),
        ]
    )
    assert code == 0
    assert json.loads(capsys.readouterr().out)["verdict"] == "match"
    assert (out / "join_graph.json").exists()
    code = cli.main(
        [
            "ingest",
            "--sqlite",
            str(path),
            "--tables",
            "customers,orders",
            "--unit",
            "customer-month",
            "--target-table",
            "customers",
            "--output",
            str(out),
            "--max-queries",
            "1000",
        ]
    )
    assert code == cli.EXIT_BLOCKED


def test_cli_source_is_read_only(tmp_path):
    path = tmp_path / "w.db"
    keeper = sqlite3.connect(path)
    keeper.execute("CREATE TABLE t (a INTEGER)")
    keeper.execute("INSERT INTO t VALUES (1)")
    keeper.commit()
    keeper.close()
    connection = ingest_plan.sqlite_factory(path)()
    try:
        connection.execute("INSERT INTO t VALUES (2)")
    except sqlite3.OperationalError:
        refused = True
    else:
        refused = False
    assert refused
