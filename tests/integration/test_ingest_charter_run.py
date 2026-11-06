"""A charter run that synthesizes its features from a database (F9)."""

import json
import pathlib
import sqlite3

import numpy as np
import pandas as pd
import pytest
import support

from scmle.contracts import errors
from scmle.data.charter import ingest
from scmle.data.charter import sealing
from scmle.data.ingest import declared
from scmle.data.ledger import ledger as ledger_module
from tests.unit.test_charter_contracts import make_charter

KEY = "ingest-key"
CUSTOMERS = 500


def database(path):
    rng = np.random.default_rng(0)
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, "
            "segment TEXT)"
        )
        connection.executemany(
            "INSERT INTO customers VALUES (?, ?)",
            [(i, "ab"[i % 2]) for i in range(CUSTOMERS)],
        )
        connection.execute(
            "CREATE TABLE orders (order_id INTEGER PRIMARY KEY, "
            "customer_id INTEGER, created REAL, recorded REAL, amount REAL)"
        )
        rate = rng.uniform(0.5, 8.0, CUSTOMERS)
        rows = []
        for customer in range(CUSTOMERS):
            for ignored in range(rng.poisson(rate[customer])):
                made = float(rng.uniform(0, 100))
                rows.append(
                    (
                        len(rows),
                        customer,
                        made,
                        made + float(rng.exponential(1.0)),
                        float(rng.integers(5, 100)),
                    )
                )
        connection.executemany(
            "INSERT INTO orders VALUES (?, ?, ?, ?, ?)", rows
        )
    return rate


def plan_file(tmp_path, unit="customer"):
    note = {
        "name_or_id": "A. Owner",
        "role": "data owner",
        "statement": "recorded is set at commit",
        "evidence": "commit log table",
        "date": "2024-01-01",
        "scope": "orders",
        "expires": "2099-01-01",
    }
    plan = {
        "database": str(tmp_path / "w.sqlite"),
        "key": "customer_id",
        "moment": "moment",
        "tables": ["customers", "orders"],
        "unit": unit,
        "target_table": "customers",
        "views": [
            {
                "name": "orders",
                "table": "orders",
                "order_by": "order_id",
                "key": "customer_id",
                "target_key": "customer_id",
                "role": "child",
                "event_time": "created",
                "rule": {
                    "mode": "transaction",
                    "known_column": "recorded",
                    "delta_rec_days": 0.0,
                    "attestation": note,
                },
            }
        ],
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    return path


def charter_file(tmp_path, block=True, entity=None, reconcile=None):
    spec = ingest.IngestSpec(
        feature_budget=40,
        windows=(30.0, 90.0),
        lag=1.0,
        entity=entity,
        reconcile=reconcile,
    )
    path = tmp_path / "charter.json"
    sealing.write_charter(
        path,
        make_charter(min_effect=0.5, k_max=3, ingest=spec if block else None),
        signing_key=KEY,
    )
    return path


def table(rate):
    rng = np.random.default_rng(1)
    y = (rate + rng.normal(scale=1.5, size=CUSTOMERS) > 4.0).astype(int)
    return pd.DataFrame(
        {
            "customer_id": np.arange(CUSTOMERS),
            "moment": np.full(CUSTOMERS, 70.0),
            "y": y,
        }
    )


def run(tmp_path, **kwargs):
    rate = database(tmp_path / "w.sqlite")
    return support.make_project(
        table(rate),
        "y",
        time_budget=300,
        output_dir=tmp_path / "out",
        charter=kwargs.pop("charter", None) or charter_file(tmp_path),
        id_columns=["customer_id", "moment"],
        ingest_plan=kwargs.pop("plan", None) or plan_file(tmp_path),
        **kwargs,
    ).run()


def read(result, name):
    return json.loads((pathlib.Path(result.run_dir) / name).read_text())


def test_features_come_from_the_database_and_reach_the_ledger(tmp_path):
    result = run(tmp_path)
    folder = pathlib.Path(result.run_dir)

    manifest = read(result, "feature_manifest.json")
    kept = [f["name"] for f in manifest["features"] if f["kept"]]
    assert manifest["declared_family_size"] == len(kept) > 3
    assert all(
        f["tier"] == "attested" for f in manifest["features"] if f["kept"]
    )
    for name in ("ingest_catalog.json", "grain_report.json", "join_graph.json"):
        assert (folder / name).exists()
    profile = read(result, "dataset_profile.json")["feature_plan"]
    assert set(kept) & set(profile["numeric"])
    ledger = ledger_module.Ledger.load(folder / ledger_module.FILE_NAME)
    registered = {h.spec.column for h in ledger.hypotheses.values()}
    assert set(kept) <= registered
    table_ = read(result, "ingest_field_provenance.json")
    assert {f["name"] for f in table_["fields"]} == set(kept)
    assert "orders__count__30d" in kept


def test_an_unknown_unit_stops_the_run_with_questions(tmp_path):
    rate = database(tmp_path / "w.sqlite")
    with pytest.raises(errors.BlockedError, match="grain"):
        support.make_project(
            table(rate),
            "y",
            time_budget=300,
            output_dir=tmp_path / "out",
            charter=charter_file(tmp_path),
            id_columns=["customer_id", "moment"],
            ingest_plan=plan_file(tmp_path, unit="household"),
        ).run()


def test_the_plan_and_the_charter_block_must_come_together(tmp_path):
    rate = database(tmp_path / "w.sqlite")
    with pytest.raises(errors.ConfigError, match="ingest block"):
        support.make_project(
            table(rate),
            "y",
            time_budget=300,
            output_dir=tmp_path / "out",
            charter=charter_file(tmp_path, block=False),
            ingest_plan=plan_file(tmp_path),
        ).run()
    with pytest.raises(errors.ConfigError, match="data\\.ingest_plan"):
        support.make_project(
            table(rate),
            "y",
            time_budget=300,
            output_dir=tmp_path / "out2",
            charter=charter_file(tmp_path),
        ).run()


def with_emails(frame):
    """Customers 250 and up repeat the email of customer i - 250."""
    frame = frame.copy()
    frame["email"] = [
        f"user{i % 250}@example.com" for i in frame["customer_id"]
    ]
    return frame


def entity_block():
    return ingest.EntitySpec(
        fields=(ingest.EntityField("email", False),),
        passes=(ingest.EntityPass("email"),),
    )


def split_sets(result):
    manifest = read(result, "split_manifest.json")
    out = {}
    for name in ("exploration_ids", "confirmation_ids", "final_test_ids"):
        out.update(dict.fromkeys(manifest[name], name))
    return out


def straddling(result):
    sets = split_sets(result)
    return [
        i
        for i in range(250)
        if sets.get(i) is not None
        and sets.get(i + 250) is not None
        and (sets[i] == "final_test_ids") != (sets[i + 250] == "final_test_ids")
    ]


def run_with_emails(tmp_path, entity):
    rate = database(tmp_path / "w.sqlite")
    return support.make_project(
        with_emails(table(rate)),
        "y",
        time_budget=300,
        output_dir=tmp_path / "out",
        charter=charter_file(tmp_path, entity=entity),
        id_columns=["customer_id", "moment", "email"],
        ingest_plan=plan_file(tmp_path),
        group_column=None,
    ).run()


def test_without_the_entity_block_duplicates_straddle_the_split(tmp_path):
    result = run_with_emails(tmp_path, None)
    assert straddling(result)


def test_the_resolved_entity_keeps_duplicates_on_one_side(tmp_path):
    result = run_with_emails(tmp_path, entity_block())
    document = read(result, "entity_resolution.json")
    assert document["group_column"] == "entity_group"
    assert document["groups"] == 250 and document["records"] == CUSTOMERS
    assert straddling(result) == []


def declared_totals(path, rows_off=0):
    with sqlite3.connect(path) as connection:
        keys = pd.read_sql("SELECT order_id, amount FROM orders", connection)
    return ingest.ReconcileSpec(
        (
            ingest.TableTotals(
                "orders",
                rows=len(keys) + rows_off,
                sums=(ingest.ColumnTotal("amount", float(keys.amount.sum())),),
                checksum_column="order_id",
                checksum=declared.key_checksum(keys.order_id),
            ),
        )
    )


def run_reconciled(tmp_path, rows_off):
    rate = database(tmp_path / "w.sqlite")
    block = declared_totals(tmp_path / "w.sqlite", rows_off)
    return support.make_project(
        table(rate),
        "y",
        time_budget=300,
        output_dir=tmp_path / "out",
        charter=charter_file(tmp_path, reconcile=block),
        id_columns=["customer_id", "moment"],
        ingest_plan=plan_file(tmp_path),
    ).run()


def test_the_extract_is_reconciled_to_the_declared_totals(tmp_path):
    result = run_reconciled(tmp_path, 0)
    report = read(result, "extract_reconciliation.json")
    assert report["passed"] is True
    assert {c["name"] for c in report["checks"]} == {
        "orders.row_count",
        "orders.sum.amount",
        "orders.checksum.order_id",
    }


def test_a_total_the_extract_does_not_reproduce_stops_the_run(tmp_path):
    with pytest.raises(errors.BlockedError, match=r"orders\.row_count"):
        run_reconciled(tmp_path, 1)


def run_with_corrupted_extract(tmp_path, monkeypatch, damage):
    from scmle.core import ingest_stage

    real = ingest_stage.read_frame

    def corrupted(entry, factory):
        frame = real(entry, factory)
        return damage(frame) if entry["table"] == "orders" else frame

    monkeypatch.setattr(ingest_stage, "read_frame", corrupted)
    return run_reconciled(tmp_path, 0)


def test_a_corrupted_extract_is_named_by_the_source_queries(
    tmp_path, monkeypatch
):
    def damage(frame):
        frame = frame.iloc[5:].copy()
        frame["amount"] = frame["amount"] + 1.0
        return frame

    with pytest.raises(errors.BlockedError) as raised:
        run_with_corrupted_extract(tmp_path, monkeypatch, damage)
    message = str(raised.value)
    for name in ("row_count", "sum.amount", "checksum.order_id"):
        assert f"orders.{name}" in message
    assert "source query" in message


def test_a_faithful_extract_passes_the_source_queries(tmp_path):
    result = run_reconciled(tmp_path, 0)
    report = read(result, "extract_reconciliation.json")
    assert report["source_passed"] is True
    assert {c["name"] for c in report["source_checks"]} == {
        "orders.row_count",
        "orders.sum.amount",
        "orders.checksum.order_id",
    }
