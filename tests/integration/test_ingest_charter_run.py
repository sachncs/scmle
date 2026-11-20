"""A charter run that synthesizes its features from a database."""

import json
import pathlib
import sqlite3

import numpy
import pandas
import pytest
import support

from scmle.assurance.charter import ingest
from scmle.assurance.charter import sealing
from scmle.assurance.ingest import declared
import scmle.assurance.ledger.ledger
from scmle.contracts import errors
from tests.unit.test_charter_contracts import make_charter

KEY = "ingest-key"
CUSTOMERS = 500


def database(path):
    rng = numpy.random.default_rng(0)
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
    spec = ingest.Ingestion(
        features=40,
        windows=(30.0, 90.0),
        lag=1.0,
        entity=entity,
        reconcile=reconcile,
    )
    path = tmp_path / "charter.json"
    sealing.inscribe(
        path,
        make_charter(effect=0.5, quota=3, ingest=spec if block else None),
        signing_key=KEY,
    )
    return path


def table(rate):
    rng = numpy.random.default_rng(1)
    y = (rate + rng.normal(scale=1.5, size=CUSTOMERS) > 4.0).astype(int)
    return pandas.DataFrame(
        {
            "customer_id": numpy.arange(CUSTOMERS),
            "moment": numpy.full(CUSTOMERS, 70.0),
            "y": y,
        }
    )


def run(tmp_path, **kwargs):
    rate = database(tmp_path / "w.sqlite")
    return support.make_project(
        table(rate),
        "y",
        budget=300,
        directory=tmp_path / "out",
        charter=kwargs.pop("charter", None) or charter_file(tmp_path),
        identifiers=["customer_id", "moment"],
        ingest=kwargs.pop("plan", None) or plan_file(tmp_path),
        **kwargs,
    ).run()


def read(result, name):
    return json.loads((pathlib.Path(result.directory) / name).read_text())


def test_features_come_from_the_database_and_reach_the_ledger(tmp_path):
    # Enforces docs/ARCHITECTURE.md:111.
    # Enforces docs/ARTIFACTS.md:73.
    # Enforces website/src/pages/docs/artifacts.astro:114.
    # Enforces website/src/pages/docs/causal-decision.astro:52.
    result = run(tmp_path)
    folder = pathlib.Path(result.directory)

    manifest = read(result, "feature_manifest.json")
    kept = [f["name"] for f in manifest["features"] if f["kept"]]
    assert manifest["family"] == len(kept) > 3
    assert all(
        f["tier"] == "attested" for f in manifest["features"] if f["kept"]
    )
    for name in ("ingest_catalog.json", "grain_report.json", "join_graph.json"):
        assert (folder / name).exists()
    profile = read(result, "dataset_profile.json")["plan"]
    assert set(kept) & set(profile["numeric"])
    ledger = scmle.assurance.ledger.ledger.Ledger.load(
        folder / scmle.assurance.ledger.ledger.FILE_NAME
    )
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
            budget=300,
            directory=tmp_path / "out",
            charter=charter_file(tmp_path),
            identifiers=["customer_id", "moment"],
            ingest=plan_file(tmp_path, unit="household"),
        ).run()


def test_the_plan_and_the_charter_block_must_come_together(tmp_path):
    rate = database(tmp_path / "w.sqlite")
    with pytest.raises(errors.ConfigError, match="ingest block"):
        support.make_project(
            table(rate),
            "y",
            budget=300,
            directory=tmp_path / "out",
            charter=charter_file(tmp_path, block=False),
            ingest=plan_file(tmp_path),
        ).run()
    with pytest.raises(errors.ConfigError, match="data\\.ingest_plan"):
        support.make_project(
            table(rate),
            "y",
            budget=300,
            directory=tmp_path / "out2",
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
    return ingest.Entity(
        fields=(ingest.Attribute("email", False),),
        passes=(ingest.Pass("email"),),
    )


def split_sets(result):
    manifest = read(result, "split_manifest.json")
    out = {}
    for name in ("exploration", "confirmation", "final"):
        out.update(dict.fromkeys(manifest["sets"][name], name))
    return out


def straddling(result):
    sets = split_sets(result)
    return [
        i
        for i in range(250)
        if sets.get(i) is not None
        and sets.get(i + 250) is not None
        and (sets[i] == "final") != (sets[i + 250] == "final")
    ]


def run_with_emails(tmp_path, entity):
    rate = database(tmp_path / "w.sqlite")
    return support.make_project(
        with_emails(table(rate)),
        "y",
        budget=300,
        directory=tmp_path / "out",
        charter=charter_file(tmp_path, entity=entity),
        identifiers=["customer_id", "moment", "email"],
        ingest=plan_file(tmp_path),
        group=None,
    ).run()


def test_without_the_entity_block_duplicates_straddle_the_split(tmp_path):
    result = run_with_emails(tmp_path, None)
    assert straddling(result)


def test_the_resolved_entity_keeps_duplicates_on_one_side(tmp_path):
    # Enforces docs/ARCHITECTURE.md:111.
    # Enforces website/src/pages/docs/artifacts.astro:114.
    # Enforces website/src/pages/docs/causal-decision.astro:52.
    # Enforces website/src/pages/docs/qualification.astro:70.
    result = run_with_emails(tmp_path, entity_block())
    document = read(result, "entity_resolution.json")
    assert document["group_column"] == "entity_group"
    assert document["groups"] == 250 and document["records"] == CUSTOMERS
    assert straddling(result) == []


def declared_totals(path, rows_off=0):
    with sqlite3.connect(path) as connection:
        keys = pandas.read_sql(
            "SELECT order_id, amount FROM orders", connection
        )
    return ingest.Expectation(
        (
            ingest.Totals(
                "orders",
                rows=len(keys) + rows_off,
                sums=(ingest.Total("amount", float(keys.amount.sum())),),
                column="order_id",
                checksum=declared.sha(keys.order_id),
            ),
        )
    )


def run_reconciled(tmp_path, rows_off):
    rate = database(tmp_path / "w.sqlite")
    block = declared_totals(tmp_path / "w.sqlite", rows_off)
    return support.make_project(
        table(rate),
        "y",
        budget=300,
        directory=tmp_path / "out",
        charter=charter_file(tmp_path, reconcile=block),
        identifiers=["customer_id", "moment"],
        ingest=plan_file(tmp_path),
    ).run()


def test_the_extract_is_reconciled_to_the_declared_totals(tmp_path):
    # Enforces docs/ARTIFACTS.md:60.
    # Enforces website/src/pages/docs/artifacts.astro:114.
    # Enforces website/src/pages/docs/causal-decision.astro:52.
    result = run_reconciled(tmp_path, 0)
    report = read(result, "extract_reconciliation.json")
    assert report["passed"] is True
    assert {c["name"] for c in report["checks"]} == {
        "orders.row_count",
        "orders.sum.amount",
        "orders.checksum.order_id",
    }


def test_a_total_the_extract_does_not_reproduce_stops_the_run(tmp_path):
    # Enforces website/src/pages/docs/artifacts.astro:114.
    # Enforces website/src/pages/docs/causal-decision.astro:52.
    # Enforces website/src/pages/docs/qualification.astro:70.
    with pytest.raises(errors.BlockedError, match=r"orders\.row_count"):
        run_reconciled(tmp_path, 1)


def run_with_corrupted_extract(tmp_path, monkeypatch, damage):
    from scmle.pipeline import ingest_stage

    real = ingest_stage.read

    def corrupted(entry, factory):
        frame = real(entry, factory)
        return damage(frame) if entry["table"] == "orders" else frame

    monkeypatch.setattr(ingest_stage, "read", corrupted)
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
    # Enforces docs/ARTIFACTS.md:60.
    # Enforces website/src/pages/docs/artifacts.astro:114.
    # Enforces website/src/pages/docs/causal-decision.astro:52.
    result = run_reconciled(tmp_path, 0)
    report = read(result, "extract_reconciliation.json")
    assert report["source_passed"] is True
    assert {c["name"] for c in report["source_checks"]} == {
        "orders.row_count",
        "orders.sum.amount",
        "orders.checksum.order_id",
    }
