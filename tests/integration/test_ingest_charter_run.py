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


def charter_file(tmp_path, block=True):
    spec = ingest.IngestSpec(feature_budget=40, windows=(30.0, 90.0), lag=1.0)
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
