"""Behavioural tests of the ingestion plan over a real SQLite source."""

import json
import sqlite3

import pytest

from scmle.assurance.ingest import budget
from scmle.assurance.ingest import grain
from scmle.assurance.ingest import joins
from scmle.assurance.ingest import plan
from scmle.contracts import errors


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "source.db"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE customers (id INTEGER, region TEXT);
        CREATE TABLE orders (oid INTEGER, customer INTEGER, amount REAL);
        """
    )
    connection.executemany(
        "INSERT INTO customers VALUES (?, ?)",
        [(i, "north" if i % 2 else "south") for i in range(20)],
    )
    connection.executemany(
        "INSERT INTO orders VALUES (?, ?, ?)",
        [(i, i % 20, float(i)) for i in range(60)],
    )
    connection.commit()
    connection.close()
    return path


def test_the_artifact_names_are_fixed():
    assert (plan.CATALOG, plan.GRAIN, plan.JOINS, plan.BUDGET) == (
        "ingest_catalog.json",
        "grain_report.json",
        "join_graph.json",
        "ingest_budget.json",
    )


def test_a_config_needs_tables_and_a_profiled_target():
    with pytest.raises(errors.InputError, match="tables must not be empty"):
        plan.Config(())
    with pytest.raises(
        errors.InputError, match="target_table must be one of tables"
    ):
        plan.Config(("a",), unit="customer", target="b")
    with pytest.raises(
        errors.InputError, match="target_table must be one of tables"
    ):
        plan.Config(("a",), unit="customer")
    settings = plan.Config(("a",), target="zzz")
    assert settings.width == grain.LEVELS
    assert (settings.violation, settings.null, settings.orphans) == (
        0.0,
        0.0,
        0.0,
    )


def test_persist_writes_sorted_indented_json(tmp_path):
    path = plan.persist(tmp_path / "new" / "dir", "a.json", {"b": 1, "a": [2]})
    assert path == tmp_path / "new" / "dir" / "a.json"
    assert path.read_text() == '{\n  "a": [\n    2\n  ],\n  "b": 1\n}\n'


def test_opener_connects_read_only(source):
    connect = plan.opener(source)
    connection = connect()
    assert connection.execute("SELECT count(*) FROM orders").fetchone() == (60,)
    with pytest.raises(sqlite3.OperationalError):
        connection.execute("DELETE FROM orders")
    connection.close()
    assert connect() is not connect()


def test_declarations_build_references_in_order():
    out = plan.declarations(
        [
            {
                "child": "orders",
                "child_columns": ["customer"],
                "parent": "customers",
                "parent_columns": ["id"],
            }
        ]
    )
    assert out == (
        joins.Reference("orders", ("customer",), "customers", ("id",)),
    )
    assert plan.declarations([]) == ()


def test_run_all_writes_every_artifact(source, tmp_path):
    run = tmp_path / "run"
    settings = plan.Config(
        ("customers", "orders"), unit="customer", target="customers"
    )
    stages = plan.Plan(run, plan.opener(source), settings, clock=lambda: 0.0)
    summary = stages.execute()
    for name in (plan.CATALOG, plan.GRAIN, plan.JOINS, plan.BUDGET):
        assert (run / name).is_file()
    assert set(stages.scans) == {"customers", "orders"}
    assert summary["partial"] is False
    assert summary["source_queries"] == stages.source.queries > 0
    assert summary["verdict"] == stages.verdict.verdict
    assert summary["accepted_links"] == len(stages.graph.accepted())
    report = json.loads((run / plan.GRAIN).read_text())
    assert set(report["tables"]) == {"customers", "orders"}
    assert report["verdict"] is not None
    spent = json.loads((run / plan.BUDGET).read_text())
    assert spent["source_queries"] == summary["source_queries"]


def test_link_profiles_and_infers_first(source, tmp_path):
    stages = plan.Plan(
        tmp_path / "run", plan.opener(source), plan.Config(("orders",))
    )
    graph = stages.link()
    assert stages.catalog is not None
    assert set(stages.scans) == {"orders"}
    assert graph is stages.graph
    assert stages.verdict is None


def test_finish_before_any_stage_reports_nothing(source, tmp_path):
    stages = plan.Plan(
        tmp_path / "run", plan.opener(source), plan.Config(("orders",))
    )
    assert stages.finish() == {
        "partial": False,
        "source_queries": 0,
        "verdict": None,
        "accepted_links": 0,
        "questions": [],
    }


def test_an_unprofiled_target_gets_an_unknown_verdict(source, tmp_path):
    settings = plan.Config(
        ("customers", "orders"), unit="customer", target="orders"
    )
    limits = budget.Budget(queries=4)
    stages = plan.Plan(tmp_path / "run", plan.opener(source), settings, limits)
    stages.infer()
    assert list(stages.scans) == ["customers"]
    assert stages.verdict == grain.Verdict(
        grain.UNKNOWN,
        "customer",
        (),
        "target table was not profiled (budget or scope)",
        ("raise the ingest budget",),
    )
    report = json.loads((tmp_path / "run" / plan.GRAIN).read_text())
    assert report["partial"] is True
    assert report["verdict"] == stages.verdict.to_dict()
