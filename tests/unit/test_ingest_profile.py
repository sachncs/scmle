"""Push-down profile, budget, and consistent reads."""

import pytest

from benchmarks.qualification import ingest_worlds
from scmle.data.ingest import budget
from scmle.data.ingest import profile
from scmle.data.readers import connectors


def test_profile_counts_are_exact():
    db = ingest_worlds.new_database("prof")
    db.execute("CREATE TABLE t (a INTEGER, b TEXT)")
    db.execute(
        "INSERT INTO t VALUES (?, ?)",
        [(1, "x"), (2, "x"), (2, None), (None, "y")],
    )
    table = profile.profile_table(db.source(), "t")
    assert table is not None
    assert table.rows == 4
    a = table.column("a")
    assert (a.non_null, a.distinct, a.kind) == (3, 2, "number")
    b = table.column("b")
    assert (b.non_null, b.distinct, b.kind) == (3, 2, "text")
    assert (b.minimum, b.maximum) == ("x", "y")


def test_catalog_fingerprint_changes_with_the_data():
    db = ingest_worlds.new_database("fp")
    db.execute("CREATE TABLE t (a INTEGER)")
    db.execute("INSERT INTO t VALUES (?)", [(1,), (2,)])
    first = profile.profile_catalog(db.source(), ["t"])
    assert profile.profile_catalog(db.source(), ["t"]).fingerprint == (
        first.fingerprint
    )
    db.execute("INSERT INTO t VALUES (?)", [(3,)])
    assert profile.profile_catalog(db.source(), ["t"]).fingerprint != (
        first.fingerprint
    )


def test_source_refuses_writes_and_keeps_data():
    db = ingest_worlds.new_database("ro")
    db.execute("CREATE TABLE t (a INTEGER)")
    db.execute("INSERT INTO t VALUES (?)", [(1,)])
    with pytest.raises(connectors.ConnectorError) as caught:
        db.source().fetch("DELETE FROM t")
    assert caught.value.code == "write_statement_refused"
    assert db.source().scalar("SELECT COUNT(*) FROM t") == 1


def test_unsafe_identifiers_are_rejected():
    with pytest.raises(ValueError):
        profile.quote('a"; DROP TABLE t; --')
    assert profile.quote("a b") == '"a b"'


def test_budget_stops_profile_at_a_table_boundary():
    db = ingest_worlds.new_database("bud")
    for name in ("t1", "t2"):
        db.execute(f"CREATE TABLE {name} (a INTEGER, b INTEGER)")
        db.execute(f"INSERT INTO {name} VALUES (?, ?)", [(1, 2), (3, 4)])
    meter = budget.BudgetMeter(budget.Budget(max_queries=5))
    source = profile.SqlSource(db.factory, meter)
    catalog = profile.profile_catalog(source, ["t1", "t2"])
    assert catalog.partial
    assert [t.name for t in catalog.tables] == ["t1"]
    assert meter.partial
    assert meter.stops[0].reason == "queries"
    assert source.queries <= 5


def test_consistent_read_on_static_data_is_verified_and_complete():
    connector = ingest_worlds.torn_world(1, None)[0]
    read = profile.consistent_read(connector, 5)
    assert read.verified
    assert read.attempts == 1
    assert read.frame["id"].tolist() == sorted(read.frame["id"].tolist())
    assert read.frame["id"].is_unique


@pytest.mark.parametrize("kind", ingest_worlds.TORN_KINDS)
def test_torn_read_is_detected(kind):
    connector = ingest_worlds.torn_world(2, kind)[0]
    with pytest.raises(profile.TornReadError):
        profile.consistent_read(connector, 5, attempts=1)


def test_torn_read_recovers_on_retry_within_budget():
    connector, db = ingest_worlds.torn_world(3, "insert")
    read = profile.consistent_read(connector, 5, attempts=2)
    assert read.attempts == 2
    assert len(read.frame) == db.frame("SELECT COUNT(*) AS n FROM t")["n"][0]


def test_unordered_query_is_refused():
    db = ingest_worlds.new_database("ord")
    db.execute("CREATE TABLE t (a INTEGER)")
    connector = connectors.DbApiConnector(
        ingest_worlds.URI, "SELECT a FROM t", db.factory
    )
    with pytest.raises(connectors.ConnectorError) as caught:
        profile.consistent_read(connector)
    assert caught.value.code == "unordered_query"
    assert profile.has_total_order("SELECT a FROM t ORDER BY a")
    assert not profile.has_total_order("SELECT 'order by' FROM t")


def test_read_without_a_token_is_marked_unverified():
    schema = connectors.SourceSchema((connectors.SourceColumn("a", "int"),))
    connector = connectors.WarehouseConnector(
        "warehouse://x/t",
        "select a from t order by a",
        lambda query: schema,
        lambda query, offset, limit: (
            schema,
            [(k,) for k in range(offset, min(offset + limit, 4))],
            offset + limit < 4,
        ),
    )
    read = profile.consistent_read(connector, 2)
    assert not read.verified
    assert len(read.frame) == 4


def test_budget_meter_is_deterministic_and_validates():
    def run():
        meter = budget.BudgetMeter(
            budget.Budget(max_rows_scanned=100, max_queries=10)
        )
        done = [meter.admit("s", f"u{k}", 30) for k in range(6)]
        return done, [s.to_dict() for s in meter.stops], meter.rows

    assert run() == run()
    done, stops, rows = run()
    assert done == [True, True, True, False, False, False]
    assert rows == 90
    assert len(stops) == 1
    with pytest.raises(ValueError):
        budget.Budget(max_queries=-1)


def test_wall_time_limit_uses_the_clock():
    ticks = iter(range(100))
    meter = budget.BudgetMeter(
        budget.Budget(max_seconds=3), lambda: float(next(ticks))
    )
    outcomes = [meter.admit("s", f"u{k}", 0) for k in range(5)]
    assert outcomes[0] and not outcomes[-1]
    assert meter.stops[0].reason == "seconds"
