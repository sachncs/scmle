"""Snapshot-consistent batches, enforced read-only, and event ingest time."""

import sqlite3

import pytest

from scmle.data.readers import connectors
from scmle.data.readers import incremental

URI = "sqlite://memory/t"
QUERY = "SELECT id, v FROM t ORDER BY id"


def make_db(rows: int = 10) -> sqlite3.Connection:
    db = sqlite3.connect(":memory:", check_same_thread=False)
    db.execute("CREATE TABLE t (id INTEGER, v INTEGER)")
    db.executemany("INSERT INTO t VALUES (?, ?)", [(i, i) for i in range(rows)])
    db.commit()
    return db


def connector(db: sqlite3.Connection, **kwargs) -> connectors.DbApiConnector:
    return connectors.DbApiConnector(URI, QUERY, lambda: db, **kwargs)


class Shared:
    """A connection whose close is a no-op so the in-memory db survives."""

    def __init__(self, db: sqlite3.Connection) -> None:
        self.db = db

    def cursor(self):
        return self.db.cursor()

    def close(self) -> None:
        return None


def test_batches_are_disjoint_and_complete_on_static_data():
    db = make_db(10)
    conn = connectors.DbApiConnector(URI, QUERY, lambda: Shared(db))
    seen, checkpoint = [], None
    while True:
        batch = conn.read_rows(checkpoint, limit=3)
        seen += [row[0] for row in batch.rows]
        checkpoint = batch.checkpoint
        if not batch.has_more:
            break
    assert seen == list(range(10))
    assert checkpoint.content_sha256 is not None


@pytest.mark.parametrize(
    "mutation",
    [
        "DELETE FROM t WHERE id = 0",
        "UPDATE t SET v = 99 WHERE id = 1",
        "INSERT INTO t VALUES (-1, -1)",
    ],
)
def test_change_before_the_offset_is_refused(mutation):
    db = make_db(10)
    conn = connectors.DbApiConnector(URI, QUERY, lambda: Shared(db))
    first = conn.read_rows(None, limit=3)
    db.execute(mutation)
    db.commit()
    with pytest.raises(connectors.CheckpointError) as caught:
        conn.read_rows(first.checkpoint, limit=3)
    assert caught.value.code == "source_changed"


def test_rows_deleted_below_the_offset_are_refused():
    db = make_db(10)
    conn = connectors.DbApiConnector(URI, QUERY, lambda: Shared(db))
    first = conn.read_rows(None, limit=8)
    db.execute("DELETE FROM t WHERE id >= 2")
    db.commit()
    with pytest.raises(connectors.CheckpointError) as caught:
        conn.read_rows(first.checkpoint, limit=3)
    assert caught.value.code == "source_changed"


def test_snapshot_token_tracks_content():
    db = make_db(5)
    conn = connectors.DbApiConnector(URI, QUERY, lambda: Shared(db))
    before = conn.snapshot_token()
    assert conn.snapshot_token() == before
    db.execute("UPDATE t SET v = 7 WHERE id = 4")
    db.commit()
    assert conn.snapshot_token() != before


def test_snapshot_query_is_the_token_source():
    db = make_db(5)
    conn = connectors.DbApiConnector(
        URI,
        QUERY,
        lambda: Shared(db),
        snapshot_query="SELECT max(id) FROM t",
    )
    before = conn.snapshot_token()
    db.execute("UPDATE t SET v = 7 WHERE id = 1")
    db.commit()
    assert conn.snapshot_token() == before
    db.execute("INSERT INTO t VALUES (50, 1)")
    db.commit()
    assert conn.snapshot_token() != before


@pytest.mark.parametrize(
    "query",
    [
        "DELETE FROM t",
        "INSERT INTO t VALUES (1, 1)",
        "UPDATE t SET v = 1",
        "DROP TABLE t",
        "SELECT 1; DROP TABLE t",
        "SELECT * INTO copy FROM t",
        "WITH x AS (DELETE FROM t RETURNING id) SELECT * FROM x",
        "PRAGMA writable_schema = 1",
        "-- comment\nCREATE TABLE z (a)",
    ],
)
def test_write_statements_are_refused_at_construction(query):
    with pytest.raises(connectors.ConnectorError) as caught:
        connectors.DbApiConnector(URI, query, make_db)
    assert caught.value.code == "write_statement_refused"


def test_reads_with_quoted_keywords_are_allowed():
    connectors.refuse_writes(
        "SELECT 'delete' AS a, \"update\" FROM t -- drop\n;", URI
    )
    connectors.refuse_writes("WITH x AS (SELECT 1) SELECT * FROM x", URI)


def test_session_is_query_only(tmp_path):
    path = tmp_path / "t.db"
    setup = sqlite3.connect(path)
    setup.execute("CREATE TABLE t (id INTEGER, v INTEGER)")
    setup.commit()
    setup.close()
    conn = connectors.DbApiConnector(URI, QUERY, lambda: sqlite3.connect(path))
    with (
        conn.session("SELECT 1") as cursor,
        pytest.raises(sqlite3.OperationalError),
    ):
        cursor.connection.execute("INSERT INTO t VALUES (1, 1)")


def test_read_only_statement_runs_before_the_query():
    log = []

    class Recording(Shared):
        def cursor(self):
            outer = self.db.cursor()

            class Wrapped:
                description = None

                def execute(self, sql, *args):
                    log.append(sql)
                    outer.execute(sql, *args)
                    self.description = outer.description

                def fetchmany(self, n):
                    return outer.fetchmany(n)

                def close(self):
                    outer.close()

            return Wrapped()

    db = make_db(3)
    conn = connectors.DbApiConnector(
        URI,
        QUERY,
        lambda: Recording(db),
        read_only_statement="SELECT 42",
    )
    conn.read_rows(None, limit=2)
    assert log[:2] == ["SELECT 42", QUERY]


def test_warehouse_snapshot_token_change_is_refused():
    schema = connectors.SourceSchema((connectors.SourceColumn("a", "int"),))
    token = ["v1"]

    def reader(query, offset, limit):
        rows = [(i,) for i in range(offset, min(offset + limit, 6))]
        return schema, rows, offset + limit < 6

    conn = connectors.WarehouseConnector(
        "warehouse://x/t",
        "select a from t",
        lambda query: schema,
        reader,
        snapshot_reader=lambda query: token[0],
    )
    first = conn.read_rows(None, limit=2)
    assert conn.read_rows(first.checkpoint, limit=2).rows == ((2,), (3,))
    token[0] = "v2"
    with pytest.raises(connectors.CheckpointError):
        conn.read_rows(first.checkpoint, limit=2)


def test_event_ingest_time_round_trips_and_is_validated():
    event = incremental.Event(
        "e1",
        1,
        "2024-01-01T00:00:00Z",
        {"a": 1},
        ingest_time="2024-01-02T00:00:00Z",
    )
    assert incremental.Event.from_dict(event.to_dict()) == event
    legacy = incremental.Event("e1", 1, "2024-01-01T00:00:00Z", {"a": 1})
    assert "ingest_time" not in legacy.to_dict()
    assert incremental.Event.from_dict(legacy.to_dict()) == legacy
    with pytest.raises(ValueError):
        incremental.Event("e1", 1, "2024-01-01T00:00:00Z", {}, ingest_time="no")
    with pytest.raises(ValueError):
        incremental.Event(
            "e1", 1, "2024-01-01T00:00:00Z", {}, ingest_time="2024-01-01"
        )
