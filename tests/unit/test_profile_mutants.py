"""Targeted tests for profile behaviours the main suite left unpinned."""

import hashlib
import json
import types

import pandas
import pytest

from benchmarks.qualification import ingest_worlds
from scmle.data.ingest import budget
from scmle.data.ingest import profile
from scmle.data.readers import connectors


class RecordingCursor:
    """Delegates to a real cursor and notes each executed statement."""

    def __init__(self, cursor, log):
        self.cursor = cursor
        self.log = log

    def execute(self, sql, *rest):
        self.log.append(sql)
        return self.cursor.execute(sql, *rest)

    def __getattr__(self, name):
        return getattr(self.cursor, name)


class RecordingConnection:
    def __init__(self, connection, log):
        self.connection = connection
        self.log = log

    def cursor(self):
        return RecordingCursor(self.connection.cursor(), self.log)

    def __getattr__(self, name):
        return getattr(self.connection, name)


def recorded_source(rows=None, meter=None, **options):
    db = ingest_worlds.new_database("pm")
    db.execute("CREATE TABLE t (a INTEGER, b TEXT)")
    db.execute(
        "INSERT INTO t VALUES (?, ?)",
        rows if rows is not None else [(1, "x"), (2, "y"), (2, None)],
    )
    log = []
    source = profile.SqlSource(
        lambda: RecordingConnection(db.factory(), log), meter, **options
    )
    return source, log, db


def test_quote_refuses_an_empty_name_and_a_lone_nul():
    with pytest.raises(ValueError, match="unsafe identifier: ''"):
        profile.quote("")
    with pytest.raises(ValueError, match=r"unsafe identifier: 'a\\x00b'"):
        profile.quote("a\x00b")
    with pytest.raises(ValueError, match=r"unsafe identifier: 'a\"b'"):
        profile.quote('a"b')
    assert profile.quote("a") == '"a"'
    assert profile.quote_all(["a", "b c"]) == '"a", "b c"'


def test_statement_fills_each_placeholder():
    assert profile.statement("SELECT {a} FROM {b}", a='"x"', b='"y"') == (
        'SELECT "x" FROM "y"'
    )


def test_source_defaults_and_overrides_reach_the_connector():
    db = ingest_worlds.new_database("pd")
    plain = profile.SqlSource(db.factory)
    assert plain.connector.source_uri == "sqlite://ingest"
    assert plain.connector.query == "SELECT 1"
    assert plain.connector.read_only_statement is None
    assert plain.meter is None
    assert plain.queries == 0
    custom = profile.SqlSource(
        db.factory,
        uri="sqlite://other",
        read_only_statement="SELECT 2",
    )
    assert custom.connector.source_uri == "sqlite://other"
    assert custom.connector.read_only_statement == "SELECT 2"


def test_the_read_only_statement_runs_before_each_query():
    source, log = recorded_source(read_only_statement="SELECT 41")[:2]
    assert source.scalar("SELECT COUNT(*) FROM t") == 3
    assert log == ["SELECT 41", "SELECT COUNT(*) FROM t"]


def test_a_refused_write_names_the_source_uri():
    source = recorded_source(uri="sqlite://named")[0]
    with pytest.raises(connectors.ConnectorError) as caught:
        source.fetch("DELETE FROM t")
    assert caught.value.source_uri == "sqlite://named"


def test_fetch_charges_the_stage_unit_and_rows_and_counts_queries():
    meter = budget.BudgetMeter(
        budget.Budget(max_queries=2, max_rows_scanned=50)
    )
    source = recorded_source(meter=meter)[0]
    first = source.fetch("SELECT a FROM t ORDER BY a", "stg", "unit-1", 20)
    assert first == [(1,), (2,), (2,)]
    assert (meter.queries, meter.rows, source.queries) == (1, 20, 1)
    second = source.fetch("SELECT a FROM t", "stg", "unit-2", 20)
    assert second is not None
    assert (meter.queries, meter.rows, source.queries) == (2, 40, 2)
    assert source.fetch("SELECT a FROM t", "stg", "unit-3", 20) is None
    stop = meter.stops[0]
    assert (stop.stage, stop.unit) == ("stg", "unit-3")
    assert source.queries == 2


def test_fetch_without_a_stage_is_not_charged():
    meter = budget.BudgetMeter(budget.Budget(max_queries=0))
    source = recorded_source(meter=meter)[0]
    assert source.fetch("SELECT a FROM t ORDER BY a") == [(1,), (2,), (2,)]
    assert meter.queries == 0
    assert meter.stops == []
    assert source.queries == 1


def test_scalar_defaults_to_no_stage_and_an_empty_unit():
    meter = budget.BudgetMeter(budget.Budget(max_queries=0))
    source = recorded_source(meter=meter)[0]
    assert source.scalar("SELECT COUNT(*) FROM t") == 3
    assert meter.queries == 0
    assert source.scalar("SELECT COUNT(*) FROM t", "stg") is None
    assert meter.stops[0].unit == ""
    assert source.scalar("SELECT 1", "other", "u") is None
    assert meter.stops[1].unit == "u"


def test_scalar_of_a_query_with_no_rows_is_none():
    source = recorded_source()[0]
    assert source.scalar("SELECT a FROM t WHERE a > 99") is None
    assert source.scalar("SELECT a FROM t WHERE a = 1") == 1


def test_columns_reads_names_with_a_zero_row_query():
    source, log = recorded_source()[:2]
    assert source.columns("t") == ("a", "b")
    assert log == ['SELECT * FROM "t" WHERE 1 = 0']
    assert source.queries == 1
    source.columns("t")
    assert source.queries == 2


def test_columns_charges_the_table_with_no_rows_and_stops_cleanly():
    meter = budget.BudgetMeter(budget.Budget(max_queries=1, max_rows_scanned=0))
    source = recorded_source(meter=meter)[0]
    assert source.columns("t") == ("a", "b")
    assert meter.queries == 0
    assert source.columns("t", "stg") == ("a", "b")
    assert meter.queries == 1
    assert source.columns("t", "stg") == ()
    assert (meter.stops[0].stage, meter.stops[0].unit) == ("stg", "t")
    refusing = budget.BudgetMeter(budget.Budget(max_queries=0))
    other = recorded_source(meter=refusing)[0]
    assert other.columns("t", "stg") == ()
    assert other.queries == 0
    assert (refusing.stops[0].stage, refusing.stops[0].unit) == ("stg", "t")


def test_columns_cost_no_rows_even_when_rows_are_limited():
    meter = budget.BudgetMeter(budget.Budget(max_rows_scanned=0))
    source = recorded_source(meter=meter)[0]
    assert source.columns("t", "stg") == ("a", "b")
    assert meter.stops == []
    assert meter.rows == 0


def test_columns_refuses_an_unsafe_table_name():
    source = recorded_source()[0]
    with pytest.raises(ValueError, match="unsafe identifier"):
        source.columns('t"')


def test_value_kind_classifies_each_type():
    assert profile.value_kind(None) == "empty"
    assert profile.value_kind(True) == "other"
    assert profile.value_kind(3) == "number"
    assert profile.value_kind(2.5) == "number"
    assert profile.value_kind("x") == "text"
    assert profile.value_kind(b"x") == "other"


def test_the_profile_issues_these_exact_statements_and_charges():
    meter = budget.BudgetMeter(budget.Budget())
    source, log = recorded_source(meter=meter)[:2]
    table = profile.profile_table(source, "t", "prof")
    assert log == [
        'SELECT COUNT(*) FROM "t"',
        'SELECT * FROM "t" WHERE 1 = 0',
        'SELECT COUNT("a"), COUNT(DISTINCT "a"), MIN("a"), MAX("a") FROM "t"',
        'SELECT COUNT("b"), COUNT(DISTINCT "b"), MIN("b"), MAX("b") FROM "t"',
    ]
    assert table.to_dict() == {
        "name": "t",
        "rows": 3,
        "columns": [
            {
                "name": "a",
                "kind": "number",
                "non_null": 3,
                "distinct": 2,
                "minimum": "1",
                "maximum": "2",
            },
            {
                "name": "b",
                "kind": "text",
                "non_null": 2,
                "distinct": 2,
                "minimum": "x",
                "maximum": "y",
            },
        ],
    }
    assert (meter.queries, meter.rows) == (4, 6)


def test_the_profile_names_the_table_and_the_column_when_it_stops():
    meter = budget.BudgetMeter(budget.Budget(max_queries=0))
    source = recorded_source(meter=meter)[0]
    assert profile.profile_table(source, "t", "prof") is None
    assert (meter.stops[0].stage, meter.stops[0].unit) == ("prof", "t")
    columns = budget.BudgetMeter(budget.Budget(max_queries=2))
    other = recorded_source(meter=columns)[0]
    assert profile.profile_table(other, "t", "prof") is None
    assert (columns.stops[0].stage, columns.stops[0].unit) == ("prof", "t.a")


def test_the_default_profile_stage_is_profile():
    meter = budget.BudgetMeter(budget.Budget(max_queries=0))
    source = recorded_source(meter=meter)[0]
    assert profile.profile_table(source, "t") is None
    assert meter.stops[0].stage == "profile"


def test_an_all_null_column_has_no_extremes_and_kind_empty():
    source = recorded_source(rows=[(None, None), (None, None)])[0]
    table = profile.profile_table(source, "t")
    only = table.column("a")
    assert (only.kind, only.non_null, only.distinct) == ("empty", 0, 0)
    assert (only.minimum, only.maximum) == (None, None)
    assert table.to_dict()["columns"][1]["minimum"] is None


def test_a_table_with_no_columns_profile_is_refused_as_none():
    class Bare:
        queries = 0

        def scalar(self, sql, stage="", unit=""):
            return 3

        def columns(self, table, stage=""):
            return ()

    assert profile.profile_table(Bare(), "t") is None


def test_profile_table_passes_the_stage_to_each_source_call():
    calls = []

    class Spy:
        def scalar(self, sql, stage="", unit=""):
            calls.append(("scalar", stage, unit))
            return 5

        def columns(self, table, stage=""):
            calls.append(("columns", stage, table))
            return ("c",)

        def fetch(self, sql, stage="", unit="", cost_rows=0):
            calls.append(("fetch", stage, unit, cost_rows))
            return [(5, 2, 1, 9)]

    table = profile.profile_table(Spy(), "tbl", "stg")
    assert calls == [
        ("scalar", "stg", "tbl"),
        ("columns", "stg", "tbl"),
        ("fetch", "stg", "tbl.c", 5),
    ]
    assert table.rows == 5
    assert table.column("c").maximum == "9"


def test_table_and_catalog_lookups_and_dicts():
    first = profile.TableProfile("one", 2, ())
    catalog = profile.Catalog((first,))
    assert catalog.table("one") is first
    assert catalog.partial is False
    assert catalog.to_dict() == {
        "partial": False,
        "fingerprint": catalog.fingerprint,
        "tables": [{"name": "one", "rows": 2, "columns": []}],
    }
    with pytest.raises(KeyError, match="table 'two' was not profiled"):
        catalog.table("two")
    with pytest.raises(KeyError, match="one has no column 'z'"):
        first.column("z")


def test_a_complete_catalog_is_not_partial_and_a_stopped_one_is():
    source = recorded_source()[0]
    complete = profile.profile_catalog(source, ["t"])
    assert complete.partial is False
    assert [t.name for t in complete.tables] == ["t"]
    meter = budget.BudgetMeter(budget.Budget(max_queries=0))
    stopped_source = recorded_source(meter=meter)[0]
    stopped = profile.profile_catalog(stopped_source, ["t"], "custom")
    assert stopped.partial is True
    assert stopped.tables == ()
    assert meter.stops[0].stage == "custom"


def test_the_catalog_fingerprint_is_the_digest_of_sorted_json():
    first = profile.TableProfile(
        "one", 2, (profile.ColumnProfile("c", "number", 2, 2, "1", "2"),)
    )
    catalog = profile.Catalog((first,))
    expected = hashlib.sha256(
        json.dumps([first.to_dict()], sort_keys=True).encode("utf-8")
    ).hexdigest()
    assert catalog.fingerprint == expected


def test_torn_read_error_records_the_attempts_and_the_reason():
    error = profile.TornReadError("sqlite://x", 3, "because")
    assert error.code == "torn_read"
    assert error.source_uri == "sqlite://x"
    assert error.attempts == 3
    assert error.reason == "because"
    assert error.details == {
        "code": "torn_read",
        "source_uri": "sqlite://x",
        "attempts": 3,
        "reason": "because",
    }
    assert "source changed during the read (because) after 3 attempt(s)" in (
        str(error)
    )


def test_has_total_order_ignores_comments_between_the_words():
    assert profile.has_total_order("SELECT a FROM t ORDER/* x */BY a")
    assert profile.has_total_order("select a from t order\n  by a")
    assert not profile.has_total_order("SELECT a FROM t -- ORDER BY a")
    assert not profile.has_total_order("SELECT a FROM t")


class Batch:
    def __init__(self, rows, checkpoint, more):
        self.schema = types.SimpleNamespace(
            columns=[types.SimpleNamespace(name="v")]
        )
        self.rows = rows
        self.checkpoint = checkpoint
        self.has_more = more


class Scripted:
    """A connector with no query attribute that serves scripted batches."""

    source_uri = "scripted://x"

    def __init__(self, scripts):
        self.scripts = list(scripts)
        self.seen = []

    def read_rows(self, checkpoint, limit):
        self.seen.append((checkpoint, limit))
        script = self.scripts[0]
        batch = script.pop(0)
        if not script:
            self.scripts.pop(0)
        if isinstance(batch, Exception):
            raise batch
        return batch


def test_read_once_starts_with_no_checkpoint_and_follows_the_batches():
    connector = Scripted(
        [[Batch([(1,)], "c1", True), Batch([(2,)], "c2", False)]]
    )
    frame = profile.read_once(connector, 7)
    assert frame["v"].tolist() == [1, 2]
    assert connector.seen == [(None, 7), ("c1", 7)]


def test_consistent_read_works_for_a_connector_without_a_query():
    connector = Scripted([[Batch([(1,)], "c", False)]])
    read = profile.consistent_read(connector, 4)
    assert read.verified is False
    assert read.token is None
    assert read.attempts == 1
    assert read.frame["v"].tolist() == [1]


def test_the_unordered_error_carries_the_message_and_uri():
    db = ingest_worlds.new_database("uo")
    db.execute("CREATE TABLE t (a INTEGER)")
    connector = connectors.DbApiConnector(
        "sqlite://ordered", "SELECT a FROM t", db.factory
    )
    with pytest.raises(connectors.ConnectorError) as caught:
        profile.consistent_read(connector)
    assert caught.value.code == "unordered_query"
    assert caught.value.source_uri == "sqlite://ordered"
    assert caught.value.details == {
        "code": "unordered_query",
        "source_uri": "sqlite://ordered",
    }
    assert caught.value.as_dict()["message"] == (
        "batched reads need an ORDER BY on a unique key"
    )
    assert profile.consistent_read(connector, require_order=False).frame.empty


def test_the_token_is_returned_with_the_read_and_flags_it_verified():
    connector = Scripted([[Batch([(1,)], "c", False)]])
    read = profile.consistent_read(connector, 4, token=lambda: "snap")
    assert read.token == "snap"
    assert read.verified is True


def test_a_changing_token_gives_a_torn_read_with_the_reason_and_attempts():
    ticks = iter(range(100))
    connector = Scripted(
        [
            [Batch([(1,)], "c", False)],
            [Batch([(1,)], "c", False)],
            [Batch([(1,)], "c", False)],
        ]
    )
    with pytest.raises(profile.TornReadError) as caught:
        profile.consistent_read(
            connector, 4, token=lambda: str(next(ticks)), attempts=3
        )
    assert caught.value.reason == "snapshot token changed"
    assert caught.value.attempts == 3
    assert caught.value.source_uri == "scripted://x"
    assert str(caught.value).endswith(
        "(snapshot token changed) after 3 attempt(s)"
    )


def test_a_moved_batch_offset_is_retried_and_then_reported():
    moved = connectors.CheckpointError(
        "scripted://x", "source_changed", "moved", {}
    )
    connector = Scripted([[moved], [moved]])
    with pytest.raises(profile.TornReadError) as caught:
        profile.consistent_read(connector, 4, attempts=2)
    assert caught.value.reason == "rows before a batch offset changed"
    assert caught.value.attempts == 2
    other = connectors.CheckpointError(
        "scripted://x", "something_else", "bad", {}
    )
    with pytest.raises(connectors.CheckpointError):
        profile.consistent_read(Scripted([[other]]), 4)


def test_at_least_one_attempt_is_made_whatever_the_count():
    connector = Scripted([[Batch([(1,)], "c", False)]])
    read = profile.consistent_read(connector, 4, attempts=0)
    assert read.attempts == 1
    assert isinstance(read.frame, pandas.DataFrame)


def test_a_moved_batch_offset_on_the_first_attempt_is_retried():
    moved = connectors.CheckpointError(
        "scripted://x", "source_changed", "moved", {}
    )
    connector = Scripted([[moved], [Batch([(1,)], "c", False)]])
    read = profile.consistent_read(connector, 4, attempts=2)
    assert read.attempts == 2
    assert read.frame["v"].tolist() == [1]
