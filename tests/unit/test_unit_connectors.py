"""Behavioural tests for the source connectors."""

import base64
import gzip
import hashlib
import pathlib
import sqlite3

import pandas
import pytest

from scmle.contracts import errors
from scmle.data.readers.connectors import adapter
from scmle.data.readers.connectors import bucket
from scmle.data.readers.connectors import database
from scmle.data.readers.connectors import failure
from scmle.data.readers.connectors import filesystem
from scmle.data.readers.connectors import lakehouse
from scmle.data.readers.connectors import layout
from scmle.data.readers.connectors import locator
from scmle.data.readers.connectors import stream

NO_RETRY = failure.Retries(limit=0)


def columns(*names):
    return layout.Columns(tuple(layout.Field(name, "int") for name in names))


# failure


def test_a_connector_error_serialises_its_context():
    error = failure.ConnectorError("c", "m", "u", {"k": 1})
    assert error.details == {"code": "c", "source_uri": "u", "k": 1}
    assert error.serialize() == {
        "category": "data",
        "code": "c",
        "message": "m",
        "source_uri": "u",
        "details": {"code": "c", "source_uri": "u", "k": 1},
    }
    assert failure.ConnectorError("c", "m", "u").details == {
        "code": "c",
        "source_uri": "u",
    }


def test_scheme_and_retry_errors_describe_the_failure():
    scheme = failure.SchemeError("x", "", ["file"])
    assert str(scheme) == (
        "unsupported URI scheme '<path>'; supported schemes: ('file',)"
    )
    assert scheme.schemes == ("file",)
    assert scheme.details["supported_schemes"] == ["file"]
    named = failure.SchemeError("x", "ftp", ("s3",))
    assert "'ftp'" in str(named)
    retry = failure.RetryError("u", "op", 3, KeyError("k"))
    assert str(retry) == "op failed after 3 attempt(s): 'k'"
    assert retry.details["last_error_type"] == "KeyError"
    assert retry.attempts == 3
    checkpoint = failure.CheckpointError("u", "c", "m", {"a": 1})
    assert checkpoint.code == "c"
    assert checkpoint.details["a"] == 1


def test_retries_validate_and_count_attempts():
    assert failure.Retries().attempts == 3
    assert failure.Retries(limit=0).attempts == 1
    assert failure.Retries(delay=0.0).delay == 0.0
    with pytest.raises(errors.InputError, match="max_retries"):
        failure.Retries(limit=-1)
    with pytest.raises(errors.InputError, match="delay_seconds"):
        failure.Retries(delay=-0.1)
    with pytest.raises(errors.InputError, match="delay_seconds"):
        failure.Retries(delay=float("inf"))


def test_persevere_retries_until_success():
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise OSError("down")
        return "ok"

    assert failure.persevere(flaky, "op", "u", failure.Retries(2)) == "ok"
    assert len(calls) == 3


def test_persevere_gives_up_after_the_budget():
    calls = []

    def broken():
        calls.append(1)
        raise OSError("down")

    with pytest.raises(failure.RetryError) as raised:
        failure.persevere(broken, "op", "u", failure.Retries(1))
    assert raised.value.attempts == 2
    assert len(calls) == 2
    assert isinstance(raised.value.error, OSError)


def test_persevere_does_not_retry_connector_errors():
    calls = []

    def refused():
        calls.append(1)
        raise failure.ConnectorError("c", "m", "u")

    with pytest.raises(failure.ConnectorError, match="m"):
        failure.persevere(refused, "op", "u", failure.Retries(3))
    assert len(calls) == 1


def test_persevere_sleeps_between_attempts(monkeypatch):
    pauses = []
    monkeypatch.setattr(failure.time, "sleep", pauses.append)
    with pytest.raises(failure.RetryError):
        failure.persevere(
            lambda: 1 / 0, "op", "u", failure.Retries(2, delay=0.5)
        )
    assert pauses == [0.5, 0.5]


# locator


def test_canonicalize_lowercases_schemes_and_resolves_paths():
    assert locator.canonicalize("S3://Bucket/Key?a=1#f") == (
        "s3://Bucket/Key?a=1#f"
    )
    assert locator.canonicalize(" data.csv ") == (
        pathlib.Path("data.csv").resolve().as_uri()
    )
    drive = locator.canonicalize("C:data.csv")
    assert drive.startswith("file://")
    with pytest.raises(errors.InputError, match="must not be empty"):
        locator.canonicalize("  ")


def test_source_identity_hashes_the_canonical_uri():
    expected = hashlib.sha256(b"s3://b/k").hexdigest()
    assert locator.identify("S3://b/k") == f"src_{expected}"


def test_bookmark_round_trips_through_unwrap():
    cursor = locator.bookmark("src_a", 12)
    assert "=" not in cursor
    assert base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4)) == (
        b"src_a:12"
    )
    assert locator.unwrap(cursor) == ("src_a", 12)
    assert locator.unwrap(locator.bookmark("a:b", 0)) == ("a:b", 0)
    with pytest.raises(errors.InputError, match="source_identity_value"):
        locator.bookmark("", 0)
    with pytest.raises(errors.InputError, match="offset"):
        locator.bookmark("a", -1)


@pytest.mark.parametrize(
    "cursor",
    [
        base64.urlsafe_b64encode(b"nocolon").decode(),
        base64.urlsafe_b64encode(b"a:x").decode(),
        base64.urlsafe_b64encode(b"\xff:1").decode(),
    ],
)
def test_unwrap_rejects_malformed_cursors(cursor):
    with pytest.raises(errors.InputError, match="not a valid connector"):
        locator.unwrap(cursor)


def test_unwrap_rejects_empty_identities_and_negative_offsets():
    with pytest.raises(errors.InputError, match="must not be empty"):
        locator.unwrap("")
    for payload in (b":1", b"a:-1"):
        cursor = base64.urlsafe_b64encode(payload).decode()
        with pytest.raises(errors.InputError, match="invalid source or offset"):
            locator.unwrap(cursor)


def test_a_bookmark_fills_fields_from_its_cursor():
    cursor = locator.bookmark("src_a", 5)
    mark = locator.Bookmark(cursor=cursor)
    assert (mark.offset, mark.identity) == (5, "src_a")
    assert locator.Bookmark(5, cursor, "src_a").offset == 5
    with pytest.raises(errors.InputError, match="identity does not match"):
        locator.Bookmark(cursor=cursor, identity="src_b")
    with pytest.raises(errors.InputError, match="offset does not match"):
        locator.Bookmark(offset=4, cursor=cursor)
    with pytest.raises(errors.InputError, match="non-negative"):
        locator.Bookmark(offset=-1)
    with pytest.raises(errors.InputError, match="content_sha256"):
        locator.Bookmark(digest="A" * 64)
    with pytest.raises(errors.InputError, match="content_sha256"):
        locator.Bookmark(digest="a" * 63)


def test_bind_seal_and_advance():
    bound = locator.Bookmark(offset=2).bind("src_a")
    assert bound.identity == "src_a"
    assert bound.cursor == locator.bookmark("src_a", 2)
    assert bound.bind("src_a") == bound
    with pytest.raises(errors.InputError, match="different source"):
        bound.bind("src_b")
    sealed = bound.seal("a" * 64)
    assert sealed.digest == "a" * 64
    assert sealed.seal("a" * 64) == sealed
    with pytest.raises(errors.InputError, match="content digest"):
        sealed.seal("b" * 64)
    moved = sealed.advance(3)
    assert moved.offset == 5
    assert moved.cursor == locator.bookmark("src_a", 5)
    assert moved.digest == "a" * 64
    assert sealed.advance(0).offset == 2
    with pytest.raises(errors.InputError, match="non-negative"):
        sealed.advance(-1)
    with pytest.raises(errors.InputError, match="bound before advancing"):
        locator.Bookmark().advance(1)


# layout


def test_fields_and_columns_validate_names():
    with pytest.raises(errors.InputError, match="column name"):
        layout.Field("", "int")
    with pytest.raises(errors.InputError, match="type_name"):
        layout.Field("a", "")
    with pytest.raises(errors.InputError, match="at least one column"):
        layout.Columns(())
    with pytest.raises(errors.InputError, match="duplicate"):
        columns("a", "a")
    assert columns("a", "b").serialize() == {
        "columns": [
            {"name": "a", "type_name": "int"},
            {"name": "b", "type_name": "int"},
        ]
    }


def test_columns_describe_a_frame():
    frame = pandas.DataFrame({"a": [1], 2: [1.5]})
    assert layout.Columns.infer(frame).serialize() == {
        "columns": [
            {"name": "a", "type_name": "int64"},
            {"name": "2", "type_name": "float64"},
        ]
    }


# adapter


class Fixed(adapter.Adapter):
    def schema(self):
        return columns("a")

    def read(self, checkpoint=None, limit=1_000):
        return None


def test_an_adapter_binds_checkpoints_to_its_source():
    connector = Fixed("S3://b/k")
    assert connector.uri == "s3://b/k"
    assert connector.identity == locator.identify("s3://b/k")
    assert connector.budget == failure.Retries()
    mark = connector.checkpoint(4)
    assert mark.offset == 4
    assert mark.identity == connector.identity
    assert connector.rebind(None).offset == 0
    assert connector.rebind(None).identity == connector.identity
    foreign = Fixed("s3://other").checkpoint(1)
    with pytest.raises(failure.CheckpointError) as raised:
        connector.rebind(foreign)
    assert raised.value.code == "checkpoint_mismatch"
    assert raised.value.details["offset"] == 1
    connector.validate(1)
    with pytest.raises(errors.InputError, match="limit must be positive"):
        connector.validate(0)


# filesystem


def test_checksum_hashes_in_chunks(tmp_path):
    path = tmp_path / "f.bin"
    path.write_bytes(b"abcdef")
    expected = hashlib.sha256(b"abcdef").hexdigest()
    assert filesystem.checksum(path, chunk_bytes=1) == expected
    assert filesystem.checksum(path) == expected
    with pytest.raises(errors.InputError, match="chunk_bytes"):
        filesystem.checksum(path, chunk_bytes=0)


def test_filesystem_reads_pages_and_resumes(tmp_path):
    path = tmp_path / "t.csv"
    path.write_text("a,b\n1,x\n2,y\n3,z\n")
    connector = filesystem.Filesystem(path, NO_RETRY)
    assert connector.schema().serialize()["columns"][0] == {
        "name": "a",
        "type_name": "int64",
    }
    first = connector.read(limit=2)
    assert first.rows == ((1, "x"), (2, "y"))
    assert first.more
    assert first.digest == hashlib.sha256(path.read_bytes()).hexdigest()
    second = connector.read(first.checkpoint, limit=2)
    assert second.rows == ((3, "z"),)
    assert not second.more
    assert second.checkpoint.offset == 3
    path.write_text("a,b\n9,q\n9,q\n9,q\n")
    with pytest.raises(failure.ConnectorError) as raised:
        connector.read(first.checkpoint)
    assert raised.value.code == "source_snapshot_changed"


def test_filesystem_accepts_file_uris_and_rejects_others(tmp_path):
    path = tmp_path / "t.csv"
    connector = filesystem.Filesystem(path.as_uri())
    assert connector.path == path.resolve()
    local = filesystem.Filesystem(
        path.as_uri().replace("file://", "file://localhost", 1)
    )
    assert local.path == path.resolve()
    with pytest.raises(failure.SchemeError) as raised:
        filesystem.Filesystem("s3://b/k")
    assert raised.value.scheme == "s3"
    with pytest.raises(failure.SchemeError) as raised:
        filesystem.Filesystem("file://remote/x.csv")
    assert raised.value.schemes == ("file",)


def test_filesystem_detects_a_change_during_the_read(tmp_path, monkeypatch):
    path = tmp_path / "t.csv"
    path.write_text("a\n1\n")
    connector = filesystem.Filesystem(path, NO_RETRY)
    digests = iter(["a" * 64, "b" * 64])
    monkeypatch.setattr(connector, "digest", lambda: next(digests))
    with pytest.raises(failure.ConnectorError) as raised:
        connector.read()
    assert raised.value.details["initial_content_sha256"] == "a" * 64


def test_filesystem_retries_a_missing_file(tmp_path):
    connector = filesystem.Filesystem(tmp_path / "missing.csv", NO_RETRY)
    with pytest.raises(failure.RetryError) as raised:
        connector.read()
    assert raised.value.operation == "read_rows"
    with pytest.raises(failure.RetryError):
        connector.schema()


# bucket


def test_a_payload_derives_its_identity():
    record = bucket.Payload("s3://b/k", b"abc")
    assert record.digest == hashlib.sha256(b"abc").hexdigest()
    assert record.size == 3
    with pytest.raises(errors.InputError, match="source_uri"):
        bucket.Payload(" ", b"")
    with pytest.raises(errors.KindError, match="bytes"):
        bucket.Payload("s3://b/k", "abc")


def test_unpack_chooses_the_format_from_the_name(tmp_path):
    frame = pandas.DataFrame({"a": [1, 2]})
    assert bucket.unpack("s3://b/k.csv", b"a\n1\n2\n").equals(frame)
    lines = b'{"a": 1}\n{"a": 2}\n'
    assert bucket.unpack("s3://b/k.NDJSON", lines)["a"].tolist() == [1, 2]
    assert bucket.unpack("s3://b/k.csv.gz", gzip.compress(b"a\n1\n2\n")).equals(
        frame
    )
    assert bucket.unpack("s3://b/k.jsonl.gz", gzip.compress(lines))[
        "a"
    ].tolist() == [1, 2]
    with pytest.raises(failure.ConnectorError) as raised:
        bucket.unpack("s3://b/k.xlsx", b"")
    assert raised.value.code == "unsupported_object_format"
    with pytest.raises(failure.ConnectorError, match="<unknown>"):
        bucket.unpack("s3://b/", b"")
    assert bucket.CSV_COMPRESSION == {".gz": "gzip", ".bz2": "bz2", ".xz": "xz"}


def test_a_bucket_validates_its_configuration():
    reader = lambda uri: b""  # noqa: E731
    with pytest.raises(failure.SchemeError):
        bucket.Bucket("ftp://b/k.csv", reader)
    with pytest.raises(errors.KindError, match="read_bytes"):
        bucket.Bucket("s3://b/k.csv", None)
    with pytest.raises(errors.InputError, match="limit"):
        bucket.Bucket("s3://b/k.csv", reader, limit=0)
    with pytest.raises(errors.KindError, match="decoder must be callable"):
        bucket.Bucket("s3://b/k.csv", reader, decoder=1, codec="d")
    with pytest.raises(errors.InputError, match="requires a caller-owned"):
        bucket.Bucket("s3://b/k.csv", reader, codec="d")
    with pytest.raises(errors.InputError, match="identify a caller-owned"):
        bucket.Bucket("s3://b/k.csv", reader, decoder=bucket.unpack)
    default = bucket.Bucket(" S3://b/k.csv ", reader)
    assert default.codec == "tabular-auto-v1"
    assert default.identity == locator.identify(
        "s3://b/k.csv|decoder=tabular-auto-v1"
    )
    custom = bucket.Bucket(
        "s3://b/k.csv", reader, decoder=bucket.unpack, codec=" mine "
    )
    assert custom.codec == "mine"


def test_a_bucket_reads_pages_bound_to_content():
    store = {"payload": b"a\n1\n2\n3\n"}
    connector = bucket.Bucket(
        "s3://b/k.csv", lambda uri: store["payload"], NO_RETRY
    )
    first = connector.read(limit=2)
    assert first.rows == ((1,), (2,))
    assert first.more
    second = connector.read(first.checkpoint, limit=5)
    assert second.rows == ((3,),)
    assert not second.more
    assert connector.schema().serialize()["columns"][0]["name"] == "a"
    store["payload"] = b"a\n7\n"
    with pytest.raises(failure.ConnectorError) as raised:
        connector.read(first.checkpoint)
    assert raised.value.code == "source_snapshot_changed"


def test_a_bucket_checks_payloads_and_decoders():
    too_big = bucket.Bucket("s3://b/k.csv", lambda uri: b"abcd", limit=3)
    with pytest.raises(failure.ConnectorError) as raised:
        too_big.fetch()
    assert raised.value.details == {
        "code": "object_too_large",
        "source_uri": "s3://b/k.csv",
        "bytes": 4,
        "max_bytes": 3,
    }
    assert (
        bucket.Bucket("s3://b/k.csv", lambda uri: b"abc", limit=3).fetch().size
        == 3
    )
    text = bucket.Bucket("s3://b/k.csv", lambda uri: "a")
    with pytest.raises(failure.ConnectorError) as raised:
        text.fetch()
    assert raised.value.details["payload_type"] == "str"

    def failing(uri, payload):
        raise ValueError("nope")

    broken = bucket.Bucket(
        "s3://b/k", lambda uri: b"x", decoder=failing, codec="d"
    )
    with pytest.raises(failure.ConnectorError) as raised:
        broken.load()
    assert raised.value.code == "decoder_failed"
    assert raised.value.details["error_type"] == "ValueError"
    wrong = bucket.Bucket(
        "s3://b/k", lambda uri: b"x", decoder=lambda u, p: [1], codec="d"
    )
    with pytest.raises(failure.ConnectorError) as raised:
        wrong.load()
    assert raised.value.code == "invalid_decoded_object"
    unsupported = bucket.Bucket("s3://b/k.bin", lambda uri: b"x")
    with pytest.raises(failure.ConnectorError) as raised:
        unsupported.load()
    assert raised.value.code == "unsupported_object_format"


# database


@pytest.mark.parametrize(
    "query",
    [
        "SELECT 1",
        "  with t as (select 1) select * from t;",
        "VALUES (1)",
        "select 'drop table x' -- update\n",
        'select "insert" from t /* delete */',
    ],
)
def test_readonly_accepts_plain_reads(query):
    database.readonly(query, "dbapi://x")


@pytest.mark.parametrize(
    ("query", "reason"),
    [
        ("select 1; select 2", "more than one statement"),
        ("insert into t values (1)", "does not start with"),
        ("", "does not start with"),
        ("select * into t2 from t", "['into']"),
        ("with x as (delete from t) select 1", "['delete']"),
        ("select 1 from t where pragma", "['pragma']"),
    ],
)
def test_readonly_refuses_writes(query, reason):
    with pytest.raises(failure.ConnectorError) as raised:
        database.readonly(query, "dbapi://x")
    assert raised.value.code == "write_statement_refused"
    assert reason in raised.value.details["reason"]


def test_feed_hashes_rows_canonically():
    hasher = hashlib.sha256()
    database.feed(hasher, [[1, "a"], (2,)])
    assert hasher.hexdigest() == hashlib.sha256(b"(1, 'a')\n(2,)\n").hexdigest()


def test_release_suppresses_close_errors():
    closed = []

    class Closing:
        def __init__(self, fail):
            self.fail = fail

        def close(self):
            closed.append(self.fail)
            if self.fail:
                raise RuntimeError

    database.release(Closing(True), Closing(True))
    database.release(None, Closing(False))
    assert closed == [True, True, False]


def factory(rows=(1, 2, 3)):
    def connect():
        connection = sqlite3.connect(":memory:")
        connection.execute("create table t (a integer)")
        connection.executemany("insert into t values (?)", [(r,) for r in rows])
        return connection

    return connect


def test_a_database_connector_reads_pages_and_detects_changes():
    connector = database.DbApiConnector(
        "sqlite://mem", "select a from t order by a", factory(), NO_RETRY
    )
    assert connector.schema().serialize() == {
        "columns": [{"name": "a", "type_name": "unknown"}]
    }
    first = connector.read(limit=2)
    assert first.rows == ((1,), (2,))
    assert first.more
    hasher = hashlib.sha256()
    database.feed(hasher, [(1,), (2,)])
    assert first.checkpoint.digest == hasher.hexdigest()
    second = connector.read(first.checkpoint, limit=2)
    assert second.rows == ((3,),)
    assert not second.more
    changed = database.DbApiConnector(
        "sqlite://mem",
        "select a from t order by a",
        factory((1, 5, 3)),
        NO_RETRY,
    )
    with pytest.raises(failure.CheckpointError) as raised:
        changed.read(first.checkpoint)
    assert raised.value.code == "source_changed"
    shrunk = database.DbApiConnector(
        "sqlite://mem", "select a from t order by a", factory((1,)), NO_RETRY
    )
    with pytest.raises(failure.CheckpointError) as raised:
        shrunk.read(first.checkpoint)
    assert raised.value.details["missing"] == 1


def test_a_database_connector_validates_its_configuration():
    with pytest.raises(failure.SchemeError):
        database.DbApiConnector("pg://x", "select 1", factory())
    with pytest.raises(errors.InputError, match="query"):
        database.DbApiConnector("dbapi://x", " ", factory())
    with pytest.raises(errors.KindError, match="callable"):
        database.DbApiConnector("dbapi://x", "select 1", None)
    with pytest.raises(failure.ConnectorError):
        database.DbApiConnector("dbapi://x", "delete from t", factory())
    with pytest.raises(failure.ConnectorError):
        database.DbApiConnector(
            "dbapi://x", "select 1", factory(), probe="drop table t"
        )
    loose = database.DbApiConnector(
        "dbapi://x", "delete from t", factory(), guarded=False
    )
    assert not loose.guarded
    assert loose.identity == locator.identify("dbapi://x|query=delete from t")


def test_a_read_only_sqlite_session_refuses_writes():
    connector = database.DbApiConnector(
        "sqlite://mem", "select a from t", factory(), NO_RETRY
    )
    with (
        pytest.raises(sqlite3.OperationalError),
        connector.session("insert into t values (9)"),
    ):
        pass
    loose = database.DbApiConnector(
        "sqlite://mem", "select a from t", factory(), NO_RETRY, guarded=False
    )
    with loose.session("insert into t values (9)"):
        pass


def test_snapshot_tokens_digest_the_chosen_query():
    plain = database.DbApiConnector("sqlite://m", "select a from t", factory())
    hasher = hashlib.sha256()
    database.feed(hasher, [(1,), (2,), (3,)])
    assert plain.snapshot() == hasher.hexdigest()
    counted = database.DbApiConnector(
        "sqlite://m",
        "select a from t",
        factory(),
        probe="select count(*) from t",
        preamble="select 1",
    )
    hasher = hashlib.sha256()
    database.feed(hasher, [(3,)])
    assert counted.snapshot() == hasher.hexdigest()


def test_translate_reads_cursor_descriptions():
    connector = database.DbApiConnector("dbapi://x", "select 1", factory())
    schema = connector.translate([("a", "INT"), ("b",), ("c", None)])
    assert schema.serialize()["columns"] == [
        {"name": "a", "type_name": "INT"},
        {"name": "b", "type_name": "unknown"},
        {"name": "c", "type_name": "unknown"},
    ]
    with pytest.raises(errors.InputError, match="result schema"):
        connector.translate(None)
    with pytest.raises(errors.InputError, match="result schema"):
        connector.translate([])


# lakehouse


def warehouse(rows, snapshot=None, **kwargs):
    def batch(query, offset, limit):
        chunk = rows[offset : offset + limit]
        return columns("a"), chunk, offset + limit < len(rows)

    return lakehouse.Lakehouse(
        "warehouse://db",
        "select a",
        lambda query: columns("a"),
        kwargs.pop("batch", batch),
        NO_RETRY,
        snapshot,
    )


def test_a_lakehouse_reads_pages_and_tracks_snapshots():
    state = {"token": "v1"}
    connector = warehouse([[1], [2], [3]], lambda query: state["token"])
    assert connector.schema() == columns("a")
    first = connector.read(limit=2)
    assert first.rows == ((1,), (2,))
    assert first.more
    assert first.checkpoint.digest == hashlib.sha256(b"v1").hexdigest()
    second = connector.read(first.checkpoint, limit=2)
    assert second.rows == ((3,),)
    assert not second.more
    state["token"] = "v2"
    with pytest.raises(failure.CheckpointError) as raised:
        connector.read(first.checkpoint)
    assert raised.value.code == "source_changed"
    untracked = warehouse([[1]]).read()
    assert untracked.checkpoint.digest is None


def test_a_lakehouse_validates_callback_results():
    for result, code in (
        ((columns("a"), [[1]], 1), "invalid_batch_more_flag"),
        ((columns("a"), [[1], [2]], False), "batch_limit_exceeded"),
        ((columns("a"), [[1, 2]], False), "row_width_mismatch"),
        (("schema", [], False), "invalid_schema"),
    ):
        connector = warehouse([], batch=lambda q, o, n, r=result: r)
        with pytest.raises(failure.ConnectorError) as raised:
            connector.read(limit=1)
        assert raised.value.code == code


def test_a_lakehouse_validates_its_configuration():
    reader = lambda query: columns("a")  # noqa: E731
    with pytest.raises(failure.SchemeError):
        lakehouse.Lakehouse("db://x", "q", reader, reader)
    with pytest.raises(errors.InputError, match="query"):
        lakehouse.Lakehouse("warehouse://x", " ", reader, reader)
    with pytest.raises(errors.KindError, match="callable"):
        lakehouse.Lakehouse("warehouse://x", "q", None, reader)
    with pytest.raises(errors.KindError, match="callable"):
        lakehouse.Lakehouse("lakehouse://x", "q", reader, None)
    built = lakehouse.Lakehouse("LAKEHOUSE://x", "q", reader, reader)
    assert built.identity == locator.identify("lakehouse://x|query=q")


# stream


def test_classify_names_json_types():
    assert [
        stream.classify(v) for v in (None, True, 1, 1.5, "s", [1], {"a": 1})
    ] == ["null", "bool", "int", "float", "str", "array", "object"]
    assert stream.classify(b"x") == "bytes"


def test_interpret_builds_schema_and_rows():
    schema, rows = stream.interpret(
        [(b'{"a": 1, "b": "x"}', 10), (b'{"a": 2, "b": null}', 20)], "u"
    )
    assert schema.serialize()["columns"] == [
        {"name": "a", "type_name": "int"},
        {"name": "b", "type_name": "str"},
    ]
    assert rows == ((1, "x"), (2, None))
    for records, code in (
        ([(b"{bad", 1)], "invalid_json_record"),
        ([(b"\xff", 1)], "invalid_json_record"),
        ([(b"[1]", 1)], "record_not_object"),
        ([(b'{"a": 1}', 1), (b'{"b": 1}', 2)], "schema_changed"),
    ):
        with pytest.raises(failure.ConnectorError) as raised:
            stream.interpret(records, "u")
        assert raised.value.code == code


def ranged(content, chunk=4, **kwargs):
    return stream.StreamingObjectStoreConnector(
        "s3://b/k.jsonl",
        lambda uri, start, end: content[start:end],
        lambda uri: len(content),
        NO_RETRY,
        chunk=chunk,
        **kwargs,
    )


def test_a_stream_reads_records_by_byte_offset():
    content = b'{"a": 1}\r\n\n{"a": 2}\n{"a": 3}'
    connector = ranged(content)
    first = connector.read(limit=1)
    assert first.rows == ((1,),)
    assert first.checkpoint.offset == 10
    assert first.more
    second = connector.read(first.checkpoint, limit=5)
    assert second.rows == ((2,), (3,))
    assert second.checkpoint.offset == len(content)
    assert not second.more
    done = connector.read(second.checkpoint)
    assert done.rows == ()
    assert done.checkpoint == second.checkpoint
    assert done.schema.serialize()["columns"][0]["name"] == "record"
    assert connector.schema().serialize()["columns"][0]["name"] == "a"


def test_a_stream_rejects_bad_objects_and_transports():
    with pytest.raises(
        failure.ConnectorError, match="no JSON records"
    ) as raised:
        ranged(b"").schema()
    assert raised.value.code == "empty_object"
    with pytest.raises(failure.CheckpointError) as raised:
        ranged(b"{}\n").read(ranged(b"{}\n").checkpoint(5))
    assert raised.value.code == "checkpoint_beyond_object"
    with pytest.raises(failure.ConnectorError) as raised:
        ranged(b'{"a": 12345}\n', chunk=2, limit=3).read()
    assert raised.value.code == "record_too_large"
    stalled = stream.StreamingObjectStoreConnector(
        "s3://b/k.jsonl", lambda u, s, e: b"", lambda u: 5, NO_RETRY
    )
    with pytest.raises(failure.ConnectorError) as raised:
        stalled.read()
    assert raised.value.code == "range_no_progress"
    greedy = stream.StreamingObjectStoreConnector(
        "s3://b/k.jsonl", lambda u, s, e: b"x" * 10, lambda u: 5, NO_RETRY
    )
    with pytest.raises(failure.ConnectorError) as raised:
        greedy.fetch(0, 2)
    assert raised.value.details["received"] == 10
    text = stream.StreamingObjectStoreConnector(
        "s3://b/k.jsonl", lambda u, s, e: "x", lambda u: 5, NO_RETRY
    )
    with pytest.raises(failure.ConnectorError) as raised:
        text.fetch(0, 2)
    assert raised.value.code == "invalid_range_payload"


@pytest.mark.parametrize("size", [-1, True, 1.0, "5"])
def test_a_stream_checks_the_object_size(size):
    connector = stream.StreamingObjectStoreConnector(
        "s3://b/k.jsonl", lambda u, s, e: b"", lambda u: size, NO_RETRY
    )
    with pytest.raises(failure.ConnectorError) as raised:
        connector.measure()
    assert raised.value.code == "invalid_object_size"


def test_a_stream_validates_its_configuration():
    reader = lambda u, s, e: b""  # noqa: E731
    size = lambda u: 0  # noqa: E731
    with pytest.raises(failure.SchemeError):
        stream.StreamingObjectStoreConnector("file:///k.jsonl", reader, size)
    with pytest.raises(failure.ConnectorError) as raised:
        stream.StreamingObjectStoreConnector("s3://b/k.csv", reader, size)
    assert raised.value.code == "unsupported_stream_format"
    with pytest.raises(errors.KindError, match="callable"):
        stream.StreamingObjectStoreConnector("s3://b/k.ndjson", None, size)
    with pytest.raises(errors.InputError, match="chunk"):
        stream.StreamingObjectStoreConnector(
            "s3://b/k.jsonl", reader, size, chunk=0
        )
    with pytest.raises(errors.InputError, match="limit"):
        stream.StreamingObjectStoreConnector(
            "s3://b/k.jsonl", reader, size, limit=0
        )


def test_scan_stops_after_the_required_records():
    connector = ranged(b'{"a":1}\n{"a":2}\n{"a":3}\n', chunk=100)
    assert connector.scan(0, 24, 2) == [(b'{"a":1}', 8), (b'{"a":2}', 16)]
    assert connector.scan(16, 24, 5) == [(b'{"a":3}', 24)]
    assert connector.scan(24, 24, 5) == []
