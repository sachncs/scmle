"""Focused tests for dependency-free source connectors."""

import contextlib
import hashlib
import sqlite3
from urllib.parse import urlsplit

import pandas as pd
import pytest

from scml.data.readers import connectors


@pytest.fixture
def frame() -> pd.DataFrame:
    """Returns a small deterministic source table."""
    return pd.DataFrame({"id": [1, 2, 3, 4], "label": ["a", "b", "c", "d"]})


def test_source_connector_is_an_abstract_public_contract():
    """The contract cannot be used without schema and row behavior."""
    with pytest.raises(TypeError):
        connectors.SourceConnector("file:///tmp/source.csv")


def test_local_file_connector_returns_schema_rows_and_resume_checkpoint(
    tmp_path, frame
):
    """Local reads return bounded rows and portable cursor checkpoints."""
    path = tmp_path / "source.csv"
    frame.to_csv(path, index=False)
    connector = connectors.LocalFileConnector(path)

    schema = connector.schema()
    first = connector.read_rows(limit=2)
    second = connector.read_rows(first.checkpoint, limit=2)

    assert [column.name for column in schema.columns] == ["id", "label"]
    assert first.rows == ((1, "a"), (2, "b"))
    assert first.has_more
    assert first.checkpoint.offset == 2
    assert first.checkpoint.cursor
    assert second.rows == ((3, "c"), (4, "d"))
    assert not second.has_more
    assert second.checkpoint.offset == 4
    assert first.content_sha256 == connector.content_sha256()
    assert second.content_sha256 == first.content_sha256


def test_local_file_checkpoint_rejects_changed_raw_snapshot(tmp_path, frame):
    """A local resume cannot silently mix two file snapshots."""
    path = tmp_path / "source.csv"
    frame.to_csv(path, index=False)
    connector = connectors.LocalFileConnector(path)
    first = connector.read_rows(limit=2)

    frame.assign(label=["changed"] * len(frame)).to_csv(path, index=False)

    with pytest.raises(
        connectors.ConnectorError, match="content changed"
    ) as raised:
        connector.read_rows(first.checkpoint, limit=2)

    assert raised.value.code == "source_snapshot_changed"
    assert raised.value.details["checkpoint_content_sha256"] == (
        first.content_sha256
    )


def test_local_file_identity_is_deterministic_for_path_and_file_uri(tmp_path):
    """Equivalent local URI spellings identify the same logical source."""
    path = tmp_path / "source.csv"
    path.write_text("id\n1\n")

    first = connectors.LocalFileConnector(path)
    second = connectors.LocalFileConnector(path.as_uri())

    assert first.source_uri == second.source_uri
    assert first.source_identity == second.source_identity


def test_local_connector_rejects_object_store_uri_without_fake_support():
    """An object-store URI fails closed instead of pretending to be local."""
    with pytest.raises(connectors.UnsupportedUriSchemeError) as raised:
        connectors.LocalFileConnector("s3://bucket/source.csv")

    assert raised.value.code == "unsupported_uri_scheme"
    assert raised.value.details["scheme"] == "s3"
    assert raised.value.as_dict()["category"] == "data"


def test_object_store_connector_reads_bounded_tabular_objects_and_resumes(
    frame,
):
    """Injected object transport supports schema and checkpointed batches."""
    payload = frame.to_csv(index=False).encode("utf-8")
    connector = connectors.ObjectStoreConnector(
        "s3://bucket/source.csv", lambda uri: payload
    )

    first = connector.read_rows(limit=2)
    second = connector.read_rows(first.checkpoint, limit=2)

    assert connector.schema().columns[0].name == "id"
    assert first.rows == ((1, "a"), (2, "b"))
    assert second.rows == ((3, "c"), (4, "d"))
    assert not second.has_more
    assert first.content_sha256 == hashlib.sha256(payload).hexdigest()
    assert second.content_sha256 == first.content_sha256


def test_object_store_checkpoint_rejects_changed_raw_snapshot(frame):
    """A resumed object read cannot silently mix two raw snapshots."""
    payloads = [
        frame.to_csv(index=False).encode("utf-8"),
        frame.assign(label=["changed"] * len(frame))
        .to_csv(index=False)
        .encode("utf-8"),
    ]

    def read_object(uri: str) -> bytes:
        return payloads.pop(0)

    connector = connectors.ObjectStoreConnector(
        "s3://bucket/source.csv", read_object
    )
    first = connector.read_rows(limit=2)

    with pytest.raises(
        connectors.ConnectorError, match="content changed"
    ) as raised:
        connector.read_rows(first.checkpoint, limit=2)

    assert raised.value.code == "source_snapshot_changed"
    assert raised.value.details["checkpoint_content_sha256"] == (
        first.content_sha256
    )


def test_object_store_connector_accepts_a_caller_owned_versioned_decoder():
    """A custom decoder admits formats outside the built-in tabular set."""
    calls = []

    def decode(uri: str, payload: bytes) -> pd.DataFrame:
        calls.append((uri, payload))
        return pd.DataFrame({"payload_length": [len(payload)]})

    connector = connectors.ObjectStoreConnector(
        "s3://bucket/sample.custom",
        lambda uri: b"custom-format",
        decoder=decode,
        decoder_name="custom-v2",
    )

    batch = connector.read_rows(limit=1)

    assert batch.rows == ((13,),)
    assert batch.schema.columns[0].name == "payload_length"
    assert connector.source_identity == connectors.source_identity(
        "s3://bucket/sample.custom|decoder=custom-v2"
    )
    assert calls == [(connector.source_uri, b"custom-format")]


def test_object_store_connector_requires_decoder_identity_for_custom_formats():
    """Custom decoding must be bound to a reproducible source identity."""
    with pytest.raises(ValueError, match="decoder_name"):
        connectors.ObjectStoreConnector(
            "s3://bucket/sample.custom",
            lambda uri: b"custom-format",
            decoder=lambda uri, payload: pd.DataFrame({"value": [1]}),
        )


def test_object_store_connector_reports_decoder_contract_failures():
    """Decoder errors are structured rather than leaked as implementation detail."""

    def fail(uri: str, payload: bytes) -> pd.DataFrame:
        raise RuntimeError("bad payload")

    failing = connectors.ObjectStoreConnector(
        "s3://bucket/sample.custom",
        lambda uri: b"custom-format",
        decoder=fail,
        decoder_name="custom-v2",
    )

    with pytest.raises(connectors.ConnectorError) as raised:
        failing.schema()

    assert raised.value.code == "decoder_failed"
    assert raised.value.details["decoder"] == "custom-v2"

    invalid = connectors.ObjectStoreConnector(
        "s3://bucket/sample.custom",
        lambda uri: b"custom-format",
        decoder=lambda uri, payload: {"value": 1},
        decoder_name="custom-v2",
    )
    with pytest.raises(connectors.ConnectorError) as raised:
        invalid.schema()
    assert raised.value.code == "invalid_decoded_object"


def test_object_store_connector_rejects_large_payloads_and_bad_formats():
    """Bounded object support fails closed instead of hiding memory risk."""
    with pytest.raises(connectors.ConnectorError, match="max_bytes"):
        connectors.ObjectStoreConnector(
            "gs://bucket/source.csv", lambda uri: b"id\n1\n", max_bytes=1
        ).schema()


def test_streaming_object_connector_reads_ranges_and_resumes_by_byte_offset():
    """JSONL range reads do not redownload preceding records."""
    payload = (
        b'{"id":1,"label":"a"}\n{"id":2,"label":"b"}\n{"id":3,"label":"c"}\n'
    )
    requests: list[tuple[int, int]] = []

    def read_range(uri: str, start: int, end: int) -> bytes:
        assert urlsplit(uri).scheme == "s3"
        requests.append((start, end))
        return payload[start:end]

    connector = connectors.StreamingObjectStoreConnector(
        "s3://bucket/events.jsonl",
        read_range,
        lambda uri: len(payload),
        chunk_bytes=10,
    )

    first = connector.read_rows(limit=2)
    second = connector.read_rows(first.checkpoint, limit=2)

    assert first.rows == ((1, "a"), (2, "b"))
    assert second.rows == ((3, "c"),)
    assert first.checkpoint.offset == payload.index(b'{"id":3')
    assert not second.has_more
    assert requests[0][0] == 0
    assert requests[-1][0] >= first.checkpoint.offset


def test_streaming_object_connector_rejects_corrupt_and_changed_records():
    """Malformed JSON and in-object schema drift fail with structured codes."""
    corrupt = b'{"id":1}\nnot-json\n'
    corrupt_connector = connectors.StreamingObjectStoreConnector(
        "https://example.test/events.ndjson",
        lambda uri, start, end: corrupt[start:end],
        lambda uri: len(corrupt),
    )
    with pytest.raises(connectors.ConnectorError) as corrupt_failure:
        corrupt_connector.read_rows(limit=2)
    assert corrupt_failure.value.code == "invalid_json_record"

    changed = b'{"id":1}\n{"label":"a"}\n'
    changed_connector = connectors.StreamingObjectStoreConnector(
        "https://example.test/events.ndjson",
        lambda uri, start, end: changed[start:end],
        lambda uri: len(changed),
    )
    with pytest.raises(connectors.ConnectorError) as changed_failure:
        changed_connector.read_rows(limit=2)
    assert changed_failure.value.code == "schema_changed"


def test_streaming_object_connector_rejects_oversized_records():
    """A missing newline cannot grow the in-memory record buffer forever."""
    payload = b'{"id":"' + b"x" * 12 + b'"}'
    connector = connectors.StreamingObjectStoreConnector(
        "s3://bucket/events.jsonl",
        lambda uri, start, end: payload[start:end],
        lambda uri: len(payload),
        chunk_bytes=4,
        max_record_bytes=8,
    )

    with pytest.raises(connectors.ConnectorError, match="max_record_bytes"):
        connector.read_rows(limit=1)

    with pytest.raises(connectors.ConnectorError, match="unsupported"):
        connectors.ObjectStoreConnector(
            "az://bucket/source.xls", lambda uri: b"data"
        ).schema()


def test_checkpoint_cannot_cross_source_boundaries(tmp_path):
    """A checkpoint from one source cannot silently read another source."""
    first_path = tmp_path / "first.csv"
    second_path = tmp_path / "second.csv"
    first_path.write_text("id\n1\n")
    second_path.write_text("id\n2\n")
    first = connectors.LocalFileConnector(first_path)
    second = connectors.LocalFileConnector(second_path)

    checkpoint = first.checkpoint(offset=1)
    with pytest.raises(connectors.CheckpointError) as raised:
        second.read_rows(checkpoint)

    assert raised.value.code == "checkpoint_mismatch"
    assert raised.value.details["offset"] == 1


def test_dbapi_connector_retrieves_schema_and_resumes_by_cursor(tmp_path):
    """Injected SQLite connections exercise generic DB-API behavior."""
    database = tmp_path / "source.sqlite"
    with (
        contextlib.closing(sqlite3.connect(database)) as connection,
        connection,
    ):
        connection.execute("CREATE TABLE records (id INTEGER, label TEXT)")
        connection.executemany(
            "INSERT INTO records VALUES (?, ?)",
            [(1, "a"), (2, "b"), (3, "c")],
        )

    def connection_factory():
        return sqlite3.connect(database)

    connector = connectors.DbApiConnector(
        "sqlite://source",
        "SELECT id, label FROM records ORDER BY id",
        connection_factory,
    )
    first = connector.read_rows(limit=2)
    second = connector.read_rows(first.checkpoint, limit=2)

    assert connector.schema().columns[0].name == "id"
    assert first.rows == ((1, "a"), (2, "b"))
    assert second.rows == ((3, "c"),)
    assert not second.has_more


def test_dbapi_connector_rejects_unsupported_uri_scheme():
    """The generic DB-API adapter does not claim HTTP or object stores."""
    with pytest.raises(connectors.UnsupportedUriSchemeError):
        connectors.DbApiConnector(
            "s3://bucket/source",
            "SELECT 1",
            lambda: None,
        )


def test_warehouse_connector_reads_bounded_batches_through_injected_engine(
    frame,
):
    """Warehouse reads preserve query identity and portable checkpoints."""
    schema = connectors.SourceSchema.from_frame(frame)

    def read_schema(query):
        assert query == "select * from records"
        return schema

    def read_batch(query, offset, limit):
        assert query == "select * from records"
        selected = frame.iloc[offset : offset + limit]
        rows = tuple(selected.itertuples(index=False, name=None))
        return schema, rows, offset + len(rows) < len(frame)

    connector = connectors.WarehouseConnector(
        "warehouse://analytics/records",
        "select * from records",
        read_schema,
        read_batch,
    )
    first = connector.read_rows(limit=2)
    second = connector.read_rows(first.checkpoint, limit=2)

    assert connector.schema() == schema
    assert first.rows == ((1, "a"), (2, "b"))
    assert second.rows == ((3, "c"), (4, "d"))
    assert second.checkpoint.source_identity == connector.source_identity
    assert not second.has_more


def test_warehouse_connector_retries_and_rejects_invalid_callback_rows(frame):
    """Engine failures retry, while unsafe callback results fail closed."""
    attempts = []

    def read_schema(query):
        attempts.append(query)
        if len(attempts) == 1:
            raise OSError("warehouse unavailable")
        return connectors.SourceSchema.from_frame(frame)

    connector = connectors.WarehouseConnector(
        "lakehouse://bronze/records",
        "records",
        read_schema,
        lambda query, offset, limit: (
            connectors.SourceSchema.from_frame(frame),
            ((1,),),
            False,
        ),
        connectors.RetryBudget(max_retries=1),
    )

    assert connector.schema().columns[0].name == "id"
    with pytest.raises(connectors.ConnectorError, match="row width"):
        connector.read_rows(limit=1)


def test_dbapi_connector_retries_transient_factory_failure(tmp_path):
    """The retry budget counts retries after the initial operation."""
    database = tmp_path / "source.sqlite"
    with (
        contextlib.closing(sqlite3.connect(database)) as connection,
        connection,
    ):
        connection.execute("CREATE TABLE records (id INTEGER)")
        connection.execute("INSERT INTO records VALUES (1)")

    attempts = 0

    def flaky_factory():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise sqlite3.OperationalError("temporary failure")
        return sqlite3.connect(database)

    connector = connectors.DbApiConnector(
        "sqlite://source",
        "SELECT id FROM records",
        flaky_factory,
        connectors.RetryBudget(max_retries=2),
    )

    assert connector.schema().columns[0].name == "id"
    assert attempts == 3


def test_dbapi_connector_reports_structured_retry_exhaustion():
    """Exhausted retries preserve machine-readable operation evidence."""
    attempts = 0

    def failing_factory():
        nonlocal attempts
        attempts += 1
        raise sqlite3.OperationalError("database unavailable")

    connector = connectors.DbApiConnector(
        "dbapi://warehouse/source",
        "SELECT 1",
        failing_factory,
        connectors.RetryBudget(max_retries=1),
    )

    with pytest.raises(connectors.RetryExhaustedError) as raised:
        connector.schema()

    assert attempts == 2
    assert raised.value.code == "retry_exhausted"
    assert raised.value.attempts == 2
    assert raised.value.details["operation"] == "schema"
    assert raised.value.details["last_error_type"] == "OperationalError"


def test_cursor_is_deterministic_and_round_trips():
    """Cursor tokens are stable and preserve source identity and offset."""
    identity = connectors.source_identity("dbapi://warehouse/source")
    cursor = connectors.cursor_for(identity, 17)

    assert connectors.cursor_for(identity, 17) == cursor
    assert connectors.decode_cursor(cursor) == (identity, 17)
    assert connectors.SourceCheckpoint(cursor=cursor).offset == 17


def test_retry_budget_rejects_invalid_values():
    """Invalid retry policy cannot create an unbounded or unsafe connector."""
    with pytest.raises(ValueError, match="max_retries"):
        connectors.RetryBudget(max_retries=-1)
    with pytest.raises(ValueError, match="delay_seconds"):
        connectors.RetryBudget(delay_seconds=-0.1)
