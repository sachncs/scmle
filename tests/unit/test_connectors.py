"""Focused tests for dependency-free source connectors."""

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
    with sqlite3.connect(database) as connection:
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


def test_dbapi_connector_retries_transient_factory_failure(tmp_path):
    """The retry budget counts retries after the initial operation."""
    database = tmp_path / "source.sqlite"
    with sqlite3.connect(database) as connection:
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
