"""Deterministic source connectors for files, objects, and DB-API queries.

Connectors expose the smallest useful ingestion contract: identify a logical
source, inspect its schema, and read bounded row batches from a checkpoint.
Network clients remain caller-owned: transport functions are injected and
their boundaries are recorded by the connector rather than hidden behind a
cloud SDK import.
"""

import abc
import base64
from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
import contextlib
import dataclasses
import hashlib
import io
import json
import math
import pathlib
import time
from typing import Any
from typing import TypeVar
from urllib.parse import unquote
from urllib.parse import urlsplit
from urllib.parse import urlunsplit

import pandas as pd

from scml.contracts import errors
from scml.data.readers import tabular

T = TypeVar("T")
Row = tuple[Any, ...]
RangeReader = Callable[[str, int, int], bytes]
SizeReader = Callable[[str], int]


class ConnectorError(errors.DataError):
    """A connector failure with a stable code and structured details."""

    def __init__(
        self,
        code: str,
        message: str,
        source_uri: str,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        """Initializes a structured connector failure.

        Args:
            code: Stable machine-readable failure code.
            message: Human-readable failure description.
            source_uri: Canonical source URI associated with the failure.
            details: Additional structured context for callers and logs.
        """
        self.code = code
        self.source_uri = source_uri
        self.details = {
            "code": code,
            "source_uri": source_uri,
            **dict(details or {}),
        }
        super().__init__(message, self.details)

    def as_dict(self) -> dict[str, Any]:
        """Returns a JSON-compatible representation of the failure."""
        return {
            "category": self.category,
            "code": self.code,
            "message": str(self),
            "source_uri": self.source_uri,
            "details": dict(self.details),
        }


class UnsupportedUriSchemeError(ConnectorError):
    """A connector was given a URI scheme it does not implement."""

    def __init__(
        self, source_uri: str, scheme: str, supported_schemes: Sequence[str]
    ) -> None:
        """Initializes an unsupported-scheme failure."""
        supported = tuple(supported_schemes)
        super().__init__(
            "unsupported_uri_scheme",
            f"unsupported URI scheme '{scheme or '<path>'}'; "
            f"supported schemes: {supported}",
            source_uri,
            {"scheme": scheme, "supported_schemes": list(supported)},
        )
        self.scheme = scheme
        self.supported_schemes = supported


class RetryExhaustedError(ConnectorError):
    """All attempts for a connector operation failed."""

    def __init__(
        self,
        source_uri: str,
        operation: str,
        attempts: int,
        last_error: Exception,
    ) -> None:
        """Initializes a retry exhaustion failure."""
        super().__init__(
            "retry_exhausted",
            f"{operation} failed after {attempts} attempt(s): {last_error}",
            source_uri,
            {
                "operation": operation,
                "attempts": attempts,
                "last_error_type": type(last_error).__name__,
                "last_error": str(last_error),
            },
        )
        self.operation = operation
        self.attempts = attempts
        self.last_error = last_error


class CheckpointError(ConnectorError):
    """A checkpoint cannot safely be used with a connector."""

    def __init__(
        self,
        source_uri: str,
        code: str,
        message: str,
        details: Mapping[str, Any],
    ) -> None:
        """Initializes a checkpoint failure."""
        super().__init__(code, message, source_uri, details)


@dataclasses.dataclass(frozen=True)
class RetryBudget:
    """Retry limits applied independently to each connector operation.

    Attributes:
        max_retries: Number of retries after the initial attempt.
        delay_seconds: Delay between attempts. Zero is recommended for tests.
    """

    max_retries: int = 2
    delay_seconds: float = 0.0

    def __post_init__(self) -> None:
        """Validates retry settings before a connector uses them."""
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if not math.isfinite(self.delay_seconds) or self.delay_seconds < 0:
            raise ValueError("delay_seconds must be finite and non-negative")

    @property
    def max_attempts(self) -> int:
        """Returns the initial attempt plus the retry budget."""
        return self.max_retries + 1


def canonical_source_uri(source_uri: str | pathlib.Path) -> str:
    """Returns a deterministic URI without changing source semantics.

    Bare paths are treated as local files. URI schemes are normalized to
    lower-case, while query strings and fragments remain part of identity.
    """
    value = str(source_uri).strip()
    if not value:
        raise ValueError("source_uri must not be empty")
    parsed = urlsplit(value)
    if parsed.scheme and not (len(parsed.scheme) == 1 and value[1:2] == ":"):
        return urlunsplit(
            (
                parsed.scheme.lower(),
                parsed.netloc,
                parsed.path,
                parsed.query,
                parsed.fragment,
            )
        )
    return pathlib.Path(value).expanduser().resolve(strict=False).as_uri()


def source_identity(source_uri: str | pathlib.Path) -> str:
    """Returns a stable identity for a logical source URI."""
    canonical = canonical_source_uri(source_uri)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"src_{digest}"


def cursor_for(source_identity_value: str, offset: int) -> str:
    """Creates a deterministic opaque cursor for a source offset."""
    if not source_identity_value:
        raise ValueError("source_identity_value must not be empty")
    if offset < 0:
        raise ValueError("offset must be non-negative")
    payload = f"{source_identity_value}:{offset}".encode()
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def decode_cursor(cursor: str) -> tuple[str, int]:
    """Decodes a cursor into its source identity and row offset."""
    if not cursor:
        raise ValueError("cursor must not be empty")
    padding = "=" * (-len(cursor) % 4)
    try:
        payload = base64.urlsafe_b64decode(cursor + padding).decode("utf-8")
        identity, offset_value = payload.rsplit(":", maxsplit=1)
        offset = int(offset_value)
    except (ValueError, UnicodeDecodeError) as error:
        raise ValueError("cursor is not a valid connector cursor") from error
    if not identity or offset < 0:
        raise ValueError("cursor contains an invalid source or offset")
    return identity, offset


@dataclasses.dataclass(frozen=True)
class SourceCheckpoint:
    """A portable source-position checkpoint bound to a source identity.

    A row-oriented connector uses row offsets; a range-oriented connector may
    use byte offsets. The unit is defined by the connector contract. A
    checkpoint can be supplied by explicit ``offset`` or by the opaque
    ``cursor`` returned in a previous batch. When both are present they must
    agree. A checkpoint returned by a connector always contains both values.
    """

    offset: int = 0
    cursor: str | None = None
    source_identity: str = ""

    def __post_init__(self) -> None:
        """Validates and normalizes cursor-derived checkpoint fields."""
        if self.offset < 0:
            raise ValueError("offset must be non-negative")
        if self.cursor is None:
            return
        cursor_identity, cursor_offset = decode_cursor(self.cursor)
        if self.source_identity and self.source_identity != cursor_identity:
            raise ValueError("cursor source identity does not match checkpoint")
        if self.offset and self.offset != cursor_offset:
            raise ValueError("cursor offset does not match checkpoint offset")
        if not self.source_identity:
            object.__setattr__(self, "source_identity", cursor_identity)
        if not self.offset:
            object.__setattr__(self, "offset", cursor_offset)

    def bind(self, identity: str) -> "SourceCheckpoint":
        """Binds an unbound checkpoint to ``identity`` safely."""
        if self.source_identity and self.source_identity != identity:
            raise ValueError("checkpoint belongs to a different source")
        return SourceCheckpoint(
            offset=self.offset,
            cursor=self.cursor or cursor_for(identity, self.offset),
            source_identity=identity,
        )

    def advance(self, rows: int) -> "SourceCheckpoint":
        """Returns the checkpoint after consuming ``rows`` rows."""
        if rows < 0:
            raise ValueError("rows must be non-negative")
        identity = self.source_identity
        if not identity:
            raise ValueError("checkpoint must be bound before advancing")
        next_offset = self.offset + rows
        return SourceCheckpoint(
            offset=next_offset,
            cursor=cursor_for(identity, next_offset),
            source_identity=identity,
        )


@dataclasses.dataclass(frozen=True)
class SourceColumn:
    """One source column and its source-reported type name."""

    name: str
    type_name: str

    def __post_init__(self) -> None:
        """Validates column identity."""
        if not self.name:
            raise ValueError("column name must not be empty")
        if not self.type_name:
            raise ValueError("column type_name must not be empty")


@dataclasses.dataclass(frozen=True)
class SourceSchema:
    """Ordered schema metadata returned by a source connector."""

    columns: tuple[SourceColumn, ...]

    def __post_init__(self) -> None:
        """Rejects empty and duplicate column declarations."""
        names = tuple(column.name for column in self.columns)
        if not names:
            raise ValueError("source schema must contain at least one column")
        if len(names) != len(set(names)):
            raise ValueError("source schema contains duplicate column names")

    def as_dict(self) -> dict[str, Any]:
        """Returns deterministic JSON-compatible schema metadata."""
        return {
            "columns": [
                {"name": column.name, "type_name": column.type_name}
                for column in self.columns
            ]
        }

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> "SourceSchema":
        """Builds a source schema from a pandas frame."""
        return cls(
            tuple(
                SourceColumn(str(name), str(dtype))
                for name, dtype in zip(frame.columns, frame.dtypes, strict=True)
            )
        )


@dataclasses.dataclass(frozen=True)
class RowBatch:
    """A bounded row result and its resume checkpoint."""

    schema: SourceSchema
    rows: tuple[Row, ...]
    checkpoint: SourceCheckpoint
    has_more: bool


def execute_with_retries(
    operation: Callable[[], T],
    operation_name: str,
    source_uri: str,
    retry_budget: RetryBudget,
) -> T:
    """Executes an operation until it succeeds or its retry budget is spent.

    Args:
        operation: Zero-argument operation to execute.
        operation_name: Stable operation name for failure evidence.
        source_uri: Canonical source URI for failure evidence.
        retry_budget: Retry policy for this operation.

    Returns:
        The operation result.

    Raises:
        RetryExhaustedError: If every attempt fails.
        ConnectorError: If the operation raises a structured connector error.
    """
    for attempt in range(1, retry_budget.max_attempts + 1):
        try:
            return operation()
        except ConnectorError:
            raise
        except Exception as error:
            if attempt == retry_budget.max_attempts:
                raise RetryExhaustedError(
                    source_uri, operation_name, attempt, error
                ) from error
            if retry_budget.delay_seconds:
                time.sleep(retry_budget.delay_seconds)
    raise AssertionError("retry loop must return or raise")


class SourceConnector(abc.ABC):
    """Public polymorphic contract for bounded source retrieval."""

    def __init__(
        self,
        source_uri: str | pathlib.Path,
        retry_budget: RetryBudget | None = None,
    ) -> None:
        """Initializes a connector identity and retry policy."""
        self.source_uri = canonical_source_uri(source_uri)
        self.source_identity = source_identity(self.source_uri)
        self.retry_budget = retry_budget or RetryBudget()

    @abc.abstractmethod
    def schema(self) -> SourceSchema:
        """Returns the ordered schema without consuming a row checkpoint."""

    @abc.abstractmethod
    def read_rows(
        self, checkpoint: SourceCheckpoint | None = None, limit: int = 1_000
    ) -> RowBatch:
        """Reads at most ``limit`` rows from a checkpoint."""

    def checkpoint(self, offset: int = 0) -> SourceCheckpoint:
        """Creates a source-bound checkpoint for explicit offset reads."""
        return SourceCheckpoint(
            offset=offset,
            cursor=cursor_for(self.source_identity, offset),
            source_identity=self.source_identity,
        )

    def validate_checkpoint(
        self, checkpoint: SourceCheckpoint | None
    ) -> SourceCheckpoint:
        """Binds and validates a checkpoint before reading rows."""
        value = checkpoint or SourceCheckpoint()
        try:
            return value.bind(self.source_identity)
        except ValueError as error:
            raise CheckpointError(
                self.source_uri,
                "checkpoint_mismatch",
                str(error),
                {
                    "checkpoint_source_identity": value.source_identity,
                    "connector_source_identity": self.source_identity,
                    "offset": value.offset,
                },
            ) from error

    def validate_limit(self, limit: int) -> None:
        """Rejects unbounded or invalid batch requests."""
        if limit <= 0:
            raise ValueError("limit must be positive")


class LocalFileConnector(SourceConnector):
    """Connector for tabular local paths and ``file://`` URIs."""

    supported_schemes = ("", "file")

    def __init__(
        self,
        source_uri: str | pathlib.Path,
        retry_budget: RetryBudget | None = None,
    ) -> None:
        """Initializes a local-file connector.

        Raises:
            UnsupportedUriSchemeError: If ``source_uri`` is not local.
        """
        value = str(source_uri).strip()
        parsed = urlsplit(value)
        scheme = (
            ""
            if len(parsed.scheme) == 1 and value[1:2] == ":"
            else parsed.scheme
        )
        if scheme.lower() not in self.supported_schemes:
            raise UnsupportedUriSchemeError(
                value, scheme.lower(), self.supported_schemes
            )
        if parsed.scheme:
            if parsed.netloc not in ("", "localhost"):
                raise UnsupportedUriSchemeError(
                    value, parsed.scheme.lower(), ("file",)
                )
            path = pathlib.Path(unquote(parsed.path))
        else:
            path = pathlib.Path(value)
        self.path = path.expanduser().resolve(strict=False)
        super().__init__(self.path, retry_budget)

    def schema(self) -> SourceSchema:
        """Reads and returns the local file schema."""
        frame = execute_with_retries(
            lambda: tabular.read_table(self.path),
            "schema",
            self.source_uri,
            self.retry_budget,
        )
        return SourceSchema.from_frame(frame)

    def read_rows(
        self, checkpoint: SourceCheckpoint | None = None, limit: int = 1_000
    ) -> RowBatch:
        """Reads a bounded local-file batch from an offset checkpoint."""
        self.validate_limit(limit)
        value = self.validate_checkpoint(checkpoint)

        def read() -> RowBatch:
            frame = tabular.read_table(self.path)
            schema = SourceSchema.from_frame(frame)
            start = value.offset
            selected = frame.iloc[start : start + limit]
            rows = tuple(selected.itertuples(index=False, name=None))
            next_checkpoint = value.advance(len(rows))
            return RowBatch(
                schema=schema,
                rows=rows,
                checkpoint=next_checkpoint,
                has_more=start + len(rows) < len(frame),
            )

        return execute_with_retries(
            read,
            "read_rows",
            self.source_uri,
            self.retry_budget,
        )


class DbApiConnector(SourceConnector):
    """Connector for a query executed through an injected DB-API factory.

    The factory is called for each operation and must return a DB-API 2.0
    connection. This keeps the connector dependency-free and makes lifecycle,
    retry, and failure behavior explicit to the caller.
    """

    supported_schemes = ("dbapi", "sqlite")

    def __init__(
        self,
        source_uri: str,
        query: str,
        connection_factory: Callable[[], Any],
        retry_budget: RetryBudget | None = None,
    ) -> None:
        """Initializes a DB-API connector.

        Args:
            source_uri: Logical ``dbapi://`` or ``sqlite://`` source URI.
            query: Read-only SQL query whose rows form the source.
            connection_factory: Callable returning a fresh DB-API connection.
            retry_budget: Per-operation retry policy.

        Raises:
            UnsupportedUriSchemeError: If the URI scheme is not supported.
            ValueError: If the query or factory is invalid.
        """
        value = source_uri.strip()
        parsed = urlsplit(value)
        scheme = parsed.scheme.lower()
        if scheme not in self.supported_schemes:
            raise UnsupportedUriSchemeError(
                value, scheme, self.supported_schemes
            )
        if not query.strip():
            raise ValueError("query must not be empty")
        if not callable(connection_factory):
            raise TypeError("connection_factory must be callable")
        self.query = query
        self.connection_factory = connection_factory
        super().__init__(value, retry_budget)
        self.source_identity = source_identity(
            f"{self.source_uri}|query={self.query}"
        )

    def schema(self) -> SourceSchema:
        """Executes the query and returns DB-API column metadata."""
        return execute_with_retries(
            self.retrieve_schema,
            "schema",
            self.source_uri,
            self.retry_budget,
        )

    def retrieve_schema(self) -> SourceSchema:
        """Performs one non-retried DB-API schema operation."""
        connection = self.connection_factory()
        cursor: Any | None = None
        try:
            cursor = connection.cursor()
            cursor.execute(self.query)
            return self.schema_from_description(cursor.description)
        finally:
            self.close_resources(cursor, connection)

    def read_rows(
        self, checkpoint: SourceCheckpoint | None = None, limit: int = 1_000
    ) -> RowBatch:
        """Reads DB-API rows using portable offset/cursor checkpoints."""
        self.validate_limit(limit)
        value = self.validate_checkpoint(checkpoint)
        return execute_with_retries(
            lambda: self.retrieve_rows(value, limit),
            "read_rows",
            self.source_uri,
            self.retry_budget,
        )

    def retrieve_rows(
        self, checkpoint: SourceCheckpoint, limit: int
    ) -> RowBatch:
        """Performs one non-retried DB-API row operation."""
        connection = self.connection_factory()
        cursor: Any | None = None
        try:
            cursor = connection.cursor()
            cursor.execute(self.query)
            schema = self.schema_from_description(cursor.description)
            self.skip_rows(cursor, checkpoint.offset, limit)
            fetched = cursor.fetchmany(limit + 1)
            rows = tuple(tuple(row) for row in fetched[:limit])
            next_checkpoint = checkpoint.advance(len(rows))
            return RowBatch(
                schema=schema,
                rows=rows,
                checkpoint=next_checkpoint,
                has_more=len(fetched) > limit,
            )
        finally:
            self.close_resources(cursor, connection)

    def schema_from_description(
        self, description: Sequence[Sequence[Any]] | None
    ) -> SourceSchema:
        """Converts DB-API cursor metadata into a validated schema."""
        if not description:
            raise ValueError("query did not provide a result schema")
        return SourceSchema(
            tuple(
                SourceColumn(
                    str(column[0]),
                    str(column[1])
                    if len(column) > 1 and column[1]
                    else "unknown",
                )
                for column in description
            )
        )

    def skip_rows(self, cursor: Any, offset: int, batch_size: int) -> None:
        """Consumes rows before a checkpoint without assuming SQL syntax."""
        remaining = offset
        while remaining:
            skipped = cursor.fetchmany(min(remaining, batch_size))
            if not skipped:
                return
            remaining -= len(skipped)

    def close_resources(self, cursor: Any, connection: Any) -> None:
        """Closes resources without masking the primary operation error."""
        if cursor is not None:
            with contextlib.suppress(Exception):
                cursor.close()
        with contextlib.suppress(Exception):
            connection.close()


class WarehouseConnector(SourceConnector):
    """Connector for caller-owned warehouse or lakehouse query engines.

    The connector injects the query engine boundary instead of importing a
    warehouse SDK. A DuckDB, Spark, BigQuery, Snowflake, or lakehouse
    integration can provide the two callbacks while this contract owns
    bounded reads, retries, checkpoints, schema validation, and failure
    evidence.
    """

    supported_schemes = ("warehouse", "lakehouse")

    def __init__(
        self,
        source_uri: str,
        query: str,
        schema_reader: Callable[[str], SourceSchema],
        batch_reader: Callable[
            [str, int, int], tuple[SourceSchema, Sequence[Row], bool]
        ],
        retry_budget: RetryBudget | None = None,
    ) -> None:
        """Initialize a warehouse connector around injected query callbacks.

        Args:
            source_uri: Logical ``warehouse://`` or ``lakehouse://`` URI.
            query: Versioned query or table expression used by the engine.
            schema_reader: Callback returning the source schema for ``query``.
            batch_reader: Callback accepting query, row offset, and row limit.
            retry_budget: Per-operation retry policy.
        """
        value = source_uri.strip()
        scheme = urlsplit(value).scheme.lower()
        if scheme not in self.supported_schemes:
            raise UnsupportedUriSchemeError(
                value, scheme, self.supported_schemes
            )
        if not query.strip():
            raise ValueError("query must not be empty")
        if not callable(schema_reader) or not callable(batch_reader):
            raise TypeError("schema_reader and batch_reader must be callable")
        self.query = query
        self.schema_reader = schema_reader
        self.batch_reader = batch_reader
        super().__init__(value, retry_budget)
        self.source_identity = source_identity(
            f"{self.source_uri}|query={self.query}"
        )

    def schema(self) -> SourceSchema:
        """Read and validate the schema without consuming a checkpoint."""
        return execute_with_retries(
            lambda: self.validate_schema(self.schema_reader(self.query)),
            "schema",
            self.source_uri,
            self.retry_budget,
        )

    def read_rows(
        self, checkpoint: SourceCheckpoint | None = None, limit: int = 1_000
    ) -> RowBatch:
        """Read one bounded batch from the injected warehouse engine."""
        self.validate_limit(limit)
        value = self.validate_checkpoint(checkpoint)
        return execute_with_retries(
            lambda: self.read_batch(value, limit),
            "read_rows",
            self.source_uri,
            self.retry_budget,
        )

    def read_batch(self, checkpoint: SourceCheckpoint, limit: int) -> RowBatch:
        """Validate one callback result and bind its next checkpoint."""
        schema, raw_rows, has_more = self.batch_reader(
            self.query, checkpoint.offset, limit
        )
        schema = self.validate_schema(schema)
        if not isinstance(has_more, bool):
            raise ConnectorError(
                "invalid_batch_more_flag",
                "batch reader has_more must be bool",
                self.source_uri,
                {"value_type": type(has_more).__name__},
            )
        rows = tuple(tuple(row) for row in raw_rows)
        if len(rows) > limit:
            raise ConnectorError(
                "batch_limit_exceeded",
                "batch reader returned more rows than requested",
                self.source_uri,
                {"limit": limit, "received": len(rows)},
            )
        width = len(schema.columns)
        if any(len(row) != width for row in rows):
            raise ConnectorError(
                "row_width_mismatch",
                "warehouse row width does not match its schema",
                self.source_uri,
                {"expected": width},
            )
        return RowBatch(
            schema=schema,
            rows=rows,
            checkpoint=checkpoint.advance(len(rows)),
            has_more=has_more,
        )

    def validate_schema(self, schema: SourceSchema) -> SourceSchema:
        """Validate a callback schema at the connector boundary."""
        if not isinstance(schema, SourceSchema):
            raise ConnectorError(
                "invalid_schema",
                "schema reader must return SourceSchema",
                self.source_uri,
                {"schema_type": type(schema).__name__},
            )
        return schema


def read_tabular_bytes(source_uri: str, payload: bytes) -> pd.DataFrame:
    """Decode one supported tabular object from bytes.

    The object-store connector accepts a byte-reader instead of importing a
    cloud SDK. Credentials and transport-level retries remain caller-owned.
    """
    name = pathlib.PurePosixPath(urlsplit(source_uri).path).name.lower()
    if name.endswith((".csv", ".csv.gz", ".csv.bz2", ".csv.xz")):
        if name.endswith(".gz"):
            return pd.read_csv(io.BytesIO(payload), compression="gzip")
        if name.endswith(".bz2"):
            return pd.read_csv(io.BytesIO(payload), compression="bz2")
        if name.endswith(".xz"):
            return pd.read_csv(io.BytesIO(payload), compression="xz")
        return pd.read_csv(io.BytesIO(payload))
    if name.endswith((".jsonl", ".ndjson", ".jsonl.gz", ".ndjson.gz")):
        if name.endswith(".gz"):
            return pd.read_json(
                io.BytesIO(payload), lines=True, compression="gzip"
            )
        return pd.read_json(io.BytesIO(payload), lines=True)
    if name.endswith((".parquet", ".pq")):
        return pd.read_parquet(io.BytesIO(payload))
    raise ConnectorError(
        "unsupported_object_format",
        f"unsupported tabular object format: {name or '<unknown>'}",
        source_uri,
        {"supported_formats": ["csv", "jsonl", "parquet"]},
    )


class ObjectStoreConnector(SourceConnector):
    """Connector for bounded object-store reads through an injected reader.

    This adapter downloads one object into memory before decoding it and
    rejects objects over max_bytes. It supports deterministic small-object
    ingestion, but does not claim range reads or streaming-scale semantics.
    """

    supported_schemes = ("s3", "gs", "az", "abfs", "http", "https")

    def __init__(
        self,
        source_uri: str,
        read_bytes: Callable[[str], bytes],
        retry_budget: RetryBudget | None = None,
        max_bytes: int = 256 * 1024 * 1024,
    ) -> None:
        """Initialize an object connector with explicit transport injection."""
        value = source_uri.strip()
        scheme = urlsplit(value).scheme.lower()
        if scheme not in self.supported_schemes:
            raise UnsupportedUriSchemeError(
                value, scheme, self.supported_schemes
            )
        if not callable(read_bytes):
            raise TypeError("read_bytes must be callable")
        if max_bytes < 1:
            raise ValueError("max_bytes must be positive")
        self.read_bytes = read_bytes
        self.max_bytes = max_bytes
        super().__init__(value, retry_budget)

    def load_frame(self) -> pd.DataFrame:
        """Download and decode the bounded object once."""
        payload = execute_with_retries(
            lambda: self.read_bytes(self.source_uri),
            "read_object",
            self.source_uri,
            self.retry_budget,
        )
        if not isinstance(payload, bytes):
            raise ConnectorError(
                "invalid_object_payload",
                "object reader must return bytes",
                self.source_uri,
                {"payload_type": type(payload).__name__},
            )
        if len(payload) > self.max_bytes:
            raise ConnectorError(
                "object_too_large",
                f"object is larger than max_bytes={self.max_bytes}",
                self.source_uri,
                {"bytes": len(payload), "max_bytes": self.max_bytes},
            )
        return read_tabular_bytes(self.source_uri, payload)

    def schema(self) -> SourceSchema:
        """Read and return the bounded object's schema."""
        return SourceSchema.from_frame(self.load_frame())

    def read_rows(
        self, checkpoint: SourceCheckpoint | None = None, limit: int = 1_000
    ) -> RowBatch:
        """Read a bounded row batch from an object checkpoint."""
        self.validate_limit(limit)
        value = self.validate_checkpoint(checkpoint)
        frame = self.load_frame()
        schema = SourceSchema.from_frame(frame)
        selected = frame.iloc[value.offset : value.offset + limit]
        rows = tuple(selected.itertuples(index=False, name=None))
        next_checkpoint = value.advance(len(rows))
        return RowBatch(
            schema=schema,
            rows=rows,
            checkpoint=next_checkpoint,
            has_more=value.offset + len(rows) < len(frame),
        )


class StreamingObjectStoreConnector(SourceConnector):
    """Range-backed connector for bounded JSONL and NDJSON objects.

    The injected range reader uses half-open byte ranges ``[start, end)``.
    Checkpoint offsets are byte positions at record boundaries, so a resumed
    read never needs to download or decode the preceding object content.
    The connector intentionally supports line-oriented formats only; quoted
    CSV records and binary containers need a format-specific streaming adapter.
    """

    supported_schemes = ObjectStoreConnector.supported_schemes

    def __init__(
        self,
        source_uri: str,
        read_range: RangeReader,
        size_bytes: SizeReader,
        retry_budget: RetryBudget | None = None,
        chunk_bytes: int = 1024 * 1024,
        max_record_bytes: int = 64 * 1024 * 1024,
    ) -> None:
        """Initialize a range-backed object connector.

        Args:
            source_uri: Object URI ending in ``.jsonl`` or ``.ndjson``.
            read_range: Caller-owned byte-range transport.
            size_bytes: Caller-owned object-size lookup.
            retry_budget: Per-operation retry policy.
            chunk_bytes: Maximum bytes requested per range call.
            max_record_bytes: Maximum buffered line size.
        """
        value = source_uri.strip()
        scheme = urlsplit(value).scheme.lower()
        suffix = pathlib.PurePosixPath(urlsplit(value).path).name.lower()
        if scheme not in self.supported_schemes:
            raise UnsupportedUriSchemeError(
                value, scheme, self.supported_schemes
            )
        if not suffix.endswith((".jsonl", ".ndjson")):
            raise ConnectorError(
                "unsupported_stream_format",
                "streaming object connector requires JSONL or NDJSON",
                value,
                {"supported_formats": ["jsonl", "ndjson"]},
            )
        if not callable(read_range) or not callable(size_bytes):
            raise TypeError("read_range and size_bytes must be callable")
        if chunk_bytes < 1:
            raise ValueError("chunk_bytes must be positive")
        if max_record_bytes < 1:
            raise ValueError("max_record_bytes must be positive")
        self.read_range = read_range
        self.size_bytes = size_bytes
        self.chunk_bytes = chunk_bytes
        self.max_record_bytes = max_record_bytes
        super().__init__(value, retry_budget)

    def object_size(self) -> int:
        """Return a non-negative object size through the retry policy."""
        size = execute_with_retries(
            lambda: self.size_bytes(self.source_uri),
            "head_object",
            self.source_uri,
            self.retry_budget,
        )
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ConnectorError(
                "invalid_object_size",
                "object size reader must return a non-negative integer",
                self.source_uri,
                {"size_type": type(size).__name__, "size": size},
            )
        return size

    def read_bytes(self, start: int, end: int) -> bytes:
        """Read one validated half-open byte range."""
        payload = execute_with_retries(
            lambda: self.read_range(self.source_uri, start, end),
            "read_range",
            self.source_uri,
            self.retry_budget,
        )
        if not isinstance(payload, bytes):
            raise ConnectorError(
                "invalid_range_payload",
                "range reader must return bytes",
                self.source_uri,
                {"payload_type": type(payload).__name__},
            )
        requested = end - start
        if len(payload) > requested:
            raise ConnectorError(
                "range_overread",
                "range reader returned more bytes than requested",
                self.source_uri,
                {"requested": requested, "received": len(payload)},
            )
        return payload

    def schema(self) -> SourceSchema:
        """Read one bounded record to discover the source schema."""
        batch = self.read_rows(limit=1)
        if not batch.rows:
            raise ConnectorError(
                "empty_object",
                "streaming object contains no JSON records",
                self.source_uri,
            )
        return batch.schema

    def read_rows(
        self, checkpoint: SourceCheckpoint | None = None, limit: int = 1_000
    ) -> RowBatch:
        """Read JSON records from a byte checkpoint without full download."""
        self.validate_limit(limit)
        value = self.validate_checkpoint(checkpoint)
        total = self.object_size()
        if value.offset > total:
            raise CheckpointError(
                self.source_uri,
                "checkpoint_beyond_object",
                "checkpoint offset is beyond the current object size",
                {"offset": value.offset, "size": total},
            )
        records = self.read_record_lines(value.offset, total, limit + 1)
        if not records:
            return RowBatch(
                schema=SourceSchema((SourceColumn("record", "object"),)),
                rows=(),
                checkpoint=value,
                has_more=False,
            )
        selected = records[:limit]
        schema, rows = self.decode_records(selected)
        next_offset = selected[-1][1]
        next_checkpoint = value.bind(self.source_identity).advance(
            next_offset - value.offset
        )
        return RowBatch(
            schema=schema,
            rows=rows,
            checkpoint=next_checkpoint,
            has_more=len(records) > limit,
        )

    def read_record_lines(
        self, start: int, total: int, required_records: int
    ) -> list[tuple[bytes, int]]:
        """Read complete newline-delimited records from a byte position."""
        records: list[tuple[bytes, int]] = []
        buffer = b""
        buffer_origin = start
        position = start
        while position < total and len(records) < required_records:
            end = min(total, position + self.chunk_bytes)
            payload = self.read_bytes(position, end)
            if not payload:
                raise ConnectorError(
                    "range_no_progress",
                    "range reader returned no bytes before object end",
                    self.source_uri,
                    {"offset": position, "requested_end": end},
                )
            buffer += payload
            position += len(payload)
            while b"\n" in buffer and len(records) < required_records:
                newline = buffer.index(b"\n")
                line = buffer[:newline].rstrip(b"\r")
                line_end = buffer_origin + newline + 1
                buffer = buffer[newline + 1 :]
                buffer_origin = line_end
                if line:
                    records.append((line, line_end))
            if len(buffer) > self.max_record_bytes:
                raise ConnectorError(
                    "record_too_large",
                    "a JSON record exceeds max_record_bytes",
                    self.source_uri,
                    {
                        "max_record_bytes": self.max_record_bytes,
                        "offset": buffer_origin,
                    },
                )
        if buffer and position >= total and len(records) < required_records:
            records.append((buffer, position))
        return records

    def decode_records(
        self, records: Sequence[tuple[bytes, int]]
    ) -> tuple[SourceSchema, tuple[Row, ...]]:
        """Decode JSON objects and reject silent schema drift."""
        decoded: list[Mapping[str, Any]] = []
        lines = [record[0] for record in records]
        for line in lines:
            try:
                record = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ConnectorError(
                    "invalid_json_record",
                    "streaming object contains invalid UTF-8 JSON",
                    self.source_uri,
                    {"error_type": type(error).__name__},
                ) from error
            if not isinstance(record, Mapping):
                raise ConnectorError(
                    "record_not_object",
                    "streaming JSON records must be objects",
                    self.source_uri,
                    {"record_type": type(record).__name__},
                )
            decoded.append(record)
        names = tuple(decoded[0])
        schema = SourceSchema(
            tuple(
                SourceColumn(name, self.json_type_name(decoded[0][name]))
                for name in names
            )
        )
        rows: list[Row] = []
        for record in decoded:
            if tuple(record) != names:
                raise ConnectorError(
                    "schema_changed",
                    "streaming JSON record fields changed within the object",
                    self.source_uri,
                    {"expected": list(names), "observed": list(record)},
                )
            rows.append(tuple(record[name] for name in names))
        return schema, tuple(rows)

    def json_type_name(self, value: Any) -> str:
        """Return a stable source type name for one JSON value."""
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "bool"
        if isinstance(value, int):
            return "int"
        if isinstance(value, float):
            return "float"
        if isinstance(value, str):
            return "str"
        if isinstance(value, list):
            return "array"
        if isinstance(value, Mapping):
            return "object"
        return type(value).__name__
