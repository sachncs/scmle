"""Deterministic source connectors for local files and DB-API queries.

Connectors expose the smallest useful ingestion contract: identify a logical
source, inspect its schema, and read bounded row batches from a checkpoint.
Object-store and network-client support deliberately remain outside this
module; accepting a URI is not the same as implementing its transport.
"""

import abc
import base64
from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
import contextlib
import dataclasses
import hashlib
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
    """A portable row checkpoint bound to a source identity.

    A checkpoint can be supplied by explicit ``offset`` or by the opaque
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
