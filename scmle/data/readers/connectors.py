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
from collections.abc import Iterator
from collections.abc import Mapping
from collections.abc import Sequence
import contextlib
import dataclasses
import hashlib
import io
import json
import math
import pathlib
import re
import sqlite3
import time
from typing import Any
from typing import cast
from typing import Literal
from typing import TypeVar
from urllib.parse import urlsplit
from urllib.parse import urlunsplit
from urllib.request import url2pathname

import pandas

from scmle.contracts import errors
from scmle.data.readers import tabular

T = TypeVar("T")

SHA256_HEX_LENGTH = 64
CSV_COMPRESSION: dict[str, Literal["gzip", "bz2", "xz"]] = {
    ".gz": "gzip",
    ".bz2": "bz2",
    ".xz": "xz",
}
JSON_TYPE_NAMES = (
    (bool, "bool"),
    (int, "int"),
    (float, "float"),
    (str, "str"),
    (list, "array"),
    (Mapping, "object"),
)
Row = tuple[Any, ...]
RangeReader = Callable[[str, int, int], bytes]
SizeReader = Callable[[str], int]
ObjectDecoder = Callable[[str, bytes], pandas.DataFrame]


@dataclasses.dataclass(frozen=True, slots=True)
class ObjectPayload:
    """Bounded raw object bytes with a stable content identity.

    Attributes:
        source_uri: Canonical URI the bytes were read from.
        payload: The raw object bytes.
        content_sha256: SHA-256 hex digest of `payload`, derived on
            construction.
        byte_size: Length of `payload` in bytes, derived on construction.
    """

    source_uri: str
    payload: bytes = dataclasses.field(repr=False)
    content_sha256: str = dataclasses.field(init=False)
    byte_size: int = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        """Derives the digest and size used by checkpoints.

        Raises:
            ValueError: If `source_uri` is blank.
            TypeError: If `payload` is not ``bytes``.
        """
        if not self.source_uri.strip():
            raise ValueError("source_uri must not be empty")
        if not isinstance(self.payload, bytes):
            raise TypeError("payload must be bytes")
        object.__setattr__(
            self,
            "content_sha256",
            hashlib.sha256(self.payload).hexdigest(),
        )
        object.__setattr__(self, "byte_size", len(self.payload))


class ConnectorError(errors.DataError):
    """A connector failure with a stable code and structured details.

    Attributes:
        code: Stable machine-readable failure code.
        source_uri: Canonical source URI associated with the failure.
        details: Structured context; always includes ``code`` and
            ``source_uri`` plus any caller-supplied entries.
    """

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
        """Serializes the failure.

        Returns:
            A JSON-compatible mapping with the error ``category``,
            ``code``, ``message``, ``source_uri`` and a copy of
            ``details``.
        """
        return {
            "category": self.category,
            "code": self.code,
            "message": str(self),
            "source_uri": self.source_uri,
            "details": dict(self.details),
        }


class UnsupportedUriSchemeError(ConnectorError):
    """A connector was given a URI scheme it does not implement.

    Attributes:
        scheme: The rejected scheme (empty for a bare path).
        supported_schemes: The schemes the connector accepts.
    """

    def __init__(
        self, source_uri: str, scheme: str, supported_schemes: Sequence[str]
    ) -> None:
        """Initializes an unsupported-scheme failure.

        Args:
            source_uri: The URI that was rejected.
            scheme: Its scheme (empty for a bare path).
            supported_schemes: Schemes the connector accepts.
        """
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
    """All attempts for a connector operation failed.

    Attributes:
        operation: Name of the failed operation.
        attempts: Number of attempts made.
        last_error: The exception raised by the final attempt.
    """

    def __init__(
        self,
        source_uri: str,
        operation: str,
        attempts: int,
        last_error: Exception,
    ) -> None:
        """Initializes a retry exhaustion failure.

        Args:
            source_uri: Canonical source URI of the operation.
            operation: Name of the failed operation.
            attempts: Number of attempts made.
            last_error: The exception raised by the final attempt.
        """
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
        """Initializes a checkpoint failure.

        Args:
            source_uri: Canonical source URI of the connector.
            code: Stable machine-readable failure code.
            message: Human-readable failure description.
            details: Structured context for callers and logs.
        """
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
        """Validates retry settings before a connector uses them.

        Raises:
            ValueError: If `max_retries` is negative or `delay_seconds` is
                negative or not finite.
        """
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if not math.isfinite(self.delay_seconds) or self.delay_seconds < 0:
            raise ValueError("delay_seconds must be finite and non-negative")

    @property
    def max_attempts(self) -> int:
        """The total number of attempts: the initial attempt plus retries."""
        return self.max_retries + 1


def canonical_source_uri(source_uri: str | pathlib.Path) -> str:
    """Returns a deterministic URI without changing source semantics.

    Bare paths are treated as local files. URI schemes are normalized to
    lower-case, while query strings and fragments remain part of identity.

    Args:
        source_uri: A URI or a local path (a single-letter drive prefix such
            as ``C:`` is treated as a path, not a scheme).

    Returns:
        The URI with a lower-cased scheme, or, for a path, the absolute
        ``file://`` URI after expanding ``~`` and resolving the path.

    Raises:
        ValueError: If `source_uri` is blank.
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
    """Returns a stable identity for a logical source URI.

    Args:
        source_uri: A URI or local path; an empty value is rejected by
            `canonical_source_uri`.

    Returns:
        ``src_`` followed by the SHA-256 hex digest of the canonical URI.
    """
    canonical = canonical_source_uri(source_uri)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"src_{digest}"


def cursor_for(source_identity_value: str, offset: int) -> str:
    """Creates a deterministic opaque cursor for a source offset.

    Args:
        source_identity_value: Identity of the source the offset belongs to.
        offset: Position in the source, in the connector's unit (rows or
            bytes).

    Returns:
        A URL-safe, unpadded base64 string encoding the identity and
        offset.

    Raises:
        ValueError: If the identity is empty or the offset is negative.
    """
    if not source_identity_value:
        raise ValueError("source_identity_value must not be empty")
    if offset < 0:
        raise ValueError("offset must be non-negative")
    payload = f"{source_identity_value}:{offset}".encode()
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def decode_cursor(cursor: str) -> tuple[str, int]:
    """Decodes a cursor into its source identity and offset.

    Args:
        cursor: A string produced by `cursor_for`.

    Returns:
        The source identity and the non-negative offset.

    Raises:
        ValueError: If the cursor is empty, cannot be decoded, or encodes
            an empty identity or a negative offset.
    """
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

    Attributes:
        offset: Position in the source, in the connector's unit. Zero is
            treated as unset when a cursor is also given.
        cursor: Opaque cursor encoding the source identity and offset, or
            None if not yet bound.
        source_identity: Identity of the source the checkpoint belongs to,
            or an empty string if not yet bound.
        content_sha256: SHA-256 hex digest of the source content (or of the
            rows before the offset) the checkpoint was taken against, or
            None if unknown.
    """

    offset: int = 0
    cursor: str | None = None
    source_identity: str = ""
    content_sha256: str | None = None

    def __post_init__(self) -> None:
        """Validates and normalizes cursor-derived checkpoint fields.

        An empty `source_identity` or a zero `offset` is filled in from the
        cursor when one is given.

        Raises:
            ValueError: If `offset` is negative, `content_sha256` is not a
                64-character lower-case hex digest, the cursor is invalid,
                or the cursor's identity or (non-zero) offset disagrees
                with the checkpoint.
        """
        if self.offset < 0:
            raise ValueError("offset must be non-negative")
        if self.content_sha256 is not None and (
            len(self.content_sha256) != SHA256_HEX_LENGTH
            or any(
                character not in "0123456789abcdef"
                for character in self.content_sha256
            )
        ):
            raise ValueError("content_sha256 must be a SHA-256 hex digest")
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
        """Binds an unbound checkpoint to ``identity`` safely.

        Args:
            identity: Identity of the source to bind to.

        Returns:
            A checkpoint with the same offset and content digest, bound to
            `identity`, with a cursor generated if none was present.

        Raises:
            ValueError: If the checkpoint is already bound to a different
                source.
        """
        if self.source_identity and self.source_identity != identity:
            raise ValueError("checkpoint belongs to a different source")
        return SourceCheckpoint(
            offset=self.offset,
            cursor=self.cursor or cursor_for(identity, self.offset),
            source_identity=identity,
            content_sha256=self.content_sha256,
        )

    def bind_content(self, content_sha256: str) -> "SourceCheckpoint":
        """Binds a checkpoint to an immutable source-content digest.

        Args:
            content_sha256: SHA-256 hex digest of the observed content.

        Returns:
            A copy of the checkpoint carrying `content_sha256`.

        Raises:
            ValueError: If the checkpoint already carries a different
                digest.
        """
        if self.content_sha256 and self.content_sha256 != content_sha256:
            raise ValueError("checkpoint content digest does not match source")
        return SourceCheckpoint(
            offset=self.offset,
            cursor=self.cursor,
            source_identity=self.source_identity,
            content_sha256=content_sha256,
        )

    def advance(self, rows: int) -> "SourceCheckpoint":
        """Returns the checkpoint after consuming ``rows`` rows.

        Args:
            rows: Amount consumed, in the connector's unit (rows or bytes).

        Returns:
            A checkpoint with the offset increased by `rows` and a matching
            cursor.

        Raises:
            ValueError: If `rows` is negative or the checkpoint is not
                bound to a source.
        """
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
            content_sha256=self.content_sha256,
        )


@dataclasses.dataclass(frozen=True)
class SourceColumn:
    """One source column and its source-reported type name.

    Attributes:
        name: Column name.
        type_name: Type name as reported by the source.
    """

    name: str
    type_name: str

    def __post_init__(self) -> None:
        """Validates column identity.

        Raises:
            ValueError: If `name` or `type_name` is empty.
        """
        if not self.name:
            raise ValueError("column name must not be empty")
        if not self.type_name:
            raise ValueError("column type_name must not be empty")


@dataclasses.dataclass(frozen=True)
class SourceSchema:
    """Ordered schema metadata returned by a source connector.

    Attributes:
        columns: The columns in source order.
    """

    columns: tuple[SourceColumn, ...]

    def __post_init__(self) -> None:
        """Rejects empty and duplicate column declarations.

        Raises:
            ValueError: If there are no columns or two columns share a
                name.
        """
        names = tuple(column.name for column in self.columns)
        if not names:
            raise ValueError("source schema must contain at least one column")
        if len(names) != len(set(names)):
            raise ValueError("source schema contains duplicate column names")

    def as_dict(self) -> dict[str, Any]:
        """Serializes the schema.

        Returns:
            A JSON-compatible mapping with a ``columns`` list of
            ``name``/``type_name`` entries in source order.
        """
        return {
            "columns": [
                {"name": column.name, "type_name": column.type_name}
                for column in self.columns
            ]
        }

    @classmethod
    def from_frame(cls, frame: pandas.DataFrame) -> "SourceSchema":
        """Builds a source schema from a pandas frame.

        Args:
            frame: The table whose columns and dtypes are described.

        Returns:
            A schema with one column per frame column, using the dtype
            string as the type name.
        """
        return cls(
            tuple(
                SourceColumn(str(name), str(dtype))
                for name, dtype in zip(frame.columns, frame.dtypes, strict=True)
            )
        )


@dataclasses.dataclass(frozen=True)
class RowBatch:
    """A bounded row result and its resume checkpoint.

    ``content_sha256`` is populated when the connector can observe the raw
    source bytes without adding a second read. A resumed checkpoint can bind
    to that digest and reject a changed source snapshot.

    Attributes:
        schema: Schema of the batch.
        rows: The rows of the batch, as tuples in schema order.
        checkpoint: Checkpoint from which to read the next batch.
        has_more: Whether more rows remain after this batch.
        content_sha256: Digest of the raw source bytes observed, or None if
            the connector did not compute one.
    """

    schema: SourceSchema
    rows: tuple[Row, ...]
    checkpoint: SourceCheckpoint
    has_more: bool
    content_sha256: str | None = None


def execute_with_retries(
    operation: Callable[[], T],
    operation_name: str,
    source_uri: str,
    retry_budget: RetryBudget,
) -> T:
    """Executes an operation until it succeeds or its retry budget is spent.

    Any exception other than a `ConnectorError` triggers a retry, after a
    pause of `delay_seconds` when that is nonzero. A `ConnectorError` raised
    by the operation is re-raised immediately without retrying.

    Args:
        operation: Zero-argument operation to execute.
        operation_name: Stable operation name for failure evidence.
        source_uri: Canonical source URI for failure evidence.
        retry_budget: Retry policy for this operation.

    Returns:
        The operation result.

    Raises:
        RetryExhaustedError: If every attempt fails.
        ConnectorError: If the operation raises one; it is not retried.
        AssertionError: If the retry loop ends without returning or
            raising, which cannot happen because the budget allows at least
            one attempt.
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


WRITE_KEYWORDS = frozenset(
    {
        "alter",
        "attach",
        "call",
        "copy",
        "create",
        "delete",
        "detach",
        "drop",
        "exec",
        "execute",
        "grant",
        "insert",
        "into",
        "merge",
        "pragma",
        "reindex",
        "revoke",
        "truncate",
        "update",
        "vacuum",
    }
)
READ_STARTS = frozenset({"select", "with", "values"})
LITERALS_AND_COMMENTS = re.compile(
    r"'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"|--[^\n]*|/\*.*?\*/", re.DOTALL
)
WORD = re.compile(r"[A-Za-z]+")


def refuse_writes(query: str, source_uri: str) -> None:
    """Rejects a statement that is not a single read-only query.

    The allow-list is textual and deliberately strict: after comments and
    quoted text are removed the statement must be one ``SELECT``, ``WITH``
    or ``VALUES`` statement that contains no data-definition, data-change,
    ``INTO`` or session-control keyword. It refuses some harmless text (a
    column named ``update`` must be quoted) and is a guard against mistakes,
    not a sandbox: a database role without write privilege is the real
    control and cannot be verified from here.

    Args:
        query: SQL text to check.
        source_uri: Source URI recorded in the failure.

    Raises:
        ConnectorError: ``write_statement_refused`` with the reason.
    """
    cleaned = LITERALS_AND_COMMENTS.sub(" ", query).strip().rstrip(";").strip()
    words = [word.lower() for word in WORD.findall(cleaned)]
    reason = ""
    if ";" in cleaned:
        reason = "more than one statement"
    elif not words or words[0] not in READ_STARTS:
        reason = "statement does not start with SELECT, WITH or VALUES"
    else:
        found = sorted(WRITE_KEYWORDS.intersection(words))
        if found:
            reason = f"contains write keyword(s) {found}"
    if reason:
        raise ConnectorError(
            "write_statement_refused",
            f"query refused by the read-only allow-list: {reason}",
            source_uri,
            {"reason": reason},
        )


def digest_rows(hasher: Any, rows: Sequence[Sequence[Any]]) -> None:
    """Feeds rows into a hash in a canonical text form.

    Args:
        hasher: A hash object with an ``update(bytes)`` method, updated in
            place.
        rows: Rows to digest; each is hashed as the UTF-8 ``repr`` of its
            tuple followed by a newline.
    """
    for row in rows:
        hasher.update(repr(tuple(row)).encode("utf-8"))
        hasher.update(b"\n")


class SourceConnector(abc.ABC):
    """Public polymorphic contract for bounded source retrieval.

    Attributes:
        source_uri: Canonical URI of the source.
        source_identity: Stable identity of the source.
        retry_budget: Retry policy applied to each operation.
    """

    def __init__(
        self,
        source_uri: str | pathlib.Path,
        retry_budget: RetryBudget | None = None,
    ) -> None:
        """Initializes a connector identity and retry policy.

        Args:
            source_uri: URI or local path of the source; it is canonicalized.
            retry_budget: Retry policy; defaults to `RetryBudget()`.
        """
        self.source_uri = canonical_source_uri(source_uri)
        self.source_identity = source_identity(self.source_uri)
        self.retry_budget = retry_budget or RetryBudget()

    @abc.abstractmethod
    def schema(self) -> SourceSchema:
        """Describes the source schema without consuming a row checkpoint.

        Returns:
            The ordered schema of the source.
        """

    @abc.abstractmethod
    def read_rows(
        self, checkpoint: SourceCheckpoint | None = None, limit: int = 1_000
    ) -> RowBatch:
        """Reads at most ``limit`` rows from a checkpoint.

        Args:
            checkpoint: Position to resume from; None starts at the
                beginning.
            limit: Maximum number of rows to return; must be positive.

        Returns:
            The batch and the checkpoint from which to continue.
        """

    def checkpoint(self, offset: int = 0) -> SourceCheckpoint:
        """Creates a source-bound checkpoint for explicit offset reads.

        Args:
            offset: Position to start from, in the connector's unit.

        Returns:
            A checkpoint bound to this connector's source at `offset`.
        """
        return SourceCheckpoint(
            offset=offset,
            cursor=cursor_for(self.source_identity, offset),
            source_identity=self.source_identity,
        )

    def validate_checkpoint(
        self, checkpoint: SourceCheckpoint | None
    ) -> SourceCheckpoint:
        """Binds and validates a checkpoint before reading rows.

        Args:
            checkpoint: Checkpoint to validate; None means the start.

        Returns:
            The checkpoint bound to this connector's source identity.

        Raises:
            CheckpointError: ``checkpoint_mismatch`` if the checkpoint is
                bound to a different source.
        """
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
        """Rejects unbounded or invalid batch requests.

        Args:
            limit: Requested maximum batch size.

        Raises:
            ValueError: If `limit` is not positive.
        """
        if limit <= 0:
            raise ValueError("limit must be positive")


def file_content_sha256(
    path: pathlib.Path, chunk_bytes: int = 1024 * 1024
) -> str:
    """Hashes a local source file without loading all of it into memory.

    Args:
        path: File to hash.
        chunk_bytes: Number of bytes read at a time.

    Returns:
        The SHA-256 hex digest of the file's bytes.

    Raises:
        ValueError: If `chunk_bytes` is less than 1.
    """
    if chunk_bytes < 1:
        raise ValueError("chunk_bytes must be positive")
    digest = hashlib.sha256()
    with path.open("rb") as opened:
        while chunk := opened.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


class LocalFileConnector(SourceConnector):
    """Connector for tabular local paths and ``file://`` URIs.

    Attributes:
        path: Absolute resolved path of the local file.
    """

    supported_schemes = ("", "file")

    def __init__(
        self,
        source_uri: str | pathlib.Path,
        retry_budget: RetryBudget | None = None,
    ) -> None:
        """Initializes a local-file connector.

        Args:
            source_uri: A local path or ``file://`` URI.
            retry_budget: Per-operation retry policy.

        Raises:
            UnsupportedUriSchemeError: If the scheme is neither empty nor
                ``file``, or a ``file`` URI names a host other than
                ``localhost``.
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
        if scheme:
            if parsed.netloc not in ("", "localhost"):
                raise UnsupportedUriSchemeError(
                    value, parsed.scheme.lower(), ("file",)
                )
            path = pathlib.Path(url2pathname(parsed.path))
        else:
            path = pathlib.Path(value)
        self.path = path.expanduser().resolve(strict=False)
        super().__init__(self.path, retry_budget)

    def schema(self) -> SourceSchema:
        """Reads the local file and describes its schema.

        Read failures are retried and then surface as `RetryExhaustedError`.

        Returns:
            The schema of the parsed file.
        """
        frame = execute_with_retries(
            lambda: tabular.read_table(self.path),
            "schema",
            self.source_uri,
            self.retry_budget,
        )
        return SourceSchema.from_frame(frame)

    def content_sha256(self) -> str:
        """Hashes the local source file as it is now.

        Returns:
            The SHA-256 hex digest of the file's bytes.
        """
        return file_content_sha256(self.path)

    def read_rows(
        self, checkpoint: SourceCheckpoint | None = None, limit: int = 1_000
    ) -> RowBatch:
        """Reads a bounded local-file batch from an offset checkpoint.

        The whole file is parsed on every call. The file is hashed before
        and after parsing, and the checkpoint is bound to that digest.
        A non-positive `limit` raises ``ValueError``; a checkpoint bound to
        a different source raises ``CheckpointError``
        (``checkpoint_mismatch``); a file that changed while being read or
        since the checkpoint was created raises ``ConnectorError``
        (``source_snapshot_changed``).

        Args:
            checkpoint: Row-offset checkpoint to resume from; None starts
                at row 0.
            limit: Maximum number of rows to return; must be positive.

        Returns:
            The rows from the checkpoint offset, the schema, a checkpoint
            advanced by the rows returned, whether rows remain, and the
            file's content digest.
        """
        self.validate_limit(limit)
        value = self.validate_checkpoint(checkpoint)

        def read() -> RowBatch:
            initial_content_sha256 = self.content_sha256()
            frame = tabular.read_table(self.path)
            observed_content_sha256 = self.content_sha256()
            if initial_content_sha256 != observed_content_sha256:
                raise ConnectorError(
                    "source_snapshot_changed",
                    "local file content changed while it was being read",
                    self.source_uri,
                    {
                        "initial_content_sha256": initial_content_sha256,
                        "observed_content_sha256": observed_content_sha256,
                    },
                )
            try:
                bound_value = value.bind_content(observed_content_sha256)
            except ValueError as error:
                raise ConnectorError(
                    "source_snapshot_changed",
                    "local file content changed since the checkpoint was "
                    "created",
                    self.source_uri,
                    {
                        "checkpoint_content_sha256": value.content_sha256,
                        "observed_content_sha256": observed_content_sha256,
                    },
                ) from error
            schema = SourceSchema.from_frame(frame)
            start = bound_value.offset
            selected = frame.iloc[start : start + limit]
            rows = tuple(selected.itertuples(index=False, name=None))
            next_checkpoint = bound_value.advance(len(rows))
            return RowBatch(
                schema=schema,
                rows=rows,
                checkpoint=next_checkpoint,
                has_more=start + len(rows) < len(frame),
                content_sha256=observed_content_sha256,
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

    The query is checked against a read-only allow-list at construction
    (``refuse_writes``) and, for ``sqlite3`` connections, the session is put
    in ``query_only`` mode while it runs. Other drivers can pass
    ``read_only_statement`` (for example ``SET TRANSACTION READ ONLY``),
    which runs before the query. A database role without write privilege is
    still the only complete control.

    Batches re-execute the query. Every batch carries a digest of the rows
    before its offset in ``SourceCheckpoint.content_sha256`` and the next
    batch refuses to continue (``source_changed``) when the rows it skips no
    longer hash to that digest. Use an ``ORDER BY`` so the order is total.

    Attributes:
        query: The read-only SQL query whose rows form the source.
        connection_factory: Callable returning a fresh DB-API connection.
        read_only: Whether the allow-list and read-only session are
            enforced.
        read_only_statement: Statement run before the query, or None.
        snapshot_query: Query whose result identifies the source state, or
            None.
    """

    supported_schemes = ("dbapi", "sqlite")

    def __init__(
        self,
        source_uri: str,
        query: str,
        connection_factory: Callable[[], Any],
        retry_budget: RetryBudget | None = None,
        read_only: bool = True,
        read_only_statement: str | None = None,
        snapshot_query: str | None = None,
    ) -> None:
        """Initializes a DB-API connector.

        Args:
            source_uri: Logical ``dbapi://`` or ``sqlite://`` source URI.
            query: Read-only SQL query whose rows form the source.
            connection_factory: Callable returning a fresh DB-API connection.
            retry_budget: Per-operation retry policy.
            read_only: Enforce the read-only allow-list (a statement that
                fails it is refused with ``ConnectorError``) and the
                ``sqlite3`` ``query_only`` session mode. When False,
                neither is applied.
            read_only_statement: Optional statement run before the query to
                start a read-only transaction on drivers that support it.
            snapshot_query: Optional cheap query whose result identifies the
                state of the source (a version counter, a maximum id). Without
                it the token is a digest of the whole query result.

        Raises:
            UnsupportedUriSchemeError: If the URI scheme is not ``dbapi``
                or ``sqlite``.
            ValueError: If `query` is blank.
            TypeError: If `connection_factory` is not callable.
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
        if read_only:
            refuse_writes(query, value)
            if snapshot_query is not None:
                refuse_writes(snapshot_query, value)
        self.query = query
        self.connection_factory = connection_factory
        self.read_only = read_only
        self.read_only_statement = read_only_statement
        self.snapshot_query = snapshot_query
        super().__init__(value, retry_budget)
        self.source_identity = source_identity(
            f"{self.source_uri}|query={self.query}"
        )

    @contextlib.contextmanager
    def session(self, sql: str) -> Iterator[Any]:
        """Opens a connection and executes ``sql`` on a cursor.

        The cursor and connection are closed on exit, ignoring errors from
        closing. The session is read-only as configured by `read_only`.

        Args:
            sql: Statement to execute before yielding.

        Yields:
            The DB-API cursor positioned on the result of ``sql``.
        """
        connection = self.connection_factory()
        cursor: Any = None
        try:
            if self.read_only and isinstance(connection, sqlite3.Connection):
                connection.execute("PRAGMA query_only = ON")
            cursor = connection.cursor()
            if self.read_only_statement:
                cursor.execute(self.read_only_statement)
            cursor.execute(sql)
            yield cursor
        finally:
            self.close_resources(cursor, connection)

    def schema(self) -> SourceSchema:
        """Executes the query and describes the result columns.

        Returns:
            The schema built from the cursor description; a missing type
            code becomes ``unknown``.
        """
        return execute_with_retries(
            self.retrieve_schema,
            "schema",
            self.source_uri,
            self.retry_budget,
        )

    def retrieve_schema(self) -> SourceSchema:
        """Performs one non-retried DB-API schema operation.

        Returns:
            The schema built from the cursor description.
        """
        with self.session(self.query) as cursor:
            return self.schema_from_description(cursor.description)

    def snapshot_token(self) -> str:
        """Returns a token that changes when the source content changes.

        With ``snapshot_query`` the token digests that query's rows; without
        it, every row of the main query (a full scan).

        Returns:
            The SHA-256 hex digest over the rows of the snapshot query, or
            of the main query when none was given.
        """
        return execute_with_retries(
            self.retrieve_token,
            "snapshot_token",
            self.source_uri,
            self.retry_budget,
        )

    def retrieve_token(self) -> str:
        """Performs one non-retried token computation.

        Returns:
            The SHA-256 hex digest over every row the token query yields.
        """
        hasher = hashlib.sha256()
        with self.session(self.snapshot_query or self.query) as cursor:
            while rows := cursor.fetchmany(10_000):
                digest_rows(hasher, rows)
        return hasher.hexdigest()

    def read_rows(
        self, checkpoint: SourceCheckpoint | None = None, limit: int = 1_000
    ) -> RowBatch:
        """Reads DB-API rows using portable offset/cursor checkpoints.

        A non-positive `limit` raises ``ValueError``; a checkpoint bound to
        a different source raises ``CheckpointError``
        (``checkpoint_mismatch``).

        Args:
            checkpoint: Row-offset checkpoint to resume from; None starts
                at row 0.
            limit: Maximum number of rows to return; must be positive.

        Returns:
            The rows after the checkpoint offset, the schema, a checkpoint
            advanced by the rows returned and carrying the digest of all
            rows up to it, and whether more rows remain.
        """
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
        """Performs one non-retried DB-API row operation.

        Args:
            checkpoint: Bound checkpoint to resume from.
            limit: Maximum number of rows to return.

        Returns:
            The batch after the checkpoint offset. ``has_more`` is true when
            one more row than `limit` could be fetched.

        Raises:
            CheckpointError: ``source_changed`` when the rows before the
                checkpoint offset no longer match the checkpoint digest.
        """
        with self.session(self.query) as cursor:
            schema = self.schema_from_description(cursor.description)
            hasher = hashlib.sha256()
            self.skip_rows(cursor, checkpoint.offset, limit, hasher)
            expected = checkpoint.content_sha256
            if expected is not None and hasher.hexdigest() != expected:
                raise CheckpointError(
                    self.source_uri,
                    "source_changed",
                    "rows before the checkpoint changed since the last batch",
                    {"offset": checkpoint.offset},
                )
            fetched = cursor.fetchmany(limit + 1)
            rows = tuple(tuple(row) for row in fetched[:limit])
            digest_rows(hasher, rows)
            next_checkpoint = dataclasses.replace(
                checkpoint.advance(len(rows)),
                content_sha256=hasher.hexdigest(),
            )
            return RowBatch(
                schema=schema,
                rows=rows,
                checkpoint=next_checkpoint,
                has_more=len(fetched) > limit,
            )

    def schema_from_description(
        self, description: Sequence[Sequence[Any]] | None
    ) -> SourceSchema:
        """Converts DB-API cursor metadata into a validated schema.

        Args:
            description: The cursor's ``description``: one sequence per
                column whose first item is the name and second the type code.

        Returns:
            The schema; a missing or empty type code is reported as
            ``unknown``.

        Raises:
            ValueError: If the description is None or empty.
        """
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

    def skip_rows(
        self, cursor: Any, offset: int, batch_size: int, hasher: Any
    ) -> None:
        """Consumes and hashes rows before a checkpoint.

        Args:
            cursor: Cursor positioned at the first row.
            offset: Number of rows to skip.
            batch_size: Maximum rows fetched per call.
            hasher: Hash object updated with the skipped rows.

        Raises:
            CheckpointError: ``source_changed`` when fewer than ``offset``
                rows remain, which means rows were deleted.
        """
        remaining = offset
        while remaining:
            skipped = cursor.fetchmany(min(remaining, batch_size))
            if not skipped:
                raise CheckpointError(
                    self.source_uri,
                    "source_changed",
                    "source has fewer rows than the checkpoint offset",
                    {"offset": offset, "missing": remaining},
                )
            digest_rows(hasher, skipped)
            remaining -= len(skipped)

    def close_resources(self, cursor: Any, connection: Any) -> None:
        """Closes resources without masking the primary operation error.

        Args:
            cursor: Cursor to close, or None if none was opened.
            connection: Connection to close.
        """
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

    Attributes:
        query: Versioned query or table expression used by the engine.
        schema_reader: Callback returning the schema for `query`.
        batch_reader: Callback returning one batch for a query, offset and
            limit.
        snapshot_reader: Callback returning a source-state token, or None.
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
        snapshot_reader: Callable[[str], str] | None = None,
    ) -> None:
        """Initializes a warehouse connector around injected query callbacks.

        Args:
            source_uri: Logical ``warehouse://`` or ``lakehouse://`` URI.
            query: Versioned query or table expression used by the engine.
            schema_reader: Callback returning the source schema for ``query``.
            batch_reader: Callback accepting query, row offset, and row
                limit and returning the batch schema, its rows, and whether
                more rows remain.
            retry_budget: Per-operation retry policy.
            snapshot_reader: Optional callback returning a token for the
                state of ``query`` (a table version, ``AS OF`` timestamp or
                content hash). When given, a resumed read whose token
                differs from the one the checkpoint was bound to is refused
                (``source_changed``). Without it the connector cannot tell
                that the source changed between batches.

        Raises:
            UnsupportedUriSchemeError: If the scheme is neither
                ``warehouse`` nor ``lakehouse``.
            ValueError: If `query` is blank.
            TypeError: If `schema_reader` or `batch_reader` is not
                callable.
        """
        value = source_uri.strip()
        scheme = urlsplit(value).scheme.lower()
        if scheme not in self.supported_schemes:
            raise UnsupportedUriSchemeError(
                value, scheme, self.supported_schemes
            )
        if not query.strip():
            raise ValueError("query must not be empty")
        schema_candidate: object = schema_reader
        batch_candidate: object = batch_reader
        if not callable(schema_candidate) or not callable(batch_candidate):
            raise TypeError("schema_reader and batch_reader must be callable")
        self.query = query
        self.schema_reader = schema_reader
        self.batch_reader = batch_reader
        self.snapshot_reader = snapshot_reader
        super().__init__(value, retry_budget)
        self.source_identity = source_identity(
            f"{self.source_uri}|query={self.query}"
        )

    def schema(self) -> SourceSchema:
        """Reads and validates the schema without consuming a checkpoint.

        The read is retried under the connector's retry budget.

        Returns:
            The schema returned by the schema callback, once validated.
        """
        return execute_with_retries(
            lambda: self.validate_schema(self.schema_reader(self.query)),
            "schema",
            self.source_uri,
            self.retry_budget,
        )

    def read_rows(
        self, checkpoint: SourceCheckpoint | None = None, limit: int = 1_000
    ) -> RowBatch:
        """Reads one bounded batch from the injected warehouse engine.

        A non-positive `limit` raises ``ValueError``; a checkpoint bound to
        a different source raises ``CheckpointError``
        (``checkpoint_mismatch``).

        Args:
            checkpoint: Row-offset checkpoint to resume from; None starts
                at row 0.
            limit: Maximum number of rows to return; must be positive.

        Returns:
            The validated batch with a checkpoint advanced by the rows
            returned.
        """
        self.validate_limit(limit)
        value = self.validate_checkpoint(checkpoint)
        return execute_with_retries(
            lambda: self.read_batch(value, limit),
            "read_rows",
            self.source_uri,
            self.retry_budget,
        )

    def read_batch(self, checkpoint: SourceCheckpoint, limit: int) -> RowBatch:
        """Validates one callback result and binds its next checkpoint.

        When a snapshot reader is configured, its token (hashed with
        SHA-256) is stored in the next checkpoint and compared with the one
        in `checkpoint`.

        Args:
            checkpoint: Bound checkpoint to resume from.
            limit: Maximum number of rows the callback may return.

        Returns:
            The batch with the next checkpoint.

        Raises:
            CheckpointError: ``source_changed`` if the snapshot token
                differs from the one recorded in the checkpoint.
            ConnectorError: If the callback's ``has_more`` is not a bool,
                it returns more than `limit` rows, or a row's width differs
                from the schema's.
        """
        token = (
            hashlib.sha256(
                self.snapshot_reader(self.query).encode()
            ).hexdigest()
            if self.snapshot_reader
            else None
        )
        if (
            token is not None
            and checkpoint.content_sha256 is not None
            and checkpoint.content_sha256 != token
        ):
            raise CheckpointError(
                self.source_uri,
                "source_changed",
                "warehouse snapshot token changed since the last batch",
                {"offset": checkpoint.offset},
            )
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
        after = checkpoint.advance(len(rows))
        if token is not None:
            after = dataclasses.replace(after, content_sha256=token)
        return RowBatch(
            schema=schema,
            rows=rows,
            checkpoint=after,
            has_more=has_more,
        )

    def validate_schema(self, schema: SourceSchema) -> SourceSchema:
        """Validates a callback schema at the connector boundary.

        Args:
            schema: The value returned by a callback.

        Returns:
            `schema`, unchanged.

        Raises:
            ConnectorError: ``invalid_schema`` if it is not a
                `SourceSchema`.
        """
        if not isinstance(schema, SourceSchema):
            raise ConnectorError(
                "invalid_schema",
                "schema reader must return SourceSchema",
                self.source_uri,
                {"schema_type": type(schema).__name__},
            )
        return schema


def read_tabular_bytes(source_uri: str, payload: bytes) -> pandas.DataFrame:
    """Decodes one supported tabular object from bytes.

    The object-store connector accepts a byte-reader instead of importing a
    cloud SDK. Credentials and transport-level retries remain caller-owned.
    The format is chosen from the lower-cased file name in the URI path:
    ``.csv`` (also ``.gz``, ``.bz2``, ``.xz``), ``.jsonl`` or ``.ndjson``
    (also ``.gz``), and ``.parquet`` or ``.pq``.

    Args:
        source_uri: Object URI whose path name selects the format.
        payload: The object's bytes.

    Returns:
        The decoded table.

    Raises:
        ConnectorError: ``unsupported_object_format`` if the name has none
            of the supported suffixes.
    """
    name = pathlib.PurePosixPath(urlsplit(source_uri).path).name.lower()
    if name.endswith((".csv", ".csv.gz", ".csv.bz2", ".csv.xz")):
        # typeshed solves the default's type of next() as str, not the literal.
        compression = cast(
            'Literal["infer", "gzip", "bz2", "xz"]',
            next(
                (
                    method
                    for suffix, method in CSV_COMPRESSION.items()
                    if name.endswith(suffix)
                ),
                "infer",
            ),
        )
        return pandas.read_csv(io.BytesIO(payload), compression=compression)
    if name.endswith((".jsonl", ".ndjson", ".jsonl.gz", ".ndjson.gz")):
        compression = "gzip" if name.endswith(".gz") else "infer"
        return pandas.read_json(
            io.BytesIO(payload), lines=True, compression=compression
        )
    if name.endswith((".parquet", ".pq")):
        return pandas.read_parquet(io.BytesIO(payload))
    raise ConnectorError(
        "unsupported_object_format",
        f"unsupported tabular object format: {name or '<unknown>'}",
        source_uri,
        {"supported_formats": ["csv", "jsonl", "parquet"]},
    )


class ObjectStoreConnector(SourceConnector):
    """Connector for bounded object-store reads through an injected reader.

    This adapter downloads one object into memory before decoding it and
    rejects objects over max_bytes. The default decoder handles common
    tabular formats; a caller-owned decoder can admit any format that can be
    represented as a bounded DataFrame. It supports deterministic small-object
    ingestion, but does not claim range reads or streaming-scale semantics.

    Attributes:
        read_bytes: Caller-owned transport returning the object bytes.
        max_bytes: Maximum object size accepted.
        decoder: Callable decoding the object bytes into a DataFrame.
        decoder_name: Stable decoder name and version, part of the source
            identity.
    """

    supported_schemes = ("s3", "gs", "az", "abfs", "http", "https")

    def __init__(
        self,
        source_uri: str,
        read_bytes: Callable[[str], bytes],
        retry_budget: RetryBudget | None = None,
        max_bytes: int = 256 * 1024 * 1024,
        decoder: ObjectDecoder | None = None,
        decoder_name: str | None = None,
    ) -> None:
        """Initializes an object connector with explicit boundaries.

        Args:
            source_uri: Object-store URI.
            read_bytes: Caller-owned transport that returns the object bytes.
            retry_budget: Per-operation retry policy.
            max_bytes: Maximum object size loaded into memory.
            decoder: Optional caller-owned decoder returning a DataFrame.
            decoder_name: Stable decoder name and version for source
                identity. Required with a custom decoder and not allowed
                without one (the default decoder is ``tabular-auto-v1``).

        Raises:
            UnsupportedUriSchemeError: If the scheme is not one of
                ``s3``, ``gs``, ``az``, ``abfs``, ``http`` or ``https``.
            ValueError: If `max_bytes` is below 1, `decoder_name` is given
                without a decoder, or a custom decoder has a blank name.
            TypeError: If a decoder or transport is not callable.
        """
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
        if decoder is not None and not callable(decoder):
            raise TypeError("decoder must be callable")
        if decoder is None:
            if decoder_name is not None:
                raise ValueError("decoder_name requires a caller-owned decoder")
            decoder = read_tabular_bytes
            decoder_name = "tabular-auto-v1"
        elif not decoder_name or not decoder_name.strip():
            raise ValueError(
                "decoder_name must identify a caller-owned decoder and version"
            )
        self.read_bytes = read_bytes
        self.max_bytes = max_bytes
        self.decoder = decoder
        self.decoder_name = decoder_name.strip()
        super().__init__(value, retry_budget)
        self.source_identity = source_identity(
            f"{self.source_uri}|decoder={self.decoder_name}"
        )

    def load_frame(self) -> pandas.DataFrame:
        """Downloads and decodes the bounded object once.

        Returns:
            The decoded table.
        """
        return self.decode_object(self.read_object())

    def read_object(self) -> ObjectPayload:
        """Reads bounded raw bytes and derives their content identity.

        Returns:
            The object bytes with their digest and size.

        Raises:
            ConnectorError: ``invalid_object_payload`` if the transport
                does not return ``bytes``; ``object_too_large`` if the
                payload exceeds `max_bytes` (checked after the read).
        """
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
        return ObjectPayload(self.source_uri, payload)

    def decode_object(self, object_payload: ObjectPayload) -> pandas.DataFrame:
        """Decodes one observed object while preserving its raw identity.

        Args:
            object_payload: The bytes to decode.

        Returns:
            The table returned by the decoder.

        Raises:
            ConnectorError: ``decoder_failed`` if the decoder raises
                (a `ConnectorError` it raises propagates unchanged);
                ``invalid_decoded_object`` if it does not return a
                DataFrame.
        """
        try:
            frame = self.decoder(self.source_uri, object_payload.payload)
        except ConnectorError:
            raise
        except Exception as error:
            raise ConnectorError(
                "decoder_failed",
                f"object decoder failed: {error}",
                self.source_uri,
                {
                    "decoder": self.decoder_name,
                    "error_type": type(error).__name__,
                },
            ) from error
        if not isinstance(frame, pandas.DataFrame):
            raise ConnectorError(
                "invalid_decoded_object",
                "object decoder must return a pandas DataFrame",
                self.source_uri,
                {
                    "decoder": self.decoder_name,
                    "result_type": type(frame).__name__,
                },
            )
        return frame

    def schema(self) -> SourceSchema:
        """Reads the bounded object and describes its schema.

        Returns:
            The schema of the decoded table.
        """
        return SourceSchema.from_frame(self.load_frame())

    def read_rows(
        self, checkpoint: SourceCheckpoint | None = None, limit: int = 1_000
    ) -> RowBatch:
        """Reads a bounded row batch from an object checkpoint.

        The whole object is downloaded and decoded on every call, and the
        checkpoint is bound to the object's content digest. A non-positive
        `limit` raises ``ValueError``; a checkpoint bound to a different
        source raises ``CheckpointError`` (``checkpoint_mismatch``).

        Args:
            checkpoint: Row-offset checkpoint to resume from; None starts
                at row 0.
            limit: Maximum number of rows to return; must be positive.

        Returns:
            The rows from the checkpoint offset, the schema, a checkpoint
            advanced by the rows returned, whether rows remain, and the
            object's content digest.

        Raises:
            ConnectorError: ``source_snapshot_changed`` if the object
                differs from the one the checkpoint was bound to.
        """
        self.validate_limit(limit)
        value = self.validate_checkpoint(checkpoint)
        object_payload = self.read_object()
        try:
            value = value.bind_content(object_payload.content_sha256)
        except ValueError as error:
            raise ConnectorError(
                "source_snapshot_changed",
                "object content changed since the checkpoint was created",
                self.source_uri,
                {
                    "checkpoint_content_sha256": value.content_sha256,
                    "observed_content_sha256": object_payload.content_sha256,
                },
            ) from error
        frame = self.decode_object(object_payload)
        schema = SourceSchema.from_frame(frame)
        selected = frame.iloc[value.offset : value.offset + limit]
        rows = tuple(selected.itertuples(index=False, name=None))
        next_checkpoint = value.advance(len(rows))
        return RowBatch(
            schema=schema,
            rows=rows,
            checkpoint=next_checkpoint,
            has_more=value.offset + len(rows) < len(frame),
            content_sha256=object_payload.content_sha256,
        )


class StreamingObjectStoreConnector(SourceConnector):
    """Range-backed connector for bounded JSONL and NDJSON objects.

    The injected range reader uses half-open byte ranges ``[start, end)``.
    Checkpoint offsets are byte positions at record boundaries, so a resumed
    read never needs to download or decode the preceding object content.
    The connector intentionally supports line-oriented formats only; quoted
    CSV records and binary containers need a format-specific streaming adapter.

    Attributes:
        read_range: Caller-owned byte-range transport.
        size_bytes: Caller-owned object-size lookup.
        chunk_bytes: Maximum bytes requested per range call.
        max_record_bytes: Maximum buffered line size.
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
        """Initializes a range-backed object connector.

        Args:
            source_uri: Object URI ending in ``.jsonl`` or ``.ndjson``.
            read_range: Caller-owned byte-range transport.
            size_bytes: Caller-owned object-size lookup.
            retry_budget: Per-operation retry policy.
            chunk_bytes: Maximum bytes requested per range call.
            max_record_bytes: Maximum buffered line size.

        Raises:
            UnsupportedUriSchemeError: If the scheme is not one of the
                object-store schemes.
            ConnectorError: ``unsupported_stream_format`` if the name does
                not end in ``.jsonl`` or ``.ndjson``.
            TypeError: If `read_range` or `size_bytes` is not callable.
            ValueError: If `chunk_bytes` or `max_record_bytes` is below 1.
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
        range_candidate: object = read_range
        size_candidate: object = size_bytes
        if not callable(range_candidate) or not callable(size_candidate):
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
        """Looks up the object size through the retry policy.

        Returns:
            The size in bytes.

        Raises:
            ConnectorError: ``invalid_object_size`` if the lookup does not
                return a non-negative integer.
        """
        size: object = execute_with_retries(
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
        """Reads one validated half-open byte range.

        Args:
            start: First byte offset, inclusive.
            end: Last byte offset, exclusive.

        Returns:
            The bytes returned by the transport, which may be fewer than
            ``end - start``.

        Raises:
            ConnectorError: ``invalid_range_payload`` if the transport does
                not return ``bytes``; ``range_overread`` if it returns more
                than requested.
        """
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
        """Reads one bounded record to discover the source schema.

        Returns:
            The schema derived from the first record.

        Raises:
            ConnectorError: ``empty_object`` if the object has no records.
        """
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
        """Reads JSON records from a byte checkpoint without full download.

        A non-positive `limit` raises ``ValueError``; a checkpoint bound to
        a different source raises ``CheckpointError``
        (``checkpoint_mismatch``).

        Args:
            checkpoint: Byte-offset checkpoint at a record boundary; None
                starts at byte 0.
            limit: Maximum number of records to return; must be positive.

        Returns:
            The decoded records and their schema, with a checkpoint at the
            byte position after the last record returned and ``has_more``
            true if further records exist. When no record remains, the rows
            are empty, the schema is a single ``record`` column and the
            checkpoint is unchanged. The batch carries no content digest.

        Raises:
            CheckpointError: ``checkpoint_beyond_object`` if the
                checkpoint offset exceeds the current object size.
        """
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
        """Reads complete newline-delimited records from a byte position.

        Blank lines are skipped and a trailing carriage return is removed.
        A final line without a newline counts as a record.

        Args:
            start: Byte offset to start reading from.
            total: Object size in bytes.
            required_records: Number of records after which to stop.

        Returns:
            Up to `required_records` pairs of a record's bytes and the byte
            offset just past its line terminator (the object end for an
            unterminated final line).

        Raises:
            ConnectorError: ``range_no_progress`` if the transport returns
                no bytes before the object end; ``record_too_large`` if
                buffered bytes exceed `max_record_bytes` without a
                complete line.
        """
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
        """Decodes JSON objects and rejects silent schema drift.

        Args:
            records: Non-empty sequence of pairs of a record's UTF-8 JSON
                bytes and its end offset.

        Returns:
            The schema taken from the first record's fields and value
            types, and the rows as tuples in that field order.

        Raises:
            ConnectorError: ``invalid_json_record`` if a record is not
                valid UTF-8 JSON; ``record_not_object`` if it is not a JSON
                object; ``schema_changed`` if a record's field names or
                order differ from the first record's.
        """
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
        """Names the source type of one JSON value.

        Args:
            value: A decoded JSON value.

        Returns:
            ``null``, ``bool``, ``int``, ``float``, ``str``, ``array`` or
            ``object``; for any other value its Python class name.
        """
        if value is None:
            return "null"
        for kind, name in JSON_TYPE_NAMES:
            if isinstance(value, kind):
                return name
        return type(value).__name__
