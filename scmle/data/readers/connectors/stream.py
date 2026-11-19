"""Range-backed connector for line-delimited JSON objects."""

from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
import json
import pathlib
from typing import Any
from urllib.parse import urlsplit

from scmle.contracts import errors
from scmle.data.readers.connectors import adapter
from scmle.data.readers.connectors import bucket
from scmle.data.readers.connectors import failure
from scmle.data.readers.connectors import layout
from scmle.data.readers.connectors import locator

JSON_TYPE_NAMES = (
    (bool, "bool"),
    (int, "int"),
    (float, "float"),
    (str, "str"),
    (list, "array"),
    (Mapping, "object"),
)


RangeReader = Callable[[str, int, int], bytes]
SizeReader = Callable[[str], int]


def interpret(
    records: Sequence[tuple[bytes, int]], source_uri: str
) -> tuple[layout.Columns, tuple[layout.Row, ...]]:
    """Decodes JSON objects and rejects silent schema drift.

    Args:
        records: Non-empty sequence of pairs of a record's UTF-8 JSON
            bytes and its end offset.
        source_uri: Canonical source URI recorded in a failure.

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
            raise failure.ConnectorError(
                "invalid_json_record",
                "streaming object contains invalid UTF-8 JSON",
                source_uri,
                {"error_type": type(error).__name__},
            ) from error
        if not isinstance(record, Mapping):
            raise failure.ConnectorError(
                "record_not_object",
                "streaming JSON records must be objects",
                source_uri,
                {"record_type": type(record).__name__},
            )
        decoded.append(record)
    names = tuple(decoded[0])
    schema = layout.Columns(
        tuple(layout.Field(name, classify(decoded[0][name])) for name in names)
    )
    rows: list[layout.Row] = []
    for record in decoded:
        if tuple(record) != names:
            raise failure.ConnectorError(
                "schema_changed",
                "streaming JSON record fields changed within the object",
                source_uri,
                {"expected": list(names), "observed": list(record)},
            )
        rows.append(tuple(record[name] for name in names))
    return schema, tuple(rows)


def classify(value: Any) -> str:
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


class StreamingObjectStoreConnector(adapter.Adapter):
    """Range-backed connector for bounded JSONL and NDJSON objects.

    The injected range reader uses half-open byte ranges ``[start, end)``.
    Checkpoint offsets are byte positions at record boundaries, so a resumed
    read never needs to download or decode the preceding object content.
    The connector intentionally supports line-oriented formats only; quoted
    CSV records and binary containers need a format-specific streaming adapter.

    Attributes:
        transport: Caller-owned byte-range transport.
        size: Caller-owned object-size lookup.
        chunk: Maximum bytes requested per range call.
        limit: Maximum buffered line size.
    """

    schemes = bucket.Bucket.schemes

    def __init__(
        self,
        source_uri: str,
        read_range: RangeReader,
        size_bytes: SizeReader,
        retry_budget: failure.Retries | None = None,
        chunk: int = 1024 * 1024,
        limit: int = 64 * 1024 * 1024,
    ) -> None:
        """Initializes a range-backed object connector.

        Args:
            source_uri: Object URI ending in ``.jsonl`` or ``.ndjson``.
            read_range: Caller-owned byte-range transport.
            size_bytes: Caller-owned object-size lookup.
            retry_budget: Per-operation retry policy.
            chunk: Maximum bytes requested per range call.
            limit: Maximum buffered line size.

        Raises:
            SchemeError: If the scheme is not one of the
                object-store schemes.
            ConnectorError: ``unsupported_stream_format`` if the name does
                not end in ``.jsonl`` or ``.ndjson``.
            TypeError: If `transport` or `size` is not callable.
            ValueError: If `chunk` or `limit` is below 1.
        """
        value = source_uri.strip()
        scheme = urlsplit(value).scheme.lower()
        suffix = pathlib.PurePosixPath(urlsplit(value).path).name.lower()
        if scheme not in self.schemes:
            raise failure.SchemeError(value, scheme, self.schemes)
        if not suffix.endswith((".jsonl", ".ndjson")):
            raise failure.ConnectorError(
                "unsupported_stream_format",
                "streaming object connector requires JSONL or NDJSON",
                value,
                {"supported_formats": ["jsonl", "ndjson"]},
            )
        range_candidate: object = read_range
        size_candidate: object = size_bytes
        if not callable(range_candidate) or not callable(size_candidate):
            raise errors.KindError("read_range and size_bytes must be callable")
        if chunk < 1:
            raise errors.InputError("chunk must be positive")
        if limit < 1:
            raise errors.InputError("limit must be positive")
        self.transport = read_range
        self.size = size_bytes
        self.chunk = chunk
        self.limit = limit
        super().__init__(value, retry_budget)

    def measure(self) -> int:
        """Looks up the object size through the retry policy.

        Returns:
            The size in bytes.

        Raises:
            ConnectorError: ``invalid_object_size`` if the lookup does not
                return a non-negative integer.
        """
        size: object = failure.persevere(
            lambda: self.size(self.uri),
            "head_object",
            self.uri,
            self.budget,
        )
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise failure.ConnectorError(
                "invalid_object_size",
                "object size reader must return a non-negative integer",
                self.uri,
                {"size_type": type(size).__name__, "size": size},
            )
        return size

    def fetch(self, start: int, end: int) -> bytes:
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
        payload = failure.persevere(
            lambda: self.transport(self.uri, start, end),
            "read_range",
            self.uri,
            self.budget,
        )
        if not isinstance(payload, bytes):
            raise failure.ConnectorError(
                "invalid_range_payload",
                "range reader must return bytes",
                self.uri,
                {"payload_type": type(payload).__name__},
            )
        requested = end - start
        if len(payload) > requested:
            raise failure.ConnectorError(
                "range_overread",
                "range reader returned more bytes than requested",
                self.uri,
                {"requested": requested, "received": len(payload)},
            )
        return payload

    def schema(self) -> layout.Columns:
        """Reads one bounded record to discover the source schema.

        Returns:
            The schema derived from the first record.

        Raises:
            ConnectorError: ``empty_object`` if the object has no records.
        """
        batch = self.read(limit=1)
        if not batch.rows:
            raise failure.ConnectorError(
                "empty_object",
                "streaming object contains no JSON records",
                self.uri,
            )
        return batch.schema

    def read(
        self, checkpoint: locator.Bookmark | None = None, limit: int = 1_000
    ) -> layout.Page:
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
        self.validate(limit)
        value = self.rebind(checkpoint)
        total = self.measure()
        if value.offset > total:
            raise failure.CheckpointError(
                self.uri,
                "checkpoint_beyond_object",
                "checkpoint offset is beyond the current object size",
                {"offset": value.offset, "size": total},
            )
        records = self.scan(value.offset, total, limit + 1)
        if not records:
            return layout.Page(
                schema=layout.Columns((layout.Field("record", "object"),)),
                rows=(),
                checkpoint=value,
                more=False,
            )
        selected = records[:limit]
        schema, rows = interpret(selected, self.uri)
        next_offset = selected[-1][1]
        next_checkpoint = value.bind(self.identity).advance(
            next_offset - value.offset
        )
        return layout.Page(
            schema=schema,
            rows=rows,
            checkpoint=next_checkpoint,
            more=len(records) > limit,
        )

    def scan(
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
                buffered bytes exceed `limit` without a
                complete line.
        """
        records: list[tuple[bytes, int]] = []
        buffer = b""
        buffer_origin = start
        position = start
        while position < total and len(records) < required_records:
            end = min(total, position + self.chunk)
            payload = self.fetch(position, end)
            if not payload:
                raise failure.ConnectorError(
                    "range_no_progress",
                    "range reader returned no bytes before object end",
                    self.uri,
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
            if len(buffer) > self.limit:
                raise failure.ConnectorError(
                    "record_too_large",
                    "a JSON record exceeds max_record_bytes",
                    self.uri,
                    {
                        "max_record_bytes": self.limit,
                        "offset": buffer_origin,
                    },
                )
        if buffer and position >= total and len(records) < required_records:
            records.append((buffer, position))
        return records
