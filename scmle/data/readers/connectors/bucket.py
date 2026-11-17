"""Connector for bounded whole-object reads from an object store."""

from collections.abc import Callable
import dataclasses
import hashlib
import io
import pathlib
from typing import cast
from typing import Literal
from urllib.parse import urlsplit

import pandas

from scmle.contracts import errors
from scmle.data.readers.connectors import adapter
from scmle.data.readers.connectors import failure
from scmle.data.readers.connectors import layout
from scmle.data.readers.connectors import locator

CSV_COMPRESSION: dict[str, Literal["gzip", "bz2", "xz"]] = {
    ".gz": "gzip",
    ".bz2": "bz2",
    ".xz": "xz",
}


ObjectDecoder = Callable[[str, bytes], pandas.DataFrame]


@dataclasses.dataclass(frozen=True, slots=True)
class Payload:
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
            raise errors.InputError("source_uri must not be empty")
        if not isinstance(self.payload, bytes):
            raise errors.KindError("payload must be bytes")
        object.__setattr__(
            self,
            "content_sha256",
            hashlib.sha256(self.payload).hexdigest(),
        )
        object.__setattr__(self, "byte_size", len(self.payload))


def unpack(source_uri: str, payload: bytes) -> pandas.DataFrame:
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
    raise failure.ConnectorError(
        "unsupported_object_format",
        f"unsupported tabular object format: {name or '<unknown>'}",
        source_uri,
        {"supported_formats": ["csv", "jsonl", "parquet"]},
    )


class Bucket(adapter.Adapter):
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
        retry_budget: failure.Retries | None = None,
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
            SchemeError: If the scheme is not one of
                ``s3``, ``gs``, ``az``, ``abfs``, ``http`` or ``https``.
            ValueError: If `max_bytes` is below 1, `decoder_name` is given
                without a decoder, or a custom decoder has a blank name.
            TypeError: If a decoder or transport is not callable.
        """
        value = source_uri.strip()
        scheme = urlsplit(value).scheme.lower()
        if scheme not in self.supported_schemes:
            raise failure.SchemeError(value, scheme, self.supported_schemes)
        if not callable(read_bytes):
            raise errors.KindError("read_bytes must be callable")
        if max_bytes < 1:
            raise errors.InputError("max_bytes must be positive")
        if decoder is not None and not callable(decoder):
            raise errors.KindError("decoder must be callable")
        if decoder is None:
            if decoder_name is not None:
                raise errors.InputError(
                    "decoder_name requires a caller-owned decoder"
                )
            decoder = unpack
            decoder_name = "tabular-auto-v1"
        elif not decoder_name or not decoder_name.strip():
            raise errors.InputError(
                "decoder_name must identify a caller-owned decoder and version"
            )
        self.read_bytes = read_bytes
        self.max_bytes = max_bytes
        self.decoder = decoder
        self.decoder_name = decoder_name.strip()
        super().__init__(value, retry_budget)
        self.source_identity = locator.source_identity(
            f"{self.source_uri}|decoder={self.decoder_name}"
        )

    def load(self) -> pandas.DataFrame:
        """Downloads and decodes the bounded object once.

        Returns:
            The decoded table.
        """
        return self.deserialize(self.read_object())

    def read_object(self) -> Payload:
        """Reads bounded raw bytes and derives their content identity.

        Returns:
            The object bytes with their digest and size.

        Raises:
            ConnectorError: ``invalid_object_payload`` if the transport
                does not return ``bytes``; ``object_too_large`` if the
                payload exceeds `max_bytes` (checked after the read).
        """
        payload = failure.persevere(
            lambda: self.read_bytes(self.source_uri),
            "read_object",
            self.source_uri,
            self.retry_budget,
        )
        if not isinstance(payload, bytes):
            raise failure.ConnectorError(
                "invalid_object_payload",
                "object reader must return bytes",
                self.source_uri,
                {"payload_type": type(payload).__name__},
            )
        if len(payload) > self.max_bytes:
            raise failure.ConnectorError(
                "object_too_large",
                f"object is larger than max_bytes={self.max_bytes}",
                self.source_uri,
                {"bytes": len(payload), "max_bytes": self.max_bytes},
            )
        return Payload(self.source_uri, payload)

    def deserialize(self, object_payload: Payload) -> pandas.DataFrame:
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
        except failure.ConnectorError:
            raise
        except Exception as error:
            raise failure.ConnectorError(
                "decoder_failed",
                f"object decoder failed: {error}",
                self.source_uri,
                {
                    "decoder": self.decoder_name,
                    "error_type": type(error).__name__,
                },
            ) from error
        if not isinstance(frame, pandas.DataFrame):
            raise failure.ConnectorError(
                "invalid_decoded_object",
                "object decoder must return a pandas DataFrame",
                self.source_uri,
                {
                    "decoder": self.decoder_name,
                    "result_type": type(frame).__name__,
                },
            )
        return frame

    def schema(self) -> layout.Columns:
        """Reads the bounded object and describes its schema.

        Returns:
            The schema of the decoded table.
        """
        return layout.Columns.from_frame(self.load())

    def read_rows(
        self, checkpoint: locator.Bookmark | None = None, limit: int = 1_000
    ) -> layout.Page:
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
        value = self.rebind(checkpoint)
        object_payload = self.read_object()
        try:
            value = value.seal(object_payload.content_sha256)
        except ValueError as error:
            raise failure.ConnectorError(
                "source_snapshot_changed",
                "object content changed since the checkpoint was created",
                self.source_uri,
                {
                    "checkpoint_content_sha256": value.content_sha256,
                    "observed_content_sha256": object_payload.content_sha256,
                },
            ) from error
        frame = self.deserialize(object_payload)
        schema = layout.Columns.from_frame(frame)
        selected = frame.iloc[value.offset : value.offset + limit]
        rows = tuple(selected.itertuples(index=False, name=None))
        next_checkpoint = value.advance(len(rows))
        return layout.Page(
            schema=schema,
            rows=rows,
            checkpoint=next_checkpoint,
            has_more=value.offset + len(rows) < len(frame),
            content_sha256=object_payload.content_sha256,
        )
