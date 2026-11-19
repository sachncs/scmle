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
        uri: Canonical URI the bytes were read from.
        payload: The raw object bytes.
        digest: SHA-256 hex digest of `payload`, derived on
            construction.
        size: Length of `payload` in bytes, derived on construction.
    """

    uri: str
    payload: bytes = dataclasses.field(repr=False)
    digest: str = dataclasses.field(init=False)
    size: int = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        """Derives the digest and size used by checkpoints.

        Raises:
            ValueError: If `uri` is blank.
            TypeError: If `payload` is not ``bytes``.
        """
        if not self.uri.strip():
            raise errors.InputError("source_uri must not be empty")
        if not isinstance(self.payload, bytes):
            raise errors.KindError("payload must be bytes")
        object.__setattr__(
            self,
            "digest",
            hashlib.sha256(self.payload).hexdigest(),
        )
        object.__setattr__(self, "size", len(self.payload))


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
        transport: Caller-owned transport returning the object bytes.
        limit: Maximum object size accepted.
        decoder: Callable decoding the object bytes into a DataFrame.
        codec: Stable decoder name and version, part of the source
            identity.
    """

    schemes = ("s3", "gs", "az", "abfs", "http", "https")

    def __init__(
        self,
        source_uri: str,
        read_bytes: Callable[[str], bytes],
        retry_budget: failure.Retries | None = None,
        limit: int = 256 * 1024 * 1024,
        decoder: ObjectDecoder | None = None,
        codec: str | None = None,
    ) -> None:
        """Initializes an object connector with explicit boundaries.

        Args:
            source_uri: Object-store URI.
            read_bytes: Caller-owned transport that returns the object bytes.
            retry_budget: Per-operation retry policy.
            limit: Maximum object size loaded into memory.
            decoder: Optional caller-owned decoder returning a DataFrame.
            codec: Stable decoder name and version for source
                identity. Required with a custom decoder and not allowed
                without one (the default decoder is ``tabular-auto-v1``).

        Raises:
            SchemeError: If the scheme is not one of
                ``s3``, ``gs``, ``az``, ``abfs``, ``http`` or ``https``.
            ValueError: If `limit` is below 1, `codec` is given
                without a decoder, or a custom decoder has a blank name.
            TypeError: If a decoder or transport is not callable.
        """
        value = source_uri.strip()
        scheme = urlsplit(value).scheme.lower()
        if scheme not in self.schemes:
            raise failure.SchemeError(value, scheme, self.schemes)
        if not callable(read_bytes):
            raise errors.KindError("read_bytes must be callable")
        if limit < 1:
            raise errors.InputError("limit must be positive")
        if decoder is not None and not callable(decoder):
            raise errors.KindError("decoder must be callable")
        if decoder is None:
            if codec is not None:
                raise errors.InputError("codec requires a caller-owned decoder")
            decoder = unpack
            codec = "tabular-auto-v1"
        elif not codec or not codec.strip():
            raise errors.InputError(
                "codec must identify a caller-owned decoder and version"
            )
        self.transport = read_bytes
        self.limit = limit
        self.decoder = decoder
        self.codec = codec.strip()
        super().__init__(value, retry_budget)
        self.identity = locator.identify(f"{self.uri}|decoder={self.codec}")

    def load(self) -> pandas.DataFrame:
        """Downloads and decodes the bounded object once.

        Returns:
            The decoded table.
        """
        return self.deserialize(self.fetch())

    def fetch(self) -> Payload:
        """Reads bounded raw bytes and derives their content identity.

        Returns:
            The object bytes with their digest and size.

        Raises:
            ConnectorError: ``invalid_object_payload`` if the transport
                does not return ``bytes``; ``object_too_large`` if the
                payload exceeds `limit` (checked after the read).
        """
        payload = failure.persevere(
            lambda: self.transport(self.uri),
            "read_object",
            self.uri,
            self.budget,
        )
        if not isinstance(payload, bytes):
            raise failure.ConnectorError(
                "invalid_object_payload",
                "object reader must return bytes",
                self.uri,
                {"payload_type": type(payload).__name__},
            )
        if len(payload) > self.limit:
            raise failure.ConnectorError(
                "object_too_large",
                f"object is larger than max_bytes={self.limit}",
                self.uri,
                {"bytes": len(payload), "max_bytes": self.limit},
            )
        return Payload(self.uri, payload)

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
            frame = self.decoder(self.uri, object_payload.payload)
        except failure.ConnectorError:
            raise
        except Exception as error:
            raise failure.ConnectorError(
                "decoder_failed",
                f"object decoder failed: {error}",
                self.uri,
                {
                    "decoder": self.codec,
                    "error_type": type(error).__name__,
                },
            ) from error
        if not isinstance(frame, pandas.DataFrame):
            raise failure.ConnectorError(
                "invalid_decoded_object",
                "object decoder must return a pandas DataFrame",
                self.uri,
                {
                    "decoder": self.codec,
                    "result_type": type(frame).__name__,
                },
            )
        return frame

    def schema(self) -> layout.Columns:
        """Reads the bounded object and describes its schema.

        Returns:
            The schema of the decoded table.
        """
        return layout.Columns.infer(self.load())

    def read(
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
        self.validate(limit)
        value = self.rebind(checkpoint)
        object_payload = self.fetch()
        try:
            value = value.seal(object_payload.digest)
        except ValueError as error:
            raise failure.ConnectorError(
                "source_snapshot_changed",
                "object content changed since the checkpoint was created",
                self.uri,
                {
                    "checkpoint_content_sha256": value.digest,
                    "observed_content_sha256": object_payload.digest,
                },
            ) from error
        frame = self.deserialize(object_payload)
        schema = layout.Columns.infer(frame)
        selected = frame.iloc[value.offset : value.offset + limit]
        rows = tuple(selected.itertuples(index=False, name=None))
        next_checkpoint = value.advance(len(rows))
        return layout.Page(
            schema=schema,
            rows=rows,
            checkpoint=next_checkpoint,
            more=value.offset + len(rows) < len(frame),
            digest=object_payload.digest,
        )
