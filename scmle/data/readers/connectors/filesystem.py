"""Connector for tabular local files."""

import hashlib
import pathlib
from urllib.parse import urlsplit
from urllib.request import url2pathname

from scmle.contracts import errors
from scmle.data.readers import tabular
from scmle.data.readers.connectors import adapter
from scmle.data.readers.connectors import failure
from scmle.data.readers.connectors import layout
from scmle.data.readers.connectors import locator


def checksum(path: pathlib.Path, chunk_bytes: int = 1024 * 1024) -> str:
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
        raise errors.InputError("chunk_bytes must be positive")
    digest = hashlib.sha256()
    with path.open("rb") as opened:
        while chunk := opened.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


class Filesystem(adapter.Adapter):
    """Connector for tabular local paths and ``file://`` URIs.

    Attributes:
        path: Absolute resolved path of the local file.
    """

    schemes = ("", "file")

    def __init__(
        self,
        source_uri: str | pathlib.Path,
        retry_budget: failure.Retries | None = None,
    ) -> None:
        """Initializes a local-file connector.

        Args:
            source_uri: A local path or ``file://`` URI.
            retry_budget: Per-operation retry policy.

        Raises:
            SchemeError: If the scheme is neither empty nor
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
        if scheme.lower() not in self.schemes:
            raise failure.SchemeError(value, scheme.lower(), self.schemes)
        if scheme:
            if parsed.netloc not in ("", "localhost"):
                raise failure.SchemeError(
                    value, parsed.scheme.lower(), ("file",)
                )
            path = pathlib.Path(url2pathname(parsed.path))
        else:
            path = pathlib.Path(value)
        self.path = path.expanduser().resolve(strict=False)
        super().__init__(self.path, retry_budget)

    def schema(self) -> layout.Columns:
        """Reads the local file and describes its schema.

        Read failures are retried and then surface as `RetryError`.

        Returns:
            The schema of the parsed file.
        """
        frame = failure.persevere(
            lambda: tabular.read(self.path),
            "schema",
            self.uri,
            self.budget,
        )
        return layout.Columns.infer(frame)

    def digest(self) -> str:
        """Hashes the local source file as it is now.

        Returns:
            The SHA-256 hex digest of the file's bytes.
        """
        return checksum(self.path)

    def read(
        self, checkpoint: locator.Bookmark | None = None, limit: int = 1_000
    ) -> layout.Page:
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
        self.validate(limit)
        value = self.rebind(checkpoint)

        def read() -> layout.Page:
            initial_content_sha256 = self.digest()
            frame = tabular.read(self.path)
            observed_content_sha256 = self.digest()
            if initial_content_sha256 != observed_content_sha256:
                raise failure.ConnectorError(
                    "source_snapshot_changed",
                    "local file content changed while it was being read",
                    self.uri,
                    {
                        "initial_content_sha256": initial_content_sha256,
                        "observed_content_sha256": observed_content_sha256,
                    },
                )
            try:
                bound_value = value.seal(observed_content_sha256)
            except ValueError as error:
                raise failure.ConnectorError(
                    "source_snapshot_changed",
                    "local file content changed since the checkpoint was "
                    "created",
                    self.uri,
                    {
                        "checkpoint_content_sha256": value.digest,
                        "observed_content_sha256": observed_content_sha256,
                    },
                ) from error
            schema = layout.Columns.infer(frame)
            start = bound_value.offset
            selected = frame.iloc[start : start + limit]
            rows = tuple(selected.itertuples(index=False, name=None))
            next_checkpoint = bound_value.advance(len(rows))
            return layout.Page(
                schema=schema,
                rows=rows,
                checkpoint=next_checkpoint,
                more=start + len(rows) < len(frame),
                digest=observed_content_sha256,
            )

        return failure.persevere(
            read,
            "read_rows",
            self.uri,
            self.budget,
        )
