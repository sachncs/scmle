"""Digest-bound JSON persistence shared by the causal estimation reports.

A report is written as an envelope of a schema version, the report's
`to_dict` record and the SHA-256 digest of that record's canonical JSON. The
sequential and longitudinal estimations both persist this way; each names
its kind in `label`, which the read errors carry.
"""

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any
from typing import Self

from scmle import durability
from scmle.contracts import errors


def created_temporary(temporary: Path | None) -> Path:
    """Returns the temporary report file, which must have been created.

    Args:
        temporary: The path of the temporary file, or None if none was made.

    Returns:
        `temporary`.

    Raises:
        RuntimeError: If no temporary file was created.
    """
    if temporary is None:
        raise RuntimeError("temporary report was not created")
    return temporary


class Envelope:
    """A report that writes and reads itself as a digest-bound envelope.

    Subclasses are frozen dataclasses that define `label`, `to_dict` and
    `from_dict`.

    Attributes:
        label: The kind of report, named in the read errors.
    """

    __slots__ = ()
    label = ""

    def to_dict(self) -> Mapping[str, Any]:
        """Returns the JSON-compatible report.

        Raises:
            NotImplementedError: Always, in the base class; subclasses
                override it.
        """
        raise NotImplementedError

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds a report from its JSON-compatible representation.

        Args:
            record: Mapping in the form produced by `to_dict`.

        Raises:
            NotImplementedError: Always, in the base class; subclasses
                override it.
        """
        raise NotImplementedError

    def digest(self) -> str:
        """Returns the digest of the canonical report content.

        Returns:
            The hexadecimal SHA-256 digest of the report serialized as
            sorted-key compact JSON.
        """
        content = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(content).hexdigest()

    def write(self, path: str | Path) -> str:
        """Atomically writes a digest-bound report.

        The envelope holds `schema_version` 1, the `to_dict` report and its
        `digest`. It is written as sorted-key JSON to a temporary file in the
        destination directory, flushed, renamed over the destination, and the
        directory is then synced. On any failure the temporary file is
        removed and the exception propagates.

        Args:
            path: Destination file; missing parent directories are created.

        Returns:
            The report digest stored in the envelope.

        Raises:
            RuntimeError: If the temporary file path was not recorded
                (not reachable in practice, since it is set on entering
                the file context).
        """
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        envelope = {
            "schema_version": 1,
            "report": self.to_dict(),
            "digest": self.digest(),
        }
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                json.dump(envelope, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            created_temporary(temporary).replace(destination)
            durability.sync_directory(destination.parent)
        except BaseException:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            raise
        return self.digest()

    @classmethod
    def read(cls, path: str | Path) -> Self:
        """Reads and verifies an atomic report envelope.

        An unreadable or malformed file raises the underlying OS or JSON
        error; a missing key raises `KeyError`, and invalid fields raise the
        validation errors of the result classes.

        Args:
            path: File previously written by `write`.

        Returns:
            The result rebuilt from the stored report.

        Raises:
            ValueError: If the schema version is not 1, the report is not a
                mapping, the stored digest does not match the report, or the
                rebuilt result serializes to a different digest.
        """
        envelope = json.loads(Path(path).read_text(encoding="utf-8"))
        if envelope.get("schema_version") != 1:
            raise errors.InputError(f"unsupported {cls.label} report schema")
        record = envelope.get("report")
        if not isinstance(record, Mapping):
            raise errors.InputError(
                f"{cls.label} report must contain a mapping"
            )
        canonical = json.dumps(
            record, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        digest = hashlib.sha256(canonical).hexdigest()
        if envelope.get("digest") != digest:
            raise errors.InputError(f"{cls.label} report digest mismatch")
        result = cls.from_dict(record)
        if result.digest() != digest:
            raise errors.InputError(
                f"{cls.label} report canonicalization mismatch"
            )
        return result
