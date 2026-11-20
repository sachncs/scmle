"""Digest-bound JSON persistence shared by the causal estimation reports.

A report is written as an envelope of a schema version, the report's
`to_dict` record and the SHA-256 digest of that record's canonical JSON. The
parameter, sequential and longitudinal estimations all persist this way;
each names its kind in `label`, which the read errors carry.

The digest names the report's content across runs, so it is anchored at
schema version 1: it is computed over the report as version 1 laid it out,
and a report stored at any version keeps the digest it was written with.
Version 2 nested the row, subject and contrast fields and the input digest
(`rows.{received, used, dropped}`, `digests.frame`); version 3 writes the
estimand, the policy and the per-period, per-regime and per-parameter
records in single-word keys (each record's own `migrations`). Each kind's
`migrations` move a report between the layouts, one version at a time.
"""

from collections.abc import Mapping
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any
from typing import ClassVar
from typing import Self

from scmle import durability
from scmle.contracts import errors
from scmle.contracts import records

VERSION = 3

Groups = Mapping[str, Mapping[str, str]]


def fold(layout: records.Layout, groups: Groups) -> records.Layout:
    """Moves flat keys into nested objects.

    Args:
        layout: The stored record.
        groups: Object key to a mapping of flat key to nested key.

    Returns:
        The record with each group's flat keys moved into its object; a
        flat key that is absent is left out of the object.

    Raises:
        InputError: If an object key is already present.
    """
    moved = dict(layout)
    for group, leaves in groups.items():
        if group in moved:
            raise errors.InputError(
                f"folding would overwrite the stored key {group!r}"
            )
        moved[group] = {
            leaf: moved.pop(key) for key, leaf in leaves.items() if key in moved
        }
    return moved


def unfold(layout: records.Layout, groups: Groups) -> records.Layout:
    """Moves nested objects back to flat keys; the inverse of `fold`.

    Args:
        layout: The stored record.
        groups: Object key to a mapping of flat key to nested key.

    Returns:
        The record with each group's object replaced by its flat keys.

    Raises:
        InputError: If an object is not a mapping, holds an unknown key, or
            a flat key is already present.
    """
    moved = dict(layout)
    for group, leaves in groups.items():
        nested = moved.pop(group, {})
        if not isinstance(nested, Mapping):
            raise errors.InputError(f"{group!r} must be an object")
        known = {leaf: key for key, leaf in leaves.items()}
        unknown = set(nested) - set(known)
        if unknown:
            raise errors.InputError(
                f"{group!r} holds unknown keys: {sorted(unknown)}"
            )
        for leaf, value in nested.items():
            if known[leaf] in moved:
                raise errors.InputError(
                    f"unfolding would overwrite the stored key {known[leaf]!r}"
                )
            moved[known[leaf]] = value
    return moved


def rename(kind: Any, part: Mapping[str, Any]) -> records.Layout:
    """Renames the version-1 keys of a stored part to its kind's fields.

    Only the keys whose new spelling is a field are renamed: a field that
    kept its old name (such as `standard_error`, written as `se`) keeps it.

    Args:
        kind: The part's dataclass; its first migration step renames the
            keys.
        part: The part with its version-1 keys.

    Returns:
        The part keyed by field name.
    """
    fields = {field.name for field in dataclasses.fields(kind)}
    renames = {
        old: new
        for old, new in kind.migrations[0].renames.items()
        if new in fields
    }
    return {renames.get(key, key): value for key, value in part.items()}


def canon(record: Mapping[str, Any]) -> str:
    """Returns the SHA-256 digest of a record's sorted compact JSON.

    Args:
        record: The JSON-compatible record.

    Returns:
        The hexadecimal digest.
    """
    content = json.dumps(record, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


@dataclasses.dataclass(frozen=True, slots=True)
class Counts:
    """Units supplied to an estimation, analysed, and removed.

    Attributes:
        received: Number of units in the supplied input.
        used: Number of units analysed.
        dropped: Number of units removed by the missingness policy.
    """

    received: int
    used: int
    dropped: int

    def __post_init__(self) -> None:
        """Checks the counts and that used plus dropped units add up.

        Raises:
            InputError: If a count is not a non-negative integer, or the
                used and dropped units do not sum to the received units.
        """
        for name in ("received", "used", "dropped"):
            count: object = getattr(self, name)
            if (
                isinstance(count, bool)
                or not isinstance(count, int)
                or count < 0
            ):
                raise errors.InputError(
                    f"{name} must be a non-negative integer"
                )
        if self.used + self.dropped != self.received:
            raise errors.InputError(
                "used and dropped units must reconcile to received units"
            )


def lowercase(digest: object, name: str) -> None:
    """Checks that a value is a lowercase SHA-256 hex digest.

    Args:
        digest: The value to check.
        name: The value's name, given in the error.

    Raises:
        InputError: If the value is not 64 lowercase hex characters.
    """
    if (
        not isinstance(digest, str)
        or len(digest) != hashlib.sha256().digest_size * 2
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise errors.InputError(f"{name} must be a lowercase SHA-256 digest")


def require(temporary: Path | None) -> Path:
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

    Subclasses are frozen dataclasses that define `label`, `migrations`,
    `to_dict` and `from_dict`.

    Attributes:
        label: The kind of report, named in the read errors.
        migrations: The report's version steps, oldest first; one step per
            version after 1, so the current version is `VERSION`.
    """

    __slots__ = ()
    label = ""
    migrations: ClassVar[tuple[records.Migration, ...]] = ()

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
            The hexadecimal SHA-256 digest of the report, laid out as at
            schema version 1, serialized as sorted-key compact JSON.
        """
        return canon(records.downgrade(self.migrations, dict(self.to_dict())))

    def write(self, path: str | Path) -> str:
        """Atomically writes a digest-bound report.

        The envelope holds `schema_version` 3, the `to_dict` report and its
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
            "schema_version": VERSION,
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
            require(temporary).replace(destination)
            durability.sync_directory(destination.parent)
        except BaseException:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            raise
        return self.digest()

    @classmethod
    def read(cls, path: str | Path) -> Self:
        """Reads and verifies an atomic report envelope.

        The stored digest is checked before the report is migrated: a
        version-1 report is hashed as stored, and a later one is hashed in
        its version-1 layout, which the migrations restore exactly. An
        unreadable or malformed file raises the underlying OS or JSON error;
        a missing key raises `KeyError`, and invalid fields raise the
        validation errors of the result classes.

        Args:
            path: File previously written by `write`.

        Returns:
            The result rebuilt from the stored report.

        Raises:
            ValueError: If the schema version is not 1 to 3, the report is
                not a mapping, the stored digest does not match the report,
                or the rebuilt result serializes to a different digest.
        """
        envelope = json.loads(Path(path).read_text(encoding="utf-8"))
        version = envelope.get("schema_version")
        if (
            isinstance(version, bool)
            or not isinstance(version, int)
            or not records.ORIGIN <= version <= VERSION
        ):
            raise errors.InputError(f"unsupported {cls.label} report schema")
        record = envelope.get("report")
        if not isinstance(record, Mapping):
            raise errors.InputError(
                f"{cls.label} report must contain a mapping"
            )
        stored = dict(record)
        older = cls.migrations[: version - records.ORIGIN]
        digest = canon(records.downgrade(older, stored))
        if envelope.get("digest") != digest:
            raise errors.InputError(f"{cls.label} report digest mismatch")
        result = cls.from_dict(records.upgrade(cls.migrations, stored, version))
        if result.digest() != digest:
            raise errors.InputError(
                f"{cls.label} report canonicalization mismatch"
            )
        return result
