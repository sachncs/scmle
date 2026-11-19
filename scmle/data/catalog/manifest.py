"""The versioned catalog of one data snapshot and its verified file."""

from collections.abc import Iterable
from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
import json
import pathlib
from typing import Any
from typing import ClassVar
from typing import Self

from scmle import durability
from scmle.contracts import errors
from scmle.contracts import records
from scmle.data.catalog import canon
from scmle.data.catalog import evidence
from scmle.data.catalog import identity
from scmle.data.catalog import inspection

PARTS = {
    "source": identity.Source.migrations[:1],
    "missingness": evidence.Missingness.migrations,
    "duplicates": evidence.Duplication.migrations,
    "label": evidence.Labeling.migrations,
}


def lift(layout: records.Layout) -> records.Layout:
    """Moves a renamed version-1 catalog's nested parts to version 2.

    Args:
        layout: The catalog after its top-level keys were renamed.

    Returns:
        The catalog with ``content_identity`` nested as
        ``digests.content`` and every part's keys renamed.
    """
    moved = {key: value for key, value in layout.items() if key in PARTS}
    lifted = {
        key: records.upgrade(PARTS[key], dict(value), records.ORIGIN)
        for key, value in moved.items()
    }
    rest = {
        key: value
        for key, value in layout.items()
        if key not in PARTS and key != "content_identity"
    }
    if "columns" in rest:
        rest["columns"] = [
            records.upgrade(
                evidence.Field.migrations, dict(field), records.ORIGIN
            )
            for field in rest["columns"]
        ]
    if "content_identity" in layout:
        rest["digests"] = {"content": layout["content_identity"]}
    return {**rest, **lifted}


def lower(layout: records.Layout) -> records.Layout:
    """Undoes `lift`.

    Args:
        layout: A version-2 catalog.

    Returns:
        The catalog with its parts and content digest in version-1 form.
    """
    moved = {key: value for key, value in layout.items() if key in PARTS}
    lowered = {
        key: records.downgrade(PARTS[key], dict(value))
        for key, value in moved.items()
    }
    rest = {
        key: value
        for key, value in layout.items()
        if key not in PARTS and key != "digests"
    }
    if "columns" in rest:
        rest["columns"] = [
            records.downgrade(evidence.Field.migrations, dict(field))
            for field in rest["columns"]
        ]
    if "digests" in layout:
        rest["content_identity"] = layout["digests"]["content"]
    return {**rest, **lowered}


MIGRATIONS = (
    records.Migration(
        renames={
            "data_kind": "kind",
            "schema": "columns",
            "sample_count": "samples",
            "release_blockers": "blockers",
        },
        change=lift,
        revert=lower,
    ),
    records.nest({}, parts={"source": identity.Source.migrations[1]}),
)
CATALOG_SCHEMA_VERSION = records.ORIGIN + len(MIGRATIONS)


def census(samples: Any, schema_version: Any) -> None:
    """Checks the two integer fields of a catalog.

    Args:
        samples: The sample count to check.
        schema_version: The schema version to check.

    Raises:
        TypeError: If either is not an integer (booleans are rejected).
        ValueError: If `samples` is negative or `schema_version` is not the
            supported version.
    """
    if isinstance(samples, bool) or not isinstance(samples, int):
        raise errors.KindError("samples must be an integer")
    if samples < 0:
        raise errors.InputError("samples must not be negative")
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        raise errors.KindError("schema_version must be an integer")
    if schema_version != CATALOG_SCHEMA_VERSION:
        raise errors.InputError("unsupported catalog schema version")


def adopt(source: identity.Source | str | pathlib.Path) -> identity.Source:
    """Returns a source, wrapping a URI or path with default governance.

    Args:
        source: A source, or a URI or path.

    Returns:
        `source` itself, or a `Source` for the URI or path.
    """
    return (
        source
        if isinstance(source, identity.Source)
        else identity.Source(source)
    )


def withhold(source: identity.Source) -> list[str]:
    """Lists the release blockers that undeclared governance implies.

    Args:
        source: The data source.

    Returns:
        A blocker for each of the owner, acquisition method and privacy
        classification that the source leaves empty, in that order.
    """
    blockers: list[str] = []
    if not source.owner:
        blockers.append("source_owner_not_declared")
    if not source.acquisition:
        blockers.append("source_acquisition_method_not_declared")
    if not source.privacy:
        blockers.append("privacy_and_retention_not_assessed")
    return blockers


def vet(catalog: "Catalog") -> tuple[evidence.Field, ...]:
    """Checks the classes and values of a catalog's fields.

    Args:
        catalog: The catalog being constructed.

    Returns:
        The columns as a tuple, read from the catalog once.

    Raises:
        TypeError: If ``source``, ``missingness``, ``duplicates``, ``label``,
            or a column has the wrong class, or if ``samples`` or
            ``schema_version`` is not an integer (booleans are rejected).
        ValueError: If ``kind`` is empty, ``samples`` is negative,
            ``schema_version`` is unsupported, ``shape`` has a negative or
            non-integer entry, ``limitations`` is empty or has a blank
            entry, or ``blockers`` has a blank entry.
    """
    if not isinstance(catalog.source, identity.Source):
        raise errors.KindError("source must be Source")
    kind: object = catalog.kind
    if not isinstance(kind, str) or not kind.strip():
        raise errors.InputError("kind must not be empty")
    census(catalog.samples, catalog.schema_version)
    dimensions: Iterable[object] = catalog.shape
    if any(
        isinstance(dimension, bool)
        or not isinstance(dimension, int)
        or dimension < 0
        for dimension in dimensions
    ):
        raise errors.InputError("shape must contain non-negative integers")
    fields = tuple(catalog.columns)
    if any(not isinstance(field, evidence.Field) for field in fields):
        raise errors.KindError("columns must contain Field records")
    if not isinstance(catalog.missingness, evidence.Missingness):
        raise errors.KindError("missingness must be Missingness")
    if not isinstance(catalog.duplicates, evidence.Duplication):
        raise errors.KindError("duplicates must be Duplication")
    if not isinstance(catalog.label, evidence.Labeling):
        raise errors.KindError("label must be Labeling")
    limitation_notes: Iterable[object] = catalog.limitations
    blocker_notes: Iterable[object] = catalog.blockers
    if not catalog.limitations or any(
        not isinstance(item, str) or not item.strip()
        for item in limitation_notes
    ):
        raise errors.InputError("limitations must contain non-empty strings")
    if any(
        not isinstance(item, str) or not item.strip() for item in blocker_notes
    ):
        raise errors.InputError("blockers must contain non-empty strings")
    return fields


def fingerprint(digests: Mapping[str, str]) -> dict[str, str]:
    """Checks and normalizes a catalog's digests.

    Args:
        digests: The digests, keyed by the hashed thing.

    Returns:
        The ``content`` digest, normalized to ``sha256:`` and lowercase hex.

    Raises:
        ValueError: If `digests` holds anything but a SHA-256 ``content``
            digest.
    """
    if set(digests) != {"content"}:
        raise errors.InputError("digests must hold the content digest")
    return {"content": identity.assay(digests["content"], "content")}


def load(path: str | pathlib.Path) -> tuple[Mapping[str, Any], str]:
    """Reads catalog evidence and separates its recorded digest.

    Args:
        path: A file written by ``Catalog.write``.

    Returns:
        The catalog content without its digest, and that digest. The digest
        is stored as ``digest``, or as ``catalog_digest`` by version 2 and
        older.

    Raises:
        ValueError: If the file cannot be read or parsed, is not a JSON
            object, or lacks a string digest.
    """
    try:
        record = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as error:
        raise errors.InputError(
            "catalog evidence is missing or invalid JSON"
        ) from error
    if not isinstance(record, dict):
        raise errors.InputError("catalog evidence must be a JSON object")
    key = "digest" if "digest" in record else "catalog_digest"
    expected_digest = record.get(key)
    if not isinstance(expected_digest, str):
        raise errors.InputError("catalog evidence is missing catalog_digest")
    payload = dict(record)
    del payload[key]
    return payload, expected_digest


@dataclasses.dataclass(frozen=True, slots=True)
class Catalog:
    """Versioned, model-agnostic curation evidence for one data snapshot.

    The contract accepts arbitrary caller-owned observations, but it never
    silently invents a content hash for an opaque object.  Use ``observe``
    for supported Python data structures or ``glean`` when a
    caller-owned adapter can provide trustworthy evidence for another format.
    ``digests`` holds ``content``, the SHA-256 digest of the data content.
    """

    __module__ = "scmle.data.catalog"

    migrations: ClassVar[tuple[records.Migration, ...]] = MIGRATIONS

    source: identity.Source
    kind: str
    shape: tuple[int, ...]
    columns: tuple[evidence.Field, ...]
    samples: int
    missingness: evidence.Missingness
    duplicates: evidence.Duplication
    label: evidence.Labeling
    digests: Mapping[str, str]
    limitations: tuple[str, ...]
    blockers: tuple[str, ...]
    schema_version: int = CATALOG_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validates evidence consistency before publication.

        Raises:
            TypeError: If ``source``, ``missingness``, ``duplicates``,
                ``label``, or a column has the wrong class, or if
                ``samples`` or ``schema_version`` is not an integer
                (booleans are rejected).
            ValueError: If ``kind`` is empty, ``samples`` is negative,
                ``schema_version`` is unsupported, ``shape`` has a
                negative or non-integer entry, ``limitations`` is empty or has
                a blank entry, ``blockers`` has a blank entry, or
                ``digests`` does not hold exactly a SHA-256 ``content``
                digest.
        """
        object.__setattr__(self, "columns", vet(self))
        object.__setattr__(self, "digests", fingerprint(self.digests))
        object.__setattr__(self, "kind", self.kind.strip())
        object.__setattr__(self, "limitations", tuple(self.limitations))
        object.__setattr__(
            self, "blockers", tuple(dict.fromkeys(self.blockers))
        )

    @classmethod
    def glean(
        cls,
        source: identity.Source | str | pathlib.Path,
        kind: str,
        shape: Sequence[int],
        columns: Sequence[evidence.Field],
        samples: int,
        missingness: evidence.Missingness,
        duplicates: evidence.Duplication,
        label: evidence.Labeling,
        content: str,
        limitations: Sequence[str],
        blockers: Sequence[str] = (),
    ) -> Self:
        """Builds a catalog from adapter-owned evidence for any data format.

        Args:
            source: The data source, or a URI or path that is wrapped in a
                ``Source`` with default governance fields.
            kind: Name of the kind of data observed.
            shape: Dimensions of the data.
            columns: Ordered fields of the observed schema.
            samples: Number of samples (rows) observed.
            missingness: Observed missing-value summary.
            duplicates: Observed duplicate-row summary.
            label: Label summary.
            content: SHA-256 digest of the data content.
            limitations: Non-empty statements of what the evidence does not
                establish.
            blockers: Conditions that block release; duplicates are
                removed, keeping first-seen order.

        Returns:
            The validated catalog.
        """
        source_value = adopt(source)
        return cls(
            source=source_value,
            kind=kind,
            shape=tuple(shape),
            columns=tuple(columns),
            samples=samples,
            missingness=missingness,
            duplicates=duplicates,
            label=label,
            digests={"content": content},
            limitations=tuple(limitations),
            blockers=tuple(blockers),
        )

    @classmethod
    def observe(
        cls,
        data: Any,
        source: identity.Source | str | pathlib.Path = "memory://",
        label_column: str | None = None,
        content_identity: str | None = None,
        limitations: Sequence[str] = (),
        release_blockers: Sequence[str] = (),
    ) -> Self:
        """Observes common Python dataset forms without fitting a model.

        Supported forms include pandas frames and series, NumPy arrays,
        mappings of columns, sequences of records or values, scalars, and
        bytes.  Opaque objects require ``content_identity`` so the catalog
        cannot publish a non-reproducible identity.

        Args:
            data: The object to observe.
            source: The data source, or a URI or path that is wrapped in a
                ``Source`` with default governance fields.
            label_column: Name of the label column, or None when no label
                is declared.
            content_identity: Caller-supplied SHA-256 digest of the
                content, or None to compute one from the observed content.
            limitations: Extra limitation statements appended to the
                observed ones.
            release_blockers: Extra release blockers appended to those
                derived from the source metadata and the observation.

        Returns:
            A catalog whose limitations and release blockers combine the
            built-in curation-only limitation, those found while
            observing, and the caller's, with duplicates removed in
            first-seen order.
        """
        source_value = adopt(source)
        observation = inspection.examine(data, label_column, content_identity)
        limitation_values = [
            "curation_only_no_model_fit_or_performance_claim",
            *observation.limitations,
            *limitations,
        ]
        blocker_values = withhold(source_value)
        blocker_values.extend(observation.blockers)
        blocker_values.extend(release_blockers)
        return cls.glean(
            source=source_value,
            kind=observation.kind,
            shape=observation.shape,
            columns=observation.schema,
            samples=observation.samples,
            missingness=observation.missingness,
            duplicates=observation.duplicates,
            label=observation.label,
            content=observation.identity,
            limitations=tuple(dict.fromkeys(limitation_values)),
            blockers=tuple(dict.fromkeys(blocker_values)),
        )

    @property
    def digest(self) -> str:
        """The digest of all catalog evidence except this digest.

        This is the ``sha256:`` digest of ``contents()``.
        """
        return canon.fingerprint(self.contents())

    def contents(self) -> dict[str, Any]:
        """Returns canonical catalog content without the external digest.

        Returns:
            A JSON-native dictionary of every catalog field, with nested
            summaries converted by their own ``to_dict``.
        """
        return {
            "schema_version": self.schema_version,
            "source": self.source.to_dict(),
            "kind": self.kind,
            "shape": list(self.shape),
            "columns": [field.to_dict() for field in self.columns],
            "samples": self.samples,
            "missingness": self.missingness.to_dict(),
            "duplicates": self.duplicates.to_dict(),
            "label": self.label.to_dict(),
            "digests": dict(self.digests),
            "limitations": list(self.limitations),
            "blockers": list(self.blockers),
        }

    def to_dict(self) -> dict[str, Any]:
        """Returns JSON-native catalog evidence with its integrity digest.

        Returns:
            The content dictionary plus the catalog digest as ``digest``.
        """
        record = self.contents()
        record["digest"] = self.digest
        return record

    def canonize(self) -> str:
        """Returns canonical catalog content suitable for verification.

        Returns:
            Canonical JSON text of ``contents()``.
        """
        return canon.canonize(self.contents())

    def write(self, path: str | pathlib.Path) -> None:
        """Persists the catalog atomically as verified JSON evidence.

        Args:
            path: Destination file.  The indented JSON, including
                digest (``digest``), is written atomically.
        """
        encoded = (
            json.dumps(
                self.to_dict(),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
        durability.write_atomically(path, encoded)

    @classmethod
    def read(cls, path: str | pathlib.Path) -> Self:
        """Reads a catalog and fail closed if its digest or schema is invalid.

        The digest is checked on the content as stored, before an older
        layout is migrated, so an older catalog verifies as it was written.

        Args:
            path: A file written by ``write``.

        Returns:
            The catalog rebuilt from the file, whose recorded digest matches
            the digest of its stored content.

        Raises:
            ValueError: If the file cannot be read or parsed, is not a JSON
                object, lacks a string digest, its digest does
                not match the stored content, or the content does not
                rebuild into a valid catalog.
        """
        payload, expected_digest = load(path)
        if not isinstance(payload.get("schema_version"), int):
            raise errors.InputError("catalog evidence schema is invalid")
        if identity.assay(expected_digest, "catalog_digest") != (
            canon.fingerprint(payload)
        ):
            raise errors.InputError(
                "catalog digest does not match catalog content"
            )
        try:
            return cls.from_dict(payload)
        except (KeyError, TypeError, ValueError) as error:
            raise errors.InputError(
                "catalog evidence schema is invalid"
            ) from error

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds a catalog from content without trusting its digest.

        A record stored at an older schema version (none means 1) is
        migrated first.

        Args:
            record: A mapping of catalog content as produced by ``to_dict``.
                Any stored digest is not checked here. A missing
                required entry raises ``KeyError``; ``blockers`` may be
                absent.

        Returns:
            The reconstructed catalog.

        Raises:
            ValueError: If ``schema_version`` is not an integer or is newer
                than this build.
        """
        layout = records.load("Catalog", cls.migrations, record)
        return cls(
            source=identity.Source.from_dict(layout["source"]),
            kind=str(layout["kind"]),
            shape=tuple(int(value) for value in layout["shape"]),
            columns=tuple(
                evidence.Field.from_dict(field) for field in layout["columns"]
            ),
            samples=int(layout["samples"]),
            missingness=evidence.Missingness.from_dict(layout["missingness"]),
            duplicates=evidence.Duplication.from_dict(layout["duplicates"]),
            label=evidence.Labeling.from_dict(layout["label"]),
            digests={"content": str(layout["digests"]["content"])},
            limitations=tuple(str(item) for item in layout["limitations"]),
            blockers=tuple(str(item) for item in layout.get("blockers", ())),
        )
