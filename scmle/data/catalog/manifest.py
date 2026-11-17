"""The versioned catalog of one data snapshot and its verified file."""

from collections.abc import Iterable
from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
import json
import pathlib
from typing import Any
from typing import Self

from scmle import durability
from scmle.contracts import errors
from scmle.data.catalog import canon
from scmle.data.catalog import evidence
from scmle.data.catalog import identity
from scmle.data.catalog import inspection

CATALOG_SCHEMA_VERSION = 1


def census(sample_count: Any, schema_version: Any) -> None:
    """Checks the two integer fields of a catalog.

    Args:
        sample_count: The sample count to check.
        schema_version: The schema version to check.

    Raises:
        TypeError: If either is not an integer (booleans are rejected).
        ValueError: If `sample_count` is negative or `schema_version` is
            not the supported version.
    """
    if isinstance(sample_count, bool) or not isinstance(sample_count, int):
        raise errors.KindError("sample_count must be an integer")
    if sample_count < 0:
        raise errors.InputError("sample_count must not be negative")
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
    if not source.acquisition_method:
        blockers.append("source_acquisition_method_not_declared")
    if not source.privacy_classification:
        blockers.append("privacy_and_retention_not_assessed")
    return blockers


def vet(catalog: "Catalog") -> tuple[evidence.Field, ...]:
    """Checks the classes and values of a catalog's fields.

    Args:
        catalog: The catalog being constructed.

    Returns:
        The schema as a tuple, read from the catalog once.

    Raises:
        TypeError: If ``source``, ``missingness``, ``duplicates``, ``label``,
            or a schema entry has the wrong class, or if ``sample_count`` or
            ``schema_version`` is not an integer (booleans are rejected).
        ValueError: If ``data_kind`` is empty, ``sample_count`` is negative,
            ``schema_version`` is unsupported, ``shape`` has a negative or
            non-integer entry, or ``limitations`` is empty or has a blank
            entry, or ``release_blockers`` has a blank entry.
    """
    if not isinstance(catalog.source, identity.Source):
        raise errors.KindError("source must be Source")
    data_kind: object = catalog.data_kind
    if not isinstance(data_kind, str) or not data_kind.strip():
        raise errors.InputError("data_kind must not be empty")
    census(catalog.sample_count, catalog.schema_version)
    dimensions: Iterable[object] = catalog.shape
    if any(
        isinstance(dimension, bool)
        or not isinstance(dimension, int)
        or dimension < 0
        for dimension in dimensions
    ):
        raise errors.InputError("shape must contain non-negative integers")
    fields = tuple(catalog.schema)
    if any(not isinstance(field, evidence.Field) for field in fields):
        raise errors.KindError("schema must contain Field records")
    if not isinstance(catalog.missingness, evidence.Missingness):
        raise errors.KindError("missingness must be Missingness")
    if not isinstance(catalog.duplicates, evidence.Duplication):
        raise errors.KindError("duplicates must be Duplication")
    if not isinstance(catalog.label, evidence.Labeling):
        raise errors.KindError("label must be Labeling")
    limitation_notes: Iterable[object] = catalog.limitations
    blocker_notes: Iterable[object] = catalog.release_blockers
    if not catalog.limitations or any(
        not isinstance(item, str) or not item.strip()
        for item in limitation_notes
    ):
        raise errors.InputError("limitations must contain non-empty strings")
    if any(
        not isinstance(item, str) or not item.strip() for item in blocker_notes
    ):
        raise errors.InputError(
            "release_blockers must contain non-empty strings"
        )
    return fields


def load(path: str | pathlib.Path) -> tuple[Mapping[str, Any], str]:
    """Reads catalog evidence and separates its recorded digest.

    Args:
        path: A file written by ``Catalog.write``.

    Returns:
        The catalog content without ``catalog_digest``, and that digest.

    Raises:
        ValueError: If the file cannot be read or parsed, is not a JSON
            object, or lacks a string ``catalog_digest``.
    """
    try:
        record = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as error:
        raise errors.InputError(
            "catalog evidence is missing or invalid JSON"
        ) from error
    if not isinstance(record, dict):
        raise errors.InputError("catalog evidence must be a JSON object")
    expected_digest = record.get("catalog_digest")
    if not isinstance(expected_digest, str):
        raise errors.InputError("catalog evidence is missing catalog_digest")
    payload = dict(record)
    del payload["catalog_digest"]
    return payload, expected_digest


@dataclasses.dataclass(frozen=True, slots=True)
class Catalog:
    """Versioned, model-agnostic curation evidence for one data snapshot.

    The contract accepts arbitrary caller-owned observations, but it never
    silently invents a content hash for an opaque object.  Use ``from_data``
    for supported Python data structures or ``glean`` when a
    caller-owned adapter can provide trustworthy evidence for another format.
    """

    __module__ = "scmle.data.catalog"

    source: identity.Source
    data_kind: str
    shape: tuple[int, ...]
    schema: tuple[evidence.Field, ...]
    sample_count: int
    missingness: evidence.Missingness
    duplicates: evidence.Duplication
    label: evidence.Labeling
    content_identity: str
    limitations: tuple[str, ...]
    release_blockers: tuple[str, ...]
    schema_version: int = CATALOG_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validates evidence consistency before publication.

        Raises:
            TypeError: If ``source``, ``missingness``, ``duplicates``,
                ``label``, or a schema entry has the wrong class, or if
                ``sample_count`` or ``schema_version`` is not an integer
                (booleans are rejected).
            ValueError: If ``data_kind`` is empty, ``sample_count`` is
                negative, ``schema_version`` is unsupported, ``shape`` has a
                negative or non-integer entry, ``limitations`` is empty or has
                a blank entry, ``release_blockers`` has a blank entry, or
                ``content_identity`` is not a SHA-256 digest.
        """
        fields = vet(self)
        object.__setattr__(self, "schema", fields)
        object.__setattr__(
            self,
            "content_identity",
            identity.assay(self.content_identity, "content_identity"),
        )
        object.__setattr__(self, "data_kind", self.data_kind.strip())
        object.__setattr__(self, "limitations", tuple(self.limitations))
        object.__setattr__(
            self,
            "release_blockers",
            tuple(dict.fromkeys(self.release_blockers)),
        )

    @classmethod
    def glean(
        cls,
        source: identity.Source | str | pathlib.Path,
        data_kind: str,
        shape: Sequence[int],
        schema: Sequence[evidence.Field],
        sample_count: int,
        missingness: evidence.Missingness,
        duplicates: evidence.Duplication,
        label: evidence.Labeling,
        content_identity: str,
        limitations: Sequence[str],
        release_blockers: Sequence[str] = (),
    ) -> Self:
        """Builds a catalog from adapter-owned evidence for any data format.

        Args:
            source: The data source, or a URI or path that is wrapped in a
                ``Source`` with default governance fields.
            data_kind: Name of the kind of data observed.
            shape: Dimensions of the data.
            schema: Ordered fields of the observed schema.
            sample_count: Number of samples (rows) observed.
            missingness: Observed missing-value summary.
            duplicates: Observed duplicate-row summary.
            label: Label summary.
            content_identity: SHA-256 digest of the data content.
            limitations: Non-empty statements of what the evidence does not
                establish.
            release_blockers: Conditions that block release; duplicates are
                removed, keeping first-seen order.

        Returns:
            The validated catalog.
        """
        source_value = adopt(source)
        return cls(
            source=source_value,
            data_kind=data_kind,
            shape=tuple(shape),
            schema=tuple(schema),
            sample_count=sample_count,
            missingness=missingness,
            duplicates=duplicates,
            label=label,
            content_identity=content_identity,
            limitations=tuple(limitations),
            release_blockers=tuple(release_blockers),
        )

    @classmethod
    def from_data(
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
        blocker_values.extend(observation.release_blockers)
        blocker_values.extend(release_blockers)
        return cls.glean(
            source=source_value,
            data_kind=observation.data_kind,
            shape=observation.shape,
            schema=observation.schema,
            sample_count=observation.sample_count,
            missingness=observation.missingness,
            duplicates=observation.duplicates,
            label=observation.label,
            content_identity=observation.content_identity,
            limitations=tuple(dict.fromkeys(limitation_values)),
            release_blockers=tuple(dict.fromkeys(blocker_values)),
        )

    @property
    def catalog_digest(self) -> str:
        """The digest of all catalog evidence except this digest.

        This is the ``sha256:`` digest of ``contents()``.
        """
        return canon.digest_value(self.contents())

    def contents(self) -> dict[str, Any]:
        """Returns canonical catalog content without the external digest.

        Returns:
            A JSON-native dictionary of every catalog field, with nested
            summaries converted by their own ``to_dict``.
        """
        return {
            "schema_version": self.schema_version,
            "source": self.source.to_dict(),
            "data_kind": self.data_kind,
            "shape": list(self.shape),
            "schema": [field.to_dict() for field in self.schema],
            "sample_count": self.sample_count,
            "missingness": self.missingness.to_dict(),
            "duplicates": self.duplicates.to_dict(),
            "label": self.label.to_dict(),
            "content_identity": self.content_identity,
            "limitations": list(self.limitations),
            "release_blockers": list(self.release_blockers),
        }

    def to_dict(self) -> dict[str, Any]:
        """Returns JSON-native catalog evidence with its integrity digest.

        Returns:
            The content dictionary plus a ``catalog_digest`` entry.
        """
        record = self.contents()
        record["catalog_digest"] = self.catalog_digest
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
                ``catalog_digest``, is written atomically.
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
    def read_verified(cls, path: str | pathlib.Path) -> Self:
        """Reads a catalog and fail closed if its digest or schema is invalid.

        Args:
            path: A file written by ``write``.

        Returns:
            The catalog rebuilt from the file, whose recorded digest matches
            its recomputed digest.

        Raises:
            ValueError: If the file cannot be read or parsed, is not a JSON
                object, lacks a string ``catalog_digest``, does not rebuild
                into a valid catalog, or its digest does not match the content.
        """
        payload, expected_digest = load(path)
        try:
            catalog = cls.from_dict(payload)
        except (KeyError, TypeError, ValueError) as error:
            raise errors.InputError(
                "catalog evidence schema is invalid"
            ) from error
        if identity.assay(expected_digest, "catalog_digest") != (
            catalog.catalog_digest
        ):
            raise errors.InputError(
                "catalog digest does not match catalog content"
            )
        return catalog

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds a catalog from content without trusting its digest.

        Args:
            record: A mapping of catalog content as produced by ``to_dict``.
                Any ``catalog_digest`` is not checked here. A missing
                required entry other than ``schema_version`` raises
                ``KeyError``; ``release_blockers`` may be absent.

        Returns:
            The reconstructed catalog.

        Raises:
            ValueError: If ``schema_version`` is missing or unsupported.
        """
        if int(record.get("schema_version", 0)) != CATALOG_SCHEMA_VERSION:
            raise errors.InputError("unsupported catalog schema version")
        return cls(
            source=identity.Source.from_dict(record["source"]),
            data_kind=str(record["data_kind"]),
            shape=tuple(int(value) for value in record["shape"]),
            schema=tuple(
                evidence.Field.from_dict(field) for field in record["schema"]
            ),
            sample_count=int(record["sample_count"]),
            missingness=evidence.Missingness.from_dict(record["missingness"]),
            duplicates=evidence.Duplication.from_dict(record["duplicates"]),
            label=evidence.Labeling.from_dict(record["label"]),
            content_identity=str(record["content_identity"]),
            limitations=tuple(str(item) for item in record["limitations"]),
            release_blockers=tuple(
                str(item) for item in record.get("release_blockers", ())
            ),
        )
