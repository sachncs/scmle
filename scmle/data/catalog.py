"""General-purpose dataset curation and catalog evidence.

This module observes caller-owned data before modelling.  It deliberately
stops at data understanding: it does not infer a task, select a model, or
claim that an arbitrary modality is supported for training.
"""

from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
import datetime
import hashlib
import json
import math
import pathlib
from typing import Any
from typing import Self
from urllib.parse import urlsplit
from urllib.parse import urlunsplit

import numpy as np
import pandas as pd

from scmle import durability

CATALOG_SCHEMA_VERSION = 1
DIGEST_LENGTH = 64


def canonical_json(value: Any) -> str:
    """Returns deterministic JSON for an already JSON-compatible value.

    Args:
        value: A JSON-compatible value (mappings, lists, strings, finite
            numbers, booleans, or None).

    Returns:
        Compact JSON text with sorted keys, non-ASCII characters kept, and no
        whitespace between tokens.
    """
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def digest_value(value: Any) -> str:
    """Returns a SHA-256 identity for deterministic JSON content.

    Args:
        value: A JSON-compatible value, serialized with ``canonical_json``.

    Returns:
        The string ``"sha256:"`` followed by 64 lowercase hex digits.
    """
    return (
        f"sha256:{hashlib.sha256(canonical_json(value).encode()).hexdigest()}"
    )


def normalize_uri(uri: str) -> str:
    """Normalizes a source URI without resolving caller-owned remote data.

    Args:
        uri: A URI or a local filesystem path. Surrounding whitespace is
            stripped.

    Returns:
        For a URI with a scheme, the same URI with the scheme lower-cased.
        A single-letter scheme followed by a colon (a Windows drive) is
        treated as a path, as is any value without a scheme: it is expanded
        and resolved to an absolute ``file://`` URI.  Nothing is fetched.
    """
    value = uri.strip()
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


def validate_digest(value: str, field_name: str) -> str:
    """Validates and normalizes a SHA-256 identity.

    Args:
        value: A 64-character hexadecimal digest, optionally prefixed with
            ``"sha256:"``.
        field_name: Name of the field being validated, used in the error
            message.

    Returns:
        The digest as ``"sha256:"`` followed by 64 lowercase hex digits.

    Raises:
        ValueError: If the value (after removing the prefix) is not exactly
            64 characters or is not hexadecimal.
    """
    normalized = value.removeprefix("sha256:")
    if len(normalized) != DIGEST_LENGTH:
        raise ValueError(f"{field_name} must be a SHA-256 digest")
    try:
        int(normalized, 16)
    except ValueError as error:
        raise ValueError(f"{field_name} must be a SHA-256 digest") from error
    return f"sha256:{normalized.lower()}"


@dataclasses.dataclass(frozen=True, slots=True)
class DatasetSource:
    """Identity and governance context for a caller-owned data source."""

    uri: str | pathlib.Path = "memory://"
    version: str = ""
    owner: str = ""
    acquisition_method: str = ""
    privacy_classification: str = ""

    def __post_init__(self) -> None:
        """Validates and normalizes source identity fields.

        Raises:
            ValueError: If ``uri`` is empty or whitespace.
            TypeError: If ``version``, ``owner``, ``acquisition_method``, or
                ``privacy_classification`` is not a string.
        """
        uri_value = str(self.uri)
        if not uri_value.strip():
            raise ValueError("uri must not be empty")
        object.__setattr__(self, "uri", normalize_uri(uri_value))
        for field_name in (
            "version",
            "owner",
            "acquisition_method",
            "privacy_classification",
        ):
            field_value = getattr(self, field_name)
            if not isinstance(field_value, str):
                raise TypeError(f"{field_name} must be a string")
            object.__setattr__(self, field_name, field_value.strip())

    @property
    def source_identity(self) -> str:
        """The deterministic identity of the declared source.

        This is the ``sha256:`` digest of the URI, version, owner,
        acquisition method, and privacy classification.
        """
        return digest_value(
            {
                "uri": self.uri,
                "version": self.version,
                "owner": self.owner,
                "acquisition_method": self.acquisition_method,
                "privacy_classification": self.privacy_classification,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        """Returns JSON-native source evidence.

        Returns:
            A dictionary of the five source fields plus ``source_identity``.
        """
        return {
            "uri": self.uri,
            "version": self.version,
            "owner": self.owner,
            "acquisition_method": self.acquisition_method,
            "privacy_classification": self.privacy_classification,
            "source_identity": self.source_identity,
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuild a source and verify its declared identity.

        Args:
            record: A mapping as produced by ``to_dict``.  ``uri`` is required;
                the other fields default to empty strings.  A missing ``uri``
                raises ``KeyError``.

        Returns:
            The reconstructed source.

        Raises:
            ValueError: If the record's ``source_identity`` is absent or differs
                from the identity recomputed from the source fields.
        """
        source = cls(
            uri=str(record["uri"]),
            version=str(record.get("version", "")),
            owner=str(record.get("owner", "")),
            acquisition_method=str(record.get("acquisition_method", "")),
            privacy_classification=str(
                record.get("privacy_classification", "")
            ),
        )
        expected = record.get("source_identity")
        if expected != source.source_identity:
            raise ValueError("source_identity does not match source fields")
        return source


@dataclasses.dataclass(frozen=True, slots=True)
class DatasetField:
    """One ordered field in an observed dataset schema."""

    name: str
    data_type: str

    def __post_init__(self) -> None:
        """Validates schema field names and types, and strips their spaces.

        Raises:
            ValueError: If ``name`` or ``data_type`` is not a string or is
                empty after stripping whitespace.
        """
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("field name must not be empty")
        if not isinstance(self.data_type, str) or not self.data_type.strip():
            raise ValueError("field data_type must not be empty")
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "data_type", self.data_type.strip())

    def to_dict(self) -> dict[str, str]:
        """Returns JSON-native field evidence.

        Returns:
            A dictionary with ``name`` and ``data_type``.
        """
        return {"name": self.name, "data_type": self.data_type}

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuild a schema field from JSON evidence.

        Args:
            record: A mapping with ``name`` and ``data_type`` entries.

        Returns:
            The reconstructed field.
        """
        return cls(str(record["name"]), str(record["data_type"]))


@dataclasses.dataclass(frozen=True, slots=True)
class MissingnessSummary:
    """Observed missing-value state, including fields with missingness."""

    status: str
    total_values: int
    missing_values: int
    rows_with_missing: int
    by_field: Mapping[str, int]

    def __post_init__(self) -> None:
        """Validates missingness accounting.

        Raises:
            TypeError: If a count is not an integer (booleans are rejected).
            ValueError: If ``status`` is empty, a count is negative,
                ``missing_values`` exceeds ``total_values``, or ``by_field``
                has a blank name or a negative or non-integer count.
        """
        if not self.status.strip():
            raise ValueError("missingness status must not be empty")
        for field_name, field_value in (
            ("total_values", self.total_values),
            ("missing_values", self.missing_values),
            ("rows_with_missing", self.rows_with_missing),
        ):
            if isinstance(field_value, bool) or not isinstance(
                field_value, int
            ):
                raise TypeError(f"{field_name} must be an integer")
            if field_value < 0:
                raise ValueError(f"{field_name} must not be negative")
        if self.missing_values > self.total_values:
            raise ValueError("missing_values cannot exceed total_values")
        if any(
            not isinstance(name, str)
            or not name.strip()
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count < 0
            for name, count in self.by_field.items()
        ):
            raise ValueError("by_field must contain non-negative named counts")
        object.__setattr__(
            self, "by_field", dict(sorted(self.by_field.items()))
        )

    @property
    def fraction(self) -> float:
        """The fraction of observed values that are missing.

        This is ``missing_values / total_values``, or 0.0 when there are no
        values.
        """
        if self.total_values == 0:
            return 0.0
        return self.missing_values / self.total_values

    def to_dict(self) -> dict[str, Any]:
        """Returns JSON-native missingness evidence.

        Returns:
            A dictionary of the counts, the per-field counts, and the computed
            ``missing_fraction``.
        """
        return {
            "status": self.status,
            "total_values": self.total_values,
            "missing_values": self.missing_values,
            "missing_fraction": self.fraction,
            "rows_with_missing": self.rows_with_missing,
            "by_field": dict(self.by_field),
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuild missingness evidence and recompute its fraction.

        Args:
            record: A mapping as produced by ``to_dict``.  A recorded
                ``missing_fraction`` is optional and is checked when present.

        Returns:
            The reconstructed summary.

        Raises:
            ValueError: If a recorded ``missing_fraction`` does not match the
                fraction computed from the counts.
        """
        result = cls(
            status=str(record["status"]),
            total_values=int(record["total_values"]),
            missing_values=int(record["missing_values"]),
            rows_with_missing=int(record["rows_with_missing"]),
            by_field={
                str(name): int(count)
                for name, count in dict(record.get("by_field", {})).items()
            },
        )
        observed_fraction = record.get("missing_fraction")
        if observed_fraction is not None and not math.isclose(
            float(observed_fraction), result.fraction
        ):
            raise ValueError("missing_fraction does not match missing values")
        return result


@dataclasses.dataclass(frozen=True, slots=True)
class DuplicateSummary:
    """Observed duplicate-row state and the method used to measure it."""

    status: str
    duplicate_rows: int
    duplicate_fraction: float
    method: str

    def __post_init__(self) -> None:
        """Validates duplicate accounting.

        Raises:
            ValueError: If ``status`` or ``method`` is empty, ``duplicate_rows``
                is negative, or ``duplicate_fraction`` is not a finite number
                in [0, 1].
            TypeError: If ``duplicate_rows`` is not an integer (booleans are
                rejected).
        """
        if not self.status.strip() or not self.method.strip():
            raise ValueError("duplicate status and method must not be empty")
        if isinstance(self.duplicate_rows, bool) or not isinstance(
            self.duplicate_rows, int
        ):
            raise TypeError("duplicate_rows must be an integer")
        if self.duplicate_rows < 0:
            raise ValueError("duplicate_rows must not be negative")
        if (
            not math.isfinite(self.duplicate_fraction)
            or not 0 <= (self.duplicate_fraction) <= 1
        ):
            raise ValueError("duplicate_fraction must be between zero and one")

    def to_dict(self) -> dict[str, Any]:
        """Returns JSON-native duplicate evidence.

        Returns:
            A dictionary of status, duplicate row count, duplicate fraction,
            and method.
        """
        return {
            "status": self.status,
            "duplicate_rows": self.duplicate_rows,
            "duplicate_fraction": self.duplicate_fraction,
            "method": self.method,
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuild duplicate evidence.

        Args:
            record: A mapping as produced by ``to_dict``.

        Returns:
            The reconstructed summary.
        """
        return cls(
            status=str(record["status"]),
            duplicate_rows=int(record["duplicate_rows"]),
            duplicate_fraction=float(record["duplicate_fraction"]),
            method=str(record["method"]),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class LabelSummary:
    """Explicit label state without inferring a learning task."""

    status: str
    field: str | None
    observed_count: int
    missing_count: int
    distinct_count: int
    distribution: Mapping[str, int]

    def __post_init__(self) -> None:
        """Validates label accounting.

        Raises:
            ValueError: If ``status`` is empty, ``field`` is blank, a count is
                negative, ``distinct_count`` exceeds ``observed_count``,
                ``distribution`` has an empty key or a negative or non-integer
                count, or its counts do not sum to ``observed_count``.
            TypeError: If a count is not an integer (booleans are rejected).
        """
        if not self.status.strip():
            raise ValueError("label status must not be empty")
        if self.field is not None and not self.field.strip():
            raise ValueError("label field must not be empty")
        for field_name, field_value in (
            ("observed_count", self.observed_count),
            ("missing_count", self.missing_count),
            ("distinct_count", self.distinct_count),
        ):
            if isinstance(field_value, bool) or not isinstance(
                field_value, int
            ):
                raise TypeError(f"{field_name} must be an integer")
            if field_value < 0:
                raise ValueError(f"{field_name} must not be negative")
        if self.distinct_count > self.observed_count:
            raise ValueError("distinct_count cannot exceed observed_count")
        if any(
            not isinstance(key, str)
            or not key
            or isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for key, value in self.distribution.items()
        ):
            raise ValueError("distribution must contain non-negative counts")
        if sum(self.distribution.values()) != self.observed_count:
            raise ValueError("distribution does not match observed_count")
        object.__setattr__(
            self, "distribution", dict(sorted(self.distribution.items()))
        )

    def to_dict(self) -> dict[str, Any]:
        """Returns JSON-native label evidence.

        Returns:
            A dictionary of status, field, counts, and the label distribution.
        """
        return {
            "status": self.status,
            "field": self.field,
            "observed_count": self.observed_count,
            "missing_count": self.missing_count,
            "distinct_count": self.distinct_count,
            "distribution": dict(self.distribution),
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuild label evidence.

        Args:
            record: A mapping as produced by ``to_dict``.  ``distribution`` is
                optional and defaults to empty.

        Returns:
            The reconstructed summary.
        """
        return cls(
            status=str(record["status"]),
            field=record.get("field"),
            observed_count=int(record["observed_count"]),
            missing_count=int(record["missing_count"]),
            distinct_count=int(record["distinct_count"]),
            distribution={
                str(label): int(count)
                for label, count in dict(record.get("distribution", {})).items()
            },
        )


@dataclasses.dataclass(frozen=True, slots=True)
class DatasetCatalog:
    """Versioned, model-agnostic curation evidence for one data snapshot.

    The contract accepts arbitrary caller-owned observations, but it never
    silently invents a content hash for an opaque object.  Use ``from_data``
    for supported Python data structures or ``from_observation`` when a
    caller-owned adapter can provide trustworthy evidence for another format.
    """

    source: DatasetSource
    data_kind: str
    shape: tuple[int, ...]
    schema: tuple[DatasetField, ...]
    sample_count: int
    missingness: MissingnessSummary
    duplicates: DuplicateSummary
    label: LabelSummary
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
        if not isinstance(self.source, DatasetSource):
            raise TypeError("source must be DatasetSource")
        if not isinstance(self.data_kind, str) or not self.data_kind.strip():
            raise ValueError("data_kind must not be empty")
        if isinstance(self.sample_count, bool) or not isinstance(
            self.sample_count, int
        ):
            raise TypeError("sample_count must be an integer")
        if self.sample_count < 0:
            raise ValueError("sample_count must not be negative")
        if isinstance(self.schema_version, bool) or not isinstance(
            self.schema_version, int
        ):
            raise TypeError("schema_version must be an integer")
        if self.schema_version != CATALOG_SCHEMA_VERSION:
            raise ValueError("unsupported catalog schema version")
        if any(
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension < 0
            for dimension in self.shape
        ):
            raise ValueError("shape must contain non-negative integers")
        fields = tuple(self.schema)
        if any(not isinstance(field, DatasetField) for field in fields):
            raise TypeError("schema must contain DatasetField records")
        if not isinstance(self.missingness, MissingnessSummary):
            raise TypeError("missingness must be MissingnessSummary")
        if not isinstance(self.duplicates, DuplicateSummary):
            raise TypeError("duplicates must be DuplicateSummary")
        if not isinstance(self.label, LabelSummary):
            raise TypeError("label must be LabelSummary")
        if not self.limitations or any(
            not isinstance(item, str) or not item.strip()
            for item in self.limitations
        ):
            raise ValueError("limitations must contain non-empty strings")
        if any(
            not isinstance(item, str) or not item.strip()
            for item in self.release_blockers
        ):
            raise ValueError("release_blockers must contain non-empty strings")
        object.__setattr__(self, "schema", fields)
        object.__setattr__(
            self,
            "content_identity",
            validate_digest(self.content_identity, "content_identity"),
        )
        object.__setattr__(self, "data_kind", self.data_kind.strip())
        object.__setattr__(self, "limitations", tuple(self.limitations))
        object.__setattr__(
            self,
            "release_blockers",
            tuple(dict.fromkeys(self.release_blockers)),
        )

    @classmethod
    def from_observation(
        cls,
        source: DatasetSource | str | pathlib.Path,
        data_kind: str,
        shape: Sequence[int],
        schema: Sequence[DatasetField],
        sample_count: int,
        missingness: MissingnessSummary,
        duplicates: DuplicateSummary,
        label: LabelSummary,
        content_identity: str,
        limitations: Sequence[str],
        release_blockers: Sequence[str] = (),
    ) -> Self:
        """Builds a catalog from adapter-owned evidence for any data format.

        Args:
            source: The data source, or a URI or path that is wrapped in a
                ``DatasetSource`` with default governance fields.
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
        source_value = (
            source
            if isinstance(source, DatasetSource)
            else DatasetSource(source)
        )
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
        source: DatasetSource | str | pathlib.Path = "memory://",
        label_column: str | None = None,
        content_identity: str | None = None,
        limitations: Sequence[str] = (),
        release_blockers: Sequence[str] = (),
    ) -> Self:
        """Observe common Python dataset forms without fitting a model.

        Supported forms include pandas frames and series, NumPy arrays,
        mappings of columns, sequences of records or values, scalars, and
        bytes.  Opaque objects require ``content_identity`` so the catalog
        cannot publish a non-reproducible identity.

        Args:
            data: The object to observe.
            source: The data source, or a URI or path that is wrapped in a
                ``DatasetSource`` with default governance fields.
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
        source_value = (
            source
            if isinstance(source, DatasetSource)
            else DatasetSource(source)
        )
        observation = observe_data(data, label_column, content_identity)
        limitation_values = [
            "curation_only_no_model_fit_or_performance_claim",
            *observation.limitations,
            *limitations,
        ]
        blocker_values = [
            *source_blockers(source_value),
            *observation.release_blockers,
            *release_blockers,
        ]
        return cls.from_observation(
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

        This is the ``sha256:`` digest of ``content_dict()``.
        """
        return digest_value(self.content_dict())

    def content_dict(self) -> dict[str, Any]:
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
        record = self.content_dict()
        record["catalog_digest"] = self.catalog_digest
        return record

    def canonical_json(self) -> str:
        """Returns canonical catalog content suitable for verification.

        Returns:
            Canonical JSON text of ``content_dict()``.
        """
        return canonical_json(self.content_dict())

    def write(self, path: str | pathlib.Path) -> None:
        """Persist the catalog atomically as verified JSON evidence.

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
        try:
            record = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError) as error:
            raise ValueError(
                "catalog evidence is missing or invalid JSON"
            ) from error
        if not isinstance(record, dict):
            raise ValueError("catalog evidence must be a JSON object")
        expected_digest = record.get("catalog_digest")
        if not isinstance(expected_digest, str):
            raise ValueError("catalog evidence is missing catalog_digest")
        payload = dict(record)
        del payload["catalog_digest"]
        try:
            catalog = cls.from_dict(payload)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("catalog evidence schema is invalid") from error
        if validate_digest(expected_digest, "catalog_digest") != (
            catalog.catalog_digest
        ):
            raise ValueError("catalog digest does not match catalog content")
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
            raise ValueError("unsupported catalog schema version")
        return cls(
            source=DatasetSource.from_dict(record["source"]),
            data_kind=str(record["data_kind"]),
            shape=tuple(int(value) for value in record["shape"]),
            schema=tuple(
                DatasetField.from_dict(field) for field in record["schema"]
            ),
            sample_count=int(record["sample_count"]),
            missingness=MissingnessSummary.from_dict(record["missingness"]),
            duplicates=DuplicateSummary.from_dict(record["duplicates"]),
            label=LabelSummary.from_dict(record["label"]),
            content_identity=str(record["content_identity"]),
            limitations=tuple(str(item) for item in record["limitations"]),
            release_blockers=tuple(
                str(item) for item in record.get("release_blockers", ())
            ),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class DataObservation:
    """Intermediate evidence produced by the supported data observers."""

    data_kind: str
    shape: tuple[int, ...]
    schema: tuple[DatasetField, ...]
    sample_count: int
    missingness: MissingnessSummary
    duplicates: DuplicateSummary
    label: LabelSummary
    content_identity: str
    limitations: tuple[str, ...]
    release_blockers: tuple[str, ...]


def source_blockers(source: DatasetSource) -> tuple[str, ...]:
    """Returns governance blockers that can be proven from source metadata.

    Args:
        source: The data source whose governance fields are checked.

    Returns:
        Blocker names, one for each of owner, acquisition method, and
        privacy classification that is empty.
    """
    blockers: list[str] = []
    if not source.owner:
        blockers.append("source_owner_not_declared")
    if not source.acquisition_method:
        blockers.append("source_acquisition_method_not_declared")
    if not source.privacy_classification:
        blockers.append("privacy_and_retention_not_assessed")
    return tuple(blockers)


def is_sequence_value(value: Any) -> bool:
    """Returns whether a value can represent an ordered column.

    Args:
        value: Any object.

    Returns:
        True for sequences and NumPy arrays other than strings, bytes,
        byte arrays, and mappings.
    """
    return isinstance(value, (Sequence, np.ndarray)) and not isinstance(
        value, (str, bytes, bytearray, Mapping)
    )


def frame_for_data(
    data: Any,
) -> tuple[pd.DataFrame | None, str, tuple[int, ...]]:
    """Converts supported record-shaped data to a bounded observation frame.

    Args:
        data: The object to convert.

    Returns:
        A tuple ``(frame, data_kind, shape)``.  ``frame`` is a DataFrame,
        or None for byte payloads, NumPy arrays with more than two
        dimensions, and opaque objects.  ``data_kind`` names the detected
        form (the type name for opaque objects).  ``shape`` is the
        dimensions of the data, ``()`` for scalars and opaque objects.

    Raises:
        ValueError: If a mapping of columns has columns of unequal length.
    """
    if isinstance(data, pd.DataFrame):
        return (
            data,
            "pandas_dataframe",
            tuple(int(value) for value in data.shape),
        )
    if isinstance(data, pd.Series):
        name = str(data.name) if data.name is not None else "value"
        frame = data.to_frame(name=name)
        return (
            frame,
            "pandas_series",
            tuple(int(value) for value in frame.shape),
        )
    if isinstance(data, np.ndarray):
        if data.ndim == 0:
            frame = pd.DataFrame({"value": [data.item()]})
            return frame, "numpy_scalar", ()
        shape = tuple(int(value) for value in data.shape)
        if data.ndim == 1:
            frame = pd.DataFrame({"value": data})
            return frame, "numpy_array", shape
        if data.ndim == 2:
            frame = pd.DataFrame(
                data, columns=[str(value) for value in range(data.shape[1])]
            )
            return frame, "numpy_array", shape
        return None, "numpy_array", shape
    if isinstance(data, Mapping):
        values = list(data.values())
        if values and all(is_sequence_value(value) for value in values):
            lengths = {len(value) for value in values}
            if len(lengths) != 1:
                raise ValueError("mapping columns must have equal lengths")
            frame = pd.DataFrame(data)
            return (
                frame,
                "column_mapping",
                tuple(int(value) for value in frame.shape),
            )
        frame = pd.DataFrame([dict(data)])
        return (
            frame,
            "record_mapping",
            tuple(int(value) for value in frame.shape),
        )
    if isinstance(data, (bytes, bytearray, memoryview)):
        return None, "bytes", (len(data),)
    if isinstance(data, (list, tuple)):
        if data and all(isinstance(value, Mapping) for value in data):
            frame = pd.DataFrame.from_records(data)
            return (
                frame,
                "record_sequence",
                tuple(int(value) for value in frame.shape),
            )
        frame = pd.DataFrame({"value": list(data)})
        return frame, "sequence", tuple(int(value) for value in frame.shape)
    if isinstance(data, (str, int, float, bool, datetime.date)) or data is None:
        frame = pd.DataFrame({"value": [data]})
        return frame, "scalar", ()
    return None, type(data).__name__, ()


def canonical_value(value: Any) -> Any:
    """Converts one supported value into deterministic JSON-native content.

    Args:
        value: A scalar, bytes, date or time, mapping, list, tuple, set, or
            NumPy value.

    Returns:
        None for missing values; a length-and-SHA-256 mapping for bytes;
        an ISO string for dates and times; a mapping with string keys
        sorted by key for mappings; lists in order for sequences; sorted
        lists for sets; a ``{"non_finite": ...}`` mapping for NaN and
        infinities; otherwise the plain string, number, or boolean.

    Raises:
        TypeError: If the value is of a type that cannot be converted
            deterministically.
    """
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        return {
            "bytes_length": len(raw),
            "bytes_sha256": hashlib.sha256(raw).hexdigest(),
        }
    if isinstance(value, np.generic):
        return canonical_value(value.item())
    if isinstance(value, (datetime.datetime, datetime.date, datetime.time)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {
            str(key): canonical_value(item)
            for key, item in sorted(
                value.items(), key=lambda pair: str(pair[0])
            )
        }
    if isinstance(value, (list, tuple)):
        return [canonical_value(item) for item in value]
    if isinstance(value, set | frozenset):
        values = [canonical_value(item) for item in value]
        return sorted(values, key=canonical_json)
    if isinstance(value, float):
        if math.isnan(value):
            return {"non_finite": "nan"}
        if math.isinf(value):
            return {"non_finite": "-inf" if value < 0 else "inf"}
    missing = pd.isna(value)
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.ndarray):
        return [canonical_value(item) for item in value.tolist()]
    raise TypeError(
        f"cannot deterministically catalog value of type {type(value).__name__}"
    )


def schema_for_frame(frame: pd.DataFrame) -> tuple[DatasetField, ...]:
    """Returns ordered field metadata, preserving duplicate column positions.

    Args:
        frame: The frame whose columns are described.

    Returns:
        One field per column, in column order, with the dtype as a string.
    """
    return tuple(
        DatasetField(str(name), str(data_type))
        for name, data_type in zip(
            frame.columns, frame.dtypes.astype(str), strict=True
        )
    )


def missingness_for_frame(frame: pd.DataFrame) -> MissingnessSummary:
    """Measure missingness over a frame without imputing or modifying it.

    Args:
        frame: The frame to inspect.

    Returns:
        An observed summary.  Per-field counts are keyed by column name,
        with ``name[position]`` used for names that occur more than once.
    """
    missing = frame.isna()
    counts = missing.sum(axis=0)
    missing_values = int(counts.sum())
    names = [str(name) for name in frame.columns]
    field_counts = {
        (name if names.count(name) == 1 else f"{name}[{position}]"): int(count)
        for position, (name, count) in enumerate(
            zip(names, counts, strict=True)
        )
    }
    return MissingnessSummary(
        status="observed",
        total_values=int(frame.shape[0] * frame.shape[1]),
        missing_values=missing_values,
        rows_with_missing=int(missing.any(axis=1).sum()),
        by_field=field_counts,
    )


def duplicates_for_frame(frame: pd.DataFrame) -> DuplicateSummary:
    """Measure duplicate rows and preserve unsupported cases as evidence.

    Args:
        frame: The frame to inspect.

    Returns:
        A summary counting every row that has at least one identical row
        (all copies, not only the repeats).  If row equality is not
        supported for some field, a ``not_observed`` summary with zero
        duplicates.
    """
    try:
        duplicate_mask = frame.duplicated(keep=False)
    except (TypeError, ValueError):
        return DuplicateSummary(
            status="not_observed",
            duplicate_rows=0,
            duplicate_fraction=0.0,
            method="row equality was not supported for one or more fields",
        )
    duplicate_rows = int(duplicate_mask.sum())
    fraction = duplicate_rows / len(frame) if len(frame) else 0.0
    return DuplicateSummary(
        status="observed",
        duplicate_rows=duplicate_rows,
        duplicate_fraction=fraction,
        method="pandas.DataFrame.duplicated(keep=False)",
    )


def label_for_frame(
    frame: pd.DataFrame | None, label_column: str | None
) -> tuple[LabelSummary, tuple[str, ...], tuple[str, ...]]:
    """Summarize only an explicitly named label column.

    Args:
        frame: The observation frame, or None when there is none.
        label_column: Name of the declared label column, or None.

    Returns:
        A tuple ``(summary, limitations, blockers)``.  The summary is
        ``not_declared`` without a label column, ``not_observed`` when the
        column is absent or appears more than once, and otherwise
        ``missing_all``, ``constant``, or ``observed`` with a distribution
        keyed by canonical JSON of each value.
    """
    if label_column is None:
        return (
            LabelSummary("not_declared", None, 0, 0, 0, {}),
            ("label_was_not_declared",),
            ("label_contract_not_declared_for_supervised_use",),
        )
    if frame is None or label_column not in frame.columns:
        return (
            LabelSummary("not_observed", label_column, 0, 0, 0, {}),
            ("declared_label_could_not_be_observed",),
            ("declared_label_field_is_not_available",),
        )
    if list(frame.columns).count(label_column) != 1:
        return (
            LabelSummary("not_observed", label_column, 0, 0, 0, {}),
            ("declared_label_field_is_ambiguous",),
            ("declared_label_field_is_ambiguous",),
        )
    series = frame[label_column]
    missing_count = int(series.isna().sum())
    observed = series[~series.isna()]
    distribution: dict[str, int] = {}
    for value in observed.tolist():
        bucket = canonical_json(canonical_value(value))
        distribution[bucket] = distribution.get(bucket, 0) + 1
    observed_count = len(observed)
    distinct_count = len(distribution)
    status = (
        "missing_all"
        if not observed_count
        else ("constant" if distinct_count == 1 else "observed")
    )
    return (
        LabelSummary(
            status,
            label_column,
            observed_count,
            missing_count,
            distinct_count,
            distribution,
        ),
        (),
        ("label_is_constant",) if status == "constant" else (),
    )


def frame_content(
    frame: pd.DataFrame, data_kind: str, shape: tuple[int, ...]
) -> dict[str, Any]:
    """Returns canonical content for a frame-backed observation.

    Args:
        frame: The observed frame.
        data_kind: Name of the detected data form.
        shape: Dimensions of the original data.

    Returns:
        A JSON-native dictionary of the kind, shape, schema, and all rows
        in order, with each value canonicalized.
    """
    rows = [
        [canonical_value(value) for value in row]
        for row in frame.itertuples(index=False, name=None)
    ]
    return {
        "data_kind": data_kind,
        "shape": list(shape),
        "schema": [field.to_dict() for field in schema_for_frame(frame)],
        "rows": rows,
    }


def bytes_content(data: bytes | bytearray | memoryview) -> str:
    """Returns direct content identity for raw byte observations.

    Args:
        data: The raw bytes.

    Returns:
        The ``sha256:``-prefixed digest of the bytes.
    """
    return f"sha256:{hashlib.sha256(bytes(data)).hexdigest()}"


def observe_data(
    data: Any,
    label_column: str | None,
    supplied_identity: str | None,
) -> DataObservation:
    """Observe a supported object and produce structured quality evidence.

    Args:
        data: The object to observe.
        label_column: Name of the declared label column, or None.
        supplied_identity: Caller-supplied SHA-256 digest of the content,
            or None to compute one when the content is observable.

    Returns:
        The observation.  Frame-backed data gets schema, missingness,
        duplicate, and label evidence; bytes get a single payload field;
        opaque objects get only the supplied identity and a
        single-sample placeholder.

    Raises:
        TypeError: If the object is opaque (neither frame-backed nor
            bytes) and no ``supplied_identity`` is given.
    """
    frame, data_kind, shape = frame_for_data(data)
    limitations: list[str] = []
    blockers: list[str] = []
    if frame is not None:
        schema = schema_for_frame(frame)
        sample_count = int(frame.shape[0])
        missingness = missingness_for_frame(frame)
        duplicates = duplicates_for_frame(frame)
        label, label_limitations, label_blockers = label_for_frame(
            frame, label_column
        )
        limitations.extend(label_limitations)
        blockers.extend(label_blockers)
        if missingness.missing_values:
            limitations.append("missing_values_are_reported_not_repaired")
            blockers.append("missingness_policy_not_declared")
        if duplicates.duplicate_rows:
            limitations.append("duplicate_rows_are_reported_not_removed")
            blockers.append("duplicate_policy_not_declared")
        if supplied_identity is None:
            content_identity = digest_value(
                frame_content(frame, data_kind, shape)
            )
        else:
            content_identity = validate_digest(
                supplied_identity, "content_identity"
            )
            limitations.append("content_identity_was_supplied_by_caller")
        return DataObservation(
            data_kind=data_kind,
            shape=shape,
            schema=schema,
            sample_count=sample_count,
            missingness=missingness,
            duplicates=duplicates,
            label=label,
            content_identity=content_identity,
            limitations=tuple(dict.fromkeys(limitations)),
            release_blockers=tuple(dict.fromkeys(blockers)),
        )
    if isinstance(data, (bytes, bytearray, memoryview)):
        schema = (DatasetField("payload", "bytes"),)
        content_identity = (
            bytes_content(data)
            if supplied_identity is None
            else validate_digest(supplied_identity, "content_identity")
        )
        if supplied_identity is not None:
            limitations.append("content_identity_was_supplied_by_caller")
        limitations.append("modality_specific_decoding_was_not_attempted")
        label, label_limitations, label_blockers = label_for_frame(
            None, label_column
        )
        limitations.extend(label_limitations)
        blockers.extend(label_blockers)
        blockers.append("modality_specific_decoder_not_declared")
        return DataObservation(
            data_kind=data_kind,
            shape=shape,
            schema=schema,
            sample_count=1,
            missingness=MissingnessSummary("not_observed", 0, 0, 0, {}),
            duplicates=DuplicateSummary(
                "not_observed", 0, 0.0, "raw bytes have no row identity"
            ),
            label=label,
            content_identity=content_identity,
            limitations=tuple(limitations),
            release_blockers=tuple(blockers),
        )
    if supplied_identity is None:
        raise TypeError(
            "opaque data needs a caller-supplied SHA-256 content_identity"
        )
    content_identity = validate_digest(supplied_identity, "content_identity")
    label, label_limitations, label_blockers = label_for_frame(
        None, label_column
    )
    limitations.extend(
        (
            "schema_and_quality_metrics_were_not_observed",
            "content_identity_was_supplied_by_caller",
            *label_limitations,
        )
    )
    blockers.extend(("schema_adapter_not_declared", *label_blockers))
    return DataObservation(
        data_kind=data_kind,
        shape=shape,
        schema=(DatasetField("value", type(data).__name__),),
        sample_count=1,
        missingness=MissingnessSummary("not_observed", 0, 0, 0, {}),
        duplicates=DuplicateSummary(
            "not_observed", 0, 0.0, "opaque object has no row identity"
        ),
        label=label,
        content_identity=content_identity,
        limitations=tuple(limitations),
        release_blockers=tuple(blockers),
    )


__all__ = [
    "DatasetCatalog",
    "DatasetField",
    "DatasetSource",
    "DuplicateSummary",
    "LabelSummary",
    "MissingnessSummary",
]
