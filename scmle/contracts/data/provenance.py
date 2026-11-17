"""Dataset identity: origin, lineage, confidentiality, manifest and passport."""

from collections.abc import Mapping
import dataclasses
from typing import Any
from typing import Self

from scmle.contracts import errors
from scmle.contracts.data import findings

SHA256_HEX_LENGTH = 64


@dataclasses.dataclass(frozen=True, slots=True)
class Origin:
    """Provenance and semantic context for the dataset source.

    Attributes:
        owner: Team or organization accountable for the source.
        acquisition_method: How the source was acquired.
        captured_at: Source capture timestamp, when known.
        timezone: Timezone used by source timestamps.
        sampling_frame: Population or sampling frame represented by the data.
        units: Units by semantic field name.
        semantic_definitions: Human-readable definitions by field name.
        extraction_query: Query or predicate used to extract the source.
        object_prefix: Object-store prefix or partition selector.
        license: License or usage grant covering the source.
        access_policy: Policy governing access to the source.
    """

    owner: str = ""
    acquisition_method: str = ""
    captured_at: str = ""
    timezone: str = ""
    sampling_frame: str = ""
    units: Mapping[str, str] = dataclasses.field(default_factory=dict)
    semantic_definitions: Mapping[str, str] = dataclasses.field(
        default_factory=dict
    )
    extraction_query: str = ""
    object_prefix: str = ""
    license: str = ""
    access_policy: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON- and Arrow-compatible mapping."""
        return {
            "owner": self.owner,
            "acquisition_method": self.acquisition_method,
            "captured_at": self.captured_at,
            "timezone": self.timezone,
            "sampling_frame": self.sampling_frame,
            "units": dict(self.units),
            "semantic_definitions": dict(self.semantic_definitions),
            "extraction_query": self.extraction_query,
            "object_prefix": self.object_prefix,
            "license": self.license,
            "access_policy": self.access_policy,
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds source metadata from its serialized representation.

        Args:
            record: A mapping as produced by `to_dict`; absent fields are empty.

        Returns:
            The source metadata.
        """
        return cls(
            owner=str(record.get("owner", "")),
            acquisition_method=str(record.get("acquisition_method", "")),
            captured_at=str(record.get("captured_at", "")),
            timezone=str(record.get("timezone", "")),
            sampling_frame=str(record.get("sampling_frame", "")),
            units=dict(record.get("units", {})),
            semantic_definitions=dict(record.get("semantic_definitions", {})),
            extraction_query=str(record.get("extraction_query", "")),
            object_prefix=str(record.get("object_prefix", "")),
            license=str(record.get("license", "")),
            access_policy=str(record.get("access_policy", "")),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Ancestry:
    """Transform and label provenance for an immutable dataset snapshot."""

    source_snapshot: str = ""
    parent_datasets: tuple[str, ...] = ()
    transformations: tuple[str, ...] = ()
    label_source: str = ""
    label_generator_version: str = ""
    annotator_version: str = ""
    adjudication_status: str = ""
    partition_ids: tuple[str, ...] = ()
    object_hashes: Mapping[str, str] = dataclasses.field(default_factory=dict)
    schema_evolution: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON- and Arrow-compatible mapping."""
        return {
            "source_snapshot": self.source_snapshot,
            "parent_datasets": list(self.parent_datasets),
            "transformations": list(self.transformations),
            "label_source": self.label_source,
            "label_generator_version": self.label_generator_version,
            "annotator_version": self.annotator_version,
            "adjudication_status": self.adjudication_status,
            "partition_ids": list(self.partition_ids),
            "object_hashes": dict(self.object_hashes),
            "schema_evolution": self.schema_evolution,
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds lineage metadata from its serialized representation.

        Args:
            record: A mapping as produced by `to_dict`; absent fields are empty.

        Returns:
            The lineage metadata.
        """
        return cls(
            source_snapshot=str(record.get("source_snapshot", "")),
            parent_datasets=tuple(record.get("parent_datasets", ())),
            transformations=tuple(record.get("transformations", ())),
            label_source=str(record.get("label_source", "")),
            label_generator_version=str(
                record.get("label_generator_version", "")
            ),
            annotator_version=str(record.get("annotator_version", "")),
            adjudication_status=str(record.get("adjudication_status", "")),
            partition_ids=tuple(record.get("partition_ids", ())),
            object_hashes=dict(record.get("object_hashes", {})),
            schema_evolution=str(record.get("schema_evolution", "")),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Confidentiality:
    """Privacy, consent, and retention controls for a dataset snapshot."""

    classification: str = "unclassified"
    contains_pii: bool = False
    sensitive_fields: tuple[str, ...] = ()
    consent_basis: str = ""
    retention_period: str = ""
    deletion_policy: str = ""
    access_policy: str = ""
    redaction_required: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON- and Arrow-compatible mapping."""
        return {
            "classification": self.classification,
            "contains_pii": self.contains_pii,
            "sensitive_fields": list(self.sensitive_fields),
            "consent_basis": self.consent_basis,
            "retention_period": self.retention_period,
            "deletion_policy": self.deletion_policy,
            "access_policy": self.access_policy,
            "redaction_required": self.redaction_required,
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds privacy metadata from its serialized representation.

        Args:
            record: A mapping as produced by `to_dict`; absent fields take
                their defaults.

        Returns:
            The privacy metadata.
        """
        return cls(
            classification=str(record.get("classification", "unclassified")),
            contains_pii=bool(record.get("contains_pii", False)),
            sensitive_fields=tuple(record.get("sensitive_fields", ())),
            consent_basis=str(record.get("consent_basis", "")),
            retention_period=str(record.get("retention_period", "")),
            deletion_policy=str(record.get("deletion_policy", "")),
            access_policy=str(record.get("access_policy", "")),
            redaction_required=bool(record.get("redaction_required", False)),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class DatasetManifest:
    """Immutable identity of an ingested dataset.

    Attributes:
        dataset_id: Short identifier derived from the content hash.
        schema_hash: Hash of column names and dtypes.
        data_hash: Hash of the row contents.
        row_count: Number of rows.
        column_count: Number of columns.
        target: Name of the target column.
        uri: Where the data was read from.
        snapshot_id: Immutable identity of the source snapshot.
        modality: Dataset modality, such as `tabular` or `image`.
        task: Intended learning task, when known.
        schema_version: Version of this manifest contract.
        source: Source ownership, acquisition, and semantic metadata.
        lineage: Source, transformation, and label lineage metadata.
        privacy: Privacy, consent, retention, and access metadata.
        findings: Structured findings attached to this snapshot.
    """

    dataset_id: str
    schema_hash: str
    data_hash: str
    row_count: int
    column_count: int
    target: str
    uri: str = ""
    snapshot_id: str = ""
    modality: str = "tabular"
    task: str = ""
    schema_version: int = 1
    source: Origin = dataclasses.field(default_factory=Origin)
    lineage: Ancestry = dataclasses.field(default_factory=Ancestry)
    privacy: Confidentiality = dataclasses.field(
        default_factory=Confidentiality
    )
    findings: tuple[findings.Finding, ...] = ()

    def __post_init__(self) -> None:
        """Validates dataset identity before it enters a run or artifact.

        Raises:
            TypeError: If a count or the schema version is not an integer,
                a metadata component has the wrong type, or a finding is not
                a `Finding`.
            ValueError: If an identity field (dataset id, hashes, target or
                modality) is blank or not a string, a count is negative, or
                the schema version is below one.
        """
        text_fields: tuple[tuple[str, object], ...] = (
            ("dataset_id", self.dataset_id),
            ("schema_hash", self.schema_hash),
            ("data_hash", self.data_hash),
            ("target", self.target),
            ("modality", self.modality),
        )
        for field_name, field_value in text_fields:
            if not isinstance(field_value, str) or not field_value.strip():
                raise errors.InputError(f"{field_name} must not be empty")
        for count_name, count_value in (
            ("row_count", self.row_count),
            ("column_count", self.column_count),
            ("schema_version", self.schema_version),
        ):
            if isinstance(count_value, bool) or not isinstance(
                count_value, int
            ):
                raise errors.KindError(f"{count_name} must be an integer")
        if self.row_count < 0:
            raise errors.InputError("row_count must not be negative")
        if self.column_count < 0:
            raise errors.InputError("column_count must not be negative")
        if self.schema_version < 1:
            raise errors.InputError("schema_version must be positive")
        for component_name, component, expected in (
            ("source", self.source, Origin),
            ("lineage", self.lineage, Ancestry),
            ("privacy", self.privacy, Confidentiality),
        ):
            if not isinstance(component, expected):
                raise errors.KindError(
                    f"{component_name} must be {expected.__name__}"
                )
        object.__setattr__(self, "findings", tuple(self.findings))
        findings.vet(self.findings)

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON- and Arrow-compatible mapping."""
        return {
            "dataset_id": self.dataset_id,
            "schema_hash": self.schema_hash,
            "data_hash": self.data_hash,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "target": self.target,
            "uri": self.uri,
            "snapshot_id": self.snapshot_id,
            "modality": self.modality,
            "task": self.task,
            "schema_version": self.schema_version,
            "source": self.source.to_dict(),
            "lineage": self.lineage.to_dict(),
            "privacy": self.privacy.to_dict(),
            "findings": [finding.to_dict() for finding in self.findings],
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds a manifest from its serialized representation.

        Args:
            record: A mapping as produced by `to_dict`.

        Returns:
            The manifest.
        """
        return cls(
            dataset_id=str(record["dataset_id"]),
            schema_hash=str(record["schema_hash"]),
            data_hash=str(record["data_hash"]),
            row_count=int(record["row_count"]),
            column_count=int(record["column_count"]),
            target=str(record["target"]),
            uri=str(record.get("uri", "")),
            snapshot_id=str(record.get("snapshot_id", "")),
            modality=str(record.get("modality", "tabular")),
            task=str(record.get("task", "")),
            schema_version=int(record.get("schema_version", 1)),
            source=Origin.from_dict(record.get("source", {})),
            lineage=Ancestry.from_dict(record.get("lineage", {})),
            privacy=Confidentiality.from_dict(record.get("privacy", {})),
            findings=tuple(
                findings.Finding.from_dict(finding)
                for finding in record.get("findings", ())
            ),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Passport:
    """Immutable identity and governance metadata for any modality.

    Unlike ``DatasetManifest``, this contract does not assume rows, columns,
    or a tabular schema. It binds a modality sample collection to its source,
    transformation, label, privacy, and curation evidence instead.
    """

    dataset_id: str
    snapshot_id: str
    modality: str
    task: str
    sample_count: int
    content_hash: str
    schema_version: int = 1
    source: Origin = dataclasses.field(default_factory=Origin)
    lineage: Ancestry = dataclasses.field(default_factory=Ancestry)
    privacy: Confidentiality = dataclasses.field(
        default_factory=Confidentiality
    )
    findings: tuple[findings.Finding, ...] = ()

    def __post_init__(self) -> None:
        """Validates modality identity and attached governance metadata.

        Raises:
            TypeError: If the sample count is not an integer, a metadata
                component has the wrong type, or a finding is not a
                `Finding`.
            ValueError: If an identity field is blank, the sample count is
                negative, the schema version is below one, or the content
                hash is not 64 lowercase hexadecimal characters.
        """
        for name, value in (
            ("dataset_id", self.dataset_id),
            ("snapshot_id", self.snapshot_id),
            ("modality", self.modality),
            ("task", self.task),
            ("content_hash", self.content_hash),
        ):
            if not value.strip():
                raise errors.InputError(f"{name} must not be empty")
        if isinstance(self.sample_count, bool) or not isinstance(
            self.sample_count, int
        ):
            raise errors.KindError("sample_count must be an integer")
        if self.sample_count < 0:
            raise errors.InputError("sample_count must not be negative")
        if self.schema_version < 1:
            raise errors.InputError("schema_version must be positive")
        if len(self.content_hash) != SHA256_HEX_LENGTH or any(
            character not in "0123456789abcdef"
            for character in self.content_hash
        ):
            raise errors.InputError("content_hash must be a SHA-256 hex digest")
        if not isinstance(self.source, Origin):
            raise errors.KindError("source must be Origin")
        if not isinstance(self.lineage, Ancestry):
            raise errors.KindError("lineage must be Ancestry")
        if not isinstance(self.privacy, Confidentiality):
            raise errors.KindError("privacy must be Confidentiality")
        object.__setattr__(self, "findings", tuple(self.findings))
        findings.vet(self.findings)

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON- and Arrow-compatible modality manifest."""
        return {
            "dataset_id": self.dataset_id,
            "snapshot_id": self.snapshot_id,
            "modality": self.modality,
            "task": self.task,
            "sample_count": self.sample_count,
            "content_hash": self.content_hash,
            "schema_version": self.schema_version,
            "source": self.source.to_dict(),
            "lineage": self.lineage.to_dict(),
            "privacy": self.privacy.to_dict(),
            "findings": [finding.to_dict() for finding in self.findings],
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds a modality manifest from serialized evidence.

        Args:
            record: A mapping as produced by `to_dict`.

        Returns:
            The manifest.
        """
        return cls(
            dataset_id=str(record["dataset_id"]),
            snapshot_id=str(record["snapshot_id"]),
            modality=str(record["modality"]),
            task=str(record["task"]),
            sample_count=int(record["sample_count"]),
            content_hash=str(record["content_hash"]),
            schema_version=int(record.get("schema_version", 1)),
            source=Origin.from_dict(record.get("source", {})),
            lineage=Ancestry.from_dict(record.get("lineage", {})),
            privacy=Confidentiality.from_dict(record.get("privacy", {})),
            findings=tuple(
                findings.Finding.from_dict(finding)
                for finding in record.get("findings", ())
            ),
        )
