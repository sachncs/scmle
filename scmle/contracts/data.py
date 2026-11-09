"""Dataset identity, feature plan and profile contracts."""

from collections.abc import Mapping
import dataclasses
import enum
import hashlib
import math
from typing import Any
from typing import Self

from scmle.contracts import events
from scmle.contracts import explanations

HIGH_CARDINALITY_LEVELS = 20
SHA256_HEX_LENGTH = 64


def normalize_quality_evidence(
    evidence: Mapping[str, Any], field_name: str = "quality_evidence"
) -> dict[str, object]:
    """Validates and normalizes one JSON-native quality evidence record.

    A floating-point value that is not finite is refused with a
    `ValueError` by the JSON validation.

    Args:
        evidence: Mapping containing only JSON-native values.
        field_name: Path used in validation errors.

    Returns:
        A detached dictionary containing JSON-native values.

    Raises:
        TypeError: If a key or value is not JSON-native, or the record does
            not serialize to an object.
    """
    normalized = explanations.validate_json_value(evidence, field_name)
    if not isinstance(normalized, dict):
        raise TypeError(f"{field_name} must serialize to an object")
    return normalized


@dataclasses.dataclass(frozen=True, slots=True)
class QualityObservation:
    """One quality check result and its reproducible evidence.

    Attributes:
        note: Human-readable finding text, if the check raised one.
        evidence: JSON-native method, threshold, sampling, metric, and
            limitation details for the check.
    """

    note: str | None
    evidence: dict[str, object]

    def __post_init__(self) -> None:
        """Validates and detaches the evidence from mutable caller state.

        The evidence is replaced by a normalized copy; invalid evidence is
        rejected as described for `normalize_quality_evidence`.

        Raises:
            ValueError: If the note is given but blank.
        """
        if self.note is not None and not self.note.strip():
            raise ValueError("note must not be empty")
        object.__setattr__(
            self,
            "evidence",
            normalize_quality_evidence(self.evidence, "evidence"),
        )


class FindingSeverity(enum.StrEnum):
    """Severity assigned to a curation finding."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class FindingDisposition(enum.StrEnum):
    """Review state assigned to a curation finding."""

    OPEN = "open"
    ACCEPTED = "accepted"
    RESOLVED = "resolved"
    WAIVED = "waived"
    FALSE_POSITIVE = "false_positive"


@dataclasses.dataclass(frozen=True, slots=True)
class CurationFinding:
    """One reviewable, reproducible observation about a dataset.

    Attributes:
        finding_id: Stable identifier for the rule finding.
        rule_version: Version of the rule that produced the finding.
        severity: Operational impact if the finding is not addressed.
        confidence: Confidence in the finding, from zero through one.
        affected_scope: Columns, rows, partitions, or objects affected.
        evidence: Reproducible measurements or references supporting it.
        remediation: Action required to resolve or assess the finding.
        owner: Person or team responsible for the finding.
        reviewer: Person who reviewed it, if review has happened.
        disposition: Current review decision.
        blocking: Whether the finding blocks training or release.
        timestamp: UTC timestamp when the finding was observed.
    """

    finding_id: str
    rule_version: str
    severity: FindingSeverity
    confidence: float
    affected_scope: Mapping[str, Any]
    evidence: Mapping[str, Any]
    remediation: str
    owner: str
    reviewer: str | None
    disposition: FindingDisposition
    blocking: bool
    timestamp: str = dataclasses.field(default_factory=events.utc_timestamp)

    def __post_init__(self) -> None:
        """Validates fields that determine finding identity and decisions.

        A severity or disposition given as a string is converted to its enum.

        Raises:
            ValueError: If the id, rule version, remediation, owner or timestamp
                is blank, the confidence is not a finite number in [0, 1], or
                the severity or disposition is not a valid value.
        """
        if not self.finding_id.strip():
            raise ValueError("finding_id must not be empty")
        if not self.rule_version.strip():
            raise ValueError("rule_version must not be empty")
        if not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError(
                "confidence must be finite and between zero and one"
            )
        if not self.remediation.strip():
            raise ValueError("remediation must not be empty")
        if not self.owner.strip():
            raise ValueError("owner must not be empty")
        if not self.timestamp.strip():
            raise ValueError("timestamp must not be empty")
        severity: Any = self.severity  # may be a raw value at runtime
        if not isinstance(severity, FindingSeverity):
            object.__setattr__(self, "severity", FindingSeverity(severity))
        disposition: Any = self.disposition  # may be a raw value at runtime
        if not isinstance(disposition, FindingDisposition):
            object.__setattr__(
                self, "disposition", FindingDisposition(disposition)
            )

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON- and Arrow-compatible mapping."""
        return {
            "id": self.finding_id,
            "rule_version": self.rule_version,
            "severity": self.severity.value,
            "confidence": self.confidence,
            "affected_scope": dict(self.affected_scope),
            "evidence": dict(self.evidence),
            "remediation": self.remediation,
            "owner": self.owner,
            "reviewer": self.reviewer,
            "disposition": self.disposition.value,
            "blocking": self.blocking,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_observation(
        cls,
        message: str,
        category: str,
        severity: FindingSeverity = FindingSeverity.WARNING,
    ) -> Self:
        """Creates a structured finding from a pipeline observation.

        The profiling layer may produce a human-readable observation while a
        rule is being developed. This constructor gives that observation a
        stable identity and evidence envelope before it is persisted. It is a
        canonical contract conversion, not a second finding representation.

        Args:
            message: Human-readable observation emitted by a profiling rule.
            category: Stable category such as ``quality`` or ``risk``.
            severity: Operational severity assigned by the caller.

        Returns:
            A reviewable finding with a deterministic ID.

        Raises:
            ValueError: If the message or category is empty.
        """
        observation = message.strip()
        finding_category = category.strip().lower()
        if not observation:
            raise ValueError("message must not be empty")
        if not finding_category:
            raise ValueError("category must not be empty")
        digest = hashlib.sha256(observation.encode("utf-8")).hexdigest()
        return cls(
            finding_id=f"{finding_category}.{digest[:16]}",
            rule_version="pipeline-observation.v1",
            severity=severity,
            confidence=1.0,
            affected_scope={"category": finding_category},
            evidence={"message": observation},
            remediation="Review the observation and record a disposition.",
            owner="pipeline",
            reviewer=None,
            disposition=FindingDisposition.OPEN,
            blocking=False,
        )

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds a finding from its serialized representation.

        A missing required key raises `KeyError`, and an invalid value is
        rejected by the constructor. Only `reviewer` may be absent.

        Args:
            record: A mapping as produced by `to_dict`.

        Returns:
            The finding.
        """
        return cls(
            finding_id=str(record["id"]),
            rule_version=str(record["rule_version"]),
            severity=FindingSeverity(record["severity"]),
            confidence=float(record["confidence"]),
            affected_scope=dict(record["affected_scope"]),
            evidence=dict(record["evidence"]),
            remediation=str(record["remediation"]),
            owner=str(record["owner"]),
            reviewer=record.get("reviewer"),
            disposition=FindingDisposition(record["disposition"]),
            blocking=bool(record["blocking"]),
            timestamp=str(record["timestamp"]),
        )


def check_findings(findings: tuple[Any, ...]) -> None:
    """Checks that every entry of a record's findings is a finding.

    Args:
        findings: The findings of a manifest.

    Raises:
        TypeError: If an entry is not a `CurationFinding`.
    """
    if any(not isinstance(finding, CurationFinding) for finding in findings):
        raise TypeError("findings must contain CurationFinding records")


@dataclasses.dataclass(frozen=True, slots=True)
class SourceMetadata:
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
class LineageMetadata:
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
class PrivacyMetadata:
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
    source: SourceMetadata = dataclasses.field(default_factory=SourceMetadata)
    lineage: LineageMetadata = dataclasses.field(
        default_factory=LineageMetadata
    )
    privacy: PrivacyMetadata = dataclasses.field(
        default_factory=PrivacyMetadata
    )
    findings: tuple[CurationFinding, ...] = ()

    def __post_init__(self) -> None:
        """Validates dataset identity before it enters a run or artifact.

        Raises:
            TypeError: If a count or the schema version is not an integer,
                a metadata component has the wrong type, or a finding is not
                a `CurationFinding`.
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
                raise ValueError(f"{field_name} must not be empty")
        for count_name, count_value in (
            ("row_count", self.row_count),
            ("column_count", self.column_count),
            ("schema_version", self.schema_version),
        ):
            if isinstance(count_value, bool) or not isinstance(
                count_value, int
            ):
                raise TypeError(f"{count_name} must be an integer")
        if self.row_count < 0:
            raise ValueError("row_count must not be negative")
        if self.column_count < 0:
            raise ValueError("column_count must not be negative")
        if self.schema_version < 1:
            raise ValueError("schema_version must be positive")
        for component_name, component, expected in (
            ("source", self.source, SourceMetadata),
            ("lineage", self.lineage, LineageMetadata),
            ("privacy", self.privacy, PrivacyMetadata),
        ):
            if not isinstance(component, expected):
                raise TypeError(f"{component_name} must be {expected.__name__}")
        object.__setattr__(self, "findings", tuple(self.findings))
        check_findings(self.findings)

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
            source=SourceMetadata.from_dict(record.get("source", {})),
            lineage=LineageMetadata.from_dict(record.get("lineage", {})),
            privacy=PrivacyMetadata.from_dict(record.get("privacy", {})),
            findings=tuple(
                CurationFinding.from_dict(finding)
                for finding in record.get("findings", ())
            ),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class ModalityManifest:
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
    source: SourceMetadata = dataclasses.field(default_factory=SourceMetadata)
    lineage: LineageMetadata = dataclasses.field(
        default_factory=LineageMetadata
    )
    privacy: PrivacyMetadata = dataclasses.field(
        default_factory=PrivacyMetadata
    )
    findings: tuple[CurationFinding, ...] = ()

    def __post_init__(self) -> None:
        """Validates modality identity and attached governance metadata.

        Raises:
            TypeError: If the sample count is not an integer, a metadata
                component has the wrong type, or a finding is not a
                `CurationFinding`.
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
                raise ValueError(f"{name} must not be empty")
        if isinstance(self.sample_count, bool) or not isinstance(
            self.sample_count, int
        ):
            raise TypeError("sample_count must be an integer")
        if self.sample_count < 0:
            raise ValueError("sample_count must not be negative")
        if self.schema_version < 1:
            raise ValueError("schema_version must be positive")
        if len(self.content_hash) != SHA256_HEX_LENGTH or any(
            character not in "0123456789abcdef"
            for character in self.content_hash
        ):
            raise ValueError("content_hash must be a SHA-256 hex digest")
        if not isinstance(self.source, SourceMetadata):
            raise TypeError("source must be SourceMetadata")
        if not isinstance(self.lineage, LineageMetadata):
            raise TypeError("lineage must be LineageMetadata")
        if not isinstance(self.privacy, PrivacyMetadata):
            raise TypeError("privacy must be PrivacyMetadata")
        object.__setattr__(self, "findings", tuple(self.findings))
        check_findings(self.findings)

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
            source=SourceMetadata.from_dict(record.get("source", {})),
            lineage=LineageMetadata.from_dict(record.get("lineage", {})),
            privacy=PrivacyMetadata.from_dict(record.get("privacy", {})),
            findings=tuple(
                CurationFinding.from_dict(finding)
                for finding in record.get("findings", ())
            ),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class FeaturePlan:
    """Which columns become features and why the others were dropped.

    Attributes:
        numeric: Numeric feature columns.
        categorical: Categorical feature columns.
        text: Free-text columns (encoded as TF-IDF terms; also listed in
            `extra_inputs` so inference requires them).
        dropped: Mapping of dropped column to the reason.
        derived: Row-wise feature-graph nodes computed inside the pipeline.
        series: Ordered (time-series) feature-graph nodes.
        extra_inputs: Raw columns only the feature graph reads.
        monotonic: Numeric feature to +1 / -1 monotonic constraint.
        high_cardinality: Categorical features with many distinct levels
            (`HIGH_CARDINALITY_LEVELS` or more), which one-hot encoding
            handles poorly.
        group_column: Entity column, so inner cross-validation keeps entities
            together.
        time_column: Ordering column, so inner cross-validation keeps time
            order.
        weight_column: Column of per-row training weights, if any.
        event_column: Survival event indicator column, if any.
        ordinal_levels: Number of ordered levels of an ordinal target.
        query_column: Ranking query identifier column, if any.
        target_columns: Every target column of a multi-target task.
        value_column: The series value column of a forecasting task.
        treatment_column: Uplift treatment indicator column, if any.
        binary_outcome: Whether an uplift outcome is a 0/1 label.
        text_encoding: `tfidf` or `embedding` for free-text columns.
        embedding_model: Sentence-embedding model for `embedding`.
        informative_missing: Features whose gaps are associated with the
            target; `None` when missingness was not assessed.
    """

    numeric: tuple[str, ...]
    categorical: tuple[str, ...]
    dropped: Mapping[str, str] = dataclasses.field(default_factory=dict)
    text: tuple[str, ...] = ()
    monotonic: Mapping[str, int] = dataclasses.field(default_factory=dict)
    derived: tuple[Any, ...] = ()
    series: tuple[Any, ...] = ()
    extra_inputs: tuple[str, ...] = ()
    high_cardinality: tuple[str, ...] = ()
    group_column: str | None = None
    time_column: str | None = None
    weight_column: str | None = None
    event_column: str | None = None
    ordinal_levels: int | None = None
    query_column: str | None = None
    target_columns: tuple[str, ...] = ()
    value_column: str | None = None
    treatment_column: str | None = None
    binary_outcome: bool = True
    text_encoding: str = "tfidf"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    informative_missing: tuple[str, ...] | None = None

    @property
    def features(self) -> tuple[str, ...]:
        """All feature columns the model consumes, numeric first."""
        return self.numeric + self.categorical

    @property
    def derived_names(self) -> frozenset[str]:
        """Columns computed by the feature graph (row-wise or ordered)."""
        return frozenset(n.name for n in (*self.derived, *self.series))

    @property
    def input_columns(self) -> tuple[str, ...]:
        """Raw columns the fitted pipeline reads (plus series features).

        These exist in the frame a pipeline is fitted on: model features that
        are not computed inside the pipeline, graph-only inputs, and the
        ordered (series) features computed at ingestion.
        """
        rowwise = frozenset(n.name for n in self.derived)
        columns = [c for c in self.features if c not in rowwise]
        return tuple(dict.fromkeys([*columns, *self.extra_inputs]))

    def columns_for(self, kind_name: str) -> tuple[str, ...]:
        """Returns the feature columns of one kind.

        The name is looked up with `getattr`, so any attribute name returns
        that attribute's value unchanged.

        Args:
            kind_name: Name of a column kind such as `numeric` or
                `categorical`.

        Returns:
            The attribute stored under that name, or an empty tuple if there
            is no such attribute.
        """
        return getattr(self, kind_name, ())

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-serializable mapping."""
        return {
            "numeric": list(self.numeric),
            "categorical": list(self.categorical),
            "text": list(self.text),
            "dropped": dict(self.dropped),
            "derived": [n.model_dump(mode="json") for n in self.derived],
            "series": [n.model_dump(mode="json") for n in self.series],
            "extra_inputs": list(self.extra_inputs),
            "high_cardinality": list(self.high_cardinality),
        }


@dataclasses.dataclass(frozen=True, slots=True)
class DatasetProfile:
    """Statistical profile and findings for a dataset.

    Attributes:
        manifest: Identity of the profiled dataset.
        column_statistics: Per-column statistics.
        quality_findings: Data-quality observations.
        risk_findings: Leakage / imbalance / shift hypotheses (not verdicts).
        target_summary: Target distribution summary.
        feature_plan: Resulting feature selection.
        dataset_statistics: Dataset-level fingerprint grouped by dimension
            (shape, types, missingness, target, features, relationships,
            structure, quality, operational, risk).
        missingness: Per-column tests of whether gaps relate to the target.
        quality_evidence: JSON-native records for reproducible quality checks.
        leakage: The leakage screen (`screen.Screen.to_dict`): its level,
            family size, suspects and the suspects that may block a run. Empty
            when the screen did not apply.
    """

    manifest: DatasetManifest
    column_statistics: Mapping[str, Any]
    quality_findings: tuple[str, ...]
    risk_findings: tuple[str, ...]
    target_summary: Mapping[str, Any]
    feature_plan: FeaturePlan
    dataset_statistics: Mapping[str, Any] = dataclasses.field(
        default_factory=dict
    )
    missingness: tuple[Mapping[str, Any], ...] = ()
    quality_evidence: tuple[dict[str, object], ...] = ()
    leakage: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def blocking_leaks(self) -> tuple[str, ...]:
        """Columns whose leakage suspicion is certified and may block a run."""
        return tuple(self.leakage.get("blocking", ()))

    @property
    def unchecked_columns(self) -> tuple[str, ...]:
        """Columns the leakage screen declined to test (not clean)."""
        return tuple(map(str, self.leakage.get("unchecked", ())))

    @property
    def screen_warning(self) -> str:
        """The loud statement of an incomplete screen, or an empty string."""
        return str(self.leakage.get("warning", ""))

    def __post_init__(self) -> None:
        """Validates and detaches structured quality evidence."""
        normalized = tuple(
            normalize_quality_evidence(item, f"quality_evidence[{index}]")
            for index, item in enumerate(self.quality_evidence)
        )
        object.__setattr__(self, "quality_evidence", normalized)

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-serializable mapping."""
        return {
            "manifest": self.manifest.to_dict(),
            "columns": dict(self.column_statistics),
            "quality_findings": list(self.quality_findings),
            "risk_findings": list(self.risk_findings),
            "target": dict(self.target_summary),
            "feature_plan": self.feature_plan.to_dict(),
            "dataset": dict(self.dataset_statistics),
            "missingness": [dict(item) for item in self.missingness],
            "quality_evidence": [dict(item) for item in self.quality_evidence],
            "leakage": dict(self.leakage),
        }


@dataclasses.dataclass(frozen=True, slots=True)
class Curation:
    """Auditable record of what happened before modelling.

    This report deliberately contains no model scores. It describes the
    source observed at ingestion, the rows used for profiling, repairs made,
    quality and risk findings, and the resulting feature decisions.
    """

    manifest: DatasetManifest
    source_rows: int
    profiled_rows: int
    read_seconds: float
    streamed: bool
    cleaning_notes: tuple[str, ...]
    quality_findings: tuple[CurationFinding, ...]
    risk_findings: tuple[CurationFinding, ...]
    feature_columns: Mapping[str, tuple[str, ...]]
    dropped_columns: Mapping[str, str]
    schema_version: int = 1

    @property
    def unresolved_blocking_findings(self) -> tuple[CurationFinding, ...]:
        """The blocking findings without an explicit safe disposition."""
        safe_dispositions = {
            FindingDisposition.RESOLVED,
            FindingDisposition.WAIVED,
            FindingDisposition.FALSE_POSITIVE,
        }
        return tuple(
            finding
            for finding in (*self.quality_findings, *self.risk_findings)
            if finding.blocking and finding.disposition not in safe_dispositions
        )

    @property
    def release_ready(self) -> bool:
        """Whether curation has no unresolved blocking findings."""
        return not self.unresolved_blocking_findings

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-serializable curation record."""
        return {
            "schema_version": self.schema_version,
            "manifest": self.manifest.to_dict(),
            "source_rows": self.source_rows,
            "profiled_rows": self.profiled_rows,
            "read_seconds": self.read_seconds,
            "streamed": self.streamed,
            "cleaning_notes": list(self.cleaning_notes),
            "quality_findings": [
                finding.to_dict() for finding in self.quality_findings
            ],
            "risk_findings": [
                finding.to_dict() for finding in self.risk_findings
            ],
            "feature_columns": {
                name: list(columns)
                for name, columns in self.feature_columns.items()
            },
            "dropped_columns": dict(self.dropped_columns),
            "release_ready": self.release_ready,
            "unresolved_blocking_findings": [
                finding.finding_id
                for finding in self.unresolved_blocking_findings
            ],
        }


@dataclasses.dataclass(frozen=True, slots=True)
class DatasetCard:
    """Human-reviewable summary of a dataset snapshot before modelling.

    A card is generated from the curation report, but it is not a release
    approval. Its explicit intended use, exclusions, limitations, privacy
    context, and open findings are the minimum context a reviewer needs before
    trusting a model result derived from the snapshot.
    """

    dataset_id: str
    snapshot_id: str
    modality: str
    task: str
    target: str
    intended_use: str
    out_of_scope: tuple[str, ...]
    composition: Mapping[str, int]
    feature_summary: Mapping[str, tuple[str, ...]]
    limitations: tuple[str, ...]
    quality_findings: tuple[CurationFinding, ...]
    risk_findings: tuple[CurationFinding, ...]
    source: SourceMetadata
    lineage: LineageMetadata
    privacy: PrivacyMetadata
    owner: str
    review_status: str = "machine_generated_unreviewed"
    schema_version: int = 1
    generated_at: str = dataclasses.field(default_factory=events.utc_timestamp)

    def __post_init__(self) -> None:
        """Validates the fields that make the card useful to reviewers.

        Raises:
            ValueError: If a required text field (dataset id, modality,
                intended use, owner, review status or generation time) is
                blank, the schema version is below one, a composition count
                is negative, or the out-of-scope or limitations list is
                empty.
        """
        for name, value in (
            ("dataset_id", self.dataset_id),
            ("modality", self.modality),
            ("intended_use", self.intended_use),
            ("owner", self.owner),
            ("review_status", self.review_status),
            ("generated_at", self.generated_at),
        ):
            if not value.strip():
                raise ValueError(f"{name} must not be empty")
        if self.schema_version < 1:
            raise ValueError("schema_version must be positive")
        if any(value < 0 for value in self.composition.values()):
            raise ValueError("composition counts must not be negative")
        if not self.out_of_scope:
            raise ValueError("out_of_scope must not be empty")
        if not self.limitations:
            raise ValueError("limitations must not be empty")

    @classmethod
    def from_curation(cls, report: Curation) -> Self:
        """Builds a card from the immutable curation evidence.

        Args:
            report: The curation report.

        Returns:
            The card, with limitations that include the report's cleaning notes.
        """
        manifest = report.manifest
        task = manifest.task or "the declared learning task"
        limitations = [
            "This machine-generated card requires human review before release.",
            "Curation findings are observations, not proof of data quality.",
        ]
        limitations.extend(report.cleaning_notes)
        if not report.risk_findings:
            limitations.append("No risk finding was emitted by the profiler.")
        return cls(
            dataset_id=manifest.dataset_id,
            snapshot_id=manifest.snapshot_id,
            modality=manifest.modality,
            task=manifest.task,
            target=manifest.target,
            intended_use=(
                f"Training and evaluation for {task} using the declared "
                "target and feature decisions."
            ),
            out_of_scope=(
                "It is not evidence of production-population coverage.",
                "It is not a release approval or a causal claim.",
                (
                    "It must not be used outside the declared task and "
                    "privacy policy."
                ),
            ),
            composition={
                "source_rows": report.source_rows,
                "profiled_rows": report.profiled_rows,
                "feature_columns": sum(
                    len(columns) for columns in report.feature_columns.values()
                ),
                "dropped_columns": len(report.dropped_columns),
            },
            feature_summary=report.feature_columns,
            limitations=tuple(limitations),
            quality_findings=report.quality_findings,
            risk_findings=report.risk_findings,
            source=manifest.source,
            lineage=manifest.lineage,
            privacy=manifest.privacy,
            owner=manifest.source.owner or "unassigned",
        )

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-serializable dataset card."""
        return {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "snapshot_id": self.snapshot_id,
            "modality": self.modality,
            "task": self.task,
            "target": self.target,
            "intended_use": self.intended_use,
            "out_of_scope": list(self.out_of_scope),
            "composition": dict(self.composition),
            "feature_summary": {
                name: list(columns)
                for name, columns in self.feature_summary.items()
            },
            "limitations": list(self.limitations),
            "quality_findings": [
                finding.to_dict() for finding in self.quality_findings
            ],
            "risk_findings": [
                finding.to_dict() for finding in self.risk_findings
            ],
            "source": self.source.to_dict(),
            "lineage": self.lineage.to_dict(),
            "privacy": self.privacy.to_dict(),
            "owner": self.owner,
            "review_status": self.review_status,
            "generated_at": self.generated_at,
        }


@dataclasses.dataclass(frozen=True, slots=True)
class Preprocessing:
    """Switches that shape the preprocessing transformers.

    Attributes:
        scale: Standardize numeric features.
        log_skewed: Log-transform strongly right-skewed numeric features.
        interactions: Add pairwise numeric interaction features.
        impute: Impute missing numeric values (off for models that handle
            missing values natively).
        high_cardinality: How high-cardinality categorical features are
            encoded: `onehot`, `target` (regularized, cross-fitted target
            encoding) or `ordinal` (integer codes for native categorical
            handling).
        sparse: Emit sparse one-hot output (for sparse-capable models).
        construct: Append differences and ratios of the numeric features
            most associated with the target.
        select: Keep only the columns an L1-penalised linear model uses.
        missing: How missing numeric values are filled: `median`,
            `indicator` (median plus a flag column per feature that has
            missing values) or `iterative` (each feature predicted from the
            others).
        text_encoding: `tfidf` or `embedding` for free-text columns.
        embedding_model: Sentence-embedding model for `embedding`.
    """

    scale: bool = False
    log_skewed: bool = False
    interactions: bool = False
    impute: bool = True
    high_cardinality: str = "onehot"
    sparse: bool = False
    construct: bool = False
    select: bool = False
    missing: str = "median"
    text_encoding: str = "tfidf"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
