"""Dataset identity: origin, lineage, confidentiality, manifest and passport."""

from collections.abc import Mapping
import dataclasses
from typing import Any
from typing import ClassVar
from typing import Self

from scmle.contracts import errors
from scmle.contracts import records
from scmle.contracts.data import findings

SHA256_HEX_LENGTH = 64


@dataclasses.dataclass(frozen=True, slots=True)
class Origin:
    """Provenance and semantic context for the dataset source.

    Attributes:
        owner: Team or organization accountable for the source.
        acquisition: How the source was acquired.
        captured: Source capture timestamp, when known.
        timezone: Timezone used by source timestamps.
        sampling: Population or sampling frame represented by the data.
        units: Units by semantic field name.
        definitions: Human-readable semantic definitions by field name.
        query: Extraction query or predicate used to extract the source.
        prefix: Object-store prefix or partition selector.
        license: License or usage grant covering the source.
        access: Policy governing access to the source.
        migrations: Version steps of the stored part, which is versioned by
            the manifest that holds it.
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = (
        records.Migration(
            renames={
                "acquisition_method": "acquisition",
                "captured_at": "captured",
                "sampling_frame": "sampling",
                "semantic_definitions": "definitions",
                "extraction_query": "query",
                "object_prefix": "prefix",
                "access_policy": "access",
            }
        ),
    )

    owner: str = ""
    acquisition: str = ""
    captured: str = ""
    timezone: str = ""
    sampling: str = ""
    units: Mapping[str, str] = dataclasses.field(default_factory=dict)
    definitions: Mapping[str, str] = dataclasses.field(default_factory=dict)
    query: str = ""
    prefix: str = ""
    license: str = ""
    access: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON- and Arrow-compatible mapping."""
        return {
            "owner": self.owner,
            "acquisition": self.acquisition,
            "captured": self.captured,
            "timezone": self.timezone,
            "sampling": self.sampling,
            "units": dict(self.units),
            "definitions": dict(self.definitions),
            "query": self.query,
            "prefix": self.prefix,
            "license": self.license,
            "access": self.access,
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
            acquisition=str(record.get("acquisition", "")),
            captured=str(record.get("captured", "")),
            timezone=str(record.get("timezone", "")),
            sampling=str(record.get("sampling", "")),
            units=dict(record.get("units", {})),
            definitions=dict(record.get("definitions", {})),
            query=str(record.get("query", "")),
            prefix=str(record.get("prefix", "")),
            license=str(record.get("license", "")),
            access=str(record.get("access", "")),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Ancestry:
    """Transform and label provenance for an immutable dataset snapshot.

    Attributes:
        snapshot: Id of the source snapshot.
        parents: Ids of the parent datasets.
        transformations: Transformations applied, in order.
        labelling: Where the labels came from.
        generator: Version of the label generator.
        annotator: Version of the annotator.
        adjudication: Adjudication status of the labels.
        partitions: Ids of the partitions.
        digests: Hash of each object, by object.
        evolution: Schema evolution of the snapshot.
        migrations: Version steps of the stored part, which is versioned by
            the manifest that holds it.
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = (
        records.Migration(
            renames={
                "source_snapshot": "snapshot",
                "parent_datasets": "parents",
                "label_source": "labelling",
                "label_generator_version": "generator",
                "annotator_version": "annotator",
                "adjudication_status": "adjudication",
                "partition_ids": "partitions",
                "object_hashes": "digests",
                "schema_evolution": "evolution",
            }
        ),
    )

    snapshot: str = ""
    parents: tuple[str, ...] = ()
    transformations: tuple[str, ...] = ()
    labelling: str = ""
    generator: str = ""
    annotator: str = ""
    adjudication: str = ""
    partitions: tuple[str, ...] = ()
    digests: Mapping[str, str] = dataclasses.field(default_factory=dict)
    evolution: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON- and Arrow-compatible mapping."""
        return {
            "snapshot": self.snapshot,
            "parents": list(self.parents),
            "transformations": list(self.transformations),
            "labelling": self.labelling,
            "generator": self.generator,
            "annotator": self.annotator,
            "adjudication": self.adjudication,
            "partitions": list(self.partitions),
            "digests": dict(self.digests),
            "evolution": self.evolution,
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
            snapshot=str(record.get("snapshot", "")),
            parents=tuple(record.get("parents", ())),
            transformations=tuple(record.get("transformations", ())),
            labelling=str(record.get("labelling", "")),
            generator=str(record.get("generator", "")),
            annotator=str(record.get("annotator", "")),
            adjudication=str(record.get("adjudication", "")),
            partitions=tuple(record.get("partitions", ())),
            digests=dict(record.get("digests", {})),
            evolution=str(record.get("evolution", "")),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Confidentiality:
    """Privacy, consent, and retention controls for a dataset snapshot.

    Attributes:
        classification: Data classification level.
        identifying: Whether the data contains personally identifying
            information.
        sensitive: Names of the sensitive fields.
        consent: Consent basis for processing.
        retention: Retention period.
        deletion: Deletion policy.
        access: Access policy.
        redaction: Whether redaction is required.
        migrations: Version steps of the stored part, which is versioned by
            the manifest that holds it.
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = (
        records.Migration(
            renames={
                "contains_pii": "identifying",
                "sensitive_fields": "sensitive",
                "consent_basis": "consent",
                "retention_period": "retention",
                "deletion_policy": "deletion",
                "access_policy": "access",
                "redaction_required": "redaction",
            }
        ),
    )

    classification: str = "unclassified"
    identifying: bool = False
    sensitive: tuple[str, ...] = ()
    consent: str = ""
    retention: str = ""
    deletion: str = ""
    access: str = ""
    redaction: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON- and Arrow-compatible mapping."""
        return {
            "classification": self.classification,
            "identifying": self.identifying,
            "sensitive": list(self.sensitive),
            "consent": self.consent,
            "retention": self.retention,
            "deletion": self.deletion,
            "access": self.access,
            "redaction": self.redaction,
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
            identifying=bool(record.get("identifying", False)),
            sensitive=tuple(record.get("sensitive", ())),
            consent=str(record.get("consent", "")),
            retention=str(record.get("retention", "")),
            deletion=str(record.get("deletion", "")),
            access=str(record.get("access", "")),
            redaction=bool(record.get("redaction", False)),
        )


PARTS = {
    "source": Origin.migrations[0],
    "lineage": Ancestry.migrations[0],
    "privacy": Confidentiality.migrations[0],
    "findings": findings.Finding.migrations[0],
}
VERSION = records.ORIGIN + 1


@dataclasses.dataclass(frozen=True, slots=True)
class DatasetManifest:
    """Immutable identity of an ingested dataset.

    Attributes:
        identifier: Short identifier derived from the content hash.
        digests: Hashes by the hashed thing: ``schema`` (column names and
            dtypes) and ``data`` (the row contents).
        rows: Number of rows.
        columns: Number of columns.
        target: Name of the target column.
        uri: Where the data was read from.
        snapshot: Immutable identity of the source snapshot.
        modality: Dataset modality, such as `tabular` or `image`.
        task: Intended learning task, when known.
        schema_version: Version of this manifest contract.
        source: Source ownership, acquisition, and semantic metadata.
        lineage: Source, transformation, and label lineage metadata.
        privacy: Privacy, consent, retention, and access metadata.
        findings: Structured findings attached to this snapshot.
        migrations: Version steps of the stored manifest and its parts.
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = (
        records.nest(
            renames={
                "dataset_id": "identifier",
                "row_count": "rows",
                "column_count": "columns",
                "snapshot_id": "snapshot",
            },
            parts=PARTS,
            groups={"digests": {"schema_hash": "schema", "data_hash": "data"}},
        ),
    )

    identifier: str
    digests: Mapping[str, str]
    rows: int
    columns: int
    target: str
    uri: str = ""
    snapshot: str = ""
    modality: str = "tabular"
    task: str = ""
    schema_version: int = VERSION
    source: Origin = dataclasses.field(default_factory=Origin)
    lineage: Ancestry = dataclasses.field(default_factory=Ancestry)
    privacy: Confidentiality = dataclasses.field(
        default_factory=Confidentiality
    )
    # Quoted: the field shadows the `findings` module in the class body,
    # and Python before 3.14 evaluates the annotation after the default.
    findings: "tuple[findings.Finding, ...]" = ()

    def __post_init__(self) -> None:
        """Validates dataset identity before it enters a run or artifact.

        Raises:
            TypeError: If a count or the schema version is not an integer,
                a metadata component has the wrong type, or a finding is not
                a `Finding`.
            ValueError: If an identity field (dataset id, schema and data
                digests, target or modality) is blank or not a string, a
                count is negative, or the schema version is below one.
        """
        object.__setattr__(self, "digests", dict(self.digests))
        text_fields: tuple[tuple[str, object], ...] = (
            ("identifier", self.identifier),
            ("digests.schema", self.digests.get("schema")),
            ("digests.data", self.digests.get("data")),
            ("target", self.target),
            ("modality", self.modality),
        )
        for field_name, field_value in text_fields:
            if not isinstance(field_value, str) or not field_value.strip():
                raise errors.InputError(f"{field_name} must not be empty")
        for count_name, count_value in (
            ("rows", self.rows),
            ("columns", self.columns),
            ("schema_version", self.schema_version),
        ):
            if isinstance(count_value, bool) or not isinstance(
                count_value, int
            ):
                raise errors.KindError(f"{count_name} must be an integer")
        if self.rows < 0:
            raise errors.InputError("rows must not be negative")
        if self.columns < 0:
            raise errors.InputError("columns must not be negative")
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
            "identifier": self.identifier,
            "digests": dict(sorted(self.digests.items())),
            "rows": self.rows,
            "columns": self.columns,
            "target": self.target,
            "uri": self.uri,
            "snapshot": self.snapshot,
            "modality": self.modality,
            "task": self.task,
            "schema_version": self.schema_version,
            "source": self.source.to_dict(),
            "lineage": self.lineage.to_dict(),
            "privacy": self.privacy.to_dict(),
            "findings": [finding.to_dict() for finding in self.findings],
        }

    def anchor(self) -> records.Layout:
        """Returns the manifest in its version-1 layout, for identity digests.

        A digest that names the manifest across runs is computed on this
        layout, so renaming a stored key never changes it.
        """
        record = self.to_dict()
        del record[records.KEY]
        record = records.downgrade(self.migrations, record)
        return {**record, records.KEY: records.ORIGIN}

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds a manifest from its serialized representation.

        A manifest stored at an older schema version is migrated first,
        parts included.

        Args:
            record: A mapping as produced by `to_dict`, at any supported
                schema version.

        Returns:
            The manifest.
        """
        record = records.load(cls.__name__, cls.migrations, record)
        return cls(
            identifier=str(record["identifier"]),
            digests={
                str(key): str(value)
                for key, value in dict(record["digests"]).items()
            },
            rows=int(record["rows"]),
            columns=int(record["columns"]),
            target=str(record["target"]),
            uri=str(record.get("uri", "")),
            snapshot=str(record.get("snapshot", "")),
            modality=str(record.get("modality", "tabular")),
            task=str(record.get("task", "")),
            schema_version=int(record[records.KEY]),
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

    Attributes:
        identifier: Id of the dataset.
        snapshot: Immutable identity of the source snapshot.
        modality: Dataset modality.
        task: Intended learning task.
        samples: Number of samples.
        digests: Hashes by the hashed thing; ``content`` is the SHA-256 of
            the sample collection.
        schema_version: Version of this manifest contract.
        source: Source ownership, acquisition, and semantic metadata.
        lineage: Source, transformation, and label lineage metadata.
        privacy: Privacy, consent, retention, and access metadata.
        findings: Structured findings attached to this snapshot.
        migrations: Version steps of the stored passport and its parts.
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = (
        records.nest(
            renames={
                "dataset_id": "identifier",
                "snapshot_id": "snapshot",
                "sample_count": "samples",
            },
            parts=PARTS,
            groups={"digests": {"content_hash": "content"}},
        ),
    )

    identifier: str
    snapshot: str
    modality: str
    task: str
    samples: int
    digests: Mapping[str, str]
    schema_version: int = VERSION
    source: Origin = dataclasses.field(default_factory=Origin)
    lineage: Ancestry = dataclasses.field(default_factory=Ancestry)
    privacy: Confidentiality = dataclasses.field(
        default_factory=Confidentiality
    )
    # Quoted: the field shadows the `findings` module in the class body,
    # and Python before 3.14 evaluates the annotation after the default.
    findings: "tuple[findings.Finding, ...]" = ()

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
        object.__setattr__(self, "digests", dict(self.digests))
        content = self.digests.get("content", "")
        for name, value in (
            ("identifier", self.identifier),
            ("snapshot", self.snapshot),
            ("modality", self.modality),
            ("task", self.task),
            ("digests.content", content),
        ):
            if not value.strip():
                raise errors.InputError(f"{name} must not be empty")
        if isinstance(self.samples, bool) or not isinstance(self.samples, int):
            raise errors.KindError("samples must be an integer")
        if self.samples < 0:
            raise errors.InputError("samples must not be negative")
        if self.schema_version < 1:
            raise errors.InputError("schema_version must be positive")
        if len(content) != SHA256_HEX_LENGTH or any(
            character not in "0123456789abcdef" for character in content
        ):
            raise errors.InputError(
                "digests.content must be a SHA-256 hex digest"
            )
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
            "identifier": self.identifier,
            "snapshot": self.snapshot,
            "modality": self.modality,
            "task": self.task,
            "samples": self.samples,
            "digests": dict(sorted(self.digests.items())),
            "schema_version": self.schema_version,
            "source": self.source.to_dict(),
            "lineage": self.lineage.to_dict(),
            "privacy": self.privacy.to_dict(),
            "findings": [finding.to_dict() for finding in self.findings],
        }

    def anchor(self) -> records.Layout:
        """Returns the manifest in its version-1 layout, for identity digests.

        A digest that names the manifest across runs is computed on this
        layout, so renaming a stored key never changes it.
        """
        record = self.to_dict()
        del record[records.KEY]
        record = records.downgrade(self.migrations, record)
        return {**record, records.KEY: records.ORIGIN}

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds a modality manifest from serialized evidence.

        A passport stored at an older schema version is migrated first,
        parts included.

        Args:
            record: A mapping as produced by `to_dict`, at any supported
                schema version.

        Returns:
            The manifest.
        """
        record = records.load(cls.__name__, cls.migrations, record)
        return cls(
            identifier=str(record["identifier"]),
            snapshot=str(record["snapshot"]),
            modality=str(record["modality"]),
            task=str(record["task"]),
            samples=int(record["samples"]),
            digests={
                str(key): str(value)
                for key, value in dict(record["digests"]).items()
            },
            schema_version=int(record[records.KEY]),
            source=Origin.from_dict(record.get("source", {})),
            lineage=Ancestry.from_dict(record.get("lineage", {})),
            privacy=Confidentiality.from_dict(record.get("privacy", {})),
            findings=tuple(
                findings.Finding.from_dict(finding)
                for finding in record.get("findings", ())
            ),
        )
