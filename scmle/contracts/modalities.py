"""Evidence contracts for modality lifecycle support decisions.

The contract separates a modality's existence from its support claim. A
modality can be catalogued while it is experimental or incomplete, but the
registry never treats an incomplete declaration as usable support.
"""

from collections.abc import Collection
from collections.abc import Iterable
from collections.abc import Mapping
import dataclasses
import enum
import json
import re
from typing import NoReturn

from scmle.contracts import capabilities
from scmle.contracts import errors


class Modality(enum.StrEnum):
    """Dataset modality named by a lifecycle support declaration."""

    TABULAR = "tabular"
    TEXT = "text"
    DOCUMENT = "document"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    GRAPH = "graph"
    GEOSPATIAL = "geospatial"
    RASTER = "raster"


@dataclasses.dataclass(frozen=True, slots=True, order=True)
class ExtensionModality:
    """Stable identity for a caller-defined dataset family.

    Extension identities let a dataset family enter the same lifecycle
    contract without pretending that the core package understands its raw
    representation. They are deliberately namespaced and can never be
    declared as production-supported by the built-in registry.
    """

    name: str

    def __post_init__(self) -> None:
        """Validates a stable, non-built-in dataset-family name.

        Raises:
            TypeError: If the name is not a string.
            ValueError: If the name is not lowercase namespaced text such as
                ``acme/point-cloud`` or conflicts with a built-in modality.
        """
        if not isinstance(self.name, str):
            raise TypeError("extension modality name must be a string")
        if not re.fullmatch(r"[a-z0-9][a-z0-9./-]*[a-z0-9]", self.name):
            raise ValueError(
                "extension modality name must use lowercase namespaced "
                "characters such as 'acme/point-cloud'"
            )
        if self.name in {item.value for item in Modality}:
            raise ValueError(
                "extension modality name conflicts with a built-in"
            )

    @property
    def value(self) -> str:
        """The serialized identity used in evidence artifacts."""
        return f"extension:{self.name}"


ModalityIdentity = Modality | ExtensionModality


def modality_value(modality: ModalityIdentity) -> str:
    """Returns the stable serialized value for a modality identity.

    Args:
        modality: A built-in or extension modality.

    Returns:
        The built-in modality's value, or ``extension:`` followed by the
        extension name.

    Raises:
        TypeError: If `modality` is not a `Modality` or `ExtensionModality`.
    """
    if not isinstance(modality, (Modality, ExtensionModality)):
        raise TypeError("modality must be Modality or ExtensionModality")
    return modality.value


def modality_from_value(value: str) -> ModalityIdentity:
    """Restore a built-in or extension modality from evidence text.

    Args:
        value: The serialized value, as returned by `modality_value`.

    Returns:
        The matching built-in or extension modality.

    Raises:
        ValueError: If the value is not a non-empty string, or is neither a
            built-in value nor an ``extension:`` name.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError("modality value must be a non-empty string")
    try:
        return Modality(value)
    except ValueError:
        prefix = "extension:"
        if not value.startswith(prefix):
            raise ValueError(f"unknown modality value: {value}") from None
        return ExtensionModality(value.removeprefix(prefix))


class LifecycleStage(enum.StrEnum):
    """Stage that a modality must support before it is production eligible."""

    INGESTION = "ingestion"
    VALIDATION = "validation"
    PROFILING = "profiling"
    CURATION = "curation"
    SPLITTING = "splitting"
    FEATURE_PREPARATION = "feature_preparation"
    MODELLING = "modelling"
    EVALUATION = "evaluation"
    DATA_EXPLAINABILITY = "data_explainability"
    MODEL_EXPLAINABILITY = "model_explainability"
    PACKAGING = "packaging"
    INFERENCE = "inference"
    MONITORING = "monitoring"
    DOCUMENTATION = "documentation"
    RECOVERY_AND_RESOURCE_TESTS = "recovery_and_resource_tests"


REQUIRED_LIFECYCLE_STAGES = tuple(LifecycleStage)


class ModalityStatus(enum.StrEnum):
    """Evidence level assigned to a modality declaration."""

    SUPPORTED = "supported"
    EXPERIMENTAL = "experimental"
    EXTENSION_ONLY = "extension-only"


@dataclasses.dataclass(frozen=True, slots=True)
class StageEvidence:
    """Evidence record for one modality lifecycle stage.

    An empty ``evidence_links`` value is allowed while a declaration is being
    assembled. It makes the incomplete state explicit and lets the registry
    reject it instead of silently treating the stage as proven.

    Attributes:
        stage: Lifecycle stage covered by the record.
        evidence_links: Test, report, or artifact references proving the
            stage.
        evidence: JSON-native structured output produced by the stage.
        notes: Human-readable boundaries or observations for the stage.
    """

    stage: LifecycleStage
    evidence_links: tuple[str, ...] = ()
    evidence: Mapping[str, object] = dataclasses.field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validates stage identity and evidence text.

        Raises:
            TypeError: If the stage is not a `LifecycleStage` or the evidence is
                not a mapping.
            ValueError: If the evidence is not JSON-native or does not serialize
                to an object, or a link or note is blank.
        """
        if not isinstance(self.stage, LifecycleStage):
            raise TypeError("stage must be LifecycleStage")
        object.__setattr__(self, "evidence_links", tuple(self.evidence_links))
        if not isinstance(self.evidence, Mapping):
            raise TypeError("evidence must be a mapping")
        try:
            normalized = json.loads(
                json.dumps(
                    dict(self.evidence),
                    allow_nan=False,
                    sort_keys=True,
                )
            )
        except (TypeError, ValueError) as error:
            raise ValueError("evidence must be JSON-native") from error
        if not isinstance(normalized, dict):
            raise ValueError("evidence must serialize to an object")
        object.__setattr__(self, "evidence", normalized)
        object.__setattr__(self, "notes", tuple(self.notes))
        for field_name, values in (
            ("evidence_links", self.evidence_links),
            ("notes", self.notes),
        ):
            if any(not value.strip() for value in values):
                raise ValueError(f"{field_name} must not contain empty values")

    @property
    def evidenced(self) -> bool:
        """Whether at least one supporting reference is present."""
        return bool(self.evidence_links)

    def to_dict(self) -> dict[str, object]:
        """Returns a stable JSON-compatible stage record."""
        return {
            "stage": self.stage.value,
            "evidence_links": list(self.evidence_links),
            "evidence": dict(self.evidence),
            "notes": list(self.notes),
        }


@dataclasses.dataclass(frozen=True, slots=True)
class ModalityDeclaration:
    """Declared lifecycle support and evidence boundary for one modality.

    Attributes:
        modality: Dataset modality covered by the declaration.
        status: Support level assigned after review.
        stage_evidence: Evidence records for lifecycle stages.
        required_extras: Package extras needed to use the modality.
        resources: Minimum runtime and environment assumptions.
        failure_modes: Known failures that must be handled or reported.
        limitations: User-visible boundaries of the declaration.
    """

    modality: ModalityIdentity
    status: ModalityStatus
    stage_evidence: tuple[StageEvidence, ...]
    required_extras: tuple[str, ...] = ()
    resources: capabilities.ResourceAssumptions = dataclasses.field(
        default_factory=capabilities.ResourceAssumptions
    )
    failure_modes: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validates declaration shape and supported-status evidence.

        Raises:
            TypeError: If the modality, status, resources or stage evidence has
                the wrong type.
            ValueError: If stages repeat, an extra, failure mode or limitation
                is
                blank, no failure mode is declared, an extension modality is not
                extension-only, or a supported modality lacks evidence for a
                required lifecycle stage.
        """
        if not isinstance(self.modality, (Modality, ExtensionModality)):
            raise TypeError("modality must be Modality or ExtensionModality")
        if not isinstance(self.status, ModalityStatus):
            raise TypeError("status must be ModalityStatus")
        if not isinstance(self.resources, capabilities.ResourceAssumptions):
            raise TypeError(
                "resources must be capabilities.ResourceAssumptions"
            )
        object.__setattr__(self, "stage_evidence", tuple(self.stage_evidence))
        object.__setattr__(self, "required_extras", tuple(self.required_extras))
        object.__setattr__(self, "failure_modes", tuple(self.failure_modes))
        object.__setattr__(self, "limitations", tuple(self.limitations))
        if any(
            not isinstance(record, StageEvidence)
            for record in self.stage_evidence
        ):
            raise TypeError("stage_evidence must contain StageEvidence records")
        stages = [record.stage for record in self.stage_evidence]
        if len(stages) != len(set(stages)):
            raise ValueError("stage_evidence must not contain duplicate stages")
        for field_name, values in (
            ("required_extras", self.required_extras),
            ("failure_modes", self.failure_modes),
            ("limitations", self.limitations),
        ):
            if any(not value.strip() for value in values):
                raise ValueError(f"{field_name} must not contain empty values")
        if not self.failure_modes:
            raise ValueError("failure_modes must declare at least one value")
        if (
            isinstance(self.modality, ExtensionModality)
            and self.status is not ModalityStatus.EXTENSION_ONLY
        ):
            raise ValueError(
                "extension modalities must use extension-only status"
            )
        if self.status is ModalityStatus.SUPPORTED and self.missing_stages:
            missing = ", ".join(stage.value for stage in self.missing_stages)
            raise ValueError(
                "supported modalities require evidenced lifecycle stages: "
                f"{missing}"
            )

    @property
    def missing_stages(self) -> tuple[LifecycleStage, ...]:
        """The required lifecycle stages that are missing or lack evidence."""
        evidence = {record.stage: record for record in self.stage_evidence}
        return tuple(
            stage
            for stage in REQUIRED_LIFECYCLE_STAGES
            if stage not in evidence or not evidence[stage].evidenced
        )

    @property
    def complete(self) -> bool:
        """Whether every required lifecycle stage has evidence."""
        return not self.missing_stages

    def to_dict(self) -> dict[str, object]:
        """Returns a stable JSON-compatible declaration."""
        stage_records = {record.stage: record for record in self.stage_evidence}
        return {
            "modality": modality_value(self.modality),
            "status": self.status.value,
            "stage_evidence": [
                stage_records[stage].to_dict()
                for stage in REQUIRED_LIFECYCLE_STAGES
                if stage in stage_records
            ],
            "required_extras": list(self.required_extras),
            "resources": self.resources.to_dict(),
            "failure_modes": list(self.failure_modes),
            "limitations": list(self.limitations),
            "complete": self.complete,
            "missing_stages": [stage.value for stage in self.missing_stages],
        }


class UnsupportedModalityError(errors.ConfigError):
    """Raised when a modality cannot cross the support boundary."""

    category = "modality"


class ModalityRegistry:
    """Registry that resolves complete modality support before execution.

    The registry has no implicit modality defaults. A declaration must be
    explicitly registered, complete, and status-eligible before a caller can
    use it.

    Attributes:
        records: Registered declarations keyed by modality.
    """

    def __init__(
        self, declarations: Iterable[ModalityDeclaration] = ()
    ) -> None:
        """Creates a registry and register supplied declarations.

        Args:
            declarations: The declarations to register.
        """
        self.records: dict[ModalityIdentity, ModalityDeclaration] = {}
        for declaration in declarations:
            self.register(declaration)

    def register(self, declaration: ModalityDeclaration) -> None:
        """Registers a declaration without replacing an existing modality.

        Args:
            declaration: Modality lifecycle declaration.

        Raises:
            TypeError: If `declaration` is not a `ModalityDeclaration`.
            errors.ConfigError: If the modality is already registered.
        """
        if not isinstance(declaration, ModalityDeclaration):
            raise TypeError("declaration must be ModalityDeclaration")
        if declaration.modality in self.records:
            raise errors.ConfigError(
                "modality already registered: "
                f"{modality_value(declaration.modality)}"
            )
        self.records[declaration.modality] = declaration

    def find(self, modality: ModalityIdentity) -> ModalityDeclaration | None:
        """Returns a declaration or ``None`` when it is unregistered.

        Args:
            modality: The modality to look up.

        Returns:
            The declaration, or None.

        Raises:
            TypeError: If `modality` is not a `Modality` or `ExtensionModality`.
        """
        if not isinstance(modality, (Modality, ExtensionModality)):
            raise TypeError("modality must be Modality or ExtensionModality")
        return self.records.get(modality)

    def require(
        self,
        modality: ModalityIdentity,
        allowed_statuses: Collection[ModalityStatus] | None = None,
    ) -> ModalityDeclaration:
        """Require complete, status-eligible modality support.

        Incomplete declarations fail regardless of requested status. This
        prevents an explicit experimental opt-in from bypassing the lifecycle
        evidence boundary. An absent, incomplete or status-ineligible modality
        is refused with `UnsupportedModalityError` (raised by
        `raise_unsupported`).

        Args:
            modality: Modality to resolve.
            allowed_statuses: Explicit statuses accepted by the caller. By
                default only ``supported`` is accepted.

        Returns:
            The registered complete declaration.
        """
        declaration = self.find(modality)
        statuses = (
            frozenset({ModalityStatus.SUPPORTED})
            if allowed_statuses is None
            else frozenset(allowed_statuses)
        )
        if declaration is None:
            self.raise_unsupported(modality, "no declaration exists")
        if not declaration.complete:
            missing = [stage.value for stage in declaration.missing_stages]
            self.raise_unsupported(
                modality,
                "declaration is incomplete",
                {"missing_stages": missing},
            )
        if declaration.status not in statuses:
            self.raise_unsupported(
                modality,
                f"registered as {declaration.status.value}; accepted statuses "
                f"are {sorted(status.value for status in statuses)}",
            )
        return declaration

    def list_declarations(self) -> tuple[ModalityDeclaration, ...]:
        """Returns declarations in deterministic modality order."""
        return tuple(
            self.records[modality]
            for modality in sorted(self.records, key=lambda item: item.value)
        )

    def to_dict(self) -> dict[str, object]:
        """Returns a deterministic registry representation."""
        return {
            "declarations": [
                declaration.to_dict()
                for declaration in self.list_declarations()
            ]
        }

    def to_json(self) -> str:
        """Returns deterministic JSON suitable for evidence artifacts."""
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    def raise_unsupported(
        self,
        modality: ModalityIdentity,
        reason: str,
        details: dict[str, object] | None = None,
    ) -> NoReturn:
        """Raises a structured pre-execution modality rejection.

        Args:
            modality: The modality that was refused.
            reason: Why it was refused.
            details: Extra facts added to the diagnostic.

        Raises:
            UnsupportedModalityError: Always, with a diagnostic naming the
                modality, the reason and the registered modalities.
        """
        diagnostic: dict[str, object] = {
            "status": "unsupported",
            "reason": reason,
            "modality": modality_value(modality),
            "registered_modalities": [
                modality_value(item.modality)
                for item in self.list_declarations()
            ],
        }
        if details:
            diagnostic.update(details)
        raise UnsupportedModalityError(
            f"unsupported modality before execution: {reason}; "
            f"modality={modality_value(modality)}",
            diagnostic,
        )
