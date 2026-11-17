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
class Bespoke:
    """Stable identity for a caller-defined dataset family.

    Extension identities let a dataset family enter the same lifecycle
    contract without pretending that the core package understands its raw
    representation. A `Declaration` for one must use the
    extension-only status, so it can never be declared as supported.
    """

    name: str

    def __post_init__(self) -> None:
        """Validates a stable, non-built-in dataset-family name.

        Raises:
            TypeError: If the name is not a string.
            ValueError: If the name is not at least two characters of
                lowercase letters, digits, ``.``, ``/`` and ``-`` that start
                and end with a letter or digit (for example
                ``acme/point-cloud``), or equals a built-in modality value.
        """
        if not isinstance(self.name, str):
            raise errors.KindError("extension modality name must be a string")
        if not re.fullmatch(r"[a-z0-9][a-z0-9./-]*[a-z0-9]", self.name):
            raise errors.InputError(
                "extension modality name must use lowercase namespaced "
                "characters such as 'acme/point-cloud'"
            )
        if self.name in {item.value for item in Modality}:
            raise errors.InputError(
                "extension modality name conflicts with a built-in"
            )

    @property
    def value(self) -> str:
        """The serialized identity used in evidence artifacts."""
        return f"extension:{self.name}"


ModalityIdentity = Modality | Bespoke


def serialise(modality: ModalityIdentity) -> str:
    """Returns the stable serialized value for a modality identity.

    Args:
        modality: A built-in or extension modality.

    Returns:
        The built-in modality's value, or ``extension:`` followed by the
        extension name.

    Raises:
        TypeError: If `modality` is not a `Modality` or `Bespoke`.
    """
    if not isinstance(modality, (Modality, Bespoke)):
        raise errors.KindError("modality must be Modality or Bespoke")
    return modality.value


def restore(value: str) -> ModalityIdentity:
    """Restores a built-in or extension modality from evidence text.

    Args:
        value: The serialized value, as returned by `serialise`.

    Returns:
        The matching built-in or extension modality.

    Raises:
        ValueError: If the value is not a non-empty string, is neither a
            built-in value nor prefixed with ``extension:``, or the text
            after the prefix is not a valid `Bespoke` name.
    """
    candidate: object = value  # callers may pass a non-string
    if not isinstance(candidate, str) or not candidate.strip():
        raise errors.InputError("modality value must be a non-empty string")
    try:
        return Modality(value)
    except ValueError:
        prefix = "extension:"
        if not value.startswith(prefix):
            raise errors.InputError(
                f"unknown modality value: {value}"
            ) from None
        return Bespoke(value.removeprefix(prefix))


class Phase(enum.StrEnum):
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


REQUIRED_LIFECYCLE_STAGES = tuple(Phase)


class Status(enum.StrEnum):
    """Evidence level assigned to a modality declaration."""

    SUPPORTED = "supported"
    EXPERIMENTAL = "experimental"
    EXTENSION_ONLY = "extension-only"


@dataclasses.dataclass(frozen=True, slots=True)
class Proof:
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

    stage: Phase
    evidence_links: tuple[str, ...] = ()
    evidence: Mapping[str, object] = dataclasses.field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validates stage identity and evidence text.

        Raises:
            TypeError: If the stage is not a `Phase` or the evidence is
                not a mapping.
            ValueError: If the evidence is not JSON-native, or a link or note
                is blank.
        """
        if not isinstance(self.stage, Phase):
            raise errors.KindError("stage must be Phase")
        object.__setattr__(self, "evidence_links", tuple(self.evidence_links))
        if not isinstance(self.evidence, Mapping):
            raise errors.KindError("evidence must be a mapping")
        try:
            normalized = json.loads(
                json.dumps(
                    dict(self.evidence),
                    allow_nan=False,
                    sort_keys=True,
                )
            )
        except (TypeError, ValueError) as error:
            raise errors.InputError("evidence must be JSON-native") from error
        if not isinstance(normalized, dict):
            raise errors.InputError("evidence must serialize to an object")
        object.__setattr__(self, "evidence", normalized)
        object.__setattr__(self, "notes", tuple(self.notes))
        for field_name, values in (
            ("evidence_links", self.evidence_links),
            ("notes", self.notes),
        ):
            if any(not value.strip() for value in values):
                raise errors.InputError(
                    f"{field_name} must not contain empty values"
                )

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


def blanks(
    fields: tuple[tuple[str, tuple[str, ...]], ...],
) -> None:
    """Checks that none of the named text collections holds a blank value.

    Args:
        fields: Pairs of a field name and its values.

    Raises:
        ValueError: If any value is blank.
    """
    for field_name, values in fields:
        if any(not value.strip() for value in values):
            raise errors.InputError(
                f"{field_name} must not contain empty values"
            )


@dataclasses.dataclass(frozen=True, slots=True)
class Declaration:
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
    status: Status
    stage_evidence: tuple[Proof, ...]
    required_extras: tuple[str, ...] = ()
    resources: capabilities.Resources = dataclasses.field(
        default_factory=capabilities.Resources
    )
    failure_modes: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validates declaration shape and supported-status evidence.

        Raises:
            TypeError: If the modality, status, resources or stage evidence has
                the wrong type.
            ValueError: If a stage appears twice, a required extra, failure
                mode or limitation is blank, no failure mode is declared, an
                extension modality does not have extension-only status, or a
                supported modality lacks evidence for a required lifecycle
                stage.
        """
        if not isinstance(self.modality, (Modality, Bespoke)):
            raise errors.KindError("modality must be Modality or Bespoke")
        if not isinstance(self.status, Status):
            raise errors.KindError("status must be Status")
        if not isinstance(self.resources, capabilities.Resources):
            raise errors.KindError("resources must be capabilities.Resources")
        object.__setattr__(self, "stage_evidence", tuple(self.stage_evidence))
        object.__setattr__(self, "required_extras", tuple(self.required_extras))
        object.__setattr__(self, "failure_modes", tuple(self.failure_modes))
        object.__setattr__(self, "limitations", tuple(self.limitations))
        if any(not isinstance(record, Proof) for record in self.stage_evidence):
            raise errors.KindError("stage_evidence must contain Proof records")
        stages = [record.stage for record in self.stage_evidence]
        if len(stages) != len(set(stages)):
            raise errors.InputError(
                "stage_evidence must not contain duplicate stages"
            )
        blanks(
            (
                ("required_extras", self.required_extras),
                ("failure_modes", self.failure_modes),
                ("limitations", self.limitations),
            )
        )
        if not self.failure_modes:
            raise errors.InputError(
                "failure_modes must declare at least one value"
            )
        if (
            isinstance(self.modality, Bespoke)
            and self.status is not Status.EXTENSION_ONLY
        ):
            raise errors.InputError(
                "extension modalities must use extension-only status"
            )
        if self.status is Status.SUPPORTED and self.missing_stages:
            missing = ", ".join(stage.value for stage in self.missing_stages)
            raise errors.InputError(
                "supported modalities require evidenced lifecycle stages: "
                f"{missing}"
            )

    @property
    def missing_stages(self) -> tuple[Phase, ...]:
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
            "modality": serialise(self.modality),
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


class UnsupportedError(errors.ConfigError):
    """Raised when a modality cannot cross the support boundary."""

    category = "modality"


class Registry:
    """Registry that resolves complete modality support before execution.

    The registry has no implicit modality defaults. A declaration must be
    explicitly registered, complete, and status-eligible before a caller can
    use it.

    Attributes:
        records: Registered declarations keyed by modality.
    """

    def __init__(self, declarations: Iterable[Declaration] = ()) -> None:
        """Creates a registry and registers the supplied declarations.

        Args:
            declarations: Declarations registered in order with `register`,
                which rejects non-declarations and repeated modalities.
        """
        self.records: dict[ModalityIdentity, Declaration] = {}
        for declaration in declarations:
            self.register(declaration)

    def register(self, declaration: Declaration) -> None:
        """Registers a declaration without replacing an existing modality.

        Args:
            declaration: Modality lifecycle declaration.

        Raises:
            TypeError: If `declaration` is not a `Declaration`.
            errors.ConfigError: If the modality is already registered.
        """
        if not isinstance(declaration, Declaration):
            raise errors.KindError("declaration must be Declaration")
        if declaration.modality in self.records:
            raise errors.ConfigError(
                "modality already registered: "
                f"{serialise(declaration.modality)}"
            )
        self.records[declaration.modality] = declaration

    def find(self, modality: ModalityIdentity) -> Declaration | None:
        """Returns a declaration or ``None`` when it is unregistered.

        Args:
            modality: The modality to look up.

        Returns:
            The declaration, or None.

        Raises:
            TypeError: If `modality` is not a `Modality` or `Bespoke`.
        """
        if not isinstance(modality, (Modality, Bespoke)):
            raise errors.KindError("modality must be Modality or Bespoke")
        return self.records.get(modality)

    def require(
        self,
        modality: ModalityIdentity,
        allowed_statuses: Collection[Status] | None = None,
    ) -> Declaration:
        """Requires complete, status-eligible modality support.

        Incomplete declarations fail regardless of requested status. This
        prevents an explicit experimental opt-in from bypassing the lifecycle
        evidence boundary. An absent, incomplete or status-ineligible modality
        is refused with `UnsupportedError` (raised by
        `refuse`); a non-modality argument raises `TypeError`.

        Args:
            modality: Modality to resolve.
            allowed_statuses: Explicit statuses accepted by the caller. By
                default only ``supported`` is accepted.

        Returns:
            The registered complete declaration.
        """
        declaration = self.find(modality)
        statuses = (
            frozenset({Status.SUPPORTED})
            if allowed_statuses is None
            else frozenset(allowed_statuses)
        )
        if declaration is None:
            self.refuse(modality, "no declaration exists")
        if not declaration.complete:
            missing = [stage.value for stage in declaration.missing_stages]
            self.refuse(
                modality,
                "declaration is incomplete",
                {"missing_stages": missing},
            )
        if declaration.status not in statuses:
            self.refuse(
                modality,
                f"registered as {declaration.status.value}; accepted statuses "
                f"are {sorted(status.value for status in statuses)}",
            )
        return declaration

    def catalog(self) -> tuple[Declaration, ...]:
        """Returns declarations in deterministic modality order."""
        return tuple(
            self.records[modality]
            for modality in sorted(self.records, key=lambda item: item.value)
        )

    def to_dict(self) -> dict[str, object]:
        """Returns a deterministic registry representation."""
        return {
            "declarations": [
                declaration.to_dict() for declaration in self.catalog()
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

    def refuse(
        self,
        modality: ModalityIdentity,
        reason: str,
        details: dict[str, object] | None = None,
    ) -> NoReturn:
        """Raises a structured pre-execution modality rejection.

        Args:
            modality: The modality that was refused; it must be a `Modality`
                or `Bespoke`, otherwise `TypeError` is raised.
            reason: Why it was refused.
            details: Extra facts added to the diagnostic.

        Raises:
            UnsupportedError: Always, with a diagnostic naming the
                modality, the reason and the registered modalities.
        """
        diagnostic: dict[str, object] = {
            "status": "unsupported",
            "reason": reason,
            "modality": serialise(modality),
            "registered_modalities": [
                serialise(item.modality) for item in self.catalog()
            ],
        }
        if details:
            diagnostic.update(details)
        raise UnsupportedError(
            f"unsupported modality before execution: {reason}; "
            f"modality={serialise(modality)}",
            diagnostic,
        )
