"""Capability contracts for pre-training support decisions.

The registry is deliberately independent from model adapters. A planner can
ask whether a complete capability is eligible before it constructs a model,
loads an optional dependency, or starts training.
"""

from collections.abc import Collection
from collections.abc import Iterable
import dataclasses
import enum
import json
import math
from typing import NoReturn

from scmle.contracts import errors
from scmle.contracts import tasks

CAPABILITY_SCHEMA_VERSION = 1
"""Schema version for serialized capability-registry evidence."""


class Modality(enum.StrEnum):
    """Data modality named by a capability boundary."""

    TABULAR = "tabular"
    TEXT = "text"
    DOCUMENT = "document"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    GRAPH = "graph"
    GEOSPATIAL = "geospatial"
    RASTER = "raster"


class Evaluation(enum.StrEnum):
    """Evaluation policy required by a capability."""

    STANDARD = "standard"
    CALIBRATED = "calibrated"
    SURVIVAL = "survival"
    RANKING = "ranking"
    FORECASTING = "forecasting"
    UPLIFT = "uplift"
    MULTILABEL = "multilabel"
    MULTIOUTPUT = "multioutput"


class Uncertainty(enum.StrEnum):
    """Uncertainty output promised by a capability."""

    NONE = "none"
    CALIBRATED_PROBABILITY = "calibrated_probability"
    PREDICTION_INTERVAL = "prediction_interval"
    CONFORMAL = "conformal"
    SURVIVAL_CURVE = "survival_curve"
    ABSTENTION = "abstention"


class Fairness(enum.StrEnum):
    """Fairness analysis or mitigation promised by a capability."""

    NONE = "none"
    GROUP_REPORT = "group_report"
    INTERSECTIONAL_REPORT = "intersectional_report"
    MITIGATION = "mitigation"


class Explainability(enum.StrEnum):
    """Explanation contract promised by a capability."""

    NONE = "none"
    GLOBAL = "global"
    LOCAL = "local"
    COUNTERFACTUAL = "counterfactual"
    MODALITY_SPECIFIC = "modality_specific"
    FULL = "full"


class Inference(enum.StrEnum):
    """Inference surface promised by a capability."""

    LIBRARY = "library"
    BATCH = "batch"
    ONNX = "onnx"
    REFERENCE_HTTP = "reference_http"
    STREAMING = "streaming"


class Status(enum.StrEnum):
    """Evidence level assigned to a registered capability."""

    SUPPORTED = "supported"
    EXPERIMENTAL = "experimental"
    EXTENSION_ONLY = "extension-only"


class ModelFamily(enum.StrEnum):
    """Named model families with an explicit support boundary.

    These values describe model-family identity only. A family is usable for a
    run only after a complete :class:`Coordinate` declaration is registered.
    """

    TRANSFORMER = "transformer"
    TEXT_TRANSFORMER = "text_transformer"
    VISION_TRANSFORMER = "vision_transformer"
    AUDIO_TRANSFORMER = "audio_transformer"
    VIDEO_TRANSFORMER = "video_transformer"
    MULTIMODAL_TRANSFORMER = "multimodal_transformer"
    JEPA = "jepa"


@dataclasses.dataclass(frozen=True, slots=True)
class ModelFamilyDeclaration:
    """Evidence boundary for a model family without full task support.

    A family declaration is intentionally not a substitute for a complete
    capability. It lets tooling answer whether a family is known, which
    modalities and optional dependencies it claims, and why it remains
    blocked before a complete task/evaluation contract exists.
    """

    model_family: str
    status: Status
    modalities: tuple[Modality, ...]
    required_extras: tuple[str, ...]
    resources: "ResourceAssumptions"
    failure_modes: tuple[str, ...]
    evidence_links: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validates and canonicalizes a family declaration.

        Raises:
            TypeError: If the family name is not a string, or the status,
                resources or a modality has the wrong type.
            ValueError: If the family name is empty, not lowercase or has
                surrounding whitespace, no modality is declared, or a list
                entry is blank.
        """
        if not isinstance(self.model_family, str):
            raise TypeError("model_family must be a string")
        if not self.model_family or self.model_family != (
            self.model_family.strip().lower()
        ):
            raise ValueError(
                "model_family must be lowercase and must not have surrounding "
                "whitespace"
            )
        if not isinstance(self.status, Status):
            raise TypeError("status must be Status")
        if not isinstance(self.resources, ResourceAssumptions):
            raise TypeError("resources must be ResourceAssumptions")
        if not self.modalities:
            raise ValueError("modalities must declare at least one value")
        if any(not isinstance(item, Modality) for item in self.modalities):
            raise TypeError("modalities must contain Modality values")
        object.__setattr__(
            self,
            "modalities",
            tuple(sorted(set(self.modalities), key=lambda item: item.value)),
        )
        for name, values in (
            ("required_extras", self.required_extras),
            ("failure_modes", self.failure_modes),
            ("evidence_links", self.evidence_links),
            ("limitations", self.limitations),
        ):
            entries: Iterable[object] = (
                values  # may hold non-strings at runtime
            )
            if any(
                not isinstance(value, str) or not value.strip()
                for value in entries
            ):
                raise ValueError(f"{name} must contain non-empty strings")
            object.__setattr__(self, name, tuple(sorted(set(values))))

    def to_dict(self) -> dict[str, object]:
        """Returns a stable JSON-compatible family declaration."""
        return {
            "model_family": self.model_family,
            "status": self.status.value,
            "modalities": [item.value for item in self.modalities],
            "required_extras": list(self.required_extras),
            "resources": self.resources.to_dict(),
            "failure_modes": list(self.failure_modes),
            "evidence_links": list(self.evidence_links),
            "limitations": list(self.limitations),
        }

    def supports_modality(self, modality: Modality) -> bool:
        """Returns whether this family declares the requested modality.

        Args:
            modality: The modality to check.

        Returns:
            True if the family declares it.

        Raises:
            TypeError: If `modality` is not a `Modality`.
        """
        if not isinstance(modality, Modality):
            raise TypeError("modality must be Modality")
        return modality in self.modalities


@dataclasses.dataclass(frozen=True, slots=True)
class Environment:
    """Observed runtime facts used to cross a capability boundary.

    Values are deliberately supplied by the caller. The registry never
    guesses that an optional extra, GPU, network, or resource budget exists.
    ``None`` means unknown and is rejected when the declaration requires that
    fact.
    """

    available_extras: frozenset[str] = frozenset()
    cpu_cores: int | None = None
    memory_gb: float | None = None
    disk_gb: float | None = None
    gpu_available: bool | None = None
    network_available: bool | None = None

    def __post_init__(self) -> None:
        """Validates runtime observations before they are used for gating.

        Raises:
            ValueError: If an available extra is blank, the CPU count is below
                one, or the memory or disk size is not finite and non-negative.
        """
        extras: Iterable[object] = self.available_extras  # may hold non-strings
        if any(
            not isinstance(extra, str) or not extra.strip() for extra in extras
        ):
            raise ValueError("available_extras must contain non-empty strings")
        object.__setattr__(
            self,
            "available_extras",
            frozenset(self.available_extras),
        )
        if self.cpu_cores is not None and self.cpu_cores < 1:
            raise ValueError("cpu_cores must be at least 1")
        for name, value in (
            ("memory_gb", self.memory_gb),
            ("disk_gb", self.disk_gb),
        ):
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and non-negative")


@dataclasses.dataclass(frozen=True, slots=True)
class Reason:
    """Machine-readable reason a capability cannot cross the boundary."""

    code: str
    message: str
    expected: object | None = None
    observed: object | None = None

    def to_dict(self) -> dict[str, object]:
        """Returns a stable JSON-compatible diagnostic."""
        return {
            "code": self.code,
            "message": self.message,
            "expected": self.expected,
            "observed": self.observed,
        }


@dataclasses.dataclass(frozen=True, slots=True)
class Coordinate:
    """Coordinates of one complete pre-training capability decision.

    Attributes:
        modality: Input data modality.
        task: Task type the capability covers.
        model_family: Lowercase family identifier, not an adapter name.
        evaluation: Evaluation policy that must be available.
        uncertainty: Uncertainty contract that must be available.
        fairness: Fairness contract that must be available.
        explainability: Explanation contract that must be available.
        inference: Inference surface that must be available.
    """

    modality: Modality
    task: tasks.Task
    model_family: str
    evaluation: Evaluation
    uncertainty: Uncertainty
    fairness: Fairness
    explainability: Explainability
    inference: Inference

    def __post_init__(self) -> None:
        """Rejects non-canonical dimension values at the contract boundary.

        Raises:
            TypeError: If a dimension has the wrong enumeration type or the
                model family is not a string.
            ValueError: If the model family is empty, not lowercase or has
                surrounding whitespace.
        """
        dimensions = (
            ("modality", self.modality, Modality),
            ("task", self.task, tasks.Task),
            ("evaluation", self.evaluation, Evaluation),
            ("uncertainty", self.uncertainty, Uncertainty),
            ("fairness", self.fairness, Fairness),
            ("explainability", self.explainability, Explainability),
            ("inference", self.inference, Inference),
        )
        for name, value, expected in dimensions:
            if not isinstance(value, expected):
                raise TypeError(f"{name} must be {expected.__name__}")
        if not isinstance(self.model_family, str):
            raise TypeError("model_family must be a string")
        if not self.model_family:
            raise ValueError("model_family must not be empty")
        if self.model_family != self.model_family.strip().lower():
            raise ValueError(
                "model_family must be lowercase and must not have surrounding "
                "whitespace"
            )

    def to_dict(self) -> dict[str, str]:
        """Returns a stable, JSON-compatible representation of the key."""
        return {
            "modality": self.modality.value,
            "task": self.task.value,
            "model_family": self.model_family,
            "evaluation": self.evaluation.value,
            "uncertainty": self.uncertainty.value,
            "fairness": self.fairness.value,
            "explainability": self.explainability.value,
            "inference": self.inference.value,
        }


@dataclasses.dataclass(frozen=True, slots=True)
class ResourceAssumptions:
    """Minimum resources and environmental assumptions for a capability.

    Attributes:
        cpu_cores: Minimum number of CPU cores required.
        memory_gb: Minimum working memory required.
        disk_gb: Minimum temporary disk space required.
        gpu_required: Whether a compatible GPU is required.
        network_required: Whether network access is required at runtime.
        max_rows: Practical row limit, if one is known.
        notes: Human-readable constraints that belong in release evidence.
    """

    cpu_cores: int = 1
    memory_gb: float = 1.0
    disk_gb: float = 0.0
    gpu_required: bool = False
    network_required: bool = False
    max_rows: int | None = None
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validates resource assumptions before they enter the registry.

        Raises:
            TypeError: If the CPU count is not an integer.
            ValueError: If the CPU count is below one, the memory is not finite
                and positive, the disk size is not finite and non-negative, the
                row limit is below one, or a note is blank.
        """
        if not isinstance(self.cpu_cores, int):
            raise TypeError("cpu_cores must be an integer")
        if self.cpu_cores < 1:
            raise ValueError("cpu_cores must be at least 1")
        if not math.isfinite(self.memory_gb) or self.memory_gb <= 0:
            raise ValueError("memory_gb must be greater than 0")
        if not math.isfinite(self.disk_gb) or self.disk_gb < 0:
            raise ValueError("disk_gb must not be negative")
        if self.max_rows is not None and self.max_rows < 1:
            raise ValueError("max_rows must be at least 1")
        notes: Iterable[object] = self.notes  # may hold non-strings at runtime
        if any(not isinstance(note, str) or not note.strip() for note in notes):
            raise ValueError("resource notes must not be empty")
        object.__setattr__(self, "notes", tuple(sorted(set(self.notes))))

    def to_dict(self) -> dict[str, object]:
        """Returns a stable, JSON-compatible representation."""
        return {
            "cpu_cores": self.cpu_cores,
            "memory_gb": self.memory_gb,
            "disk_gb": self.disk_gb,
            "gpu_required": self.gpu_required,
            "network_required": self.network_required,
            "max_rows": self.max_rows,
            "notes": list(self.notes),
        }


@dataclasses.dataclass(frozen=True, slots=True)
class Capability:
    """Declared support and evidence boundary for one capability key.

    Attributes:
        key: The complete eight-dimension capability coordinate.
        status: Supported, experimental, or extension-only.
        required_extras: Optional package extras needed to use the capability.
        resources: Minimum resource and environment assumptions.
        failure_modes: Known conditions that must be handled or reported.
        evidence_links: References to tests or evidence artifacts.
        limitations: User-visible limitations of the declaration.
    """

    key: Coordinate
    status: Status
    required_extras: tuple[str, ...]
    resources: ResourceAssumptions
    failure_modes: tuple[str, ...]
    evidence_links: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Requires complete declarations rather than optimistic defaults.

        Raises:
            TypeError: If the key, status or resources have the wrong type.
            ValueError: If no failure mode is declared, a list entry is blank,
                or a supported capability has no evidence links.
        """
        if not isinstance(self.key, Coordinate):
            raise TypeError("key must be Coordinate")
        if not isinstance(self.status, Status):
            raise TypeError("status must be Status")
        if not isinstance(self.resources, ResourceAssumptions):
            raise TypeError("resources must be ResourceAssumptions")
        object.__setattr__(self, "required_extras", tuple(self.required_extras))
        object.__setattr__(self, "failure_modes", tuple(self.failure_modes))
        object.__setattr__(self, "evidence_links", tuple(self.evidence_links))
        object.__setattr__(self, "limitations", tuple(self.limitations))
        if not self.failure_modes:
            raise ValueError("failure_modes must declare at least one value")
        for name, values in (
            ("required_extras", self.required_extras),
            ("failure_modes", self.failure_modes),
            ("evidence_links", self.evidence_links),
            ("limitations", self.limitations),
        ):
            entries: Iterable[object] = (
                values  # may hold non-strings at runtime
            )
            if any(
                not isinstance(value, str) or not value.strip()
                for value in entries
            ):
                raise ValueError(f"{name} must contain non-empty strings")
        if self.status is Status.SUPPORTED and not self.evidence_links:
            raise ValueError(
                "supported capabilities must declare evidence_links"
            )

    def to_dict(self) -> dict[str, object]:
        """Returns a stable, JSON-compatible capability declaration."""
        return {
            "key": self.key.to_dict(),
            "status": self.status.value,
            "required_extras": list(self.required_extras),
            "resources": self.resources.to_dict(),
            "failure_modes": list(self.failure_modes),
            "evidence_links": list(self.evidence_links),
            "limitations": list(self.limitations),
        }


DEFAULT_MODEL_FAMILY_DECLARATIONS = (
    ModelFamilyDeclaration(
        model_family=ModelFamily.TRANSFORMER.value,
        status=Status.EXPERIMENTAL,
        modalities=(Modality.TABULAR,),
        required_extras=("deeplearning",),
        resources=ResourceAssumptions(memory_gb=4.0, gpu_required=True),
        failure_modes=(
            "torch runtime or tabular transformer adapter is unavailable",
            "numeric tabular feature tokenization fails",
            "end-to-end transformer evaluation contract is incomplete",
        ),
        limitations=(
            "experimental ft_transformer tabular feature-token path only",
            "not a text, image, audio, video, or multimodal transformer",
            "not a complete production capability declaration",
        ),
    ),
    ModelFamilyDeclaration(
        model_family=ModelFamily.TEXT_TRANSFORMER.value,
        status=Status.EXTENSION_ONLY,
        modalities=(Modality.TEXT, Modality.DOCUMENT),
        required_extras=("embeddings",),
        resources=ResourceAssumptions(memory_gb=4.0),
        failure_modes=(
            "tokenization or text normalization fails",
            "pretrained weights are unavailable",
            "evaluation or explanation evidence is incomplete",
        ),
        limitations=(
            "extension-only family declaration",
            "no built-in text or document transformer training path",
            "no production inference guarantee",
        ),
    ),
    ModelFamilyDeclaration(
        model_family=ModelFamily.VISION_TRANSFORMER.value,
        status=Status.EXTENSION_ONLY,
        modalities=(Modality.IMAGE, Modality.VIDEO),
        required_extras=("multimodal",),
        resources=ResourceAssumptions(memory_gb=8.0, gpu_required=True),
        failure_modes=(
            "image or video decoder is unavailable",
            "pretrained weights are unavailable",
            "modality-specific evaluation is incomplete",
        ),
        limitations=(
            "extension-only family declaration",
            "no built-in image or video transformer training path",
            "no production lifecycle evidence",
        ),
    ),
    ModelFamilyDeclaration(
        model_family=ModelFamily.AUDIO_TRANSFORMER.value,
        status=Status.EXTENSION_ONLY,
        modalities=(Modality.AUDIO,),
        required_extras=("multimodal",),
        resources=ResourceAssumptions(memory_gb=8.0, gpu_required=True),
        failure_modes=(
            "audio decoder or resampler is unavailable",
            "pretrained weights are unavailable",
            "audio evaluation is incomplete",
        ),
        limitations=(
            "extension-only family declaration",
            "no built-in audio transformer training path",
            "no production lifecycle evidence",
        ),
    ),
    ModelFamilyDeclaration(
        model_family=ModelFamily.VIDEO_TRANSFORMER.value,
        status=Status.EXTENSION_ONLY,
        modalities=(Modality.VIDEO,),
        required_extras=("multimodal",),
        resources=ResourceAssumptions(memory_gb=12.0, gpu_required=True),
        failure_modes=(
            "video decoder or frame sampler is unavailable",
            "pretrained weights are unavailable",
            "video evaluation is incomplete",
        ),
        limitations=(
            "extension-only family declaration",
            "no built-in video transformer training path",
            "no production lifecycle evidence",
        ),
    ),
    ModelFamilyDeclaration(
        model_family=ModelFamily.MULTIMODAL_TRANSFORMER.value,
        status=Status.EXTENSION_ONLY,
        modalities=(
            Modality.TEXT,
            Modality.DOCUMENT,
            Modality.IMAGE,
            Modality.AUDIO,
            Modality.VIDEO,
        ),
        required_extras=("multimodal",),
        resources=ResourceAssumptions(memory_gb=12.0, gpu_required=True),
        failure_modes=(
            "modality adapter is caller-supplied",
            "cross-modal alignment is not evidenced",
            "deployment and monitoring contracts are incomplete",
        ),
        limitations=(
            "extension-only family declaration",
            "must provide independent lifecycle evidence",
        ),
    ),
    ModelFamilyDeclaration(
        model_family=ModelFamily.JEPA.value,
        status=Status.EXPERIMENTAL,
        modalities=(Modality.TABULAR,),
        required_extras=("deeplearning",),
        resources=ResourceAssumptions(memory_gb=2.0),
        failure_modes=(
            "torch runtime is unavailable",
            "numeric tabular input is not compatible",
            "representation quality and downstream task evidence are absent",
        ),
        limitations=(
            "tabular representation learning only",
            "not a multimodal JEPA implementation",
            "no complete supervised capability declaration",
        ),
    ),
)


class UnsupportedCapabilityError(errors.ConfigError):
    """Raised when a requested capability cannot cross the support boundary."""

    category = "capability"


class Registry:
    """Registry that resolves complete capabilities before training.

    The registry has no implicit defaults. Callers must register an explicit
    declaration for every combination they intend to train, which prevents a
    model adapter's existence from being mistaken for end-to-end support.

    Attributes:
        records: Registered capabilities keyed by their complete coordinate.
        model_families: Registered family declarations keyed by family name.
        runtime_environment: The bound runtime environment, or None until
            :meth:`bind_runtime_environment` is called.
    """

    def __init__(
        self,
        capabilities: Iterable[Capability] = (),
        model_families: Iterable[ModelFamilyDeclaration] = (),
    ) -> None:
        """Creates a registry from complete capability declarations.

        Args:
            capabilities: The capabilities to register.
            model_families: The model-family declarations to register.
        """
        self.records: dict[Coordinate, Capability] = {}
        self.model_families: dict[str, ModelFamilyDeclaration] = {}
        self.runtime_environment: Environment | None = None
        for capability in capabilities:
            self.register(capability)
        for declaration in model_families:
            self.register_model_family(declaration)

    @classmethod
    def with_declared_model_families(
        cls,
        capabilities: Iterable[Capability] = (),
    ) -> "Registry":
        """Creates a registry with the explicit transformer/JEPA catalog.

        The catalog is metadata, not support promotion. Callers still need a
        complete capability declaration for every task and contract dimension
        before :meth:`require` can succeed.

        Args:
            capabilities: The capabilities to register.

        Returns:
            The registry, with the default model-family declarations.
        """
        return cls(capabilities, DEFAULT_MODEL_FAMILY_DECLARATIONS)

    def register(self, capability: Capability) -> None:
        """Registers a capability without silently replacing an existing one.

        Args:
            capability: Complete capability declaration.

        Raises:
            TypeError: If `capability` is not a `Capability`.
            errors.ConfigError: If the key is already registered.
        """
        if not isinstance(capability, Capability):
            raise TypeError("capability must be Capability")
        if capability.key in self.records:
            raise errors.ConfigError(
                f"capability already registered: {capability.key.to_dict()}"
            )
        self.records[capability.key] = capability

    def register_model_family(
        self,
        declaration: ModelFamilyDeclaration,
    ) -> None:
        """Registers a model-family status without replacing another one.

        Args:
            declaration: The family declaration.

        Raises:
            ConfigError: If the family is already registered.
            TypeError: If `declaration` is not a `ModelFamilyDeclaration`.
        """
        if not isinstance(declaration, ModelFamilyDeclaration):
            raise TypeError("declaration must be ModelFamilyDeclaration")
        if declaration.model_family in self.model_families:
            raise errors.ConfigError(
                f"model family already registered: {declaration.model_family}"
            )
        self.model_families[declaration.model_family] = declaration

    def find(self, key: Coordinate) -> Capability | None:
        """Returns a declaration or ``None`` when the key is unregistered.

        Args:
            key: The capability request.

        Returns:
            The registered capability, or None.
        """
        return self.records.get(key)

    def find_model_family(
        self,
        model_family: str,
    ) -> ModelFamilyDeclaration | None:
        """Returns family metadata or ``None`` when it is not catalogued.

        Args:
            model_family: The family name.

        Returns:
            The family declaration, or None.

        Raises:
            TypeError: If `model_family` is not a string.
        """
        if not isinstance(model_family, str):
            raise TypeError("model_family must be a string")
        return self.model_families.get(model_family)

    def bind_runtime_environment(self, environment: Environment) -> None:
        """Binds observed facts used when a caller omits an environment.

        Args:
            environment: The observed runtime environment.

        Raises:
            TypeError: If `environment` is not a `Environment`.
        """
        if not isinstance(environment, Environment):
            raise TypeError("environment must be Environment")
        self.runtime_environment = environment

    def diagnose(
        self,
        key: Coordinate,
        allowed_statuses: Collection[Status] | None = None,
        *,
        environment: Environment | None = None,
        row_count: int | None = None,
    ) -> tuple[Reason, ...]:
        """Returns every pre-training rejection for a capability request.

        Diagnostics are collected rather than short-circuited so a caller can
        fix one configuration or environment and see the remaining blockers
        in the same response. An empty tuple is an affirmative decision.

        Args:
            key: Complete capability request.
            allowed_statuses: Statuses the caller accepts; only ``supported``
                if None.
            environment: Observed extras and resources; the bound runtime
                environment if None.
            row_count: Dataset size used to check a declared row limit.

        Returns:
            The diagnostics. A key with no declaration yields a missing
            declaration entry plus, when its model family is catalogued, an
            entry for an unsupported modality or a disallowed family status.
            A declared key yields an entry for a disallowed status and the
            resource diagnostics. Resource checks run only when an
            environment is available, except that the row limit is checked
            whenever a row count is given or the declaration has a limit.

        Raises:
            TypeError: If `key`, `environment` or a status has the wrong type.
        """
        if not isinstance(key, Coordinate):
            raise TypeError("key must be Coordinate")
        if environment is not None and not isinstance(environment, Environment):
            raise TypeError("environment must be Environment")
        if environment is None:
            environment = self.runtime_environment
        statuses = (
            frozenset({Status.SUPPORTED})
            if allowed_statuses is None
            else frozenset(allowed_statuses)
        )
        if any(not isinstance(status, Status) for status in statuses):
            raise TypeError("allowed_statuses must contain Status")

        capability = self.find(key)
        if capability is None:
            return self.undeclared_diagnostics(key, statuses)

        diagnostics: list[Reason] = []
        if capability.status not in statuses:
            diagnostics.append(
                Reason(
                    code="status_not_allowed",
                    message=(
                        f"registered as {capability.status.value}; accepted "
                        "statuses do not include it"
                    ),
                    expected=sorted(status.value for status in statuses),
                    observed=capability.status.value,
                )
            )
        if environment is not None:
            diagnostics.extend(
                self.resource_diagnostics(capability, environment, row_count)
            )
        elif row_count is not None or capability.resources.max_rows is not None:
            diagnostics.extend(
                self.resource_diagnostics(capability, None, row_count)
            )
        return tuple(diagnostics)

    def undeclared_diagnostics(
        self, key: Coordinate, statuses: frozenset[Status]
    ) -> tuple[Reason, ...]:
        """Returns the rejections for a request that has no declaration.

        Args:
            key: Complete capability request.
            statuses: Statuses the caller accepts.

        Returns:
            A missing declaration entry plus, when the model family is
            catalogued, an entry for an unsupported modality or a disallowed
            family status.
        """
        family = self.find_model_family(key.model_family)
        family_status = None if family is None else family.status.value
        family_diagnostics = [
            Reason(
                code="missing_declaration",
                message=("no declaration exists for this complete capability"),
                expected=key.to_dict(),
                observed={"model_family_status": family_status},
            )
        ]
        if family is not None and not family.supports_modality(key.modality):
            family_diagnostics.append(
                Reason(
                    code="model_family_modality_unsupported",
                    message=(
                        "model family is not declared for the requested "
                        "modality"
                    ),
                    expected=[item.value for item in family.modalities],
                    observed=key.modality.value,
                )
            )
        if family is not None and family.status not in statuses:
            family_diagnostics.append(
                Reason(
                    code="model_family_status_not_allowed",
                    message=(
                        f"model family is registered as "
                        f"{family.status.value}; accepted statuses do "
                        "not include it"
                    ),
                    expected=sorted(status.value for status in statuses),
                    observed=family.status.value,
                )
            )
        return tuple(family_diagnostics)

    def resource_diagnostics(
        self,
        capability: Capability,
        environment: Environment | None,
        row_count: int | None = None,
    ) -> tuple[Reason, ...]:
        """Checks extras and resource assumptions without starting work.

        Args:
            capability: The declared capability.
            environment: Observed extras and resources; the bound runtime
                environment if None.
            row_count: Dataset size used to check a declared row limit.

        Returns:
            A diagnostic for each missing extra and unmet numeric, GPU or
            network assumption, which are checked only when an environment is
            available. If the capability declares a row limit, a diagnostic
            is also returned for an exceeded limit or an unknown row count.

        Raises:
            TypeError: If `capability` or `environment` has the wrong type.
        """
        if not isinstance(capability, Capability):
            raise TypeError("capability must be Capability")
        if environment is not None and not isinstance(environment, Environment):
            raise TypeError("environment must be Environment")
        if environment is None:
            environment = self.runtime_environment
        diagnostics: list[Reason] = []
        if environment is not None:
            missing_extras = sorted(
                set(capability.required_extras)
                - set(environment.available_extras)
            )
            if missing_extras:
                diagnostics.append(
                    Reason(
                        code="missing_extra",
                        message=(
                            "required optional dependency extra is unavailable"
                        ),
                        expected=missing_extras,
                        observed=sorted(environment.available_extras),
                    )
                )
            diagnostics.extend(
                self.numeric_resource_diagnostics(capability, environment)
            )
        if capability.resources.max_rows is not None:
            if row_count is None:
                diagnostics.append(
                    Reason(
                        code="unknown_row_count",
                        message=(
                            "row count is required for this capability limit"
                        ),
                        expected=capability.resources.max_rows,
                    )
                )
            elif row_count > capability.resources.max_rows:
                diagnostics.append(
                    Reason(
                        code="row_limit_exceeded",
                        message=(
                            "requested dataset exceeds capability row limit"
                        ),
                        expected=capability.resources.max_rows,
                        observed=row_count,
                    )
                )
        return tuple(diagnostics)

    def numeric_resource_diagnostics(
        self,
        capability: Capability,
        environment: Environment,
    ) -> tuple[Reason, ...]:
        """Checks CPU, memory, disk, GPU, and network assumptions.

        Args:
            capability: The declared capability.
            environment: Observed resources.

        Returns:
            A diagnostic for each assumption that is unobserved or not met.
        """
        requirements = capability.resources
        diagnostics: list[Reason] = []
        numeric_values = (
            ("cpu_cores", requirements.cpu_cores, environment.cpu_cores),
            ("memory_gb", requirements.memory_gb, environment.memory_gb),
            ("disk_gb", requirements.disk_gb, environment.disk_gb),
        )
        for name, expected, observed in numeric_values:
            if observed is None:
                diagnostics.append(
                    Reason(
                        code=f"unknown_{name}",
                        message=f"{name} is required to verify resources",
                        expected=expected,
                    )
                )
            elif observed < expected:
                diagnostics.append(
                    Reason(
                        code=f"insufficient_{name}",
                        message=f"runtime {name} is below the declared minimum",
                        expected=expected,
                        observed=observed,
                    )
                )
        boolean_values = (
            (
                "gpu_available",
                requirements.gpu_required,
                environment.gpu_available,
                "GPU",
            ),
            (
                "network_available",
                requirements.network_required,
                environment.network_available,
                "network",
            ),
        )
        for name, required, observed, label in boolean_values:
            if not required:
                continue
            if observed is None:
                diagnostics.append(
                    Reason(
                        code=f"unknown_{name}",
                        message=(
                            f"{label} availability is required to verify "
                            "resources"
                        ),
                        expected=True,
                    )
                )
            elif not observed:
                diagnostics.append(
                    Reason(
                        code=f"missing_{name}",
                        message=f"declared {label} resource is unavailable",
                        expected=True,
                        observed=False,
                    )
                )
        return tuple(diagnostics)

    def require(
        self,
        key: Coordinate,
        allowed_statuses: Collection[Status] | None = None,
        *,
        environment: Environment | None = None,
        row_count: int | None = None,
    ) -> Capability:
        """Requires a capability to be eligible before training begins.

        By default only ``supported`` declarations cross the boundary.
        Experimental or extension-only declarations may be used only when the
        caller explicitly includes their status in ``allowed_statuses``. A
        request with any diagnostic is refused with
        `UnsupportedCapabilityError` (raised by `raise_unsupported`) whose
        message is the first diagnostic. `TypeError` from `diagnose` for
        arguments of the wrong type propagates unchanged.

        Args:
            key: Complete capability request.
            allowed_statuses: Explicit statuses accepted by the caller.
            environment: Observed optional extras and runtime resources.
            row_count: Dataset size used to check a declared row limit.

        Returns:
            The registered capability declaration.

        Raises:
            ConfigError: Defensive guard that cannot trigger in practice,
                since an unregistered key always produces a diagnostic.
        """
        capability = self.find(key)
        diagnostics = self.diagnose(
            key,
            allowed_statuses,
            environment=environment,
            row_count=row_count,
        )
        if diagnostics:
            self.raise_unsupported(
                key,
                diagnostics[0].message,
                diagnostics,
            )
        if capability is None:
            raise errors.ConfigError("capability resolution failed")
        return capability

    def list_capabilities(self) -> tuple[Capability, ...]:
        """Returns declarations in deterministic key order."""
        return tuple(
            self.records[key]
            for key in sorted(
                self.records,
                key=lambda item: tuple(item.to_dict().values()),
            )
        )

    def list_model_families(self) -> tuple[ModelFamilyDeclaration, ...]:
        """Returns model-family declarations in deterministic order."""
        return tuple(
            self.model_families[name] for name in sorted(self.model_families)
        )

    def to_dict(self) -> dict[str, object]:
        """Returns a deterministic JSON-compatible registry record."""
        return {
            "schema_version": CAPABILITY_SCHEMA_VERSION,
            "model_families": [
                declaration.to_dict()
                for declaration in self.list_model_families()
            ],
            "capabilities": [
                capability.to_dict() for capability in self.list_capabilities()
            ],
        }

    def canonical_json(self) -> str:
        """Returns stable JSON suitable for hashing and release evidence."""
        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )

    def to_json(self) -> str:
        """Returns the canonical serialized registry."""
        return self.canonical_json()

    def raise_unsupported(
        self,
        key: Coordinate,
        reason: str,
        diagnostics: Iterable[Reason] = (),
    ) -> NoReturn:
        """Raises a structured pre-training rejection for ``key``.

        Args:
            key: The capability request that was refused.
            reason: Why it was refused.
            diagnostics: The diagnostics behind the refusal.

        Raises:
            UnsupportedCapabilityError: Always, with details naming the key, the
                reason, the diagnostics and the registered keys.
        """
        details = {
            "status": "unsupported",
            "reason": reason,
            "key": key.to_dict(),
            "diagnostics": [diagnostic.to_dict() for diagnostic in diagnostics],
            "registered_keys": [
                item.key.to_dict() for item in self.list_capabilities()
            ],
        }
        raise UnsupportedCapabilityError(
            f"unsupported capability before training: {reason}; "
            f"key={key.to_dict()}",
            details,
        )
