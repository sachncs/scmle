"""Capability contracts for pre-training support decisions.

The registry is deliberately independent from model adapters. A planner can
ask whether a complete capability is eligible before it constructs a model,
loads an optional dependency, or starts training.
"""

from collections.abc import Collection
from collections.abc import Iterable
import dataclasses
import enum
from typing import NoReturn

from scml.contracts import errors
from scml.contracts import tasks


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


class CapabilityStatus(enum.StrEnum):
    """Evidence level assigned to a registered capability."""

    SUPPORTED = "supported"
    EXPERIMENTAL = "experimental"
    EXTENSION_ONLY = "extension-only"


@dataclasses.dataclass(frozen=True, slots=True)
class CapabilityKey:
    """Coordinates of one complete pre-training capability decision.

    Attributes:
        modality: Input data modality.
        task: Existing framework task contract.
        model_family: Lowercase family identifier, not an adapter name.
        evaluation: Evaluation policy that must be available.
        uncertainty: Uncertainty contract that must be available.
        fairness: Fairness contract that must be available.
        explainability: Explanation contract that must be available.
        inference: Inference surface that must be available.
    """

    modality: Modality
    task: tasks.TaskType
    model_family: str
    evaluation: Evaluation
    uncertainty: Uncertainty
    fairness: Fairness
    explainability: Explainability
    inference: Inference

    def __post_init__(self) -> None:
        """Reject non-canonical dimension values at the contract boundary."""
        dimensions = (
            ("modality", self.modality, Modality),
            ("task", self.task, tasks.TaskType),
            ("evaluation", self.evaluation, Evaluation),
            ("uncertainty", self.uncertainty, Uncertainty),
            ("fairness", self.fairness, Fairness),
            ("explainability", self.explainability, Explainability),
            ("inference", self.inference, Inference),
        )
        for name, value, expected in dimensions:
            if not isinstance(value, expected):
                raise TypeError(f"{name} must be {expected.__name__}")
        if not self.model_family:
            raise ValueError("model_family must not be empty")
        if self.model_family != self.model_family.strip().lower():
            raise ValueError(
                "model_family must be lowercase and must not have surrounding "
                "whitespace"
            )

    def to_dict(self) -> dict[str, str]:
        """Return a stable, JSON-compatible representation of the key."""
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
        """Validate resource assumptions before they enter the registry."""
        if self.cpu_cores < 1:
            raise ValueError("cpu_cores must be at least 1")
        if self.memory_gb <= 0:
            raise ValueError("memory_gb must be greater than 0")
        if self.disk_gb < 0:
            raise ValueError("disk_gb must not be negative")
        if self.max_rows is not None and self.max_rows < 1:
            raise ValueError("max_rows must be at least 1")
        if any(not note.strip() for note in self.notes):
            raise ValueError("resource notes must not be empty")

    def to_dict(self) -> dict[str, object]:
        """Return a stable, JSON-compatible representation."""
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

    key: CapabilityKey
    status: CapabilityStatus
    required_extras: tuple[str, ...]
    resources: ResourceAssumptions
    failure_modes: tuple[str, ...]
    evidence_links: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Require complete declarations rather than optimistic defaults."""
        if not isinstance(self.status, CapabilityStatus):
            raise TypeError("status must be CapabilityStatus")
        if not self.failure_modes:
            raise ValueError("failure_modes must declare at least one value")
        for name, values in (
            ("required_extras", self.required_extras),
            ("failure_modes", self.failure_modes),
            ("evidence_links", self.evidence_links),
            ("limitations", self.limitations),
        ):
            if any(not value.strip() for value in values):
                raise ValueError(f"{name} must not contain empty values")
        if (
            self.status is CapabilityStatus.SUPPORTED
            and not self.evidence_links
        ):
            raise ValueError(
                "supported capabilities must declare evidence_links"
            )

    def to_dict(self) -> dict[str, object]:
        """Return a stable, JSON-compatible capability declaration."""
        return {
            "key": self.key.to_dict(),
            "status": self.status.value,
            "required_extras": list(self.required_extras),
            "resources": self.resources.to_dict(),
            "failure_modes": list(self.failure_modes),
            "evidence_links": list(self.evidence_links),
            "limitations": list(self.limitations),
        }


class UnsupportedCapabilityError(errors.ConfigError):
    """Raised when a requested capability cannot cross the support boundary."""

    category = "capability"


class CapabilityRegistry:
    """Registry that resolves complete capabilities before training.

    The registry has no implicit defaults. Callers must register an explicit
    declaration for every combination they intend to train, which prevents a
    model adapter's existence from being mistaken for end-to-end support.

    Attributes:
        records: Registered capabilities keyed by their complete coordinate.
    """

    def __init__(self, capabilities: Iterable[Capability] = ()) -> None:
        """Create a registry and register the supplied declarations."""
        self.records: dict[CapabilityKey, Capability] = {}
        for capability in capabilities:
            self.register(capability)

    def register(self, capability: Capability) -> None:
        """Register a capability without silently replacing an existing one.

        Args:
            capability: Complete capability declaration.

        Raises:
            errors.ConfigError: If the key is already registered.
        """
        if capability.key in self.records:
            raise errors.ConfigError(
                f"capability already registered: {capability.key.to_dict()}"
            )
        self.records[capability.key] = capability

    def find(self, key: CapabilityKey) -> Capability | None:
        """Return a declaration or ``None`` when the key is unregistered."""
        return self.records.get(key)

    def require(
        self,
        key: CapabilityKey,
        allowed_statuses: Collection[CapabilityStatus] | None = None,
    ) -> Capability:
        """Require a capability to be eligible before training begins.

        By default only ``supported`` declarations cross the boundary.
        Experimental or extension-only declarations may be used only when the
        caller explicitly includes their status in ``allowed_statuses``.

        Args:
            key: Complete capability request.
            allowed_statuses: Explicit statuses accepted by the caller.

        Returns:
            The registered capability declaration.

        Raises:
            UnsupportedCapabilityError: If the key is absent or its status is
                not allowed.
        """
        capability = self.find(key)
        statuses = (
            frozenset({CapabilityStatus.SUPPORTED})
            if allowed_statuses is None
            else frozenset(allowed_statuses)
        )
        if capability is None:
            self.raise_unsupported(
                key,
                "no declaration exists for this complete capability",
            )
        if capability.status not in statuses:
            self.raise_unsupported(
                key,
                f"registered as {capability.status.value}; accepted statuses "
                f"are {sorted(status.value for status in statuses)}",
            )
        return capability

    def list_capabilities(self) -> tuple[Capability, ...]:
        """Return declarations in deterministic key order."""
        return tuple(
            self.records[key]
            for key in sorted(
                self.records,
                key=lambda item: tuple(item.to_dict().values()),
            )
        )

    def raise_unsupported(self, key: CapabilityKey, reason: str) -> NoReturn:
        """Raise a structured pre-training rejection for ``key``."""
        details = {
            "status": "unsupported",
            "reason": reason,
            "key": key.to_dict(),
            "registered_keys": [
                item.key.to_dict() for item in self.list_capabilities()
            ],
        }
        raise UnsupportedCapabilityError(
            f"unsupported capability before training: {reason}; "
            f"key={key.to_dict()}",
            details,
        )
