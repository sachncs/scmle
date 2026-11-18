"""Capability declarations: model families, environments and coordinates."""

import dataclasses
import math
from typing import ClassVar
from typing import TYPE_CHECKING

from scmle.contracts import errors
from scmle.contracts import records
from scmle.contracts import tasks
from scmle.contracts.capabilities import axes

if TYPE_CHECKING:
    from collections.abc import Iterable

RESOURCES = records.Migration(
    renames={
        "cpu_cores": "cpu",
        "memory_gb": "memory",
        "disk_gb": "disk",
        "gpu_required": "gpu",
        "network_required": "network",
        "max_rows": "rows",
    }
)
"""Version 2 of stored resources: Kubernetes-like single words; the units
(cores, gigabytes) are in the field descriptions."""


@dataclasses.dataclass(frozen=True, slots=True)
class Declaration:
    """Evidence boundary for a model family without full task support.

    A family declaration is intentionally not a substitute for a complete
    capability. It lets tooling answer whether a family is known, which
    modalities and optional dependencies it claims, and why it remains
    blocked before a complete task/evaluation contract exists.

    Version 2 of the stored declaration writes `family`, `extras`,
    `failures` and `links` for the fields `model_family`, `required_extras`,
    `failure_modes` and `evidence_links`; `migrations` moves a stored layout
    between the versions.
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = (
        records.nest(
            renames={
                "model_family": "family",
                "required_extras": "extras",
                "failure_modes": "failures",
                "evidence_links": "links",
            },
            parts={"resources": RESOURCES},
        ),
    )

    model_family: str
    status: axes.Status
    modalities: tuple[axes.Modality, ...]
    required_extras: tuple[str, ...]
    resources: "Resources"
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
            raise errors.KindError("model_family must be a string")
        if not self.model_family or self.model_family != (
            self.model_family.strip().lower()
        ):
            raise errors.InputError(
                "model_family must be lowercase and must not have surrounding "
                "whitespace"
            )
        if not isinstance(self.status, axes.Status):
            raise errors.KindError("status must be Status")
        if not isinstance(self.resources, Resources):
            raise errors.KindError("resources must be Resources")
        if not self.modalities:
            raise errors.InputError(
                "modalities must declare at least one value"
            )
        if any(not isinstance(item, axes.Modality) for item in self.modalities):
            raise errors.KindError("modalities must contain Modality values")
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
                raise errors.InputError(
                    f"{name} must contain non-empty strings"
                )
            object.__setattr__(self, name, tuple(sorted(set(values))))

    def to_dict(self) -> dict[str, object]:
        """Returns a stable JSON-compatible family declaration."""
        return {
            "family": self.model_family,
            "status": self.status.value,
            "modalities": [item.value for item in self.modalities],
            "extras": list(self.required_extras),
            "resources": self.resources.to_dict(),
            "failures": list(self.failure_modes),
            "links": list(self.evidence_links),
            "limitations": list(self.limitations),
        }

    def accommodates(self, modality: axes.Modality) -> bool:
        """Returns whether this family declares the requested modality.

        Args:
            modality: The modality to check.

        Returns:
            True if the family declares it.

        Raises:
            TypeError: If `modality` is not a `Modality`.
        """
        if not isinstance(modality, axes.Modality):
            raise errors.KindError("modality must be Modality")
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
            raise errors.InputError(
                "available_extras must contain non-empty strings"
            )
        object.__setattr__(
            self,
            "available_extras",
            frozenset(self.available_extras),
        )
        if self.cpu_cores is not None and self.cpu_cores < 1:
            raise errors.InputError("cpu_cores must be at least 1")
        for name, value in (
            ("memory_gb", self.memory_gb),
            ("disk_gb", self.disk_gb),
        ):
            if value is not None and (not math.isfinite(value) or value < 0):
                raise errors.InputError(
                    f"{name} must be finite and non-negative"
                )


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
        migrations: Version steps of the stored key; version 2 writes
            `model_family` as `family`.
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = (
        records.Migration(renames={"model_family": "family"}),
    )

    modality: axes.Modality
    task: tasks.Task
    model_family: str
    evaluation: axes.Evaluation
    uncertainty: axes.Uncertainty
    fairness: axes.Fairness
    explainability: axes.Explainability
    inference: axes.Inference

    def __post_init__(self) -> None:
        """Rejects non-canonical dimension values at the contract boundary.

        Raises:
            TypeError: If a dimension has the wrong enumeration type or the
                model family is not a string.
            ValueError: If the model family is empty, not lowercase or has
                surrounding whitespace.
        """
        dimensions = (
            ("modality", self.modality, axes.Modality),
            ("task", self.task, tasks.Task),
            ("evaluation", self.evaluation, axes.Evaluation),
            ("uncertainty", self.uncertainty, axes.Uncertainty),
            ("fairness", self.fairness, axes.Fairness),
            ("explainability", self.explainability, axes.Explainability),
            ("inference", self.inference, axes.Inference),
        )
        for name, value, expected in dimensions:
            if not isinstance(value, expected):
                raise errors.KindError(f"{name} must be {expected.__name__}")
        if not isinstance(self.model_family, str):
            raise errors.KindError("model_family must be a string")
        if not self.model_family:
            raise errors.InputError("model_family must not be empty")
        if self.model_family != self.model_family.strip().lower():
            raise errors.InputError(
                "model_family must be lowercase and must not have surrounding "
                "whitespace"
            )

    def to_dict(self) -> dict[str, str]:
        """Returns a stable, JSON-compatible representation of the key."""
        return {
            "modality": self.modality.value,
            "task": self.task.value,
            "family": self.model_family,
            "evaluation": self.evaluation.value,
            "uncertainty": self.uncertainty.value,
            "fairness": self.fairness.value,
            "explainability": self.explainability.value,
            "inference": self.inference.value,
        }


@dataclasses.dataclass(frozen=True, slots=True)
class Resources:
    """Minimum resources and environmental assumptions for a capability.

    Attributes:
        cpu_cores: Minimum number of CPU cores required; stored as `cpu`.
        memory_gb: Minimum working memory required, in gigabytes; stored as
            `memory`.
        disk_gb: Minimum temporary disk space required, in gigabytes; stored
            as `disk`.
        gpu_required: Whether a compatible GPU is required; stored as `gpu`.
        network_required: Whether network access is required at runtime;
            stored as `network`.
        max_rows: Practical row limit, if one is known; stored as `rows`.
        notes: Human-readable constraints that belong in release evidence.
        migrations: Version steps of the stored resources.
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = (RESOURCES,)

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
            raise errors.KindError("cpu_cores must be an integer")
        if self.cpu_cores < 1:
            raise errors.InputError("cpu_cores must be at least 1")
        if not math.isfinite(self.memory_gb) or self.memory_gb <= 0:
            raise errors.InputError("memory_gb must be greater than 0")
        if not math.isfinite(self.disk_gb) or self.disk_gb < 0:
            raise errors.InputError("disk_gb must not be negative")
        if self.max_rows is not None and self.max_rows < 1:
            raise errors.InputError("max_rows must be at least 1")
        notes: Iterable[object] = self.notes  # may hold non-strings at runtime
        if any(not isinstance(note, str) or not note.strip() for note in notes):
            raise errors.InputError("resource notes must not be empty")
        object.__setattr__(self, "notes", tuple(sorted(set(self.notes))))

    def to_dict(self) -> dict[str, object]:
        """Returns a stable, JSON-compatible representation."""
        return {
            "cpu": self.cpu_cores,
            "memory": self.memory_gb,
            "disk": self.disk_gb,
            "gpu": self.gpu_required,
            "network": self.network_required,
            "rows": self.max_rows,
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
        migrations: Version steps of the stored capability; version 2
            writes `extras`, `failures` and `links`, and the key and
            resources in their version-2 layouts.
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = (
        records.nest(
            renames={
                "required_extras": "extras",
                "failure_modes": "failures",
                "evidence_links": "links",
            },
            parts={
                "key": Coordinate.migrations[0],
                "resources": RESOURCES,
            },
        ),
    )

    key: Coordinate
    status: axes.Status
    required_extras: tuple[str, ...]
    resources: Resources
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
            raise errors.KindError("key must be Coordinate")
        if not isinstance(self.status, axes.Status):
            raise errors.KindError("status must be Status")
        if not isinstance(self.resources, Resources):
            raise errors.KindError("resources must be Resources")
        object.__setattr__(self, "required_extras", tuple(self.required_extras))
        object.__setattr__(self, "failure_modes", tuple(self.failure_modes))
        object.__setattr__(self, "evidence_links", tuple(self.evidence_links))
        object.__setattr__(self, "limitations", tuple(self.limitations))
        if not self.failure_modes:
            raise errors.InputError(
                "failure_modes must declare at least one value"
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
                raise errors.InputError(
                    f"{name} must contain non-empty strings"
                )
        if self.status is axes.Status.SUPPORTED and not self.evidence_links:
            raise errors.InputError(
                "supported capabilities must declare evidence_links"
            )

    def to_dict(self) -> dict[str, object]:
        """Returns a stable, JSON-compatible capability declaration."""
        return {
            "key": self.key.to_dict(),
            "status": self.status.value,
            "extras": list(self.required_extras),
            "resources": self.resources.to_dict(),
            "failures": list(self.failure_modes),
            "links": list(self.evidence_links),
            "limitations": list(self.limitations),
        }


DEFAULT_MODEL_FAMILY_DECLARATIONS = (
    Declaration(
        model_family=axes.ModelFamily.TRANSFORMER.value,
        status=axes.Status.EXPERIMENTAL,
        modalities=(axes.Modality.TABULAR,),
        required_extras=("deeplearning",),
        resources=Resources(memory_gb=4.0, gpu_required=True),
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
    Declaration(
        model_family=axes.ModelFamily.TEXT_TRANSFORMER.value,
        status=axes.Status.EXTENSION_ONLY,
        modalities=(axes.Modality.TEXT, axes.Modality.DOCUMENT),
        required_extras=("embeddings",),
        resources=Resources(memory_gb=4.0),
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
    Declaration(
        model_family=axes.ModelFamily.VISION_TRANSFORMER.value,
        status=axes.Status.EXTENSION_ONLY,
        modalities=(axes.Modality.IMAGE, axes.Modality.VIDEO),
        required_extras=("multimodal",),
        resources=Resources(memory_gb=8.0, gpu_required=True),
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
    Declaration(
        model_family=axes.ModelFamily.AUDIO_TRANSFORMER.value,
        status=axes.Status.EXTENSION_ONLY,
        modalities=(axes.Modality.AUDIO,),
        required_extras=("multimodal",),
        resources=Resources(memory_gb=8.0, gpu_required=True),
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
    Declaration(
        model_family=axes.ModelFamily.VIDEO_TRANSFORMER.value,
        status=axes.Status.EXTENSION_ONLY,
        modalities=(axes.Modality.VIDEO,),
        required_extras=("multimodal",),
        resources=Resources(memory_gb=12.0, gpu_required=True),
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
    Declaration(
        model_family=axes.ModelFamily.MULTIMODAL_TRANSFORMER.value,
        status=axes.Status.EXTENSION_ONLY,
        modalities=(
            axes.Modality.TEXT,
            axes.Modality.DOCUMENT,
            axes.Modality.IMAGE,
            axes.Modality.AUDIO,
            axes.Modality.VIDEO,
        ),
        required_extras=("multimodal",),
        resources=Resources(memory_gb=12.0, gpu_required=True),
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
    Declaration(
        model_family=axes.ModelFamily.JEPA.value,
        status=axes.Status.EXPERIMENTAL,
        modalities=(axes.Modality.TABULAR,),
        required_extras=("deeplearning",),
        resources=Resources(memory_gb=2.0),
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


class UnsupportedError(errors.ConfigError):
    """Raised when a requested capability cannot cross the support boundary."""

    category = "capability"
