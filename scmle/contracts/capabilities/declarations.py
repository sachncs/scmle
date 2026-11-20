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
    `failures` and `links` for the version-1 keys `model_family`,
    `required_extras`, `failure_modes` and `evidence_links`; `migrations`
    moves a stored layout
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

    family: str
    status: axes.Status
    modalities: tuple[axes.Modality, ...]
    extras: tuple[str, ...]
    resources: "Resources"
    modes: tuple[str, ...]
    links: tuple[str, ...] = ()
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
        if not isinstance(self.family, str):
            raise errors.KindError("model_family must be a string")
        if not self.family or self.family != (self.family.strip().lower()):
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
        for label, name, values in (
            ("required_extras", "extras", self.extras),
            ("failure_modes", "modes", self.modes),
            ("evidence_links", "links", self.links),
            ("limitations", "limitations", self.limitations),
        ):
            entries: Iterable[object] = (
                values  # may hold non-strings at runtime
            )
            if any(
                not isinstance(value, str) or not value.strip()
                for value in entries
            ):
                raise errors.InputError(
                    f"{label} must contain non-empty strings"
                )
            object.__setattr__(self, name, tuple(sorted(set(values))))

    def to_dict(self) -> dict[str, object]:
        """Returns a stable JSON-compatible family declaration."""
        return {
            "family": self.family,
            "status": self.status.value,
            "modalities": [item.value for item in self.modalities],
            "extras": list(self.extras),
            "resources": self.resources.to_dict(),
            "failures": list(self.modes),
            "links": list(self.links),
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

    extras: frozenset[str] = frozenset()
    cores: int | None = None
    memory: float | None = None
    disk: float | None = None
    available: bool | None = None
    online: bool | None = None

    def __post_init__(self) -> None:
        """Validates runtime observations before they are used for gating.

        Raises:
            ValueError: If an available extra is blank, the CPU count is below
                one, or the memory or disk size is not finite and non-negative.
        """
        extras: Iterable[object] = self.extras  # may hold non-strings
        if any(
            not isinstance(extra, str) or not extra.strip() for extra in extras
        ):
            raise errors.InputError(
                "available_extras must contain non-empty strings"
            )
        object.__setattr__(
            self,
            "extras",
            frozenset(self.extras),
        )
        if self.cores is not None and self.cores < 1:
            raise errors.InputError("cpu_cores must be at least 1")
        for name, value in (
            ("memory_gb", self.memory),
            ("disk_gb", self.disk),
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
        family: Lowercase family identifier, not an adapter name.
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
    family: str
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
        if not isinstance(self.family, str):
            raise errors.KindError("model_family must be a string")
        if not self.family:
            raise errors.InputError("model_family must not be empty")
        if self.family != self.family.strip().lower():
            raise errors.InputError(
                "model_family must be lowercase and must not have surrounding "
                "whitespace"
            )

    def to_dict(self) -> dict[str, str]:
        """Returns a stable, JSON-compatible representation of the key."""
        return {
            "modality": self.modality.value,
            "task": self.task.value,
            "family": self.family,
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
        cores: Minimum number of CPU cores required; stored as `cpu`.
        memory: Minimum working memory required, in gigabytes; stored as
            `memory`.
        disk: Minimum temporary disk space required, in gigabytes; stored
            as `disk`.
        required: Whether a compatible GPU is required; stored as `gpu`.
        network: Whether network access is required at runtime;
            stored as `network`.
        rows: Practical row limit, if one is known; stored as `rows`.
        notes: Human-readable constraints that belong in release evidence.
        migrations: Version steps of the stored resources.
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = (RESOURCES,)

    cores: int = 1
    memory: float = 1.0
    disk: float = 0.0
    required: bool = False
    network: bool = False
    rows: int | None = None
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validates resource assumptions before they enter the registry.

        Raises:
            TypeError: If the CPU count is not an integer.
            ValueError: If the CPU count is below one, the memory is not finite
                and positive, the disk size is not finite and non-negative, the
                row limit is below one, or a note is blank.
        """
        if not isinstance(self.cores, int):
            raise errors.KindError("cpu_cores must be an integer")
        if self.cores < 1:
            raise errors.InputError("cpu_cores must be at least 1")
        if not math.isfinite(self.memory) or self.memory <= 0:
            raise errors.InputError("memory_gb must be greater than 0")
        if not math.isfinite(self.disk) or self.disk < 0:
            raise errors.InputError("disk_gb must not be negative")
        if self.rows is not None and self.rows < 1:
            raise errors.InputError("max_rows must be at least 1")
        notes: Iterable[object] = self.notes  # may hold non-strings at runtime
        if any(not isinstance(note, str) or not note.strip() for note in notes):
            raise errors.InputError("resource notes must not be empty")
        object.__setattr__(self, "notes", tuple(sorted(set(self.notes))))

    def to_dict(self) -> dict[str, object]:
        """Returns a stable, JSON-compatible representation."""
        return {
            "cpu": self.cores,
            "memory": self.memory,
            "disk": self.disk,
            "gpu": self.required,
            "network": self.network,
            "rows": self.rows,
            "notes": list(self.notes),
        }


@dataclasses.dataclass(frozen=True, slots=True)
class Capability:
    """Declared support and evidence boundary for one capability key.

    Attributes:
        key: The complete eight-dimension capability coordinate.
        status: Supported, experimental, or extension-only.
        extras: Optional package extras needed to use the capability.
        resources: Minimum resource and environment assumptions.
        modes: Known conditions that must be handled or reported.
        links: References to tests or evidence artifacts.
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
    extras: tuple[str, ...]
    resources: Resources
    modes: tuple[str, ...]
    links: tuple[str, ...] = ()
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
        object.__setattr__(self, "extras", tuple(self.extras))
        object.__setattr__(self, "modes", tuple(self.modes))
        object.__setattr__(self, "links", tuple(self.links))
        object.__setattr__(self, "limitations", tuple(self.limitations))
        if not self.modes:
            raise errors.InputError(
                "failure_modes must declare at least one value"
            )
        for name, values in (
            ("required_extras", self.extras),
            ("failure_modes", self.modes),
            ("evidence_links", self.links),
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
        if self.status is axes.Status.SUPPORTED and not self.links:
            raise errors.InputError(
                "supported capabilities must declare evidence_links"
            )

    def to_dict(self) -> dict[str, object]:
        """Returns a stable, JSON-compatible capability declaration."""
        return {
            "key": self.key.to_dict(),
            "status": self.status.value,
            "extras": list(self.extras),
            "resources": self.resources.to_dict(),
            "failures": list(self.modes),
            "links": list(self.links),
            "limitations": list(self.limitations),
        }


DEFAULT_MODEL_FAMILY_DECLARATIONS = (
    Declaration(
        family=axes.Family.TRANSFORMER.value,
        status=axes.Status.EXPERIMENTAL,
        modalities=(axes.Modality.TABULAR,),
        extras=("deeplearning",),
        resources=Resources(memory=4.0, required=True),
        modes=(
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
        family=axes.Family.TEXT_TRANSFORMER.value,
        status=axes.Status.EXTENSION_ONLY,
        modalities=(axes.Modality.TEXT, axes.Modality.DOCUMENT),
        extras=("embeddings",),
        resources=Resources(memory=4.0),
        modes=(
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
        family=axes.Family.VISION_TRANSFORMER.value,
        status=axes.Status.EXTENSION_ONLY,
        modalities=(axes.Modality.IMAGE, axes.Modality.VIDEO),
        extras=("multimodal",),
        resources=Resources(memory=8.0, required=True),
        modes=(
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
        family=axes.Family.AUDIO_TRANSFORMER.value,
        status=axes.Status.EXTENSION_ONLY,
        modalities=(axes.Modality.AUDIO,),
        extras=("multimodal",),
        resources=Resources(memory=8.0, required=True),
        modes=(
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
        family=axes.Family.VIDEO_TRANSFORMER.value,
        status=axes.Status.EXTENSION_ONLY,
        modalities=(axes.Modality.VIDEO,),
        extras=("multimodal",),
        resources=Resources(memory=12.0, required=True),
        modes=(
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
        family=axes.Family.MULTIMODAL_TRANSFORMER.value,
        status=axes.Status.EXTENSION_ONLY,
        modalities=(
            axes.Modality.TEXT,
            axes.Modality.DOCUMENT,
            axes.Modality.IMAGE,
            axes.Modality.AUDIO,
            axes.Modality.VIDEO,
        ),
        extras=("multimodal",),
        resources=Resources(memory=12.0, required=True),
        modes=(
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
        family=axes.Family.JEPA.value,
        status=axes.Status.EXPERIMENTAL,
        modalities=(axes.Modality.TABULAR,),
        extras=("deeplearning",),
        resources=Resources(memory=2.0),
        modes=(
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
