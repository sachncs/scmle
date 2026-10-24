"""Immutable image samples, bounded profiling, and baseline features.

Pillow is intentionally optional. Importing this module remains safe without
it, while every operation that needs image decoding raises
``ImageDependencyError`` with an actionable, structured error envelope.
"""

from collections.abc import Iterable
from collections.abc import Mapping
import dataclasses
import hashlib
import io
import json
import pathlib
import random
from types import MappingProxyType
from typing import Any
from typing import Self

import numpy as np

from scmle.contracts import data as data_contracts
from scmle.contracts import errors
from scmle.data.modalities.curation import build_finding
from scmle.data.modalities.curation import conflicting_label_groups
from scmle.data.modalities.curation import dataset_hash
from scmle.data.modalities.curation import label_counts
from scmle.data.modalities.curation import ModalityCurationReport

PillowModule: Any = None
try:
    from PIL import Image as ImportedImage
except ImportError:
    pass
else:
    PillowModule = ImportedImage


PILLOW_AVAILABLE = PillowModule is not None
DEFAULT_MAX_BYTES = 50 * 1024 * 1024
DEFAULT_MAX_PIXELS = 25_000_000
DEFAULT_PROFILE_SIZE = (16, 16)
SUPPORTED_FEATURE_MODES = ("L", "RGB", "RGBA")
SENSITIVE_METADATA_TAGS = {
    271: "camera_make",
    272: "camera_model",
    305: "software",
    315: "artist",
    33432: "copyright",
    34853: "gps_info",
    36867: "date_time_original",
}


class ImageError(errors.DataError):
    """Base error for image ingestion and feature preparation."""

    category = "image"

    def __init__(
        self,
        message: str,
        code: str,
        source: str = "",
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initializes an image error with machine-readable context."""
        context = {"code": code}
        if source:
            context["source"] = source
        if details:
            context.update(details)
        super().__init__(message, context)
        self.code = code
        self.source = source


class ImageDependencyError(ImageError):
    """Raised when Pillow is required but is not installed."""

    category = "dependency"

    def __init__(self, operation: str) -> None:
        """Initializes a dependency error for ``operation``."""
        super().__init__(
            "image operation requires the optional 'Pillow' dependency; "
            "install it with 'pip install Pillow'",
            code="missing_dependency",
            details={
                "dependency": "Pillow",
                "operation": operation,
                "install": "pip install Pillow",
            },
        )


class ImageValidationError(ImageError):
    """Raised when an image violates a safety or validity constraint."""


def require_pillow(operation: str = "image data") -> Any:
    """Returns the optional Pillow module or raises a structured error.

    Args:
        operation: Operation that requires image decoding.

    Returns:
        The imported Pillow image module.

    Raises:
        ImageDependencyError: If Pillow is not installed.
    """
    if not PILLOW_AVAILABLE or PillowModule is None:
        raise ImageDependencyError(operation)
    return PillowModule


def canonical_source(source: str | pathlib.Path) -> str:
    """Returns a stable source identifier for a path or URI-like value."""
    value = str(source).strip()
    if not value:
        return "memory://"
    if "://" in value:
        return value
    return pathlib.Path(value).expanduser().resolve(strict=False).as_uri()


def validate_feature_options(
    target_size: tuple[int, int], mode: str
) -> tuple[int, int]:
    """Validates the deterministic baseline feature configuration."""
    if len(target_size) != 2 or any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0
        for value in target_size
    ):
        raise ValueError("target_size must contain two positive integers")
    if mode not in SUPPORTED_FEATURE_MODES:
        raise ValueError(
            f"mode must be one of {SUPPORTED_FEATURE_MODES}, got {mode!r}"
        )
    return target_size


def image_content_hash(
    image: Any,
    width: int,
    height: int,
) -> str:
    """Returns a format-independent hash of decoded RGBA content."""
    canonical = image.convert("RGBA")
    payload = b"scmle-image-content-v1\0"
    payload += f"{width}x{height}\0".encode("ascii")
    payload += canonical.tobytes()
    return hashlib.sha256(payload).hexdigest()


@dataclasses.dataclass(frozen=True, slots=True)
class ImageAugmentation:
    """Caller-declared provenance for a transformed image sample."""

    parent_source: str
    operation: str
    parameters: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    tool: str = ""

    def __post_init__(self) -> None:
        """Validate and normalize transformation provenance."""
        parent = canonical_source(self.parent_source)
        operation = self.operation.strip()
        tool = self.tool.strip()
        if not operation:
            raise ValueError("augmentation operation must not be empty")
        if any(
            not isinstance(key, str) or not key.strip()
            for key in self.parameters
        ):
            raise ValueError(
                "augmentation parameter keys must be non-empty text"
            )
        try:
            normalized = json.loads(
                json.dumps(
                    dict(self.parameters), allow_nan=False, sort_keys=True
                )
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                "augmentation parameters must be JSON-native"
            ) from error
        object.__setattr__(self, "parent_source", parent)
        object.__setattr__(self, "operation", operation)
        object.__setattr__(self, "parameters", MappingProxyType(normalized))
        object.__setattr__(self, "tool", tool)

    @property
    def identity(self) -> str:
        """Return a deterministic identity for the transformation record."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-native transformation provenance."""
        return {
            "parent_source": self.parent_source,
            "operation": self.operation,
            "parameters": dict(self.parameters),
            "tool": self.tool,
        }


@dataclasses.dataclass(frozen=True, slots=True)
class ImageSample:
    """An immutable, validated image snapshot.

    The encoded bytes are retained so a sample remains tied to the identities
    and metadata that were validated at ingestion time. ``file_hash`` hashes
    the encoded source bytes; ``content_hash`` hashes canonical decoded RGBA
    pixels and therefore supports content-based deduplication across formats.
    """

    source: str
    encoded_bytes: bytes = dataclasses.field(repr=False)
    file_hash: str
    content_hash: str
    width: int
    height: int
    channels: int
    mode: str
    format: str
    byte_size: int
    label: str | int | float | bool | None = None
    group_id: str | None = None
    exif_tags: tuple[str, ...] = ()
    sensitive_metadata: tuple[str, ...] = ()
    augmentation: ImageAugmentation | None = None

    def __post_init__(self) -> None:
        """Validates immutable identity and geometry metadata."""
        object.__setattr__(self, "encoded_bytes", bytes(self.encoded_bytes))
        if not self.source.strip():
            raise ValueError("source must not be empty")
        if self.group_id is not None and not self.group_id.strip():
            raise ValueError("group_id must not be empty when provided")
        object.__setattr__(self, "exif_tags", tuple(self.exif_tags))
        object.__setattr__(
            self, "sensitive_metadata", tuple(self.sensitive_metadata)
        )
        if self.augmentation is not None and not isinstance(
            self.augmentation, ImageAugmentation
        ):
            raise TypeError("augmentation must be ImageAugmentation or None")
        if len(self.encoded_bytes) != self.byte_size:
            raise ValueError("byte_size does not match encoded_bytes")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("image dimensions must be positive")
        if self.channels <= 0:
            raise ValueError("channels must be positive")
        if len(self.file_hash) != 64 or len(self.content_hash) != 64:
            raise ValueError("image hashes must be SHA-256 hex digests")

    @classmethod
    def from_bytes(
        cls,
        encoded_bytes: bytes,
        source: str = "memory://",
        max_bytes: int | None = DEFAULT_MAX_BYTES,
        max_pixels: int | None = DEFAULT_MAX_PIXELS,
        label: str | int | float | bool | None = None,
        group_id: str | None = None,
        augmentation: ImageAugmentation | None = None,
    ) -> Self:
        """Builds a validated image sample from encoded bytes.

        Args:
            encoded_bytes: Complete encoded image payload.
            source: Stable source label or URI for lineage.
            max_bytes: Maximum encoded payload size, or ``None`` for no limit.
            max_pixels: Maximum decoded pixel count, or ``None`` for no limit.
            label: Optional task label retained for curation evidence.
            group_id: Optional identity used to prevent split leakage.
            augmentation: Optional caller-declared transformation provenance.

        Returns:
            A validated immutable image snapshot.

        Raises:
            ImageDependencyError: If Pillow is unavailable.
            ImageValidationError: If the payload is empty, corrupt, or too
                large.
        """
        require_pillow("image ingestion")
        raw = bytes(encoded_bytes)
        stable_source = canonical_source(source)
        cls.validate_limits(max_bytes, max_pixels)
        if not raw:
            raise ImageValidationError(
                "image payload is empty",
                code="empty_payload",
                source=stable_source,
            )
        if max_bytes is not None and len(raw) > max_bytes:
            raise ImageValidationError(
                f"image payload is {len(raw)} bytes; limit is {max_bytes}",
                code="encoded_size_limit",
                source=stable_source,
                details={"byte_size": len(raw), "max_bytes": max_bytes},
            )
        try:
            with PillowModule.open(io.BytesIO(raw)) as opened:
                opened.verify()
            with PillowModule.open(io.BytesIO(raw)) as opened:
                width, height = opened.size
                if width <= 0 or height <= 0:
                    raise ImageValidationError(
                        "image dimensions must be positive",
                        code="empty_dimensions",
                        source=stable_source,
                    )
                pixel_count = width * height
                if max_pixels is not None and pixel_count > max_pixels:
                    raise ImageValidationError(
                        f"image has {pixel_count} pixels; "
                        f"limit is {max_pixels}",
                        code="pixel_count_limit",
                        source=stable_source,
                        details={
                            "pixel_count": pixel_count,
                            "max_pixels": max_pixels,
                        },
                    )
                mode = opened.mode
                channels = len(opened.getbands())
                format_name = opened.format or "unknown"
                opened.load()
                content_hash = image_content_hash(opened, width, height)
                exif_tags = tuple(sorted(str(tag) for tag in opened.getexif()))
                sensitive_metadata = tuple(
                    sorted(
                        SENSITIVE_METADATA_TAGS[tag]
                        for tag in opened.getexif()
                        if tag in SENSITIVE_METADATA_TAGS
                    )
                )
        except ImageValidationError:
            raise
        except Exception as error:
            raise ImageValidationError(
                f"could not decode image: {error}",
                code="corrupt_payload",
                source=stable_source,
                details={"exception": type(error).__name__},
            ) from error
        return cls(
            source=stable_source,
            encoded_bytes=raw,
            file_hash=hashlib.sha256(raw).hexdigest(),
            content_hash=content_hash,
            width=width,
            height=height,
            channels=channels,
            mode=mode,
            format=format_name,
            byte_size=len(raw),
            label=label,
            group_id=group_id,
            exif_tags=exif_tags,
            sensitive_metadata=sensitive_metadata,
            augmentation=augmentation,
        )

    @classmethod
    def from_path(
        cls,
        path: str | pathlib.Path,
        max_bytes: int | None = DEFAULT_MAX_BYTES,
        max_pixels: int | None = DEFAULT_MAX_PIXELS,
        label: str | int | float | bool | None = None,
        group_id: str | None = None,
        augmentation: ImageAugmentation | None = None,
    ) -> Self:
        """Builds a validated image sample from a local file."""
        file_path = pathlib.Path(path).expanduser()
        source = canonical_source(file_path)
        if not file_path.exists():
            raise ImageValidationError(
                "image file was not found",
                code="missing_file",
                source=source,
            )
        if not file_path.is_file():
            raise ImageValidationError(
                "image source is not a regular file",
                code="not_a_file",
                source=source,
            )
        cls.validate_limits(max_bytes, max_pixels)
        file_size = file_path.stat().st_size
        if max_bytes is not None and file_size > max_bytes:
            raise ImageValidationError(
                f"image file is {file_size} bytes; limit is {max_bytes}",
                code="encoded_size_limit",
                source=source,
                details={"byte_size": file_size, "max_bytes": max_bytes},
            )
        try:
            payload = file_path.read_bytes()
        except OSError as error:
            raise ImageValidationError(
                f"could not read image file: {error}",
                code="read_failure",
                source=source,
            ) from error
        return cls.from_bytes(
            payload,
            source,
            max_bytes,
            max_pixels,
            label,
            group_id,
            augmentation,
        )

    def perceptual_hash(self, hash_size: int = 8) -> str:
        """Return a bounded luminance signature for near-duplicate review."""
        if (
            not isinstance(hash_size, int)
            or isinstance(hash_size, bool)
            or not 2 <= hash_size <= 32
        ):
            raise ValueError("hash_size must be an integer between 2 and 32")
        pixels = self.decode("L", (hash_size, hash_size))[..., 0]
        quantized = np.asarray(pixels, dtype=np.uint8) // 16
        return "".join(f"{int(value):x}" for value in quantized.flat)

    @staticmethod
    def validate_limits(
        max_bytes: int | None,
        max_pixels: int | None,
    ) -> None:
        """Validates optional resource limits before reading or decoding."""
        for name, value in (
            ("max_bytes", max_bytes),
            ("max_pixels", max_pixels),
        ):
            if value is not None and (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer or None")

    def decode(
        self,
        mode: str | None = None,
        target_size: tuple[int, int] | None = None,
    ) -> np.ndarray:
        """Decodes pixels into a deterministic ``uint8`` NumPy array.

        Args:
            mode: Optional Pillow mode, limited to ``L``, ``RGB`` or ``RGBA``.
            target_size: Optional ``(width, height)`` resize target.

        Returns:
            Array shaped ``(height, width, channels)``.
        """
        image_library = require_pillow("image decoding")
        selected_mode = mode or self.mode
        if selected_mode not in SUPPORTED_FEATURE_MODES:
            raise ValueError(
                f"mode must be one of {SUPPORTED_FEATURE_MODES}, got "
                f"{selected_mode!r}"
            )
        if target_size is not None:
            validate_feature_options(target_size, selected_mode)
        try:
            with image_library.open(io.BytesIO(self.encoded_bytes)) as opened:
                converted = opened.convert(selected_mode)
                if target_size is not None:
                    resampling = image_library.Resampling.BILINEAR
                    converted = converted.resize(target_size, resampling)
                pixels = np.asarray(converted, dtype=np.uint8)
        except Exception as error:
            raise ImageValidationError(
                f"could not decode validated image: {error}",
                code="decode_failure",
                source=self.source,
                details={"exception": type(error).__name__},
            ) from error
        if pixels.ndim == 2:
            pixels = pixels[..., np.newaxis]
        return np.ascontiguousarray(pixels)

    def to_dict(self) -> dict[str, Any]:
        """Returns JSON-compatible metadata without embedding image bytes."""
        return {
            "source": self.source,
            "file_hash": self.file_hash,
            "content_hash": self.content_hash,
            "width": self.width,
            "height": self.height,
            "channels": self.channels,
            "mode": self.mode,
            "format": self.format,
            "byte_size": self.byte_size,
            "label": self.label,
            "group_id": self.group_id,
            "exif_tags": list(self.exif_tags),
            "sensitive_metadata": list(self.sensitive_metadata),
            "augmentation": (
                self.augmentation.to_dict()
                if self.augmentation is not None
                else None
            ),
        }


@dataclasses.dataclass(frozen=True, slots=True)
class ImageProfile:
    """Bounded, deterministic summary of an image dataset."""

    dataset_count: int
    profiled_count: int
    truncated: bool
    profiled_bytes: int
    unique_content_count: int
    duplicate_count: int
    modes: tuple[tuple[str, int], ...]
    channels: tuple[tuple[int, int], ...]
    width_min: int
    width_max: int
    width_mean: float
    height_min: int
    height_max: int
    height_mean: float
    mean_pixel: tuple[float, ...]
    pixel_std: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        """Returns JSON-compatible profile data."""
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, slots=True)
class ImageSplit:
    """Group-aware image split that keeps related samples together."""

    train: tuple[ImageSample, ...]
    validation: tuple[ImageSample, ...]
    test: tuple[ImageSample, ...]

    def group_ids(self) -> dict[str, tuple[str, ...]]:
        """Return the stable groups assigned to each partition."""
        return {
            name: tuple(
                sorted({sample.group_id or sample.source for sample in samples})
            )
            for name, samples in (
                ("train", self.train),
                ("validation", self.validation),
                ("test", self.test),
            )
        }


@dataclasses.dataclass(frozen=True, slots=True)
class ImageDataset:
    """Immutable collection of validated image samples."""

    samples: tuple[ImageSample, ...] = ()
    content_hash: str = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        """Normalizes samples to an immutable tuple and validates types."""
        normalized = tuple(self.samples)
        if any(not isinstance(sample, ImageSample) for sample in normalized):
            raise TypeError("samples must contain only ImageSample objects")
        object.__setattr__(self, "samples", normalized)
        identities = (
            f"{sample.source}\0{sample.content_hash}"
            + (
                f"\0augmentation={sample.augmentation.identity}"
                if sample.augmentation is not None
                else ""
            )
            for sample in normalized
        )
        object.__setattr__(self, "content_hash", dataset_hash(identities))

    @classmethod
    def from_paths(
        cls,
        paths: Iterable[str | pathlib.Path],
        max_bytes: int | None = DEFAULT_MAX_BYTES,
        max_pixels: int | None = DEFAULT_MAX_PIXELS,
        deduplicate: bool = False,
    ) -> Self:
        """Builds a dataset from local paths in caller-provided order."""
        dataset = cls(
            tuple(
                ImageSample.from_path(path, max_bytes, max_pixels)
                for path in paths
            )
        )
        return dataset.deduplicate() if deduplicate else dataset

    @property
    def sample_count(self) -> int:
        """Returns the number of samples in the dataset."""
        return len(self.samples)

    def deduplicate(self) -> Self:
        """Returns a content-deduplicated dataset with stable ordering.

        For each content identity, the lexicographically smallest source is
        retained. Sorting by content and source makes the result independent
        of input iteration order.
        """
        selected: dict[str, ImageSample] = {}
        for sample in self.samples:
            previous = selected.get(sample.content_hash)
            if previous is None or (
                sample.source,
                sample.file_hash,
            ) < (previous.source, previous.file_hash):
                selected[sample.content_hash] = sample
        ordered = tuple(
            selected[key]
            for key in sorted(
                selected,
                key=lambda content: (
                    content,
                    selected[content].source,
                ),
            )
        )
        return type(self)(ordered)

    def near_duplicate_groups(
        self,
        max_samples: int = 1024,
        hash_size: int = 8,
        max_hamming_distance: int = 8,
    ) -> tuple[tuple[str, ...], ...]:
        """Return bounded groups of visually similar, distinct images.

        The signature is a quantized, resized luminance representation. It is
        review evidence for likely near duplicates, not a semantic similarity
        model or a proof that two images depict the same object.
        """
        if not isinstance(max_samples, int) or isinstance(max_samples, bool):
            raise ValueError("max_samples must be a positive integer")
        if max_samples <= 0:
            raise ValueError("max_samples must be a positive integer")
        if (
            not isinstance(max_hamming_distance, int)
            or isinstance(max_hamming_distance, bool)
            or max_hamming_distance < 0
        ):
            raise ValueError(
                "max_hamming_distance must be a non-negative integer"
            )
        candidates = self.samples[:max_samples]
        signatures = [
            int(sample.perceptual_hash(hash_size), 16) for sample in candidates
        ]
        neighbors: dict[int, set[int]] = {
            index: set() for index in range(len(candidates))
        }
        for left in range(len(candidates)):
            for right in range(left + 1, len(candidates)):
                if (
                    signatures[left] ^ signatures[right]
                ).bit_count() <= max_hamming_distance:
                    neighbors[left].add(right)
                    neighbors[right].add(left)
        groups = []
        visited: set[int] = set()
        for start in range(len(candidates)):
            if start in visited or not neighbors[start]:
                continue
            pending = [start]
            component: set[int] = set()
            while pending:
                current = pending.pop()
                if current in component:
                    continue
                component.add(current)
                visited.add(current)
                pending.extend(neighbors[current] - component)
            if len({candidates[index].content_hash for index in component}) > 1:
                groups.append(
                    tuple(
                        sorted(candidates[index].source for index in component)
                    )
                )
        return tuple(sorted(groups))

    def profile(
        self,
        max_samples: int = 1024,
        profile_size: tuple[int, int] = DEFAULT_PROFILE_SIZE,
    ) -> ImageProfile:
        """Profiles at most ``max_samples`` images and bounded pixels."""
        if not isinstance(max_samples, int) or isinstance(max_samples, bool):
            raise ValueError("max_samples must be a positive integer")
        if max_samples <= 0:
            raise ValueError("max_samples must be a positive integer")
        validate_feature_options(profile_size, "RGB")
        profiled = self.samples[:max_samples]
        if not profiled:
            return ImageProfile(
                dataset_count=0,
                profiled_count=0,
                truncated=False,
                profiled_bytes=0,
                unique_content_count=0,
                duplicate_count=0,
                modes=(),
                channels=(),
                width_min=0,
                width_max=0,
                width_mean=0.0,
                height_min=0,
                height_max=0,
                height_mean=0.0,
                mean_pixel=(0.0, 0.0, 0.0),
                pixel_std=(0.0, 0.0, 0.0),
            )
        means: list[np.ndarray] = []
        variances: list[np.ndarray] = []
        for sample in profiled:
            pixels = sample.decode("RGB", profile_size).astype(np.float64)
            means.append(pixels.mean(axis=(0, 1)))
            variances.append(pixels.var(axis=(0, 1)))
        mean = np.mean(np.stack(means), axis=0)
        variance = np.mean(np.stack(variances), axis=0)
        modes = tuple(
            sorted(
                (mode, sum(sample.mode == mode for sample in profiled))
                for mode in {sample.mode for sample in profiled}
            )
        )
        channels = tuple(
            sorted(
                (
                    channel,
                    sum(sample.channels == channel for sample in profiled),
                )
                for channel in {sample.channels for sample in profiled}
            )
        )
        unique_count = len({sample.content_hash for sample in profiled})
        return ImageProfile(
            dataset_count=len(self.samples),
            profiled_count=len(profiled),
            truncated=len(profiled) < len(self.samples),
            profiled_bytes=sum(sample.byte_size for sample in profiled),
            unique_content_count=unique_count,
            duplicate_count=len(profiled) - unique_count,
            modes=modes,
            channels=channels,
            width_min=min(sample.width for sample in profiled),
            width_max=max(sample.width for sample in profiled),
            width_mean=float(np.mean([sample.width for sample in profiled])),
            height_min=min(sample.height for sample in profiled),
            height_max=max(sample.height for sample in profiled),
            height_mean=float(np.mean([sample.height for sample in profiled])),
            mean_pixel=tuple(float(value) for value in mean),
            pixel_std=tuple(float(value) for value in np.sqrt(variance)),
        )

    def curate(
        self,
        max_samples: int = 1024,
        profile_size: tuple[int, int] = DEFAULT_PROFILE_SIZE,
    ) -> ModalityCurationReport:
        """Build bounded image statistics and reviewable findings."""
        profile = self.profile(max_samples, profile_size)
        labels = label_counts(sample.label for sample in self.samples)
        conflicts = conflicting_label_groups(
            (sample.content_hash, sample.source, sample.label)
            for sample in self.samples
        )
        profile_data = profile.to_dict()
        profile_data["label_counts"] = labels
        profile_data["conflicting_label_groups"] = list(conflicts)
        sensitive_metadata = sorted(
            {
                tag
                for sample in self.samples
                for tag in sample.sensitive_metadata
            }
        )
        profile_data["sensitive_metadata"] = sensitive_metadata
        near_duplicate_groups = self.near_duplicate_groups(max_samples)
        profile_data["near_duplicate_groups"] = [
            list(group) for group in near_duplicate_groups
        ]
        augmented_samples = [
            sample for sample in self.samples if sample.augmentation is not None
        ]
        source_names = {sample.source for sample in self.samples}
        profile_data["augmentation"] = {
            "status": "absent" if not augmented_samples else "declared",
            "augmented_sample_count": len(augmented_samples),
            "operations": sorted(
                {
                    sample.augmentation.operation
                    for sample in augmented_samples
                    if sample.augmentation is not None
                }
            ),
            "external_parent_sources": sorted(
                {
                    sample.augmentation.parent_source
                    for sample in augmented_samples
                    if sample.augmentation is not None
                    and sample.augmentation.parent_source not in source_names
                }
            ),
        }
        quality_findings = []
        risk_findings = []
        if not self.samples:
            quality_findings.append(
                build_finding(
                    "quality.image-empty-dataset",
                    data_contracts.FindingSeverity.CRITICAL,
                    {"dataset": "empty"},
                    {"sample_count": 0},
                    "Provide at least one validated image before training.",
                    True,
                )
            )
        if profile.duplicate_count:
            quality_findings.append(
                build_finding(
                    "quality.image-duplicate-content",
                    data_contracts.FindingSeverity.WARNING,
                    {"dataset": "profiled_samples"},
                    {"duplicate_count": profile.duplicate_count},
                    "Review duplicate content and prevent split contamination.",
                    False,
                )
            )
        if near_duplicate_groups:
            quality_findings.append(
                build_finding(
                    "quality.image-near-duplicate-content",
                    data_contracts.FindingSeverity.WARNING,
                    {"dataset": "profiled_samples"},
                    {
                        "group_count": len(near_duplicate_groups),
                        "groups": [
                            list(group) for group in near_duplicate_groups
                        ],
                    },
                    "Review visually similar images and keep related content "
                    "within one split before training.",
                    False,
                )
            )
        if conflicts:
            risk_findings.append(
                build_finding(
                    "risk.image-conflicting-duplicate-labels",
                    data_contracts.FindingSeverity.ERROR,
                    {"dataset": "all_samples"},
                    {"groups": list(conflicts)},
                    "Resolve the label conflict or explicitly exclude the "
                    "affected content.",
                    True,
                )
            )
        if sensitive_metadata:
            risk_findings.append(
                build_finding(
                    "risk.image-sensitive-metadata",
                    data_contracts.FindingSeverity.ERROR,
                    {"dataset": "all_samples"},
                    {"tags": sensitive_metadata},
                    "Review, redact, or explicitly approve image metadata "
                    "before training or distribution.",
                    True,
                )
            )
        if profile.truncated:
            quality_findings.append(
                build_finding(
                    "quality.image-profile-truncated",
                    data_contracts.FindingSeverity.INFO,
                    {"dataset": "profiled_samples"},
                    {
                        "max_samples": max_samples,
                        "profiled_count": profile.profiled_count,
                    },
                    "Run a full-content audit before release or record the "
                    "sampling policy.",
                    False,
                )
            )
        if "unlabeled" in labels and len(labels) > 1:
            risk_findings.append(
                build_finding(
                    "risk.image-mixed-label-state",
                    data_contracts.FindingSeverity.WARNING,
                    {"dataset": "all_samples"},
                    {"label_counts": labels},
                    "Declare whether unlabeled images are valid for the "
                    "intended task before training.",
                    False,
                )
            )
        return ModalityCurationReport(
            modality="image",
            dataset_hash=self.content_hash,
            sample_count=len(self.samples),
            profile=profile_data,
            quality_findings=tuple(quality_findings),
            risk_findings=tuple(risk_findings),
        )

    def prepare_tabular_features(
        self,
        target_size: tuple[int, int] = (16, 16),
        mode: str = "RGB",
        normalize: bool = True,
    ) -> np.ndarray:
        """Flattens resized pixels for a numeric baseline adapter.

        The result has one row per sample and a stable column order of
        channel-major? No: Pillow/NumPy row-major order is retained, with
        channels in the selected mode's order. Values are ``[0, 1]`` when
        ``normalize`` is true, otherwise ``uint8`` values in ``[0, 255]``.
        """
        validate_feature_options(target_size, mode)
        channels = 1 if mode == "L" else len(mode)
        feature_count = target_size[0] * target_size[1] * channels
        if not self.samples:
            dtype = np.float32 if normalize else np.uint8
            return np.empty((0, feature_count), dtype=dtype)
        rows = [
            sample.decode(mode, target_size).reshape(-1)
            for sample in self.samples
        ]
        features = np.stack(rows)
        if normalize:
            return features.astype(np.float32) / 255.0
        return features

    def split(
        self,
        validation_fraction: float = 0.2,
        test_fraction: float = 0.2,
        seed: int = 0,
    ) -> ImageSplit:
        """Split samples by explicit group identity to prevent leakage.

        A caller should use a subject, capture session, or other identity as
        ``group_id``. Without one, source identity is the conservative group.
        """
        if not self.samples:
            raise ValueError("cannot split an empty image dataset")
        if not 0 <= validation_fraction < 1:
            raise ValueError("validation_fraction must be in [0, 1)")
        if not 0 <= test_fraction < 1:
            raise ValueError("test_fraction must be in [0, 1)")
        if validation_fraction + test_fraction >= 1:
            raise ValueError("validation_fraction + test_fraction must be < 1")
        grouped: dict[str, list[ImageSample]] = {}
        for sample in self.samples:
            grouped.setdefault(sample.group_id or sample.source, []).append(
                sample
            )
        groups = sorted(grouped)
        randomizer = random.Random(seed)  # nosec B311 - seeded data split
        randomizer.shuffle(groups)
        test_count = int(len(groups) * test_fraction)
        validation_count = int(len(groups) * validation_fraction)
        if test_fraction and test_count == 0 and len(groups) >= 3:
            test_count = 1
        if validation_fraction and validation_count == 0 and len(groups) >= 3:
            validation_count = 1
        while test_count + validation_count >= len(groups):
            if validation_count > 0:
                validation_count -= 1
            elif test_count > 0:
                test_count -= 1
            else:
                break
        test_groups = set(groups[:test_count])
        validation_groups = set(
            groups[test_count : test_count + validation_count]
        )

        def samples_for(excluded: set[str]) -> tuple[ImageSample, ...]:
            return tuple(
                sample
                for sample in self.samples
                if (sample.group_id or sample.source) not in excluded
            )

        validation = tuple(
            sample
            for sample in self.samples
            if (sample.group_id or sample.source) in validation_groups
        )
        test = tuple(
            sample
            for sample in self.samples
            if (sample.group_id or sample.source) in test_groups
        )
        return ImageSplit(
            train=samples_for(test_groups | validation_groups),
            validation=validation,
            test=test,
        )
