"""Video data contracts with explicit, caller-owned probing."""

from collections.abc import Callable
import dataclasses
import hashlib
import pathlib
import random
from typing import Self

import numpy as np

from scml.contracts import data as data_contracts
from scml.contracts import errors
from scml.data.modalities.curation import build_finding
from scml.data.modalities.curation import conflicting_label_groups
from scml.data.modalities.curation import dataset_hash
from scml.data.modalities.curation import label_counts
from scml.data.modalities.curation import ModalityCurationReport


class VideoValidationError(errors.DataError):
    """A video sample violates an ingestion or metadata invariant."""

    category = "video_validation"


@dataclasses.dataclass(frozen=True, slots=True)
class VideoMetadata:
    """Decoded container metadata required for bounded video profiling."""

    format: str
    width: int
    height: int
    frame_count: int
    frames_per_second: float
    codec: str
    audio_tracks: int = 0

    def __post_init__(self) -> None:
        """Validate dimensions, timing, and track counts."""
        if not self.format.strip() or not self.codec.strip():
            raise ValueError("format and codec must not be empty")
        if self.width < 1 or self.height < 1:
            raise ValueError("video dimensions must be positive")
        if self.frame_count < 1 or self.frames_per_second <= 0:
            raise ValueError("video frame timing must be positive")
        if self.audio_tracks < 0:
            raise ValueError("audio_tracks must not be negative")

    @property
    def duration_seconds(self) -> float:
        """Return the duration implied by frame count and rate."""
        return self.frame_count / self.frames_per_second


@dataclasses.dataclass(frozen=True, slots=True)
class VideoProfile:
    """Bounded data-understanding summary for one video sample."""

    source: str
    format: str
    codec: str
    width: int
    height: int
    frame_count: int
    frames_per_second: float
    duration_seconds: float
    audio_tracks: int
    byte_size: int


@dataclasses.dataclass(frozen=True, slots=True)
class VideoFeatures:
    """Numeric metadata features without a hidden frame-decoding policy."""

    sources: tuple[str, ...]
    feature_names: tuple[str, ...]
    values: tuple[tuple[float, ...], ...]

    @property
    def shape(self) -> tuple[int, int]:
        """Return the number of videos and metadata features."""
        return (len(self.sources), len(self.feature_names))

    def to_numpy(self) -> np.ndarray:
        """Return a writable copy of the metadata feature matrix."""
        return np.asarray(self.values, dtype=np.float64).copy()


@dataclasses.dataclass(frozen=True, slots=True)
class VideoSplit:
    """Group-aware video split that keeps related clips together."""

    train: tuple["VideoSample", ...]
    validation: tuple["VideoSample", ...]
    test: tuple["VideoSample", ...]

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


VideoProbe = Callable[[bytes], VideoMetadata]


def canonical_source(source: str | pathlib.Path) -> str:
    """Return a stable source identifier for a path or URI-like value."""
    value = str(source).strip()
    if not value:
        return "memory://"
    if "://" in value:
        return value
    return pathlib.Path(value).expanduser().resolve(strict=False).as_uri()


@dataclasses.dataclass(frozen=True, slots=True)
class VideoSample:
    """Immutable encoded video tied to metadata observed by a probe."""

    source: str
    encoded_bytes: bytes = dataclasses.field(repr=False)
    file_hash: str
    metadata: VideoMetadata
    label: str | int | float | bool | None = None
    group_id: str | None = None
    byte_size: int = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        """Validate the payload and its content identity."""
        raw = bytes(self.encoded_bytes)
        object.__setattr__(self, "encoded_bytes", raw)
        if not self.source.strip():
            raise ValueError("source must not be empty")
        if self.group_id is not None and not self.group_id.strip():
            raise ValueError("group_id must not be empty when provided")
        if not raw:
            raise VideoValidationError(
                "video payload is empty", {"code": "empty_payload"}
            )
        if len(self.file_hash) != 64 or any(
            character not in "0123456789abcdef" for character in self.file_hash
        ):
            raise ValueError("file_hash must be a SHA-256 hex digest")
        if hashlib.sha256(raw).hexdigest() != self.file_hash:
            raise ValueError("file_hash does not match encoded_bytes")
        object.__setattr__(self, "byte_size", len(raw))

    @classmethod
    def from_bytes(
        cls,
        encoded_bytes: bytes,
        metadata: VideoMetadata,
        source: str = "memory://",
        max_bytes: int | None = 2 * 1024 * 1024 * 1024,
        label: str | int | float | bool | None = None,
        group_id: str | None = None,
    ) -> Self:
        """Create a sample from bytes and explicit probe output.

        Video codecs are intentionally not imported by core. A production
        adapter should record the probe version alongside the sample manifest.
        """
        raw = bytes(encoded_bytes)
        if not raw:
            raise VideoValidationError(
                "video payload is empty", {"code": "empty_payload"}
            )
        if max_bytes is not None and (max_bytes < 1 or len(raw) > max_bytes):
            raise VideoValidationError(
                "video payload exceeds the configured byte limit",
                {
                    "code": "encoded_size_limit",
                    "byte_size": len(raw),
                    "max_bytes": max_bytes,
                },
            )
        return cls(
            source=canonical_source(source),
            encoded_bytes=raw,
            file_hash=hashlib.sha256(raw).hexdigest(),
            metadata=metadata,
            label=label,
            group_id=group_id,
        )

    def profile(self) -> VideoProfile:
        """Return bounded metadata suitable for curation and data cards."""
        return VideoProfile(
            source=self.source,
            format=self.metadata.format,
            codec=self.metadata.codec,
            width=self.metadata.width,
            height=self.metadata.height,
            frame_count=self.metadata.frame_count,
            frames_per_second=self.metadata.frames_per_second,
            duration_seconds=self.metadata.duration_seconds,
            audio_tracks=self.metadata.audio_tracks,
            byte_size=self.byte_size,
        )


@dataclasses.dataclass(frozen=True, slots=True)
class VideoDataset:
    """Immutable video collection with deterministic source identity."""

    samples: tuple[VideoSample, ...]
    content_hash: str = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        """Reject duplicate sources and derive a stable dataset hash."""
        normalized = tuple(self.samples)
        if not normalized:
            raise VideoValidationError(
                "video dataset must contain at least one sample",
                {"code": "empty_dataset"},
            )
        sources = [sample.source for sample in normalized]
        if len(sources) != len(set(sources)):
            raise VideoValidationError(
                "video dataset contains duplicate sources",
                {"code": "duplicate_source"},
            )
        digest = hashlib.sha256(
            "\n".join(
                f"{sample.source}\0{sample.file_hash}" for sample in normalized
            ).encode("utf-8")
        ).hexdigest()
        object.__setattr__(self, "samples", normalized)
        object.__setattr__(self, "content_hash", digest)

    def profiles(self) -> tuple[VideoProfile, ...]:
        """Return profiles in source order for deterministic curation."""
        return tuple(sample.profile() for sample in self.samples)

    def prepare_metadata_features(self) -> VideoFeatures:
        """Prepare deterministic numeric features from probed metadata.

        This method never decodes or samples frames. A future video baseline
        must add an explicit decoder, temporal sampling policy, resource
        budget, and synchronization evidence rather than inheriting any of
        those decisions from this metadata-only contract.
        """
        feature_names = (
            "duration_seconds",
            "width",
            "height",
            "pixel_count",
            "aspect_ratio",
            "frame_count",
            "frames_per_second",
            "audio_tracks",
            "byte_size",
            "bytes_per_frame",
        )
        rows = []
        for sample in self.samples:
            metadata = sample.metadata
            rows.append(
                (
                    metadata.duration_seconds,
                    float(metadata.width),
                    float(metadata.height),
                    float(metadata.width * metadata.height),
                    metadata.width / metadata.height,
                    float(metadata.frame_count),
                    metadata.frames_per_second,
                    float(metadata.audio_tracks),
                    float(sample.byte_size),
                    sample.byte_size / metadata.frame_count,
                )
            )
        values = np.asarray(rows, dtype=np.float64)
        if not np.isfinite(values).all():
            raise VideoValidationError(
                "video metadata features must be finite",
                {"code": "non_finite_features"},
            )
        return VideoFeatures(
            sources=tuple(sample.source for sample in self.samples),
            feature_names=feature_names,
            values=tuple(
                tuple(float(value) for value in row) for row in values
            ),
        )

    def split(
        self,
        validation_fraction: float = 0.2,
        test_fraction: float = 0.2,
        seed: int = 0,
    ) -> VideoSplit:
        """Split clips by explicit group identity to prevent leakage.

        A caller should use a subject, session, camera, or other identity as
        ``group_id``. Without one, source identity is the conservative group
        and repeated clips are not assumed to share an identity.
        """
        if not 0 <= validation_fraction < 1:
            raise ValueError("validation_fraction must be in [0, 1)")
        if not 0 <= test_fraction < 1:
            raise ValueError("test_fraction must be in [0, 1)")
        if validation_fraction + test_fraction >= 1:
            raise ValueError("validation_fraction + test_fraction must be < 1")
        grouped: dict[str, list[VideoSample]] = {}
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

        def samples_for(excluded: set[str]) -> tuple[VideoSample, ...]:
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
        return VideoSplit(
            train=samples_for(test_groups | validation_groups),
            validation=validation,
            test=test,
        )

    def curate(self) -> ModalityCurationReport:
        """Build reviewable metadata and compatibility findings."""
        profiles = self.profiles()
        formats = sorted({profile.format for profile in profiles})
        codecs = sorted({profile.codec for profile in profiles})
        dimensions = sorted(
            {(profile.width, profile.height) for profile in profiles}
        )
        durations = [profile.duration_seconds for profile in profiles]
        labels = label_counts(sample.label for sample in self.samples)
        conflicts = conflicting_label_groups(
            (sample.file_hash, sample.source, sample.label)
            for sample in self.samples
        )
        quality_findings = []
        if len(formats) > 1 or len(codecs) > 1:
            quality_findings.append(
                build_finding(
                    "quality.video-encoding-drift",
                    data_contracts.FindingSeverity.WARNING,
                    {"sources": [profile.source for profile in profiles]},
                    {"formats": formats, "codecs": codecs},
                    "Choose an explicit decoder and normalization policy "
                    "before training.",
                    False,
                )
            )
        if len(dimensions) > 1:
            quality_findings.append(
                build_finding(
                    "quality.video-dimension-drift",
                    data_contracts.FindingSeverity.WARNING,
                    {"sources": [profile.source for profile in profiles]},
                    {"dimensions": [list(value) for value in dimensions]},
                    "Record frame resizing or bucketing as a versioned "
                    "transformation.",
                    False,
                )
            )
        risk_findings = []
        if "unlabeled" in labels and len(labels) > 1:
            risk_findings.append(
                build_finding(
                    "risk.video-mixed-label-state",
                    data_contracts.FindingSeverity.WARNING,
                    {"dataset": self.content_hash},
                    {"label_counts": labels},
                    "Declare whether unlabeled clips are valid for the "
                    "intended task.",
                    False,
                )
            )
        if conflicts:
            risk_findings.append(
                build_finding(
                    "risk.video-conflicting-duplicate-labels",
                    data_contracts.FindingSeverity.ERROR,
                    {"dataset": self.content_hash},
                    {"groups": list(conflicts)},
                    "Resolve the label conflict or explicitly exclude the "
                    "affected video content.",
                    True,
                )
            )
        return ModalityCurationReport(
            modality="video",
            dataset_hash=dataset_hash(
                f"{sample.source}\0{sample.file_hash}"
                for sample in self.samples
            ),
            sample_count=len(self.samples),
            profile={
                "formats": formats,
                "codecs": codecs,
                "dimensions": [list(value) for value in dimensions],
                "duration_min_seconds": min(durations),
                "duration_max_seconds": max(durations),
                "duration_mean_seconds": sum(durations) / len(durations),
                "total_bytes": sum(profile.byte_size for profile in profiles),
                "label_counts": labels,
                "conflicting_label_groups": list(conflicts),
            },
            quality_findings=tuple(quality_findings),
            risk_findings=tuple(risk_findings),
        )


__all__ = [
    "VideoDataset",
    "VideoFeatures",
    "VideoMetadata",
    "VideoProbe",
    "VideoProfile",
    "VideoSample",
    "VideoSplit",
    "VideoValidationError",
    "canonical_source",
]
