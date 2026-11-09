"""Video data contracts with explicit, caller-owned probing."""

from collections.abc import Callable
import dataclasses
from datetime import datetime
from datetime import UTC
import hashlib
import math
import pathlib
import random
from typing import Any
from typing import Self

import numpy as np

from scmle.contracts import data as data_contracts
from scmle.contracts import errors
from scmle.data.modalities.curation import build_finding
from scmle.data.modalities.curation import conflicting_label_groups
from scmle.data.modalities.curation import label_counts
from scmle.data.modalities.curation import ModalityCurationReport


class VideoValidationError(errors.DataError):
    """A video sample violates an ingestion or metadata invariant."""

    category = "video_validation"


class VideoTemporalError(VideoValidationError):
    """Temporal video evidence is incomplete or invalid."""


@dataclasses.dataclass(frozen=True, slots=True)
class VideoMetadata:
    """Decoded container metadata required for bounded video profiling.

    Attributes:
        format: Non-empty container name.
        width: Frame width in pixels; positive.
        height: Frame height in pixels; positive.
        frame_count: Number of frames; positive.
        frames_per_second: Frame rate; positive.
        codec: Non-empty video codec name.
        audio_tracks: Number of audio tracks; not negative.
        audio_duration_seconds: Measured audio duration, or None. Positive
            and finite when given; needs at least one audio track.
        sync_offset_seconds: Measured audio/video offset, or None. Finite
            when given; needs at least one audio track.
        sync_tolerance_seconds: Allowed absolute offset, or None. Finite and
            not negative when given; needs at least one audio track.
    """

    format: str
    width: int
    height: int
    frame_count: int
    frames_per_second: float
    codec: str
    audio_tracks: int = 0
    audio_duration_seconds: float | None = None
    sync_offset_seconds: float | None = None
    sync_tolerance_seconds: float | None = None

    def __post_init__(self) -> None:
        """Validate dimensions, timing, and track counts.

        Raises:
            ValueError: If format or codec is blank, a dimension is below
                one, frame_count or frames_per_second is not positive,
                audio_tracks is negative, synchronization values are given
                without an audio track, or an audio duration, offset or
                tolerance is not finite, non-positive (duration) or
                negative (tolerance).
        """
        if not self.format.strip() or not self.codec.strip():
            raise ValueError("format and codec must not be empty")
        if self.width < 1 or self.height < 1:
            raise ValueError("video dimensions must be positive")
        if self.frame_count < 1 or self.frames_per_second <= 0:
            raise ValueError("video frame timing must be positive")
        if self.audio_tracks < 0:
            raise ValueError("audio_tracks must not be negative")
        if self.audio_tracks == 0 and any(
            value is not None
            for value in (
                self.audio_duration_seconds,
                self.sync_offset_seconds,
                self.sync_tolerance_seconds,
            )
        ):
            raise ValueError(
                "synchronization metadata requires at least one audio track"
            )
        if self.audio_duration_seconds is not None and (
            not math.isfinite(self.audio_duration_seconds)
            or self.audio_duration_seconds <= 0
        ):
            raise ValueError("audio_duration_seconds must be positive")
        if self.sync_offset_seconds is not None and not math.isfinite(
            self.sync_offset_seconds
        ):
            raise ValueError("sync_offset_seconds must be finite")
        if self.sync_tolerance_seconds is not None and (
            not math.isfinite(self.sync_tolerance_seconds)
            or self.sync_tolerance_seconds < 0
        ):
            raise ValueError("sync_tolerance_seconds must not be negative")

    @property
    def duration_seconds(self) -> float:
        """The duration in seconds implied by frame count and rate."""
        return self.frame_count / self.frames_per_second

    @property
    def synchronization_status(self) -> str:
        """The measured audio/video synchronization state.

        One of ``not_applicable`` (no audio track), ``unknown`` (audio
        duration, offset or tolerance missing), ``out_of_tolerance`` (the
        absolute offset exceeds the tolerance) or ``within_tolerance``.
        """
        if self.audio_tracks == 0:
            return "not_applicable"
        sync_offset = self.sync_offset_seconds
        sync_tolerance = self.sync_tolerance_seconds
        if self.audio_duration_seconds is None:
            return "unknown"
        if sync_offset is None or sync_tolerance is None:
            return "unknown"
        if abs(sync_offset) > sync_tolerance:
            return "out_of_tolerance"
        return "within_tolerance"


@dataclasses.dataclass(frozen=True, slots=True)
class VideoProfile:
    """Bounded data-understanding summary for one video sample.

    Attributes:
        source: Source identifier of the sample.
        format: Container name.
        codec: Video codec name.
        width: Frame width in pixels.
        height: Frame height in pixels.
        frame_count: Number of frames.
        frames_per_second: Frame rate.
        duration_seconds: Duration in seconds.
        audio_tracks: Number of audio tracks.
        byte_size: Size of the encoded payload in bytes.
        audio_duration_seconds: Measured audio duration, if any.
        sync_offset_seconds: Measured audio/video offset, if any.
        sync_tolerance_seconds: Allowed absolute offset, if any.
        synchronization_status: Synchronization state of the metadata.
        observation_time: UTC time the clip was observed, if known.
    """

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
    audio_duration_seconds: float | None = None
    sync_offset_seconds: float | None = None
    sync_tolerance_seconds: float | None = None
    synchronization_status: str = "not_applicable"
    observation_time: datetime | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class VideoFeatures:
    """Numeric video features with explicit sampled-frame provenance.

    Attributes:
        sources: Source identifier of each row.
        feature_names: Name of each column.
        values: One tuple of finite feature values per source.
        sampled_frame_indices: Per source, the indices of the frames the
            features came from; empty when not frame-based.
    """

    sources: tuple[str, ...]
    feature_names: tuple[str, ...]
    values: tuple[tuple[float, ...], ...]
    sampled_frame_indices: tuple[tuple[int, ...], ...] = ()

    def __post_init__(self) -> None:
        """Validate feature dimensions and sampled-frame metadata.

        Raises:
            ValueError: If rows do not match sources, columns do not match
                feature names, a value is not finite, sampled frame
                indices are given for a different number of sources, or an
                index is negative.
        """
        if len(self.values) != len(self.sources):
            raise ValueError("feature rows must match video sources")
        if any(len(row) != len(self.feature_names) for row in self.values):
            raise ValueError("feature columns must match feature names")
        if not np.isfinite(np.asarray(self.values, dtype=np.float64)).all():
            raise ValueError("video features must be finite")
        sampled = tuple(
            tuple(int(index) for index in indices)
            for indices in self.sampled_frame_indices
        )
        if sampled and len(sampled) != len(self.sources):
            raise ValueError("sampled frame metadata must match video sources")
        if any(index < 0 for indices in sampled for index in indices):
            raise ValueError("sampled frame indices must not be negative")
        object.__setattr__(self, "sources", tuple(self.sources))
        object.__setattr__(self, "feature_names", tuple(self.feature_names))
        object.__setattr__(
            self,
            "values",
            tuple(tuple(float(value) for value in row) for row in self.values),
        )
        object.__setattr__(self, "sampled_frame_indices", sampled)

    @property
    def shape(self) -> tuple[int, int]:
        """The number of videos and the number of features, as a pair."""
        return (len(self.sources), len(self.feature_names))

    def to_numpy(self) -> np.ndarray:
        """Return a writable copy of the feature matrix.

        Returns:
            A float64 array of shape ``(videos, features)``.
        """
        return np.asarray(self.values, dtype=np.float64).copy()


@dataclasses.dataclass(frozen=True, slots=True)
class VideoSplit:
    """Group-aware video split that keeps related clips together.

    Attributes:
        train: Samples assigned to training.
        validation: Samples assigned to validation.
        test: Samples assigned to testing.
    """

    train: tuple["VideoSample", ...]
    validation: tuple["VideoSample", ...]
    test: tuple["VideoSample", ...]

    def group_ids(self) -> dict[str, tuple[str, ...]]:
        """Return the stable groups assigned to each partition.

        Returns:
            A dict from partition name (`train`, `validation`, `test`) to
            the sorted distinct group ids; a sample without a group id uses
            its source.
        """
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
class VideoTemporalSplit:
    """Time-ordered video clips separated at a UTC cutoff.

    Attributes:
        cutoff: Timezone-aware cutoff, converted to UTC.
        train: Clips observed at or before the cutoff.
        future: Clips observed after the cutoff.
    """

    cutoff: datetime
    train: tuple["VideoSample", ...]
    future: tuple["VideoSample", ...]

    def __post_init__(self) -> None:
        """Validate the cutoff and timestamp completeness of each partition.

        Raises:
            ValueError: If the cutoff is not timezone-aware or a clip in
                either partition has no observation time.
        """
        if self.cutoff.tzinfo is None or self.cutoff.utcoffset() is None:
            raise ValueError("cutoff must be timezone-aware")
        object.__setattr__(self, "cutoff", self.cutoff.astimezone(UTC))
        if any(sample.observation_time is None for sample in self.train):
            raise ValueError("train videos must contain observation times")
        if any(sample.observation_time is None for sample in self.future):
            raise ValueError("future videos must contain observation times")

    @property
    def has_future_leakage(self) -> bool:
        """Whether a training clip was observed after the cutoff."""
        return any(
            isinstance(sample.observation_time, datetime)
            and sample.observation_time > self.cutoff
            for sample in self.train
        )


VideoProbe = Callable[[bytes], VideoMetadata]
VideoFrameDecoder = Callable[[bytes, int], np.ndarray]


def canonical_source(source: str | pathlib.Path) -> str:
    """Return a stable source identifier for a path or URI-like value.

    Args:
        source: A filesystem path, a value containing ``://``, or empty.

    Returns:
        ``memory://`` for an empty value, the stripped value itself when it
        contains ``://``, otherwise the absolute resolved path as a file URI.
    """
    value = str(source).strip()
    if not value:
        return "memory://"
    if "://" in value:
        return value
    return pathlib.Path(value).expanduser().resolve(strict=False).as_uri()


def canonical_observation_time(value: datetime | str) -> datetime:
    """Parse and normalize a timezone-aware observation time to UTC.

    Args:
        value: A datetime, or ISO 8601 text; a trailing ``Z`` means UTC.

    Returns:
        The same instant as a UTC datetime.

    Raises:
        VideoTemporalError: If the text is not ISO 8601, the value is
            neither datetime nor text, or the time has no timezone.
    """
    if isinstance(value, str):
        normalized = value.strip()
        if normalized.endswith(("Z", "z")):
            normalized = normalized[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError as error:
            raise VideoTemporalError(
                f"invalid observation time: {value!r}",
                {"code": "invalid_observation_time"},
            ) from error
    elif isinstance(value, datetime):
        parsed = value
    else:
        raise VideoTemporalError(
            "observation time must be datetime or text",
            {"code": "invalid_observation_time"},
        )
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise VideoTemporalError(
            "observation time must be timezone-aware",
            {"code": "timezone_required"},
        )
    return parsed.astimezone(UTC)


@dataclasses.dataclass(frozen=True, slots=True)
class VideoSample:
    """Immutable encoded video tied to metadata observed by a probe.

    Attributes:
        source: Non-blank source label or URI.
        encoded_bytes: The non-empty encoded payload.
        file_hash: SHA-256 hex digest of the payload.
        metadata: Metadata observed by the probe.
        label: Optional task label.
        group_id: Optional identity used to keep related clips in one split.
        observation_time: UTC time the clip was observed, if known.
        byte_size: Length of the payload; set on construction.
    """

    source: str
    encoded_bytes: bytes = dataclasses.field(repr=False)
    file_hash: str
    metadata: VideoMetadata
    label: str | int | float | bool | None = None
    group_id: str | None = None
    observation_time: datetime | None = None
    byte_size: int = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        """Validate the payload and its content identity.

        Raises:
            ValueError: If source or group_id is blank, or file_hash is not
                a lowercase SHA-256 hex digest of the payload.
            VideoValidationError: If the payload is empty.
        """
        raw = bytes(self.encoded_bytes)
        object.__setattr__(self, "encoded_bytes", raw)
        if not self.source.strip():
            raise ValueError("source must not be empty")
        if self.group_id is not None and not self.group_id.strip():
            raise ValueError("group_id must not be empty when provided")
        if self.observation_time is not None:
            object.__setattr__(
                self,
                "observation_time",
                canonical_observation_time(self.observation_time),
            )
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
        observation_time: datetime | str | None = None,
    ) -> Self:
        """Create a sample from bytes and explicit probe output.

        Video codecs are intentionally not imported by core. A production
        adapter should record the probe version alongside the sample manifest.

        Args:
            encoded_bytes: The encoded video payload.
            metadata: Metadata produced by the caller's probe.
            source: Source label or URI, normalised with `canonical_source`.
            max_bytes: Largest payload size in bytes, or None for no limit.
            label: Optional task label.
            group_id: Optional identity used to prevent split leakage.
            observation_time: Optional timezone-aware time, as a datetime or
                ISO 8601 text, stored in UTC.

        Returns:
            The new sample.

        Raises:
            VideoValidationError: If the payload is empty, or exceeds
                ``max_bytes`` (or ``max_bytes`` is below one).
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
            observation_time=(
                canonical_observation_time(observation_time)
                if observation_time is not None
                else None
            ),
        )

    def profile(self) -> VideoProfile:
        """Return bounded metadata suitable for curation and data cards.

        Returns:
            The profile of this sample.
        """
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
            audio_duration_seconds=self.metadata.audio_duration_seconds,
            sync_offset_seconds=self.metadata.sync_offset_seconds,
            sync_tolerance_seconds=self.metadata.sync_tolerance_seconds,
            synchronization_status=self.metadata.synchronization_status,
            observation_time=self.observation_time,
        )


@dataclasses.dataclass(frozen=True, slots=True)
class VideoDataset:
    """Immutable video collection with deterministic source identity.

    Attributes:
        samples: The samples, in caller order.
        content_hash: SHA-256 hex digest of the sources, file hashes and
            observation times, in sample order; set on construction.
    """

    samples: tuple[VideoSample, ...]
    content_hash: str = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        """Reject duplicate sources and derive a stable dataset hash.

        Raises:
            VideoValidationError: If there are no samples or two samples
                share a source.
        """
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
                f"{sample.source}\0{sample.file_hash}"
                + (
                    f"\0{sample.observation_time.isoformat()}"
                    if sample.observation_time is not None
                    else ""
                )
                for sample in normalized
            ).encode("utf-8")
        ).hexdigest()
        object.__setattr__(self, "samples", normalized)
        object.__setattr__(self, "content_hash", digest)

    def profiles(self) -> tuple[VideoProfile, ...]:
        """Return profiles in sample order for deterministic curation.

        Returns:
            One profile per sample.
        """
        return tuple(sample.profile() for sample in self.samples)

    def prepare_metadata_features(self) -> VideoFeatures:
        """Prepare deterministic numeric features from probed metadata.

        This method never decodes or samples frames. Frame-based features
        need an explicit decoder, sampling policy and resource budget; see
        `prepare_frame_features`.

        Returns:
            One row of ten features per sample: duration in seconds, width,
            height, pixel count, aspect ratio, frame count, frame rate,
            audio track count, byte size and bytes per frame.

        Raises:
            VideoValidationError: If a feature is not finite.
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

    def prepare_frame_features(
        self,
        decoder: VideoFrameDecoder,
        max_frames: int = 8,
        max_frame_pixels: int = 1_000_000,
    ) -> VideoFeatures:
        """Prepare bounded temporal statistics through an explicit decoder.

        The core package does not select or import a video codec. The caller
        supplies a decoder that accepts encoded bytes and an absolute frame
        index. The contract samples evenly spaced frames, validates each
        decoded array, and records the selected indices beside the features.
        Each sampled frame contributes mean, standard deviation, minimum,
        maximum, and an observation flag; absent frame slots are zero-filled.

        Args:
            decoder: Caller-owned frame decoder.
            max_frames: Maximum number of frames decoded per sample.
            max_frame_pixels: Maximum number of decoded array elements per
                frame, including channels.

        Returns:
            Numeric frame-statistics features with temporal provenance.

        Raises:
            TypeError: If ``decoder`` is not callable.
            ValueError: If ``max_frames`` or ``max_frame_pixels`` is below
                one.
            VideoValidationError: If a decoded frame violates its metadata
                or cannot be converted to finite numeric values.
        """
        if not callable(decoder):
            raise TypeError("decoder must be callable")
        if max_frames < 1 or max_frame_pixels < 1:
            raise ValueError("frame bounds must be positive")
        slot_names = ("mean", "std", "min", "max", "observed")
        feature_names = tuple(
            f"frame[{slot}].{name}"
            for slot in range(max_frames)
            for name in slot_names
        )
        rows: list[tuple[float, ...]] = []
        sampled_indices: list[tuple[int, ...]] = []
        for sample in self.samples:
            indices = self.frame_indices_for(sample, max_frames)
            values = np.zeros(max_frames * len(slot_names), dtype=np.float64)
            for slot, frame_index in enumerate(indices):
                try:
                    frame = np.asarray(
                        decoder(sample.encoded_bytes, frame_index)
                    )
                except Exception as error:
                    raise VideoValidationError(
                        "video frame decoding failed",
                        {
                            "code": "frame_decode_failed",
                            "source": sample.source,
                            "frame_index": frame_index,
                        },
                    ) from error
                if frame.ndim not in {2, 3}:
                    raise VideoValidationError(
                        "decoded video frames must be grayscale or "
                        "channel-last",
                        {
                            "code": "invalid_frame_shape",
                            "source": sample.source,
                            "frame_index": frame_index,
                            "shape": list(frame.shape),
                        },
                    )
                if frame.size < 1 or frame.size > max_frame_pixels:
                    raise VideoValidationError(
                        "decoded video frame exceeds the configured pixel "
                        "limit",
                        {
                            "code": "frame_pixel_limit",
                            "source": sample.source,
                            "frame_index": frame_index,
                            "pixel_count": int(frame.size),
                            "max_frame_pixels": max_frame_pixels,
                        },
                    )
                if (
                    frame.shape[0] > sample.metadata.height
                    or frame.shape[1] > sample.metadata.width
                ):
                    raise VideoValidationError(
                        "decoded video frame exceeds probed dimensions",
                        {
                            "code": "frame_dimension_mismatch",
                            "source": sample.source,
                            "frame_index": frame_index,
                            "shape": list(frame.shape),
                            "declared": [
                                sample.metadata.height,
                                sample.metadata.width,
                            ],
                        },
                    )
                numeric = np.asarray(frame, dtype=np.float64)
                if not np.isfinite(numeric).all():
                    raise VideoValidationError(
                        "decoded video frame contains non-finite values",
                        {
                            "code": "non_finite_frame",
                            "source": sample.source,
                            "frame_index": frame_index,
                        },
                    )
                start = slot * len(slot_names)
                values[start : start + len(slot_names)] = (
                    float(numeric.mean()),
                    float(numeric.std()),
                    float(np.min(numeric)),
                    float(np.max(numeric)),
                    1.0,
                )
            rows.append(tuple(float(value) for value in values))
            sampled_indices.append(indices)
        return VideoFeatures(
            sources=tuple(sample.source for sample in self.samples),
            feature_names=feature_names,
            values=tuple(rows),
            sampled_frame_indices=tuple(sampled_indices),
        )

    @staticmethod
    def frame_indices_for(
        sample: VideoSample, max_frames: int
    ) -> tuple[int, ...]:
        """Return deterministic, evenly spaced frame indices for a sample.

        Args:
            sample: The video sample.
            max_frames: Largest number of indices to return; at least one.

        Returns:
            Ascending frame indices from the first to the last frame,
            numbering ``min(max_frames, frame_count)``.

        Raises:
            ValueError: If ``max_frames`` is below one.
        """
        if max_frames < 1:
            raise ValueError("max_frames must be positive")
        count = min(max_frames, sample.metadata.frame_count)
        return tuple(
            int(index)
            for index in np.linspace(
                0,
                sample.metadata.frame_count - 1,
                count,
                dtype=int,
            )
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

        Args:
            validation_fraction: Share of groups for validation, in [0, 1).
            test_fraction: Share of groups for testing, in [0, 1).
            seed: Seed of the shuffle of the sorted group ids.

        Returns:
            The split. Whole groups go to one partition; with at least three
            groups a non-zero fraction gets at least one group, and the
            remaining groups form the training set.

        Raises:
            ValueError: If a fraction is outside [0, 1) or the fractions sum
                to 1 or more.
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

    def split_temporal(self, cutoff: datetime | str) -> VideoTemporalSplit:
        """Split fully timestamped videos into past and future partitions.

        Videos at the cutoff belong to training. Videos after the cutoff
        belong to future evaluation. Untimestamped videos fail closed because
        assigning them a temporal position would invent ordering evidence.

        Args:
            cutoff: Timezone-aware datetime or ISO 8601 text.

        Returns:
            The split, with the cutoff in UTC.

        Raises:
            VideoTemporalError: If any video has no observation time, or the
                cutoff is invalid (see `canonical_observation_time`).
        """
        cutoff_time = canonical_observation_time(cutoff)
        if any(sample.observation_time is None for sample in self.samples):
            raise VideoTemporalError(
                "temporal splitting requires an observation time on every "
                "video",
                {"code": "incomplete_observation_times"},
            )
        train = tuple(
            sample
            for sample in self.samples
            if isinstance(sample.observation_time, datetime)
            and sample.observation_time <= cutoff_time
        )
        future = tuple(
            sample
            for sample in self.samples
            if isinstance(sample.observation_time, datetime)
            and sample.observation_time > cutoff_time
        )
        return VideoTemporalSplit(
            cutoff=cutoff_time,
            train=train,
            future=future,
        )

    def curate(self) -> ModalityCurationReport:
        """Build reviewable metadata and compatibility findings.

        Returns:
            A curation report for modality ``video`` with format, codec,
            dimension, duration, synchronization, label and temporal
            statistics, and findings for audio/video synchronization,
            encoding or dimension drift, partly timestamped data, mixed
            labelled and unlabelled clips, and conflicting labels.
        """
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
        synchronization_statuses = sorted(
            {sample.metadata.synchronization_status for sample in self.samples}
        )
        out_of_tolerance = [
            sample.source
            for sample in self.samples
            if sample.metadata.synchronization_status == "out_of_tolerance"
        ]
        unknown_synchronization = [
            sample.source
            for sample in self.samples
            if sample.metadata.synchronization_status == "unknown"
        ]
        if out_of_tolerance:
            quality_findings.append(
                build_finding(
                    "quality.video-audio-sync-out-of-tolerance",
                    data_contracts.FindingSeverity.ERROR,
                    {"sources": out_of_tolerance},
                    {"count": len(out_of_tolerance)},
                    "Align audio and video or exclude synchronization-"
                    "dependent tasks from the capability declaration.",
                    True,
                )
            )
        if unknown_synchronization:
            quality_findings.append(
                build_finding(
                    "quality.video-audio-sync-unknown",
                    data_contracts.FindingSeverity.WARNING,
                    {"sources": unknown_synchronization},
                    {"count": len(unknown_synchronization)},
                    "Supply measured audio duration, offset, and tolerance "
                    "before making audio-video timing claims.",
                    False,
                )
            )
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
        profile = {
            "formats": formats,
            "codecs": codecs,
            "dimensions": [list(value) for value in dimensions],
            "duration_min_seconds": min(durations),
            "duration_max_seconds": max(durations),
            "duration_mean_seconds": sum(durations) / len(durations),
            "total_bytes": sum(profile.byte_size for profile in profiles),
            "synchronization_statuses": synchronization_statuses,
            "label_counts": labels,
            "conflicting_label_groups": list(conflicts),
        }
        observation_times = tuple(
            sample.observation_time
            for sample in self.samples
            if sample.observation_time is not None
        )
        temporal_evidence: dict[str, Any]
        if not observation_times:
            temporal_evidence = {"status": "absent"}
        elif len(observation_times) == len(self.samples):
            temporal_evidence = {
                "status": "complete",
                "observation_time_min": min(observation_times).isoformat(),
                "observation_time_max": max(observation_times).isoformat(),
            }
        else:
            temporal_evidence = {
                "status": "mixed",
                "timed_video_count": len(observation_times),
                "untimed_video_count": len(self.samples)
                - len(observation_times),
            }
        profile["temporal"] = temporal_evidence
        if temporal_evidence["status"] == "mixed":
            quality_findings.append(
                build_finding(
                    "quality.video-mixed-temporal-state",
                    data_contracts.FindingSeverity.WARNING,
                    {"dataset": self.content_hash},
                    temporal_evidence,
                    "Supply observation times for every clip before using "
                    "temporal evaluation or declare the untimed subset.",
                    False,
                )
            )
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
            dataset_hash=self.content_hash,
            sample_count=len(self.samples),
            profile=profile,
            quality_findings=tuple(quality_findings),
            risk_findings=tuple(risk_findings),
        )


__all__ = [
    "VideoDataset",
    "VideoFeatures",
    "VideoFrameDecoder",
    "VideoMetadata",
    "VideoProbe",
    "VideoProfile",
    "VideoSample",
    "VideoSplit",
    "VideoTemporalError",
    "VideoTemporalSplit",
    "VideoValidationError",
    "canonical_observation_time",
    "canonical_source",
]
