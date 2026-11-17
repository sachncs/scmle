"""Video samples, their probed metadata and source identity."""

from collections.abc import Callable
import dataclasses
from datetime import datetime
from datetime import UTC
import hashlib
import math
import pathlib
from typing import Self

import numpy

from scmle.contracts import errors

SHA256_HEX_LENGTH = 64


class VideoError(errors.DataError):
    """A video sample violates an ingestion or metadata invariant."""

    category = "video_validation"


class VideoTemporalError(VideoError):
    """Temporal video evidence is incomplete or invalid."""


@dataclasses.dataclass(frozen=True, slots=True)
class Metadata:
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
        """Validates dimensions, timing, and track counts.

        Raises:
            ValueError: If format or codec is blank, a dimension is below
                one, frame_count or frames_per_second is not positive,
                audio_tracks is negative, synchronization values are given
                without an audio track, or an audio duration, offset or
                tolerance is not finite, non-positive (duration) or
                negative (tolerance).
        """
        if not self.format.strip() or not self.codec.strip():
            raise errors.InputError("format and codec must not be empty")
        if self.width < 1 or self.height < 1:
            raise errors.InputError("video dimensions must be positive")
        if self.frame_count < 1 or self.frames_per_second <= 0:
            raise errors.InputError("video frame timing must be positive")
        if self.audio_tracks < 0:
            raise errors.InputError("audio_tracks must not be negative")
        if self.audio_tracks == 0 and any(
            value is not None
            for value in (
                self.audio_duration_seconds,
                self.sync_offset_seconds,
                self.sync_tolerance_seconds,
            )
        ):
            raise errors.InputError(
                "synchronization metadata requires at least one audio track"
            )
        if self.audio_duration_seconds is not None and (
            not math.isfinite(self.audio_duration_seconds)
            or self.audio_duration_seconds <= 0
        ):
            raise errors.InputError("audio_duration_seconds must be positive")
        if self.sync_offset_seconds is not None and not math.isfinite(
            self.sync_offset_seconds
        ):
            raise errors.InputError("sync_offset_seconds must be finite")
        if self.sync_tolerance_seconds is not None and (
            not math.isfinite(self.sync_tolerance_seconds)
            or self.sync_tolerance_seconds < 0
        ):
            raise errors.InputError(
                "sync_tolerance_seconds must not be negative"
            )

    @property
    def duration_seconds(self) -> float:
        """The duration in seconds implied by frame count and rate."""
        return self.frame_count / self.frames_per_second

    @property
    def synchrony(self) -> str:
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
class Profile:
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
        synchrony: Synchronization state of the metadata.
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
    synchrony: str = "not_applicable"
    observation_time: datetime | None = None


VideoProbe = Callable[[bytes], Metadata]
VideoFrameDecoder = Callable[[bytes, int], numpy.ndarray]


def canonical_source(source: str | pathlib.Path) -> str:
    """Returns a stable source identifier for a path or URI-like value.

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
    """Parses and normalizes a timezone-aware observation time to UTC.

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
class Sample:
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
    metadata: Metadata
    label: str | int | float | bool | None = None
    group_id: str | None = None
    observation_time: datetime | None = None
    byte_size: int = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        """Validates the payload and its content identity.

        Raises:
            ValueError: If source or group_id is blank, or file_hash is not
                a lowercase SHA-256 hex digest of the payload.
            VideoError: If the payload is empty.
        """
        raw = bytes(self.encoded_bytes)
        object.__setattr__(self, "encoded_bytes", raw)
        if not self.source.strip():
            raise errors.InputError("source must not be empty")
        if self.group_id is not None and not self.group_id.strip():
            raise errors.InputError("group_id must not be empty when provided")
        if self.observation_time is not None:
            object.__setattr__(
                self,
                "observation_time",
                canonical_observation_time(self.observation_time),
            )
        if not raw:
            raise VideoError(
                "video payload is empty", {"code": "empty_payload"}
            )
        if len(self.file_hash) != SHA256_HEX_LENGTH or any(
            character not in "0123456789abcdef" for character in self.file_hash
        ):
            raise errors.InputError("file_hash must be a SHA-256 hex digest")
        if hashlib.sha256(raw).hexdigest() != self.file_hash:
            raise errors.InputError("file_hash does not match encoded_bytes")
        object.__setattr__(self, "byte_size", len(raw))

    @classmethod
    def from_bytes(
        cls,
        encoded_bytes: bytes,
        metadata: Metadata,
        source: str = "memory://",
        max_bytes: int | None = 2 * 1024 * 1024 * 1024,
        label: str | float | bool | None = None,
        group_id: str | None = None,
        observation_time: datetime | str | None = None,
    ) -> Self:
        """Creates a sample from bytes and explicit probe output.

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
            VideoError: If the payload is empty, or exceeds
                ``max_bytes`` (or ``max_bytes`` is below one).
        """
        raw = bytes(encoded_bytes)
        if not raw:
            raise VideoError(
                "video payload is empty", {"code": "empty_payload"}
            )
        if max_bytes is not None and (max_bytes < 1 or len(raw) > max_bytes):
            raise VideoError(
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

    def profile(self) -> Profile:
        """Returns bounded metadata suitable for curation and data cards.

        Returns:
            The profile of this sample.
        """
        return Profile(
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
            synchrony=self.metadata.synchrony,
            observation_time=self.observation_time,
        )
