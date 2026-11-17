"""Audio metadata, transcripts, profiles and their errors."""

import dataclasses
import math

from scmle.contracts import errors

SHA256_HEX_LENGTH = 64


class ValidationError(errors.DataError):
    """An audio sample violates an ingestion or metadata invariant."""

    category = "audio_validation"


class DecoderError(errors.DataError):
    """A non-WAV audio operation needs an explicitly supplied decoder."""

    category = "audio_dependency"


@dataclasses.dataclass(frozen=True, slots=True)
class Metadata:
    """Decoded metadata required to identify an audio sample.

    Attributes:
        format: Non-empty container or codec name, such as ``wav``.
        sample_rate: Frames per second; positive.
        channels: Number of channels; positive.
        frame_count: Number of frames; positive.
        sample_width: Bytes per sample, or None when unknown; positive when
            given.
    """

    format: str
    sample_rate: int
    channels: int
    frame_count: int
    sample_width: int | None = None

    def __post_init__(self) -> None:
        """Validates metadata before it enters a dataset contract.

        Raises:
            ValueError: If format is blank, or sample_rate, channels,
                frame_count or a given sample_width is below one.
        """
        if not self.format.strip():
            raise errors.InputError("format must not be empty")
        if self.sample_rate < 1:
            raise errors.InputError("sample_rate must be positive")
        if self.channels < 1:
            raise errors.InputError("channels must be positive")
        if self.frame_count < 1:
            raise errors.InputError("frame_count must be positive")
        if self.sample_width is not None and self.sample_width < 1:
            raise errors.InputError(
                "sample_width must be positive when provided"
            )

    @property
    def duration_seconds(self) -> float:
        """The decoded duration in seconds, never rounded to whole seconds."""
        return self.frame_count / self.sample_rate


@dataclasses.dataclass(frozen=True, slots=True)
class Segment:
    """One time-local transcript segment with provenance and confidence.

    Attributes:
        start_seconds: Start offset in seconds; finite and non-negative.
        end_seconds: End offset in seconds; finite and after the start.
        text: Non-blank transcribed text.
        confidence: Optional confidence in [0, 1].
    """

    start_seconds: float
    end_seconds: float
    text: str
    confidence: float | None = None

    def __post_init__(self) -> None:
        """Validates time bounds, text, and optional confidence.

        Raises:
            ValueError: If the bounds are not finite, start is negative or
                end is not after start, the text is blank, or the confidence
                is a bool, not finite or outside [0, 1].
        """
        if (
            not math.isfinite(self.start_seconds)
            or not math.isfinite(self.end_seconds)
            or self.start_seconds < 0
            or self.end_seconds <= self.start_seconds
        ):
            raise errors.InputError(
                "transcript segment time bounds are invalid"
            )
        if not self.text.strip():
            raise errors.InputError("transcript segment text must not be empty")
        if self.confidence is not None and (
            isinstance(self.confidence, bool)
            or not math.isfinite(self.confidence)
            or not 0.0 <= self.confidence <= 1.0
        ):
            raise errors.InputError("transcript confidence must be in [0, 1]")


@dataclasses.dataclass(frozen=True, slots=True)
class Transcript:
    """Immutable transcript output tied to its extraction provenance.

    Attributes:
        text: Non-blank full transcript text.
        source: Non-blank identifier of the transcribed audio.
        provider: Non-blank name of the transcription provider.
        provider_version: Non-blank version of the provider.
        language: Non-blank language of the transcript.
        segments: Time-local segments, ordered by start time.
        confidence: Optional overall confidence in [0, 1].
    """

    text: str
    source: str
    provider: str
    provider_version: str
    language: str
    segments: tuple[Segment, ...] = ()
    confidence: float | None = None

    def __post_init__(self) -> None:
        """Validates provenance, ordering, and optional confidence.

        Raises:
            ValueError: If a text field is blank, the confidence is a bool,
                not finite or outside [0, 1], or the segments are not
                ordered by start time.
            TypeError: If segments is not a tuple of
                Segment.
        """
        for field_name, value in (
            ("text", self.text),
            ("source", self.source),
            ("provider", self.provider),
            ("provider_version", self.provider_version),
            ("language", self.language),
        ):
            if not value.strip():
                raise errors.InputError(
                    f"transcript {field_name} must not be empty"
                )
        if self.confidence is not None and (
            isinstance(self.confidence, bool)
            or not math.isfinite(self.confidence)
            or not 0.0 <= self.confidence <= 1.0
        ):
            raise errors.InputError("transcript confidence must be in [0, 1]")
        if not isinstance(self.segments, tuple):
            raise errors.KindError("segments must be a tuple")
        if not all(isinstance(segment, Segment) for segment in self.segments):
            raise errors.KindError("segments must contain transcript segments")
        if any(
            current.start_seconds < previous.start_seconds
            for previous, current in zip(
                self.segments, self.segments[1:], strict=False
            )
        ):
            raise errors.InputError(
                "transcript segments must be ordered by start time"
            )

    def to_dict(self) -> dict[str, object]:
        """Returns JSON-native transcript provenance.

        Returns:
            A dict of the fields, with segments as nested dicts.
        """
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, slots=True)
class Profile:
    """Bounded data-understanding summary for one audio sample.

    Attributes:
        source: Source identifier of the sample.
        format: Container or codec name.
        sample_rate: Frames per second.
        channels: Number of channels.
        frame_count: Number of frames.
        duration_seconds: Duration in seconds.
        byte_size: Size of the encoded payload in bytes.
        transcript_present: Whether a transcript is attached.
        transcript_segment_count: Number of transcript segments.
    """

    source: str
    format: str
    sample_rate: int
    channels: int
    frame_count: int
    duration_seconds: float
    byte_size: int
    transcript_present: bool = False
    transcript_segment_count: int = 0
