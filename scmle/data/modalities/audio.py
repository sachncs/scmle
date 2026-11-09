"""Dependency-light audio data contracts and bounded profiling.

The core package decodes WAV files with the Python standard library. Other
audio containers must provide an explicit decoder/probe function so that a
codec dependency never becomes an implicit trust or reproducibility boundary.
"""

from collections.abc import Callable
import dataclasses
import hashlib
import io
import math
import pathlib
import random
from typing import Self
import wave

import numpy

from scmle.contracts import data
from scmle.contracts import errors
from scmle.data.modalities.curation import build_finding
from scmle.data.modalities.curation import conflicting_label_groups
from scmle.data.modalities.curation import label_counts
from scmle.data.modalities.curation import ModalityCurationReport


class AudioValidationError(errors.DataError):
    """An audio sample violates an ingestion or metadata invariant."""

    category = "audio_validation"


class AudioDependencyError(errors.DataError):
    """A non-WAV audio operation needs an explicitly supplied decoder."""

    category = "audio_dependency"


@dataclasses.dataclass(frozen=True, slots=True)
class AudioMetadata:
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
            raise ValueError("format must not be empty")
        if self.sample_rate < 1:
            raise ValueError("sample_rate must be positive")
        if self.channels < 1:
            raise ValueError("channels must be positive")
        if self.frame_count < 1:
            raise ValueError("frame_count must be positive")
        if self.sample_width is not None and self.sample_width < 1:
            raise ValueError("sample_width must be positive when provided")

    @property
    def duration_seconds(self) -> float:
        """The decoded duration in seconds, never rounded to whole seconds."""
        return self.frame_count / self.sample_rate


@dataclasses.dataclass(frozen=True, slots=True)
class AudioTranscriptSegment:
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
            raise ValueError("transcript segment time bounds are invalid")
        if not self.text.strip():
            raise ValueError("transcript segment text must not be empty")
        if self.confidence is not None and (
            isinstance(self.confidence, bool)
            or not math.isfinite(self.confidence)
            or not 0.0 <= self.confidence <= 1.0
        ):
            raise ValueError("transcript confidence must be in [0, 1]")


@dataclasses.dataclass(frozen=True, slots=True)
class AudioTranscript:
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
    segments: tuple[AudioTranscriptSegment, ...] = ()
    confidence: float | None = None

    def __post_init__(self) -> None:
        """Validates provenance, ordering, and optional confidence.

        Raises:
            ValueError: If a text field is blank, the confidence is a bool,
                not finite or outside [0, 1], or the segments are not
                ordered by start time.
            TypeError: If segments is not a tuple of
                AudioTranscriptSegment.
        """
        for field_name, value in (
            ("text", self.text),
            ("source", self.source),
            ("provider", self.provider),
            ("provider_version", self.provider_version),
            ("language", self.language),
        ):
            if not value.strip():
                raise ValueError(f"transcript {field_name} must not be empty")
        if self.confidence is not None and (
            isinstance(self.confidence, bool)
            or not math.isfinite(self.confidence)
            or not 0.0 <= self.confidence <= 1.0
        ):
            raise ValueError("transcript confidence must be in [0, 1]")
        if not isinstance(self.segments, tuple):
            raise TypeError("segments must be a tuple")
        if not all(
            isinstance(segment, AudioTranscriptSegment)
            for segment in self.segments
        ):
            raise TypeError("segments must contain transcript segments")
        if any(
            current.start_seconds < previous.start_seconds
            for previous, current in zip(
                self.segments, self.segments[1:], strict=False
            )
        ):
            raise ValueError(
                "transcript segments must be ordered by start time"
            )

    def to_dict(self) -> dict[str, object]:
        """Returns JSON-native transcript provenance.

        Returns:
            A dict of the fields, with segments as nested dicts.
        """
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, slots=True)
class AudioProfile:
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


@dataclasses.dataclass(frozen=True, slots=True)
class AudioFeatures:
    """Bounded numeric signal features with their observation boundary.

    Attributes:
        sources: Source identifier of each row.
        feature_names: Name of each column.
        values: One tuple of feature values per source.
        frames_observed: Number of frames each row was computed from.
        frames_truncated: Whether each row saw fewer frames than the clip
            holds.
    """

    sources: tuple[str, ...]
    feature_names: tuple[str, ...]
    values: tuple[tuple[float, ...], ...]
    frames_observed: tuple[int, ...]
    frames_truncated: tuple[bool, ...]

    @property
    def shape(self) -> tuple[int, int]:
        """The number of samples and the number of features, as a pair."""
        return (len(self.sources), len(self.feature_names))

    def to_numpy(self) -> numpy.ndarray:
        """Returns a writable copy of the feature matrix.

        Returns:
            A float64 array of shape ``(samples, features)``.
        """
        return numpy.asarray(self.values, dtype=numpy.float64).copy()


@dataclasses.dataclass(frozen=True, slots=True)
class AudioSplit:
    """Group-aware audio split that keeps related clips together.

    Attributes:
        train: Samples assigned to training.
        validation: Samples assigned to validation.
        test: Samples assigned to testing.
    """

    train: tuple["AudioSample", ...]
    validation: tuple["AudioSample", ...]
    test: tuple["AudioSample", ...]

    def group_ids(self) -> dict[str, tuple[str, ...]]:
        """Returns the stable groups assigned to each partition.

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


AudioProbe = Callable[[bytes], AudioMetadata]


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


def wav_metadata(encoded_bytes: bytes) -> AudioMetadata:
    """Reads WAV metadata using the standard library.

    Args:
        encoded_bytes: A complete WAV file.

    Returns:
        The format (``wav``), sample rate, channel count, frame count and
        sample width in bytes.

    Raises:
        AudioValidationError: If the payload is not a readable WAV file.
    """
    try:
        with wave.open(io.BytesIO(encoded_bytes), "rb") as opened:
            return AudioMetadata(
                format="wav",
                sample_rate=opened.getframerate(),
                channels=opened.getnchannels(),
                frame_count=opened.getnframes(),
                sample_width=opened.getsampwidth(),
            )
    except (EOFError, wave.Error) as error:
        raise AudioValidationError(
            "audio payload is not a readable WAV file",
            {"code": "invalid_wav"},
        ) from error


def wav_signal_features(
    encoded_bytes: bytes,
    max_frames: int,
    start_frame: int = 0,
) -> tuple[tuple[float, ...], int]:
    """Extracts bounded, normalized PCM statistics from one WAV payload.

    The returned statistics describe the bounded window beginning at
    'start_frame'. Callers receive the observed frame count so a truncated
    boundary cannot be mistaken for a full-clip measurement.

    Args:
        encoded_bytes: A complete WAV file.
        max_frames: Largest number of frames to read; at least one.
        start_frame: Index of the first frame to read; not negative.

    Returns:
        A pair. The first item holds nine statistics of the window, with
        samples scaled to [-1, 1]: mean, mean absolute value, RMS, peak
        absolute value, share of samples at or above 0.999 in magnitude,
        share at or below 0.01, window duration in seconds, frame count
        and PCM scale. The second item is the number of frames read.

    Raises:
        ValueError: If ``max_frames`` is below one or ``start_frame`` is
            negative.
        AudioValidationError: If the payload is not a readable WAV file,
            ``start_frame`` is at or beyond the last frame, the window has
            no complete frame, the sample width is not 1, 2, 3 or 4 bytes,
            or a statistic is not finite.
    """
    if max_frames < 1:
        raise ValueError("max_frames must be at least one")
    if start_frame < 0:
        raise ValueError("start_frame must not be negative")
    try:
        with wave.open(io.BytesIO(encoded_bytes), "rb") as opened:
            sample_width = opened.getsampwidth()
            channels = opened.getnchannels()
            sample_rate = opened.getframerate()
            if start_frame >= opened.getnframes():
                raise AudioValidationError(
                    "audio window starts beyond the available frames",
                    {
                        "code": "window_start_out_of_range",
                        "start_frame": start_frame,
                    },
                )
            opened.setpos(start_frame)
            requested_frames = min(
                opened.getnframes() - start_frame,
                max_frames,
            )
            raw = opened.readframes(requested_frames)
    except (EOFError, wave.Error) as error:
        raise AudioValidationError(
            "audio payload is not a readable WAV file",
            {"code": "invalid_wav"},
        ) from error

    frame_width = sample_width * channels
    complete_bytes = len(raw) - (len(raw) % frame_width)
    raw = raw[:complete_bytes]
    if not raw:
        raise AudioValidationError(
            "WAV payload contains no complete audio frames",
            {"code": "empty_audio_frames"},
        )
    frame_count = len(raw) // frame_width
    if sample_width == 1:
        samples = numpy.frombuffer(raw, dtype=numpy.uint8).astype(numpy.float64)
        samples = (samples - 128.0) / 128.0
        scale = 1.0
    elif sample_width in (2, 4):
        dtype = numpy.dtype("<i2" if sample_width == 2 else "<i4")
        samples = numpy.frombuffer(raw, dtype=dtype).astype(numpy.float64)
        scale = float(2 ** (sample_width * 8 - 1))
        samples /= scale
    elif sample_width == 3:
        packed = numpy.frombuffer(raw, dtype=numpy.uint8).reshape(-1, 3)
        unsigned = (
            packed[:, 0].astype(numpy.int64)
            | (packed[:, 1].astype(numpy.int64) << 8)
            | (packed[:, 2].astype(numpy.int64) << 16)
        )
        signed = unsigned - ((unsigned & 0x800000) << 1)
        samples = signed.astype(numpy.float64)
        scale = float(2**23)
        samples /= scale
    else:
        raise AudioValidationError(
            "WAV sample width is unsupported",
            {"code": "unsupported_sample_width", "sample_width": sample_width},
        )

    matrix = samples.reshape(-1, channels)
    absolute = numpy.abs(matrix)
    frame_values = numpy.asarray(
        [
            float(matrix.mean()),
            float(absolute.mean()),
            float(numpy.sqrt(numpy.mean(numpy.square(matrix)))),
            float(absolute.max()),
            float(numpy.mean(absolute >= 0.999)),
            float(numpy.mean(absolute <= 0.01)),
            float(frame_count / sample_rate),
            float(frame_count),
            float(scale),
        ],
        dtype=numpy.float64,
    )
    if not numpy.isfinite(frame_values).all():
        raise AudioValidationError(
            "WAV signal features must be finite",
            {"code": "non_finite_features"},
        )
    return tuple(float(value) for value in frame_values), frame_count


def wav_pcm_values(
    encoded_bytes: bytes,
) -> tuple[tuple[int, int, int, int, str, str], numpy.ndarray]:
    """Decodes WAV PCM values and retain the writer parameters.

    Values are normalized to ``[-1, 1]`` and shaped as frames by channels.
    This is a bounded dependency-light primitive for caller-owned perturbation
    probes; it is not a general-purpose audio decoder.

    Args:
        encoded_bytes: A complete WAV file.

    Returns:
        A pair. The first item is the writer parameters as channels, sample
        width in bytes, frame rate, frame count, compression type and
        compression name. The second is a float64 array of shape
        ``(frames, channels)`` with values in [-1, 1].

    Raises:
        AudioValidationError: If the payload is not a readable WAV file,
            holds no complete frame, has an unsupported sample width, or
            decodes to non-finite values.
    """
    try:
        with wave.open(io.BytesIO(encoded_bytes), "rb") as opened:
            parameters = (
                opened.getnchannels(),
                opened.getsampwidth(),
                opened.getframerate(),
                opened.getnframes(),
                opened.getcomptype(),
                opened.getcompname(),
            )
            raw = opened.readframes(opened.getnframes())
    except (EOFError, wave.Error) as error:
        raise AudioValidationError(
            "audio payload is not a readable WAV file",
            {"code": "invalid_wav"},
        ) from error
    channels = parameters[0]
    sample_width = parameters[1]
    frame_width = sample_width * channels
    complete_bytes = len(raw) - (len(raw) % frame_width)
    raw = raw[:complete_bytes]
    if not raw:
        raise AudioValidationError(
            "WAV payload contains no complete audio frames",
            {"code": "empty_audio_frames"},
        )
    if sample_width == 1:
        samples = numpy.frombuffer(raw, dtype=numpy.uint8).astype(numpy.float64)
        samples = (samples - 128.0) / 128.0
    elif sample_width in (2, 4):
        dtype = numpy.dtype("<i2" if sample_width == 2 else "<i4")
        samples = numpy.frombuffer(raw, dtype=dtype).astype(numpy.float64)
        samples /= float(2 ** (sample_width * 8 - 1))
    elif sample_width == 3:
        packed = numpy.frombuffer(raw, dtype=numpy.uint8).reshape(-1, 3)
        unsigned = (
            packed[:, 0].astype(numpy.int64)
            | (packed[:, 1].astype(numpy.int64) << 8)
            | (packed[:, 2].astype(numpy.int64) << 16)
        )
        signed = unsigned - ((unsigned & 0x800000) << 1)
        samples = signed.astype(numpy.float64) / float(2**23)
    else:
        raise AudioValidationError(
            "WAV sample width is unsupported",
            {"code": "unsupported_sample_width", "sample_width": sample_width},
        )
    values = samples.reshape(-1, channels)
    if not numpy.isfinite(values).all():
        raise AudioValidationError(
            "WAV PCM values must be finite",
            {"code": "non_finite_pcm"},
        )
    return parameters, values


def wav_payload_from_pcm(
    parameters: tuple[int, int, int, int, str, str], values: numpy.ndarray
) -> bytes:
    """Encodes normalized PCM values with the supplied WAV parameters.

    Values are clipped to [-1, 1] and rounded to the sample width given in
    ``parameters``.

    Args:
        parameters: Writer parameters in the layout `wav_pcm_values`
            returns: channels, sample width in bytes, frame rate, frame
            count, compression type and compression name.
        values: Frame-by-channel matrix of normalized values.

    Returns:
        The bytes of a WAV file holding the values.

    Raises:
        ValueError: If ``values`` is not a finite two-dimensional matrix.
        AudioValidationError: If the sample width is not 1, 2, 3 or 4.
    """
    matrix = numpy.asarray(values, dtype=numpy.float64)
    if matrix.ndim != 2 or not numpy.isfinite(matrix).all():
        raise ValueError("PCM values must be a finite frame-by-channel matrix")
    sample_width = parameters[1]
    clipped = numpy.clip(matrix, -1.0, 1.0)
    if sample_width == 1:
        encoded = numpy.rint(clipped * 127.0 + 128.0).astype(numpy.uint8)
    elif sample_width == 2:
        encoded = numpy.rint(clipped * 32767.0).astype("<i2")
    elif sample_width == 3:
        integer = numpy.rint(clipped * (2**23 - 1)).astype(numpy.int32)
        unsigned = integer.astype(numpy.int64) & 0xFFFFFF
        encoded = numpy.empty((*integer.shape, 3), dtype=numpy.uint8)
        encoded[..., 0] = unsigned & 0xFF
        encoded[..., 1] = (unsigned >> 8) & 0xFF
        encoded[..., 2] = (unsigned >> 16) & 0xFF
        raw = encoded.tobytes()
    elif sample_width == 4:
        encoded = numpy.rint(clipped * (2**31 - 1)).astype("<i4")
    else:
        raise AudioValidationError(
            "WAV sample width is unsupported",
            {"code": "unsupported_sample_width", "sample_width": sample_width},
        )
    if sample_width != 3:
        raw = encoded.tobytes()
    payload = io.BytesIO()
    with wave.open(payload, "wb") as writer:
        writer.setparams(parameters)
        writer.writeframes(raw)
    return payload.getvalue()


@dataclasses.dataclass(frozen=True, slots=True)
class AudioSample:
    """Immutable encoded audio plus the metadata observed at ingestion.

    Attributes:
        source: Non-blank source label or URI.
        encoded_bytes: The non-empty encoded payload.
        file_hash: SHA-256 hex digest of the payload.
        metadata: Metadata observed at ingestion.
        label: Optional task label.
        group_id: Optional identity used to keep related clips in one split.
        transcript: Optional transcript; no segment may end after the audio.
        byte_size: Length of the payload; set on construction.
    """

    source: str
    encoded_bytes: bytes = dataclasses.field(repr=False)
    file_hash: str
    metadata: AudioMetadata
    label: str | int | float | bool | None = None
    group_id: str | None = None
    transcript: AudioTranscript | None = None
    byte_size: int = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        """Validates bytes, source identity, and hash integrity.

        Raises:
            ValueError: If source or group_id is blank, or file_hash is not
                a lowercase SHA-256 hex digest of the payload.
            AudioValidationError: If a transcript segment ends after the
                audio does, or the payload is empty.
        """
        raw = bytes(self.encoded_bytes)
        object.__setattr__(self, "encoded_bytes", raw)
        if not self.source.strip():
            raise ValueError("source must not be empty")
        if self.group_id is not None and not self.group_id.strip():
            raise ValueError("group_id must not be empty when provided")
        if self.transcript is not None and any(
            segment.end_seconds > self.metadata.duration_seconds + 1e-9
            for segment in self.transcript.segments
        ):
            raise AudioValidationError(
                "transcript segment exceeds audio duration",
                {"code": "transcript_out_of_range", "source": self.source},
            )
        if not raw:
            raise AudioValidationError(
                "audio payload is empty", {"code": "empty_payload"}
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
        source: str = "memory://",
        probe: AudioProbe | None = None,
        max_bytes: int | None = 500 * 1024 * 1024,
        label: str | int | float | bool | None = None,
        group_id: str | None = None,
        transcript: AudioTranscript | None = None,
    ) -> Self:
        """Creates a sample from bytes and an explicit codec probe.

        WAV uses the standard-library probe by default. Every other format
        requires a caller-owned probe, making optional codec behavior visible
        in the run configuration and evidence.

        Args:
            encoded_bytes: The encoded audio payload.
            source: Source label or URI, normalised with `canonical_source`.
            probe: Function that reads metadata from the payload. When None
                the payload must be WAV.
            max_bytes: Largest payload size in bytes, or None for no limit.
            label: Optional task label.
            group_id: Optional identity used to prevent split leakage.
            transcript: Optional transcript.

        Returns:
            The new sample.

        Raises:
            AudioValidationError: If the payload is empty, exceeds
                ``max_bytes`` (or ``max_bytes`` is below one), or the probe
                raises it.
            AudioDependencyError: If no probe is given for a payload that is
                not WAV, or the probe fails with any other exception.
        """
        raw = bytes(encoded_bytes)
        if not raw:
            raise AudioValidationError(
                "audio payload is empty", {"code": "empty_payload"}
            )
        if max_bytes is not None and (max_bytes < 1 or len(raw) > max_bytes):
            raise AudioValidationError(
                "audio payload exceeds the configured byte limit",
                {
                    "code": "encoded_size_limit",
                    "byte_size": len(raw),
                    "max_bytes": max_bytes,
                },
            )
        if probe is None and not (raw[:4] == b"RIFF" and raw[8:12] == b"WAVE"):
            raise AudioDependencyError(
                "this audio container requires an explicit codec probe"
            )
        try:
            metadata = (probe or wav_metadata)(raw)
        except AudioValidationError:
            raise
        except Exception as error:
            raise AudioDependencyError(
                "the supplied audio probe could not decode the payload"
            ) from error
        stable_source = canonical_source(source)
        return cls(
            source=stable_source,
            encoded_bytes=raw,
            file_hash=hashlib.sha256(raw).hexdigest(),
            metadata=metadata,
            label=label,
            group_id=group_id,
            transcript=transcript,
        )

    def profile(self) -> AudioProfile:
        """Returns bounded metadata suitable for curation and data cards.

        Returns:
            The profile of this sample.
        """
        return AudioProfile(
            source=self.source,
            format=self.metadata.format,
            sample_rate=self.metadata.sample_rate,
            channels=self.metadata.channels,
            frame_count=self.metadata.frame_count,
            duration_seconds=self.metadata.duration_seconds,
            byte_size=self.byte_size,
            transcript_present=self.transcript is not None,
            transcript_segment_count=(
                len(self.transcript.segments) if self.transcript else 0
            ),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class AudioDataset:
    """Immutable audio collection with deterministic identity and splitting.

    Attributes:
        samples: The samples, in caller order.
        content_hash: SHA-256 hex digest of the sources and file hashes, in
            sample order; set on construction.
    """

    samples: tuple[AudioSample, ...]
    content_hash: str = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        """Rejects duplicate source identities and derive dataset identity.

        Raises:
            AudioValidationError: If there are no samples or two samples
                share a source.
        """
        normalized = tuple(self.samples)
        if not normalized:
            raise AudioValidationError(
                "audio dataset must contain at least one sample",
                {"code": "empty_dataset"},
            )
        sources = [sample.source for sample in normalized]
        if len(sources) != len(set(sources)):
            raise AudioValidationError(
                "audio dataset contains duplicate sources",
                {"code": "duplicate_source"},
            )
        digest = hashlib.sha256(
            "\n".join(
                f"{sample.source}\0{sample.file_hash}" for sample in normalized
            ).encode("utf-8")
        ).hexdigest()
        object.__setattr__(self, "samples", normalized)
        object.__setattr__(self, "content_hash", digest)

    def profiles(self) -> tuple[AudioProfile, ...]:
        """Returns profiles in sample order for deterministic curation.

        Returns:
            One profile per sample.
        """
        return tuple(sample.profile() for sample in self.samples)

    def prepare_features(
        self,
        max_frames: int = 100_000,
        start_frame: int = 0,
    ) -> AudioFeatures:
        """Prepares bounded PCM and metadata features for an audio baseline.

        Core decoding is intentionally limited to WAV. Samples ingested with
        a caller-owned probe remain valid data contracts, but require a codec
        adapter before signal features can be prepared. The feature rows use
        only an observed prefix and record whether that prefix was truncated.

        Args:
            max_frames: Largest number of frames read per sample; at least
                one.
            start_frame: Index of the first frame read; not negative.

        Returns:
            Thirteen features per sample: nine signal statistics (see
            `wav_signal_features`), then duration in seconds, sample rate,
            channel count and sample width (0 when unknown).

        Raises:
            ValueError: If ``max_frames`` is below one or ``start_frame`` is
                negative.
            AudioDependencyError: If a sample is not in WAV format.
        """
        if max_frames < 1:
            raise ValueError("max_frames must be at least one")
        if start_frame < 0:
            raise ValueError("start_frame must not be negative")
        feature_names = (
            "mean_amplitude",
            "mean_absolute_amplitude",
            "rms_amplitude",
            "peak_amplitude",
            "clipping_fraction",
            "silence_fraction",
            "observed_duration_seconds",
            "frames_observed",
            "pcm_scale",
            "duration_seconds",
            "sample_rate",
            "channels",
            "sample_width",
        )
        rows = []
        frames_observed = []
        frames_truncated = []
        for sample in self.samples:
            if sample.metadata.format.lower() != "wav":
                raise AudioDependencyError(
                    "audio signal features require a WAV decoder for "
                    f"'{sample.source}'"
                )
            signal_values, observed = wav_signal_features(
                sample.encoded_bytes, max_frames, start_frame
            )
            rows.append(
                (
                    *signal_values,
                    sample.metadata.duration_seconds,
                    float(sample.metadata.sample_rate),
                    float(sample.metadata.channels),
                    float(sample.metadata.sample_width or 0),
                )
            )
            frames_observed.append(observed)
            frames_truncated.append(observed < sample.metadata.frame_count)
        return AudioFeatures(
            sources=tuple(sample.source for sample in self.samples),
            feature_names=feature_names,
            values=tuple(rows),
            frames_observed=tuple(frames_observed),
            frames_truncated=tuple(frames_truncated),
        )

    def split(
        self,
        validation_fraction: float = 0.2,
        test_fraction: float = 0.2,
        seed: int = 0,
    ) -> AudioSplit:
        """Splits clips by explicit group identity to prevent leakage.

        A caller should use a speaker, session, device, or other identity as
        ``group_id``. Without one, source identity is the conservative group
        and repeated clips are not assumed to share a speaker.

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
        grouped: dict[str, list[AudioSample]] = {}
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

        def samples_for(excluded: set[str]) -> tuple[AudioSample, ...]:
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
        return AudioSplit(
            train=samples_for(test_groups | validation_groups),
            validation=validation,
            test=test,
        )

    def curate(self) -> ModalityCurationReport:
        """Builds reviewable metadata and compatibility findings.

        Returns:
            A curation report for modality ``audio`` with format, sample
            rate, channel, duration, label and transcript statistics, and
            findings for missing transcript confidence, format drift,
            sample-rate or channel drift, mixed labelled and unlabelled
            clips, and conflicting labels.
        """
        profiles = self.profiles()
        formats = sorted({profile.format for profile in profiles})
        sample_rates = sorted({profile.sample_rate for profile in profiles})
        channels = sorted({profile.channels for profile in profiles})
        durations = [profile.duration_seconds for profile in profiles]
        labels = label_counts(sample.label for sample in self.samples)
        conflicts = conflicting_label_groups(
            (sample.file_hash, sample.source, sample.label)
            for sample in self.samples
        )
        quality_findings = []
        transcripts = [
            sample.transcript
            for sample in self.samples
            if sample.transcript is not None
        ]
        transcript_confidences = [
            confidence
            for transcript in transcripts
            for confidence in (
                [transcript.confidence]
                + [segment.confidence for segment in transcript.segments]
            )
            if confidence is not None
        ]
        missing_transcript_confidence = sum(
            transcript.confidence is None
            and all(
                segment.confidence is None for segment in transcript.segments
            )
            for transcript in transcripts
        )
        if missing_transcript_confidence:
            quality_findings.append(
                build_finding(
                    "quality.audio-transcript-confidence-missing",
                    data.FindingSeverity.WARNING,
                    {
                        "sources": [
                            sample.source
                            for sample in self.samples
                            if sample.transcript is not None
                        ]
                    },
                    {"sample_count": missing_transcript_confidence},
                    "Obtain confidence from the transcript provider or "
                    "record why confidence is unavailable before using "
                    "transcript-dependent claims.",
                    False,
                )
            )
        if len(formats) > 1:
            quality_findings.append(
                build_finding(
                    "quality.audio-format-drift",
                    data.FindingSeverity.WARNING,
                    {"sources": [profile.source for profile in profiles]},
                    {"formats": formats},
                    "Select an explicit codec normalization policy before "
                    "training.",
                    False,
                )
            )
        if len(sample_rates) > 1 or len(channels) > 1:
            quality_findings.append(
                build_finding(
                    "quality.audio-shape-drift",
                    data.FindingSeverity.WARNING,
                    {"sources": [profile.source for profile in profiles]},
                    {
                        "sample_rates": sample_rates,
                        "channels": channels,
                    },
                    "Resample or channel-normalize with the transformation "
                    "recorded.",
                    False,
                )
            )
        risk_findings = []
        if "unlabeled" in labels and len(labels) > 1:
            risk_findings.append(
                build_finding(
                    "risk.audio-mixed-label-state",
                    data.FindingSeverity.WARNING,
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
                    "risk.audio-conflicting-duplicate-labels",
                    data.FindingSeverity.ERROR,
                    {"dataset": self.content_hash},
                    {"groups": list(conflicts)},
                    "Resolve the label conflict or explicitly exclude the "
                    "affected audio content.",
                    True,
                )
            )
        return ModalityCurationReport(
            modality="audio",
            dataset_hash=self.content_hash,
            sample_count=len(self.samples),
            profile={
                "formats": formats,
                "sample_rates": sample_rates,
                "channels": channels,
                "duration_min_seconds": min(durations),
                "duration_max_seconds": max(durations),
                "duration_mean_seconds": sum(durations) / len(durations),
                "total_bytes": sum(profile.byte_size for profile in profiles),
                "label_counts": labels,
                "conflicting_label_groups": list(conflicts),
                "transcript_coverage": len(transcripts) / len(self.samples),
                "transcript_sources": sorted(
                    {transcript.source for transcript in transcripts}
                ),
                "transcript_providers": sorted(
                    {transcript.provider for transcript in transcripts}
                ),
                "transcript_languages": sorted(
                    {transcript.language for transcript in transcripts}
                ),
                "transcript_confidence_min": (
                    min(transcript_confidences)
                    if transcript_confidences
                    else None
                ),
                "transcript_confidence_max": (
                    max(transcript_confidences)
                    if transcript_confidences
                    else None
                ),
                "transcript_segment_count": sum(
                    len(transcript.segments) for transcript in transcripts
                ),
            },
            quality_findings=tuple(quality_findings),
            risk_findings=tuple(risk_findings),
        )


__all__ = [
    "AudioDataset",
    "AudioDependencyError",
    "AudioFeatures",
    "AudioMetadata",
    "AudioProbe",
    "AudioProfile",
    "AudioSample",
    "AudioSplit",
    "AudioTranscript",
    "AudioTranscriptSegment",
    "AudioValidationError",
    "canonical_source",
    "wav_metadata",
    "wav_signal_features",
]
