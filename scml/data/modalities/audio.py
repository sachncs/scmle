"""Dependency-light audio data contracts and bounded profiling.

The core package decodes WAV files with the Python standard library. Other
audio containers must provide an explicit decoder/probe function so that a
codec dependency never becomes an implicit trust or reproducibility boundary.
"""

from collections.abc import Callable
import dataclasses
import hashlib
import io
import pathlib
from typing import Self
import wave

import numpy as np

from scml.contracts import data as data_contracts
from scml.contracts import errors
from scml.data.modalities.curation import build_finding
from scml.data.modalities.curation import dataset_hash
from scml.data.modalities.curation import label_counts
from scml.data.modalities.curation import ModalityCurationReport


class AudioValidationError(errors.DataError):
    """An audio sample violates an ingestion or metadata invariant."""

    category = "audio_validation"


class AudioDependencyError(errors.DataError):
    """A non-WAV audio operation needs an explicitly supplied decoder."""

    category = "audio_dependency"


@dataclasses.dataclass(frozen=True, slots=True)
class AudioMetadata:
    """Decoded metadata required to identify an audio sample."""

    format: str
    sample_rate: int
    channels: int
    frame_count: int
    sample_width: int | None = None

    def __post_init__(self) -> None:
        """Validate metadata before it enters a dataset contract."""
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
        """Return the decoded duration without rounding away short clips."""
        return self.frame_count / self.sample_rate


@dataclasses.dataclass(frozen=True, slots=True)
class AudioProfile:
    """Bounded data-understanding summary for one audio sample."""

    source: str
    format: str
    sample_rate: int
    channels: int
    frame_count: int
    duration_seconds: float
    byte_size: int


@dataclasses.dataclass(frozen=True, slots=True)
class AudioFeatures:
    """Bounded numeric signal features with their observation boundary."""

    sources: tuple[str, ...]
    feature_names: tuple[str, ...]
    values: tuple[tuple[float, ...], ...]
    frames_observed: tuple[int, ...]
    frames_truncated: tuple[bool, ...]

    @property
    def shape(self) -> tuple[int, int]:
        """Return the number of samples and features."""
        return (len(self.sources), len(self.feature_names))

    def to_numpy(self) -> np.ndarray:
        """Return a writable copy of the feature matrix."""
        return np.asarray(self.values, dtype=np.float64).copy()


AudioProbe = Callable[[bytes], AudioMetadata]


def canonical_source(source: str | pathlib.Path) -> str:
    """Return a stable source identifier for a path or URI-like value."""
    value = str(source).strip()
    if not value:
        return "memory://"
    if "://" in value:
        return value
    return pathlib.Path(value).expanduser().resolve(strict=False).as_uri()


def wav_metadata(encoded_bytes: bytes) -> AudioMetadata:
    """Read WAV metadata using the standard library."""
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
) -> tuple[tuple[float, ...], int]:
    """Extract bounded, normalized PCM statistics from one WAV payload.

    The returned statistics describe the observed prefix when ``max_frames``
    is smaller than the clip. Callers receive the observed frame count so the
    truncation boundary cannot be mistaken for a full-clip measurement.
    """
    try:
        with wave.open(io.BytesIO(encoded_bytes), "rb") as opened:
            sample_width = opened.getsampwidth()
            channels = opened.getnchannels()
            sample_rate = opened.getframerate()
            requested_frames = min(opened.getnframes(), max_frames)
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
        samples = np.frombuffer(raw, dtype=np.uint8).astype(np.float64)
        samples = (samples - 128.0) / 128.0
        scale = 1.0
    elif sample_width in (2, 4):
        dtype = np.dtype("<i2" if sample_width == 2 else "<i4")
        samples = np.frombuffer(raw, dtype=dtype).astype(np.float64)
        scale = float(2 ** (sample_width * 8 - 1))
        samples /= scale
    elif sample_width == 3:
        packed = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        unsigned = (
            packed[:, 0].astype(np.int64)
            | (packed[:, 1].astype(np.int64) << 8)
            | (packed[:, 2].astype(np.int64) << 16)
        )
        signed = unsigned - ((unsigned & 0x800000) << 1)
        samples = signed.astype(np.float64)
        scale = float(2**23)
        samples /= scale
    else:
        raise AudioValidationError(
            "WAV sample width is unsupported",
            {"code": "unsupported_sample_width", "sample_width": sample_width},
        )

    matrix = samples.reshape(-1, channels)
    absolute = np.abs(matrix)
    frame_values = np.asarray(
        [
            float(matrix.mean()),
            float(absolute.mean()),
            float(np.sqrt(np.mean(np.square(matrix)))),
            float(absolute.max()),
            float(np.mean(absolute >= 0.999)),
            float(np.mean(absolute <= 0.01)),
            float(frame_count / sample_rate),
            float(frame_count),
            float(scale),
        ],
        dtype=np.float64,
    )
    if not np.isfinite(frame_values).all():
        raise AudioValidationError(
            "WAV signal features must be finite",
            {"code": "non_finite_features"},
        )
    return tuple(float(value) for value in frame_values), frame_count


@dataclasses.dataclass(frozen=True, slots=True)
class AudioSample:
    """Immutable encoded audio plus the metadata observed at ingestion."""

    source: str
    encoded_bytes: bytes = dataclasses.field(repr=False)
    file_hash: str
    metadata: AudioMetadata
    label: str | int | float | bool | None = None
    byte_size: int = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        """Validate bytes, source identity, and hash integrity."""
        raw = bytes(self.encoded_bytes)
        object.__setattr__(self, "encoded_bytes", raw)
        if not self.source.strip():
            raise ValueError("source must not be empty")
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
    ) -> Self:
        """Create a sample from bytes and an explicit codec probe.

        WAV uses the standard-library probe by default. Every other format
        requires a caller-owned probe, making optional codec behavior visible
        in the run configuration and evidence.
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
        )

    def profile(self) -> AudioProfile:
        """Return bounded metadata suitable for curation and data cards."""
        return AudioProfile(
            source=self.source,
            format=self.metadata.format,
            sample_rate=self.metadata.sample_rate,
            channels=self.metadata.channels,
            frame_count=self.metadata.frame_count,
            duration_seconds=self.metadata.duration_seconds,
            byte_size=self.byte_size,
        )


@dataclasses.dataclass(frozen=True, slots=True)
class AudioDataset:
    """Immutable audio collection with deterministic identity and splitting."""

    samples: tuple[AudioSample, ...]
    content_hash: str = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        """Reject duplicate source identities and derive dataset identity."""
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
        """Return profiles in source order for deterministic curation."""
        return tuple(sample.profile() for sample in self.samples)

    def prepare_features(self, max_frames: int = 100_000) -> AudioFeatures:
        """Prepare bounded PCM and metadata features for an audio baseline.

        Core decoding is intentionally limited to WAV. Samples ingested with
        a caller-owned probe remain valid data contracts, but require a codec
        adapter before signal features can be prepared. The feature rows use
        only an observed prefix and record whether that prefix was truncated.
        """
        if max_frames < 1:
            raise ValueError("max_frames must be at least one")
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
                sample.encoded_bytes, max_frames
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

    def curate(self) -> ModalityCurationReport:
        """Build reviewable metadata and compatibility findings."""
        profiles = self.profiles()
        formats = sorted({profile.format for profile in profiles})
        sample_rates = sorted({profile.sample_rate for profile in profiles})
        channels = sorted({profile.channels for profile in profiles})
        durations = [profile.duration_seconds for profile in profiles]
        labels = label_counts(sample.label for sample in self.samples)
        quality_findings = []
        if len(formats) > 1:
            quality_findings.append(
                build_finding(
                    "quality.audio-format-drift",
                    data_contracts.FindingSeverity.WARNING,
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
                    data_contracts.FindingSeverity.WARNING,
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
                    data_contracts.FindingSeverity.WARNING,
                    {"dataset": self.content_hash},
                    {"label_counts": labels},
                    "Declare whether unlabeled clips are valid for the "
                    "intended task.",
                    False,
                )
            )
        return ModalityCurationReport(
            modality="audio",
            dataset_hash=dataset_hash(
                f"{sample.source}\0{sample.file_hash}"
                for sample in self.samples
            ),
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
    "AudioValidationError",
    "canonical_source",
    "wav_metadata",
    "wav_signal_features",
]
