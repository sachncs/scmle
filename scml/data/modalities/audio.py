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
    "AudioMetadata",
    "AudioProbe",
    "AudioProfile",
    "AudioSample",
    "AudioValidationError",
    "canonical_source",
    "wav_metadata",
]
