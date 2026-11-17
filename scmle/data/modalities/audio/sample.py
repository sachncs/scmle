"""Audio samples and their source identity."""

from collections.abc import Callable
import dataclasses
import hashlib
import pathlib
from typing import Self

from scmle.contracts import errors
from scmle.data.modalities.audio.metadata import DecoderError
from scmle.data.modalities.audio.metadata import Metadata
from scmle.data.modalities.audio.metadata import Profile
from scmle.data.modalities.audio.metadata import SHA256_HEX_LENGTH
from scmle.data.modalities.audio.metadata import Transcript
from scmle.data.modalities.audio.metadata import ValidationError
from scmle.data.modalities.audio.wav import parse

AudioProbe = Callable[[bytes], Metadata]


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


@dataclasses.dataclass(frozen=True, slots=True)
class Sample:
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
    metadata: Metadata
    label: str | int | float | bool | None = None
    group_id: str | None = None
    transcript: Transcript | None = None
    byte_size: int = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        """Validates bytes, source identity, and hash integrity.

        Raises:
            ValueError: If source or group_id is blank, or file_hash is not
                a lowercase SHA-256 hex digest of the payload.
            ValidationError: If a transcript segment ends after the
                audio does, or the payload is empty.
        """
        raw = bytes(self.encoded_bytes)
        object.__setattr__(self, "encoded_bytes", raw)
        if not self.source.strip():
            raise errors.InputError("source must not be empty")
        if self.group_id is not None and not self.group_id.strip():
            raise errors.InputError("group_id must not be empty when provided")
        if self.transcript is not None and any(
            segment.end_seconds > self.metadata.duration_seconds + 1e-9
            for segment in self.transcript.segments
        ):
            raise ValidationError(
                "transcript segment exceeds audio duration",
                {"code": "transcript_out_of_range", "source": self.source},
            )
        if not raw:
            raise ValidationError(
                "audio payload is empty", {"code": "empty_payload"}
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
        source: str = "memory://",
        probe: AudioProbe | None = None,
        max_bytes: int | None = 500 * 1024 * 1024,
        label: str | float | bool | None = None,
        group_id: str | None = None,
        transcript: Transcript | None = None,
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
            ValidationError: If the payload is empty, exceeds
                ``max_bytes`` (or ``max_bytes`` is below one), or the probe
                raises it.
            DecoderError: If no probe is given for a payload that is
                not WAV, or the probe fails with any other exception.
        """
        raw = bytes(encoded_bytes)
        if not raw:
            raise ValidationError(
                "audio payload is empty", {"code": "empty_payload"}
            )
        if max_bytes is not None and (max_bytes < 1 or len(raw) > max_bytes):
            raise ValidationError(
                "audio payload exceeds the configured byte limit",
                {
                    "code": "encoded_size_limit",
                    "byte_size": len(raw),
                    "max_bytes": max_bytes,
                },
            )
        if probe is None and not (raw[:4] == b"RIFF" and raw[8:12] == b"WAVE"):
            raise DecoderError(
                "this audio container requires an explicit codec probe"
            )
        try:
            metadata = (probe or parse)(raw)
        except ValidationError:
            raise
        except Exception as error:
            raise DecoderError(
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

    def profile(self) -> Profile:
        """Returns bounded metadata suitable for curation and data cards.

        Returns:
            The profile of this sample.
        """
        return Profile(
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
