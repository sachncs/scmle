"""The immutable audio collection."""

import dataclasses
import hashlib

from scmle.contracts import errors
from scmle.data.modalities.audio.evidence import Features
from scmle.data.modalities.audio.evidence import Split
from scmle.data.modalities.audio.metadata import DecoderError
from scmle.data.modalities.audio.metadata import ValidationError
from scmle.data.modalities.audio.sample import Sample
from scmle.data.modalities.audio.survey import Survey
from scmle.data.modalities.audio.wav import extract
from scmle.data.modalities.curation import allocate
from scmle.data.modalities.curation import proportions


@dataclasses.dataclass(frozen=True, slots=True)
class Dataset(Survey):
    """Immutable audio collection with deterministic identity and splitting.

    Attributes:
        samples: The samples, in caller order.
        digest: SHA-256 hex digest of the sources and file hashes, in
            sample order; set on construction.
    """

    samples: tuple[Sample, ...]
    digest: str = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        """Rejects duplicate source identities and derive dataset identity.

        Raises:
            ValidationError: If there are no samples or two samples
                share a source.
        """
        normalized = tuple(self.samples)
        if not normalized:
            raise ValidationError(
                "audio dataset must contain at least one sample",
                {"code": "empty_dataset"},
            )
        sources = [sample.source for sample in normalized]
        if len(sources) != len(set(sources)):
            raise ValidationError(
                "audio dataset contains duplicate sources",
                {"code": "duplicate_source"},
            )
        digest = hashlib.sha256(
            "\n".join(
                f"{sample.source}\0{sample.file_hash}" for sample in normalized
            ).encode("utf-8")
        ).hexdigest()
        object.__setattr__(self, "samples", normalized)
        object.__setattr__(self, "digest", digest)

    def prepare(
        self,
        max_frames: int = 100_000,
        start_frame: int = 0,
    ) -> Features:
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
            `extract`), then duration in seconds, sample rate,
            channel count and sample width (0 when unknown).

        Raises:
            ValueError: If ``max_frames`` is below one or ``start_frame`` is
                negative.
            DecoderError: If a sample is not in WAV format.
        """
        if max_frames < 1:
            raise errors.InputError("max_frames must be at least one")
        if start_frame < 0:
            raise errors.InputError("start_frame must not be negative")
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
                raise DecoderError(
                    "audio signal features require a WAV decoder for "
                    f"'{sample.source}'"
                )
            signal_values, observed = extract(
                sample.encoded, max_frames, start_frame
            )
            rows.append(
                (
                    *signal_values,
                    sample.metadata.seconds,
                    float(sample.metadata.sample_rate),
                    float(sample.metadata.channels),
                    float(sample.metadata.sample_width or 0),
                )
            )
            frames_observed.append(observed)
            frames_truncated.append(observed < sample.metadata.frame_count)
        return Features(
            sources=tuple(sample.source for sample in self.samples),
            names=feature_names,
            values=tuple(rows),
            observed=tuple(frames_observed),
            truncated=tuple(frames_truncated),
        )

    def split(
        self,
        validation_fraction: float = 0.2,
        test_fraction: float = 0.2,
        seed: int = 0,
    ) -> Split:
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
        proportions(validation_fraction, test_fraction)
        grouped: dict[str, list[Sample]] = {}
        for sample in self.samples:
            grouped.setdefault(sample.group or sample.source, []).append(sample)
        validation_groups, test_groups = allocate(
            grouped, validation_fraction, test_fraction, seed
        )

        def sieve(excluded: set[str]) -> tuple[Sample, ...]:
            return tuple(
                sample
                for sample in self.samples
                if (sample.group or sample.source) not in excluded
            )

        validation = tuple(
            sample
            for sample in self.samples
            if (sample.group or sample.source) in validation_groups
        )
        test = tuple(
            sample
            for sample in self.samples
            if (sample.group or sample.source) in test_groups
        )
        return Split(
            train=sieve(test_groups | validation_groups),
            validation=validation,
            test=test,
        )
