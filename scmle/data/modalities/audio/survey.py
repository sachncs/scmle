"""Profiling and curation of an audio collection."""

from collections.abc import Sequence
from typing import Any

from scmle.contracts import data
from scmle.data.modalities.audio.metadata import Profile
from scmle.data.modalities.audio.metadata import Transcript
from scmle.data.modalities.audio.sample import Sample
from scmle.data.modalities.curation import conflicting_label_groups
from scmle.data.modalities.curation import Curation
from scmle.data.modalities.curation import flaw
from scmle.data.modalities.curation import label_counts


def confidence(
    samples: Sequence[Sample], transcripts: Sequence[Transcript]
) -> list[data.Finding]:
    """Builds the finding on transcripts that carry no confidence.

    Args:
        samples: The samples of a dataset.
        transcripts: Their transcripts, where present.

    Returns:
        One warning when a transcript has no confidence on itself or on any
        segment; otherwise nothing.
    """
    missing_transcript_confidence = sum(
        transcript.confidence is None
        and all(segment.confidence is None for segment in transcript.segments)
        for transcript in transcripts
    )
    if not missing_transcript_confidence:
        return []
    return [
        flaw(
            "quality.audio-transcript-confidence-missing",
            data.FindingSeverity.WARNING,
            {
                "sources": [
                    sample.source
                    for sample in samples
                    if sample.transcript is not None
                ]
            },
            {"sample_count": missing_transcript_confidence},
            "Obtain confidence from the transcript provider or "
            "record why confidence is unavailable before using "
            "transcript-dependent claims.",
            blocking=False,
        )
    ]


def drift(
    profiles: Sequence[Profile],
    formats: list[str],
    sample_rates: list[int],
    channels: list[int],
) -> list[data.Finding]:
    """Builds the findings on format, sample-rate and channel drift.

    Args:
        profiles: The sample profiles.
        formats: Sorted distinct formats.
        sample_rates: Sorted distinct sample rates.
        channels: Sorted distinct channel counts.

    Returns:
        A warning when there is more than one format, then one when there is
        more than one sample rate or channel count.
    """
    quality_findings = []
    if len(formats) > 1:
        quality_findings.append(
            flaw(
                "quality.audio-format-drift",
                data.FindingSeverity.WARNING,
                {"sources": [profile.source for profile in profiles]},
                {"formats": formats},
                "Select an explicit codec normalization policy before "
                "training.",
                blocking=False,
            )
        )
    if len(sample_rates) > 1 or len(channels) > 1:
        quality_findings.append(
            flaw(
                "quality.audio-shape-drift",
                data.FindingSeverity.WARNING,
                {"sources": [profile.source for profile in profiles]},
                {
                    "sample_rates": sample_rates,
                    "channels": channels,
                },
                "Resample or channel-normalize with the transformation "
                "recorded.",
                blocking=False,
            )
        )
    return quality_findings


def risks(
    content_hash: str,
    labels: dict[str, int],
    conflicts: Sequence[Any],
) -> list[data.Finding]:
    """Builds the label risk findings of an audio curation.

    Args:
        content_hash: Content hash of the dataset.
        labels: Label counts.
        conflicts: Groups of identical content with conflicting labels.

    Returns:
        A warning for a mix of labelled and unlabelled clips, then a
        blocking finding for conflicting labels.
    """
    risk_findings = []
    if "unlabeled" in labels and len(labels) > 1:
        risk_findings.append(
            flaw(
                "risk.audio-mixed-label-state",
                data.FindingSeverity.WARNING,
                {"dataset": content_hash},
                {"label_counts": labels},
                "Declare whether unlabeled clips are valid for the "
                "intended task.",
                blocking=False,
            )
        )
    if conflicts:
        risk_findings.append(
            flaw(
                "risk.audio-conflicting-duplicate-labels",
                data.FindingSeverity.ERROR,
                {"dataset": content_hash},
                {"groups": list(conflicts)},
                "Resolve the label conflict or explicitly exclude the "
                "affected audio content.",
                blocking=True,
            )
        )
    return risk_findings


def transcription(
    transcripts: Sequence[Transcript], sample_count: int
) -> dict[str, float | list[str] | None]:
    """Summarises the transcripts of a dataset for its curation profile.

    Args:
        transcripts: The transcripts present.
        sample_count: Number of samples in the dataset.

    Returns:
        Coverage, sorted sources, providers and languages, the confidence
        range over transcripts and segments (None when no confidence is
        known) and the segment count.
    """
    transcript_confidences = [
        value
        for transcript in transcripts
        for value in (
            [transcript.confidence]
            + [segment.confidence for segment in transcript.segments]
        )
        if value is not None
    ]
    return {
        "transcript_coverage": len(transcripts) / sample_count,
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
            min(transcript_confidences) if transcript_confidences else None
        ),
        "transcript_confidence_max": (
            max(transcript_confidences) if transcript_confidences else None
        ),
        "transcript_segment_count": sum(
            len(transcript.segments) for transcript in transcripts
        ),
    }


class Survey:
    """Profiling and curation of an audio collection.

    Attributes:
        samples: The samples, in caller order.
        content_hash: Content hash of the dataset.
    """

    __slots__ = ()

    samples: tuple[Sample, ...]
    content_hash: str

    def profiles(self) -> tuple[Profile, ...]:
        """Returns profiles in sample order for deterministic curation.

        Returns:
            One profile per sample.
        """
        return tuple(sample.profile() for sample in self.samples)

    def curate(self) -> Curation:
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
        transcripts = [
            sample.transcript
            for sample in self.samples
            if sample.transcript is not None
        ]
        quality_findings = confidence(self.samples, transcripts)
        quality_findings.extend(
            drift(profiles, formats, sample_rates, channels)
        )
        return Curation(
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
                **transcription(transcripts, len(self.samples)),
            },
            quality_findings=tuple(quality_findings),
            risk_findings=tuple(risks(self.content_hash, labels, conflicts)),
        )
