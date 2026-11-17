"""Profiling and curation of a video collection."""

from collections.abc import Sequence
from typing import Any

from scmle.contracts import data
from scmle.data.modalities.curation import conflicting_label_groups
from scmle.data.modalities.curation import Curation
from scmle.data.modalities.curation import flaw
from scmle.data.modalities.curation import label_counts
from scmle.data.modalities.video.sample import Profile
from scmle.data.modalities.video.sample import Sample


def synchronization(samples: Sequence[Sample]) -> list[data.Finding]:
    """Builds the findings on audio/video synchronization.

    Args:
        samples: The samples of a dataset.

    Returns:
        A blocking finding when a clip is out of tolerance, then a warning
        when a clip's synchronization is unknown.
    """
    quality_findings = []
    out_of_tolerance = [
        sample.source
        for sample in samples
        if sample.metadata.synchrony == "out_of_tolerance"
    ]
    unknown_synchronization = [
        sample.source
        for sample in samples
        if sample.metadata.synchrony == "unknown"
    ]
    if out_of_tolerance:
        quality_findings.append(
            flaw(
                "quality.video-audio-sync-out-of-tolerance",
                data.FindingSeverity.ERROR,
                {"sources": out_of_tolerance},
                {"count": len(out_of_tolerance)},
                "Align audio and video or exclude synchronization-"
                "dependent tasks from the capability declaration.",
                blocking=True,
            )
        )
    if unknown_synchronization:
        quality_findings.append(
            flaw(
                "quality.video-audio-sync-unknown",
                data.FindingSeverity.WARNING,
                {"sources": unknown_synchronization},
                {"count": len(unknown_synchronization)},
                "Supply measured audio duration, offset, and tolerance "
                "before making audio-video timing claims.",
                blocking=False,
            )
        )
    return quality_findings


def drift(
    profiles: Sequence[Profile],
    formats: list[str],
    codecs: list[str],
    dimensions: list[tuple[int, int]],
) -> list[data.Finding]:
    """Builds the findings on encoding and dimension drift.

    Args:
        profiles: The sample profiles.
        formats: Sorted distinct container names.
        codecs: Sorted distinct codec names.
        dimensions: Sorted distinct (width, height) pairs.

    Returns:
        A warning when there is more than one format or codec, then one when
        there is more than one frame size.
    """
    quality_findings = []
    if len(formats) > 1 or len(codecs) > 1:
        quality_findings.append(
            flaw(
                "quality.video-encoding-drift",
                data.FindingSeverity.WARNING,
                {"sources": [profile.source for profile in profiles]},
                {"formats": formats, "codecs": codecs},
                "Choose an explicit decoder and normalization policy "
                "before training.",
                blocking=False,
            )
        )
    if len(dimensions) > 1:
        quality_findings.append(
            flaw(
                "quality.video-dimension-drift",
                data.FindingSeverity.WARNING,
                {"sources": [profile.source for profile in profiles]},
                {"dimensions": [list(value) for value in dimensions]},
                "Record frame resizing or bucketing as a versioned "
                "transformation.",
                blocking=False,
            )
        )
    return quality_findings


def timeline(samples: Sequence[Sample]) -> dict[str, str | int]:
    """Summarises how completely the clips carry observation times.

    Args:
        samples: The samples of a dataset.

    Returns:
        Status ``"absent"`` when no clip is timed, ``"complete"`` with the
        earliest and latest time when every clip is, and ``"mixed"`` with
        the timed and untimed counts otherwise.
    """
    observation_times = tuple(
        sample.observation_time
        for sample in samples
        if sample.observation_time is not None
    )
    if not observation_times:
        return {"status": "absent"}
    if len(observation_times) == len(samples):
        return {
            "status": "complete",
            "observation_time_min": min(observation_times).isoformat(),
            "observation_time_max": max(observation_times).isoformat(),
        }
    return {
        "status": "mixed",
        "timed_video_count": len(observation_times),
        "untimed_video_count": len(samples) - len(observation_times),
    }


def risks(
    content_hash: str,
    labels: dict[str, int],
    conflicts: Sequence[Any],
) -> list[data.Finding]:
    """Builds the label risk findings of a video curation.

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
                "risk.video-mixed-label-state",
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
                "risk.video-conflicting-duplicate-labels",
                data.FindingSeverity.ERROR,
                {"dataset": content_hash},
                {"groups": list(conflicts)},
                "Resolve the label conflict or explicitly exclude the "
                "affected video content.",
                blocking=True,
            )
        )
    return risk_findings


class Survey:
    """Profiling and curation of a video collection.

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
        quality_findings = synchronization(self.samples)
        quality_findings.extend(drift(profiles, formats, codecs, dimensions))
        profile = {
            "formats": formats,
            "codecs": codecs,
            "dimensions": [list(value) for value in dimensions],
            "duration_min_seconds": min(durations),
            "duration_max_seconds": max(durations),
            "duration_mean_seconds": sum(durations) / len(durations),
            "total_bytes": sum(profile.byte_size for profile in profiles),
            "synchronization_statuses": sorted(
                {sample.metadata.synchrony for sample in self.samples}
            ),
            "label_counts": labels,
            "conflicting_label_groups": list(conflicts),
        }
        temporal_evidence = timeline(self.samples)
        profile["temporal"] = temporal_evidence
        if temporal_evidence["status"] == "mixed":
            quality_findings.append(
                flaw(
                    "quality.video-mixed-temporal-state",
                    data.FindingSeverity.WARNING,
                    {"dataset": self.content_hash},
                    temporal_evidence,
                    "Supply observation times for every clip before using "
                    "temporal evaluation or declare the untimed subset.",
                    blocking=False,
                )
            )
        return Curation(
            modality="video",
            dataset_hash=self.content_hash,
            sample_count=len(self.samples),
            profile=profile,
            quality_findings=tuple(quality_findings),
            risk_findings=tuple(risks(self.content_hash, labels, conflicts)),
        )
