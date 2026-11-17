"""Profiling and curation of an image collection."""

from collections.abc import Sequence
from typing import Any

import numpy

from scmle.contracts import data
from scmle.contracts import errors
from scmle.data.modalities.curation import conflicting_label_groups
from scmle.data.modalities.curation import Curation
from scmle.data.modalities.curation import flaw
from scmle.data.modalities.curation import label_counts
from scmle.data.modalities.image.duplicates import Duplicates
from scmle.data.modalities.image.evidence import Profile
from scmle.data.modalities.image.pixels import configure
from scmle.data.modalities.image.sample import DEFAULT_PROFILE_SIZE
from scmle.data.modalities.image.sample import Sample

EMPTY_PROFILE = Profile(
    dataset_count=0,
    profiled_count=0,
    truncated=False,
    profiled_bytes=0,
    unique_content_count=0,
    duplicate_count=0,
    modes=(),
    channels=(),
    width_min=0,
    width_max=0,
    width_mean=0.0,
    height_min=0,
    height_max=0,
    height_mean=0.0,
    mean_pixel=(0.0, 0.0, 0.0),
    pixel_std=(0.0, 0.0, 0.0),
)


def palette(
    profiled: Sequence[Sample], profile_size: tuple[int, int]
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Averages per-channel RGB statistics over resized copies.

    Args:
        profiled: The samples profiled; at least one.
        profile_size: ``(width, height)`` each image is resized to.

    Returns:
        The mean over samples of each sample's per-channel mean, and the
        mean over samples of each sample's per-channel variance.
    """
    means: list[numpy.ndarray] = []
    variances: list[numpy.ndarray] = []
    for sample in profiled:
        pixels = sample.decode("RGB", profile_size).astype(numpy.float64)
        means.append(pixels.mean(axis=(0, 1)))
        variances.append(pixels.var(axis=(0, 1)))
    mean = numpy.mean(numpy.stack(means), axis=0)
    variance = numpy.mean(numpy.stack(variances), axis=0)
    return mean, variance


def augmentation(
    samples: Sequence[Sample],
) -> dict[str, str | int | list[str]]:
    """Summarises the declared augmentation provenance of a dataset.

    Args:
        samples: The samples of a dataset.

    Returns:
        The status (``absent`` or ``declared``), the augmented sample count,
        the sorted operations and the sorted parent sources that are not in
        the dataset.
    """
    augmented_samples = [
        sample for sample in samples if sample.augmentation is not None
    ]
    source_names = {sample.source for sample in samples}
    return {
        "status": "absent" if not augmented_samples else "declared",
        "augmented_sample_count": len(augmented_samples),
        "operations": sorted(
            {
                sample.augmentation.operation
                for sample in augmented_samples
                if sample.augmentation is not None
            }
        ),
        "external_parent_sources": sorted(
            {
                sample.augmentation.parent_source
                for sample in augmented_samples
                if sample.augmentation is not None
                and sample.augmentation.parent_source not in source_names
            }
        ),
    }


def defects(
    sample_count: int,
    profile: Profile,
    near_duplicate_groups: Sequence[tuple[str, ...]],
    max_samples: int,
) -> list[data.Finding]:
    """Builds the quality findings of an image curation.

    Args:
        sample_count: Number of samples in the dataset.
        profile: The bounded profile.
        near_duplicate_groups: Groups of visually similar images.
        max_samples: Number of leading samples profiled.

    Returns:
        Findings for an empty dataset, duplicate content, near-duplicate
        content and a truncated profile, in that order.
    """
    quality_findings = []
    if not sample_count:
        quality_findings.append(
            flaw(
                "quality.image-empty-dataset",
                data.FindingSeverity.CRITICAL,
                {"dataset": "empty"},
                {"sample_count": 0},
                "Provide at least one validated image before training.",
                blocking=True,
            )
        )
    if profile.duplicate_count:
        quality_findings.append(
            flaw(
                "quality.image-duplicate-content",
                data.FindingSeverity.WARNING,
                {"dataset": "profiled_samples"},
                {"duplicate_count": profile.duplicate_count},
                "Review duplicate content and prevent split contamination.",
                blocking=False,
            )
        )
    if near_duplicate_groups:
        quality_findings.append(
            flaw(
                "quality.image-near-duplicate-content",
                data.FindingSeverity.WARNING,
                {"dataset": "profiled_samples"},
                {
                    "group_count": len(near_duplicate_groups),
                    "groups": [list(group) for group in near_duplicate_groups],
                },
                "Review visually similar images and keep related content "
                "within one split before training.",
                blocking=False,
            )
        )
    if profile.truncated:
        quality_findings.append(
            flaw(
                "quality.image-profile-truncated",
                data.FindingSeverity.INFO,
                {"dataset": "profiled_samples"},
                {
                    "max_samples": max_samples,
                    "profiled_count": profile.profiled_count,
                },
                "Run a full-content audit before release or record the "
                "sampling policy.",
                blocking=False,
            )
        )
    return quality_findings


def hazards(
    conflicts: Sequence[Any],
    sensitive_metadata: list[str],
    labels: dict[str, int],
) -> list[data.Finding]:
    """Builds the risk findings of an image curation.

    Args:
        conflicts: Groups of identical content with conflicting labels.
        sensitive_metadata: Sorted names of sensitive EXIF fields present.
        labels: Label counts.

    Returns:
        Findings for conflicting labels, sensitive metadata and a mix of
        labelled and unlabelled images, in that order.
    """
    risk_findings = []
    if conflicts:
        risk_findings.append(
            flaw(
                "risk.image-conflicting-duplicate-labels",
                data.FindingSeverity.ERROR,
                {"dataset": "all_samples"},
                {"groups": list(conflicts)},
                "Resolve the label conflict or explicitly exclude the "
                "affected content.",
                blocking=True,
            )
        )
    if sensitive_metadata:
        risk_findings.append(
            flaw(
                "risk.image-sensitive-metadata",
                data.FindingSeverity.ERROR,
                {"dataset": "all_samples"},
                {"tags": sensitive_metadata},
                "Review, redact, or explicitly approve image metadata "
                "before training or distribution.",
                blocking=True,
            )
        )
    if "unlabeled" in labels and len(labels) > 1:
        risk_findings.append(
            flaw(
                "risk.image-mixed-label-state",
                data.FindingSeverity.WARNING,
                {"dataset": "all_samples"},
                {"label_counts": labels},
                "Declare whether unlabeled images are valid for the "
                "intended task before training.",
                blocking=False,
            )
        )
    return risk_findings


class Survey(Duplicates):
    """Profiling and curation of an image collection.

    Attributes:
        content_hash: Content hash of the dataset.
    """

    __slots__ = ()

    content_hash: str

    def profile(
        self,
        max_samples: int = 1024,
        profile_size: tuple[int, int] = DEFAULT_PROFILE_SIZE,
    ) -> Profile:
        """Profiles at most ``max_samples`` images and bounded pixels.

        Args:
            max_samples: Number of leading samples profiled.
            profile_size: ``(width, height)`` each image is resized to
                before pixel statistics are computed.

        Returns:
            The profile; an empty dataset gives all-zero counts.

        Raises:
            ValueError: If ``max_samples`` is not a positive integer.
                ``profile_size`` that is not two positive integers also
                raises ValueError, from `configure`.
        """
        samples_limit: object = max_samples
        if not isinstance(samples_limit, int) or isinstance(
            samples_limit, bool
        ):
            raise errors.InputError("max_samples must be a positive integer")
        if max_samples <= 0:
            raise errors.InputError("max_samples must be a positive integer")
        configure(profile_size, "RGB")
        profiled = self.samples[:max_samples]
        if not profiled:
            return EMPTY_PROFILE
        mean, variance = palette(profiled, profile_size)
        modes = tuple(
            sorted(
                (mode, sum(sample.mode == mode for sample in profiled))
                for mode in {sample.mode for sample in profiled}
            )
        )
        channels = tuple(
            sorted(
                (
                    channel,
                    sum(sample.channels == channel for sample in profiled),
                )
                for channel in {sample.channels for sample in profiled}
            )
        )
        unique_count = len({sample.content_hash for sample in profiled})
        return Profile(
            dataset_count=len(self.samples),
            profiled_count=len(profiled),
            truncated=len(profiled) < len(self.samples),
            profiled_bytes=sum(sample.byte_size for sample in profiled),
            unique_content_count=unique_count,
            duplicate_count=len(profiled) - unique_count,
            modes=modes,
            channels=channels,
            width_min=min(sample.width for sample in profiled),
            width_max=max(sample.width for sample in profiled),
            width_mean=float(numpy.mean([sample.width for sample in profiled])),
            height_min=min(sample.height for sample in profiled),
            height_max=max(sample.height for sample in profiled),
            height_mean=float(
                numpy.mean([sample.height for sample in profiled])
            ),
            mean_pixel=tuple(float(value) for value in mean),
            pixel_std=tuple(float(value) for value in numpy.sqrt(variance)),
        )

    def curate(
        self,
        max_samples: int = 1024,
        profile_size: tuple[int, int] = DEFAULT_PROFILE_SIZE,
    ) -> Curation:
        """Builds bounded image statistics and reviewable findings.

        Args:
            max_samples: Number of leading samples profiled and checked for
                near duplicates.
            profile_size: Resize target used for pixel statistics.

        Returns:
            A curation report for modality ``image`` carrying the profile,
            label counts, conflicting labels, sensitive metadata,
            near-duplicate groups and augmentation summary, with quality and
            risk findings. Invalid ``max_samples`` or ``profile_size``
            (ValueError) and an undecodable image (ImageValidationError)
            propagate from `profile`.
        """
        profile = self.profile(max_samples, profile_size)
        labels = label_counts(sample.label for sample in self.samples)
        conflicts = conflicting_label_groups(
            (sample.content_hash, sample.source, sample.label)
            for sample in self.samples
        )
        profile_data = profile.to_dict()
        profile_data["label_counts"] = labels
        profile_data["conflicting_label_groups"] = list(conflicts)
        sensitive_metadata = sorted(
            {
                tag
                for sample in self.samples
                for tag in sample.sensitive_metadata
            }
        )
        profile_data["sensitive_metadata"] = sensitive_metadata
        near_duplicate_groups = self.near_duplicate_groups(max_samples)
        profile_data["near_duplicate_groups"] = [
            list(group) for group in near_duplicate_groups
        ]
        profile_data["augmentation"] = augmentation(self.samples)
        return Curation(
            modality="image",
            dataset_hash=self.content_hash,
            sample_count=len(self.samples),
            profile=profile_data,
            quality_findings=tuple(
                defects(
                    len(self.samples),
                    profile,
                    near_duplicate_groups,
                    max_samples,
                )
            ),
            risk_findings=tuple(hazards(conflicts, sensitive_metadata, labels)),
        )
