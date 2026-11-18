"""The profile and split produced from an image collection."""

import dataclasses
from typing import Any
from typing import ClassVar

from scmle.contracts import records
from scmle.data.modalities.image.sample import Sample


@dataclasses.dataclass(frozen=True, slots=True)
class Profile:
    """Bounded, deterministic summary of an image dataset.

    Attributes:
        dataset_count: Number of samples in the dataset.
        profiled_count: Number of samples profiled.
        truncated: Whether samples were left out of the profile.
        profiled_bytes: Total encoded size of the profiled samples.
        unique_content_count: Distinct content hashes among them.
        duplicate_count: Profiled samples minus unique content.
        modes: Sorted ``(mode, count)`` pairs.
        channels: Sorted ``(channel count, count)`` pairs.
        width_min: Smallest width in pixels.
        width_max: Largest width in pixels.
        width_mean: Mean width in pixels.
        height_min: Smallest height in pixels.
        height_max: Largest height in pixels.
        height_mean: Mean height in pixels.
        mean_pixel: Mean RGB value per channel on 0-255, from resized
            copies.
        pixel_std: Per-channel standard deviation on 0-255, from resized
            copies.
    """

    dataset_count: int
    profiled_count: int
    truncated: bool
    profiled_bytes: int
    unique_content_count: int
    duplicate_count: int
    modes: tuple[tuple[str, int], ...]
    channels: tuple[tuple[int, int], ...]
    width_min: int
    width_max: int
    width_mean: float
    height_min: int
    height_max: int
    height_mean: float
    mean_pixel: tuple[float, ...]
    pixel_std: tuple[float, ...]

    KEYS: ClassVar[records.Migration] = records.nest(
        {
            "dataset_count": "samples",
            "profiled_count": "profiled",
            "profiled_bytes": "size",
            "unique_content_count": "unique",
            "duplicate_count": "duplicates",
        },
        groups={
            "width": {
                "width_min": "low",
                "width_max": "high",
                "width_mean": "mean",
            },
            "height": {
                "height_min": "low",
                "height_max": "high",
                "height_mean": "mean",
            },
            "pixel": {"mean_pixel": "mean", "pixel_std": "std"},
        },
    )
    """Version-1 keys to the written ones; `downgrade` restores version 1."""

    def to_dict(self) -> dict[str, Any]:
        """Returns JSON-compatible profile data.

        Returns:
            A dict of every field of the profile in single-word keys, with
            extremes and means under `width`, `height` and `pixel`.
        """
        return self.KEYS.upgrade(dataclasses.asdict(self))


@dataclasses.dataclass(frozen=True, slots=True)
class Split:
    """Group-aware image split that keeps related samples together.

    Attributes:
        train: Samples assigned to training.
        validation: Samples assigned to validation.
        test: Samples assigned to testing.
    """

    train: tuple[Sample, ...]
    validation: tuple[Sample, ...]
    test: tuple[Sample, ...]

    def cohorts(self) -> dict[str, tuple[str, ...]]:
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
