"""Feature matrices and splits produced from an audio collection."""

import dataclasses

import numpy

from scmle.data.modalities.audio.sample import Sample


@dataclasses.dataclass(frozen=True, slots=True)
class Features:
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
class Split:
    """Group-aware audio split that keeps related clips together.

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
