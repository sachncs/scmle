"""Feature matrices and splits produced from a video collection."""

import dataclasses
from datetime import datetime
from datetime import UTC

import numpy

from scmle.contracts import errors
from scmle.data.modalities.video.sample import Sample


@dataclasses.dataclass(frozen=True, slots=True)
class Features:
    """Numeric video features with explicit sampled-frame provenance.

    Attributes:
        sources: Source identifier of each row.
        names: Name of each column.
        values: One tuple of finite feature values per source.
        frames: Per source, the indices of the frames the
            features came from; empty when not frame-based.
    """

    sources: tuple[str, ...]
    names: tuple[str, ...]
    values: tuple[tuple[float, ...], ...]
    frames: tuple[tuple[int, ...], ...] = ()

    def __post_init__(self) -> None:
        """Validates feature dimensions and sampled-frame metadata.

        Raises:
            ValueError: If rows do not match sources, columns do not match
                feature names, a value is not finite, sampled frame
                indices are given for a different number of sources, or an
                index is negative.
        """
        if len(self.values) != len(self.sources):
            raise errors.InputError("feature rows must match video sources")
        if any(len(row) != len(self.names) for row in self.values):
            raise errors.InputError("feature columns must match feature names")
        if not numpy.isfinite(
            numpy.asarray(self.values, dtype=numpy.float64)
        ).all():
            raise errors.InputError("video features must be finite")
        sampled = tuple(
            tuple(int(index) for index in indices) for indices in self.frames
        )
        if sampled and len(sampled) != len(self.sources):
            raise errors.InputError(
                "sampled frame metadata must match video sources"
            )
        if any(index < 0 for indices in sampled for index in indices):
            raise errors.InputError(
                "sampled frame indices must not be negative"
            )
        object.__setattr__(self, "sources", tuple(self.sources))
        object.__setattr__(self, "names", tuple(self.names))
        object.__setattr__(
            self,
            "values",
            tuple(tuple(float(value) for value in row) for row in self.values),
        )
        object.__setattr__(self, "frames", sampled)

    @property
    def shape(self) -> tuple[int, int]:
        """The number of videos and the number of features, as a pair."""
        return (len(self.sources), len(self.names))

    def materialize(self) -> numpy.ndarray:
        """Returns a writable copy of the feature matrix.

        Returns:
            A float64 array of shape ``(videos, features)``.
        """
        return numpy.asarray(self.values, dtype=numpy.float64).copy()


@dataclasses.dataclass(frozen=True, slots=True)
class Split:
    """Group-aware video split that keeps related clips together.

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
                sorted({sample.group or sample.source for sample in samples})
            )
            for name, samples in (
                ("train", self.train),
                ("validation", self.validation),
                ("test", self.test),
            )
        }


@dataclasses.dataclass(frozen=True, slots=True)
class Chronology:
    """Time-ordered video clips separated at a UTC cutoff.

    Attributes:
        cutoff: Timezone-aware cutoff, converted to UTC.
        train: Clips observed at or before the cutoff.
        future: Clips observed after the cutoff.
    """

    cutoff: datetime
    train: tuple[Sample, ...]
    future: tuple[Sample, ...]

    def __post_init__(self) -> None:
        """Validates the cutoff and timestamp completeness of each partition.

        Raises:
            ValueError: If the cutoff is not timezone-aware or a clip in
                either partition has no observation time.
        """
        if self.cutoff.tzinfo is None or self.cutoff.utcoffset() is None:
            raise errors.InputError("cutoff must be timezone-aware")
        object.__setattr__(self, "cutoff", self.cutoff.astimezone(UTC))
        if any(sample.time is None for sample in self.train):
            raise errors.InputError(
                "train videos must contain observation times"
            )
        if any(sample.time is None for sample in self.future):
            raise errors.InputError(
                "future videos must contain observation times"
            )

    @property
    def lookahead(self) -> bool:
        """Whether a training clip was observed after the cutoff."""
        return any(
            isinstance(sample.time, datetime) and sample.time > self.cutoff
            for sample in self.train
        )
