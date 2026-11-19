"""The immutable image collection."""

import dataclasses
from typing import Self

import numpy

from scmle.contracts import errors
from scmle.data.modalities.curation import allocate
from scmle.data.modalities.curation import digest
from scmle.data.modalities.curation import proportions
from scmle.data.modalities.image.evidence import Split
from scmle.data.modalities.image.pixels import configure
from scmle.data.modalities.image.sample import Sample
from scmle.data.modalities.image.survey import Survey


@dataclasses.dataclass(frozen=True, slots=True)
class Dataset(Survey):
    """Immutable collection of validated image samples.

    Attributes:
        samples: The samples, in caller order.
        digest: Order-independent digest of the sources, content hashes
            and augmentation identities; set on construction.
    """

    samples: tuple[Sample, ...] = ()
    digest: str = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        """Normalizes samples to an immutable tuple and validates types.

        Raises:
            TypeError: If an element is not an Sample.
        """
        normalized = tuple(self.samples)
        if any(not isinstance(sample, Sample) for sample in normalized):
            raise errors.KindError("samples must contain only Sample objects")
        object.__setattr__(self, "samples", normalized)
        identities = (
            f"{sample.source}\0{sample.content_hash}"
            + (
                f"\0augmentation={sample.augmentation.identity}"
                if sample.augmentation is not None
                else ""
            )
            for sample in normalized
        )
        object.__setattr__(self, "digest", digest(identities))

    @property
    def size(self) -> int:
        """The number of samples in the dataset."""
        return len(self.samples)

    def deduplicate(self) -> Self:
        """Returns a content-deduplicated dataset with stable ordering.

        For each content identity, the lexicographically smallest source is
        retained. Sorting by content and source makes the result independent
        of input iteration order.

        Returns:
            A new dataset with one sample per distinct content hash, ordered
            by content hash and then source.
        """
        selected: dict[str, Sample] = {}
        for sample in self.samples:
            previous = selected.get(sample.content_hash)
            if previous is None or (
                sample.source,
                sample.file_hash,
            ) < (previous.source, previous.file_hash):
                selected[sample.content_hash] = sample
        ordered = tuple(
            selected[key]
            for key in sorted(
                selected,
                key=lambda content: (
                    content,
                    selected[content].source,
                ),
            )
        )
        return type(self)(ordered)

    def flatten(
        self,
        target_size: tuple[int, int] = (16, 16),
        mode: str = "RGB",
        normalize: bool = True,
    ) -> numpy.ndarray:
        """Flattens resized pixels for a numeric baseline adapter.

        The result has one row per sample. Columns keep Pillow/NumPy
        row-major order, with channels in the selected mode's order.

        Args:
            target_size: ``(width, height)`` each image is resized to.
            mode: Pillow mode; one of ``L``, ``RGB`` or ``RGBA``.
            normalize: Whether to scale values to ``[0, 1]`` as float32;
                otherwise they are ``uint8`` in ``[0, 255]``.

        Returns:
            Array of shape ``(sample_count, width * height * channels)``;
            with no samples, shape ``(0, width * height * channels)``. An
            invalid ``target_size`` or ``mode`` raises ValueError from
            `configure`.
        """
        configure(target_size, mode)
        channels = 1 if mode == "L" else len(mode)
        feature_count = target_size[0] * target_size[1] * channels
        if not self.samples:
            dtype = numpy.float32 if normalize else numpy.uint8
            return numpy.empty((0, feature_count), dtype=dtype)
        rows = [
            sample.decode(mode, target_size).reshape(-1)
            for sample in self.samples
        ]
        features = numpy.stack(rows)
        if normalize:
            return features.astype(numpy.float32) / 255.0
        return features

    def split(
        self,
        validation_fraction: float = 0.2,
        test_fraction: float = 0.2,
        seed: int = 0,
    ) -> Split:
        """Splits samples by explicit group identity to prevent leakage.

        A caller should use a subject, capture session, or other identity as
        ``group_id``. Without one, source identity is the conservative group.

        Args:
            validation_fraction: Share of groups for validation, in [0, 1).
            test_fraction: Share of groups for testing, in [0, 1).
            seed: Seed of the shuffle of the sorted group ids.

        Returns:
            The split. Whole groups go to one partition; with at least three
            groups a non-zero fraction gets at least one group, and the
            remaining groups form the training set.

        Raises:
            ValueError: If the dataset is empty, a fraction is outside
                [0, 1), or the fractions sum to 1 or more.
        """
        if not self.samples:
            raise errors.InputError("cannot split an empty image dataset")
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
