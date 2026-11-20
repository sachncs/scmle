"""Image-profile monitoring against a reference snapshot."""

import numpy

from scmle.contracts import errors
from scmle.data.modalities import curation
from scmle.data.modalities import image
from scmle.models.images.records import Drift


class Monitor:
    """Monitors bounded image-profile signals against a reference snapshot.

    Attributes:
        reference: Curation report of the reference images.
        threshold: Level above which a signal counts as shifted.
    """

    def __init__(
        self,
        reference: curation.Curation,
        threshold: float = 0.25,
    ) -> None:
        """Initializes a monitor with an explicit normalized threshold.

        Args:
            reference: Curation report of the reference images.
            threshold: Level above which a signal counts as shifted; in
                (0, 1].

        Raises:
            ValueError: If `threshold` is outside (0, 1].
        """
        if not 0 < threshold <= 1:
            raise errors.InputError("threshold must be in (0, 1]")
        self.reference = reference
        self.threshold = threshold

    def observe(self, observed: image.Dataset) -> Drift:
        """Compares observed mean pixels and duplicate rate with reference.

        Args:
            observed: Images to compare against the reference.

        Returns:
            Drift report with the mean pixel distance and duplicate rate.
        """
        current = observed.curate()
        reference_mean = numpy.asarray(
            self.reference.profile["pixel"]["mean"],
            dtype=float,
        )
        current_mean = numpy.asarray(
            current.profile["pixel"]["mean"],
            dtype=float,
        )
        distance = float(
            numpy.mean(numpy.abs(current_mean - reference_mean)) / 255.0
        )
        duplicate_rate = current.profile["duplicates"] / max(current.samples, 1)
        return Drift(
            reference_hash=self.reference.dataset_hash,
            observed_hash=current.dataset_hash,
            pixels=distance,
            duplicates=float(duplicate_rate),
            threshold=self.threshold,
        )
