"""Group-aware and temporal splits of a video collection."""

from datetime import datetime

from scmle.data.modalities.curation import allocate
from scmle.data.modalities.curation import proportions
from scmle.data.modalities.video.evidence import Chronology
from scmle.data.modalities.video.evidence import Split
from scmle.data.modalities.video.sample import canonical_observation_time
from scmle.data.modalities.video.sample import Sample
from scmle.data.modalities.video.sample import VideoTemporalError


class Splits:
    """Group-aware and temporal splits of a video collection.

    Attributes:
        samples: The samples, in caller order.
    """

    __slots__ = ()

    samples: tuple[Sample, ...]

    def split(
        self,
        validation_fraction: float = 0.2,
        test_fraction: float = 0.2,
        seed: int = 0,
    ) -> Split:
        """Splits clips by explicit group identity to prevent leakage.

        A caller should use a subject, session, camera, or other identity as
        ``group_id``. Without one, source identity is the conservative group
        and repeated clips are not assumed to share an identity.

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
            grouped.setdefault(sample.group_id or sample.source, []).append(
                sample
            )
        validation_groups, test_groups = allocate(
            grouped, validation_fraction, test_fraction, seed
        )

        def sieve(excluded: set[str]) -> tuple[Sample, ...]:
            return tuple(
                sample
                for sample in self.samples
                if (sample.group_id or sample.source) not in excluded
            )

        validation = tuple(
            sample
            for sample in self.samples
            if (sample.group_id or sample.source) in validation_groups
        )
        test = tuple(
            sample
            for sample in self.samples
            if (sample.group_id or sample.source) in test_groups
        )
        return Split(
            train=sieve(test_groups | validation_groups),
            validation=validation,
            test=test,
        )

    def cutover(self, cutoff: datetime | str) -> Chronology:
        """Splits fully timestamped videos into past and future partitions.

        Videos at the cutoff belong to training. Videos after the cutoff
        belong to future evaluation. Untimestamped videos fail closed because
        assigning them a temporal position would invent ordering evidence.

        Args:
            cutoff: Timezone-aware datetime or ISO 8601 text.

        Returns:
            The split, with the cutoff in UTC.

        Raises:
            VideoTemporalError: If any video has no observation time, or the
                cutoff is invalid (see `canonical_observation_time`).
        """
        cutoff_time = canonical_observation_time(cutoff)
        if any(sample.observation_time is None for sample in self.samples):
            raise VideoTemporalError(
                "temporal splitting requires an observation time on every "
                "video",
                {"code": "incomplete_observation_times"},
            )
        train = tuple(
            sample
            for sample in self.samples
            if isinstance(sample.observation_time, datetime)
            and sample.observation_time <= cutoff_time
        )
        future = tuple(
            sample
            for sample in self.samples
            if isinstance(sample.observation_time, datetime)
            and sample.observation_time > cutoff_time
        )
        return Chronology(
            cutoff=cutoff_time,
            train=train,
            future=future,
        )
