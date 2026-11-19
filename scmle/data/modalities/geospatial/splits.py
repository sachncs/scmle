"""Spatial-block and temporal splits of a geospatial vector dataset."""

from datetime import datetime
import math

from scmle.contracts import errors
from scmle.data.modalities.curation import allocate
from scmle.data.modalities.curation import proportions
from scmle.data.modalities.geospatial.evidence import Chronology
from scmle.data.modalities.geospatial.evidence import Leakage
from scmle.data.modalities.geospatial.evidence import Partition
from scmle.data.modalities.geospatial.geometry import GeometryError
from scmle.data.modalities.geospatial.geometry import Placemark
from scmle.data.modalities.geospatial.geometry import TemporalError
from scmle.data.modalities.geospatial.geometry import timestamp
from scmle.data.modalities.geospatial.survey import Survey


class Splits(Survey):
    """Spatial-block and temporal splits of a vector dataset."""

    __slots__ = ()

    def envelop(
        self, block_size: float
    ) -> dict[str, tuple[int, int, int, int]]:
        """Returns each feature envelope in CRS-unit block coordinates.

        The envelope is conservative: a feature is considered to touch every
        block intersected by its axis-aligned bounds. It is intended for
        curation and leakage review, not as a replacement for a geometry
        engine's exact intersection operation.

        Args:
            block_size: Block edge length in CRS units; positive and finite.

        Returns:
            A mapping from feature ID to (min x block, min y block, max x
            block, max y block) indexes, counted from the CRS origin so that
            adding a feature moves no other feature's block (MM-11).

        Raises:
            GeometryError: If the dataset has no CRS.
            ValueError: If ``block_size`` is not positive and finite.
        """
        if not self.crs.strip():
            raise GeometryError(
                "spatial block envelopes require a declared CRS",
                {"code": "missing_crs"},
            )
        if not math.isfinite(block_size) or block_size <= 0:
            raise errors.InputError(
                "block_size must be a positive finite number"
            )
        origin_x = origin_y = 0.0  # the CRS origin, fixed (MM-11)
        envelopes: dict[str, tuple[int, int, int, int]] = {}
        for feature in self.features:
            minimum_x, minimum_y, maximum_x, maximum_y = feature.bounds
            envelopes[feature.identifier] = (
                math.floor((minimum_x - origin_x) / block_size),
                math.floor((minimum_y - origin_y) / block_size),
                math.floor((maximum_x - origin_x) / block_size),
                math.floor((maximum_y - origin_y) / block_size),
            )
        return envelopes

    def straddle(
        self,
        block_size: float,
        max_blocks_per_feature: int = 10_000,
    ) -> Leakage:
        """Measures conservative envelope crossings for a spatial split.

        A crossing feature is not automatically leaking: the geometry may not
        occupy every block in its envelope. The evidence identifies the
        feature for domain review before a centroid-based split is promoted.
        Large envelopes are counted arithmetically and marked truncated rather
        than expanded into an unbounded block list.

        Args:
            block_size: Block edge length in CRS units; positive and finite.
            max_blocks_per_feature: Block count above which a feature is
                marked truncated.

        Returns:
            Per-feature block counts and the IDs of crossing and truncated
            features. A dataset without a CRS raises
            ``GeometryError``.

        Raises:
            ValueError: If ``max_blocks_per_feature`` is below one or
                ``block_size`` is not positive and finite.
        """
        if max_blocks_per_feature < 1:
            raise errors.InputError("max_blocks_per_feature must be positive")
        envelopes = self.envelop(block_size)
        block_counts: list[tuple[str, int]] = []
        crossing_feature_ids: list[str] = []
        truncated_feature_ids: list[str] = []
        for feature in self.features:
            minimum_x, minimum_y, maximum_x, maximum_y = envelopes[
                feature.identifier
            ]
            block_count = (maximum_x - minimum_x + 1) * (
                maximum_y - minimum_y + 1
            )
            block_counts.append((feature.identifier, block_count))
            if block_count > 1:
                crossing_feature_ids.append(feature.identifier)
            if block_count > max_blocks_per_feature:
                truncated_feature_ids.append(feature.identifier)
        return Leakage(
            block=block_size,
            features=len(self.features),
            blocks=tuple(block_counts),
            crossing=tuple(crossing_feature_ids),
            truncated=tuple(truncated_feature_ids),
            spread=max_blocks_per_feature,
        )

    def tile(
        self,
        block_size: float,
        validation_fraction: float = 0.2,
        test_fraction: float = 0.2,
        seed: int = 0,
    ) -> Partition:
        """Splits features by CRS-unit spatial blocks.

        ``block_size`` is expressed in the dataset CRS units. The method uses
        geometry centroids for block assignment and keeps every block in one
        partition. Large geometries that cross block boundaries need a
        domain-specific spatial join policy before this split is suitable for
        release.

        Args:
            block_size: Block edge length in CRS units; positive and finite.
            validation_fraction: Fraction of blocks for validation, in
                ``[0, 1)``.
            test_fraction: Fraction of blocks for testing, in ``[0, 1)``.
                The two fractions must sum to less than one.
            seed: Seed for the block shuffle.

        Returns:
            Train, validation, and test features with the block IDs assigned
            to each. At least one block stays in training. With three or
            more blocks a nonzero fraction is first raised to one block, but
            if the counts then leave no block for training, validation
            gives blocks back before testing does, so a small validation
            fraction can end with none.

        Raises:
            ValueError: If ``block_size`` is not positive and finite, a
                fraction is outside ``[0, 1)``, or the two fractions sum to
                one or more.
        """
        if not math.isfinite(block_size) or block_size <= 0:
            raise errors.InputError(
                "block_size must be a positive finite number"
            )
        proportions(validation_fraction, test_fraction)
        origin_x = origin_y = 0.0  # the CRS origin, fixed (MM-11)

        def locate(feature: Placemark) -> str:
            minimum_x, minimum_y, maximum_x, maximum_y = feature.bounds
            centroid_x = (minimum_x + maximum_x) / 2
            centroid_y = (minimum_y + maximum_y) / 2
            block_x = math.floor((centroid_x - origin_x) / block_size)
            block_y = math.floor((centroid_y - origin_y) / block_size)
            return f"{block_x}:{block_y}"

        grouped: dict[str, list[Placemark]] = {}
        for feature in self.features:
            grouped.setdefault(locate(feature), []).append(feature)
        blocks = sorted(grouped)
        validation_blocks, test_blocks = allocate(
            grouped, validation_fraction, test_fraction, seed
        )

        def features_for(selected: set[str]) -> tuple[Placemark, ...]:
            return tuple(
                feature
                for feature in self.features
                if locate(feature) in selected
            )

        training = set(blocks) - test_blocks - validation_blocks
        return Partition(
            train=features_for(training),
            validation=features_for(validation_blocks),
            test=features_for(test_blocks),
            block_size=block_size,
            blocks={
                "train": tuple(sorted(training)),
                "validation": tuple(sorted(validation_blocks)),
                "test": tuple(sorted(test_blocks)),
            },
        )

    def cutover(self, cutoff: datetime | str) -> Chronology:
        """Splits fully timestamped features into past and future partitions.

        Features at the cutoff belong to training. Features after the cutoff
        belong to future evaluation. Untimestamped features fail closed because
        assigning them a temporal position would invent ordering evidence.

        Args:
            cutoff: Timezone-aware cutoff datetime or ISO 8601 string.

        Returns:
            The features observed at or before the cutoff and those observed
            after it, with the cutoff in UTC.

        Raises:
            TemporalError: If the cutoff is invalid or lacks a
                timezone, or any feature has no observation time.
        """
        cutoff_time = timestamp(cutoff)
        if any(feature.time is None for feature in self.features):
            raise TemporalError(
                "temporal splitting requires an observation time on every "
                "feature",
                {"code": "incomplete_observation_times"},
            )
        train = tuple(
            feature
            for feature in self.features
            if isinstance(feature.time, datetime)
            and feature.time <= cutoff_time
        )
        future = tuple(
            feature
            for feature in self.features
            if isinstance(feature.time, datetime) and feature.time > cutoff_time
        )
        return Chronology(
            cutoff=cutoff_time,
            train=train,
            future=future,
        )
