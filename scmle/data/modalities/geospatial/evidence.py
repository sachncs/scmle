"""Evidence records that geospatial operations return."""

from collections.abc import Mapping
import dataclasses
from datetime import datetime
from datetime import UTC
import math
from typing import Any
from typing import ClassVar

import numpy

from scmle.contracts import errors
from scmle.contracts import records
from scmle.data.modalities.geospatial.geometry import MIN_OBSERVATIONS
from scmle.data.modalities.geospatial.geometry import Placemark
from scmle.data.modalities.geospatial.geometry import verify


@dataclasses.dataclass(frozen=True, slots=True)
class Match:
    """One deterministic pair emitted by a bounded spatial join.

    Attributes:
        left: ID of the matched feature in the left dataset.
        right: ID of the matched feature in the right dataset.
    """

    left: str
    right: str

    KEYS: ClassVar[records.Migration] = records.Migration(
        renames={"left_feature_id": "left", "right_feature_id": "right"}
    )
    """Version-1 keys to the written ones; `downgrade` restores version 1."""

    def __post_init__(self) -> None:
        """Validates both feature identities before recording a match.

        Raises:
            ValueError: If either feature ID is empty.
        """
        if not self.left.strip():
            raise errors.InputError("left_feature_id must not be empty")
        if not self.right.strip():
            raise errors.InputError("right_feature_id must not be empty")

    def to_dict(self) -> dict[str, str]:
        """Returns JSON-native join-match evidence.

        Returns:
            A mapping with the left and right feature IDs under `left` and
            `right`.
        """
        return self.KEYS.upgrade(dataclasses.asdict(self))


@dataclasses.dataclass(frozen=True, slots=True)
class Join:
    """Bounded, same-CRS evidence from a caller-owned spatial predicate.

    Attributes:
        left_dataset_hash: Content hash of the left dataset.
        right_dataset_hash: Content hash of the right dataset.
        crs: The shared CRS of both datasets.
        predicate: Stable name and version of the predicate.
        left_feature_count: Number of features in the left dataset.
        right_feature_count: Number of features in the right dataset.
        comparisons: Number of feature pairs the predicate was applied
            to.
        matched: Number of matching pairs.
        unmatched_left_count: Number of left features with no match.
        unmatched_right_count: Number of right features with no match.
        limit: Maximum comparisons the join was allowed.
        match_limit: Maximum matches the join was allowed to retain.
        matches: The matching feature-ID pairs, in comparison order.
    """

    left_dataset_hash: str
    right_dataset_hash: str
    crs: str
    predicate: str
    left_feature_count: int
    right_feature_count: int
    comparisons: int
    matched: int
    unmatched_left_count: int
    unmatched_right_count: int
    limit: int
    match_limit: int
    matches: tuple[Match, ...]

    KEYS: ClassVar[records.Migration] = records.nest(
        {
            "predicate_name": "predicate",
            "comparison_count": "comparisons",
            "match_count": "matched",
        },
        groups={
            "digests": {
                "left_dataset_hash": "left",
                "right_dataset_hash": "right",
            },
            "features": {
                "left_feature_count": "left",
                "right_feature_count": "right",
            },
            "unmatched": {
                "unmatched_left_count": "left",
                "unmatched_right_count": "right",
            },
            "limits": {
                "comparison_limit": "comparisons",
                "match_limit": "matches",
            },
        },
    )
    """Version-1 keys to the written ones; `downgrade` restores version 1."""

    def __post_init__(self) -> None:
        """Rejects incomplete or internally inconsistent join evidence.

        Raises:
            ValueError: If a dataset hash is not a lowercase SHA-256 hex
                digest, text fields are empty, counts or limits are out of
                range or inconsistent with each other, ``matched`` differs
                from the number of matches, or matches are duplicated.
        """
        for field_name, value in (
            ("left_dataset_hash", self.left_dataset_hash),
            ("right_dataset_hash", self.right_dataset_hash),
        ):
            verify(field_name, value)
        if not self.crs.strip():
            raise errors.InputError("crs must not be empty")
        if not self.predicate.strip():
            raise errors.InputError("predicate_name must not be empty")
        self.census()
        if self.limit < 1 or self.match_limit < 1:
            raise errors.InputError("join limits must be positive")
        identities = tuple((match.left, match.right) for match in self.matches)
        if len(identities) != len(set(identities)):
            raise errors.InputError("join matches must be unique")

    def census(self) -> None:
        """Checks that the join counts are positive and consistent.

        Raises:
            ValueError: If feature counts are not positive, a count is
                negative, a count exceeds the one it is bounded by, or
                ``matched`` differs from the number of matches.
        """
        if min(self.left_feature_count, self.right_feature_count) < 1:
            raise errors.InputError("join feature counts must be positive")
        if (
            min(
                self.comparisons,
                self.matched,
                self.unmatched_left_count,
                self.unmatched_right_count,
            )
            < 0
        ):
            raise errors.InputError("join counts must not be negative")
        if self.comparisons < self.matched:
            raise errors.InputError("comparison_count must cover every match")
        if self.unmatched_left_count > self.left_feature_count:
            raise errors.InputError(
                "unmatched_left_count exceeds left feature count"
            )
        if self.unmatched_right_count > self.right_feature_count:
            raise errors.InputError(
                "unmatched_right_count exceeds right feature count"
            )
        if self.matched != len(self.matches):
            raise errors.InputError(
                "match_count must match the recorded matches"
            )

    def to_dict(self) -> dict[str, Any]:
        """Returns JSON-native join evidence without raw geometry.

        Returns:
            A mapping of every report field in single-word keys, with
            matches as a list of mappings.
        """
        return self.KEYS.upgrade(
            {
                "left_dataset_hash": self.left_dataset_hash,
                "right_dataset_hash": self.right_dataset_hash,
                "crs": self.crs,
                "predicate_name": self.predicate,
                "left_feature_count": self.left_feature_count,
                "right_feature_count": self.right_feature_count,
                "comparison_count": self.comparisons,
                "match_count": self.matched,
                "unmatched_left_count": self.unmatched_left_count,
                "unmatched_right_count": self.unmatched_right_count,
                "comparison_limit": self.limit,
                "match_limit": self.match_limit,
                "matches": [match.to_dict() for match in self.matches],
            }
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Provenance:
    """Evidence binding a caller-owned coordinate transformation.

    Attributes:
        source_dataset_hash: Content hash of the dataset before
            transformation.
        target_dataset_hash: Content hash of the transformed dataset.
        source_crs: CRS of the source dataset.
        target_crs: CRS of the transformed dataset.
        transformer: Stable name and version of the transformer.
        features: Number of features transformed.
    """

    source_dataset_hash: str
    target_dataset_hash: str
    source_crs: str
    target_crs: str
    transformer: str
    features: int

    KEYS: ClassVar[records.Migration] = records.nest(
        {"transformer_name": "transformer", "feature_count": "features"},
        groups={
            "digests": {
                "source_dataset_hash": "source",
                "target_dataset_hash": "target",
            },
            "crs": {"source_crs": "source", "target_crs": "target"},
        },
    )
    """Version-1 keys to the written ones; `downgrade` restores version 1."""

    def __post_init__(self) -> None:
        """Validates reprojection identity and scope.

        Raises:
            ValueError: If a dataset hash is not a lowercase SHA-256 hex
                digest, a CRS or the transformer name is empty, the two CRS
                values are equal, or ``features`` is not positive.
        """
        for field_name, value in (
            ("source_dataset_hash", self.source_dataset_hash),
            ("target_dataset_hash", self.target_dataset_hash),
        ):
            verify(field_name, value)
        for field_name, value in (
            ("source_crs", self.source_crs),
            ("target_crs", self.target_crs),
            ("transformer_name", self.transformer),
        ):
            if not value.strip():
                raise errors.InputError(f"{field_name} must not be empty")
        if self.source_crs == self.target_crs:
            raise errors.InputError("source_crs and target_crs must differ")
        if self.features < 1:
            raise errors.InputError("feature_count must be positive")

    def to_dict(self) -> dict[str, Any]:
        """Returns JSON-native reprojection evidence.

        Returns:
            A mapping of every report field in single-word keys.
        """
        return self.KEYS.upgrade(dataclasses.asdict(self))


@dataclasses.dataclass(frozen=True, slots=True)
class Leakage:
    """Potential envelope leakage across CRS-unit spatial blocks.

    Attributes:
        block: Block edge length in CRS units.
        features: Number of features examined.
        blocks: Pairs of feature ID and the number of blocks
            its envelope touches, one per feature.
        crossing: IDs of features whose envelope touches more
            than one block.
        truncated: IDs of crossing features whose block count
            exceeds ``spread``.
        spread: Block count above which a feature is marked
            truncated.
    """

    block: float
    features: int
    blocks: tuple[tuple[str, int], ...]
    crossing: tuple[str, ...]
    truncated: tuple[str, ...]
    spread: int

    KEYS: ClassVar[records.Migration] = records.Migration(
        renames={
            "block_size": "block",
            "feature_count": "features",
            "feature_block_counts": "blocks",
            "crossing_feature_ids": "crossing",
            "truncated_feature_ids": "truncated",
            "max_blocks_per_feature": "spread",
        }
    )
    """Version-1 keys to the written ones; `downgrade` restores version 1."""

    def __post_init__(self) -> None:
        """Validates bounded and internally consistent leakage evidence.

        Raises:
            ValueError: If ``block`` is not positive and finite, counts
                or limits are not positive, the block counts do not cover
                each feature exactly once, or the crossing and truncated ID
                sets are not nested within the reported features.
        """
        if not math.isfinite(self.block) or self.block <= 0:
            raise errors.InputError(
                "block_size must be a positive finite number"
            )
        if self.features < 1:
            raise errors.InputError("feature_count must be positive")
        if self.spread < 1:
            raise errors.InputError("max_blocks_per_feature must be positive")
        if len(self.blocks) != self.features:
            raise errors.InputError(
                "feature block counts must match feature count"
            )
        feature_ids = tuple(item[0] for item in self.blocks)
        if len(set(feature_ids)) != len(feature_ids):
            raise errors.InputError("feature block counts must have unique IDs")
        if any(item[1] < 1 for item in self.blocks):
            raise errors.InputError("feature block counts must be positive")
        crossing = set(self.crossing)
        truncated = set(self.truncated)
        if not crossing.issubset(feature_ids):
            raise errors.InputError(
                "crossing feature IDs must belong to the report"
            )
        if not truncated.issubset(crossing):
            raise errors.InputError(
                "truncated feature IDs must cross block boundaries"
            )

    @property
    def status(self) -> str:
        """The leakage status, ``"potential_leakage"`` or ``"clear"``."""
        return "potential_leakage" if self.crossing else "clear"

    def to_dict(self) -> dict[str, Any]:
        """Returns JSON-native boundary-leakage evidence.

        Returns:
            A mapping of the status and every report field in single-word
            keys, with block counts as a list of ID and count mappings.
        """
        return self.KEYS.upgrade(
            {
                "status": self.status,
                "block_size": self.block,
                "feature_count": self.features,
                "feature_block_counts": [
                    {"feature_id": feature_id, "block_count": block_count}
                    for feature_id, block_count in self.blocks
                ],
                "crossing_feature_ids": list(self.crossing),
                "truncated_feature_ids": list(self.truncated),
                "max_blocks_per_feature": self.spread,
            }
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Profile:
    """Bounded data-understanding summary for a vector dataset.

    Attributes:
        features: Number of features in the dataset.
        geometries: Sorted distinct GeoJSON geometry type names.
        crs: Declared CRS of the dataset, possibly empty.
        bounds: Dataset-wide (min x, min y, max x, max y) in CRS units.
        content_hash: Content hash of the dataset.
    """

    features: int
    geometries: tuple[str, ...]
    crs: str
    bounds: tuple[float, float, float, float]
    content_hash: str

    KEYS: ClassVar[records.Migration] = records.nest(
        {"feature_count": "features", "geometry_types": "geometries"},
        groups={"digests": {"content_hash": "content"}},
    )
    """Version-1 keys to the written ones; `downgrade` restores version 1."""

    def to_dict(self) -> dict[str, Any]:
        """Returns JSON-compatible geospatial profile data.

        Returns:
            A mapping of every profile field in single-word keys.
        """
        return self.KEYS.upgrade(dataclasses.asdict(self))


@dataclasses.dataclass(frozen=True, slots=True)
class Partition:
    """Spatial-block split that keeps nearby feature blocks together.

    Attributes:
        train: Features assigned to training.
        validation: Features assigned to validation.
        test: Features assigned to testing.
        block_size: Block edge length in CRS units.
        blocks: Block IDs assigned to each partition, keyed by ``"train"``,
            ``"validation"``, and ``"test"``.
    """

    train: tuple[Placemark, ...]
    validation: tuple[Placemark, ...]
    test: tuple[Placemark, ...]
    block_size: float
    blocks: Mapping[str, tuple[str, ...]]

    def tiles(self) -> dict[str, tuple[str, ...]]:
        """Returns the stable spatial blocks assigned to each partition.

        Returns:
            A new mapping from partition name to a tuple of block IDs.
        """
        return {name: tuple(values) for name, values in self.blocks.items()}


@dataclasses.dataclass(frozen=True, slots=True)
class Chronology:
    """Time-ordered geospatial features separated at a UTC cutoff.

    Attributes:
        cutoff: Timezone-aware cutoff, converted to UTC.
        train: Features observed at or before the cutoff.
        future: Features observed after the cutoff.
    """

    cutoff: datetime
    train: tuple[Placemark, ...]
    future: tuple[Placemark, ...]

    def __post_init__(self) -> None:
        """Validates the cutoff and timestamp completeness of each partition.

        Raises:
            ValueError: If the cutoff has no timezone or any feature lacks an
                observation time.
        """
        if self.cutoff.tzinfo is None or self.cutoff.utcoffset() is None:
            raise errors.InputError("cutoff must be timezone-aware")
        object.__setattr__(self, "cutoff", self.cutoff.astimezone(UTC))
        if any(feature.time is None for feature in self.train):
            raise errors.InputError(
                "train features must contain observation times"
            )
        if any(feature.time is None for feature in self.future):
            raise errors.InputError(
                "future features must contain observation times"
            )

    @property
    def lookahead(self) -> bool:
        """Whether any training feature was observed after the cutoff."""
        return any(
            isinstance(feature.time, datetime) and feature.time > self.cutoff
            for feature in self.train
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Autocorrelation:
    """Bounded global Moran's I evidence for a numeric spatial variable.

    Attributes:
        features: Number of features analysed.
        neighbors: Nearest neighbors used per feature.
        edges: Number of undirected neighbor edges.
        statistic: Observed global Moran's I.
        permutations: Number of permutations used for the p-value.
        p_value: Two-sided permutation p-value, or None when no permutations
            were run.
        seed: Seed of the permutation random generator.
    """

    features: int
    neighbors: int
    edges: int
    statistic: float
    permutations: int
    p_value: float | None
    seed: int

    KEYS: ClassVar[records.Migration] = records.Migration(
        renames={
            "feature_count": "features",
            "neighbor_count": "neighbors",
            "edge_count": "edges",
            "permutation_count": "permutations",
        }
    )
    """Version-1 keys to the written ones; `downgrade` restores version 1."""

    def __post_init__(self) -> None:
        """Validates a finite and reproducible spatial statistic.

        Raises:
            ValueError: If there are fewer than three features, the neighbor
                count is outside ``[1, feature_count)``, there are no edges,
                the statistic is not finite, the permutation count is
                negative, or the p-value is outside ``[0, 1]``.
        """
        if self.features < MIN_OBSERVATIONS:
            raise errors.InputError("feature_count must be at least three")
        if not 1 <= self.neighbors < self.features:
            raise errors.InputError(
                "neighbor_count must be in [1, feature_count)"
            )
        if self.edges < 1:
            raise errors.InputError("edge_count must contain at least one edge")
        if not math.isfinite(self.statistic):
            raise errors.InputError("statistic must be finite")
        if self.permutations < 0:
            raise errors.InputError("permutation_count must not be negative")
        if self.p_value is not None and not 0.0 <= self.p_value <= 1.0:
            raise errors.InputError("p_value must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        """Returns JSON-native autocorrelation evidence.

        Returns:
            A mapping of every field in single-word keys.
        """
        return self.KEYS.upgrade(dataclasses.asdict(self))


@dataclasses.dataclass(frozen=True, slots=True)
class Features:
    """Numeric geometry features tied to an explicit CRS.

    Attributes:
        feature_ids: Feature IDs, one per row of ``values``.
        feature_names: Names of the columns of ``values``.
        values: Row-major feature matrix of floats.
        crs: CRS in which the values are expressed.
    """

    feature_ids: tuple[str, ...]
    feature_names: tuple[str, ...]
    values: tuple[tuple[float, ...], ...]
    crs: str

    @property
    def shape(self) -> tuple[int, int]:
        """The matrix shape as (number of geometries, number of features)."""
        return (len(self.feature_ids), len(self.feature_names))

    def materialize(self) -> numpy.ndarray:
        """Returns a writable copy of the numeric feature matrix.

        Returns:
            A float64 array with one row per geometry and one column per
            feature name.
        """
        return numpy.asarray(self.values, dtype=numpy.float64).copy()
