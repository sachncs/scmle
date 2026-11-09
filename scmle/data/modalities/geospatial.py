"""Dependency-light vector and raster geospatial data contracts."""

from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
from datetime import datetime
from datetime import UTC
import hashlib
import json
import math
from numbers import Real
import pathlib
import random
from types import MappingProxyType
from typing import Any
from typing import cast
from typing import Self

import numpy as np

from scmle.contracts import data as data_contracts
from scmle.contracts import errors
from scmle.data.modalities.curation import build_finding
from scmle.data.modalities.curation import label_counts
from scmle.data.modalities.curation import ModalityCurationReport

GEOMETRY_TYPES = frozenset(
    {
        "Point",
        "MultiPoint",
        "LineString",
        "MultiLineString",
        "Polygon",
        "MultiPolygon",
        "GeometryCollection",
    }
)

RasterDecoder = Callable[[bytes], np.ndarray]


class GeospatialValidationError(errors.DataError):
    """A vector or raster sample violates a geospatial invariant."""

    category = "geospatial_validation"


class GeospatialTemporalError(GeospatialValidationError):
    """Temporal geospatial evidence is incomplete or invalid."""


def canonical_source(source: str | pathlib.Path) -> str:
    """Returns a stable source identifier for a path or URI-like value.

    Args:
        source: A filesystem path or a URI-like string.

    Returns:
        ``"memory://"`` when the stripped value is empty, the stripped value
        unchanged when it contains ``"://"``, and otherwise the absolute
        ``file://`` URI of the user-expanded, resolved path.
    """
    value = str(source).strip()
    if not value:
        return "memory://"
    if "://" in value:
        return value
    return pathlib.Path(value).expanduser().resolve(strict=False).as_uri()


def canonical_observation_time(value: datetime | str) -> datetime:
    """Parses and normalizes a timezone-aware observation time to UTC.

    Args:
        value: A timezone-aware datetime or an ISO 8601 string. A trailing
            ``Z`` or ``z`` is read as UTC.

    Returns:
        The observation time converted to UTC.

    Raises:
        GeospatialTemporalError: If the text is not a valid ISO 8601
            timestamp, if the value is neither a datetime nor text, or if the
            parsed time has no timezone offset.
    """
    if isinstance(value, str):
        normalized = value.strip()
        if normalized.endswith(("Z", "z")):
            normalized = normalized[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError as error:
            raise GeospatialTemporalError(
                f"invalid observation time: {value!r}",
                {"code": "invalid_observation_time"},
            ) from error
    elif isinstance(value, datetime):
        parsed = value
    else:
        raise GeospatialTemporalError(
            "observation time must be datetime or text",
            {"code": "invalid_observation_time"},
        )
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise GeospatialTemporalError(
            "observation time must be timezone-aware",
            {"code": "timezone_required"},
        )
    return parsed.astimezone(UTC)


def validate_coordinates(value: Any) -> tuple[float, ...]:
    """Validates a nested coordinate array and return its numeric leaves.

    Args:
        value: A list or tuple that is either a position (at least two finite
            non-boolean numbers) or a nested list or tuple of such arrays.

    Returns:
        Every coordinate number, as floats, flattened in document order.

    Raises:
        GeospatialValidationError: If an array is empty, a position has fewer
            than two numbers or a non-finite number, or a value is not a
            list, tuple, or number.
    """
    if isinstance(value, (list, tuple)):
        if not value:
            raise GeospatialValidationError(
                "coordinate arrays must not be empty",
                {"code": "empty_coordinates"},
            )
        if all(
            isinstance(item, (int, float)) and not isinstance(item, bool)
            for item in value
        ):
            numbers = tuple(float(item) for item in value)
            if len(numbers) < 2 or not all(
                math.isfinite(item) for item in numbers
            ):
                raise GeospatialValidationError(
                    "coordinates must contain finite x/y values",
                    {"code": "invalid_coordinates"},
                )
            return numbers
        flattened: list[float] = []
        for item in value:
            flattened.extend(validate_coordinates(item))
        return tuple(flattened)
    raise GeospatialValidationError(
        "coordinates must be nested numeric arrays",
        {"code": "invalid_coordinates"},
    )


def geometry_coordinates(geometry: Mapping[str, Any]) -> tuple[float, ...]:
    """Validates a GeoJSON geometry and return all coordinate leaves.

    Args:
        geometry: A GeoJSON geometry object. Geometry collections are
            validated recursively.

    Returns:
        Every coordinate number, as floats, flattened in document order
        (x, y, then any further dimensions of each position).

    Raises:
        GeospatialValidationError: If the geometry type is not a supported
            GeoJSON type, a geometry collection has no geometries or has a
            member that is not an object, or the coordinates are invalid.
    """
    geometry_type = geometry.get("type")
    if geometry_type not in GEOMETRY_TYPES:
        raise GeospatialValidationError(
            "geometry type is not supported",
            {"code": "unsupported_geometry", "geometry_type": geometry_type},
        )
    if geometry_type == "GeometryCollection":
        geometries = geometry.get("geometries")
        if not isinstance(geometries, list) or not geometries:
            raise GeospatialValidationError(
                "geometry collection must contain geometries",
                {"code": "empty_geometry_collection"},
            )
        values: list[float] = []
        for nested in geometries:
            if not isinstance(nested, Mapping):
                raise GeospatialValidationError(
                    "geometry collection members must be objects",
                    {"code": "invalid_geometry_collection"},
                )
            values.extend(geometry_coordinates(nested))
        return tuple(values)
    return validate_coordinates(geometry.get("coordinates"))


def moran_statistic(
    values: np.ndarray, edges: Sequence[tuple[int, int]]
) -> float:
    """Calculates global Moran's I for an undirected binary graph.

    Args:
        values: One-dimensional numeric values aligned with the feature
            order. At least three observations are required.
        edges: Undirected feature-neighbor pairs, given as indexes into
            ``values``. Each pair is counted in both directions.

    Returns:
        The global Moran's I statistic.

    Raises:
        ValueError: If values are not one-dimensional with at least three
            observations, are not finite, have no variation, or if no edges
            are given.
    """
    numeric = np.asarray(values, dtype=np.float64)
    if numeric.ndim != 1 or len(numeric) < 3:
        raise ValueError("values must contain at least three observations")
    if not np.isfinite(numeric).all():
        raise ValueError("values must be finite")
    centered = numeric - numeric.mean()
    denominator = float(np.dot(centered, centered))
    if denominator == 0.0:
        raise ValueError("values must contain more than one unique value")
    weight_total = 2 * len(edges)
    if weight_total == 0:
        raise ValueError("edges must contain at least one neighbor pair")
    numerator = sum(
        centered[first] * centered[second] * 2 for first, second in edges
    )
    return float(len(numeric) * numerator / (weight_total * denominator))


def mask_geometry_axis(
    geometry: Mapping[str, Any], axis: int, baseline: float
) -> dict[str, Any]:
    """Replaces one coordinate axis while preserving geometry structure.

    The replacement is performed in the source CRS and keeps the geometry
    type, nesting, and non-masked coordinate dimensions intact. It is a
    bounded perturbation primitive for raw-geometry evidence; it does not
    validate domain-specific topology or spatial semantics.

    Args:
        geometry: A GeoJSON geometry object. Geometry collections are masked
            member by member.
        axis: The coordinate index to replace, 0 for x or 1 for y.
        baseline: The finite value written to that axis of every position.

    Returns:
        A new geometry mapping with the same structure as ``geometry`` and
        the chosen axis of every position set to ``baseline``. Other
        coordinates are converted to floats. A coordinate array that is
        empty or not a nested list or tuple, or a geometry collection with no
        geometries, raises ``GeospatialValidationError``.

    Raises:
        ValueError: If ``axis`` is not 0 or 1 or ``baseline`` is not finite.
    """
    if axis not in (0, 1):
        raise ValueError("axis must be 0 for x or 1 for y")
    if not math.isfinite(float(baseline)):
        raise ValueError("baseline must be finite")

    def masked_coordinates(value: Any) -> list[Any]:
        if not isinstance(value, (list, tuple)) or not value:
            raise GeospatialValidationError(
                "coordinates must be non-empty nested arrays",
                {"code": "invalid_coordinates"},
            )
        if all(
            isinstance(item, (int, float)) and not isinstance(item, bool)
            for item in value
        ):
            return [
                float(baseline) if index == axis else float(item)
                for index, item in enumerate(value)
            ]
        return [masked_coordinates(item) for item in value]

    def masked_geometry(value: Mapping[str, Any]) -> dict[str, Any]:
        geometry_type = value.get("type")
        result = dict(value)
        if geometry_type == "GeometryCollection":
            geometries = value.get("geometries")
            if not isinstance(geometries, list) or not geometries:
                raise GeospatialValidationError(
                    "geometry collection must contain geometries",
                    {"code": "empty_geometry_collection"},
                )
            result["geometries"] = [
                masked_geometry(item) for item in geometries
            ]
        else:
            result["coordinates"] = masked_coordinates(value.get("coordinates"))
        return result

    return masked_geometry(geometry)


@dataclasses.dataclass(frozen=True, slots=True)
class GeoFeature:
    """Validated GeoJSON feature with stable serialized identity.

    Attributes:
        feature_id: Non-empty identifier of the feature.
        geometry: GeoJSON geometry object.
        properties: JSON-native feature properties, exposed as a read-only
            mapping after validation.
        label: Optional JSON-native label of the feature.
        observation_time: Optional observation time, normalized to a UTC
            datetime when given.
        geometry_hash: SHA-256 hex digest of the canonical geometry JSON.
        content_hash: SHA-256 hex digest of the canonical feature identity,
            covering ID, geometry hash, label, observation time, and
            properties.
    """

    feature_id: str
    geometry: Mapping[str, Any]
    properties: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    label: str | int | float | bool | None = None
    observation_time: datetime | str | None = None
    geometry_hash: str = dataclasses.field(init=False)
    content_hash: str = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        """Validates feature data and derive canonical content identities.

        Raises:
            ValueError: If ``feature_id`` is empty.
            GeospatialValidationError: If the geometry is not an object or is
                invalid, if a property name is not a string, or if the
                properties and label are not JSON-native. An invalid or
                timezone-naive observation time raises
                ``GeospatialTemporalError``.
        """
        if not self.feature_id.strip():
            raise ValueError("feature_id must not be empty")
        if not isinstance(self.geometry, Mapping):
            raise GeospatialValidationError(
                "geometry must be an object", {"code": "invalid_geometry"}
            )
        geometry_coordinates(self.geometry)
        if self.observation_time is not None:
            object.__setattr__(
                self,
                "observation_time",
                canonical_observation_time(self.observation_time),
            )
        properties = dict(self.properties)
        if any(not isinstance(key, str) for key in properties):
            raise GeospatialValidationError(
                "feature property names must be strings",
                {"code": "invalid_property_name"},
            )
        try:
            json.dumps(
                {"label": self.label, "properties": properties},
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError) as error:
            raise GeospatialValidationError(
                "feature properties and label must be JSON-native",
                {"code": "invalid_feature_metadata"},
            ) from error
        object.__setattr__(self, "properties", MappingProxyType(properties))
        payload = json.dumps(
            self.geometry,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        object.__setattr__(
            self, "geometry_hash", hashlib.sha256(payload).hexdigest()
        )
        feature_payload = json.dumps(
            {
                "feature_id": self.feature_id,
                "geometry_hash": self.geometry_hash,
                "label": self.label,
                "observation_time": self.observation_time,
                "properties": properties,
            },
            default=lambda value: (
                value.isoformat() if isinstance(value, datetime) else value
            ),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        object.__setattr__(
            self, "content_hash", hashlib.sha256(feature_payload).hexdigest()
        )

    @property
    def geometry_type(self) -> str:
        """The validated GeoJSON geometry type name."""
        return str(self.geometry["type"])

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """The x/y bounds of the geometry as (min x, min y, max x, max y)."""
        coordinates = geometry_coordinates(self.geometry)
        xs = coordinates[0::2]
        ys = coordinates[1::2]
        return min(xs), min(ys), max(xs), max(ys)

    @property
    def centroid(self) -> tuple[float, float]:
        """The centre of the bounds as (x, y) in the source CRS units."""
        minimum_x, minimum_y, maximum_x, maximum_y = self.bounds
        return (minimum_x + maximum_x) / 2, (minimum_y + maximum_y) / 2

    def masked_geometry_axis(self, axis: int, baseline: float) -> "GeoFeature":
        """Returns a copy with one source-coordinate axis replaced.

        Args:
            axis: The coordinate index to replace, 0 for x or 1 for y.
            baseline: The finite value written to that axis of every
                position.

        Returns:
            A new feature with the masked geometry and the same ID,
            properties, label, and observation time.
        """
        return dataclasses.replace(
            self,
            geometry=mask_geometry_axis(self.geometry, axis, baseline),
        )


SpatialJoinPredicate = Callable[[GeoFeature, GeoFeature], bool]
CoordinateTransformer = Callable[[GeoFeature, str, str], GeoFeature]


@dataclasses.dataclass(frozen=True, slots=True)
class SpatialJoinMatch:
    """One deterministic pair emitted by a bounded spatial join.

    Attributes:
        left_feature_id: ID of the matched feature in the left dataset.
        right_feature_id: ID of the matched feature in the right dataset.
    """

    left_feature_id: str
    right_feature_id: str

    def __post_init__(self) -> None:
        """Validates both feature identities before recording a match.

        Raises:
            ValueError: If either feature ID is empty.
        """
        if not self.left_feature_id.strip():
            raise ValueError("left_feature_id must not be empty")
        if not self.right_feature_id.strip():
            raise ValueError("right_feature_id must not be empty")

    def to_dict(self) -> dict[str, str]:
        """Returns JSON-native join-match evidence.

        Returns:
            A mapping with the left and right feature IDs.
        """
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, slots=True)
class SpatialJoinReport:
    """Bounded, same-CRS evidence from a caller-owned spatial predicate.

    Attributes:
        left_dataset_hash: Content hash of the left dataset.
        right_dataset_hash: Content hash of the right dataset.
        crs: The shared CRS of both datasets.
        predicate_name: Stable name and version of the predicate.
        left_feature_count: Number of features in the left dataset.
        right_feature_count: Number of features in the right dataset.
        comparison_count: Number of feature pairs the predicate was applied
            to.
        match_count: Number of matching pairs.
        unmatched_left_count: Number of left features with no match.
        unmatched_right_count: Number of right features with no match.
        comparison_limit: Maximum comparisons the join was allowed.
        match_limit: Maximum matches the join was allowed to retain.
        matches: The matching feature-ID pairs, in comparison order.
    """

    left_dataset_hash: str
    right_dataset_hash: str
    crs: str
    predicate_name: str
    left_feature_count: int
    right_feature_count: int
    comparison_count: int
    match_count: int
    unmatched_left_count: int
    unmatched_right_count: int
    comparison_limit: int
    match_limit: int
    matches: tuple[SpatialJoinMatch, ...]

    def __post_init__(self) -> None:
        """Rejects incomplete or internally inconsistent join evidence.

        Raises:
            ValueError: If a dataset hash is not a lowercase SHA-256 hex
                digest, text fields are empty, counts or limits are out of
                range or inconsistent with each other, ``match_count`` differs
                from the number of matches, or matches are duplicated.
        """
        for field_name, value in (
            ("left_dataset_hash", self.left_dataset_hash),
            ("right_dataset_hash", self.right_dataset_hash),
        ):
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(f"{field_name} must be a SHA-256 hex digest")
        if not self.crs.strip():
            raise ValueError("crs must not be empty")
        if not self.predicate_name.strip():
            raise ValueError("predicate_name must not be empty")
        if min(self.left_feature_count, self.right_feature_count) < 1:
            raise ValueError("join feature counts must be positive")
        if (
            min(
                self.comparison_count,
                self.match_count,
                self.unmatched_left_count,
                self.unmatched_right_count,
            )
            < 0
        ):
            raise ValueError("join counts must not be negative")
        if self.comparison_count < self.match_count:
            raise ValueError("comparison_count must cover every match")
        if self.unmatched_left_count > self.left_feature_count:
            raise ValueError("unmatched_left_count exceeds left feature count")
        if self.unmatched_right_count > self.right_feature_count:
            raise ValueError(
                "unmatched_right_count exceeds right feature count"
            )
        if self.match_count != len(self.matches):
            raise ValueError("match_count must match the recorded matches")
        if self.comparison_limit < 1 or self.match_limit < 1:
            raise ValueError("join limits must be positive")
        identities = tuple(
            (match.left_feature_id, match.right_feature_id)
            for match in self.matches
        )
        if len(identities) != len(set(identities)):
            raise ValueError("join matches must be unique")

    def to_dict(self) -> dict[str, Any]:
        """Returns JSON-native join evidence without raw geometry.

        Returns:
            A mapping of every report field, with matches as a list of
            mappings.
        """
        return {
            "left_dataset_hash": self.left_dataset_hash,
            "right_dataset_hash": self.right_dataset_hash,
            "crs": self.crs,
            "predicate_name": self.predicate_name,
            "left_feature_count": self.left_feature_count,
            "right_feature_count": self.right_feature_count,
            "comparison_count": self.comparison_count,
            "match_count": self.match_count,
            "unmatched_left_count": self.unmatched_left_count,
            "unmatched_right_count": self.unmatched_right_count,
            "comparison_limit": self.comparison_limit,
            "match_limit": self.match_limit,
            "matches": [match.to_dict() for match in self.matches],
        }


@dataclasses.dataclass(frozen=True, slots=True)
class SpatialReprojectionReport:
    """Evidence binding a caller-owned coordinate transformation.

    Attributes:
        source_dataset_hash: Content hash of the dataset before
            transformation.
        target_dataset_hash: Content hash of the transformed dataset.
        source_crs: CRS of the source dataset.
        target_crs: CRS of the transformed dataset.
        transformer_name: Stable name and version of the transformer.
        feature_count: Number of features transformed.
    """

    source_dataset_hash: str
    target_dataset_hash: str
    source_crs: str
    target_crs: str
    transformer_name: str
    feature_count: int

    def __post_init__(self) -> None:
        """Validates reprojection identity and scope.

        Raises:
            ValueError: If a dataset hash is not a lowercase SHA-256 hex
                digest, a CRS or the transformer name is empty, the two CRS
                values are equal, or ``feature_count`` is not positive.
        """
        for field_name, value in (
            ("source_dataset_hash", self.source_dataset_hash),
            ("target_dataset_hash", self.target_dataset_hash),
        ):
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(f"{field_name} must be a SHA-256 hex digest")
        for field_name, value in (
            ("source_crs", self.source_crs),
            ("target_crs", self.target_crs),
            ("transformer_name", self.transformer_name),
        ):
            if not value.strip():
                raise ValueError(f"{field_name} must not be empty")
        if self.source_crs == self.target_crs:
            raise ValueError("source_crs and target_crs must differ")
        if self.feature_count < 1:
            raise ValueError("feature_count must be positive")

    def to_dict(self) -> dict[str, Any]:
        """Returns JSON-native reprojection evidence.

        Returns:
            A mapping of every report field.
        """
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, slots=True)
class SpatialBoundaryLeakage:
    """Potential envelope leakage across CRS-unit spatial blocks.

    Attributes:
        block_size: Block edge length in CRS units.
        feature_count: Number of features examined.
        feature_block_counts: Pairs of feature ID and the number of blocks
            its envelope touches, one per feature.
        crossing_feature_ids: IDs of features whose envelope touches more
            than one block.
        truncated_feature_ids: IDs of crossing features whose block count
            exceeds ``max_blocks_per_feature``.
        max_blocks_per_feature: Block count above which a feature is marked
            truncated.
    """

    block_size: float
    feature_count: int
    feature_block_counts: tuple[tuple[str, int], ...]
    crossing_feature_ids: tuple[str, ...]
    truncated_feature_ids: tuple[str, ...]
    max_blocks_per_feature: int

    def __post_init__(self) -> None:
        """Validates bounded and internally consistent leakage evidence.

        Raises:
            ValueError: If ``block_size`` is not positive and finite, counts
                or limits are not positive, the block counts do not cover
                each feature exactly once, or the crossing and truncated ID
                sets are not nested within the reported features.
        """
        if not math.isfinite(self.block_size) or self.block_size <= 0:
            raise ValueError("block_size must be a positive finite number")
        if self.feature_count < 1:
            raise ValueError("feature_count must be positive")
        if self.max_blocks_per_feature < 1:
            raise ValueError("max_blocks_per_feature must be positive")
        if len(self.feature_block_counts) != self.feature_count:
            raise ValueError("feature block counts must match feature count")
        feature_ids = tuple(item[0] for item in self.feature_block_counts)
        if len(set(feature_ids)) != len(feature_ids):
            raise ValueError("feature block counts must have unique IDs")
        if any(item[1] < 1 for item in self.feature_block_counts):
            raise ValueError("feature block counts must be positive")
        crossing = set(self.crossing_feature_ids)
        truncated = set(self.truncated_feature_ids)
        if not crossing.issubset(feature_ids):
            raise ValueError("crossing feature IDs must belong to the report")
        if not truncated.issubset(crossing):
            raise ValueError(
                "truncated feature IDs must cross block boundaries"
            )

    @property
    def status(self) -> str:
        """The leakage status, ``"potential_leakage"`` or ``"clear"``."""
        return "potential_leakage" if self.crossing_feature_ids else "clear"

    def to_dict(self) -> dict[str, Any]:
        """Returns JSON-native boundary-leakage evidence.

        Returns:
            A mapping of the status and every report field, with block counts
            as a list of ID and count mappings.
        """
        return {
            "status": self.status,
            "block_size": self.block_size,
            "feature_count": self.feature_count,
            "feature_block_counts": [
                {"feature_id": feature_id, "block_count": block_count}
                for feature_id, block_count in self.feature_block_counts
            ],
            "crossing_feature_ids": list(self.crossing_feature_ids),
            "truncated_feature_ids": list(self.truncated_feature_ids),
            "max_blocks_per_feature": self.max_blocks_per_feature,
        }


@dataclasses.dataclass(frozen=True, slots=True)
class GeospatialProfile:
    """Bounded data-understanding summary for a vector dataset.

    Attributes:
        feature_count: Number of features in the dataset.
        geometry_types: Sorted distinct GeoJSON geometry type names.
        crs: Declared CRS of the dataset, possibly empty.
        bounds: Dataset-wide (min x, min y, max x, max y) in CRS units.
        content_hash: Content hash of the dataset.
    """

    feature_count: int
    geometry_types: tuple[str, ...]
    crs: str
    bounds: tuple[float, float, float, float]
    content_hash: str

    def to_dict(self) -> dict[str, Any]:
        """Returns JSON-compatible geospatial profile data.

        Returns:
            A mapping of every profile field.
        """
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, slots=True)
class SpatialSplit:
    """Spatial-block split that keeps nearby feature blocks together.

    Attributes:
        train: Features assigned to training.
        validation: Features assigned to validation.
        test: Features assigned to testing.
        block_size: Block edge length in CRS units.
        blocks: Block IDs assigned to each partition, keyed by ``"train"``,
            ``"validation"``, and ``"test"``.
    """

    train: tuple[GeoFeature, ...]
    validation: tuple[GeoFeature, ...]
    test: tuple[GeoFeature, ...]
    block_size: float
    blocks: Mapping[str, tuple[str, ...]]

    def block_ids(self) -> dict[str, tuple[str, ...]]:
        """Returns the stable spatial blocks assigned to each partition.

        Returns:
            A new mapping from partition name to a tuple of block IDs.
        """
        return {name: tuple(values) for name, values in self.blocks.items()}


@dataclasses.dataclass(frozen=True, slots=True)
class GeospatialTemporalSplit:
    """Time-ordered geospatial features separated at a UTC cutoff.

    Attributes:
        cutoff: Timezone-aware cutoff, converted to UTC.
        train: Features observed at or before the cutoff.
        future: Features observed after the cutoff.
    """

    cutoff: datetime
    train: tuple[GeoFeature, ...]
    future: tuple[GeoFeature, ...]

    def __post_init__(self) -> None:
        """Validates the cutoff and timestamp completeness of each partition.

        Raises:
            ValueError: If the cutoff has no timezone or any feature lacks an
                observation time.
        """
        if self.cutoff.tzinfo is None or self.cutoff.utcoffset() is None:
            raise ValueError("cutoff must be timezone-aware")
        object.__setattr__(self, "cutoff", self.cutoff.astimezone(UTC))
        if any(feature.observation_time is None for feature in self.train):
            raise ValueError("train features must contain observation times")
        if any(feature.observation_time is None for feature in self.future):
            raise ValueError("future features must contain observation times")

    @property
    def has_future_leakage(self) -> bool:
        """Whether any training feature was observed after the cutoff."""
        return any(
            isinstance(feature.observation_time, datetime)
            and feature.observation_time > self.cutoff
            for feature in self.train
        )


@dataclasses.dataclass(frozen=True, slots=True)
class SpatialAutocorrelation:
    """Bounded global Moran's I evidence for a numeric spatial variable.

    Attributes:
        feature_count: Number of features analysed.
        neighbor_count: Nearest neighbors used per feature.
        edge_count: Number of undirected neighbor edges.
        statistic: Observed global Moran's I.
        permutation_count: Number of permutations used for the p-value.
        p_value: Two-sided permutation p-value, or None when no permutations
            were run.
        seed: Seed of the permutation random generator.
    """

    feature_count: int
    neighbor_count: int
    edge_count: int
    statistic: float
    permutation_count: int
    p_value: float | None
    seed: int

    def __post_init__(self) -> None:
        """Validates a finite and reproducible spatial statistic.

        Raises:
            ValueError: If there are fewer than three features, the neighbor
                count is outside ``[1, feature_count)``, there are no edges,
                the statistic is not finite, the permutation count is
                negative, or the p-value is outside ``[0, 1]``.
        """
        if self.feature_count < 3:
            raise ValueError("feature_count must be at least three")
        if not 1 <= self.neighbor_count < self.feature_count:
            raise ValueError("neighbor_count must be in [1, feature_count)")
        if self.edge_count < 1:
            raise ValueError("edge_count must contain at least one edge")
        if not math.isfinite(self.statistic):
            raise ValueError("statistic must be finite")
        if self.permutation_count < 0:
            raise ValueError("permutation_count must not be negative")
        if self.p_value is not None and not 0.0 <= self.p_value <= 1.0:
            raise ValueError("p_value must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        """Returns JSON-native autocorrelation evidence.

        Returns:
            A mapping of every field.
        """
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, slots=True)
class GeospatialFeatures:
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

    def to_numpy(self) -> np.ndarray:
        """Returns a writable copy of the numeric feature matrix.

        Returns:
            A float64 array with one row per geometry and one column per
            feature name.
        """
        return np.asarray(self.values, dtype=np.float64).copy()


@dataclasses.dataclass(frozen=True, slots=True)
class GeospatialDataset:
    """Immutable GeoJSON-like vector collection with CRS provenance.

    Attributes:
        features: The features of the dataset, with unique IDs.
        crs: Declared CRS, stripped of surrounding whitespace; may be empty.
        source: Canonical source identifier.
        content_hash: SHA-256 hex digest over the CRS and the feature
            identities, independent of feature order.
    """

    features: tuple[GeoFeature, ...]
    crs: str = ""
    source: str = "memory://"
    content_hash: str = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        """Rejects duplicate identities and derive CRS-bound content.

        Raises:
            GeospatialValidationError: If there are no features or feature
                IDs are duplicated.
            TypeError: If ``crs`` is not a string.
        """
        normalized = tuple(self.features)
        if not normalized:
            raise GeospatialValidationError(
                "geospatial dataset must contain at least one feature",
                {"code": "empty_dataset"},
            )
        if not isinstance(self.crs, str):
            raise TypeError("crs must be a string")
        canonical_crs_value = self.crs.strip()
        ids = [feature.feature_id for feature in normalized]
        if len(ids) != len(set(ids)):
            raise GeospatialValidationError(
                "geospatial dataset contains duplicate feature IDs",
                {"code": "duplicate_feature_id"},
            )
        feature_identities = sorted(
            f"{feature.feature_id}\0{feature.content_hash}"
            for feature in normalized
        )
        identity_payload = f"crs:{canonical_crs_value}\n" + "\n".join(
            feature_identities
        )
        digest = hashlib.sha256(identity_payload.encode("utf-8")).hexdigest()
        object.__setattr__(self, "features", normalized)
        object.__setattr__(self, "crs", canonical_crs_value)
        object.__setattr__(self, "source", canonical_source(self.source))
        object.__setattr__(self, "content_hash", digest)

    @classmethod
    def from_geojson(
        cls,
        document: Mapping[str, Any],
        source: str = "memory://",
        crs: str = "",
    ) -> Self:
        """Builds a dataset from a GeoJSON FeatureCollection mapping.

        Feature IDs come from each feature's ``id`` member (as text), falling
        back to its index. Feature labels and observation times are not read.
        Errors raised by the feature and dataset constructors propagate.

        Args:
            document: A GeoJSON FeatureCollection.
            source: Source identifier recorded on the dataset.
            crs: CRS identifier declared for the dataset.

        Returns:
            A dataset with one feature per GeoJSON feature.

        Raises:
            GeospatialValidationError: If the root is not a
                FeatureCollection, ``features`` is not a list, or a member is
                not a Feature with object geometry and properties.
        """
        if document.get("type") != "FeatureCollection":
            raise GeospatialValidationError(
                "GeoJSON document must be a FeatureCollection",
                {"code": "unsupported_geojson_root"},
            )
        raw_features = document.get("features")
        if not isinstance(raw_features, list):
            raise GeospatialValidationError(
                "FeatureCollection features must be a list",
                {"code": "invalid_features"},
            )
        features: list[GeoFeature] = []
        for index, raw_feature in enumerate(raw_features):
            if (
                not isinstance(raw_feature, Mapping)
                or raw_feature.get("type") != "Feature"
            ):
                raise GeospatialValidationError(
                    "FeatureCollection members must be GeoJSON Features",
                    {"code": "invalid_feature", "index": index},
                )
            feature_id = str(raw_feature.get("id", index))
            geometry = raw_feature.get("geometry")
            properties = raw_feature.get("properties", {})
            if not isinstance(geometry, Mapping) or not isinstance(
                properties, Mapping
            ):
                raise GeospatialValidationError(
                    "feature geometry and properties must be objects",
                    {"code": "invalid_feature", "index": index},
                )
            features.append(GeoFeature(feature_id, geometry, properties))
        return cls(tuple(features), crs=crs, source=source)

    def profile(self) -> GeospatialProfile:
        """Returns geometry counts, bounds, CRS, and immutable identity.

        Returns:
            A profile with the feature count, sorted geometry types, CRS,
            dataset-wide bounds, and content hash.
        """
        bounds = tuple(
            value for feature in self.features for value in feature.bounds
        )
        return GeospatialProfile(
            feature_count=len(self.features),
            geometry_types=tuple(
                sorted({feature.geometry_type for feature in self.features})
            ),
            crs=self.crs,
            bounds=(
                min(bounds[0::4]),
                min(bounds[1::4]),
                max(bounds[2::4]),
                max(bounds[3::4]),
            ),
            content_hash=self.content_hash,
        )

    def curate(self) -> ModalityCurationReport:
        """Builds geometry, CRS, and spatial-interpretation evidence.

        The profile includes label counts, temporal completeness, and global
        Moran's I when a CRS is declared, at least three features exist, and
        every label is numeric. Findings cover a missing CRS, mixed geometry
        types, mixed temporal state, and a mix of labeled and unlabeled
        features.

        Returns:
            A curation report for the geospatial modality.
        """
        profile = self.profile()
        labels = label_counts(feature.label for feature in self.features)
        profile_data = profile.to_dict()
        profile_data["label_counts"] = labels
        observation_times = tuple(
            cast(datetime, feature.observation_time)
            for feature in self.features
            if feature.observation_time is not None
        )
        if observation_times and len(observation_times) == len(self.features):
            profile_data["temporal"] = {
                "status": "complete",
                "observation_time_min": min(observation_times).isoformat(),
                "observation_time_max": max(observation_times).isoformat(),
            }
        elif observation_times:
            profile_data["temporal"] = {
                "status": "mixed",
                "timed_feature_count": len(observation_times),
                "untimed_feature_count": len(self.features)
                - len(observation_times),
            }
        else:
            profile_data["temporal"] = {"status": "absent"}
        quality_findings = []
        risk_findings = []
        if observation_times and len(observation_times) != len(self.features):
            quality_findings.append(
                build_finding(
                    "quality.geospatial-mixed-temporal-state",
                    data_contracts.FindingSeverity.WARNING,
                    {"source": self.source},
                    {"temporal": profile_data["temporal"]},
                    "Supply observation times for every feature or explicitly "
                    "use a non-temporal geospatial capability.",
                    False,
                )
            )
        numeric_labels = [feature.label for feature in self.features]
        if not self.crs.strip():
            profile_data["spatial_autocorrelation"] = {
                "status": "blocked_missing_crs"
            }
        elif len(self.features) < 3:
            profile_data["spatial_autocorrelation"] = {
                "status": "not_estimable",
                "reason": "at least three features are required",
            }
        elif any(
            isinstance(value, bool) or not isinstance(value, Real)
            for value in numeric_labels
        ):
            profile_data["spatial_autocorrelation"] = {
                "status": "not_applicable",
                "reason": "numeric values are required for every feature",
            }
        else:
            profile_data["spatial_autocorrelation"] = (
                self.spatial_autocorrelation().to_dict()
            )
        if not self.crs.strip():
            quality_findings.append(
                build_finding(
                    "quality.geospatial-missing-crs",
                    data_contracts.FindingSeverity.ERROR,
                    {"source": self.source},
                    {"feature_count": len(self.features)},
                    "Declare the source CRS before spatial transforms or "
                    "modelling.",
                    True,
                )
            )
        if len(profile.geometry_types) > 1:
            quality_findings.append(
                build_finding(
                    "quality.geospatial-mixed-geometry",
                    data_contracts.FindingSeverity.WARNING,
                    {"source": self.source},
                    {"geometry_types": list(profile.geometry_types)},
                    "Choose a geometry normalization or task-specific "
                    "handling policy.",
                    False,
                )
            )
        if "unlabeled" in labels and len(labels) > 1:
            risk_findings.append(
                build_finding(
                    "risk.geospatial-mixed-label-state",
                    data_contracts.FindingSeverity.WARNING,
                    {"source": self.source},
                    {"label_counts": labels},
                    "Declare whether unlabeled features are valid for the "
                    "intended spatial task before training.",
                    False,
                )
            )
        return ModalityCurationReport(
            modality="geospatial",
            dataset_hash=self.content_hash,
            sample_count=len(self.features),
            profile=profile_data,
            quality_findings=tuple(quality_findings),
            risk_findings=tuple(risk_findings),
        )

    def spatial_neighbors(
        self, neighbor_count: int = 4
    ) -> tuple[tuple[int, int], ...]:
        """Builds deterministic symmetric k-nearest-neighbor edges.

        Distances are measured between envelope centroids in the declared
        source CRS. The method exposes the exact topology used by the
        autocorrelation statistic so a reviewer can reproduce the evidence.

        Args:
            neighbor_count: Number of nearest neighbors per feature, in
                ``[1, feature_count)``.

        Returns:
            Sorted, de-duplicated undirected edges as (lower, higher) feature
            index pairs. Ties in distance break by feature index.

        Raises:
            GeospatialValidationError: If the dataset declares no CRS.
            ValueError: If ``neighbor_count`` is outside
                ``[1, feature_count)``.
        """
        if not self.crs.strip():
            raise GeospatialValidationError(
                "spatial neighbors require a declared CRS",
                {"code": "missing_crs"},
            )
        if not 1 <= neighbor_count < len(self.features):
            raise ValueError("neighbor_count must be in [1, feature_count)")
        centroids = np.asarray(
            [feature.centroid for feature in self.features],
            dtype=np.float64,
        )
        edges: set[tuple[int, int]] = set()
        for first in range(len(centroids)):
            distances = [
                (
                    float(np.sum((centroids[first] - centroids[second]) ** 2)),
                    second,
                )
                for second in range(len(centroids))
                if second != first
            ]
            nearest = sorted(distances)[:neighbor_count]
            for distance_and_index in nearest:
                second = distance_and_index[1]
                edges.add(
                    (first, second) if first < second else (second, first)
                )
        return tuple(sorted(edges))

    def spatial_join(
        self,
        other: "GeospatialDataset",
        predicate: SpatialJoinPredicate,
        predicate_name: str,
        comparison_limit: int = 100_000,
        match_limit: int = 100_000,
    ) -> SpatialJoinReport:
        """Joins two same-CRS datasets through a bounded caller predicate.

        The core package does not choose a geometry engine, infer CRS
        transformations, or silently compare coordinates in different units.
        A caller supplies a versioned predicate such as an ``intersects/v1``
        implementation. This method owns the identity checks, bounded
        execution, deterministic evidence, and fail-closed limits.

        Args:
            other: Right-hand dataset with the same declared CRS.
            predicate: Caller-owned predicate receiving one left and one right
                feature and returning a Python boolean.
            predicate_name: Stable name and version of the predicate.
            comparison_limit: Maximum Cartesian comparisons allowed.
            match_limit: Maximum matches retained in the evidence report.

        Returns:
            Dataset identities, counts, and matched feature IDs.

        Raises:
            TypeError: If ``other`` is not a dataset or ``predicate`` is not
                callable.
            ValueError: If ``predicate_name`` is empty or a limit is not
                positive.
            GeospatialValidationError: If either dataset lacks a CRS, the
                CRS values differ, the number of comparisons exceeds
                ``comparison_limit``, the predicate raises or returns a
                non-boolean, or the matches exceed ``match_limit``.
        """
        if not isinstance(other, GeospatialDataset):
            raise TypeError("other must be a GeospatialDataset")
        if not callable(predicate):
            raise TypeError("predicate must be callable")
        if not predicate_name.strip():
            raise ValueError("predicate_name must not be empty")
        if comparison_limit < 1 or match_limit < 1:
            raise ValueError("join limits must be positive")
        if not self.crs.strip() or not other.crs.strip():
            raise GeospatialValidationError(
                "spatial joins require a declared CRS on both datasets",
                {"code": "missing_crs"},
            )
        if self.crs.strip() != other.crs.strip():
            raise GeospatialValidationError(
                "spatial joins require matching CRS declarations; reproject "
                "through an explicit caller-owned policy first",
                {
                    "code": "crs_mismatch",
                    "left_crs": self.crs,
                    "right_crs": other.crs,
                },
            )
        total_comparisons = len(self.features) * len(other.features)
        if total_comparisons > comparison_limit:
            raise GeospatialValidationError(
                "spatial join exceeds the comparison limit; provide a "
                "bounded indexed policy",
                {
                    "code": "comparison_limit_exceeded",
                    "comparison_count": total_comparisons,
                    "comparison_limit": comparison_limit,
                },
            )
        matches: list[SpatialJoinMatch] = []
        for left in self.features:
            for right in other.features:
                try:
                    result = predicate(left, right)
                except Exception as error:
                    raise GeospatialValidationError(
                        "caller-owned spatial join predicate failed",
                        {
                            "code": "predicate_failure",
                            "predicate_name": predicate_name,
                            "exception": type(error).__name__,
                        },
                    ) from error
                if not isinstance(result, (bool, np.bool_)):
                    raise GeospatialValidationError(
                        "spatial join predicate must return a boolean",
                        {
                            "code": "invalid_predicate_result",
                            "predicate_name": predicate_name,
                            "result_type": type(result).__name__,
                        },
                    )
                if bool(result):
                    matches.append(
                        SpatialJoinMatch(left.feature_id, right.feature_id)
                    )
                    if len(matches) > match_limit:
                        raise GeospatialValidationError(
                            "spatial join exceeds the match limit; provide a "
                            "bounded output policy",
                            {
                                "code": "match_limit_exceeded",
                                "match_limit": match_limit,
                            },
                        )
        matched_left = {match.left_feature_id for match in matches}
        matched_right = {match.right_feature_id for match in matches}
        return SpatialJoinReport(
            left_dataset_hash=self.content_hash,
            right_dataset_hash=other.content_hash,
            crs=self.crs.strip(),
            predicate_name=predicate_name.strip(),
            left_feature_count=len(self.features),
            right_feature_count=len(other.features),
            comparison_count=total_comparisons,
            match_count=len(matches),
            unmatched_left_count=len(
                {feature.feature_id for feature in self.features} - matched_left
            ),
            unmatched_right_count=len(
                {feature.feature_id for feature in other.features}
                - matched_right
            ),
            comparison_limit=comparison_limit,
            match_limit=match_limit,
            matches=tuple(matches),
        )

    def reproject(
        self,
        target_crs: str,
        transformer: CoordinateTransformer,
        transformer_name: str,
    ) -> "GeospatialReprojection":
        """Transforms coordinates through an explicit caller-owned policy.

        The core package does not select a CRS database or geometry engine.
        The caller supplies a versioned transformer that changes geometry only;
        feature IDs, properties, labels, and observation times must remain
        stable. The returned report binds the source and target snapshots so a
        transformed dataset cannot be mistaken for the original source.

        Args:
            target_crs: Non-empty target CRS identifier.
            transformer: Caller-owned coordinate transformation.
            transformer_name: Stable name and version of the transformer.

        Returns:
            A transformed dataset and its source/target evidence report.

        Raises:
            GeospatialValidationError: If the source CRS is missing, the
                transformer raises, returns a non-feature, or changes a
                feature's ID, properties, label, or observation time.
            ValueError: If ``target_crs`` is empty or equals the source CRS,
                or ``transformer_name`` is empty.
            TypeError: If ``transformer`` is not callable.
        """
        if not self.crs.strip():
            raise GeospatialValidationError(
                "reprojection requires a declared source CRS",
                {"code": "missing_crs"},
            )
        if not isinstance(target_crs, str) or not target_crs.strip():
            raise ValueError("target_crs must be a non-empty string")
        canonical_target_crs = target_crs.strip()
        if canonical_target_crs == self.crs:
            raise ValueError("target_crs must differ from the source CRS")
        if not callable(transformer):
            raise TypeError("transformer must be callable")
        if not transformer_name.strip():
            raise ValueError("transformer_name must not be empty")
        transformed_features: list[GeoFeature] = []
        for feature in self.features:
            try:
                transformed = transformer(
                    feature, self.crs, canonical_target_crs
                )
            except Exception as error:
                raise GeospatialValidationError(
                    "caller-owned coordinate transformer failed",
                    {
                        "code": "transformer_failure",
                        "transformer_name": transformer_name,
                        "exception": type(error).__name__,
                    },
                ) from error
            if not isinstance(transformed, GeoFeature):
                raise GeospatialValidationError(
                    "coordinate transformer must return a GeoFeature",
                    {
                        "code": "invalid_transformer_result",
                        "transformer_name": transformer_name,
                    },
                )
            if (
                transformed.feature_id != feature.feature_id
                or dict(transformed.properties) != dict(feature.properties)
                or transformed.label != feature.label
                or transformed.observation_time != feature.observation_time
            ):
                raise GeospatialValidationError(
                    "coordinate transformation changed feature metadata",
                    {
                        "code": "transformer_metadata_changed",
                        "feature_id": feature.feature_id,
                    },
                )
            transformed_features.append(transformed)
        dataset = GeospatialDataset(
            tuple(transformed_features),
            crs=canonical_target_crs,
            source=self.source,
        )
        report = SpatialReprojectionReport(
            source_dataset_hash=self.content_hash,
            target_dataset_hash=dataset.content_hash,
            source_crs=self.crs,
            target_crs=canonical_target_crs,
            transformer_name=transformer_name.strip(),
            feature_count=len(transformed_features),
        )
        return GeospatialReprojection(dataset=dataset, report=report)

    def spatial_autocorrelation(
        self,
        values: Sequence[Real] | None = None,
        neighbor_count: int = 4,
        permutations: int = 99,
        seed: int = 0,
    ) -> SpatialAutocorrelation:
        """Measures global Moran's I with optional deterministic permutations.

        When ``values`` is omitted, numeric feature labels are used. This
        method reports association in the observed spatial arrangement; it
        does not establish causality or replace a domain-specific spatial
        model.

        Args:
            values: Numeric values aligned with the feature order. Defaults
                to the feature labels.
            neighbor_count: Requested nearest neighbors per feature, reduced
                to one less than the feature count when larger.
            permutations: Number of random permutations for the p-value, in
                ``[0, 10000]``. Zero skips the p-value.
            seed: Seed of the permutation random generator.

        Returns:
            The statistic, neighbor topology size, and a two-sided
            permutation p-value, which is None when no permutations are run.

        Raises:
            ValueError: If ``permutations`` is outside ``[0, 10000]`` or
                ``neighbor_count`` is below one.
            GeospatialValidationError: If there are fewer than three
                features, the value count differs from the feature count,
                a value is not numeric, the dataset has no CRS, or the
                statistic is not estimable.
        """
        if permutations < 0 or permutations > 10_000:
            raise ValueError("permutations must be in [0, 10000]")
        if len(self.features) < 3:
            raise GeospatialValidationError(
                "spatial autocorrelation requires at least three features",
                {"code": "insufficient_features"},
            )
        if neighbor_count < 1:
            raise ValueError("neighbor_count must be positive")
        effective_neighbor_count = min(neighbor_count, len(self.features) - 1)
        if values is None:
            raw_values: list[Any] = [feature.label for feature in self.features]
        else:
            raw_values = list(values)
        if len(raw_values) != len(self.features):
            raise GeospatialValidationError(
                "spatial values must match feature count",
                {
                    "code": "value_count_mismatch",
                    "feature_count": len(self.features),
                    "value_count": len(raw_values),
                },
            )
        if any(
            isinstance(value, bool) or not isinstance(value, Real)
            for value in raw_values
        ):
            raise GeospatialValidationError(
                "spatial autocorrelation requires numeric values",
                {"code": "numeric_values_required"},
            )
        numeric = np.asarray(raw_values, dtype=np.float64)
        edges = self.spatial_neighbors(effective_neighbor_count)
        try:
            statistic = moran_statistic(numeric, edges)
        except ValueError as error:
            raise GeospatialValidationError(
                str(error), {"code": "statistic_not_estimable"}
            ) from error

        p_value: float | None = None
        if permutations:
            randomizer = np.random.default_rng(seed)
            exceedances = 0
            permutation_index = 0
            while permutation_index < permutations:
                shuffled = randomizer.permutation(numeric)
                permuted_statistic = moran_statistic(shuffled, edges)
                if abs(permuted_statistic) >= abs(statistic):
                    exceedances += 1
                permutation_index += 1
            p_value = (exceedances + 1) / (permutations + 1)
        return SpatialAutocorrelation(
            feature_count=len(self.features),
            neighbor_count=effective_neighbor_count,
            edge_count=len(edges),
            statistic=statistic,
            permutation_count=permutations,
            p_value=p_value,
            seed=seed,
        )

    def prepare_features(self) -> GeospatialFeatures:
        """Prepares bounded geometry features without changing CRS units.

        The output is a geometry envelope and centroid representation. It does
        not perform reprojection, spatial joins, rasterization, or distance
        calculations; those operations require an explicit, versioned policy.

        Returns:
            One row per feature with its bounds, width, height, centroid, and
            count of coordinate values, in the dataset CRS units.

        Raises:
            GeospatialValidationError: If the dataset has no CRS or a
                prepared value is not finite.
        """
        if not self.crs.strip():
            raise GeospatialValidationError(
                "geospatial feature preparation requires a declared CRS",
                {"code": "missing_crs"},
            )
        feature_names = (
            "min_x",
            "min_y",
            "max_x",
            "max_y",
            "width",
            "height",
            "centroid_x",
            "centroid_y",
            "coordinate_value_count",
        )
        rows = []
        for feature in self.features:
            minimum_x, minimum_y, maximum_x, maximum_y = feature.bounds
            coordinates = geometry_coordinates(feature.geometry)
            rows.append(
                (
                    minimum_x,
                    minimum_y,
                    maximum_x,
                    maximum_y,
                    maximum_x - minimum_x,
                    maximum_y - minimum_y,
                    feature.centroid[0],
                    feature.centroid[1],
                    float(len(coordinates)),
                )
            )
        values = np.asarray(rows, dtype=np.float64)
        if not np.isfinite(values).all():
            raise GeospatialValidationError(
                "geospatial features must be finite",
                {"code": "non_finite_features"},
            )
        return GeospatialFeatures(
            feature_ids=tuple(feature.feature_id for feature in self.features),
            feature_names=feature_names,
            values=tuple(
                tuple(float(value) for value in row) for row in values
            ),
            crs=self.crs,
        )

    def spatial_block_envelopes(
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
            block, max y block) indexes, counted from the dataset's minimum
            x and y.

        Raises:
            GeospatialValidationError: If the dataset has no CRS.
            ValueError: If ``block_size`` is not positive and finite.
        """
        if not self.crs.strip():
            raise GeospatialValidationError(
                "spatial block envelopes require a declared CRS",
                {"code": "missing_crs"},
            )
        if not math.isfinite(block_size) or block_size <= 0:
            raise ValueError("block_size must be a positive finite number")
        dataset_bounds = self.profile().bounds
        origin_x, origin_y = dataset_bounds[:2]
        envelopes: dict[str, tuple[int, int, int, int]] = {}
        for feature in self.features:
            minimum_x, minimum_y, maximum_x, maximum_y = feature.bounds
            envelopes[feature.feature_id] = (
                math.floor((minimum_x - origin_x) / block_size),
                math.floor((minimum_y - origin_y) / block_size),
                math.floor((maximum_x - origin_x) / block_size),
                math.floor((maximum_y - origin_y) / block_size),
            )
        return envelopes

    def boundary_leakage(
        self,
        block_size: float,
        max_blocks_per_feature: int = 10_000,
    ) -> SpatialBoundaryLeakage:
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
            ``GeospatialValidationError``.

        Raises:
            ValueError: If ``max_blocks_per_feature`` is below one or
                ``block_size`` is not positive and finite.
        """
        if max_blocks_per_feature < 1:
            raise ValueError("max_blocks_per_feature must be positive")
        envelopes = self.spatial_block_envelopes(block_size)
        block_counts: list[tuple[str, int]] = []
        crossing_feature_ids: list[str] = []
        truncated_feature_ids: list[str] = []
        for feature in self.features:
            minimum_x, minimum_y, maximum_x, maximum_y = envelopes[
                feature.feature_id
            ]
            block_count = (maximum_x - minimum_x + 1) * (
                maximum_y - minimum_y + 1
            )
            block_counts.append((feature.feature_id, block_count))
            if block_count > 1:
                crossing_feature_ids.append(feature.feature_id)
            if block_count > max_blocks_per_feature:
                truncated_feature_ids.append(feature.feature_id)
        return SpatialBoundaryLeakage(
            block_size=block_size,
            feature_count=len(self.features),
            feature_block_counts=tuple(block_counts),
            crossing_feature_ids=tuple(crossing_feature_ids),
            truncated_feature_ids=tuple(truncated_feature_ids),
            max_blocks_per_feature=max_blocks_per_feature,
        )

    def split_spatial(
        self,
        block_size: float,
        validation_fraction: float = 0.2,
        test_fraction: float = 0.2,
        seed: int = 0,
    ) -> SpatialSplit:
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
            raise ValueError("block_size must be a positive finite number")
        if not 0 <= validation_fraction < 1:
            raise ValueError("validation_fraction must be in [0, 1)")
        if not 0 <= test_fraction < 1:
            raise ValueError("test_fraction must be in [0, 1)")
        if validation_fraction + test_fraction >= 1:
            raise ValueError("validation_fraction + test_fraction must be < 1")
        dataset_bounds = self.profile().bounds
        origin_x = dataset_bounds[0]
        origin_y = dataset_bounds[1]

        def block_for(feature: GeoFeature) -> str:
            minimum_x, minimum_y, maximum_x, maximum_y = feature.bounds
            centroid_x = (minimum_x + maximum_x) / 2
            centroid_y = (minimum_y + maximum_y) / 2
            block_x = math.floor((centroid_x - origin_x) / block_size)
            block_y = math.floor((centroid_y - origin_y) / block_size)
            return f"{block_x}:{block_y}"

        grouped: dict[str, list[GeoFeature]] = {}
        for feature in self.features:
            grouped.setdefault(block_for(feature), []).append(feature)
        blocks = sorted(grouped)
        randomizer = random.Random(seed)  # nosec B311 - seeded data split
        randomizer.shuffle(blocks)
        test_count = int(len(blocks) * test_fraction)
        validation_count = int(len(blocks) * validation_fraction)
        if test_fraction and test_count == 0 and len(blocks) >= 3:
            test_count = 1
        if validation_fraction and validation_count == 0 and len(blocks) >= 3:
            validation_count = 1
        while test_count + validation_count >= len(blocks):
            if validation_count > 0:
                validation_count -= 1
            elif test_count > 0:
                test_count -= 1
            else:
                break
        test_blocks = set(blocks[:test_count])
        validation_blocks = set(
            blocks[test_count : test_count + validation_count]
        )

        def features_for(selected: set[str]) -> tuple[GeoFeature, ...]:
            return tuple(
                feature
                for feature in self.features
                if block_for(feature) in selected
            )

        return SpatialSplit(
            train=features_for(set(blocks) - test_blocks - validation_blocks),
            validation=features_for(validation_blocks),
            test=features_for(test_blocks),
            block_size=block_size,
            blocks={
                "train": tuple(
                    sorted(set(blocks) - test_blocks - validation_blocks)
                ),
                "validation": tuple(sorted(validation_blocks)),
                "test": tuple(sorted(test_blocks)),
            },
        )

    def split_temporal(self, cutoff: datetime | str) -> GeospatialTemporalSplit:
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
            GeospatialTemporalError: If the cutoff is invalid or lacks a
                timezone, or any feature has no observation time.
        """
        cutoff_time = canonical_observation_time(cutoff)
        if any(feature.observation_time is None for feature in self.features):
            raise GeospatialTemporalError(
                "temporal splitting requires an observation time on every "
                "feature",
                {"code": "incomplete_observation_times"},
            )
        train = tuple(
            feature
            for feature in self.features
            if isinstance(feature.observation_time, datetime)
            and feature.observation_time <= cutoff_time
        )
        future = tuple(
            feature
            for feature in self.features
            if isinstance(feature.observation_time, datetime)
            and feature.observation_time > cutoff_time
        )
        return GeospatialTemporalSplit(
            cutoff=cutoff_time,
            train=train,
            future=future,
        )


@dataclasses.dataclass(frozen=True, slots=True)
class GeospatialReprojection:
    """A transformed dataset paired with its source-bound evidence.

    Attributes:
        dataset: The transformed dataset.
        report: Evidence binding the source and transformed datasets.
    """

    dataset: GeospatialDataset
    report: SpatialReprojectionReport

    def __post_init__(self) -> None:
        """Ensures the report describes the returned transformed dataset.

        Raises:
            ValueError: If the report's target hash or CRS differs from the
                dataset's.
        """
        if self.report.target_dataset_hash != self.dataset.content_hash:
            raise ValueError(
                "reprojection report must match the transformed dataset"
            )
        if self.report.target_crs != self.dataset.crs:
            raise ValueError("reprojection report CRS must match the dataset")

    def to_dict(self) -> dict[str, Any]:
        """Returns the transformed identity and its reprojection evidence.

        Returns:
            A mapping with the dataset hash, CRS, and the report as a
            mapping.
        """
        return {
            "dataset_hash": self.dataset.content_hash,
            "crs": self.dataset.crs,
            "report": self.report.to_dict(),
        }


@dataclasses.dataclass(frozen=True, slots=True)
class RasterSample:
    """Raster identity and metadata without assuming a raster codec.

    Attributes:
        source: Non-empty source identifier.
        encoded_bytes: Encoded raster payload, which must not be empty.
        file_hash: SHA-256 hex digest of ``encoded_bytes``.
        width: Raster width in pixels.
        height: Raster height in pixels.
        bands: Number of bands.
        dtype: Non-empty pixel data type name.
        crs: Non-empty CRS identifier.
        transform: Empty, a six-value affine transform, or a nine-value
            row-major homogeneous matrix.
        nodata_value: Declared nodata sentinel, stored as a float, or None.
        nodata_fraction: Measured fraction of nodata pixels in ``[0, 1]``,
            or None.
        byte_size: Length of ``encoded_bytes``.
    """

    source: str
    encoded_bytes: bytes = dataclasses.field(repr=False)
    file_hash: str
    width: int
    height: int
    bands: int
    dtype: str
    crs: str
    transform: tuple[float, ...]
    nodata_value: float | int | None = None
    nodata_fraction: float | None = None
    byte_size: int = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        """Validates raster shape, CRS provenance, and source identity.

        Raises:
            ValueError: If the source, CRS, or dtype is empty, dimensions or
                bands are not positive, the transform length is not 0, 6, or
                9, nodata values are non-numeric, non-finite, or out of
                range, or ``file_hash`` does not match the payload.
            GeospatialValidationError: If the payload is empty.
        """
        raw = bytes(self.encoded_bytes)
        object.__setattr__(self, "encoded_bytes", raw)
        if (
            not self.source.strip()
            or not self.crs.strip()
            or not self.dtype.strip()
        ):
            raise ValueError("source, crs, and dtype must not be empty")
        if min(self.width, self.height, self.bands) < 1:
            raise ValueError("raster dimensions and bands must be positive")
        if len(self.transform) not in {0, 6, 9}:
            raise ValueError("transform must be empty or an affine transform")
        if self.nodata_value is not None:
            if isinstance(self.nodata_value, bool) or not isinstance(
                self.nodata_value, Real
            ):
                raise ValueError("nodata_value must be numeric")
            if not math.isfinite(float(self.nodata_value)):
                raise ValueError("nodata_value must be finite")
            object.__setattr__(self, "nodata_value", float(self.nodata_value))
        if (
            self.nodata_fraction is not None
            and not 0.0 <= float(self.nodata_fraction) <= 1.0
        ):
            raise ValueError("nodata_fraction must be in [0, 1]")
        if self.nodata_fraction is not None and not math.isfinite(
            float(self.nodata_fraction)
        ):
            raise ValueError("nodata_fraction must be finite")
        if not raw:
            raise GeospatialValidationError(
                "raster payload is empty", {"code": "empty_payload"}
            )
        if hashlib.sha256(raw).hexdigest() != self.file_hash:
            raise ValueError("file_hash does not match encoded_bytes")
        object.__setattr__(self, "byte_size", len(raw))

    @property
    def nodata_status(self) -> str:
        """``"measured"``, ``"declared"``, or ``"unknown"`` nodata state."""
        if self.nodata_fraction is not None:
            return "measured"
        if self.nodata_value is not None:
            return "declared"
        return "unknown"

    def pixel_window_bounds(
        self,
        row_start: int,
        row_end: int,
        column_start: int,
        column_end: int,
    ) -> tuple[float, float, float, float] | None:
        """Returns the CRS bounds of a validated pixel window.

        Pixel boundaries are transformed rather than pixel centers so the
        returned envelope describes the full window. Empty transforms have no
        spatial reference and therefore return ``None``. Six-value transforms
        use the conventional affine order ``a, b, c, d, e, f``; nine-value
        transforms use a row-major homogeneous matrix.

        Args:
            row_start: Inclusive pixel-row boundary.
            row_end: Exclusive pixel-row boundary.
            column_start: Inclusive pixel-column boundary.
            column_end: Exclusive pixel-column boundary.

        Returns:
            Minimum x, minimum y, maximum x, and maximum y in the source CRS,
            or ``None`` when the sample has no transform. A homogeneous
            transform that is singular at a window boundary raises
            ``GeospatialValidationError``.

        Raises:
            ValueError: If the window is outside the raster or empty.
        """
        if not (
            0 <= row_start < row_end <= self.height
            and 0 <= column_start < column_end <= self.width
        ):
            raise ValueError("pixel window must be non-empty and in bounds")
        if not self.transform:
            return None
        corners = (
            (column_start, row_start),
            (column_start, row_end),
            (column_end, row_start),
            (column_end, row_end),
        )
        if len(self.transform) == 6:
            first, second, third, fourth, fifth, sixth = self.transform

            def project(column: int, row: int) -> tuple[float, float]:
                return (
                    first * column + second * row + third,
                    fourth * column + fifth * row + sixth,
                )

        else:
            (
                first,
                second,
                third,
                fourth,
                fifth,
                sixth,
                seventh,
                eighth,
                ninth,
            ) = self.transform

            def project(column: int, row: int) -> tuple[float, float]:
                denominator = seventh * column + eighth * row + ninth
                if denominator == 0:
                    raise GeospatialValidationError(
                        "raster transform is singular at a pixel boundary",
                        {"code": "singular_transform"},
                    )
                return (
                    (first * column + second * row + third) / denominator,
                    (fourth * column + fifth * row + sixth) / denominator,
                )

        projected = tuple(project(column, row) for column, row in corners)
        x_values = tuple(point[0] for point in projected)
        y_values = tuple(point[1] for point in projected)
        return (min(x_values), min(y_values), max(x_values), max(y_values))

    def decode(
        self,
        decoder: RasterDecoder,
        max_pixels: int = 4_000_000,
    ) -> np.ndarray:
        """Decodes bounded numeric pixels through a caller-owned decoder.

        The core package does not select a raster codec. The caller owns the
        decoder and must return a grayscale ``(height, width)`` array or a
        channel-last ``(height, width, bands)`` array. The decoded values are
        copied into a read-only float matrix after shape, size, type, and
        finiteness checks.

        Args:
            decoder: Caller-owned function that decodes the encoded payload.
            max_pixels: Maximum decoded array elements allowed per sample.

        Returns:
            A read-only, finite, channel-last-compatible pixel matrix.

        Raises:
            TypeError: If the decoder is not callable.
            ValueError: If ``max_pixels`` is not positive.
            GeospatialValidationError: If the decoder raises, the decoded
                array is not 2-D or 3-D, is empty or larger than
                ``max_pixels``, does not match the declared dimensions, is
                not real numeric, or contains non-finite values.
        """
        if not callable(decoder):
            raise TypeError("decoder must be callable")
        if max_pixels < 1:
            raise ValueError("max_pixels must be positive")
        try:
            decoded = np.asarray(decoder(self.encoded_bytes))
        except Exception as error:
            raise GeospatialValidationError(
                "raster decoding failed",
                {"code": "decode_failed", "source": self.source},
            ) from error
        if decoded.ndim not in {2, 3}:
            raise GeospatialValidationError(
                "decoded raster pixels must be grayscale or channel-last",
                {
                    "code": "invalid_pixel_shape",
                    "source": self.source,
                    "shape": list(decoded.shape),
                },
            )
        if decoded.size < 1 or decoded.size > max_pixels:
            raise GeospatialValidationError(
                "decoded raster exceeds the configured pixel limit",
                {
                    "code": "pixel_limit",
                    "source": self.source,
                    "pixel_count": int(decoded.size),
                    "max_pixels": max_pixels,
                },
            )
        expected_shape = (self.height, self.width)
        if self.bands == 1 and decoded.shape == expected_shape:
            pass
        elif decoded.shape != (self.height, self.width, self.bands):
            raise GeospatialValidationError(
                "decoded raster dimensions do not match metadata",
                {
                    "code": "pixel_dimension_mismatch",
                    "source": self.source,
                    "expected": [self.height, self.width, self.bands],
                    "observed": list(decoded.shape),
                },
            )
        if not np.issubdtype(decoded.dtype, np.number) or np.issubdtype(
            decoded.dtype, np.complexfloating
        ):
            raise GeospatialValidationError(
                "decoded raster pixels must be real numeric values",
                {"code": "invalid_pixel_type", "source": self.source},
            )
        values = np.asarray(decoded, dtype=np.float64).copy()
        if not np.isfinite(values).all():
            raise GeospatialValidationError(
                "decoded raster pixels must be finite",
                {"code": "non_finite_pixels", "source": self.source},
            )
        values.setflags(write=False)
        return values

    def measured_nodata_fraction(
        self,
        decoder: RasterDecoder,
        max_pixels: int = 4_000_000,
    ) -> float | None:
        """Measures the declared nodata sentinel in decoded pixels.

        Args:
            decoder: Caller-owned function that decodes the encoded payload.
            max_pixels: Maximum decoded array elements allowed per sample.

        Returns:
            The fraction of decoded values equal to the declared nodata
            value, or None when no nodata value is declared. Decoding errors
            are those of ``decode``.
        """
        if self.nodata_value is None:
            return None
        pixels = self.decode(decoder, max_pixels=max_pixels)
        return float(np.equal(pixels, self.nodata_value).mean())


__all__ = [
    "GEOMETRY_TYPES",
    "GeoFeature",
    "GeospatialDataset",
    "GeospatialFeatures",
    "GeospatialProfile",
    "GeospatialTemporalError",
    "GeospatialTemporalSplit",
    "GeospatialValidationError",
    "RasterDecoder",
    "RasterSample",
    "SpatialAutocorrelation",
    "SpatialSplit",
    "canonical_observation_time",
    "canonical_source",
    "geometry_coordinates",
    "moran_statistic",
    "validate_coordinates",
]
