"""Geometry primitives: validation, identity and the GeoJSON feature."""

from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
from datetime import datetime
from datetime import UTC
import hashlib
import json
import math
import pathlib
from types import MappingProxyType
from typing import Any

import numpy

from scmle.contracts import errors

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

SHA256_HEX_LENGTH = 64
MIN_COORDINATE_VALUES = 2
MIN_OBSERVATIONS = 3


class GeometryError(errors.DataError):
    """A vector or raster sample violates a geospatial invariant."""

    category = "geospatial_validation"


def verify(field_name: str, value: str) -> None:
    """Checks that a value is a lowercase SHA-256 hex digest.

    Args:
        field_name: Field name, for the message.
        value: The text to check.

    Raises:
        ValueError: If the value is not 64 lowercase hexadecimal characters.
    """
    if len(value) != SHA256_HEX_LENGTH or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise errors.InputError(f"{field_name} must be a SHA-256 hex digest")


class TemporalError(GeometryError):
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
        TemporalError: If the text is not a valid ISO 8601
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
            raise TemporalError(
                f"invalid observation time: {value!r}",
                {"code": "invalid_observation_time"},
            ) from error
    elif isinstance(value, datetime):
        parsed = value
    else:
        raise TemporalError(
            "observation time must be datetime or text",
            {"code": "invalid_observation_time"},
        )
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise TemporalError(
            "observation time must be timezone-aware",
            {"code": "timezone_required"},
        )
    return parsed.astimezone(UTC)


def flatten(value: Any) -> tuple[float, ...]:
    """Validates a nested coordinate array and return its numeric leaves.

    Args:
        value: A list or tuple that is either a position (at least two finite
            non-boolean numbers) or a nested list or tuple of such arrays.

    Returns:
        Every coordinate number, as floats, flattened in document order.

    Raises:
        GeometryError: If an array is empty, a position has fewer
            than two numbers or a non-finite number, or a value is not a
            list, tuple, or number.
    """
    if isinstance(value, (list, tuple)):
        if not value:
            raise GeometryError(
                "coordinate arrays must not be empty",
                {"code": "empty_coordinates"},
            )
        if all(
            isinstance(item, (int, float)) and not isinstance(item, bool)
            for item in value
        ):
            numbers = tuple(float(item) for item in value)
            if len(numbers) < MIN_COORDINATE_VALUES or not all(
                math.isfinite(item) for item in numbers
            ):
                raise GeometryError(
                    "coordinates must contain finite x/y values",
                    {"code": "invalid_coordinates"},
                )
            return numbers
        flattened: list[float] = []
        for item in value:
            flattened.extend(flatten(item))
        return tuple(flattened)
    raise GeometryError(
        "coordinates must be nested numeric arrays",
        {"code": "invalid_coordinates"},
    )


def unpack(geometry: Mapping[str, Any]) -> tuple[float, ...]:
    """Validates a GeoJSON geometry and return all coordinate leaves.

    Args:
        geometry: A GeoJSON geometry object. Geometry collections are
            validated recursively.

    Returns:
        Every coordinate number, as floats, flattened in document order
        (x, y, then any further dimensions of each position).

    Raises:
        GeometryError: If the geometry type is not a supported
            GeoJSON type, a geometry collection has no geometries or has a
            member that is not an object, or the coordinates are invalid.
    """
    geometry_type = geometry.get("type")
    if geometry_type not in GEOMETRY_TYPES:
        raise GeometryError(
            "geometry type is not supported",
            {"code": "unsupported_geometry", "geometry_type": geometry_type},
        )
    if geometry_type == "GeometryCollection":
        geometries = geometry.get("geometries")
        if not isinstance(geometries, list) or not geometries:
            raise GeometryError(
                "geometry collection must contain geometries",
                {"code": "empty_geometry_collection"},
            )
        values: list[float] = []
        for nested in geometries:
            if not isinstance(nested, Mapping):
                raise GeometryError(
                    "geometry collection members must be objects",
                    {"code": "invalid_geometry_collection"},
                )
            values.extend(unpack(nested))
        return tuple(values)
    return flatten(geometry.get("coordinates"))


def autocorrelate(
    values: numpy.ndarray, edges: Sequence[tuple[int, int]]
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
    numeric = numpy.asarray(values, dtype=numpy.float64)
    if numeric.ndim != 1 or len(numeric) < MIN_OBSERVATIONS:
        raise errors.InputError(
            "values must contain at least three observations"
        )
    if not numpy.isfinite(numeric).all():
        raise errors.InputError("values must be finite")
    centered = numeric - numeric.mean()
    denominator = float(numpy.dot(centered, centered))
    if denominator == 0.0:
        raise errors.InputError(
            "values must contain more than one unique value"
        )
    weight_total = 2 * len(edges)
    if weight_total == 0:
        raise errors.InputError("edges must contain at least one neighbor pair")
    pairs = numpy.asarray(edges, dtype=numpy.int64).reshape(-1, 2)
    numerator = 2 * float(
        numpy.dot(centered[pairs[:, 0]], centered[pairs[:, 1]])
    )
    return float(len(numeric) * numerator / (weight_total * denominator))


def substitute(
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
        geometries, raises ``GeometryError``.

    Raises:
        ValueError: If ``axis`` is not 0 or 1 or ``baseline`` is not finite.
    """
    if axis not in (0, 1):
        raise errors.InputError("axis must be 0 for x or 1 for y")
    if not math.isfinite(float(baseline)):
        raise errors.InputError("baseline must be finite")

    def overwrite(value: Any) -> list[Any]:
        if not isinstance(value, (list, tuple)) or not value:
            raise GeometryError(
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
        return [overwrite(item) for item in value]

    def mask(value: Mapping[str, Any]) -> dict[str, Any]:
        geometry_type = value.get("type")
        result = dict(value)
        if geometry_type == "GeometryCollection":
            geometries = value.get("geometries")
            if not isinstance(geometries, list) or not geometries:
                raise GeometryError(
                    "geometry collection must contain geometries",
                    {"code": "empty_geometry_collection"},
                )
            result["geometries"] = [mask(item) for item in geometries]
        else:
            result["coordinates"] = overwrite(value.get("coordinates"))
        return result

    return mask(geometry)


@dataclasses.dataclass(frozen=True, slots=True)
class Placemark:
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
            GeometryError: If the geometry is not an object or is
                invalid, if a property name is not a string, or if the
                properties and label are not JSON-native. An invalid or
                timezone-naive observation time raises
                ``TemporalError``.
        """
        if not self.feature_id.strip():
            raise errors.InputError("feature_id must not be empty")
        if not isinstance(self.geometry, Mapping):
            raise GeometryError(
                "geometry must be an object", {"code": "invalid_geometry"}
            )
        unpack(self.geometry)
        if self.observation_time is not None:
            object.__setattr__(
                self,
                "observation_time",
                canonical_observation_time(self.observation_time),
            )
        properties = dict(self.properties)
        if any(not isinstance(key, str) for key in properties):
            raise GeometryError(
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
            raise GeometryError(
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
        """The x/y bounds of the geometry as (min x, min y, max x, max y).

        Only the first two values of each position count, so an elevation
        or measure in a 3-D or 4-D position never enters the bounds.
        """
        unpack(self.geometry)
        xs: list[float] = []
        ys: list[float] = []

        def visit(value: Any) -> None:
            if all(
                isinstance(item, (int, float)) and not isinstance(item, bool)
                for item in value
            ):
                xs.append(float(value[0]))
                ys.append(float(value[1]))
                return
            for item in value:
                visit(item)

        def walk(geometry: Mapping[str, Any]) -> None:
            if geometry["type"] == "GeometryCollection":
                for nested in geometry["geometries"]:
                    walk(nested)
            else:
                visit(geometry["coordinates"])

        walk(self.geometry)
        return min(xs), min(ys), max(xs), max(ys)

    @property
    def centroid(self) -> tuple[float, float]:
        """The centre of the bounds as (x, y) in the source CRS units."""
        minimum_x, minimum_y, maximum_x, maximum_y = self.bounds
        return (minimum_x + maximum_x) / 2, (minimum_y + maximum_y) / 2

    def substituted(self, axis: int, baseline: float) -> "Placemark":
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
            geometry=substitute(self.geometry, axis, baseline),
        )


SpatialJoinPredicate = Callable[[Placemark, Placemark], bool]
CoordinateTransformer = Callable[[Placemark, str, str], Placemark]


def placemarks(document: Mapping[str, Any]) -> tuple[Placemark, ...]:
    """Reads the features of a GeoJSON FeatureCollection mapping.

    Feature IDs come from each feature's ``id`` member (as text), falling
    back to its index. Feature labels and observation times are not read.

    Args:
        document: A GeoJSON FeatureCollection.

    Returns:
        One feature per GeoJSON feature, in document order. Errors raised by
        the feature constructor propagate.

    Raises:
        GeometryError: If the root is not a FeatureCollection, ``features``
            is not a list, or a member is not a Feature with object geometry
            and properties.
    """
    if document.get("type") != "FeatureCollection":
        raise GeometryError(
            "GeoJSON document must be a FeatureCollection",
            {"code": "unsupported_geojson_root"},
        )
    raw_features = document.get("features")
    if not isinstance(raw_features, list):
        raise GeometryError(
            "FeatureCollection features must be a list",
            {"code": "invalid_features"},
        )
    features: list[Placemark] = []
    for index, raw_feature in enumerate(raw_features):
        if (
            not isinstance(raw_feature, Mapping)
            or raw_feature.get("type") != "Feature"
        ):
            raise GeometryError(
                "FeatureCollection members must be GeoJSON Features",
                {"code": "invalid_feature", "index": index},
            )
        feature_id = str(raw_feature.get("id", index))
        geometry = raw_feature.get("geometry")
        properties = raw_feature.get("properties", {})
        if not isinstance(geometry, Mapping) or not isinstance(
            properties, Mapping
        ):
            raise GeometryError(
                "feature geometry and properties must be objects",
                {"code": "invalid_feature", "index": index},
            )
        features.append(Placemark(feature_id, geometry, properties))
    return tuple(features)
