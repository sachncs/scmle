"""Dependency-light vector and raster geospatial data contracts."""

from collections.abc import Mapping
import dataclasses
import hashlib
import json
import math
import pathlib
import random
from typing import Any
from typing import Self

from scml.contracts import data as data_contracts
from scml.contracts import errors
from scml.data.modalities.curation import build_finding
from scml.data.modalities.curation import label_counts
from scml.data.modalities.curation import ModalityCurationReport

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


class GeospatialValidationError(errors.DataError):
    """A vector or raster sample violates a geospatial invariant."""

    category = "geospatial_validation"


def canonical_source(source: str | pathlib.Path) -> str:
    """Return a stable source identifier for a path or URI-like value."""
    value = str(source).strip()
    if not value:
        return "memory://"
    if "://" in value:
        return value
    return pathlib.Path(value).expanduser().resolve(strict=False).as_uri()


def validate_coordinates(value: Any) -> tuple[float, ...]:
    """Validate a nested coordinate array and return its numeric leaves."""
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
    """Validate a GeoJSON geometry and return all coordinate leaves."""
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


@dataclasses.dataclass(frozen=True, slots=True)
class GeoFeature:
    """Validated GeoJSON feature with stable serialized identity."""

    feature_id: str
    geometry: Mapping[str, Any]
    properties: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    label: str | int | float | bool | None = None
    geometry_hash: str = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        """Validate geometry and derive a canonical geometry hash."""
        if not self.feature_id.strip():
            raise ValueError("feature_id must not be empty")
        if not isinstance(self.geometry, Mapping):
            raise GeospatialValidationError(
                "geometry must be an object", {"code": "invalid_geometry"}
            )
        geometry_coordinates(self.geometry)
        payload = json.dumps(
            self.geometry,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        object.__setattr__(
            self, "geometry_hash", hashlib.sha256(payload).hexdigest()
        )

    @property
    def geometry_type(self) -> str:
        """Return the validated GeoJSON geometry type."""
        return str(self.geometry["type"])

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Return the axis-aligned x/y bounds of the geometry."""
        coordinates = geometry_coordinates(self.geometry)
        xs = coordinates[0::2]
        ys = coordinates[1::2]
        return min(xs), min(ys), max(xs), max(ys)


@dataclasses.dataclass(frozen=True, slots=True)
class GeospatialProfile:
    """Bounded data-understanding summary for a vector dataset."""

    feature_count: int
    geometry_types: tuple[str, ...]
    crs: str
    bounds: tuple[float, float, float, float]
    content_hash: str

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible geospatial profile data."""
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, slots=True)
class SpatialSplit:
    """Spatial-block split that keeps nearby feature blocks together."""

    train: tuple[GeoFeature, ...]
    validation: tuple[GeoFeature, ...]
    test: tuple[GeoFeature, ...]
    block_size: float
    blocks: Mapping[str, tuple[str, ...]]

    def block_ids(self) -> dict[str, tuple[str, ...]]:
        """Return the stable spatial blocks assigned to each partition."""
        return {name: tuple(values) for name, values in self.blocks.items()}


@dataclasses.dataclass(frozen=True, slots=True)
class GeospatialDataset:
    """Immutable GeoJSON-like vector collection with CRS provenance."""

    features: tuple[GeoFeature, ...]
    crs: str = ""
    source: str = "memory://"
    content_hash: str = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        """Reject duplicate identities and derive order-independent content."""
        normalized = tuple(self.features)
        if not normalized:
            raise GeospatialValidationError(
                "geospatial dataset must contain at least one feature",
                {"code": "empty_dataset"},
            )
        ids = [feature.feature_id for feature in normalized]
        if len(ids) != len(set(ids)):
            raise GeospatialValidationError(
                "geospatial dataset contains duplicate feature IDs",
                {"code": "duplicate_feature_id"},
            )
        digest = hashlib.sha256(
            "\n".join(
                f"{feature.feature_id}\0{feature.geometry_hash}"
                for feature in normalized
            ).encode("utf-8")
        ).hexdigest()
        object.__setattr__(self, "features", normalized)
        object.__setattr__(self, "source", canonical_source(self.source))
        object.__setattr__(self, "content_hash", digest)

    @classmethod
    def from_geojson(
        cls,
        document: Mapping[str, Any],
        source: str = "memory://",
        crs: str = "",
    ) -> Self:
        """Build a dataset from a GeoJSON FeatureCollection mapping."""
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
        """Return geometry counts, bounds, CRS, and immutable identity."""
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
        """Build geometry, CRS, and spatial-interpretation evidence."""
        profile = self.profile()
        labels = label_counts(feature.label for feature in self.features)
        profile_data = profile.to_dict()
        profile_data["label_counts"] = labels
        quality_findings = []
        risk_findings = []
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

    def split_spatial(
        self,
        block_size: float,
        validation_fraction: float = 0.2,
        test_fraction: float = 0.2,
        seed: int = 0,
    ) -> SpatialSplit:
        """Split features by CRS-unit spatial blocks.

        ``block_size`` is expressed in the dataset CRS units. The method uses
        geometry centroids for block assignment and keeps every block in one
        partition. Large geometries that cross block boundaries need a
        domain-specific spatial join policy before this split is suitable for
        release.
        """
        if not math.isfinite(block_size) or block_size <= 0:
            raise ValueError("block_size must be a positive finite number")
        if not 0 <= validation_fraction < 1:
            raise ValueError("validation_fraction must be in [0, 1)")
        if not 0 <= test_fraction < 1:
            raise ValueError("test_fraction must be in [0, 1)")
        if validation_fraction + test_fraction >= 1:
            raise ValueError("validation_fraction + test_fraction must be < 1")
        origin_x, origin_y, _, _ = self.profile().bounds

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


@dataclasses.dataclass(frozen=True, slots=True)
class RasterSample:
    """Raster identity and metadata without assuming a raster codec."""

    source: str
    encoded_bytes: bytes = dataclasses.field(repr=False)
    file_hash: str
    width: int
    height: int
    bands: int
    dtype: str
    crs: str
    transform: tuple[float, ...]
    byte_size: int = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        """Validate raster shape, CRS provenance, and source identity."""
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
        if not raw:
            raise GeospatialValidationError(
                "raster payload is empty", {"code": "empty_payload"}
            )
        if hashlib.sha256(raw).hexdigest() != self.file_hash:
            raise ValueError("file_hash does not match encoded_bytes")
        object.__setattr__(self, "byte_size", len(raw))


__all__ = [
    "GEOMETRY_TYPES",
    "GeoFeature",
    "GeospatialDataset",
    "GeospatialProfile",
    "GeospatialValidationError",
    "RasterSample",
    "SpatialSplit",
    "canonical_source",
    "geometry_coordinates",
    "validate_coordinates",
]
