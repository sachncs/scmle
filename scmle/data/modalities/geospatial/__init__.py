"""Dependency-light vector and raster geospatial data contracts."""

from scmle.data.modalities.geospatial.dataset import Dataset
from scmle.data.modalities.geospatial.dataset import Reprojection
from scmle.data.modalities.geospatial.evidence import Autocorrelation
from scmle.data.modalities.geospatial.evidence import Chronology
from scmle.data.modalities.geospatial.evidence import Features
from scmle.data.modalities.geospatial.evidence import Join
from scmle.data.modalities.geospatial.evidence import Leakage
from scmle.data.modalities.geospatial.evidence import Match
from scmle.data.modalities.geospatial.evidence import Partition
from scmle.data.modalities.geospatial.evidence import Profile
from scmle.data.modalities.geospatial.evidence import Provenance
from scmle.data.modalities.geospatial.geometry import autocorrelate
from scmle.data.modalities.geospatial.geometry import canonical_observation_time
from scmle.data.modalities.geospatial.geometry import canonical_source
from scmle.data.modalities.geospatial.geometry import CoordinateTransformer
from scmle.data.modalities.geospatial.geometry import flatten
from scmle.data.modalities.geospatial.geometry import GEOMETRY_TYPES
from scmle.data.modalities.geospatial.geometry import GeometryError
from scmle.data.modalities.geospatial.geometry import MIN_COORDINATE_VALUES
from scmle.data.modalities.geospatial.geometry import MIN_OBSERVATIONS
from scmle.data.modalities.geospatial.geometry import Placemark
from scmle.data.modalities.geospatial.geometry import SHA256_HEX_LENGTH
from scmle.data.modalities.geospatial.geometry import SpatialJoinPredicate
from scmle.data.modalities.geospatial.geometry import substitute
from scmle.data.modalities.geospatial.geometry import TemporalError
from scmle.data.modalities.geospatial.geometry import unpack
from scmle.data.modalities.geospatial.geometry import verify
from scmle.data.modalities.geospatial.linkage import probe
from scmle.data.modalities.geospatial.raster import AFFINE_TRANSFORM_LENGTH
from scmle.data.modalities.geospatial.raster import RasterDecoder
from scmle.data.modalities.geospatial.raster import Tile
from scmle.data.modalities.geospatial.topology import MAX_PERMUTATIONS
from scmle.data.modalities.geospatial.topology import permutation_p_value

__all__ = [
    "AFFINE_TRANSFORM_LENGTH",
    "GEOMETRY_TYPES",
    "MAX_PERMUTATIONS",
    "MIN_COORDINATE_VALUES",
    "MIN_OBSERVATIONS",
    "SHA256_HEX_LENGTH",
    "Autocorrelation",
    "Chronology",
    "CoordinateTransformer",
    "Dataset",
    "Features",
    "GeometryError",
    "Join",
    "Leakage",
    "Match",
    "Partition",
    "Placemark",
    "Profile",
    "Provenance",
    "RasterDecoder",
    "Reprojection",
    "SpatialJoinPredicate",
    "TemporalError",
    "Tile",
    "autocorrelate",
    "canonical_observation_time",
    "canonical_source",
    "flatten",
    "permutation_p_value",
    "probe",
    "substitute",
    "unpack",
    "verify",
]
