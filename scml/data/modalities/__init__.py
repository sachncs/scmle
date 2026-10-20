"""First-class data foundations for non-tabular modality lifecycles."""

from scml.data.modalities.audio import AudioDataset
from scml.data.modalities.audio import AudioMetadata
from scml.data.modalities.audio import AudioProfile
from scml.data.modalities.audio import AudioSample
from scml.data.modalities.document import DocumentDataset
from scml.data.modalities.document import DocumentFormat
from scml.data.modalities.document import DocumentReader
from scml.data.modalities.geospatial import GeoFeature
from scml.data.modalities.geospatial import GeospatialDataset
from scml.data.modalities.geospatial import GeospatialProfile
from scml.data.modalities.geospatial import RasterSample
from scml.data.modalities.graph import GraphData
from scml.data.modalities.graph import GraphEdge
from scml.data.modalities.graph import GraphNode
from scml.data.modalities.image import ImageDataset
from scml.data.modalities.image import ImageSample
from scml.data.modalities.video import VideoDataset
from scml.data.modalities.video import VideoMetadata
from scml.data.modalities.video import VideoProfile
from scml.data.modalities.video import VideoSample

__all__ = [
    "AudioDataset",
    "AudioMetadata",
    "AudioProfile",
    "AudioSample",
    "DocumentDataset",
    "DocumentFormat",
    "DocumentReader",
    "GeoFeature",
    "GeospatialDataset",
    "GeospatialProfile",
    "GraphData",
    "GraphEdge",
    "GraphNode",
    "ImageDataset",
    "ImageSample",
    "RasterSample",
    "VideoDataset",
    "VideoMetadata",
    "VideoProfile",
    "VideoSample",
]
