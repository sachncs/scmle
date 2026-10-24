"""First-class data foundations for non-tabular modality lifecycles."""

from scmle.data.modalities.audio import AudioDataset
from scmle.data.modalities.audio import AudioFeatures
from scmle.data.modalities.audio import AudioMetadata
from scmle.data.modalities.audio import AudioProfile
from scmle.data.modalities.audio import AudioSample
from scmle.data.modalities.audio import AudioTranscript
from scmle.data.modalities.audio import AudioTranscriptSegment
from scmle.data.modalities.curation import ModalityCurationReport
from scmle.data.modalities.curation import ModalityDatasetCard
from scmle.data.modalities.document import DocumentCitation
from scmle.data.modalities.document import DocumentCurationReport
from scmle.data.modalities.document import DocumentDataset
from scmle.data.modalities.document import DocumentElement
from scmle.data.modalities.document import DocumentFormat
from scmle.data.modalities.document import DocumentLayout
from scmle.data.modalities.document import DocumentReader
from scmle.data.modalities.document import redact_sensitive_text
from scmle.data.modalities.document import sensitive_matches
from scmle.data.modalities.geospatial import canonical_observation_time
from scmle.data.modalities.geospatial import GeoFeature
from scmle.data.modalities.geospatial import GeospatialDataset
from scmle.data.modalities.geospatial import GeospatialFeatures
from scmle.data.modalities.geospatial import GeospatialProfile
from scmle.data.modalities.geospatial import GeospatialTemporalError
from scmle.data.modalities.geospatial import GeospatialTemporalSplit
from scmle.data.modalities.geospatial import moran_statistic
from scmle.data.modalities.geospatial import RasterDecoder
from scmle.data.modalities.geospatial import RasterSample
from scmle.data.modalities.geospatial import SpatialAutocorrelation
from scmle.data.modalities.graph import canonical_event_time
from scmle.data.modalities.graph import GraphData
from scmle.data.modalities.graph import GraphEdge
from scmle.data.modalities.graph import GraphNode
from scmle.data.modalities.graph import GraphTemporalError
from scmle.data.modalities.graph import TemporalGraphSplit
from scmle.data.modalities.image import ImageAugmentation
from scmle.data.modalities.image import ImageDataset
from scmle.data.modalities.image import ImageSample
from scmle.data.modalities.image import ImageSplit
from scmle.data.modalities.video import VideoDataset
from scmle.data.modalities.video import VideoFeatures
from scmle.data.modalities.video import VideoFrameDecoder
from scmle.data.modalities.video import VideoMetadata
from scmle.data.modalities.video import VideoProfile
from scmle.data.modalities.video import VideoSample
from scmle.data.modalities.video import VideoSplit
from scmle.data.modalities.video import VideoTemporalError
from scmle.data.modalities.video import VideoTemporalSplit

__all__ = [
    "AudioDataset",
    "AudioFeatures",
    "AudioMetadata",
    "AudioProfile",
    "AudioSample",
    "AudioTranscript",
    "AudioTranscriptSegment",
    "DocumentCitation",
    "DocumentCurationReport",
    "DocumentDataset",
    "DocumentElement",
    "DocumentFormat",
    "DocumentLayout",
    "DocumentReader",
    "GeoFeature",
    "GeospatialDataset",
    "GeospatialFeatures",
    "GeospatialProfile",
    "GeospatialTemporalError",
    "GeospatialTemporalSplit",
    "GraphData",
    "GraphEdge",
    "GraphNode",
    "GraphTemporalError",
    "ImageAugmentation",
    "ImageDataset",
    "ImageSample",
    "ImageSplit",
    "ModalityCurationReport",
    "ModalityDatasetCard",
    "RasterDecoder",
    "RasterSample",
    "SpatialAutocorrelation",
    "TemporalGraphSplit",
    "VideoDataset",
    "VideoFeatures",
    "VideoFrameDecoder",
    "VideoMetadata",
    "VideoProfile",
    "VideoSample",
    "VideoSplit",
    "VideoTemporalError",
    "VideoTemporalSplit",
    "canonical_event_time",
    "canonical_observation_time",
    "moran_statistic",
    "redact_sensitive_text",
    "sensitive_matches",
]
