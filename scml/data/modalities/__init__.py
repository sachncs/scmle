"""First-class data foundations for non-tabular modality lifecycles."""

from scml.data.modalities.document import DocumentDataset
from scml.data.modalities.document import DocumentFormat
from scml.data.modalities.document import DocumentReader
from scml.data.modalities.graph import GraphData
from scml.data.modalities.graph import GraphEdge
from scml.data.modalities.graph import GraphNode
from scml.data.modalities.image import ImageDataset
from scml.data.modalities.image import ImageSample

__all__ = [
    "DocumentDataset",
    "DocumentFormat",
    "DocumentReader",
    "GraphData",
    "GraphEdge",
    "GraphNode",
    "ImageDataset",
    "ImageSample",
]
