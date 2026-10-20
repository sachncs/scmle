"""Tests for the public data-foundation import surfaces."""

from scml.data import modalities
from scml.data import readers


def test_modality_and_connector_surfaces_are_public():
    assert modalities.DocumentReader is not None
    assert modalities.GraphData is not None
    assert modalities.ImageDataset is not None
    assert readers.SourceConnector is not None
    assert readers.LocalFileConnector is not None
    assert readers.DbApiConnector is not None
