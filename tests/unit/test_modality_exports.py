"""Tests for the public data-foundation import surfaces."""

import scmle.data.modalities.document
import scmle.data.modalities.video
import scmle.data.readers.connectors


def test_modality_and_connector_surfaces_are_public():
    assert scmle.data.modalities.document.Reader is not None
    assert scmle.data.modalities.video.Dataset is not None
    assert scmle.data.modalities.video.Dataset is not None
    assert scmle.data.readers.connectors.SourceConnector is not None
    assert scmle.data.readers.connectors.LocalFileConnector is not None
    assert scmle.data.readers.connectors.DbApiConnector is not None
