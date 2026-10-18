"""Prediction from saved bundles through the public functions."""

import numpy as np
import pytest
import support

import scml
from scml.contracts import errors


def test_a_saved_bundle_predicts_without_the_project(binary_frame, tmp_path):
    result = support.make_project(
        binary_frame, "y", time_budget=300, output_dir=tmp_path
    ).run()
    result.save(tmp_path / "model")
    rows = binary_frame.drop(columns="y").head(10)

    labels = scml.predict(tmp_path / "model", rows, trusted=True)

    np.testing.assert_array_equal(labels, result.predict(rows))
    model = scml.load_model(tmp_path / "model", trusted=True)
    assert model.info.model_version == result.bundle.info.model_version


def test_loading_an_unsigned_bundle_needs_trust(binary_frame, tmp_path):
    result = support.make_project(
        binary_frame, "y", time_budget=300, output_dir=tmp_path
    ).run()
    result.save(tmp_path / "model")

    with pytest.raises(errors.ArtifactError):
        scml.load_model(tmp_path / "model")
