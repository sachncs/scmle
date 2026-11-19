"""Exact refusals of a modality package manifest and its lifecycle report."""

import json
import types

import pytest

from scmle.contracts import errors
from scmle.contracts import modalities
from scmle.pipeline.modality_lifecycle import predictors

ADAPTER = types.SimpleNamespace(modality=modalities.Modality.AUDIO)
MODEL = types.SimpleNamespace(n_features_in_=2)


def manifest(tmp_path, **changes):
    fields = {
        "modality": "audio",
        "feature_names": ["a", "b"],
        "dataset_hash": "d" * 64,
        "model_sha256": "m" * 64,
        "signature": {"value": "s"},
    }
    fields.update(changes)
    (tmp_path / "manifest.json").write_text(
        json.dumps(fields), encoding="utf-8"
    )
    return tmp_path


def test_declared_returns_what_the_manifest_records(tmp_path):
    names, identity = predictors.declared(manifest(tmp_path), ADAPTER, MODEL)

    assert names == ("a", "b")
    assert identity == ("d" * 64, "m" * 64, "s")


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"modality": "video"}, "package modality does not match adapter"),
        (
            {"feature_names": ["a"]},
            "package model and feature schema do not match",
        ),
        ({"signature": "s"}, "package signature is invalid"),
        ({"signature": None}, "package signature is invalid"),
    ],
)
def test_declared_refusals_are_exact(tmp_path, changes, message):
    with pytest.raises(errors.InputError) as caught:
        predictors.declared(manifest(tmp_path, **changes), ADAPTER, MODEL)

    assert str(caught.value) == message


def test_a_manifest_that_is_not_an_object_is_refused(tmp_path):
    (tmp_path / "manifest.json").write_text("[]", encoding="utf-8")

    with pytest.raises(errors.InputError) as caught:
        predictors.declared(tmp_path, ADAPTER, MODEL)

    assert str(caught.value) == "modality package manifest must be an object"


def report(tmp_path):
    return types.SimpleNamespace(
        modality="audio",
        digests={"dataset": "d"},
        package=types.SimpleNamespace(
            path=str(tmp_path), model_sha256="m", signature="s"
        ),
    )


@pytest.mark.parametrize(
    ("identity", "message"),
    [
        (("x", "m", "s"), "lifecycle report dataset identity does not match"),
        (("d", "x", "s"), "lifecycle report model digest does not match"),
    ],
)
def test_identify_refusals_are_exact(tmp_path, identity, message):
    with pytest.raises(errors.InputError) as caught:
        predictors.Predictor.identify(
            report(tmp_path), ADAPTER, tmp_path, identity
        )

    assert str(caught.value) == message
