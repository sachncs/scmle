"""Tests for the model bundle and run store."""

import json

import numpy as np
import pandas as pd
import pytest

from mml.artifacts import bundle as bundle_module
from mml.artifacts import store as store_module
from mml.contracts import data as data_contracts
from mml.contracts import errors
from mml.contracts import tasks
from mml.data import preprocessing
from mml.models import registry as registry_module

PLAN = data_contracts.FeaturePlan(("num",), ("cat",))


@pytest.fixture
def trained():
    rng = np.random.default_rng(0)
    features = pd.DataFrame(
        {"num": rng.normal(size=200), "cat": rng.choice(["a", "b"], 200)}
    )
    target = pd.Series((features.num > 0).astype(int))
    registry = registry_module.ModelRegistry.with_builtins()
    pipe = preprocessing.build_pipeline(
        PLAN, registry.get("random_forest"), tasks.TaskType.BINARY, 0
    ).fit(features, target)
    bundle = bundle_module.ModelBundle.create(
        pipe, PLAN, features, "y", "binary_classification", "roc_auc", "rf"
    )
    return bundle, features


def test_roundtrip_prediction_equivalence(trained, tmp_path):
    bundle, features = trained
    bundle.save(tmp_path / "m")
    loaded = bundle_module.ModelBundle.load(tmp_path / "m", trusted=True)
    np.testing.assert_array_equal(
        bundle.predict(features), loaded.predict(features)
    )
    np.testing.assert_allclose(
        bundle.predict_proba(features), loaded.predict_proba(features)
    )
    assert loaded.classes == (0, 1)
    assert loaded.info.model_name == "rf"


def test_predict_from_file_with_extra_columns(trained, tmp_path):
    bundle, features = trained
    features.assign(junk=1).to_csv(tmp_path / "n.csv", index=False)
    assert len(bundle.predict(tmp_path / "n.csv")) == len(features)
    assert bundle.validate(features.assign(junk=1)).extra == ("junk",)


def test_schema_errors_are_structured(trained):
    bundle, features = trained
    with pytest.raises(errors.SchemaError) as caught:
        bundle.predict(features.drop(columns=["num"]))
    assert caught.value.details["missing"] == ["num"]
    with pytest.raises(errors.SchemaError, match="non-numeric"):
        bundle.predict(features.assign(num="oops"))


def test_unknown_categories_and_missing_values_are_reported(trained):
    bundle, features = trained
    novel = features.head(3).assign(cat="zzz")
    novel.loc[0, "num"] = np.nan
    report = bundle.validate(novel)
    assert report.unseen_categories == {"cat": ["zzz"]}
    assert report.missing_value_counts == {"num": 1}
    assert len(bundle.predict(novel)) == 3


def test_tampering_is_detected(trained, tmp_path):
    bundle = trained[0]
    directory = bundle.save(tmp_path / "m")
    with (directory / bundle_module.MODEL_FILE).open("ab") as handle:
        handle.write(b"x")
    with pytest.raises(errors.ArtifactError, match="checksum"):
        bundle_module.ModelBundle.load(directory, trusted=True)


def test_version_mismatch_blocks_unless_allowed(trained, tmp_path):
    bundle, features = trained
    directory = bundle.save(tmp_path / "m")
    path = directory / bundle_module.MANIFEST_FILE
    manifest = json.loads(path.read_text())
    manifest["sklearn"] = "0.1.0"
    path.write_text(json.dumps(manifest))
    with pytest.raises(errors.ArtifactError, match="different dependencies"):
        bundle_module.ModelBundle.load(directory, trusted=True)
    relaxed = bundle_module.ModelBundle.load(
        directory, trusted=True, allow_version_mismatch=True
    )
    assert len(relaxed.predict(features)) == len(features)


def test_invalid_bundles(tmp_path, trained):
    with pytest.raises(errors.ArtifactError, match="not a model bundle"):
        bundle_module.ModelBundle.load(tmp_path, trusted=True)
    bundle = trained[0]
    directory = bundle.save(tmp_path / "m")
    path = directory / bundle_module.MANIFEST_FILE
    manifest = json.loads(path.read_text())
    manifest["format"] = 99
    path.write_text(json.dumps(manifest))
    with pytest.raises(errors.ArtifactError, match="unsupported"):
        bundle_module.ModelBundle.load(directory, trusted=True)


def test_regression_bundle_has_no_probabilities(tmp_path):
    rng = np.random.default_rng(1)
    features = pd.DataFrame({"num": rng.normal(size=80)})
    registry = registry_module.ModelRegistry.with_builtins()
    plan = data_contracts.FeaturePlan(("num",), ())
    pipe = preprocessing.build_pipeline(
        plan, registry.get("ridge"), tasks.TaskType.REGRESSION, 0
    ).fit(features, features.num * 2)
    bundle = bundle_module.ModelBundle.create(
        pipe, plan, features, "y", "regression", "rmse", "ridge"
    )
    assert bundle.classes is None
    with pytest.raises(errors.ArtifactError, match="probabilities"):
        bundle.predict_proba(features)


def test_run_store_allocates_unique_directories(tmp_path):
    first = store_module.RunStore.create(tmp_path / "runs")
    second = store_module.RunStore.create(tmp_path / "runs")
    assert first.run_id != second.run_id and first.directory.is_dir()
    first.write_json("a.json", {"x": 1})
    first.write_text("b.txt", "hi")
    sub = first.make_directory("trials")
    assert json.loads(first.path("a.json").read_text()) == {"x": 1}
    assert sub.is_dir() and first.path("b.txt").read_text() == "hi"


def test_reference_profile_roundtrips_and_is_checksummed(trained, tmp_path):
    from mml.monitoring import reference

    bundle, features = trained
    scores = pd.Series(bundle.predict_proba(features)[:, 1], name="score")
    bundle.reference = reference.ReferenceProfile.build(
        features, PLAN, scores, "score", {"roc_auc": 0.9}
    )
    directory = bundle.save(tmp_path / "m")
    loaded = bundle_module.ModelBundle.load(directory, trusted=True)
    assert loaded.reference == bundle.reference
    assert bundle_module.REFERENCE_FILE in loaded.info.files
    (directory / bundle_module.REFERENCE_FILE).write_text("{}")
    with pytest.raises(errors.ArtifactError, match="reference profile"):
        bundle_module.ModelBundle.load(directory, trusted=True)


def test_bundle_without_reference_still_loads(trained, tmp_path):
    bundle = trained[0]
    loaded = bundle_module.ModelBundle.load(
        bundle.save(tmp_path / "m"), trusted=True
    )
    assert loaded.reference is None


def test_unsigned_bundles_are_refused_unless_trusted(
    trained, tmp_path, monkeypatch
):
    monkeypatch.delenv(bundle_module.SIGNING_KEY_ENV, raising=False)
    bundle, features = trained
    directory = bundle.save(tmp_path / "m")
    with pytest.raises(errors.ArtifactError, match="unauthenticated"):
        bundle_module.ModelBundle.load(directory)
    assert len(
        bundle_module.ModelBundle.load(directory, trusted=True).predict(
            features
        )
    ) == len(features)


def test_signed_bundles_load_without_trust_and_detect_tampering(
    trained, tmp_path, monkeypatch
):
    monkeypatch.delenv(bundle_module.SIGNING_KEY_ENV, raising=False)
    bundle, features = trained
    directory = bundle.save(tmp_path / "m", signing_key="s3cret")
    assert (directory / bundle_module.SIGNATURE_FILE).exists()
    loaded = bundle_module.ModelBundle.load(directory, signing_key="s3cret")
    assert len(loaded.predict(features)) == len(features)
    with pytest.raises(errors.ArtifactError, match="signature is invalid"):
        bundle_module.ModelBundle.load(directory, signing_key="wrong-key")
    with pytest.raises(errors.ArtifactError, match="signature is invalid"):
        bundle_module.ModelBundle.load(
            directory, trusted=True, signing_key="wrong-key"
        )


def test_a_swapped_model_file_cannot_ride_a_valid_signature(
    trained, tmp_path, monkeypatch
):
    monkeypatch.delenv(bundle_module.SIGNING_KEY_ENV, raising=False)
    bundle = trained[0]
    good = bundle.save(tmp_path / "good", signing_key="k")
    evil = bundle.save(tmp_path / "evil", signing_key="k")
    (evil / bundle_module.MODEL_FILE).write_bytes(b"malicious pickle")
    (evil / bundle_module.MANIFEST_FILE).write_text(
        (good / bundle_module.MANIFEST_FILE).read_text()
    )
    (evil / bundle_module.SIGNATURE_FILE).write_text(
        (good / bundle_module.SIGNATURE_FILE).read_text()
    )
    with pytest.raises(errors.ArtifactError, match="checksum"):
        bundle_module.ModelBundle.load(evil, signing_key="k")


def test_key_from_environment_signs_and_verifies(
    trained, tmp_path, monkeypatch
):
    bundle, features = trained
    monkeypatch.setenv(bundle_module.SIGNING_KEY_ENV, "env-key")
    directory = bundle.save(tmp_path / "m")
    assert len(
        bundle_module.ModelBundle.load(directory).predict(features)
    ) == len(features)
    monkeypatch.delenv(bundle_module.SIGNING_KEY_ENV)
    with pytest.raises(errors.ArtifactError, match="unauthenticated"):
        bundle_module.ModelBundle.load(directory)


def test_resaving_unsigned_removes_a_stale_signature(
    trained, tmp_path, monkeypatch
):
    monkeypatch.delenv(bundle_module.SIGNING_KEY_ENV, raising=False)
    bundle = trained[0]
    directory = bundle.save(tmp_path / "m", signing_key="k")
    bundle.save(directory)
    assert not (directory / bundle_module.SIGNATURE_FILE).exists()
