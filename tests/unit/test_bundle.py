"""Tests for the model bundle and run store."""

import dataclasses
import json
import platform

import numpy as np
import pandas as pd
import pytest

from scml.artifacts import bundle as bundle_module
from scml.artifacts import store as store_module
from scml.contracts import data as data_contracts
from scml.contracts import errors
from scml.contracts import tasks
from scml.data.curation import preprocessing
from scml.inference import mapping as mapping_module
from scml.models import registry as registry_module

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


def test_schema_tampering_is_detected(trained, tmp_path):
    bundle = trained[0]
    directory = bundle.save(tmp_path / "m")
    schema = directory / bundle_module.SCHEMA_FILE
    schema.write_text(schema.read_text() + "\n")
    with pytest.raises(errors.ArtifactError, match="schema file"):
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


def test_legacy_bundle_requires_opt_in_and_can_be_migrated(
    tmp_path, trained
):
    directory = trained[0].save(tmp_path / "legacy")
    manifest_path = directory / bundle_module.MANIFEST_FILE
    manifest = json.loads(manifest_path.read_text())
    manifest["format"] = 1
    manifest["files"].pop(bundle_module.SCHEMA_FILE)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(errors.ArtifactError, match="unsupported"):
        bundle_module.ModelBundle.load(directory, trusted=True)
    legacy = bundle_module.ModelBundle.load(
        directory, trusted=True, allow_legacy_format=True
    )
    migrated = legacy.save(tmp_path / "migrated")
    assert json.loads(
        (migrated / bundle_module.MANIFEST_FILE).read_text()
    )["format"] == bundle_module.BUNDLE_FORMAT
    assert bundle_module.ModelBundle.load(migrated, trusted=True).info.format == 2


def test_malformed_bundle_metadata_is_structured_error(tmp_path, trained):
    directory = trained[0].save(tmp_path / "m")
    (directory / bundle_module.MANIFEST_FILE).write_text("not-json")
    with pytest.raises(errors.ArtifactError, match="invalid model bundle"):
        bundle_module.ModelBundle.load(directory, trusted=True)


def test_unknown_signature_algorithm_is_rejected(trained, tmp_path):
    directory = trained[0].save(tmp_path / "m", signing_key="secret")
    signature_path = directory / bundle_module.SIGNATURE_FILE
    record = json.loads(signature_path.read_text())
    record["algorithm"] = "sha256"
    signature_path.write_text(json.dumps(record))
    with pytest.raises(errors.ArtifactError, match="signature algorithm"):
        bundle_module.ModelBundle.load(directory, signing_key="secret")


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
    assert not list(first.directory.glob(".b.txt.*"))


def test_reference_profile_roundtrips_and_is_checksummed(trained, tmp_path):
    from scml.monitoring import reference

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
    (evil / bundle_module.MANIFEST_FILE).write_bytes(
        (good / bundle_module.MANIFEST_FILE).read_bytes()
    )
    (evil / bundle_module.SIGNATURE_FILE).write_bytes(
        (good / bundle_module.SIGNATURE_FILE).read_bytes()
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


def test_signing_key_rotation_preserves_bundle_and_rejects_old_key(
    trained, tmp_path
):
    bundle, features = trained
    directory = bundle.save(tmp_path / "m", signing_key="old-key")
    bundle_module.ModelBundle.rotate_signing_key(
        directory, "old-key", "new-key"
    )
    with pytest.raises(errors.ArtifactError, match="signature is invalid"):
        bundle_module.ModelBundle.load(directory, signing_key="old-key")
    loaded = bundle_module.ModelBundle.load(directory, signing_key="new-key")
    assert len(loaded.predict(features)) == len(features)


def test_resaving_unsigned_removes_a_stale_signature(
    trained, tmp_path, monkeypatch
):
    monkeypatch.delenv(bundle_module.SIGNING_KEY_ENV, raising=False)
    bundle = trained[0]
    directory = bundle.save(tmp_path / "m", signing_key="k")
    bundle.save(directory)
    assert not (directory / bundle_module.SIGNATURE_FILE).exists()


def test_default_policy_requires_exact_column_names(trained):
    bundle, features = trained
    renamed = features.rename(columns={"num": "NUM"})
    with pytest.raises(errors.SchemaError, match="missing"):
        bundle.predict(renamed)


def test_explicit_renames_map_incoming_names(trained):
    bundle, features = trained
    renamed = features.rename(columns={"num": "customer_age", "cat": "segment"})
    mapping = mapping_module.ColumnMapping(
        {"customer_age": "num", "segment": "cat"}
    )
    np.testing.assert_array_equal(
        bundle.predict(renamed, mapping), bundle.predict(features)
    )
    report = bundle.validate(renamed, mapping)
    assert report.valid and report.extra == ()


def test_case_and_whitespace_insensitive_matching_is_opt_in(trained):
    bundle, features = trained
    messy = features.rename(columns={"num": "  NUM ", "cat": "Cat"})
    with pytest.raises(errors.SchemaError):
        bundle.predict(messy)
    loose = mapping_module.ColumnMapping(
        case_insensitive=True, strip_whitespace=True
    )
    np.testing.assert_array_equal(
        bundle.predict(messy, loose), bundle.predict(features)
    )
    only_case = mapping_module.ColumnMapping(case_insensitive=True)
    with pytest.raises(errors.SchemaError):  # whitespace still differs
        bundle.predict(messy, only_case)


def test_ambiguous_or_unknown_mappings_are_rejected(trained):
    bundle, features = trained
    clash = features.rename(columns={"num": "NUM"}).assign(Num=1.0)
    with pytest.raises(errors.SchemaError, match="ambiguous"):
        bundle.predict(
            clash, mapping_module.ColumnMapping(case_insensitive=True)
        )
    double = features.assign(other=features["num"]).rename(columns={"num": "a"})
    with pytest.raises(errors.SchemaError, match="ambiguous"):
        bundle.predict(
            double, mapping_module.ColumnMapping({"a": "num", "other": "num"})
        )
    with pytest.raises(errors.SchemaError, match="not in the model schema"):
        bundle.predict(features, mapping_module.ColumnMapping({"num": "ghost"}))


def test_exact_matches_are_never_renamed(trained):
    bundle, features = trained
    mapping = mapping_module.ColumnMapping(
        {"num": "cat"}, case_insensitive=True
    )
    # "num" already exists exactly, so the rename request is ignored.
    np.testing.assert_array_equal(
        bundle.predict(features, mapping), bundle.predict(features)
    )


def test_round_trip_verification_passes_and_reports_the_tolerance(
    trained, tmp_path
):
    bundle, features = trained
    bundle.save(tmp_path / "m")
    record = bundle.verify_round_trip(tmp_path / "m", features.head(50))
    assert record["rows"] == 50
    assert record["largest_difference"] <= record["tolerance"]


def test_round_trip_verification_catches_a_bundle_that_drifted(
    trained, tmp_path
):
    bundle, features = trained
    bundle.save(tmp_path / "m")
    other = bundle_module.ModelBundle.load(tmp_path / "m", trusted=True)
    flipped = features.assign(num=-features.num)
    other.pipeline = other.pipeline.fit(flipped, (flipped.num > 0).astype(int))
    with pytest.raises(errors.ArtifactError, match="does not reproduce"):
        other.verify_round_trip(tmp_path / "m", features.head(50))


def test_inference_time_calibration_is_optional_persisted_and_verified(
    trained, tmp_path
):
    bundle, features = trained
    labels = (features.num > 0.3).astype(int)  # a noisier target than trained
    plain = bundle.predict_proba(features)
    report = bundle.calibrate(features, labels, method="sigmoid")
    calibrated = bundle.predict_proba(features)
    assert not np.allclose(plain, calibrated)
    np.testing.assert_allclose(calibrated.sum(axis=1), 1.0, atol=1e-9)
    assert report["brier_after"] <= report["brier_before"] + 1e-9
    assert bundle.info.calibration.startswith("sigmoid on 200 rows")
    bundle.save(tmp_path / "m")
    loaded = bundle_module.ModelBundle.load(tmp_path / "m", trusted=True)
    np.testing.assert_allclose(loaded.predict_proba(features), calibrated)
    assert bundle.verify_round_trip(tmp_path / "m", features)["rows"] == 200
    (tmp_path / "m" / bundle_module.CALIBRATOR_FILE).write_bytes(b"broken")
    with pytest.raises(errors.ArtifactError, match="calibrator"):
        bundle_module.ModelBundle.load(tmp_path / "m", trusted=True)


def test_calibration_rejects_unknown_methods_and_regression(trained):
    bundle, features = trained
    with pytest.raises(errors.ConfigError, match="unknown calibration"):
        bundle.calibrate(features, (features.num > 0).astype(int), "magic")
    bundle.schema = dataclasses.replace(bundle.schema, classes=None)
    with pytest.raises(errors.ArtifactError, match="only classifiers"):
        bundle.calibrate(features, features.num)


def test_compatibility_checks_every_recorded_library():
    info = bundle_module.BundleInfo("m", "v1", "roc_auc")
    assert info.compatibility_problems() == []
    older = dataclasses.replace(
        info, libraries={**info.libraries, "numpy": "1.0.0"}
    )
    assert any("numpy 1.0.0" in p for p in older.compatibility_problems())
    pandas_off = dataclasses.replace(info, pandas="0.1.0")
    assert any("pandas" in p for p in pandas_off.compatibility_problems())


def test_score_returns_the_structured_inference_result(trained):
    bundle, features = trained
    messy = features.head(5).assign(junk=1)
    result = bundle.score(messy)
    assert len(result["predictions"]) == 5
    assert result["model_version"] == "v1" and result["schema_valid"] is True
    assert result["diagnostics"]["ignored_columns"] == ["junk"]
    assert result["classes"] == [0, 1]
    assert np.allclose(np.sum(result["probabilities"], axis=1), 1.0)
    with pytest.raises(errors.SchemaError):
        bundle.score(features.drop(columns="num"))


def test_the_environment_lock_pins_the_build_environment(trained, tmp_path):
    bundle = trained[0]
    path = tmp_path / "lock.txt"

    text = bundle.environment_lock(path)

    assert path.read_text() == text
    assert text.startswith("# scml environment lock")
    assert f"# python {platform.python_version()}" in text
    assert f"numpy=={np.__version__}" in text
    assert "# platform " in text and "unknown" not in text.splitlines()[2]
    names = [line.split("==")[0] for line in text.splitlines() if "==" in line]
    assert names == sorted(names)


def test_the_lock_survives_saving_and_loading(trained, tmp_path):
    bundle = trained[0]
    bundle.save(tmp_path / "m")

    loaded = bundle_module.ModelBundle.load(tmp_path / "m", trusted=True)

    assert loaded.environment_lock() == bundle.environment_lock()


def test_a_version_mismatch_arrives_with_the_lock_attached(trained, tmp_path):
    bundle = trained[0]
    bundle.save(tmp_path / "m")
    manifest = tmp_path / "m" / bundle_module.MANIFEST_FILE
    recorded = json.loads(manifest.read_text())
    recorded["sklearn"] = "0.1.0"
    manifest.write_text(json.dumps(recorded))

    with pytest.raises(errors.ArtifactError, match="scikit-learn") as caught:
        bundle_module.ModelBundle.load(tmp_path / "m", trusted=True)

    lock = caught.value.details["environment_lock"]
    assert lock == bundle.environment_lock()
    assert f"numpy=={np.__version__}" in lock
