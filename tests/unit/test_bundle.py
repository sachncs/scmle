"""Tests for the model bundle and run store."""

import dataclasses
import json
import platform
import sys
import types

import numpy
import pandas
import pytest

import scmle.artifacts.bundle
import scmle.artifacts.store
from scmle.contracts import data
from scmle.contracts import errors
from scmle.contracts import tasks
import scmle.contracts.signing
import scmle.inference.mapping
import scmle.models.registry
from scmle.preparation import preprocessing

PLAN = data.Blueprint(("num",), ("cat",))


@pytest.fixture
def trained():
    rng = numpy.random.default_rng(0)
    features = pandas.DataFrame(
        {"num": rng.normal(size=200), "cat": rng.choice(["a", "b"], 200)}
    )
    target = pandas.Series((features.num > 0).astype(int))
    registry = scmle.models.registry.Catalog.stock()
    pipe = preprocessing.instantiate(
        PLAN, registry.get("random_forest"), tasks.Task.BINARY, 0
    ).fit(features, target)
    bundle = scmle.artifacts.bundle.Bundle.create(
        pipe, PLAN, features, "y", "binary_classification", "roc_auc", "rf"
    )
    return bundle, features


def test_roundtrip_prediction_equivalence(trained, tmp_path):
    bundle, features = trained
    bundle.save(tmp_path / "m")
    loaded = scmle.artifacts.bundle.Bundle.load(tmp_path / "m", trusted=True)
    numpy.testing.assert_array_equal(
        bundle.predict(features), loaded.predict(features)
    )
    numpy.testing.assert_allclose(
        bundle.predict_proba(features), loaded.predict_proba(features)
    )
    assert loaded.classes == (0, 1)
    assert loaded.info.model_name == "rf"


def test_predict_from_file_with_extra_columns(trained, tmp_path):
    # Enforces docs/GLOSSARY.md:588.
    # Enforces website/src/learn/glossary.html:294.
    bundle, features = trained
    features.assign(junk=1).to_csv(tmp_path / "n.csv", index=False)
    assert len(bundle.predict(tmp_path / "n.csv")) == len(features)
    assert bundle.validate(features.assign(junk=1)).extra == ("junk",)


def test_schema_errors_are_structured(trained):
    # Enforces docs/GLOSSARY.md:588.
    # Enforces website/src/learn/glossary.html:294.
    bundle, features = trained
    with pytest.raises(errors.SchemaError) as caught:
        bundle.predict(features.drop(columns=["num"]))
    assert caught.value.details["missing"] == ["num"]
    with pytest.raises(errors.SchemaError, match="non-numeric"):
        bundle.predict(features.assign(num="oops"))


def test_unknown_categories_and_missing_values_are_reported(trained):
    bundle, features = trained
    novel = features.head(3).assign(cat="zzz")
    novel.loc[0, "num"] = numpy.nan
    report = bundle.validate(novel)
    assert report.unseen_categories == {"cat": ["zzz"]}
    assert report.missing_value_counts == {"num": 1}
    assert len(bundle.predict(novel)) == 3


def test_tampering_is_detected(trained, tmp_path):
    # Enforces README.md:154.
    # Enforces website/src/pages/index.astro:446.
    bundle = trained[0]
    directory = bundle.save(tmp_path / "m")
    with (directory / scmle.artifacts.bundle.MODEL_FILE).open("ab") as handle:
        handle.write(b"x")
    with pytest.raises(errors.ArtifactError, match="checksum"):
        scmle.artifacts.bundle.Bundle.load(directory, trusted=True)


def test_schema_tampering_is_detected(trained, tmp_path):
    bundle = trained[0]
    directory = bundle.save(tmp_path / "m")
    schema = directory / scmle.artifacts.bundle.SCHEMA_FILE
    schema.write_text(schema.read_text() + "\n")
    with pytest.raises(errors.ArtifactError, match="schema file"):
        scmle.artifacts.bundle.Bundle.load(directory, trusted=True)


def test_version_mismatch_blocks_unless_allowed(trained, tmp_path):
    # Enforces docs/SECURITY.md:27.
    # Enforces docs/WHEN_THINGS_GO_WRONG.md:473.
    # Enforces website/src/learn/when-things-go-wrong.html:202.
    # Enforces website/src/pages/docs/threat-model.astro:8.
    bundle, features = trained
    directory = bundle.save(tmp_path / "m")
    path = directory / scmle.artifacts.bundle.MANIFEST_FILE
    manifest = json.loads(path.read_text())
    manifest["sklearn"] = "0.1.0"
    path.write_text(json.dumps(manifest))
    with pytest.raises(errors.ArtifactError, match="different dependencies"):
        scmle.artifacts.bundle.Bundle.load(directory, trusted=True)
    relaxed = scmle.artifacts.bundle.Bundle.load(
        directory, trusted=True, allow_version_mismatch=True
    )
    assert len(relaxed.predict(features)) == len(features)


def test_invalid_bundles(tmp_path, trained):
    with pytest.raises(errors.ArtifactError, match="not a model bundle"):
        scmle.artifacts.bundle.Bundle.load(tmp_path, trusted=True)
    bundle = trained[0]
    directory = bundle.save(tmp_path / "m")
    path = directory / scmle.artifacts.bundle.MANIFEST_FILE
    manifest = json.loads(path.read_text())
    manifest["format"] = 99
    path.write_text(json.dumps(manifest))
    with pytest.raises(errors.ArtifactError, match="unsupported"):
        scmle.artifacts.bundle.Bundle.load(directory, trusted=True)


def test_a_bundle_from_before_the_class_renames_is_refused_with_a_reason(
    tmp_path, trained
):
    # Enforces docs/API_STABILITY.md:337.
    directory = trained[0].save(tmp_path / "earlier")
    path = directory / scmle.artifacts.bundle.MANIFEST_FILE
    manifest = json.loads(path.read_text())
    manifest["format"] = 3
    path.write_text(json.dumps(manifest))

    with pytest.raises(errors.ArtifactError) as raised:
        scmle.artifacts.bundle.Bundle.load(directory, trusted=True)

    message = str(raised.value)
    assert "unsupported bundle format 3" in message
    assert "module path" in message
    assert "retrain and save it again" in message


def test_a_member_naming_a_missing_module_is_refused(tmp_path, trained):
    # Enforces docs/API_STABILITY.md:342.
    vanished = types.ModuleType("vanished")
    vanished.Thing = type("Thing", (), {"__module__": "vanished"})
    sys.modules["vanished"] = vanished
    bundle = trained[0]
    kept = bundle.calibrator
    try:
        bundle.calibrator = vanished.Thing()
        directory = bundle.save(tmp_path / "moved")
    finally:
        bundle.calibrator = kept
        del sys.modules["vanished"]

    with pytest.raises(errors.ArtifactError, match="retrain") as raised:
        scmle.artifacts.bundle.Bundle.load(directory, trusted=True)

    assert isinstance(raised.value.__cause__, ModuleNotFoundError)


def test_legacy_bundle_requires_opt_in_and_can_be_migrated(tmp_path, trained):
    # Enforces docs/API_STABILITY.md:337.
    # Enforces website/src/pages/docs/api-stability.astro:83.
    # Enforces website/src/pages/docs/api.astro:489.
    directory = trained[0].save(tmp_path / "legacy")
    manifest_path = directory / scmle.artifacts.bundle.MANIFEST_FILE
    manifest = json.loads(manifest_path.read_text())
    manifest["format"] = 1
    manifest["files"].pop(scmle.artifacts.bundle.SCHEMA_FILE)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(errors.ArtifactError, match="unsupported"):
        scmle.artifacts.bundle.Bundle.load(directory, trusted=True)
    legacy = scmle.artifacts.bundle.Bundle.load(
        directory, trusted=True, allow_legacy_format=True
    )
    migrated = legacy.save(tmp_path / "migrated")
    assert (
        json.loads(
            (migrated / scmle.artifacts.bundle.MANIFEST_FILE).read_text()
        )["format"]
        == scmle.artifacts.bundle.BUNDLE_FORMAT
    )
    assert (
        scmle.artifacts.bundle.Bundle.load(migrated, trusted=True).info.format
        == scmle.artifacts.bundle.BUNDLE_FORMAT
    )


def test_malformed_bundle_metadata_is_structured_error(tmp_path, trained):
    directory = trained[0].save(tmp_path / "m")
    (directory / scmle.artifacts.bundle.MANIFEST_FILE).write_text("not-json")
    with pytest.raises(errors.ArtifactError, match="invalid model bundle"):
        scmle.artifacts.bundle.Bundle.load(directory, trusted=True)


def test_unknown_signature_algorithm_is_rejected(trained, tmp_path):
    directory = trained[0].save(tmp_path / "m", signing_key="secret")
    signature_path = directory / scmle.artifacts.bundle.SIGNATURE_FILE
    record = json.loads(signature_path.read_text())
    record["algorithm"] = "sha256"
    signature_path.write_text(json.dumps(record))
    with pytest.raises(errors.ArtifactError, match="signature algorithm"):
        scmle.artifacts.bundle.Bundle.load(directory, signing_key="secret")


def test_regression_bundle_has_no_probabilities(tmp_path):
    rng = numpy.random.default_rng(1)
    features = pandas.DataFrame({"num": rng.normal(size=80)})
    registry = scmle.models.registry.Catalog.stock()
    plan = data.Blueprint(("num",), ())
    pipe = preprocessing.instantiate(
        plan, registry.get("ridge"), tasks.Task.REGRESSION, 0
    ).fit(features, features.num * 2)
    bundle = scmle.artifacts.bundle.Bundle.create(
        pipe, plan, features, "y", "regression", "rmse", "ridge"
    )
    assert bundle.classes is None
    with pytest.raises(errors.ArtifactError, match="probabilities"):
        bundle.predict_proba(features)


def test_run_store_allocates_unique_directories(tmp_path):
    first = scmle.artifacts.store.Workspace.create(tmp_path / "runs")
    second = scmle.artifacts.store.Workspace.create(tmp_path / "runs")
    assert first.run_id != second.run_id and first.directory.is_dir()
    first.write_json("a.json", {"x": 1})
    first.write_text("b.txt", "hi")
    sub = first.path("trials")
    sub.mkdir()
    assert json.loads(first.path("a.json").read_text()) == {"x": 1}
    assert sub.is_dir() and first.path("b.txt").read_text() == "hi"
    assert not list(first.directory.glob(".b.txt.*"))


def test_run_store_cleans_partial_write_when_replace_fails(
    tmp_path, monkeypatch
):
    store = scmle.artifacts.store.Workspace.create(tmp_path / "runs")
    target = store.path("payload.json")
    store.write_text("payload.json", "original")

    def fail_replace(source, destination):
        raise OSError("disk full")

    monkeypatch.setattr(scmle.artifacts.store.os, "replace", fail_replace)
    with pytest.raises(OSError, match="disk full"):
        store.write_text("payload.json", "replacement")

    assert target.read_text() == "original"
    assert not list(store.directory.glob(".payload.json.*"))


def test_reference_profile_roundtrips_and_is_checksummed(trained, tmp_path):
    from scmle.data import reference

    bundle, features = trained
    scores = pandas.Series(bundle.predict_proba(features)[:, 1], name="score")
    bundle.reference = reference.Baseline.build(
        features, PLAN, scores, "score", {"roc_auc": 0.9}
    )
    directory = bundle.save(tmp_path / "m")
    loaded = scmle.artifacts.bundle.Bundle.load(directory, trusted=True)
    assert loaded.reference == bundle.reference
    assert scmle.artifacts.bundle.REFERENCE_FILE in loaded.info.files
    (directory / scmle.artifacts.bundle.REFERENCE_FILE).write_text("{}")
    with pytest.raises(errors.ArtifactError, match="reference profile"):
        scmle.artifacts.bundle.Bundle.load(directory, trusted=True)


def test_bundle_without_reference_still_loads(trained, tmp_path):
    bundle = trained[0]
    loaded = scmle.artifacts.bundle.Bundle.load(
        bundle.save(tmp_path / "m"), trusted=True
    )
    assert loaded.reference is None


def test_unsigned_bundles_are_refused_unless_trusted(
    trained, tmp_path, monkeypatch
):
    # Enforces docs/GLOSSARY.md:603.
    # Enforces docs/ARTIFACTS.md:108.
    # Enforces README.md:154 and docs/START_HERE.md:436.
    # Enforces docs/BEGINNER_FINDINGS.md:37.
    # Enforces website/src/learn/beginner-findings.html:22.
    # Enforces website/src/learn/glossary.html:302.
    # Enforces website/src/learn/start-here.html:221.
    # Enforces website/src/pages/docs/artifacts.astro:130.
    # Enforces website/src/pages/docs/quickstart.astro:122.
    # Enforces website/src/pages/docs/quickstart.astro:94.
    # Enforces website/src/pages/docs/quickstart.astro:96.
    # Enforces website/src/pages/docs/threat-model.astro:6.
    # Enforces website/src/pages/docs/threat-model.astro:67.
    # Enforces website/src/pages/index.astro:236.
    # Enforces website/src/pages/index.astro:250.
    # Enforces website/src/pages/index.astro:446.
    # Enforces website/src/pages/docs/cli.astro:144.
    monkeypatch.delenv(scmle.contracts.signing.SIGNING_KEY_ENV, raising=False)
    bundle, features = trained
    directory = bundle.save(tmp_path / "m")
    with pytest.raises(errors.ArtifactError, match="unauthenticated"):
        scmle.artifacts.bundle.Bundle.load(directory)
    assert len(
        scmle.artifacts.bundle.Bundle.load(directory, trusted=True).predict(
            features
        )
    ) == len(features)


def test_signed_bundles_load_without_trust_and_detect_tampering(
    trained, tmp_path, monkeypatch
):
    # Enforces docs/ARTIFACTS.md:108.
    # Enforces docs/BENCHMARKS.md:18.
    # Enforces docs/WHEN_THINGS_GO_WRONG.md:458.
    # Enforces README.md:154.
    # Enforces website/src/learn/when-things-go-wrong.html:196.
    # Enforces website/src/pages/docs/benchmarks.astro:719-721.
    # Enforces website/src/pages/docs/artifacts.astro:130.
    # Enforces website/src/pages/docs/quickstart.astro:96.
    # Enforces website/src/pages/docs/threat-model.astro:6.
    # Enforces website/src/pages/index.astro:236.
    # Enforces website/src/pages/index.astro:426.
    # Enforces website/src/pages/index.astro:446.
    monkeypatch.delenv(scmle.contracts.signing.SIGNING_KEY_ENV, raising=False)
    bundle, features = trained
    directory = bundle.save(tmp_path / "m", signing_key="s3cret")
    assert (directory / scmle.artifacts.bundle.SIGNATURE_FILE).exists()
    loaded = scmle.artifacts.bundle.Bundle.load(directory, signing_key="s3cret")
    assert len(loaded.predict(features)) == len(features)
    with pytest.raises(errors.ArtifactError, match="signature is invalid"):
        scmle.artifacts.bundle.Bundle.load(directory, signing_key="wrong-key")
    with pytest.raises(errors.ArtifactError, match="signature is invalid"):
        scmle.artifacts.bundle.Bundle.load(
            directory, trusted=True, signing_key="wrong-key"
        )


def test_a_swapped_model_file_cannot_ride_a_valid_signature(
    trained, tmp_path, monkeypatch
):
    # Enforces docs/BENCHMARKS.md:18.
    # Enforces README.md:154.
    # Enforces website/src/pages/docs/benchmarks.astro:719-721.
    # Enforces website/src/pages/index.astro:236.
    # Enforces website/src/pages/index.astro:426.
    monkeypatch.delenv(scmle.contracts.signing.SIGNING_KEY_ENV, raising=False)
    bundle = trained[0]
    good = bundle.save(tmp_path / "good", signing_key="k")
    evil = bundle.save(tmp_path / "evil", signing_key="k")
    (evil / scmle.artifacts.bundle.MODEL_FILE).write_bytes(b"malicious pickle")
    (evil / scmle.artifacts.bundle.MANIFEST_FILE).write_bytes(
        (good / scmle.artifacts.bundle.MANIFEST_FILE).read_bytes()
    )
    (evil / scmle.artifacts.bundle.SIGNATURE_FILE).write_bytes(
        (good / scmle.artifacts.bundle.SIGNATURE_FILE).read_bytes()
    )
    with pytest.raises(errors.ArtifactError, match="checksum"):
        scmle.artifacts.bundle.Bundle.load(evil, signing_key="k")


def test_a_signed_bundle_with_no_key_is_refused_even_when_trusted(
    trained, tmp_path, monkeypatch
):
    # Enforces website/src/pages/docs/quickstart.astro:116.
    # Enforces website/src/pages/docs/threat-model.astro:6.
    monkeypatch.delenv(scmle.contracts.signing.SIGNING_KEY_ENV, raising=False)
    monkeypatch.delenv(
        scmle.contracts.signing.SIGNING_KEY_FILE_ENV, raising=False
    )
    directory = trained[0].save(tmp_path / "m", signing_key="k")
    for trusted in (False, True):
        with pytest.raises(errors.ArtifactError, match="refusing"):
            scmle.artifacts.bundle.Bundle.load(directory, trusted=trusted)


def test_key_from_environment_signs_and_verifies(
    trained, tmp_path, monkeypatch
):
    # Enforces docs/BEGINNER_FINDINGS.md:37.
    # Enforces website/src/learn/beginner-findings.html:22.
    bundle, features = trained
    monkeypatch.setenv(scmle.contracts.signing.SIGNING_KEY_ENV, "env-key")
    directory = bundle.save(tmp_path / "m")
    assert len(
        scmle.artifacts.bundle.Bundle.load(directory).predict(features)
    ) == len(features)
    monkeypatch.delenv(scmle.contracts.signing.SIGNING_KEY_ENV)
    with pytest.raises(errors.ArtifactError, match="unauthenticated"):
        scmle.artifacts.bundle.Bundle.load(directory)


def test_signing_key_rotation_preserves_bundle_and_rejects_old_key(
    trained, tmp_path
):
    bundle, features = trained
    directory = bundle.save(tmp_path / "m", signing_key="old-key")
    scmle.artifacts.bundle.Bundle.rotate_signing_key(
        directory, "old-key", "new-key"
    )
    with pytest.raises(errors.ArtifactError, match="signature is invalid"):
        scmle.artifacts.bundle.Bundle.load(directory, signing_key="old-key")
    loaded = scmle.artifacts.bundle.Bundle.load(
        directory, signing_key="new-key"
    )
    assert len(loaded.predict(features)) == len(features)


def test_resaving_unsigned_removes_a_stale_signature(
    trained, tmp_path, monkeypatch
):
    monkeypatch.delenv(scmle.contracts.signing.SIGNING_KEY_ENV, raising=False)
    bundle = trained[0]
    directory = bundle.save(tmp_path / "m", signing_key="k")
    bundle.save(directory)
    assert not (directory / scmle.artifacts.bundle.SIGNATURE_FILE).exists()


def test_default_policy_requires_exact_column_names(trained):
    # Enforces docs/GLOSSARY.md:588.
    # Enforces docs/WHEN_THINGS_GO_WRONG.md:482.
    # Enforces docs/START_HERE.md:541.
    # Enforces website/src/learn/glossary.html:294.
    # Enforces website/src/learn/start-here.html:279.
    # Enforces website/src/learn/when-things-go-wrong.html:204.
    bundle, features = trained
    renamed = features.rename(columns={"num": "NUM"})
    with pytest.raises(errors.SchemaError, match="missing"):
        bundle.predict(renamed)


def test_explicit_renames_map_incoming_names(trained):
    # Enforces docs/START_HERE.md:541.
    # Enforces website/src/learn/start-here.html:279.
    bundle, features = trained
    renamed = features.rename(columns={"num": "customer_age", "cat": "segment"})
    mapping = scmle.inference.mapping.Mapper(
        {"customer_age": "num", "segment": "cat"}
    )
    numpy.testing.assert_array_equal(
        bundle.predict(renamed, mapping), bundle.predict(features)
    )
    report = bundle.validate(renamed, mapping)
    assert report.valid and report.extra == ()


def test_case_and_whitespace_insensitive_matching_is_opt_in(trained):
    # Enforces docs/WHEN_THINGS_GO_WRONG.md:482.
    # Enforces website/src/learn/when-things-go-wrong.html:204.
    bundle, features = trained
    messy = features.rename(columns={"num": "  NUM ", "cat": "Cat"})
    with pytest.raises(errors.SchemaError):
        bundle.predict(messy)
    loose = scmle.inference.mapping.Mapper(insensitive=True, whitespace=True)
    numpy.testing.assert_array_equal(
        bundle.predict(messy, loose), bundle.predict(features)
    )
    only_case = scmle.inference.mapping.Mapper(insensitive=True)
    with pytest.raises(errors.SchemaError):  # whitespace still differs
        bundle.predict(messy, only_case)


def test_ambiguous_or_unknown_mappings_are_rejected(trained):
    bundle, features = trained
    clash = features.rename(columns={"num": "NUM"}).assign(Num=1.0)
    with pytest.raises(errors.SchemaError, match="ambiguous"):
        bundle.predict(clash, scmle.inference.mapping.Mapper(insensitive=True))
    double = features.assign(other=features["num"]).rename(columns={"num": "a"})
    with pytest.raises(errors.SchemaError, match="ambiguous"):
        bundle.predict(
            double,
            scmle.inference.mapping.Mapper({"a": "num", "other": "num"}),
        )
    with pytest.raises(errors.SchemaError, match="not in the model schema"):
        bundle.predict(
            features, scmle.inference.mapping.Mapper({"num": "ghost"})
        )


def test_exact_matches_are_never_renamed(trained):
    bundle, features = trained
    mapping = scmle.inference.mapping.Mapper({"num": "cat"}, insensitive=True)
    # "num" already exists exactly, so the rename request is ignored.
    numpy.testing.assert_array_equal(
        bundle.predict(features, mapping), bundle.predict(features)
    )


def test_round_trip_verification_passes_and_reports_the_tolerance(
    trained, tmp_path
):
    bundle, features = trained
    bundle.save(tmp_path / "m")
    record = bundle.roundtrip(tmp_path / "m", features.head(50))
    assert record["rows"] == 50
    assert record["largest_difference"] <= record["tolerance"]


def test_round_trip_verification_catches_a_bundle_that_drifted(
    trained, tmp_path
):
    bundle, features = trained
    bundle.save(tmp_path / "m")
    other = scmle.artifacts.bundle.Bundle.load(tmp_path / "m", trusted=True)
    flipped = features.assign(num=-features.num)
    other.pipeline = other.pipeline.fit(flipped, (flipped.num > 0).astype(int))
    with pytest.raises(errors.ArtifactError, match="does not reproduce"):
        other.roundtrip(tmp_path / "m", features.head(50))


def test_inference_time_calibration_is_optional_persisted_and_verified(
    trained, tmp_path
):
    bundle, features = trained
    labels = (features.num > 0.3).astype(int)  # a noisier target than trained
    plain = bundle.predict_proba(features)
    report = bundle.calibrate(features, labels, method="sigmoid")
    calibrated = bundle.predict_proba(features)
    assert not numpy.allclose(plain, calibrated)
    numpy.testing.assert_allclose(calibrated.sum(axis=1), 1.0, atol=1e-9)
    assert report["brier_after"] <= report["brier_before"] + 1e-9
    assert bundle.info.calibration.startswith("sigmoid on 200 rows")
    bundle.save(tmp_path / "m")
    loaded = scmle.artifacts.bundle.Bundle.load(tmp_path / "m", trusted=True)
    numpy.testing.assert_allclose(loaded.predict_proba(features), calibrated)
    assert bundle.roundtrip(tmp_path / "m", features)["rows"] == 200
    (tmp_path / "m" / scmle.artifacts.bundle.CALIBRATOR_FILE).write_bytes(
        b"broken"
    )
    with pytest.raises(errors.ArtifactError, match="calibrator"):
        scmle.artifacts.bundle.Bundle.load(tmp_path / "m", trusted=True)


def test_calibration_rejects_unknown_methods_and_regression(trained):
    bundle, features = trained
    with pytest.raises(errors.ConfigError, match="unknown calibration"):
        bundle.calibrate(features, (features.num > 0).astype(int), "magic")
    bundle.schema = dataclasses.replace(bundle.schema, classes=None)
    with pytest.raises(errors.ArtifactError, match="only classifiers"):
        bundle.calibrate(features, features.num)


def test_compatibility_checks_every_recorded_library():
    info = scmle.artifacts.bundle.Manifest("m", "v1", "roc_auc")
    assert info.incompatibilities() == []
    older = dataclasses.replace(
        info, libraries={**info.libraries, "numpy": "1.0.0"}
    )
    assert any("numpy 1.0.0" in p for p in older.incompatibilities())
    pandas_off = dataclasses.replace(info, pandas="0.1.0")
    assert any("pandas" in p for p in pandas_off.incompatibilities())


def test_score_returns_the_structured_inference_result(trained):
    bundle, features = trained
    messy = features.head(5).assign(junk=1)
    result = bundle.score(messy)
    assert len(result["predictions"]) == 5
    assert result["model_version"] == "v1" and result["schema_valid"] is True
    assert result["diagnostics"]["ignored_columns"] == ["junk"]
    assert result["classes"] == [0, 1]
    assert numpy.allclose(numpy.sum(result["probabilities"], axis=1), 1.0)
    with pytest.raises(errors.SchemaError):
        bundle.score(features.drop(columns="num"))


def test_the_environment_lock_pins_the_build_environment(trained, tmp_path):
    bundle = trained[0]
    path = tmp_path / "lock.txt"

    text = bundle.environment_lock(path)

    assert path.read_text() == text
    assert text.startswith("# scmle environment lock")
    assert f"# python {platform.python_version()}" in text
    assert f"numpy=={numpy.__version__}" in text
    assert "# platform " in text and "unknown" not in text.splitlines()[2]
    names = [line.split("==")[0] for line in text.splitlines() if "==" in line]
    assert names == sorted(names)


def test_the_manifest_traces_the_bundle_to_its_inputs(tmp_path):
    # ML-10 / C6: every bundle names the seed, configuration, data, split,
    # code revision and math-library threads it was built with.
    rng = numpy.random.default_rng(0)
    features = pandas.DataFrame(
        {"num": rng.normal(size=200), "cat": rng.choice(["a", "b"], 200)}
    )
    target = pandas.Series((features.num > 0).astype(int))
    pipe = preprocessing.instantiate(
        PLAN,
        scmle.models.registry.Catalog.stock().get("random_forest"),
        tasks.Task.BINARY,
        0,
    ).fit(features, target)
    bundle = scmle.artifacts.bundle.Bundle.create(
        pipe,
        PLAN,
        features,
        "y",
        "binary_classification",
        "roc_auc",
        "rf",
        seed=7,
        configuration="c" * 64,
        dataset="d" * 64,
        split="s" * 64,
    )
    bundle.save(tmp_path / "m")
    info = scmle.artifacts.bundle.Bundle.load(tmp_path / "m", trusted=True).info

    assert (info.seed, info.configuration, info.dataset, info.split) == (
        7,
        "c" * 64,
        "d" * 64,
        "s" * 64,
    )
    assert len(info.commit) == 40  # this checkout is a git repository
    assert info.threads and all(n >= 1 for n in info.threads.values())


def test_the_lock_survives_saving_and_loading(trained, tmp_path):
    bundle = trained[0]
    bundle.save(tmp_path / "m")

    loaded = scmle.artifacts.bundle.Bundle.load(tmp_path / "m", trusted=True)

    assert loaded.environment_lock() == bundle.environment_lock()


def test_a_version_mismatch_arrives_with_the_lock_attached(trained, tmp_path):
    bundle = trained[0]
    bundle.save(tmp_path / "m")
    manifest = tmp_path / "m" / scmle.artifacts.bundle.MANIFEST_FILE
    recorded = json.loads(manifest.read_text())
    recorded["sklearn"] = "0.1.0"
    manifest.write_text(json.dumps(recorded))

    with pytest.raises(errors.ArtifactError, match="scikit-learn") as caught:
        scmle.artifacts.bundle.Bundle.load(tmp_path / "m", trusted=True)

    lock = caught.value.details["environment_lock"]
    assert lock == bundle.environment_lock()
    assert f"numpy=={numpy.__version__}" in lock


def test_an_empty_bytes_key_is_rejected_when_resolved():
    with pytest.raises(ValueError, match="empty"):
        scmle.contracts.signing.keying(b"")
    with pytest.raises(ValueError, match="empty"):
        scmle.contracts.signing.keying("")
