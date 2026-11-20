"""The parts of a model bundle, module by module.

Covers the layout and checksums, the manifest signature, the environment
probes, the archive (save, verify, load, re-sign), the core predictions, the
manifest builder and Brier score, the outputs, the intervals and the
recourse goals.
"""

import hashlib
import hmac
import io
import json
import pathlib

import joblib
import numpy
import pandas
import pytest

from scmle.artifacts.bundle import archive
from scmle.artifacts.bundle import core
from scmle.artifacts.bundle import environment
from scmle.artifacts.bundle import explanation
from scmle.artifacts.bundle import intervals
from scmle.artifacts.bundle import layout
from scmle.artifacts.bundle import model
from scmle.artifacts.bundle import outputs
from scmle.artifacts.bundle import seal
from scmle.contracts import data
from scmle.contracts import errors
from scmle.contracts import tasks
import scmle.evaluation.conformal
import scmle.models.registry
from scmle.preparation import preprocessing

PLAN = data.Blueprint(("num",), ("cat",))
KEY = b"0123456789abcdef0123456789abcdef"


def table(rows=120, seed=0):
    rng = numpy.random.default_rng(seed)
    return pandas.DataFrame(
        {"num": rng.normal(size=rows), "cat": rng.choice(["a", "b"], rows)}
    )


@pytest.fixture
def binary():
    features = table()
    target = pandas.Series((features.num > 0).astype(int))
    adapter = scmle.models.registry.Catalog.stock().get("logistic_regression")
    pipe = preprocessing.instantiate(PLAN, adapter, tasks.Task.BINARY, 0).fit(
        features, target
    )
    bundle = model.Bundle.create(
        pipe, PLAN, features, "y", "binary_classification", "roc_auc", "lr"
    )
    return bundle, features, target


@pytest.fixture
def regressor():
    features = table()
    target = pandas.Series(2.0 * features.num + 1.0)
    adapter = scmle.models.registry.Catalog.stock().get("ridge")
    pipe = preprocessing.instantiate(
        PLAN, adapter, tasks.Task.REGRESSION, 0
    ).fit(features, target)
    bundle = model.Bundle.create(
        pipe, PLAN, features, "y", "regression", "rmse", "ridge"
    )
    return bundle, features, target


def test_a_shasum_is_the_sha256_of_the_whole_file(tmp_path):
    path = tmp_path / "f.bin"
    payload = b"x" * ((1 << 20) + 7)
    path.write_bytes(payload)
    assert layout.shasum(path) == hashlib.sha256(payload).hexdigest()


def test_the_layout_names_the_format_and_every_member():
    assert layout.BUNDLE_FORMAT == 6
    assert layout.MODEL_FILE == "pipeline.joblib"
    assert layout.SCHEMA_FILE == "schema.json"
    assert layout.MANIFEST_FILE == "model_manifest.json"
    assert layout.CONFORMAL_FILE == "conformal.json"
    assert layout.HISTORY_FILE == "series_history.parquet"


def test_a_signature_is_the_hmac_sha256_of_the_manifest():
    expected = hmac.new(KEY, b"manifest", hashlib.sha256).hexdigest()
    assert seal.sign(b"manifest", KEY) == expected
    assert seal.SIGNATURE_ALGORITHM == "hmac-sha256"
    assert seal.SIGNATURE_FILE == "signature.json"


def signed(tmp_path, record):
    (tmp_path / layout.MANIFEST_FILE).write_bytes(b"{}")
    (tmp_path / seal.SIGNATURE_FILE).write_text(
        json.dumps(record), encoding="utf-8"
    )
    return tmp_path


def test_an_unsigned_bundle_or_a_missing_key_does_not_verify(tmp_path):
    assert seal.verify_signature(tmp_path, KEY) is False
    signed(tmp_path, {"algorithm": "hmac-sha256", "signature": "x"})
    assert seal.verify_signature(tmp_path, None) is False


def test_a_matching_signature_verifies(tmp_path):
    root = signed(
        tmp_path,
        {"algorithm": "hmac-sha256", "signature": seal.sign(b"{}", KEY)},
    )
    assert seal.verify_signature(root, KEY) is True


def test_a_signature_of_other_bytes_is_refused(tmp_path):
    root = signed(tmp_path, {"algorithm": "hmac-sha256", "signature": "0"})
    with pytest.raises(errors.ArtifactError, match="signature is invalid"):
        seal.verify_signature(root, KEY)


def test_an_unknown_signature_algorithm_is_named(tmp_path):
    root = signed(tmp_path, {"algorithm": "md5", "signature": "0"})
    with pytest.raises(errors.ArtifactError, match="algorithm 'md5'"):
        seal.verify_signature(root, KEY)


def test_unreadable_signature_metadata_is_refused(tmp_path):
    (tmp_path / seal.SIGNATURE_FILE).write_text("[1]", encoding="utf-8")
    with pytest.raises(errors.ArtifactError, match="invalid bundle signature"):
        seal.verify_signature(tmp_path, KEY)
    (tmp_path / seal.SIGNATURE_FILE).write_text("{", encoding="utf-8")
    with pytest.raises(errors.ArtifactError, match="invalid bundle signature"):
        seal.verify_signature(tmp_path, KEY)


def test_the_inventory_is_sorted_and_normalised():
    found = environment.inventory()
    assert list(found) == sorted(found)
    assert "numpy" in found
    assert found["numpy"] == numpy.__version__
    assert all(name == name.lower() and "_" not in name for name in found)


def test_detect_reports_each_importable_tracked_library():
    found = environment.detect()
    assert found["numpy"] == numpy.__version__
    assert set(found) <= set(environment.TRACKED_LIBRARIES)
    assert environment.TRACKED_LIBRARIES[0] == "numpy"


def test_detect_leaves_out_a_library_that_cannot_be_imported(monkeypatch):
    monkeypatch.setattr(
        environment, "TRACKED_LIBRARIES", ("numpy", "not_a_module_xyz")
    )
    assert environment.detect() == {"numpy": numpy.__version__}


def test_pools_report_sorted_thread_counts_and_the_blas(monkeypatch):
    info = [
        {"internal_api": "openmp", "num_threads": 4, "user_api": "openmp"},
        {"internal_api": "mkl", "num_threads": 2, "user_api": "blas"},
        {"internal_api": "openblas", "num_threads": 3, "user_api": "blas"},
    ]
    monkeypatch.setattr(
        environment.threadpoolctl, "threadpool_info", lambda: info
    )
    threads, blas = environment.pools()
    assert threads == {"mkl": 2, "openblas": 3, "openmp": 4}
    assert list(threads) == ["mkl", "openblas", "openmp"]
    assert blas == "mkl"
    monkeypatch.setattr(
        environment.threadpoolctl, "threadpool_info", lambda: info[:1]
    )
    assert environment.pools() == ({"openmp": 4}, "")


def test_revision_is_a_full_commit_or_empty():
    commit = environment.revision()
    assert commit == "" or len(commit) == environment.COMMIT_LENGTH
    assert environment.COMMIT_LENGTH == 40
    assert environment.GIT_SECONDS == 5.0


def test_revision_is_empty_without_git(monkeypatch):
    monkeypatch.setattr(environment.shutil, "which", lambda name: None)
    assert environment.revision() == ""


def test_revision_is_empty_when_git_fails(monkeypatch):
    class Done:
        returncode = 1
        stdout = "a" * 40

    monkeypatch.setattr(environment.shutil, "which", lambda name: "git")
    monkeypatch.setattr(environment.pathlib.Path, "exists", lambda self: True)
    monkeypatch.setattr(
        environment.subprocess, "run", lambda *args, **kwargs: Done()
    )
    assert environment.revision() == ""
    Done.returncode = 0
    assert environment.revision() == "a" * 40
    Done.stdout = "a" * 39
    assert environment.revision() == ""


def test_revision_is_empty_when_git_hangs(monkeypatch):
    def hang(*args, **kwargs):
        raise environment.subprocess.TimeoutExpired("git", 5)

    monkeypatch.setattr(environment.shutil, "which", lambda name: "git")
    monkeypatch.setattr(environment.pathlib.Path, "exists", lambda self: True)
    monkeypatch.setattr(environment.subprocess, "run", hang)
    assert environment.revision() == ""


def test_brier_is_zero_for_certainty_and_two_for_confident_error():
    probabilities = numpy.array([[1.0, 0.0], [0.0, 1.0]])
    labels = numpy.array([0, 1])
    assert model.brier(probabilities, labels, (0, 1)) == 0.0
    assert model.brier(probabilities, labels[::-1], (0, 1)) == 2.0
    half = numpy.array([[0.5, 0.5]])
    assert model.brier(half, numpy.array([1]), (0, 1)) == 0.5


def test_describe_records_the_columns_classes_and_categories(binary):
    bundle = binary[0]
    schema = bundle.schema
    assert schema.numeric == ("num",)
    assert schema.categorical == ("cat",)
    assert schema.categories == {"cat": ["a", "b"]}
    assert schema.classes == (0, 1)
    assert schema.target == "y"
    assert schema.task == "binary_classification"
    assert schema.digests == {}
    assert schema.optional == ()


def test_describe_caps_the_known_categories(monkeypatch, binary):
    bundle, features = binary[0], binary[1]
    monkeypatch.setattr(model, "MAX_KNOWN_CATEGORIES", 1)
    schema = model.describe(
        bundle.pipeline, PLAN, features, "y", "binary", None, "g", None, None
    )
    assert schema.categories == {"cat": ["a"]}
    assert schema.digests == {"graph": "g"}


def test_describe_prefers_the_ordinal_labels(binary):
    bundle, features = binary[0], binary[1]
    schema = model.describe(
        bundle.pipeline,
        PLAN,
        features,
        "y",
        "ordinal",
        ("low", "high"),
        "",
        "g",
        "t",
    )
    assert schema.classes == ("low", "high")
    assert (schema.group, schema.time) == ("g", "t")


def test_stamp_records_the_build_and_its_inputs():
    info = model.stamp(
        "lr", "v2", "auc", None, "s", 0.4, 7, "cfg", "data", "split"
    )
    assert (info.model_name, info.model_version, info.metric) == (
        "lr",
        "v2",
        "auc",
    )
    assert info.model_params == {}
    assert info.strategy == "s"
    assert info.decision_threshold == 0.4
    assert info.seed == 7
    assert (info.configuration, info.dataset, info.split) == (
        "cfg",
        "data",
        "split",
    )
    assert info.environment["numpy"] == numpy.__version__


def test_core_predicts_through_the_pipeline_in_schema_order(binary):
    bundle, features = binary[0], binary[1]
    shuffled = features[["cat", "num"]].assign(extra=1)
    prepared = bundle.prepare(shuffled)
    assert list(prepared.columns) == ["num", "cat"]
    numpy.testing.assert_array_equal(
        bundle.predict(shuffled), bundle.pipeline.predict(features)
    )
    assert bundle.features == ["num", "cat"]
    assert bundle.classes == (0, 1)
    assert isinstance(bundle, core.Core)


def test_core_reads_a_table_from_a_path(binary, tmp_path):
    bundle, features = binary[0], binary[1]
    features.to_csv(tmp_path / "x.csv", index=False)
    frame = bundle.read_frame(tmp_path / "x.csv")
    assert list(frame.columns) == ["num", "cat"]
    assert len(frame) == len(features)
    copy = bundle.read_frame(features)
    assert copy is not features


def test_a_regressor_gives_no_probabilities(regressor):
    bundle, features = regressor[0], regressor[1]
    with pytest.raises(errors.ArtifactError, match="does not provide"):
        bundle.predict_proba(features)


def test_an_attached_calibrator_supplies_the_probabilities(binary):
    bundle, features = binary[0], binary[1]

    class Fixed:
        def predict_proba(self, frame):
            return numpy.full((len(frame), 2), 0.5)

    bundle.calibrator = Fixed()
    assert bundle.predict_proba(features.head(3)).tolist() == [[0.5, 0.5]] * 3


def test_a_bundle_without_a_detector_cannot_flag_rows(binary):
    bundle, features = binary[0], binary[1]
    with pytest.raises(errors.ArtifactError, match="no novelty detector"):
        bundle.atypical(features)


def test_remember_does_nothing_without_series(binary):
    bundle, features = binary[0], binary[1]
    bundle.remember(features)
    assert bundle.history is None


def test_the_ordinal_codes_map_back_to_labels(binary):
    bundle, features = binary[0], binary[1]
    bundle.schema = model.describe(
        bundle.pipeline,
        PLAN,
        features,
        "y",
        tasks.Task.ORDINAL.value,
        ("low", "high"),
        "",
        None,
        None,
    )
    expected = numpy.where(
        bundle.pipeline.predict(features) == 1, "high", "low"
    )
    numpy.testing.assert_array_equal(bundle.predict(features), expected)


def test_calibration_reports_brier_and_ece_and_names_the_method(binary):
    bundle, features, target = binary
    report = bundle.calibrate(features, target, "sigmoid")
    assert set(report) == {
        "brier_before",
        "brier_after",
        "ece_before",
        "ece_after",
    }
    assert bundle.info.calibration == "sigmoid on 120 rows"
    assert report["brier_before"] == pytest.approx(
        model.brier(
            bundle.pipeline.predict_proba(features), target.to_numpy(), (0, 1)
        )
    )


def test_calibration_refuses_unknown_methods_and_two_class_dirichlet(binary):
    bundle, features, target = binary
    with pytest.raises(errors.ConfigError, match="unknown calibration"):
        bundle.calibrate(features, target, "platt")
    with pytest.raises(errors.ConfigError, match="three or more classes"):
        bundle.calibrate(features, target, "dirichlet")


def test_a_regressor_cannot_be_calibrated(regressor):
    bundle, features, target = regressor
    with pytest.raises(errors.ArtifactError, match="only classifiers"):
        bundle.calibrate(features, target)


def test_the_saved_bundle_lists_and_checksums_its_members(binary, tmp_path):
    bundle = binary[0]
    root = bundle.save(tmp_path / "b", signing_key=KEY)
    manifest = json.loads((root / layout.MANIFEST_FILE).read_text("utf-8"))
    assert manifest["format"] == layout.BUNDLE_FORMAT
    assert manifest["files"] == {
        layout.MODEL_FILE: layout.shasum(root / layout.MODEL_FILE),
        layout.SCHEMA_FILE: layout.shasum(root / layout.SCHEMA_FILE),
    }
    signature = json.loads((root / seal.SIGNATURE_FILE).read_text("utf-8"))
    assert signature == {
        "algorithm": "hmac-sha256",
        "signature": seal.sign((root / layout.MANIFEST_FILE).read_bytes(), KEY),
    }


def test_a_signed_bundle_loads_without_trust(binary, tmp_path):
    bundle, features = binary[0], binary[1]
    root = bundle.save(tmp_path / "b", signing_key=KEY)
    loaded = model.Bundle.load(root, signing_key=KEY)
    numpy.testing.assert_array_equal(
        loaded.predict(features), bundle.predict(features)
    )
    assert loaded.cv_plus is None
    assert loaded.coverage_report is None
    assert loaded.shift_weights is None


def test_an_unsigned_bundle_is_refused_unless_trusted(binary, tmp_path):
    bundle = binary[0]
    root = bundle.save(tmp_path / "b")
    assert not (root / seal.SIGNATURE_FILE).exists()
    with pytest.raises(errors.ArtifactError, match="unauthenticated bundle"):
        model.Bundle.load(root)
    assert model.Bundle.load(root, trusted=True).info.model_name == "lr"


def test_a_signed_bundle_without_a_key_is_refused_even_trusted(
    binary, tmp_path, monkeypatch
):
    monkeypatch.delenv("SCMLE_SIGNING_KEY", raising=False)
    bundle = binary[0]
    root = bundle.save(tmp_path / "b", signing_key=KEY)
    with pytest.raises(errors.ArtifactError, match="no signing key set"):
        model.Bundle.load(root, trusted=True)


def test_resaving_without_a_key_removes_the_stale_signature(binary, tmp_path):
    bundle = binary[0]
    root = bundle.save(tmp_path / "b", signing_key=KEY)
    bundle.save(root)
    assert not (root / seal.SIGNATURE_FILE).exists()


def test_unpack_explains_a_missing_or_broken_bundle(tmp_path):
    with pytest.raises(errors.ArtifactError, match="not a model bundle"):
        archive.unpack(tmp_path)
    (tmp_path / layout.MANIFEST_FILE).write_text("{", encoding="utf-8")
    (tmp_path / layout.SCHEMA_FILE).write_text("{}", encoding="utf-8")
    with pytest.raises(errors.ArtifactError, match="invalid model bundle"):
        archive.unpack(tmp_path)


def manifest(binary, tmp_path):
    bundle = binary[0]
    root = bundle.save(tmp_path / "b")
    return root, json.loads((root / layout.MANIFEST_FILE).read_text("utf-8"))


def test_admit_refuses_another_format(binary, tmp_path):
    document = manifest(binary, tmp_path)[1]
    info, legacy = archive.admit(document, False, False)
    assert legacy is False
    assert info.model_name == "lr"
    for value in (5, 7, None):
        with pytest.raises(errors.ArtifactError, match=f"format {value}"):
            archive.admit({**document, "format": value}, True, True)


def test_admit_takes_a_trusted_legacy_bundle_only_when_allowed(
    binary, tmp_path
):
    document = manifest(binary, tmp_path)[1]
    legacy = {**document, "format": 1}
    assert archive.admit(legacy, True, True)[1] is True
    for trusted, allowed in ((False, True), (True, False)):
        with pytest.raises(errors.ArtifactError, match="format 1"):
            archive.admit(legacy, trusted, allowed)


def test_admit_refuses_a_bundle_without_a_schema_checksum(binary, tmp_path):
    document = manifest(binary, tmp_path)[1]
    files = {layout.MODEL_FILE: document["files"][layout.MODEL_FILE]}
    with pytest.raises(errors.ArtifactError, match="schema checksum"):
        archive.admit({**document, "files": files}, True, False)


def test_authenticate_detects_a_modified_model_or_schema(binary, tmp_path):
    root, document = manifest(binary, tmp_path)
    info, legacy = archive.admit(document, True, False)
    snapshot = archive.unpack(root)
    archive.authenticate(root, snapshot, info, legacy, True, None)
    shape = root / layout.SCHEMA_FILE
    shape.write_text(shape.read_text("utf-8") + " ", encoding="utf-8")
    changed = archive.unpack(root)
    with pytest.raises(errors.ArtifactError, match="schema file was modified"):
        archive.authenticate(root, changed, info, False, True, None)
    archive.authenticate(root, changed, info, True, True, None)
    (root / layout.MODEL_FILE).write_bytes(b"0")
    with pytest.raises(errors.ArtifactError, match="model file was modified"):
        archive.authenticate(root, snapshot, info, legacy, True, None)
    (root / layout.MODEL_FILE).unlink()
    with pytest.raises(errors.ArtifactError, match=r"missing pipeline\.joblib"):
        archive.authenticate(root, snapshot, info, legacy, True, None)


def test_the_schema_checksum_covers_the_bytes_that_were_parsed(
    binary, tmp_path
):
    root, document = manifest(binary, tmp_path)
    info, legacy = archive.admit(document, True, False)
    snapshot = archive.unpack(root)
    forged = archive.Snapshot(
        snapshot.manifest, snapshot.schema, snapshot.text, b"{}"
    )
    with pytest.raises(errors.ArtifactError, match="schema file was modified"):
        archive.authenticate(root, forged, info, legacy, True, None)


def test_the_signature_covers_the_manifest_bytes_that_were_parsed(
    binary, tmp_path
):
    root = binary[0].save(tmp_path / "b", signing_key=KEY)
    snapshot = archive.unpack(root)
    assert seal.verify_signature(root, KEY, snapshot.text) is True
    with pytest.raises(errors.ArtifactError, match="signature is invalid"):
        seal.verify_signature(root, KEY, snapshot.text + b" ")


def test_unpack_reads_the_manifest_and_schema_bytes_once(binary, tmp_path):
    root, document = manifest(binary, tmp_path)
    snapshot = archive.unpack(root)
    assert snapshot.manifest == document
    assert snapshot.text == (root / layout.MANIFEST_FILE).read_bytes()
    assert snapshot.layout == (root / layout.SCHEMA_FILE).read_bytes()
    (root / layout.MANIFEST_FILE).write_text("[]", encoding="utf-8")
    with pytest.raises(errors.ArtifactError, match="invalid model bundle"):
        archive.unpack(root)


def test_resolve_returns_listed_members_and_refuses_changed_ones(
    binary, tmp_path
):
    root, document = manifest(binary, tmp_path)
    info = archive.admit(document, True, False)[0]
    assert archive.resolve(root, info, layout.NOVELTY_FILE, "x") is None
    assert (
        archive.resolve(root, info, layout.MODEL_FILE, "x")
        == (root / layout.MODEL_FILE).read_bytes()
    )
    (root / layout.MODEL_FILE).write_bytes(b"0")
    with pytest.raises(errors.ArtifactError, match="checksum mismatch: gone"):
        archive.resolve(root, info, layout.MODEL_FILE, "gone")
    (root / layout.MODEL_FILE).unlink()
    with pytest.raises(errors.ArtifactError, match="checksum mismatch: gone"):
        archive.resolve(root, info, layout.MODEL_FILE, "gone")


def test_unpickle_explains_a_member_from_another_build(monkeypatch):
    buffer = io.BytesIO()
    joblib.dump({"a": 1}, buffer)
    assert archive.unpickle(buffer.getvalue(), "m.joblib") == {"a": 1}

    def missing(source):
        raise ModuleNotFoundError("No module named 'gone'")

    monkeypatch.setattr(archive.joblib, "load", missing)
    with pytest.raises(errors.ArtifactError, match=r"m\.joblib names"):
        archive.unpickle(buffer.getvalue(), "m.joblib")


def test_every_optional_member_round_trips(regressor, tmp_path):
    bundle = regressor[0]
    bundle.conformal = scmle.evaluation.conformal.ConformalCalibrator.calibrate(
        numpy.arange(40.0), numpy.arange(40.0) + 1.0
    )
    bundle.coverage_report = {"b": 2, "a": 1}
    bundle.quantile = {"q": 1}
    bundle.cv_plus = {"c": 1}
    bundle.shift_weights = {"w": 1}
    bundle.novelty = {"n": 1}
    bundle.calibrator = {"k": 1}
    bundle.history = pandas.DataFrame({"num": [1.0]})
    root = bundle.save(tmp_path / "b")
    files = json.loads((root / layout.MANIFEST_FILE).read_text("utf-8"))[
        "files"
    ]
    assert list(files) == [
        layout.MODEL_FILE,
        layout.SCHEMA_FILE,
        layout.CALIBRATOR_FILE,
        layout.CONFORMAL_FILE,
        layout.NOVELTY_FILE,
        layout.QUANTILE_FILE,
        layout.CV_PLUS_FILE,
        layout.COVERAGE_FILE,
        layout.SHIFT_WEIGHTS_FILE,
        layout.HISTORY_FILE,
    ]
    assert (root / layout.COVERAGE_FILE).read_text("utf-8") == (
        '{"a": 1, "b": 2}'
    )
    loaded = model.Bundle.load(root, trusted=True)
    assert loaded.quantile == {"q": 1}
    assert loaded.cv_plus == {"c": 1}
    assert loaded.shift_weights == {"w": 1}
    assert loaded.novelty == {"n": 1}
    assert loaded.calibrator == {"k": 1}
    assert loaded.coverage_report == {"a": 1, "b": 2}
    assert loaded.history.to_dict() == {"num": {0: 1.0}}
    assert loaded.conformal.scores == bundle.conformal.scores
    bundle.calibrator = None
    bundle.history = None
    bundle.save(root)
    assert not (root / layout.CALIBRATOR_FILE).exists()
    assert not (root / layout.HISTORY_FILE).exists()


def test_a_modified_optional_member_is_refused_on_load(regressor, tmp_path):
    bundle = regressor[0]
    bundle.coverage_report = {"a": 1}
    root = bundle.save(tmp_path / "b")
    (root / layout.COVERAGE_FILE).write_text("{}", encoding="utf-8")
    with pytest.raises(errors.ArtifactError, match="coverage report was"):
        model.Bundle.load(root, trusted=True)


def test_rotation_re_signs_and_retires_the_old_key(binary, tmp_path):
    bundle = binary[0]
    root = bundle.save(tmp_path / "b", signing_key=KEY)
    other = b"fedcba9876543210fedcba9876543210"
    assert archive.Archive.rotate_signing_key(root, KEY, other) == root
    assert seal.verify_signature(root, other) is True
    with pytest.raises(errors.ArtifactError, match="signature is invalid"):
        seal.verify_signature(root, KEY)
    assert not [p for p in root.iterdir() if p.name.startswith(".signature")]


def test_rotation_refuses_empty_keys_and_unsigned_bundles(binary, tmp_path):
    bundle = binary[0]
    root = bundle.save(tmp_path / "b")
    with pytest.raises(errors.ArtifactError, match="non-empty current"):
        archive.Archive.rotate_signing_key(root, "", KEY)
    with pytest.raises(errors.ArtifactError, match="unauthenticated"):
        archive.Archive.rotate_signing_key(root, KEY, KEY)
    with pytest.raises(errors.ArtifactError, match="incomplete"):
        archive.Archive.rotate_signing_key(tmp_path / "none", KEY, KEY)


def test_roundtrip_reports_rows_tolerance_and_the_difference(binary, tmp_path):
    bundle, features = binary[0], binary[1]
    bundle.save(tmp_path / "b")
    report = bundle.roundtrip(tmp_path / "b", features.head(5))
    assert report == {
        "rows": 5,
        "tolerance": model.PARITY_TOLERANCE,
        "largest_difference": 0.0,
    }


def test_newest_picks_the_latest_row_of_each_series():
    frame = pandas.DataFrame({"e": ["a", "b", "a", "b", "a"]})
    order = numpy.array([1, 5, 3, 5, 2])
    assert outputs.newest(frame, "e", order).tolist() == [2, 3]
    assert outputs.newest(frame, None, order).tolist() == [3]


def test_score_reports_predictions_probabilities_and_diagnostics(binary):
    bundle, features = binary[0], binary[1]
    rows = features.head(2).assign(extra=1)
    result = bundle.score(rows)
    assert result["predictions"] == bundle.predict(rows).tolist()
    assert result["model_version"] == "v1"
    assert result["schema_valid"] is True
    assert result["diagnostics"]["ignored_columns"] == ["extra"]
    assert result["classes"] == [0, 1]
    assert result["probabilities"] == bundle.predict_proba(rows).tolist()
    assert "out_of_distribution" not in result


def test_the_task_specific_outputs_refuse_other_tasks(binary):
    bundle, features = binary[0], binary[1]
    with pytest.raises(errors.ArtifactError, match="not a survival model"):
        bundle.predict_survival(features, [1])
    with pytest.raises(errors.ArtifactError, match="not a forecaster"):
        bundle.forecast(features)
    with pytest.raises(errors.ArtifactError, match="not a ranking model"):
        bundle.rank(features, "q")


def test_rank_orders_rows_within_each_query(regressor):
    bundle, features = regressor[0], regressor[1]
    bundle.schema = model.describe(
        bundle.pipeline,
        PLAN,
        features,
        "y",
        tasks.Task.RANKING.value,
        None,
        "",
        None,
        None,
    )
    rows = pandas.DataFrame(
        {"num": [0.0, 2.0, 1.0, -1.0], "cat": "a", "q": [1, 1, 2, 2]}
    )
    ranked = bundle.rank(rows, "q")
    assert ranked["q"].tolist() == [1, 1, 2, 2]
    assert ranked["rank"].tolist() == [1, 2, 1, 2]
    assert ranked["num"].tolist() == [2.0, 0.0, 1.0, -1.0]
    with pytest.raises(errors.ArtifactError, match="query column 'z'"):
        bundle.rank(rows, "z")


def test_the_environment_lock_is_written_when_asked(binary, tmp_path):
    bundle = binary[0]
    text = bundle.environment_lock(tmp_path / "lock.txt")
    assert (tmp_path / "lock.txt").read_text("utf-8") == text
    assert f"numpy=={numpy.__version__}" in text.splitlines()


def test_residual_intervals_add_the_conformal_margin(regressor):
    bundle, features = regressor[0], regressor[1]
    with pytest.raises(errors.ArtifactError, match="no regression conformal"):
        bundle.predict_interval(features)
    bundle.conformal = scmle.evaluation.conformal.ConformalCalibrator.calibrate(
        numpy.zeros(99), numpy.ones(99)
    )
    frame = bundle.predict_interval(features.head(3), coverage=0.9)
    assert list(frame.columns) == ["prediction", "lower", "upper"]
    numpy.testing.assert_allclose(frame.upper - frame.prediction, 1.0)
    numpy.testing.assert_allclose(frame.prediction - frame.lower, 1.0)


def test_interval_methods_refuse_what_the_bundle_lacks(regressor):
    bundle, features = regressor[0], regressor[1]
    with pytest.raises(errors.ArtifactError, match="'residual' method only"):
        bundle.predict_interval(features, method="quantile", shift=True)
    with pytest.raises(errors.ArtifactError, match="no quantile"):
        bundle.predict_interval(features, method="quantile")
    with pytest.raises(errors.ArtifactError, match="no CV\\+ fold models"):
        bundle.predict_interval(features, method="cv_plus")
    with pytest.raises(errors.ArtifactError, match="unknown interval method"):
        bundle.predict_interval(features, method="box")
    assert isinstance(bundle, intervals.Intervals)


def test_quantile_intervals_always_contain_the_prediction(regressor):
    bundle, features = regressor[0], regressor[1]

    class Quantile:
        def interval(self, frame, coverage):
            middle = numpy.zeros(len(frame))
            return middle + 100.0, middle - 100.0

    bundle.quantile = Quantile()
    frame = bundle.predict_interval(features.head(4), method="quantile")
    numpy.testing.assert_array_equal(frame.lower, frame.prediction)
    numpy.testing.assert_array_equal(frame.upper, frame.prediction)


def test_prediction_sets_need_a_classification_calibrator(binary):
    bundle, features = binary[0], binary[1]
    with pytest.raises(errors.ArtifactError, match="no classification"):
        bundle.predict_set(features)
    bundle.conformal = scmle.evaluation.conformal.ConformalCalibrator.calibrate(
        numpy.zeros(30), numpy.ones(30)
    )
    with pytest.raises(errors.ArtifactError, match="no classification"):
        bundle.predict_set(features)


def test_a_label_goal_tracks_the_wanted_class(binary):
    bundle, features = binary[0], binary[1]
    goal = explanation.Label(bundle, 1)
    assert goal.column == 1
    rows = features.head(6)
    numpy.testing.assert_allclose(
        goal.progress(rows), bundle.predict_proba(rows)[:, 1]
    )
    numpy.testing.assert_array_equal(
        goal.reached(rows), bundle.predict(rows) == 1
    )
    assert explanation.BINARY_CLASSES == 2


def test_a_range_goal_scores_the_distance_outside_it(regressor):
    bundle = regressor[0]
    goal = explanation.Range(bundle, (0.0, 1.0))
    rows = pandas.DataFrame({"num": [-2.0, 0.0, 3.0], "cat": "a"})
    values = bundle.predict(rows).astype(float)
    expected = -(numpy.maximum(0 - values, 0) + numpy.maximum(values - 1, 0))
    numpy.testing.assert_allclose(goal.progress(rows), expected)
    numpy.testing.assert_array_equal(
        goal.reached(rows), (values >= 0) & (values <= 1)
    )
    edge = explanation.Range(bundle, (values[1], values[1]))
    assert edge.reached(rows).tolist() == [False, True, False]
    assert edge.progress(rows)[1] == 0.0


def test_the_bundle_path_helpers_take_paths(binary, tmp_path):
    bundle = binary[0]
    assert isinstance(bundle.save(str(tmp_path / "s")), pathlib.Path)
