"""ONNX export reproduces the pipeline or refuses with a reason."""

import dataclasses
import importlib.util
import json

import numpy
import pandas
import pytest

import scmle.artifacts.bundle
from scmle.contracts import data
from scmle.contracts import errors
from scmle.contracts import tasks
from scmle.inference import onnx_export
from scmle.models import registry
from scmle.preparation import preprocessing

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("skl2onnx") is None
    or importlib.util.find_spec("onnxruntime") is None,
    reason="the onnx extra is not installed",
)

PLAN = data.Blueprint(("num", "num2"), ("cat",))
REGISTRY = registry.Catalog.stock()


def frame(rows=400, seed=0):
    rng = numpy.random.default_rng(seed)
    features = pandas.DataFrame(
        {
            "num": rng.normal(size=rows),
            "num2": rng.normal(size=rows),
            "cat": rng.choice(["a", "b", "c"], rows),
        }
    )
    features.loc[::17, "num"] = numpy.nan
    return features


def make_bundle(name, task, features=None):
    features = frame() if features is None else features
    if task is tasks.Task.REGRESSION:
        target = features.num2 * 3 + (features.cat == "a") * 2
    else:
        target = pandas.Series(
            numpy.where(
                features.num2 + (features.cat == "a") > 0.5, "yes", "no"
            )
        )
    fitted = preprocessing.instantiate(PLAN, REGISTRY.get(name), task, 0).fit(
        features, target
    )
    return scmle.artifacts.bundle.Bundle.create(
        fitted, PLAN, features, "y", task.value, "roc_auc", name
    ), features


@pytest.mark.parametrize(
    "name", ["logistic_regression", "random_forest", "decision_tree"]
)
def test_classifiers_match_the_pipeline_within_tolerance(name, tmp_path):
    bundle, features = make_bundle(name, tasks.Task.BINARY)

    path = bundle.export_onnx(tmp_path / "m.onnx", check=features)
    report = onnx_export.verify(bundle, path, features)

    assert report["max_difference"] <= onnx_export.DEFAULT_TOLERANCE
    assert report["label_agreement"] >= 0.99


def test_hist_gradient_boosting_exports_when_the_converter_supports_it(
    tmp_path,
):
    bundle, features = make_bundle("hist_gradient_boosting", tasks.Task.BINARY)

    refusal = None
    path = None
    try:
        path = bundle.export_onnx(tmp_path / "m.onnx", check=features)
    except errors.ArtifactError as error:
        refusal = error
    if refusal is not None:
        assert "hist_gradient_boosting" in str(refusal)
    else:
        assert path.exists()


def test_a_regressor_matches_on_scaled_values(tmp_path):
    bundle, features = make_bundle("random_forest", tasks.Task.REGRESSION)

    path = bundle.export_onnx(tmp_path / "m.onnx", check=features)

    assert path.exists()
    assert onnx_export.verify(bundle, path, features)["max_difference"] < 1e-3


def test_the_exported_classifier_agrees_in_class_order(tmp_path):
    bundle, features = make_bundle("logistic_regression", tasks.Task.BINARY)
    path = bundle.export_onnx(tmp_path / "m.onnx", check=features)

    result = onnx_export.verify(bundle, path, features)

    assert result["label_agreement"] == 1.0
    assert list(bundle.classes) == ["no", "yes"]


def test_models_the_converter_cannot_express_are_refused_by_name(tmp_path):
    bundle, features = make_bundle("random_forest", tasks.Task.BINARY)
    bundle.pipeline = tasks  # Not an estimator at all.

    with pytest.raises(errors.ArtifactError, match="not supported"):
        bundle.export_onnx(tmp_path / "m.onnx", check=features)


def test_an_attached_calibrator_blocks_the_export(tmp_path):
    bundle, features = make_bundle("random_forest", tasks.Task.BINARY)
    bundle.calibrate(features, bundle.predict(features), method="sigmoid")

    with pytest.raises(errors.ArtifactError, match="calibrator"):
        bundle.export_onnx(tmp_path / "m.onnx", check=features)


def test_survival_models_are_refused(tmp_path):
    bundle, features = make_bundle("random_forest", tasks.Task.BINARY)
    bundle.schema = dataclasses.replace(bundle.schema, task="survival")

    with pytest.raises(errors.ArtifactError, match="survival"):
        bundle.export_onnx(tmp_path / "m.onnx", check=features)


def test_a_failed_parity_check_names_the_difference(tmp_path):
    bundle, features = make_bundle("logistic_regression", tasks.Task.BINARY)
    path = bundle.export_onnx(tmp_path / "m.onnx", unverified=True)

    with pytest.raises(errors.BlockedError, match="differs"):
        onnx_export.verify(bundle, path, features, tolerance=0.0)


def test_an_export_without_held_out_rows_is_blocked_by_default(tmp_path):
    bundle = make_bundle("logistic_regression", tasks.Task.BINARY)[0]

    with pytest.raises(errors.BlockedError, match="at least"):
        bundle.export_onnx(tmp_path / "m.onnx")

    assert not (tmp_path / "m.onnx").exists()


def test_a_diverging_export_is_blocked_and_its_graph_removed(tmp_path):
    bundle, features = make_bundle("logistic_regression", tasks.Task.BINARY)

    with pytest.raises(errors.BlockedError, match="differs"):
        bundle.export_onnx(tmp_path / "m.onnx", check=features, tolerance=0.0)

    assert not (tmp_path / "m.onnx").exists()


def test_the_manifest_records_the_check_or_its_skipping(tmp_path):
    bundle, features = make_bundle("logistic_regression", tasks.Task.BINARY)

    checked = bundle.export_onnx(tmp_path / "a.onnx", check=features)
    skipped = bundle.export_onnx(tmp_path / "b.onnx", unverified=True)

    def read(path):
        manifest = path.with_name(path.name + onnx_export.MANIFEST)
        return json.loads(manifest.read_text())

    assert read(checked)["verified"] is True
    assert read(checked)["rows"] == len(features)
    assert read(skipped)["verified"] is False
    assert read(skipped)["rows"] is None
    assert read(skipped)["graph_sha256"].startswith("sha256:")


def test_rows_and_unverified_together_are_a_config_error(tmp_path):
    bundle, features = make_bundle("logistic_regression", tasks.Task.BINARY)

    with pytest.raises(errors.ConfigError, match="not both"):
        bundle.export_onnx(tmp_path / "m.onnx", check=features, unverified=True)


needs_boosters = pytest.mark.skipif(
    any(
        importlib.util.find_spec(name) is None
        for name in ("onnxmltools", "lightgbm", "xgboost")
    ),
    reason="onnxmltools, lightgbm and xgboost are not all installed",
)


@needs_boosters
@pytest.mark.parametrize("name", ["lightgbm", "xgboost"])
def test_boosted_models_export_and_decode_their_string_labels(name, tmp_path):
    bundle, features = make_bundle(name, tasks.Task.BINARY)

    path = bundle.export_onnx(tmp_path / "m.onnx", check=features)

    result = onnx_export.verify(bundle, path, features)
    assert result["label_agreement"] >= 0.99


@needs_boosters
@pytest.mark.parametrize("name", ["lightgbm", "xgboost"])
def test_boosted_regressors_export(name, tmp_path):
    bundle, features = make_bundle(name, tasks.Task.REGRESSION)

    path = bundle.export_onnx(tmp_path / "m.onnx", check=features)

    assert path.stat().st_size > 0


def test_a_refusal_points_to_the_service_package(tmp_path):
    bundle, features = make_bundle("random_forest", tasks.Task.BINARY)
    bundle.pipeline = object()

    with pytest.raises(errors.ArtifactError, match="scaffold"):
        bundle.export_onnx(tmp_path / "m.onnx", check=features)


def test_nan_outputs_do_not_pass_the_parity_check(monkeypatch, tmp_path):
    import types

    class Pipeline:
        def predict(self, rows):
            return numpy.zeros(len(rows))

    class Session:
        def __init__(self, path):
            pass

        def run(self, names, inputs):
            return [numpy.full(onnx_export.ROWS, numpy.nan)]

    bundle = types.SimpleNamespace(
        classes=None,
        pipeline=Pipeline(),
        prepare=lambda rows: rows,
    )
    monkeypatch.setattr(
        onnx_export,
        "load",
        lambda: (None, types.SimpleNamespace(InferenceSession=Session)),
    )
    monkeypatch.setattr(onnx_export, "feed", lambda b, rows: {})
    with pytest.raises(errors.BlockedError, match="differs"):
        onnx_export.verify(
            bundle,
            tmp_path / "m.onnx",
            pandas.DataFrame({"a": range(onnx_export.ROWS)}),
        )
