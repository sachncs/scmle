"""ONNX export reproduces the pipeline or refuses with a reason."""

import dataclasses
import importlib.util

import numpy as np
import pandas as pd
import pytest

from scmle.artifacts import bundle as bundle_module
from scmle.contracts import data as data_contracts
from scmle.contracts import errors
from scmle.contracts import tasks
from scmle.data.curation import preprocessing
from scmle.inference import onnx_export
from scmle.models import registry as registry_module

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("skl2onnx") is None
    or importlib.util.find_spec("onnxruntime") is None,
    reason="the onnx extra is not installed",
)

PLAN = data_contracts.FeaturePlan(("num", "num2"), ("cat",))
REGISTRY = registry_module.ModelRegistry.with_builtins()


def frame(rows=400, seed=0):
    rng = np.random.default_rng(seed)
    features = pd.DataFrame(
        {
            "num": rng.normal(size=rows),
            "num2": rng.normal(size=rows),
            "cat": rng.choice(["a", "b", "c"], rows),
        }
    )
    features.loc[::17, "num"] = np.nan
    return features


def make_bundle(name, task, features=None):
    features = frame() if features is None else features
    if task is tasks.TaskType.REGRESSION:
        target = features.num2 * 3 + (features.cat == "a") * 2
    else:
        target = pd.Series(
            np.where(features.num2 + (features.cat == "a") > 0.5, "yes", "no")
        )
    fitted = preprocessing.build_pipeline(
        PLAN, REGISTRY.get(name), task, 0
    ).fit(features, target)
    return bundle_module.ModelBundle.create(
        fitted, PLAN, features, "y", task.value, "roc_auc", name
    ), features


@pytest.mark.parametrize(
    "name", ["logistic_regression", "random_forest", "decision_tree"]
)
def test_classifiers_match_the_pipeline_within_tolerance(name, tmp_path):
    bundle, features = make_bundle(name, tasks.TaskType.BINARY)

    path = bundle.export_onnx(tmp_path / "m.onnx")
    report = onnx_export.verify(bundle, path, features)

    assert report["max_difference"] <= onnx_export.DEFAULT_TOLERANCE
    assert report["label_agreement"] >= 0.99


def test_hist_gradient_boosting_exports_when_the_converter_supports_it(
    tmp_path,
):
    bundle, features = make_bundle(
        "hist_gradient_boosting", tasks.TaskType.BINARY
    )

    try:
        path = bundle.export_onnx(tmp_path / "m.onnx", check=features)
    except errors.ArtifactError as error:
        assert "hist_gradient_boosting" in str(error)
    else:
        assert path.exists()


def test_a_regressor_matches_on_scaled_values(tmp_path):
    bundle, features = make_bundle("random_forest", tasks.TaskType.REGRESSION)

    path = bundle.export_onnx(tmp_path / "m.onnx", check=features)

    scorer = onnx_export.OnnxScorer(bundle, path)
    np.testing.assert_allclose(
        scorer.predict(features),
        bundle.predict(features),
        rtol=1e-3,
        atol=1e-3,
    )


def test_the_scorer_returns_probabilities_in_class_order(tmp_path):
    bundle, features = make_bundle("logistic_regression", tasks.TaskType.BINARY)
    path = bundle.export_onnx(tmp_path / "m.onnx")

    scorer = onnx_export.OnnxScorer(bundle, path)

    np.testing.assert_allclose(
        scorer.predict_proba(features),
        bundle.predict_proba(features),
        atol=1e-4,
    )
    assert list(bundle.classes) == ["no", "yes"]


def test_models_the_converter_cannot_express_are_refused_by_name(tmp_path):
    bundle = make_bundle("random_forest", tasks.TaskType.BINARY)[0]
    bundle.pipeline = tasks  # Not an estimator at all.

    with pytest.raises(errors.ArtifactError, match="not supported"):
        bundle.export_onnx(tmp_path / "m.onnx")


def test_an_attached_calibrator_blocks_the_export(tmp_path):
    bundle, features = make_bundle("random_forest", tasks.TaskType.BINARY)
    bundle.calibrate(features, bundle.predict(features), method="sigmoid")

    with pytest.raises(errors.ArtifactError, match="calibrator"):
        bundle.export_onnx(tmp_path / "m.onnx")


def test_survival_models_are_refused(tmp_path):
    bundle = make_bundle("random_forest", tasks.TaskType.BINARY)[0]
    bundle.schema = dataclasses.replace(bundle.schema, task="survival")

    with pytest.raises(errors.ArtifactError, match="survival"):
        bundle.export_onnx(tmp_path / "m.onnx")


def test_a_failed_parity_check_names_the_difference(tmp_path):
    bundle, features = make_bundle("logistic_regression", tasks.TaskType.BINARY)
    path = bundle.export_onnx(tmp_path / "m.onnx")

    with pytest.raises(errors.ArtifactError, match="differs"):
        onnx_export.verify(bundle, path, features, tolerance=0.0)


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
    bundle, features = make_bundle(name, tasks.TaskType.BINARY)

    path = bundle.export_onnx(tmp_path / "m.onnx", check=features)

    scorer = onnx_export.OnnxScorer(bundle, path)
    assert set(scorer.predict(features)) <= {"no", "yes"}
    assert (scorer.predict(features) == bundle.predict(features)).mean() >= 0.99


@needs_boosters
@pytest.mark.parametrize("name", ["lightgbm", "xgboost"])
def test_boosted_regressors_export(name, tmp_path):
    bundle, features = make_bundle(name, tasks.TaskType.REGRESSION)

    path = bundle.export_onnx(tmp_path / "m.onnx", check=features)

    assert path.stat().st_size > 0


def test_a_refusal_points_to_the_service_package(tmp_path):
    bundle = make_bundle("random_forest", tasks.TaskType.BINARY)[0]
    bundle.pipeline = object()

    with pytest.raises(errors.ArtifactError, match="export_service"):
        bundle.export_onnx(tmp_path / "m.onnx")
