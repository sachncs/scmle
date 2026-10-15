"""Tests for the optional boosting adapters."""

import pytest

from mml.contracts import errors
from mml.contracts import tasks
from mml.models import boosting_adapters
from mml.models import conformance
from mml.models import registry as registry_module


@pytest.mark.parametrize("library", ["lightgbm", "xgboost"])
def test_adapters_pass_conformance_when_installed(library):
    pytest.importorskip(library)
    adapter = registry_module.ModelRegistry.with_builtins().get(library)
    conformance.check_conformance(adapter)


def test_adapters_are_rejected_not_failing_when_library_missing(monkeypatch):
    registry = registry_module.ModelRegistry.with_builtins()
    monkeypatch.setattr(
        boosting_adapters.XgboostAdapter, "requires", ("not_installed_pkg",)
    )
    compatible, rejected = registry.candidates(tasks.TaskType.BINARY)
    assert "missing dependency" in rejected["xgboost"]
    assert "xgboost" not in {a.name for a in compatible}


def test_build_without_library_raises_config_error(monkeypatch):
    monkeypatch.setattr(
        boosting_adapters.XgboostAdapter, "requires", ("not_installed_pkg",)
    )
    with pytest.raises(errors.ConfigError, match="boosting"):
        boosting_adapters.XgboostAdapter().build(tasks.TaskType.BINARY, 0)


def test_params_override_fixed_params_and_seed_is_applied():
    pytest.importorskip("lightgbm")
    estimator = boosting_adapters.LightGbmAdapter().build(
        tasks.TaskType.REGRESSION, 5, {"num_leaves": 9}
    )
    params = estimator.get_params()
    assert params["num_leaves"] == 9 and params["random_state"] == 5
    assert params["verbose"] == -1


def test_end_to_end_with_string_labels(multiclass_frame, tmp_path):
    pytest.importorskip("xgboost")
    pytest.importorskip("lightgbm")
    import support

    project = support.make_project(
        multiclass_frame,
        "y",
        task="classification",
        time_budget=300,
        output_dir=tmp_path,
    )
    result = project.run()
    names = {t.model_name for t in result.trials if t.status == "ok"}
    assert {"lightgbm", "xgboost"} <= names
    assert set(result.predict(multiclass_frame.drop(columns="y"))) <= {
        "lo",
        "mid",
        "hi",
    }


def test_label_encoded_classifier_roundtrips_string_labels():
    import numpy as np
    import pandas as pd
    from sklearn import tree

    from mml.models import wrappers

    rng = np.random.default_rng(0)
    features = pd.DataFrame({"a": rng.normal(size=60)})
    labels = np.where(features.a > 0, "pos", "neg")
    model = wrappers.LabelEncodedClassifier(tree.DecisionTreeClassifier()).fit(
        features, labels
    )
    assert list(model.classes_) == ["neg", "pos"]
    assert set(model.predict(features)) <= {"neg", "pos"}
    assert model.predict_proba(features).shape == (60, 2)
    assert model.score(features, labels) > 0.9


def test_wrapper_survives_pickling_and_cloning():
    import pickle

    from sklearn import base
    from sklearn import tree

    from mml.models import wrappers

    model = wrappers.LabelEncodedClassifier(tree.DecisionTreeClassifier())
    assert base.clone(model).get_params()["estimator"] is not None
    assert pickle.loads(pickle.dumps(model)).estimator is not None
