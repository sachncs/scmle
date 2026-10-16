"""Tests for the optional boosting adapters."""

import pytest

from mml.contracts import errors
from mml.contracts import tasks
from mml.models import conformance
from mml.models import registry as registry_module
from mml.models.boosting import adapters as boosting_adapters
from mml.models.boosting import early_stopping


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
    assert params["estimator__num_leaves"] == 9
    assert params["estimator__random_state"] == 5
    assert params["estimator__verbose"] == -1
    assert params["rounds"] == early_stopping.DEFAULT_ROUNDS


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


def test_catboost_passes_conformance_when_installed():
    pytest.importorskip("catboost")
    adapter = registry_module.ModelRegistry.with_builtins().get("catboost")
    conformance.check_conformance(adapter)


def test_catboost_handles_string_labels_natively(multiclass_frame, tmp_path):
    pytest.importorskip("catboost")
    import support

    from mml.contracts import tasks

    only = tasks.StrategySpec(top_models=0)
    project = support.make_project(
        multiclass_frame,
        "y",
        task="classification",
        time_budget=600,
        output_dir=tmp_path,
        strategies=only,
    )
    plan = project.plan()
    assert "catboost" in plan.candidate_models
    result = project.run()
    assert {"catboost"} <= {
        t.model_name for t in result.trials if t.status == "ok"
    }
    assert set(result.predict(multiclass_frame.drop(columns="y"))) <= {
        "lo",
        "mid",
        "hi",
    }


def test_flat_prediction_wrapper_flattens_column_vectors():
    import numpy as np
    import pandas as pd
    from sklearn import tree

    from mml.models import wrappers

    class ColumnVectorTree(tree.DecisionTreeClassifier):
        def predict(self, features):
            return super().predict(features).reshape(-1, 1)

    rng = np.random.default_rng(0)
    features = pd.DataFrame({"a": rng.normal(size=80)})
    target = pd.Series((features.a > 0).astype(int))
    plain = ColumnVectorTree().fit(features, target)
    assert plain.predict(features).ndim == 2
    wrapped = wrappers.FlatPredictionClassifier(ColumnVectorTree()).fit(
        features, target
    )
    assert wrapped.predict(features).shape == (80,)
    assert wrapped.predict_proba(features).shape == (80, 2)
    assert list(wrapped.classes_) == [0, 1]


def test_catboost_multiclass_predictions_are_one_dimensional(multiclass_frame):
    pytest.importorskip("catboost")
    from mml.contracts import tasks

    adapter = registry_module.ModelRegistry.with_builtins().get("catboost")
    estimator = adapter.build(tasks.TaskType.MULTICLASS, 0, {"iterations": 20})
    features = multiclass_frame.drop(columns="y")
    estimator.fit(features, multiclass_frame["y"])
    assert estimator.predict(features).shape == (len(features),)
