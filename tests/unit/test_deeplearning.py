"""Tests for the PyTorch model families."""

import dataclasses
import importlib.util
import io
import subprocess
import sys
import textwrap

import joblib
import numpy as np
import pytest
from scipy import sparse
import support

import scmle
from scmle.contracts import tasks
from scmle.models import conformance
from scmle.models import registry as registry_module
from scmle.models.deeplearning import estimators
from scmle.models.deeplearning import worker

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="torch is not installed"
)

REGISTRY = registry_module.ModelRegistry.with_builtins()
FAST = {"epochs": 60, "patience": 6}


def separable(rows=400, seed=0):
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(rows, 4))
    labels = np.where(features[:, 0] + features[:, 1] > 0, "yes", "no")
    return features.astype(np.float32), labels


def test_classifier_learns_and_keeps_string_labels():
    features, labels = separable()
    model = estimators.TorchClassifier(**FAST).fit(features, labels)

    assert set(model.classes_) == {"no", "yes"}
    assert (model.predict(features) == labels).mean() > 0.9
    probabilities = model.predict_proba(features)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, rtol=1e-5)


def test_regressor_recovers_a_linear_signal_in_original_units():
    rng = np.random.default_rng(1)
    features = rng.normal(size=(400, 3)).astype(np.float32)
    target = 50 + 10 * features[:, 0] - 4 * features[:, 1]

    model = estimators.TorchRegressor(**FAST).fit(features, target)

    residual = target - model.predict(features)
    assert 1 - residual.var() / target.var() > 0.9


def test_residual_architecture_learns_too():
    features, labels = separable()
    model = estimators.TorchClassifier(
        architecture="residual", hidden_sizes=(32, 32), **FAST
    ).fit(features, labels)

    assert (model.predict(features) == labels).mean() > 0.9


def test_same_seed_gives_the_same_network():
    features, labels = separable()
    first = estimators.TorchClassifier(random_state=3, **FAST).fit(
        features, labels
    )
    second = estimators.TorchClassifier(random_state=3, **FAST).fit(
        features, labels
    )

    np.testing.assert_allclose(
        first.predict_proba(features), second.predict_proba(features)
    )


def test_pickling_keeps_predictions_and_leaves_the_live_model_intact():
    features, labels = separable()
    model = estimators.TorchClassifier(**FAST).fit(features, labels)
    before = model.predict_proba(features)

    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    buffer.seek(0)
    restored = joblib.load(buffer)

    np.testing.assert_allclose(restored.predict_proba(features), before)
    np.testing.assert_allclose(model.predict_proba(features), before)


def test_early_stopping_ends_training_before_the_epoch_limit():
    features, labels = separable()
    model = estimators.TorchClassifier(
        epochs=300, patience=3, learning_rate=1e-2
    ).fit(features, labels)

    assert model.n_iter_ < 300


def test_balanced_class_weight_raises_minority_probability():
    rng = np.random.default_rng(2)
    features = rng.normal(size=(600, 3)).astype(np.float32)
    labels = (features[:, 0] + rng.normal(0, 1.5, 600) > 1.8).astype(int)
    plain = estimators.TorchClassifier(**FAST).fit(features, labels)
    balanced = estimators.TorchClassifier(class_weight="balanced", **FAST).fit(
        features, labels
    )

    assert (
        balanced.predict_proba(features)[:, 1].mean()
        > plain.predict_proba(features)[:, 1].mean()
    )


def test_sparse_input_and_sample_weights_are_accepted():
    features, labels = separable()
    weights = np.where(labels == "yes", 2.0, 1.0)

    model = estimators.TorchClassifier(**FAST).fit(
        sparse.csr_matrix(features), labels, sample_weight=weights
    )

    assert len(model.predict(sparse.csr_matrix(features))) == len(labels)


def test_unknown_architecture_is_a_configuration_error():
    features, labels = separable()
    with pytest.raises(worker.WorkerError, match="unknown architecture"):
        estimators.TorchClassifier(architecture="transformer").fit(
            features, labels
        )


@pytest.mark.parametrize("name", ["torch_mlp", "torch_residual"])
def test_adapters_pass_conformance_and_support_every_task(name):
    adapter = REGISTRY.get(name)

    conformance.check_conformance(adapter)

    assert adapter.capabilities.tasks == tasks.STANDARD_TASKS
    assert adapter.rejection_reason(tasks.TaskType.BINARY) is None


def test_a_pytorch_model_family_trains_packages_and_predicts(
    binary_frame, tmp_path
):
    project = support.make_project(
        binary_frame,
        "y",
        time_budget=600,
        output_dir=tmp_path,
        enforcement="hard",
        models=tasks.ModelSpec(),
    )
    plan = project.plan()
    only_torch = dataclasses.replace(
        plan, candidate_models=("dummy", "torch_residual")
    )

    result = project.run(only_torch)

    assert result.selected_model == "torch_residual"
    features = binary_frame.drop(columns="y").head(20)
    result.save(tmp_path / "net")
    reloaded = scmle.load_model(tmp_path / "net", trusted=True)
    np.testing.assert_array_equal(
        reloaded.predict(features), result.predict(features)
    )
    assert reloaded.predict_proba(features).shape == (20, 2)


def test_serving_limits_are_checked_for_a_pytorch_candidate(
    binary_frame, tmp_path
):
    project = support.make_project(
        binary_frame,
        "y",
        time_budget=600,
        output_dir=tmp_path,
        enforcement="hard",
        models=tasks.ModelSpec(),
        selection=tasks.SelectionSpec(max_model_mb=50.0),
    )
    only_torch = dataclasses.replace(
        project.plan(), candidate_models=("torch_mlp",)
    )

    result = project.run(only_torch)

    assert result.selected_model == "torch_mlp"
    log = (result.run_dir / "diagnostics" / "decision_log.json").read_text()
    assert "within serving limits" in log


def test_the_framework_process_never_imports_torch():
    program = textwrap.dedent(
        """
        import sys
        import numpy as np
        from scmle.models.deeplearning import estimators
        features = np.random.rand(120, 3).astype("float32")
        target = (features[:, 0] > 0.5).astype(int)
        model = estimators.TorchClassifier(epochs=5).fit(features, target)
        model.predict_proba(features)
        assert "torch" not in sys.modules
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True
    )

    assert completed.returncode == 0, completed.stderr[-800:]


def test_torch_shares_a_process_with_xgboost_and_lightgbm():
    xgboost = pytest.importorskip("xgboost")
    lightgbm = pytest.importorskip("lightgbm")
    features, labels = separable()
    numeric = (labels == "yes").astype(int)

    network = estimators.TorchClassifier(**FAST).fit(features, labels)
    xgboost.XGBClassifier(n_estimators=10).fit(features, numeric)
    lightgbm.LGBMClassifier(n_estimators=10, verbose=-1).fit(features, numeric)

    assert (network.predict(features) == labels).mean() > 0.9


def test_a_killed_worker_is_restarted_on_the_next_request():
    features, labels = separable()
    model = estimators.TorchClassifier(**FAST).fit(features, labels)
    before = model.predict_proba(features)

    worker.client().process.kill()
    worker.client().process.wait()

    np.testing.assert_allclose(model.predict_proba(features), before)


def test_training_reports_its_health_and_clips_gradients():
    features, labels = separable()

    model = estimators.TorchClassifier(**FAST).fit(features, labels)

    report = model.training_report_
    assert report["verdict"] in {"ok", "stalled"} and report["steps"] > 0
    assert report["peak_gradient_norm"] >= report["median_gradient_norm"]


def test_a_huge_learning_rate_is_caught_instead_of_silently_accepted():
    features, labels = separable()
    unstable = estimators.TorchClassifier(
        learning_rate=1e6, max_grad_norm=None, epochs=20, hidden_sizes=(256,)
    ).fit(features * 1e3, labels)

    assert unstable.training_report_["verdict"] in {
        "diverged",
        "exploding",
        "stalled",
    }
    assert unstable.training_report_["verdict"] != "ok"


def test_auto_batch_size_and_both_schedules_train():
    features, labels = separable()

    for schedule in ("cosine", "constant"):
        model = estimators.TorchClassifier(
            batch_size="auto", schedule=schedule, **FAST
        ).fit(features, labels)
        assert (model.predict(features) == labels).mean() > 0.85


@pytest.mark.parametrize(
    ("architecture", "sizes"),
    [
        ("tabm", (64,)),
        ("ft_transformer", (32,)),
        ("robust_mlp", (64,)),
    ],
)
def test_modern_architectures_learn_classification(architecture, sizes):
    features, labels = separable()
    model = estimators.TorchClassifier(
        architecture=architecture, hidden_sizes=sizes, **FAST
    ).fit(features, labels)

    assert (model.predict(features) == labels).mean() > 0.9
    probabilities = model.predict_proba(features)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, rtol=1e-5)


@pytest.mark.parametrize(
    "architecture", ["tabm", "ft_transformer", "robust_mlp"]
)
def test_modern_architectures_learn_regression(architecture):
    rng = np.random.default_rng(3)
    features = rng.normal(size=(400, 4)).astype(np.float32)
    target = 20 + 5 * features[:, 0] - 3 * features[:, 2]

    model = estimators.TorchRegressor(
        architecture=architecture,
        hidden_sizes=(32,) if architecture == "ft_transformer" else (64,),
        **FAST,
    ).fit(features, target)

    residual = target - model.predict(features)
    assert 1 - residual.var() / target.var() > 0.85


def test_the_ensemble_architecture_averages_several_members():
    features, labels = separable()
    model = estimators.TorchClassifier(architecture="tabm", **FAST).fit(
        features, labels
    )

    raw = worker.client().request(
        "call",
        (
            model.engine_path,
            model.state_id_,
            model.state_,
            "raw_outputs",
            model.dense(features),
        ),
    )

    assert raw.shape == (len(features), 8, 2)
