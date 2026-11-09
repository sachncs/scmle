"""Functional lifecycle contracts for the optional PyTorch model families."""

from importlib import util

import joblib
import numpy
import pytest

from scmle.contracts import capabilities
from scmle.contracts import tasks
from scmle.models.deeplearning import contracts
from scmle.models.deeplearning import estimators
from scmle.models.deeplearning import jepa
from scmle.models.deeplearning import networks

pytestmark = pytest.mark.skipif(
    util.find_spec("torch") is None,
    reason="the optional deeplearning extra is not installed",
)


def features() -> numpy.ndarray:
    """Return a finite numeric table with a reproducible signal."""
    rng = numpy.random.default_rng(41)
    return rng.normal(size=(24, 4)).astype(numpy.float32)


def labels(rows: int) -> numpy.ndarray:
    """Return a binary target for functional classifier tests."""
    return numpy.arange(rows) % 2


def test_transformer_contract_is_tabular_and_fails_closed_for_images():
    """The experimental Transformer cannot be mistaken for multimodal support."""
    family = contracts.TABULAR_TRANSFORMER_CONTRACT

    assert family.status is capabilities.CapabilityStatus.EXPERIMENTAL
    assert family.supports_modality(capabilities.Modality.TABULAR)
    assert not family.supports_modality(capabilities.Modality.IMAGE)
    with pytest.raises(ValueError, match="supports only tabular modality"):
        family.require_modality(capabilities.Modality.IMAGE)

    boundary = contracts.ModelFamilyContract.unsupported_multimodal()
    assert boundary.status is capabilities.CapabilityStatus.EXTENSION_ONLY
    with pytest.raises(ValueError, match="supports only tabular modality"):
        boundary.require_modality(capabilities.Modality.TEXT)


def test_transformer_runs_real_torch_inference_explanation_and_checkpoint(
    tmp_path,
):
    """A real FT-Transformer fit exercises every declared lifecycle surface."""
    matrix = features()
    model = estimators.TorchClassifier(
        architecture="ft_transformer",
        hidden_sizes=(16,),
        epochs=3,
        batch_size=12,
        validation_fraction=0.0,
        random_state=19,
    ).fit(matrix, labels(len(matrix)))

    probabilities = model.predict_proba(matrix[:6])
    explanation = model.explain(matrix[:6])

    assert model.model_contract() is contracts.TABULAR_TRANSFORMER_CONTRACT
    assert model.inference_contract().methods == ("predict", "predict_proba")
    assert probabilities.shape == (6, 2)
    assert numpy.isfinite(probabilities).all()
    assert explanation.values.shape == (6, 4)
    assert numpy.isfinite(explanation.values).all()
    assert explanation.method == "gradient_x_input"
    assert explanation.target == "predicted_output"

    checkpoint = model.save_checkpoint(tmp_path / "transformer.joblib")
    restored = estimators.TorchClassifier.load_checkpoint(checkpoint)

    numpy.testing.assert_allclose(
        restored.predict_proba(matrix[:6]), probabilities, rtol=0.0, atol=0.0
    )
    assert restored.checkpoint_manifest()["model_family"] == "transformer"


def test_transformer_rejects_non_tabular_shape_before_worker_start():
    """The input contract reports shape errors at the estimator boundary."""
    model = estimators.TorchClassifier(
        architecture="ft_transformer", epochs=1, validation_fraction=0.0
    )

    with pytest.raises(ValueError, match="requires a 2-D numeric matrix"):
        model.fit(numpy.zeros((8, 2, 1), dtype=numpy.float32), labels(8))


def test_jepa_runs_real_torch_embedding_explanation_and_checkpoint(tmp_path):
    """A real JEPA fit preserves its embedding and explanation contracts."""
    matrix = features()
    model = jepa.JepaEstimator(
        hidden_size=12,
        embedding_size=6,
        epochs=2,
        batch_size=12,
        random_state=29,
    ).fit(matrix)

    embedding = model.transform(matrix[:5])
    explanation = model.explain(matrix[:5])

    assert model.model_contract() is contracts.TABULAR_JEPA_CONTRACT
    assert model.inference_contract().methods == ("encode", "transform")
    assert embedding.shape == (5, 6)
    assert numpy.isfinite(embedding).all()
    assert explanation.values.shape == (5, 4)
    assert numpy.isfinite(explanation.values).all()
    assert explanation.method == "embedding_gradient_x_input"
    assert explanation.target == "embedding_energy"

    checkpoint = model.save_checkpoint(tmp_path / "jepa.joblib")
    restored = jepa.JepaEstimator.load_checkpoint(checkpoint)

    numpy.testing.assert_allclose(
        restored.transform(matrix[:5]), embedding, rtol=0.0, atol=0.0
    )
    assert restored.checkpoint_manifest()["model_family"] == "jepa"


def test_transformer_checkpoint_metadata_tampering_is_rejected(tmp_path):
    """A valid payload cannot be relabeled as another model family."""
    matrix = features()
    model = estimators.TorchClassifier(
        architecture="ft_transformer",
        hidden_sizes=(16,),
        epochs=1,
        validation_fraction=0.0,
        random_state=7,
    ).fit(matrix, labels(len(matrix)))
    checkpoint = model.save_checkpoint(tmp_path / "transformer.joblib")
    envelope = joblib.load(checkpoint)
    envelope["model_family"] = "jepa"
    joblib.dump(envelope, checkpoint)

    with pytest.raises(ValueError, match="metadata does not match"):
        estimators.TorchClassifier.load_checkpoint(checkpoint)


def test_torch_adapters_expose_the_same_fail_closed_modality_contract():
    """Adapter-level declarations match the estimator-level boundary."""
    transformer = networks.TorchTransformerAdapter()
    jepa_adapter = networks.TorchJepaAdapter()

    assert transformer.supports_modality("tabular")
    assert not transformer.supports_modality("image")
    assert jepa_adapter.family_contract is contracts.TABULAR_JEPA_CONTRACT
    with pytest.raises(ValueError, match="supports only tabular modality"):
        jepa_adapter.require_modality("audio")


def test_jepa_rejection_reason_includes_both_training_boundaries():
    """Small JEPA datasets report the complete actionable boundary."""
    reason = networks.TorchJepaAdapter().rejection_reason(
        tasks.TaskType.BINARY, row_count=15
    )

    assert reason == (
        "tabular JEPA requires at least 16 rows for its training and "
        "validation boundaries"
    )
