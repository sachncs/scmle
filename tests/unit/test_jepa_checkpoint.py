"""Checkpoint and deterministic-output contracts for tabular JEPA."""

import importlib.util

import numpy as np
import pytest

from scmle.models.deeplearning.jepa import JepaEstimator

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="torch is not installed",
)


def features() -> np.ndarray:
    """Return a small finite numeric table with repeatable values."""
    rng = np.random.default_rng(17)
    return rng.normal(size=(32, 5)).astype(np.float32)


def estimator() -> JepaEstimator:
    """Return a short-running JEPA configuration for contract tests."""
    return JepaEstimator(
        hidden_size=12,
        embedding_size=6,
        epochs=3,
        batch_size=16,
        random_state=23,
    )


def test_checkpoint_round_trip_preserves_embeddings(tmp_path):
    """A saved representation reloads without retraining or torch state."""
    rows = features()
    original = estimator().fit(rows)
    before = original.transform(rows[:7])

    checkpoint = original.save_checkpoint(tmp_path / "jepa.joblib")
    restored = JepaEstimator.load_checkpoint(checkpoint)

    np.testing.assert_allclose(restored.transform(rows[:7]), before)
    assert restored.training_report_ == original.training_report_


def test_same_seed_reproduces_training_and_transform_output():
    """The declared random state makes repeated CPU fits reproducible."""
    rows = features()
    first = estimator().fit(rows).transform(rows)
    second = estimator().fit(rows).transform(rows)

    np.testing.assert_allclose(first, second, rtol=0.0, atol=0.0)


def test_invalid_checkpoint_format_is_rejected(tmp_path):
    """A random joblib payload cannot be treated as a JEPA checkpoint."""
    import joblib

    path = tmp_path / "invalid.joblib"
    joblib.dump({"format": "unknown"}, path)

    with pytest.raises(ValueError, match="checkpoint format"):
        JepaEstimator.load_checkpoint(path)


def test_checkpoint_payload_tampering_is_rejected(tmp_path):
    """A modified serialized payload fails the integrity check."""
    import joblib

    checkpoint = (
        estimator().fit(features()).save_checkpoint(tmp_path / "jepa.joblib")
    )
    payload = joblib.load(checkpoint)
    payload["payload"] = payload["payload"][:-1] + b"x"
    joblib.dump(payload, checkpoint)

    with pytest.raises(ValueError, match="integrity check"):
        JepaEstimator.load_checkpoint(checkpoint)


def test_checkpoint_metadata_mismatch_is_rejected(tmp_path):
    """Envelope metadata cannot silently describe another representation."""
    import joblib

    checkpoint = (
        estimator().fit(features()).save_checkpoint(tmp_path / "jepa.joblib")
    )
    payload = joblib.load(checkpoint)
    payload["input_width"] += 1
    joblib.dump(payload, checkpoint)

    with pytest.raises(ValueError, match="metadata does not match"):
        JepaEstimator.load_checkpoint(checkpoint)
