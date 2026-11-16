"""Checkpoint and deterministic-output contracts for tabular JEPA."""

import importlib.util
import json

import numpy
import pytest

from scmle import envelope
from scmle.contracts import errors
from scmle.models.deeplearning.jepa import JepaEstimator

KEY = "checkpoint-test-key"

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="torch is not installed",
)


def features() -> numpy.ndarray:
    """Return a small finite numeric table with repeatable values."""
    rng = numpy.random.default_rng(17)
    return rng.normal(size=(32, 5)).astype(numpy.float32)


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
    checkpoint = original.save_checkpoint(tmp_path / "jepa.bin", KEY)
    restored = JepaEstimator.load_checkpoint(checkpoint, KEY)
    numpy.testing.assert_allclose(restored.transform(rows[:7]), before)
    assert restored.training_report_ == original.training_report_


def test_an_unsigned_checkpoint_needs_trust(tmp_path, monkeypatch):
    """SEC-1: nothing is unpickled from a checkpoint nobody signed."""
    monkeypatch.delenv("SCMLE_SIGNING_KEY", raising=False)
    checkpoint = estimator().fit(features()).save_checkpoint(tmp_path / "j.bin")
    with pytest.raises(errors.ArtifactError, match="not signed"):
        JepaEstimator.load_checkpoint(checkpoint, KEY)
    assert JepaEstimator.load_checkpoint(checkpoint, trusted=True)


def test_same_seed_reproduces_training_and_transform_output():
    """The declared random state makes repeated CPU fits reproducible."""
    rows = features()
    first = estimator().fit(rows).transform(rows)
    second = estimator().fit(rows).transform(rows)

    numpy.testing.assert_allclose(first, second, rtol=0.0, atol=0.0)


def test_invalid_checkpoint_format_is_rejected(tmp_path):
    """A random joblib payload cannot be treated as a JEPA checkpoint."""
    import joblib

    path = tmp_path / "invalid.joblib"
    joblib.dump({"format": "unknown"}, path)
    with pytest.raises(ValueError, match="not an scmle envelope"):
        JepaEstimator.load_checkpoint(path, KEY, trusted=True)


def test_checkpoint_payload_tampering_is_rejected(tmp_path):
    """A modified serialized payload fails the integrity check."""
    checkpoint = (
        estimator().fit(features()).save_checkpoint(tmp_path / "j.bin", KEY)
    )
    raw = bytearray(checkpoint.read_bytes())
    raw[-1] ^= 1
    checkpoint.write_bytes(bytes(raw))
    with pytest.raises(ValueError, match="digest"):
        JepaEstimator.load_checkpoint(checkpoint, KEY)


def test_checkpoint_metadata_mismatch_is_rejected(tmp_path):
    """Envelope metadata cannot silently describe another representation."""
    checkpoint = (
        estimator().fit(features()).save_checkpoint(tmp_path / "j.bin", KEY)
    )
    meta, payload = envelope.read(checkpoint, KEY)
    edited = {**meta, "input_width": meta["input_width"] + 1}
    magic, header, body = checkpoint.read_bytes().split(b"\n", 2)
    record = json.loads(header)
    record["meta"] = edited
    checkpoint.write_bytes(
        magic + b"\n" + json.dumps(record).encode() + b"\n" + body
    )
    with pytest.raises(errors.ArtifactError, match="not authentic"):
        JepaEstimator.load_checkpoint(checkpoint, KEY)
    envelope.seal(checkpoint, edited, payload, KEY)
    with pytest.raises(ValueError, match="metadata does not match"):
        JepaEstimator.load_checkpoint(checkpoint, KEY)
