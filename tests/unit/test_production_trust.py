"""Production refuses vouched-for pickles, as it refuses bundles (SEC-1, SEC-2).

`trusted=True` lets an unsigned deep-learning checkpoint or document or image
package load anywhere but production. With `SCMLE_ENV=production` each loader
refuses it before reading the file, so `joblib.load` is never reached.
"""

import hashlib
import io
import json

import joblib
import pytest

from scmle import envelope
from scmle.contracts import errors
from scmle.models import documents
from scmle.models import images
from scmle.models.deeplearning import estimators
from scmle.models.deeplearning import jepa


@pytest.fixture(autouse=True)
def production(monkeypatch):
    """Runs each test in production with no key and a tripwired unpickler."""

    def refuse(*args, **kwargs):
        raise AssertionError("joblib.load was reached")

    monkeypatch.setenv("SCMLE_ENV", " Production ")
    monkeypatch.delenv("SCMLE_SIGNING_KEY", raising=False)
    monkeypatch.delenv("SCMLE_SIGNING_KEY_FILE", raising=False)
    monkeypatch.setattr(joblib, "load", refuse)


def pickled() -> bytes:
    buffer = io.BytesIO()
    joblib.dump({"weights": [1, 2, 3]}, buffer)
    return buffer.getvalue()


@pytest.mark.parametrize(
    "restore",
    [estimators.TorchClassifier.restore, jepa.Embedder.restore],
)
def test_production_refuses_a_trusted_checkpoint(tmp_path, restore):
    path = envelope.seal(tmp_path / "unsigned.bin", {}, pickled())
    with pytest.raises(errors.ArtifactError, match="SCMLE_ENV=production"):
        restore(path, trusted=True)


@pytest.mark.parametrize(
    "load", [documents.Classifier.load, images.Classifier.load]
)
def test_production_refuses_a_trusted_package(tmp_path, load):
    (tmp_path / "model.joblib").write_bytes(pickled())
    manifest = {
        "schema_version": 1,
        "model_path": "model.joblib",
        "artifact_digest": hashlib.sha256(pickled()).hexdigest(),
        "signature": None,
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(errors.ArtifactError, match="SCMLE_ENV=production"):
        load(tmp_path, trusted=True)
