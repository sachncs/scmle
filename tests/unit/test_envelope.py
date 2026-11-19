"""Pickled payloads are authenticated before they are unpickled (SEC-1, SEC-2).

Each refusal test replaces `joblib.load` with a tripwire, so a test fails if
any byte of an unauthenticated payload would have been unpickled.
"""

import hashlib
import io
import json

import joblib
import pytest

from scmle import envelope
from scmle.contracts import errors

KEY = "envelope-test-key"


@pytest.fixture
def tripwire(monkeypatch):
    """Makes any unpickling during the test an error."""

    def refuse(*args, **kwargs):
        raise AssertionError("joblib.load was reached")

    monkeypatch.setattr(joblib, "load", refuse)


@pytest.fixture(autouse=True)
def nokey(monkeypatch):
    monkeypatch.delenv("SCMLE_SIGNING_KEY", raising=False)
    monkeypatch.delenv("SCMLE_SIGNING_KEY_FILE", raising=False)


def payload():
    buffer = io.BytesIO()
    joblib.dump({"weights": [1, 2, 3]}, buffer)
    return buffer.getvalue()


def test_a_signed_envelope_round_trips(tmp_path):
    path = envelope.seal(tmp_path / "e.bin", {"kind": "test"}, payload(), KEY)
    meta, data = envelope.read(path, KEY)
    assert meta == {"kind": "test"}
    assert joblib.load(io.BytesIO(data)) == {"weights": [1, 2, 3]}


def test_a_tampered_payload_is_refused_before_unpickling(tmp_path, tripwire):
    path = envelope.seal(tmp_path / "e.bin", {"kind": "test"}, payload(), KEY)
    raw = bytearray(path.read_bytes())
    raw[-1] ^= 1
    path.write_bytes(bytes(raw))
    with pytest.raises(ValueError, match="digest"):
        envelope.read(path, KEY)


def test_a_rewritten_header_and_digest_fail_the_signature(tmp_path, tripwire):
    path = envelope.seal(tmp_path / "e.bin", {"kind": "test"}, payload(), KEY)
    magic, header, ignored = path.read_bytes().split(b"\n", 2)
    forged = b"forged payload"
    record = json.loads(header)
    record["sha256"] = hashlib.sha256(forged).hexdigest()
    path.write_bytes(
        magic + b"\n" + json.dumps(record).encode() + b"\n" + forged
    )
    with pytest.raises(errors.ArtifactError, match="not authentic"):
        envelope.read(path, KEY)
    with pytest.raises(errors.ArtifactError, match="not authentic"):
        envelope.read(path, "another-key")


def test_an_invalid_signature_is_refused_even_when_trusted(tmp_path, tripwire):
    path = envelope.seal(tmp_path / "e.bin", {"kind": "test"}, payload(), KEY)
    with pytest.raises(errors.ArtifactError, match="not authentic"):
        envelope.read(path, "another-key", trusted=True)
    changed = {**envelope.sign({"artifact_digest": "a" * 64}, KEY)}
    changed["artifact_digest"] = "b" * 64
    with pytest.raises(errors.ArtifactError, match="not authentic"):
        envelope.check(changed, "pkg", KEY, trusted=True)


def test_an_unsigned_envelope_needs_trust(tmp_path, tripwire):
    path = envelope.seal(tmp_path / "e.bin", {"kind": "test"}, payload())
    with pytest.raises(errors.ArtifactError, match="not signed"):
        envelope.read(path, KEY)
    meta, ignored = envelope.read(path, trusted=True)
    assert meta == {"kind": "test"}


def test_a_signed_envelope_without_a_key_needs_trust(tmp_path, tripwire):
    path = envelope.seal(tmp_path / "e.bin", {"kind": "test"}, payload(), KEY)
    with pytest.raises(errors.ArtifactError, match="no key is set"):
        envelope.read(path)


def test_a_file_that_is_not_an_envelope_is_refused(tmp_path, tripwire):
    path = tmp_path / "old.joblib"
    path.write_bytes(payload())
    with pytest.raises(ValueError, match="not an scmle envelope"):
        envelope.read(path, KEY, trusted=True)


def test_a_signed_manifest_verifies_and_a_changed_one_does_not():
    signed = envelope.sign({"artifact_digest": "a" * 64}, KEY)
    envelope.check(signed, "pkg", KEY)
    changed = {**signed, "artifact_digest": "b" * 64}
    with pytest.raises(errors.ArtifactError, match="not authentic"):
        envelope.check(changed, "pkg", KEY)
    unsigned = envelope.sign({"artifact_digest": "a" * 64})
    assert unsigned["signature"] is None
    with pytest.raises(errors.ArtifactError, match="not signed"):
        envelope.check(unsigned, "pkg", KEY)
    envelope.check(unsigned, "pkg", KEY, trusted=True)


def test_a_document_package_is_checked_before_its_payload_is_loaded(
    tmp_path, monkeypatch
):
    from scmle.models import documents

    calls = []
    real = envelope.check

    def spy(*args, **kwargs):
        calls.append("check")
        return real(*args, **kwargs)

    def loader(*args, **kwargs):
        calls.append("load")
        raise AssertionError("stop after the order is known")

    monkeypatch.setattr(envelope, "check", spy)
    monkeypatch.setattr(joblib, "load", loader)
    package = tmp_path / "pkg"
    package.mkdir()
    model = package / "model.joblib"
    model.write_bytes(payload())
    manifest = envelope.sign(
        {
            "schema_version": 1,
            "model_path": "model.joblib",
            "artifact_digest": documents.file_digest(model),
        },
        KEY,
    )
    (package / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(AssertionError, match="order is known"):
        documents.Classifier.load(package, signing_key=KEY)
    assert calls == ["check", "load"]
    calls.clear()
    with pytest.raises(errors.ArtifactError):
        documents.Classifier.load(package, signing_key="another-key")
    assert calls == ["check"]
