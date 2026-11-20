"""A payload swapped after it is hashed never loads (SEC-13).

Each package loader reads the payload once, checks the digest of those bytes
and unpickles the same bytes. The test replaces the file on disk as soon as
its first open (the hash step) is closed, so a loader that opened it again
to unpickle would run the swapped pickle. The swap waits for the close
because Windows refuses to replace a file that is open.
"""

import builtins
import io
import json
import os
import pathlib

import joblib
import pytest

from scmle import envelope
from scmle.contracts import errors
from scmle.contracts import signing
from scmle.models import documents
from scmle.models import images
from scmle.pipeline import modality_lifecycle

KEY = "swap-test-key"
TRIPPED: list[str] = []


def trip() -> None:
    """Records that the swapped pickle was unpickled."""
    TRIPPED.append("swapped payload ran")


class Swapped:
    """A pickle that calls `trip` when it is loaded."""

    def __reduce__(self):
        return (trip, ())


def dumped(value: object) -> bytes:
    buffer = io.BytesIO()
    joblib.dump(value, buffer)
    return buffer.getvalue()


@pytest.fixture
def swap(monkeypatch):
    """Swaps the payload for `Swapped` when its first open is closed."""
    TRIPPED.clear()
    real = builtins.open
    opened: list[pathlib.Path] = []

    def arm(model: pathlib.Path) -> None:
        def wrapper(file, mode="r", *args, **kwargs):
            stream = real(file, mode, *args, **kwargs)
            if (
                isinstance(file, (str, os.PathLike))
                and pathlib.Path(file) == model
                and not opened
            ):
                opened.append(model)
                close = stream.close

                def closed() -> None:
                    if stream.closed:
                        return
                    close()
                    staged = model.with_suffix(".swap")
                    staged.write_bytes(dumped(Swapped()))
                    staged.replace(model)

                stream.close = closed
            return stream

        monkeypatch.setattr(builtins, "open", wrapper)
        monkeypatch.setattr(io, "open", wrapper)

    yield arm
    assert opened, "the payload was never opened"
    assert opened[0].read_bytes() == dumped(Swapped()), "never swapped"


@pytest.mark.parametrize("module", [documents, images])
def test_a_package_payload_swapped_after_hashing_never_loads(
    tmp_path, swap, module
):
    model = tmp_path / "model.joblib"
    model.write_bytes(dumped({"not": "a classifier"}))
    manifest = envelope.sign(
        {
            "schema_version": 1,
            "model_path": "model.joblib",
            "artifact_digest": module.digest(model),
        },
        KEY,
    )
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    swap(model)
    with pytest.raises(errors.KindError):
        module.Classifier.load(tmp_path, signing_key=KEY)
    assert not TRIPPED


def test_a_modality_payload_swapped_after_hashing_never_loads(tmp_path, swap):
    model = tmp_path / "model.joblib"
    model.write_bytes(dumped({"not": "a classifier"}))
    unsigned = {
        "schema_version": modality_lifecycle.PACKAGE_SCHEMA_VERSION,
        "modality": "tabular",
        "dataset_hash": "d" * 64,
        "feature_names": ["a"],
        "model_sha256": modality_lifecycle.digest(model),
        "evaluation": {},
    }
    manifest = {
        **unsigned,
        "signature": {
            "algorithm": signing.SIGNATURE_ALGORITHM,
            "value": signing.mac(unsigned, KEY),
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    swap(model)
    with pytest.raises(errors.KindError):
        modality_lifecycle.load(tmp_path, KEY)
    assert not TRIPPED
