"""A bundle member or manifest swapped after it is checked never loads.

`Bundle.load` used to checksum a member and then open the file again to
unpickle it, so a file replaced between the two reads ran unauthenticated
code. Each test replaces one file as soon as its n-th open is closed, for
every n the load performs, and asserts the swapped pickle never runs.
"""

import builtins
import hashlib
import io
import json
import os
import pathlib

import joblib
import numpy
import pandas
import pytest

from scmle.artifacts.bundle import layout
from scmle.artifacts.bundle import model
from scmle.artifacts.bundle import seal
from scmle.contracts import data
from scmle.contracts import errors
from scmle.contracts import tasks
import scmle.models.registry
from scmle.preparation import preprocessing

KEY = b"0123456789abcdef0123456789abcdef"
TRIPPED: list[str] = []


def trip() -> None:
    """Records that the swapped pickle was unpickled."""
    TRIPPED.append("swapped member ran")


class Swapped:
    """A pickle that calls `trip` when it is loaded."""

    def __reduce__(self):
        return (trip, ())


def dumped(value: object) -> bytes:
    buffer = io.BytesIO()
    joblib.dump(value, buffer)
    return buffer.getvalue()


@pytest.fixture
def signed(tmp_path):
    rng = numpy.random.default_rng(0)
    features = pandas.DataFrame(
        {"num": rng.normal(size=80), "cat": rng.choice(["a", "b"], 80)}
    )
    target = pandas.Series((features.num > 0).astype(int))
    plan = data.Blueprint(("num",), ("cat",))
    adapter = scmle.models.registry.Catalog.stock().get("logistic_regression")
    pipe = preprocessing.instantiate(plan, adapter, tasks.Task.BINARY, 0).fit(
        features, target
    )
    bundle = model.Bundle.create(
        pipe, plan, features, "y", "binary_classification", "roc_auc", "lr"
    )
    return bundle.save(tmp_path / "b", signing_key=KEY)


def armed(monkeypatch, target: pathlib.Path, after: int, content: bytes):
    """Replaces `target` with `content` once its `after`-th open is closed.

    The swap waits for the close because Windows refuses to replace a file
    that is open; a loader that reads a file once has closed it before it
    parses or unpickles, so the swap still lands between check and load.
    """
    real = builtins.open
    opened: list[int] = []

    def wrapper(file, mode="r", *args, **kwargs):
        stream = real(file, mode, *args, **kwargs)
        if (
            isinstance(file, (str, os.PathLike))
            and pathlib.Path(file) == target
        ):
            opened.append(1)
            if len(opened) == after:
                close = stream.close

                def closed() -> None:
                    if stream.closed:
                        return
                    close()
                    staged = target.with_suffix(".swap")
                    staged.write_bytes(content)
                    staged.replace(target)

                stream.close = closed
        return stream

    monkeypatch.setattr(builtins, "open", wrapper)
    monkeypatch.setattr(io, "open", wrapper)
    return opened


def test_a_model_swapped_after_its_checksum_read_is_refused(
    signed, monkeypatch
):
    """The first read checks the checksum; the swap lands before the load."""
    TRIPPED.clear()
    target = signed / layout.MODEL_FILE
    opened = armed(monkeypatch, target, 1, dumped(Swapped()))
    with pytest.raises(errors.ArtifactError, match="checksum mismatch"):
        model.Bundle.load(signed, signing_key=KEY)
    assert not TRIPPED
    assert len(opened) == 2


def test_a_model_swapped_after_the_bytes_it_loads_are_read_is_ignored(
    signed, monkeypatch
):
    """The second read is the one loaded, from memory; the file is not
    opened again, so the swapped file never runs."""
    TRIPPED.clear()
    target = signed / layout.MODEL_FILE
    opened = armed(monkeypatch, target, 2, dumped(Swapped()))
    loaded = model.Bundle.load(signed, signing_key=KEY)
    assert not isinstance(loaded.pipeline, Swapped)
    assert loaded.pipeline is not None
    assert not TRIPPED
    assert len(opened) == 2


def test_a_model_swapped_before_any_read_is_refused(signed):
    TRIPPED.clear()
    (signed / layout.MODEL_FILE).write_bytes(dumped(Swapped()))
    with pytest.raises(errors.ArtifactError, match="checksum mismatch"):
        model.Bundle.load(signed, signing_key=KEY)
    assert not TRIPPED


@pytest.mark.parametrize("after", [1, 2])
def test_a_manifest_swapped_after_parsing_is_never_trusted(
    signed, monkeypatch, after
):
    """A forged manifest listing a malicious model never passes the check.

    The forgery names the digest of a swapped model; whichever read of the
    manifest the swap follows, the signature is checked over the bytes that
    were parsed, so the forged digest is never trusted.
    """
    TRIPPED.clear()
    payload = dumped(Swapped())
    manifest = json.loads((signed / layout.MANIFEST_FILE).read_text("utf-8"))
    original = (signed / layout.MANIFEST_FILE).read_bytes()
    manifest["files"][layout.MODEL_FILE] = hashlib.sha256(payload).hexdigest()
    forged = json.dumps(manifest, indent=2).encode()
    (signed / layout.MANIFEST_FILE).write_bytes(forged)
    (signed / layout.MODEL_FILE).write_bytes(payload)
    armed(monkeypatch, signed / layout.MANIFEST_FILE, after, original)
    with pytest.raises(errors.ArtifactError):
        model.Bundle.load(signed, signing_key=KEY)
    assert not TRIPPED


def test_an_unsigned_bundle_holding_a_tripwire_is_refused(signed):
    TRIPPED.clear()
    (signed / seal.SIGNATURE_FILE).unlink()
    (signed / layout.MODEL_FILE).write_bytes(dumped(Swapped()))
    with pytest.raises(errors.ArtifactError, match="unauthenticated bundle"):
        model.Bundle.load(signed, signing_key=KEY)
    assert not TRIPPED
