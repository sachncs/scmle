"""One attack per deserialisation path: unauthenticated bytes never run (C9).

`scmle.security.deserialisation` lists every call site that unpickles. For
each, an attack below writes a tripwire pickle (one that records it ran when
loaded) into the file that path reads, unsigned and then signed with the
wrong key, and calls the public entry point with the right key. The path
must refuse, and the tripwire must never fire. A site the scanner finds with
no attack here fails `test_every_site_has_an_attack`.

The positive control, `test_attack_reaches_the_sink`, signs the same
tripwire with the right key and requires it to fire: a site that refuses
for another reason before it reaches its deserialiser proves nothing. A
round trip reads no input, so it has one attack, that it opens no file, and
no control.
"""

import builtins
from collections.abc import Callable
import contextlib
import dataclasses
import hashlib
import io
import json
import pathlib

import joblib
import numpy
import pandas
import pytest

from scmle import envelope
from scmle.artifacts.bundle import layout
from scmle.artifacts.bundle import model
from scmle.artifacts.bundle import seal
from scmle.contracts import data
from scmle.contracts import errors
from scmle.contracts import signing
from scmle.contracts import tasks
from scmle.models import conformance
from scmle.models import documents
from scmle.models import images
from scmle.models.deeplearning import estimators
from scmle.models.deeplearning import jepa
import scmle.models.registry
from scmle.pipeline import modality_lifecycle
from scmle.pipeline.stages import training
from scmle.preparation import preprocessing
from scmle.security import deserialisation

ROOT = pathlib.Path(__file__).resolve().parents[2]
KEY = "the right key, long enough for hmac"
WRONG = "the wrong key, also long enough"
TRIPPED: list[str] = []


def trip() -> None:
    """Records that a tripwire pickle was unpickled."""
    TRIPPED.append("tripwire ran")


class Tripwire:
    """A pickle that calls `trip` when it is loaded."""

    def __reduce__(self):
        return (trip, ())


def tripwire() -> bytes:
    buffer = io.BytesIO()
    joblib.dump(Tripwire(), buffer)
    return buffer.getvalue()


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    TRIPPED.clear()
    monkeypatch.delenv(signing.SIGNING_KEY_ENV, raising=False)
    monkeypatch.delenv(signing.SIGNING_KEY_FILE_ENV, raising=False)
    monkeypatch.delenv(envelope.ENVIRONMENT, raising=False)
    yield
    assert not TRIPPED, "a tripwire pickle was unpickled"


def bundle(folder: pathlib.Path, key: str | None) -> None:
    rng = numpy.random.default_rng(0)
    features = pandas.DataFrame(
        {"num": rng.normal(size=60), "cat": rng.choice(["a", "b"], 60)}
    )
    target = pandas.Series((features.num > 0).astype(int))
    plan = data.Blueprint(("num",), ("cat",))
    adapter = scmle.models.registry.Catalog.stock().get("logistic_regression")
    pipe = preprocessing.instantiate(plan, adapter, tasks.Task.BINARY, 0).fit(
        features, target
    )
    made = model.Bundle.create(
        pipe, plan, features, "y", "binary_classification", "roc_auc", "lr"
    )
    made.save(folder, signing_key=key)
    payload = tripwire()
    (folder / layout.MODEL_FILE).write_bytes(payload)
    manifest = json.loads((folder / layout.MANIFEST_FILE).read_text("utf-8"))
    manifest["files"][layout.MODEL_FILE] = hashlib.sha256(payload).hexdigest()
    text = json.dumps(manifest, indent=2)
    (folder / layout.MANIFEST_FILE).write_text(text, encoding="utf-8")
    if key is not None:
        (folder / seal.SIGNATURE_FILE).write_text(
            json.dumps(
                {
                    "algorithm": seal.SIGNATURE_ALGORITHM,
                    "signature": seal.sign(text.encode(), key.encode()),
                }
            ),
            encoding="utf-8",
        )


@dataclasses.dataclass(frozen=True)
class Attack:
    """How to plant a tripwire for one site, and how that site refuses it.

    Attributes:
        plant: Writes the tripwire, signed with a key or unsigned (None),
            and returns the call that loads it through the site.
        refusal: The error the site raises on unauthenticated bytes; None
            when it refuses by returning None.
    """

    plant: Callable[[pathlib.Path, str | None], Callable[[], object]]
    refusal: type[Exception] | None = errors.ArtifactError


def bundled(tmp_path: pathlib.Path, key: str | None) -> Callable[[], object]:
    bundle(tmp_path / "b", key)
    return lambda: model.Bundle.load(tmp_path / "b", signing_key=KEY)


def checkpointed(kind):
    def plant(tmp_path: pathlib.Path, key: str | None) -> Callable[[], object]:
        path = tmp_path / "checkpoint.envelope"
        meta = {"schema_version": kind.CHECKPOINT_SCHEMA_VERSION}
        if hasattr(kind, "CHECKPOINT_FORMAT"):
            meta["format"] = kind.CHECKPOINT_FORMAT
        envelope.seal(path, meta, tripwire(), key)
        return lambda: kind.restore(path, signing_key=KEY)

    return plant


def packaged(module):
    def plant(tmp_path: pathlib.Path, key: str | None) -> Callable[[], object]:
        payload = tmp_path / "model.joblib"
        payload.write_bytes(tripwire())
        manifest = {
            "schema_version": 1,
            "model_path": "model.joblib",
            "artifact_digest": module.file_digest(payload),
        }
        if key is not None:
            manifest = dict(envelope.sign(manifest, key))
        (tmp_path / "manifest.json").write_text(json.dumps(manifest))
        return lambda: module.Classifier.load(tmp_path, signing_key=KEY)

    return plant


def modal(tmp_path: pathlib.Path, key: str | None) -> Callable[[], object]:
    payload = tmp_path / "model.joblib"
    payload.write_bytes(tripwire())
    unsigned = {
        "schema_version": modality_lifecycle.PACKAGE_SCHEMA_VERSION,
        "modality": "tabular",
        "dataset_hash": "d" * 64,
        "feature_names": ["a"],
        "model_sha256": modality_lifecycle.file_digest(payload),
        "evaluation": {},
    }
    manifest: dict[str, object] = dict(unsigned)
    if key is not None:
        manifest["signature"] = {
            "algorithm": signing.SIGNATURE_ALGORITHM,
            "value": signing.mac(unsigned, key),
        }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return lambda: modality_lifecycle.load_package(tmp_path, KEY)


def kept(tmp_path: pathlib.Path, key: str | None) -> Callable[[], object]:
    path = tmp_path / "04-trainstage.product"
    payload = tripwire()
    envelope.seal(path, {"product": "Search"}, payload, key)
    expected = hashlib.sha256(payload).hexdigest()

    def load() -> object:
        patch = pytest.MonkeyPatch()
        patch.setenv(signing.SIGNING_KEY_ENV, KEY)
        try:
            return training.Search.load(path, expected)
        finally:
            patch.undo()

    return load


def roundtrip(tmp_path: pathlib.Path, key: str | None) -> Callable[[], object]:
    """The round trip loads only what it dumped: it opens no file at all.

    It reads no input, so there is nothing to sign or plant; the call
    returns None when it ran without opening a file.
    """
    adapter = scmle.models.registry.Catalog.stock().get("logistic_regression")
    features, target = conformance.toy(tasks.Task.BINARY)

    def load() -> None:
        real = builtins.open

        def refuse(*args, **kwargs):
            raise AssertionError(f"the round trip opened {args[:1]}")

        builtins.open = refuse
        try:
            conformance.smoke(adapter, tasks.Task.BINARY, features, target)
        finally:
            builtins.open = real

    return load


ATTACKS = {
    "scmle.artifacts.bundle.archive:unpickle": Attack(bundled),
    "scmle.models.conformance:smoke": Attack(roundtrip, None),
    "scmle.models.deeplearning.estimators:NetworkEstimator.restore": Attack(
        checkpointed(estimators.TorchClassifier)
    ),
    "scmle.models.deeplearning.jepa:Embedder.restore": Attack(
        checkpointed(jepa.Embedder)
    ),
    "scmle.models.documents.archive:Archive.load": Attack(packaged(documents)),
    "scmle.models.images.archive:Archive.load": Attack(packaged(images)),
    "scmle.pipeline.modality_lifecycle.verification:load_package": Attack(
        modal, errors.InputError
    ),
    "scmle.pipeline.products:Kept.load": Attack(kept, None),
}
# Sites that read no input: one attack each, and no positive control.
ROUNDTRIP = {"scmle.models.conformance:smoke"}
CASES = [
    pytest.param(site, key, id=f"{site}-{label}")
    for site in sorted(ATTACKS)
    for key, label in (
        [(None, "no-input")]
        if site in ROUNDTRIP
        else [(None, "unsigned"), (WRONG, "wrong-key")]
    )
]


def test_every_site_has_an_attack():
    sites = deserialisation.scan(ROOT / "scmle")
    assert {s.key for s in sites} == set(ATTACKS)
    assert {s.key for s in sites if s.verdict == "roundtrip"} == ROUNDTRIP


@pytest.mark.parametrize(("site", "key"), CASES)
def test_unauthenticated_bytes_never_run(site, key, tmp_path):
    attack = ATTACKS[site]
    load = attack.plant(tmp_path, key)
    if attack.refusal is None:
        assert load() is None
    else:
        with pytest.raises(attack.refusal):
            load()
    assert not TRIPPED


@pytest.mark.parametrize("site", sorted(set(ATTACKS) - ROUNDTRIP))
def test_attack_reaches_the_sink(site, tmp_path):
    load = ATTACKS[site].plant(tmp_path, KEY)
    with contextlib.suppress(Exception):
        # what the site does with the tripwire's None is not the question
        load()
    assert TRIPPED, f"{site} refused correctly signed bytes before its sink"
    TRIPPED.clear()
