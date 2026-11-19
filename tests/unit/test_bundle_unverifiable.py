"""A signed bundle read without a key is refused, even when trusted (C8)."""

import hashlib
import json

import numpy
import pandas
import pytest

from scmle import cli
import scmle.artifacts.bundle
from scmle.contracts import data
from scmle.contracts import errors
from scmle.contracts import tasks
import scmle.contracts.signing
import scmle.models.registry
from scmle.preparation import preprocessing

PLAN = data.Blueprint(("num",), ())


@pytest.fixture
def rehashed(tmp_path, monkeypatch):
    """A signed bundle whose model changed and whose checksum was updated.

    Its signature.json is left in place, so it no longer verifies.
    """
    monkeypatch.delenv(scmle.contracts.signing.SIGNING_KEY_ENV, raising=False)
    monkeypatch.delenv("SCMLE_ENV", raising=False)
    rng = numpy.random.default_rng(0)
    features = pandas.DataFrame({"num": rng.normal(size=60)})
    target = pandas.Series((features.num > 0).astype(int))
    pipe = preprocessing.instantiate(
        PLAN,
        scmle.models.registry.Catalog.stock().get("logistic_regression"),
        tasks.Task.BINARY,
        0,
    ).fit(features, target)
    bundle = scmle.artifacts.bundle.Bundle.create(
        pipe, PLAN, features, "y", "binary_classification", "roc_auc", "lr"
    )
    directory = bundle.save(tmp_path / "m", signing_key="k")
    member = directory / scmle.artifacts.bundle.MODEL_FILE
    member.write_bytes(member.read_bytes() + b"\0")
    manifest = directory / scmle.artifacts.bundle.MANIFEST_FILE
    record = json.loads(manifest.read_text(encoding="utf-8"))
    record["files"][member.name] = hashlib.sha256(
        member.read_bytes()
    ).hexdigest()
    manifest.write_text(json.dumps(record), encoding="utf-8")
    return directory


def test_trusted_load_refuses_a_signature_it_cannot_check(rehashed):
    # Enforces docs/CLI.md:272.
    # Enforces docs/GLOSSARY.md:613.
    # Enforces website/src/learn/glossary.html:306.
    # Enforces website/src/pages/docs/quickstart.astro:118.
    # Enforces website/src/pages/docs/threat-model.astro:67.
    # Enforces website/src/pages/docs/cli.astro:99.
    with pytest.raises(errors.ArtifactError, match="no signing key set"):
        scmle.artifacts.bundle.Bundle.load(rehashed, trusted=True)


def test_the_invalid_signature_is_still_reported_with_the_key(rehashed):
    with pytest.raises(errors.ArtifactError, match="signature is invalid"):
        scmle.artifacts.bundle.Bundle.load(
            rehashed, trusted=True, signing_key="k"
        )


def test_package_refuses_a_signature_it_cannot_check(rehashed, tmp_path):
    code = cli.main(
        [
            "package",
            str(rehashed),
            "--trusted",
            "--output",
            str(tmp_path / "svc"),
        ]
    )
    assert code != cli.EXIT_OK
    assert not (tmp_path / "svc" / "serve.py").exists()


def test_trusted_production_load_is_refused_before_reading(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("SCMLE_ENV", " Production ")
    with pytest.raises(errors.ArtifactError, match="SCMLE_ENV=production"):
        scmle.artifacts.bundle.Bundle.load(tmp_path / "absent", trusted=True)
