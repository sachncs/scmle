"""Guarantees in docs/GLOSSARY.md that no other test enforces."""

import numpy
import pandas
import pytest

from scmle import cli
from scmle.artifacts import bundle
from scmle.contracts import data
from scmle.contracts import errors
from scmle.contracts import signing
from scmle.contracts import tasks
from scmle.data.curation import preprocessing
from scmle.models import registry

PLAN = data.Blueprint(("num",), ("cat",))


@pytest.fixture
def trained():
    """A small fitted bundle and the features it was trained on."""
    rng = numpy.random.default_rng(0)
    features = pandas.DataFrame(
        {"num": rng.normal(size=120), "cat": rng.choice(["a", "b"], 120)}
    )
    target = pandas.Series((features.num > 0).astype(int))
    model = registry.Catalog.stock().get("logistic_regression")
    pipe = preprocessing.instantiate(PLAN, model, tasks.Task.BINARY, 0).fit(
        features, target
    )
    made = bundle.Bundle.create(
        pipe, PLAN, features, "y", "binary_classification", "roc_auc", "lr"
    )
    return made, features


def test_a_signed_bundle_without_the_key_is_refused_even_when_trusted(
    trained, tmp_path, monkeypatch
):
    """Enforces docs/GLOSSARY.md:729."""
    monkeypatch.delenv(signing.SIGNING_KEY_ENV, raising=False)
    monkeypatch.delenv(signing.SIGNING_KEY_FILE_ENV, raising=False)
    monkeypatch.delenv("SCMLE_ENV", raising=False)
    made, features = trained
    folder = made.save(tmp_path / "m", signing_key="glossary-key")
    with pytest.raises(errors.ArtifactError, match="unauthenticated"):
        bundle.Bundle.load(folder)
    with pytest.raises(errors.ArtifactError, match="no signing key set"):
        bundle.Bundle.load(folder, trusted=True)
    key = tmp_path / "key"
    key.write_text("glossary-key", encoding="utf-8")
    monkeypatch.setenv(signing.SIGNING_KEY_FILE_ENV, str(key))
    assert len(bundle.Bundle.load(folder).predict(features)) == len(features)


def test_a_blocked_release_exits_with_code_two():
    """Enforces docs/GLOSSARY.md:700 (the code the gate's refusal uses)."""
    assert cli.EXIT_BLOCKED == 2
