"""Guarantees of docs/START_HERE.md that no other test enforced."""

import logging

import numpy
import pandas
import pytest

from scmle import cli
import scmle.artifacts.bundle
from scmle.contracts import data
from scmle.contracts import signing
from scmle.contracts import tasks
import scmle.models.registry
from scmle.preparation import preprocessing

PLAN = data.Blueprint(("num",), ("cat",))


@pytest.fixture
def signed(tmp_path):
    """A small bundle signed with the key `demo-key`, with its input rows.

    Args:
        tmp_path: Scratch directory.

    Returns:
        The bundle directory and a CSV of rows it can score.
    """
    rng = numpy.random.default_rng(0)
    features = pandas.DataFrame(
        {"num": rng.normal(size=120), "cat": rng.choice(["a", "b"], 120)}
    )
    target = pandas.Series((features.num > 0).astype(int))
    catalog = scmle.models.registry.Catalog.stock()
    pipe = preprocessing.instantiate(
        PLAN, catalog.get("logistic_regression"), tasks.Task.BINARY, 0
    ).fit(features, target)
    bundle = scmle.artifacts.bundle.Bundle.create(
        pipe, PLAN, features, "y", "binary_classification", "roc_auc", "lr"
    )
    directory = bundle.save(tmp_path / "model", signing_key="demo-key")
    rows = tmp_path / "rows.csv"
    features.head(5).to_csv(rows, index=False)
    return directory, rows


def test_predict_refuses_a_wrong_key_and_missing_columns_with_exit_one(
    signed, tmp_path, monkeypatch, caplog
):
    """Enforces docs/START_HERE.md:521.

    Also enforces website/src/learn/start-here.html:265.
    Also enforces website/src/pages/docs/quickstart.astro:113.
    """
    directory, rows = signed
    caplog.set_level(logging.INFO)
    monkeypatch.setenv(signing.SIGNING_KEY_ENV, "wrong-key")
    assert cli.main(["predict", str(directory), str(rows)]) == 1
    assert "artifact error: bundle signature is invalid" in caplog.text
    caplog.clear()
    monkeypatch.setenv(signing.SIGNING_KEY_ENV, "demo-key")
    bad = tmp_path / "bad.csv"
    bad.write_text("class,sex\n1,m\n")
    assert cli.main(["predict", str(directory), str(bad)]) == 1
    assert "schema error: input is missing required columns" in caplog.text
