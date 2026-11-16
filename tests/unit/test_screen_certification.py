"""A screen without a current certificate cannot pass `block` silently (ERR-1).

Before the fix, a stale or missing certificate made every mechanism
descriptive, so no suspicion could block and `leakage_policy="block"` trained
as if the screen had passed.
"""

import dataclasses
import logging

import numpy
import pandas
import pytest
import support

from scmle.contracts import errors
from scmle.core import stages
from scmle.data.admissibility import screen
from scmle.data.detectors import certificates


def table(seed=0, rows=240):
    rng = numpy.random.default_rng(seed)
    frame = pandas.DataFrame({f"x{i}": rng.normal(size=rows) for i in range(4)})
    frame["y"] = (frame["x0"] + rng.normal(size=rows) > 0).astype(int)
    return frame


def stale():
    """The shipped store, reissued for code that is not the current code."""
    shipped = screen.load_store()
    return certificates.Store(
        dataclasses.replace(c, code_hash="0" * 64) for c in shipped.certificates
    )


def build(tmp_path, policy):
    return support.make_project(
        table(), "y", output_dir=tmp_path, leakage_policy=policy
    )


def test_the_shipped_certificate_is_current():
    assert stages.certification() == "not passing"


def test_block_refuses_a_stale_certificate(monkeypatch, tmp_path):
    store = stale()
    monkeypatch.setattr(screen, "shipped_store", lambda: store)
    assert stages.certification() == "stale"
    with pytest.raises(
        errors.BlockedError, match="certificate stale"
    ) as raised:
        build(tmp_path, "block").run()
    assert "'warn'" in str(raised.value)
    assert set(raised.value.details["leakage"]["gates"].values()) == {
        "descriptive"
    }


def test_block_refuses_a_missing_certificate(monkeypatch, tmp_path):
    monkeypatch.setattr(screen, "shipped_store", lambda: None)
    with pytest.raises(errors.BlockedError, match="certificate missing"):
        build(tmp_path, "block").run()


def test_warn_proceeds_and_says_so(monkeypatch, tmp_path, caplog):
    store = stale()
    monkeypatch.setattr(screen, "shipped_store", lambda: store)
    with caplog.at_level(logging.WARNING, logger=stages.__name__):
        build(tmp_path, "warn").run()
    assert any(
        "certificate stale" in r.getMessage() and r.levelno == logging.WARNING
        for r in caplog.records
    )


def test_block_runs_with_the_shipped_certificate(tmp_path):
    result = build(tmp_path, "block").run()
    assert result.profile.leakage["gates"]
    assert set(result.profile.leakage["gates"].values()) == {"gating"}
