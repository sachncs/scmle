"""`block` stops every flagged column and says what it could not see (ERR-1).

Before, a stale or missing certificate made every mechanism descriptive, so
no suspicion could block and `leakage_policy="block"` trained as if the
screen had passed. Now every suspect may block, because its false-alarm
control holds by construction, and the screen probes each run's rows with
planted leaks, so a clean result states which kinds of leak it would have
missed there. Certificates are evidence and decide nothing.
"""

import logging

import numpy
import pandas
import pytest
import support

from scmle.assurance.admissibility import plants
from scmle.assurance.admissibility import screen
from scmle.contracts import errors
from scmle.pipeline import stages


def table(rows=240, seed=0, leak=False):
    rng = numpy.random.default_rng(seed)
    frame = pandas.DataFrame({f"x{i}": rng.normal(size=rows) for i in range(4)})
    frame["y"] = (frame["x0"] + rng.normal(size=rows) > 0).astype(int)
    if leak:
        frame["leak"] = frame["y"] * 1.0
    return frame


def build(tmp_path, policy, frame):
    return support.make_project(frame, "y", directory=tmp_path, leakage=policy)


def test_block_stops_a_flagged_column_even_with_a_stale_certificate(
    monkeypatch, tmp_path
):
    # Enforces docs/ARCHITECTURE.md:165.
    # Enforces docs/API_STABILITY.md:54 and docs/ARTIFACTS.md:32.
    # Enforces docs/WHAT_SCMLE_CANNOT_DO.md:156.
    # Enforces website/src/learn/what-scmle-cannot-do.html:36-37.
    # Enforces website/src/pages/docs/api-stability.astro:64.
    # Enforces website/src/pages/docs/artifacts.astro:15.
    # Enforces website/src/pages/docs/qualification.astro:22.
    # Enforces website/src/pages/index.astro:149.
    monkeypatch.setattr(screen, "fingerprint", lambda: "0" * 64)
    with pytest.raises(errors.BlockedError, match="leak") as raised:
        build(tmp_path, "block", table(leak=True)).run()
    assert raised.value.details["leakage"]["blocking"] == ["leak"]
    assert any("stale" in n for n in raised.value.details["leakage"]["notes"])


def test_block_runs_a_clean_table_and_records_the_probe(tmp_path):
    result = build(tmp_path, "block", table(rows=600)).run()
    assert set(result.profile.leakage["probe"]) == set(plants.KINDS)
    assert set(result.profile.leakage["gates"].values()) == {"gating"}


def test_a_missed_plant_is_said_loudly_and_does_not_refuse(
    monkeypatch, tmp_path, caplog
):
    found = {**dict.fromkeys(plants.KINDS, True), "leak_rare_record": False}
    monkeypatch.setattr(screen, "probe", lambda *a, **k: found)
    with caplog.at_level(logging.WARNING, logger=stages.__name__):
        result = build(tmp_path, "block", table()).run()
    assert result.profile.leakage["probe"]["leak_rare_record"] is False
    assert any(
        "leak_rare_record" in r.getMessage()
        and "may remain undetected" in r.getMessage()
        and r.levelno == logging.WARNING
        for r in caplog.records
    )
