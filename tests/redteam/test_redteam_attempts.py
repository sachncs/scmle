"""Claim C8: a regression test per bypass attempt (benchmarks/evidence/bypass).

Each attempt in `bypass.ATTEMPTS` runs end to end here and must end the way
its control promises. It is `blocked` by the control it attacks, or by one
it lists in `Attempt.also` (`Attempt.stops`). An attempt that relies on a
documented override may also go through it (`override`). It is never
`bypass` and never `error`: an attack that did not reach its control fails
here, whatever stopped it. An attempt that got through when it was added
names its defect and is a strict xfail that expects only `Bypassed`. It
fails the suite the day the control holds, so the defect field is removed
with the fix. It also fails if the attack breaks or is stopped by
something other than its control.
"""

import os
import pathlib

import pytest

from benchmarks.evidence import bypass


class Bypassed(Exception):  # noqa: N818 -- the finding, not a fault
    """An attempt got through its control with no documented override."""


def cases() -> list:
    """One case per attempt; a known defect is a strict xfail on `Bypassed`."""
    found = []
    for name, attempt in sorted(bypass.ATTEMPTS.items()):
        marks = []
        if attempt.defect:
            marks.append(
                pytest.mark.xfail(
                    strict=True,
                    raises=Bypassed,
                    reason=f"bypass: {attempt.defect}",
                )
            )
        found.append(pytest.param(name, marks=marks, id=name))
    return found


@pytest.fixture(scope="session")
def signed(tmp_path_factory) -> pathlib.Path:
    """The signed bundle the CLI attempts copy and tamper with."""
    return bypass.fixture(tmp_path_factory.mktemp("signed"))


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    """No `SCMLE_` variable of the caller's shell reaches an attempt."""
    for name in list(os.environ):
        if name.startswith("SCMLE_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("OMP_NUM_THREADS", "1")


@pytest.mark.parametrize("name", cases())
def test_the_control_holds(name, signed):
    attempt = bypass.ATTEMPTS[name]
    record = bypass.execute(name, str(signed))
    assert record["outcome"] != "error", record
    if record["outcome"] == "bypass":
        raise Bypassed(record)
    if record["outcome"] == "override":
        assert attempt.override, record
        return
    assert record["outcome"] == "blocked", record
    assert record["stopped"] in attempt.stops, record


def test_every_known_bypass_names_existing_product_code():
    root = pathlib.Path(bypass.__file__).resolve().parents[2]
    known = {n: a.defect for n, a in bypass.ATTEMPTS.items() if a.defect}
    assert sorted(known) == [
        "charter_rerun_elsewhere",
        "charter_rerun_forgotten",
        "checkpoint_approved",
        "checkpoint_rechained",
        "checkpoint_restarted",
        "journal_rechained",
        "journal_restarted",
        "journal_truncated",
        "pickle_checkpoint_trusted_nokey",
        "plugin_peek",
    ]
    for defect in known.values():
        assert (root / defect.split(":", 1)[0]).is_file(), defect


def test_a_known_bypass_is_never_also_an_override():
    both = [n for n, a in bypass.ATTEMPTS.items() if a.defect and a.override]
    assert both == []


def test_every_defence_in_depth_is_another_control():
    same = [n for n, a in bypass.ATTEMPTS.items() if a.gate in a.also]
    assert same == []
