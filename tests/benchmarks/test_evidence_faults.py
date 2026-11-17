"""Claim C1 evidence: every registered gate blocks its injected fault."""

from typing import Any

import pytest

from benchmarks import acceptance
from benchmarks import registry
from benchmarks.evidence import faults
from scmle.contracts import errors


def records(outcome: str = "blocked") -> list[dict[str, Any]]:
    """One record per job, faults and overrides ending in `outcome`."""
    found = []
    for name, path in faults.jobs(registry.load()):
        if path == "control":
            found.append({"gate": name, "path": path, "outcome": "passed"})
            continue
        ending = "override" if path == "override" else outcome
        found.append(
            {"gate": name, "path": path, "outcome": ending, "by": name}
        )
    return found


def document(rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """A current document holding `rows`."""
    return {
        "floor": acceptance.load()[faults.VARIABLE].value,
        "code": faults.digest(),
        "records": records() if rows is None else rows,
    }


def test_every_registered_gate_has_an_injection() -> None:
    assert set(faults.INJECTIONS) == set(registry.load())


def test_every_overridable_gate_injects_its_override() -> None:
    for name, gate in registry.load().items():
        if faults.overridable(gate):
            assert faults.INJECTIONS[name].override is not None, name


def test_every_fault_blocked_passes() -> None:
    assert faults.problems(document()) == []
    text, verdict = faults.summary(document())
    assert verdict == "pass"
    assert "not real data" in text


def test_one_silent_fault_fails_the_row() -> None:
    rows = records()
    rows[0] = {**rows[0], "outcome": "silent"}
    assert faults.summary(document(rows))[1] == "fail"
    assert faults.tally(rows)["outcomes"]["silent"] == 1


def test_a_gate_not_injected_counts_against_the_claim() -> None:
    rows = records()
    rows[0] = {**rows[0], "outcome": "not injected"}
    counted = faults.tally(rows)
    assert counted["satisfied"] == counted["units"] - 1
    assert faults.summary(document(rows))[1] == "fail"


def test_a_missing_gate_is_not_measured() -> None:
    rows = [r for r in records() if r["gate"] != "vault.open"]
    issues = faults.problems(document(rows))
    assert "faults: vault.open fault not attempted" in issues
    assert faults.summary(document(rows))[1] == "not measured"


def test_a_missing_override_is_not_measured() -> None:
    rows = [
        r
        for r in records()
        if not (r["gate"] == "task.auto" and r["path"] == "override")
    ]
    assert "faults: task.auto override not attempted" in faults.problems(
        document(rows)
    )


def test_stale_code_is_not_measured() -> None:
    stale = {**document(), "code": "0" * 64}
    assert faults.summary(stale)[1] == "not measured"


def test_a_block_by_another_gate_is_reported() -> None:
    rows = records()
    rows[0] = {**rows[0], "by": "task.auto"}
    assert faults.tally(rows)["elsewhere"] == [
        f"{rows[0]['gate']} blocked by task.auto"
    ]


def test_an_unknown_injection_is_an_input_error() -> None:
    with pytest.raises(errors.InputError):
        faults.attempt("no.such.gate", "fault")


def test_a_site_fault_is_recorded_as_blocked_by_its_gate() -> None:
    record = faults.attempt("ingest.multiplicity", "fault")
    assert (record["outcome"], record["by"]) == (
        "blocked",
        "ingest.multiplicity",
    )
    assert faults.attempt("ingest.multiplicity", "control")["outcome"] == (
        "passed"
    )


def test_the_stored_measurement_is_current() -> None:
    assert faults.problems() == []
