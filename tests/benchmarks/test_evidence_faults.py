"""Claim C1 evidence: every registered gate blocks its injected fault."""

from typing import Any

import pytest

from benchmarks import acceptance
from benchmarks import registry
from benchmarks.evidence import faults
from scmle.contracts import errors


def records(outcome: str = "blocked") -> list[dict[str, Any]]:
    """One record per job of both parts, faults ending in `outcome`."""
    found = []
    for name, path, part in faults.jobs(registry.load()):
        level = faults.injections(part)[name].level
        base = {"gate": name, "path": path, "part": part, "level": level}
        if path == "control":
            found.append({**base, "outcome": "passed"})
            continue
        ending = "override" if path == "override" else outcome
        found.append({**base, "outcome": ending, "by": name})
    return found


def document(rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """A current document holding `rows`."""
    return {
        "floor": acceptance.load()[faults.VARIABLE].value,
        "code": faults.digest(),
        "records": records() if rows is None else rows,
    }


def real(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every real record a run can inject, ending blocked or overridden."""
    return [r for r in rows if r["part"] == "real" and r["level"] == "run"]


def test_every_registered_gate_has_an_injection_in_both_parts() -> None:
    assert set(faults.INJECTIONS) == set(registry.load())
    assert set(faults.REAL) == set(registry.load())


def test_every_overridable_gate_injects_its_override() -> None:
    for name, gate in registry.load().items():
        if faults.overridable(gate):
            assert faults.INJECTIONS[name].override is not None, name


def test_real_injections_go_through_runs_or_say_why_not() -> None:
    # Enforces docs/preregistration/c1.md:18.
    for name, injection in faults.REAL.items():
        if injection.fault is None:
            assert injection.reason, name
            assert injection.level == "none", name
        else:
            assert injection.level == "run", name


def test_every_fault_blocked_in_both_parts_still_fails_on_gaps() -> None:
    # Gates no run reaches on real data are `not injected` and count
    # against the real clause, so the row cannot pass while they exist.
    assert faults.problems(document()) == []
    text, verdict = faults.summary(document())
    gaps = [n for n, i in faults.REAL.items() if i.fault is None]
    assert verdict == ("fail" if gaps else "pass")
    assert "real data" in text and "simulated" in text


def test_the_row_passes_only_when_every_real_unit_is_satisfied() -> None:
    # Enforces docs/preregistration/c1.md:10.
    rows = [
        {**r, "level": "run"} if r["part"] == "real" else r for r in records()
    ]
    assert faults.verdicts(document(rows))["real"][1] == "pass"
    assert faults.summary(document(rows))[1] == "pass"


def test_a_real_unit_not_through_a_run_does_not_pass() -> None:
    rows = [
        {**r, "level": "run"} if r["part"] == "real" else r for r in records()
    ]
    first = next(i for i, r in enumerate(rows) if r["part"] == "real")
    rows[first] = {**rows[first], "level": "site"}
    counted = faults.tally(faults.select(rows, "real"))
    assert counted["satisfied"] == counted["units"] - 1
    assert faults.summary(document(rows))[1] == "fail"


def test_no_real_part_is_not_measured() -> None:
    rows = [r for r in records() if r["part"] != "real"]
    assert "faults: real part not measured" in faults.problems(document(rows))
    assert faults.summary(document(rows))[1] == "not measured"


def test_a_silent_simulated_fault_fails_the_row() -> None:
    # Enforces docs/preregistration/c1.md:10.
    rows = [
        {**r, "level": "run"} if r["part"] == "real" else r for r in records()
    ]
    first = next(
        i
        for i, r in enumerate(rows)
        if r["part"] == "simulated" and r["path"] == "fault"
    )
    rows[first] = {**rows[first], "outcome": "silent"}
    assert faults.verdicts(document(rows))["simulated"][1] == "fail"
    assert faults.summary(document(rows))[1] == "fail"
    simulated = faults.select(rows, "simulated")
    assert faults.tally(simulated)["outcomes"]["silent"] == 1


def test_a_gate_not_injected_counts_against_the_claim() -> None:
    # Enforces docs/preregistration/c1.md:18.
    rows = real(records())
    rows[0] = {**rows[0], "outcome": "not injected"}
    counted = faults.tally(rows)
    assert counted["satisfied"] == counted["units"] - 1


def test_a_missing_gate_is_not_measured() -> None:
    rows = [r for r in records() if r["gate"] != "vault.open"]
    issues = faults.problems(document(rows))
    assert "faults: vault.open fault not attempted (simulated)" in issues
    assert "faults: vault.open fault not attempted (real)" in issues
    assert faults.summary(document(rows))[1] == "not measured"


def test_a_missing_override_is_not_measured() -> None:
    rows = [
        r
        for r in records()
        if not (
            r["gate"] == "task.auto"
            and r["path"] == "override"
            and r["part"] == "real"
        )
    ]
    assert "faults: task.auto override not attempted (real)" in (
        faults.problems(document(rows))
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
    with pytest.raises(errors.InputError):
        faults.attempt("task.auto", "fault", "imagined")


def test_a_site_fault_is_recorded_as_blocked_by_its_gate() -> None:
    record = faults.attempt("ingest.stage.run", "fault")
    assert (record["outcome"], record["by"], record["level"]) == (
        "blocked",
        "ingest.stage.run",
        "site",
    )
    assert faults.attempt("ingest.stage.run", "control")["outcome"] == (
        "passed"
    )


def test_a_real_gate_without_a_run_is_recorded_not_injected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = {
        **faults.REAL,
        "task.auto": faults.absent("a condition", "no run reaches it"),
    }
    monkeypatch.setattr(faults, "REAL", table)
    record = faults.attempt("task.auto", "fault", "real")
    assert record["outcome"] == "not injected"
    assert record["reason"] == "no run reaches it"
    assert not faults.tally([record])["satisfied"]


@pytest.mark.parametrize("part", faults.PARTS)
def test_a_test_set_missing_a_class_is_blocked_by_its_gate(part: str) -> None:
    """Two minority rows used to complete with NaN ROC AUC and F1 of 1."""
    record = faults.attempt("split.test_support", "fault", part)
    assert (record["outcome"], record.get("by"), record["level"]) == (
        "blocked",
        "split.test_support",
        "run",
    )
    assert "test rows" in record["message"]
    control = faults.attempt("split.test_support", "control", part)
    assert control["outcome"] == "passed"


def test_a_real_fault_runs_on_a_cached_table() -> None:
    record = faults.attempt("task.regression", "fault", "real")
    assert (record["outcome"], record["level"], record["part"]) == (
        "blocked",
        "run",
        "real",
    )


@pytest.mark.parametrize(
    "name",
    [
        "task.survival",
        "split.treatment.stratify",
        "split.group.partition",
        "ingest.cli.unit_verdict",
        "split.resident.open",
    ],
)
def test_a_formerly_uninjected_real_fault_is_blocked_by_its_gate(
    name: str,
) -> None:
    # The churn table (survival, uplift), the house sales by zip code
    # (ranking, the ingest source) and a plugin stage reach these gates.
    record = faults.attempt(name, "fault", "real")
    assert (record["outcome"], record.get("by"), record["level"]) == (
        "blocked",
        name,
        "run",
    )


def test_the_stored_measurement_is_current() -> None:
    assert faults.problems() == []
