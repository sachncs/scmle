"""Tests for the governance audit of a finished run directory."""

import json
import pathlib
from typing import Any

from scmle.governance import audit

RUN_ID = "run-1"


def stamp(index: int) -> str:
    """Return a strictly increasing timestamp for the given position."""
    return f"2026-01-01T00:00:{index:02d}+00:00"


def event(
    index: int,
    kind: str,
    reason: str = "",
    artifacts: tuple[str, ...] = (),
    run_id: str = RUN_ID,
) -> dict[str, Any]:
    """Return one event record as written to the event log."""
    return {
        "run_id": run_id,
        "kind": kind,
        "reason": reason,
        "artifacts": list(artifacts),
        "timestamp": stamp(index),
    }


def write_log(root: pathlib.Path, records: list[dict[str, Any]]) -> None:
    """Write records as a JSON-lines event log inside the run directory."""
    lines = [json.dumps(record) for record in records]
    (root / audit.EVENTS_FILE).write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def lifecycle(final: str = "COMPLETED") -> list[dict[str, Any]]:
    """Return a legal state lifecycle ending in the given state."""
    names = [
        "CREATED",
        "VALIDATING",
        "PROFILING",
        "PLANNING",
        "TRAINING",
        "SELECTING",
        "TESTING",
        "PACKAGING",
    ]
    if final != "COMPLETED":
        names = names[:5]
    names.append(final)
    return [event(i, f"state.{name}") for i, name in enumerate(names)]


def codes(report: audit.Audit) -> list[str]:
    """Return the finding codes of a report."""
    return [finding.code for finding in report.findings]


def test_missing_event_log_is_reported(tmp_path):
    report = audit.RunAuditor().audit(tmp_path)

    assert not report.ok
    assert report.run_id == tmp_path.name
    assert report.events == 0
    assert codes(report) == ["log.missing"]


def test_clean_run_has_no_findings(tmp_path):
    write_log(tmp_path, lifecycle())

    report = audit.RunAuditor().audit(str(tmp_path))

    assert report.ok
    assert report.events == 9
    assert report.findings == ()


def test_blank_lines_in_log_are_ignored(tmp_path):
    path = tmp_path / audit.EVENTS_FILE
    path.write_text(
        "\n" + json.dumps(event(0, "state.CREATED")) + "\n   \n",
        encoding="utf-8",
    )

    report = audit.RunAuditor().audit(tmp_path)

    assert report.events == 1
    assert codes(report) == ["state.unfinished"]


def test_backwards_timestamps_and_mixed_runs_are_found(tmp_path):
    records = lifecycle()
    records[3]["timestamp"] = stamp(0)
    records[4]["run_id"] = "other"
    write_log(tmp_path, records)

    report = audit.RunAuditor().audit(tmp_path)

    assert "log.order" in codes(report)
    mixed = [f for f in report.findings if f.code == "log.runs"]
    assert mixed and "other" in mixed[0].message


def test_skipped_state_is_found(tmp_path):
    records = [
        event(0, "state.CREATED"),
        event(1, "state.PROFILING"),
        event(2, "state.FAILED"),
    ]
    write_log(tmp_path, records)

    report = audit.RunAuditor().audit(tmp_path)

    assert codes(report) == ["state.order"]
    assert "skips a step" in report.findings[0].message


def test_abort_state_is_allowed_from_any_step(tmp_path):
    write_log(tmp_path, lifecycle("CANCELLED"))

    assert audit.RunAuditor().audit(tmp_path).ok


def test_state_after_terminal_is_found(tmp_path):
    records = [*lifecycle(), event(20, "state.TRAINING")]
    write_log(tmp_path, records)

    report = audit.RunAuditor().audit(tmp_path)

    assert "state.order" in codes(report)


def test_first_state_must_be_created(tmp_path):
    write_log(tmp_path, [event(0, "state.TRAINING"), event(1, "state.FAILED")])

    report = audit.RunAuditor().audit(tmp_path)

    assert codes(report) == ["state.order"]
    assert "None" in report.findings[0].message


def test_unfinished_run_is_found(tmp_path):
    write_log(tmp_path, lifecycle()[:4])

    report = audit.RunAuditor().audit(tmp_path)

    assert codes(report) == ["state.unfinished"]


def test_resume_restarts_the_lifecycle(tmp_path):
    records = [
        event(0, "state.CREATED"),
        event(1, "state.VALIDATING"),
        event(2, "state.FAILED"),
        event(3, "state.CREATED", reason=f"{audit.RESUME_REASON} (retry)"),
        *[
            event(4 + i, record["kind"])
            for i, record in enumerate(lifecycle()[1:])
        ],
    ]
    write_log(tmp_path, records)

    assert audit.RunAuditor().audit(tmp_path).ok


def test_restart_without_resume_reason_is_found(tmp_path):
    records = [
        event(0, "state.CREATED"),
        event(1, "state.FAILED"),
        event(2, "state.CREATED", reason="again"),
        event(3, "state.FAILED"),
    ]
    write_log(tmp_path, records)

    assert "state.order" in codes(audit.RunAuditor().audit(tmp_path))


def test_test_set_opened_correctly_is_clean(tmp_path):
    records = lifecycle()
    records[5:5] = [event(0, "trial.ok"), event(0, "selection.frozen")]
    records.insert(8, event(0, audit.TEST_OPENED))
    for i, record in enumerate(records):
        record["timestamp"] = stamp(i)
    write_log(tmp_path, records)

    assert audit.RunAuditor().audit(tmp_path).ok


def test_test_set_reopened_before_selection_and_after_trials(tmp_path):
    records = lifecycle()
    records += [
        event(20, audit.TEST_OPENED),
        event(21, "trial.ok"),
        event(22, audit.TEST_OPENED),
    ]
    write_log(tmp_path, records)

    report = audit.RunAuditor().audit(tmp_path)

    found = codes(report)
    assert "test.reopened" in found
    assert "test.before_selection" in found
    assert "test.before_trials" in found
    reopened = [f for f in report.findings if f.code == "test.reopened"]
    assert "2 times" in reopened[0].message


def test_selection_frozen_after_opening_is_found(tmp_path):
    records = [
        *lifecycle(),
        event(20, audit.TEST_OPENED),
        event(21, audit.SELECTION_FROZEN),
    ]
    write_log(tmp_path, records)

    assert "test.before_selection" in codes(audit.RunAuditor().audit(tmp_path))


def test_missing_artifact_is_reported_once(tmp_path):
    (tmp_path / "present.json").write_text("{}", encoding="utf-8")
    records = lifecycle()
    records[1]["artifacts"] = ["gone.json", "present.json"]
    records[2]["artifacts"] = ["gone.json"]
    write_log(tmp_path, records)

    report = audit.RunAuditor().audit(tmp_path)

    assert codes(report) == ["artifact.missing"]
    assert "gone.json" in report.findings[0].message


def test_decisions_combine_planner_log_and_events(tmp_path):
    log = tmp_path / audit.DECISION_LOG
    log.parent.mkdir()
    log.write_text(
        json.dumps({"plan": [{"step": "scale", "why": "skew"}]}),
        encoding="utf-8",
    )
    records = lifecycle()
    records += [
        event(20, "strategy.chosen", reason="cheap"),
        event(21, "ensemble.built", reason="stack"),
        event(22, "plan.created", reason="plan"),
        event(23, "approval.granted", reason="ok"),
        event(24, audit.SELECTION_FROZEN, reason="frozen"),
        event(25, audit.TEST_OPENED, reason="final"),
        event(26, "trial.ok", reason="ignored"),
    ]
    write_log(tmp_path, records)

    report = audit.RunAuditor().audit(tmp_path)

    assert report.decisions[0] == {
        "source": "planner",
        "step": "scale",
        "why": "skew",
    }
    sources = [d["source"] for d in report.decisions[1:]]
    assert sources == [
        "strategy.chosen",
        "ensemble.built",
        "plan.created",
        "approval.granted",
        audit.SELECTION_FROZEN,
        audit.TEST_OPENED,
    ]
    assert report.decisions[1]["reason"] == "cheap"
    assert report.decisions[1]["timestamp"] == stamp(20)


def test_decision_log_without_plan_adds_nothing(tmp_path):
    log = tmp_path / audit.DECISION_LOG
    log.parent.mkdir()
    log.write_text("{}", encoding="utf-8")
    write_log(tmp_path, lifecycle())

    assert audit.RunAuditor().audit(tmp_path).decisions == ()
