"""Audit of run directories against governance invariants."""

import json

import support

from scmle.governance import audit


def finish(binary_frame, tmp_path):
    return support.make_project(
        binary_frame, "y", time_budget=300, output_dir=tmp_path
    ).run()


def rewrite(path, edit):
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    lines = edit(lines)
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n")


def test_a_clean_run_passes_and_lists_its_decisions(binary_frame, tmp_path):
    result = finish(binary_frame, tmp_path)

    report = audit.RunAuditor().audit(result.run_dir)

    assert report.ok, report.findings
    sources = {d["source"] for d in report.decisions}
    assert {"planner", "selection.frozen", "test.opened"} <= sources


def test_opening_the_test_set_twice_is_a_finding(binary_frame, tmp_path):
    result = finish(binary_frame, tmp_path)
    events = result.run_dir / "events.jsonl"
    rewrite(
        events,
        lambda lines: (
            [*lines, next(r for r in lines if r["kind"] == "test.opened")]
        ),
    )

    codes = {f.code for f in audit.RunAuditor().audit(result.run_dir).findings}

    assert "test.reopened" in codes


def test_a_test_set_opened_before_selection_is_a_finding(
    binary_frame, tmp_path
):
    result = finish(binary_frame, tmp_path)

    def swap(lines):
        opened = next(r for r in lines if r["kind"] == "test.opened")
        rest = [r for r in lines if r is not opened]
        first_trial = next(
            i for i, r in enumerate(rest) if r["kind"].startswith("trial.")
        )
        return [*rest[:first_trial], opened, *rest[first_trial:]]

    rewrite(result.run_dir / "events.jsonl", swap)

    codes = {f.code for f in audit.RunAuditor().audit(result.run_dir).findings}

    assert {"test.before_selection", "test.before_trials"} <= codes


def test_missing_artifacts_and_skipped_states_are_findings(
    binary_frame, tmp_path
):
    result = finish(binary_frame, tmp_path)
    (result.run_dir / "dataset_profile.json").unlink()
    rewrite(
        result.run_dir / "events.jsonl",
        lambda lines: [r for r in lines if r["kind"] != "state.PLANNING"],
    )

    codes = {f.code for f in audit.RunAuditor().audit(result.run_dir).findings}

    assert {"artifact.missing", "state.order"} <= codes


def test_a_run_without_a_log_is_reported(tmp_path):
    report = audit.RunAuditor().audit(tmp_path)

    assert not report.ok and report.findings[0].code == "log.missing"
