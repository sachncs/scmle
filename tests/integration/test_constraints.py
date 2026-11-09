"""Serving constraints in model selection."""

import json
import pathlib

import pytest
import support

from scmle.contracts import errors
from scmle.contracts import tasks


def run(frame, tmp_path, selection=None, **kwargs):
    return support.make_project(
        frame,
        "y",
        time_budget=600,
        output_dir=tmp_path,
        selection=selection,
        **kwargs,
    ).run()


def test_resources_are_always_measured_and_recorded(binary_frame, tmp_path):
    result = run(binary_frame, tmp_path)
    usage = result.evaluation["resources"]
    assert usage["model_mb"] > 0 and usage["p95_latency_ms"] > 0
    assert usage["rows_timed"] == 30 and "python" in usage["machine"]


def test_size_limit_forces_a_smaller_model(binary_frame, tmp_path):
    free = run(binary_frame, tmp_path / "free")
    limit = tasks.Selection(max_model_mb=0.05)
    constrained = run(binary_frame, tmp_path / "limited", limit)
    assert constrained.evaluation["resources"]["model_mb"] <= 0.05
    log = json.loads(
        (constrained.run_dir / "diagnostics" / "decision_log.json").read_text()
    )
    assert any("within serving limits" in o["reason"] for o in log["outcomes"])
    if free.evaluation["resources"]["model_mb"] > 0.05:
        rejected = [o for o in log["outcomes"] if "MB >" in o["reason"]]
        assert rejected and constrained.selected_model != free.selected_model


def test_latency_limit_is_enforced_and_reported(binary_frame, tmp_path):
    limit = tasks.Selection(max_p95_latency_ms=1000.0)
    result = run(binary_frame, tmp_path, limit)
    assert result.evaluation["resources"]["p95_latency_ms"] <= 1000.0


def test_impossible_constraints_block_the_run(binary_frame, tmp_path):
    limit = tasks.Selection(max_model_mb=1e-6, max_candidates_checked=3)
    with pytest.raises(errors.BlockedError, match="satisfies") as caught:
        run(binary_frame, tmp_path, limit)
    events_path = pathlib.Path(caught.value.details["run_dir"]) / "events.jsonl"
    lines = events_path.read_text().splitlines()
    assert json.loads(lines[-1])["kind"] == "state.BLOCKED"


def test_selection_is_unchanged_without_constraints(binary_frame, tmp_path):
    first = run(binary_frame, tmp_path / "a")
    second = run(binary_frame, tmp_path / "b", tasks.Selection())
    assert first.selected_model == second.selected_model
