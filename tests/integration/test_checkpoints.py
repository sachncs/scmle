"""Human-approval gates inside the orchestrator."""

import json
import pathlib

import pytest
import support

import scmle
from scmle.contracts import errors
from scmle.core import policies
from scmle.core import state


def project(frame, tmp_path, *gates):
    return support.make_project(
        frame,
        "y",
        time_budget=300,
        output_dir=tmp_path,
        checkpoints=tuple(gates),
    )


def kinds(run_dir):
    lines = (pathlib.Path(run_dir) / "events.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines]


def test_auto_approval_runs_and_audits_both_gates(binary_frame, tmp_path):
    result = project(binary_frame, tmp_path, policies.AutoApprove()).run()
    events = kinds(result.run_dir)
    approvals = [e for e in events if e["kind"] == "policy.approved"]
    assert [a["data"]["gate"] for a in approvals] == ["PLANNING", "SELECTING"]
    assert all(a["data"]["reviewer"] == "auto" for a in approvals)


def test_reviewers_see_the_plan_and_the_selection(binary_frame, tmp_path):
    seen = []

    def decide(review):
        seen.append(review)
        return True

    project(binary_frame, tmp_path, policies.CallbackCheckpoint(decide)).run()
    plan_review, selection_review = seen
    assert plan_review.gate is state.RunState.PLANNING
    assert "Candidates:" in plan_review.summary
    assert plan_review.details["candidate_models"]
    assert selection_review.gate is state.RunState.SELECTING
    assert "test set is still locked" in selection_review.summary
    assert selection_review.details["status"] == "ok"


def test_rejecting_the_plan_stops_before_any_training(binary_frame, tmp_path):
    reject = policies.CallbackCheckpoint(
        lambda review: policies.Approval(False, "alice", "too many models"),
        gates=frozenset({policies.PLAN_GATE}),
    )
    with pytest.raises(
        errors.BlockedError, match="alice: too many models"
    ) as caught:
        project(binary_frame, tmp_path, reject).run()
    run_dir = pathlib.Path(caught.value.details["run_dir"])
    events = kinds(run_dir)
    assert events[-1]["kind"] == "state.BLOCKED"
    assert any(
        e["kind"] == "policy.rejected" and e["data"]["reviewer"] == "alice"
        for e in events
    )
    assert not (run_dir / "trials").exists()


def test_rejecting_the_selection_keeps_the_test_set_locked(
    binary_frame, tmp_path
):
    reject = policies.CallbackCheckpoint(
        lambda review: review.gate is not policies.SELECTION_GATE
    )
    with pytest.raises(
        errors.BlockedError, match="SELECTING not approved"
    ) as caught:
        project(binary_frame, tmp_path, reject).run()
    run_dir = pathlib.Path(caught.value.details["run_dir"])
    names = [e["kind"] for e in kinds(run_dir)]
    assert "selection.frozen" in names
    assert "test.opened" not in names
    assert not (run_dir / "evaluation.json").exists()


def test_every_checkpoint_must_approve(binary_frame, tmp_path):
    yes = policies.CallbackCheckpoint(lambda review: True)
    no = policies.CallbackCheckpoint(
        lambda review: policies.Approval(False, "bob", "veto"),
        gates=frozenset({policies.SELECTION_GATE}),
    )
    with pytest.raises(errors.BlockedError, match="bob"):
        project(binary_frame, tmp_path, yes, no).run()


def test_a_blocked_run_can_be_resumed_once_approved(binary_frame, tmp_path):
    gate = {"open": False}
    reviewer = policies.CallbackCheckpoint(lambda review: gate["open"])
    first = project(binary_frame, tmp_path, reviewer)
    with pytest.raises(errors.BlockedError) as caught:
        first.run()
    gate["open"] = True
    resumed = first.orchestrator.resume(caught.value.details["run_dir"])
    assert resumed.state == "COMPLETED"
    names = [e["kind"] for e in kinds(resumed.run_dir)]
    assert "policy.rejected" in names and names.count("test.opened") == 1


def test_checkpoints_do_not_change_results(binary_frame, tmp_path):
    plain = project(binary_frame, tmp_path / "a").run()
    gated = project(binary_frame, tmp_path / "b", policies.AutoApprove()).run()
    assert gated.selected_model == plain.selected_model
    assert gated.evaluation["selected_cv_score"] == pytest.approx(
        plain.evaluation["selected_cv_score"]
    )
    assert isinstance(scmle.MLProject, type)
