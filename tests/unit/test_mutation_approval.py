"""Exact audit records, history and errors of hand transitions and approvals."""

import json

import pytest

from scmle.contracts import errors
from scmle.deploy.release_state import approval
from scmle.deploy.release_state import status
from tests import support_release
from tests import support_runtime

MODEL = "churn"


def audits(store):
    lines = (store.root / "audit.jsonl").read_text(encoding="utf-8")
    rows = [json.loads(line) for line in lines.splitlines()]
    return [{k: v for k, v in row.items() if k != "timestamp"} for row in rows]


def test_an_approval_records_its_history_and_audit(tmp_path):
    store = support_runtime.make_registry(tmp_path)
    gate = approval.Approval(store)

    gate.approve(MODEL, 1, support_runtime.sanction(1), support_release.KEY)

    (step,) = gate.read()["models"][MODEL]["versions"]["1"]["history"]
    assert (step["from"], step["to"], step["reason"]) == (
        "candidate",
        "approved",
        "approved",
    )
    assert audits(store)[-1] == {
        "action": "release_state",
        "model": MODEL,
        "version": 1,
        "state": "approved",
        "reviewer": "reviewer-1",
    }


def test_a_hand_transition_is_audited_with_its_reason(tmp_path):
    store = support_runtime.make_registry(tmp_path)
    gate = approval.Approval(store)

    gate.transition(MODEL, 3, status.ReleaseState.CHALLENGER, "compare")

    assert audits(store)[-1] == {
        "action": "release_state",
        "model": MODEL,
        "version": 3,
        "state": "challenger",
        "reason": "compare",
    }


def test_a_rejected_approval_keeps_the_reason_of_its_rejection(tmp_path):
    store = support_runtime.make_registry(tmp_path)
    gate = approval.Approval(store)
    sanction = support_runtime.sanction(2)
    record = store.resolve(MODEL, 3)
    with pytest.raises((TypeError, ValueError)) as expected:
        sanction.validate(MODEL, 3, record.content_id, support_release.KEY)

    with pytest.raises(errors.ArtifactError) as caught:
        gate.approve(MODEL, 3, sanction, support_release.KEY)

    assert str(caught.value) == str(expected.value)
    assert str(caught.value) != "None"
