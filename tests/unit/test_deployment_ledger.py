import json

import pytest

from scmle.governance import deployment_ledger

KEY = "human-key-not-held-by-the-system"
SYSTEM_KEY = "system-key"


def make(path=None):
    return deployment_ledger.DeploymentLedger.create(
        path,
        charter_hash="c" * 64,
        policy={"r_star": 0.05},
        alpha_promo="1/20",
        system_keys=[deployment_ledger.key_fingerprint(SYSTEM_KEY)],
    )


def escalate(ledger, action="promote"):
    question = {"action": action, "detail": {"version": 3}}
    number = len(ledger.rows)
    ledger.commit(
        deployment_ledger.ESCALATION,
        {
            "phase": deployment_ledger.PHASE_OPEN,
            "id": f"esc-{number}",
            "question": question,
            "question_hash": deployment_ledger.question_hash(question),
            "evidence": ["alarm-1"],
            "proposed_default": "keep_champion",
        },
    )
    return f"esc-{number}", deployment_ledger.question_hash(question)


def reply(identity, digest, key=KEY, **changes):
    fields = {
        "escalation_id": identity,
        "question_hash": digest,
        "decision": "approve",
        "statement": "reviewed the evidence",
        "evidence": "ticket-17",
        "role": "model owner",
        "name_or_id": "A. Person",
        "date": "2026-11-01",
        "scope": "v3:stream-a",
        "expires": "2026-12-01",
    }
    fields.update(changes)
    return deployment_ledger.Attestation.create(key, **fields)


def filled(tmp_path):
    ledger = make(tmp_path / deployment_ledger.FILE_NAME)
    ledger.commit(deployment_ledger.OBSERVE_SUMMARY, {"batch": 0})
    ledger.commit(deployment_ledger.ALARM, {"batch": 1})
    escalate(ledger)
    ledger.commit(deployment_ledger.OBSERVE_SUMMARY, {"batch": 2})
    return ledger


def test_a_valid_ledger_loads_and_has_the_open_row_first(tmp_path):
    ledger = filled(tmp_path)
    again = deployment_ledger.DeploymentLedger.load(
        tmp_path / deployment_ledger.FILE_NAME
    )
    assert again.head == ledger.head
    assert again.rows[0]["kind"] == deployment_ledger.OPEN
    assert len(again.pending()) == 1


def rewrite(tmp_path, change):
    path = tmp_path / deployment_ledger.FILE_NAME
    lines = path.read_text().splitlines()
    path.write_text("\n".join(change(lines)) + "\n")


def test_deleting_a_middle_row_fails_verification(tmp_path):
    filled(tmp_path)
    rewrite(tmp_path, lambda lines: lines[:2] + lines[3:])
    with pytest.raises(
        deployment_ledger.DeploymentLedgerError, match="chain broken"
    ):
        deployment_ledger.DeploymentLedger.load(
            tmp_path / deployment_ledger.FILE_NAME
        )


def test_reordering_rows_fails_verification(tmp_path):
    filled(tmp_path)
    rewrite(tmp_path, lambda lines: [lines[0], lines[2], lines[1], *lines[3:]])
    with pytest.raises(
        deployment_ledger.DeploymentLedgerError, match="chain broken"
    ):
        deployment_ledger.DeploymentLedger.load(
            tmp_path / deployment_ledger.FILE_NAME
        )


def test_editing_a_row_fails_verification(tmp_path):
    filled(tmp_path)

    def edit(lines):
        row = json.loads(lines[1])
        row["body"]["batch"] = 99
        return [lines[0], json.dumps(row, sort_keys=True), *lines[2:]]

    rewrite(tmp_path, edit)
    with pytest.raises(
        deployment_ledger.DeploymentLedgerError, match="chain broken"
    ):
        deployment_ledger.DeploymentLedger.load(
            tmp_path / deployment_ledger.FILE_NAME
        )


def test_tail_truncation_is_caught_with_the_recorded_head(tmp_path):
    ledger = filled(tmp_path)
    head = ledger.head
    rewrite(tmp_path, lambda lines: lines[:-1])
    assert deployment_ledger.DeploymentLedger.load(
        tmp_path / deployment_ledger.FILE_NAME
    )
    with pytest.raises(
        deployment_ledger.DeploymentLedgerError, match="recorded head"
    ):
        deployment_ledger.DeploymentLedger.load(
            tmp_path / deployment_ledger.FILE_NAME, head
        )


def test_unknown_kinds_and_a_second_open_are_refused():
    ledger = make()
    with pytest.raises(
        deployment_ledger.DeploymentLedgerError, match="unknown"
    ):
        ledger.commit("confirmation", {})
    with pytest.raises(
        deployment_ledger.DeploymentLedgerError, match="exactly one"
    ):
        ledger.commit(deployment_ledger.OPEN, {})


def test_a_challenger_must_be_registered_before_it_is_scored():
    ledger = make()
    body = {
        "name": "c1",
        "phase": deployment_ledger.DECIDED,
        "e_value": 100.0,
        "level": "1/40",
        "rejected": True,
        "wealth_after": "1/40",
    }
    with pytest.raises(
        deployment_ledger.DeploymentLedgerError, match="not registered"
    ):
        ledger.commit(deployment_ledger.CHALLENGER_RESULT, body)


def test_challenger_decisions_are_replayed_exactly():
    ledger = make()
    ledger.commit(
        deployment_ledger.CHALLENGER_RESULT,
        {"name": "c1", "phase": deployment_ledger.REGISTERED},
    )
    step = ledger.require_promotion().rule.step(
        ledger.require_promotion().state,
        deployment_ledger.sequential_tests.online_fdr.Evidence(e=100.0),
    )
    good = {
        "name": "c1",
        "phase": deployment_ledger.DECIDED,
        "e_value": 100.0,
        "level": str(step.level),
        "rejected": step.rejected,
        "wealth_after": str(step.wealth_after),
    }
    bad = {**good, "rejected": not step.rejected}
    with pytest.raises(deployment_ledger.DeploymentLedgerError, match="replay"):
        make_with_registered(bad)
    row = make_with_registered(good)
    assert row["kind"] == deployment_ledger.CHALLENGER_RESULT


def make_with_registered(body):
    ledger = make()
    ledger.commit(
        deployment_ledger.CHALLENGER_RESULT,
        {"name": "c1", "phase": deployment_ledger.REGISTERED},
    )
    return ledger.commit(deployment_ledger.CHALLENGER_RESULT, body)


def test_an_attestation_is_bound_to_its_question_hash():
    ledger = make()
    identity, digest = escalate(ledger)
    wrong = reply(identity, "0" * 64)
    with pytest.raises(
        deployment_ledger.DeploymentLedgerError, match="different question"
    ):
        ledger.commit(deployment_ledger.ATTESTATION, wrong.to_dict())
    ledger.commit(
        deployment_ledger.ATTESTATION, reply(identity, digest).to_dict()
    )
    assert ledger.pending() == []


def test_an_attestation_signed_with_a_system_key_is_refused():
    ledger = make()
    identity, digest = escalate(ledger)
    with pytest.raises(
        deployment_ledger.DeploymentLedgerError, match="system holds"
    ):
        ledger.commit(
            deployment_ledger.ATTESTATION,
            reply(identity, digest, key=SYSTEM_KEY).to_dict(),
        )


def test_an_escalation_is_answered_once_and_timeouts_close_it():
    ledger = make()
    identity, digest = escalate(ledger)
    ledger.commit(
        deployment_ledger.ATTESTATION, reply(identity, digest).to_dict()
    )
    with pytest.raises(
        deployment_ledger.DeploymentLedgerError, match="already closed"
    ):
        ledger.commit(
            deployment_ledger.ATTESTATION, reply(identity, digest).to_dict()
        )
    other, ignored = escalate(ledger, "rollback")
    ledger.commit(
        deployment_ledger.ESCALATION,
        {"phase": deployment_ledger.PHASE_TIMEOUT, "id": other},
    )
    assert ledger.pending() == []
    with pytest.raises(
        deployment_ledger.DeploymentLedgerError, match="not pending"
    ):
        ledger.commit(
            deployment_ledger.ESCALATION,
            {"phase": deployment_ledger.PHASE_TIMEOUT, "id": other},
        )


def test_escalation_needs_a_matching_hash_id_and_evidence():
    ledger = make()
    question = {"action": "x"}
    body = {
        "phase": deployment_ledger.PHASE_OPEN,
        "id": "esc-1",
        "question": question,
        "question_hash": deployment_ledger.question_hash(question),
        "evidence": [],
        "proposed_default": "d",
    }
    with pytest.raises(deployment_ledger.DeploymentLedgerError, match="hash"):
        ledger.commit(
            deployment_ledger.ESCALATION, {**body, "question_hash": "1" * 64}
        )
    with pytest.raises(deployment_ledger.DeploymentLedgerError, match="id"):
        ledger.commit(deployment_ledger.ESCALATION, {**body, "id": "esc-7"})
    del body["proposed_default"]
    with pytest.raises(
        deployment_ledger.DeploymentLedgerError, match="proposed_default"
    ):
        ledger.commit(deployment_ledger.ESCALATION, body)


def test_attestation_signature_and_fields():
    answer = reply("esc-1", "a" * 64)
    assert answer.verify(KEY)
    assert not answer.verify("another key")
    assert not deployment_ledger.Attestation.from_dict(
        {**answer.to_dict(), "statement": "changed"}
    ).verify(KEY)
    with pytest.raises(
        deployment_ledger.DeploymentLedgerError, match="expires"
    ):
        reply("esc-1", "a" * 64, expires="2026-01-01")
    with pytest.raises(
        deployment_ledger.DeploymentLedgerError, match="YYYY-MM-DD"
    ):
        reply("esc-1", "a" * 64, date="11/01/2026")
    with pytest.raises(
        deployment_ledger.DeploymentLedgerError, match="statement"
    ):
        reply("esc-1", "a" * 64, statement=" ")
    with pytest.raises(
        deployment_ledger.DeploymentLedgerError, match="unexpected"
    ):
        deployment_ledger.Attestation.from_dict({"escalation_id": "x"})


def test_halt_is_released_only_with_a_recalibration():
    ledger = make()
    with pytest.raises(
        deployment_ledger.DeploymentLedgerError, match="without a halt"
    ):
        ledger.commit(
            deployment_ledger.HALT,
            {"phase": deployment_ledger.RELEASE, "recalibration": {"p": 0}},
        )
    ledger.commit(
        deployment_ledger.HALT,
        {"phase": deployment_ledger.ENTER, "reason": "risk"},
    )
    assert ledger.halted
    with pytest.raises(
        deployment_ledger.DeploymentLedgerError, match="recalibration"
    ):
        ledger.commit(
            deployment_ledger.HALT, {"phase": deployment_ledger.RELEASE}
        )
    ledger.commit(
        deployment_ledger.HALT,
        {"phase": deployment_ledger.RELEASE, "recalibration": {"tau": 0.5}},
    )
    assert not ledger.halted


def test_in_memory_verify_replays_the_rows():
    ledger = make()
    ledger.commit(deployment_ledger.ALARM, {"batch": 1})
    ledger.verify()
    ledger.rows[1]["body"]["batch"] = 2
    with pytest.raises(deployment_ledger.DeploymentLedgerError):
        ledger.verify()


def test_replay_is_deterministic_across_loads(tmp_path):
    ledger = filled(tmp_path)
    first = deployment_ledger.DeploymentLedger.load(
        tmp_path / deployment_ledger.FILE_NAME
    )
    second = deployment_ledger.DeploymentLedger.load(
        tmp_path / deployment_ledger.FILE_NAME
    )
    assert first.head == second.head == ledger.head


def test_an_experiment_level_with_a_zero_denominator_is_a_ledger_error():
    ledger = make()
    with pytest.raises(
        deployment_ledger.DeploymentLedgerError, match="malformed"
    ):
        ledger.commit(
            deployment_ledger.ROLLOUT_EXPERIMENT,
            {"experiment_id": "e1", "level": "1/0"},
        )
