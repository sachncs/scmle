"""Unit tests of the deployment ledger submodules, each by its full name."""

import datetime
import hashlib
import json

import pytest

import scmle.assurance.ledger.ledger
from scmle.governance.deployment_ledger import chain
from scmle.governance.deployment_ledger import grants
from scmle.governance.deployment_ledger import kinds
from scmle.governance.deployment_ledger import rules
from scmle.monitoring import sequential_tests

KEY = "person-key"
SYSTEM_KEY = "system-key"


def make(path=None):
    return chain.DeploymentLedger.create(
        path,
        charter_hash="c" * 64,
        policy={"r": 1},
        alpha_promo="1/20",
        system_keys=[kinds.thumbprint(SYSTEM_KEY)],
    )


def escalate(book):
    question = {"action": "promote"}
    number = len(book.rows)
    digest = scmle.assurance.ledger.ledger.digest(question)
    book.commit(
        kinds.ESCALATION,
        {
            "phase": kinds.PHASE_OPEN,
            "id": f"esc-{number}",
            "question": question,
            "question_hash": digest,
            "evidence": [],
            "proposed_default": "keep",
        },
    )
    return f"esc-{number}", digest


def reply_fields(identity="esc-1", digest="d" * 64, **changes):
    fields = {
        "escalation": identity,
        "digests": {"question": digest},
        "decision": "approve",
        "statement": "reviewed",
        "evidence": "ticket-1",
        "role": "owner",
        "attestor": "A. Person",
        "date": "2026-11-01",
        "scope": "v3:a",
        "expires": "2026-12-01",
    }
    fields.update(changes)
    return fields


def envelope_fields(**changes):
    fields = {
        "identifier": "env-1",
        "limits": {"train_challenger": 2},
        "statement": "may retrain twice",
        "role": "owner",
        "attestor": "A. Person",
        "date": "2026-11-01",
        "scope": "v3:a",
        "expires": "2026-12-01",
    }
    fields.update(changes)
    return fields


# ---- kinds -----------------------------------------------------------------


def test_the_kind_constants_have_their_values():
    assert kinds.FILE_NAME == "deployment_ledger.jsonl"
    assert kinds.LEDGER_VERSION == 1
    assert len(kinds.KINDS) == 21
    assert kinds.KINDS[0] == "open"
    assert kinds.ENVELOPE_ACTIONS == ("train_challenger",)
    assert kinds.CONTROLLER_PHASES == ("intent", "applied", "rejected", "tick")
    assert kinds.CONTROLLER_INPUTS == ("batch", "labels")
    assert [r.value for r in kinds.Reply] == [
        "approve",
        "reject",
        "override",
        "cannot_decide",
    ]


def test_mapped_refuses_a_body_that_is_not_an_object():
    kinds.mapped({})
    with pytest.raises(TypeError, match="body must be an object"):
        kinds.mapped([])


def test_thumbprint_is_a_domain_separated_prefix():
    expected = hashlib.sha256(b"scmle-key-fingerprint:k").hexdigest()[:16]
    assert kinds.thumbprint("k") == expected
    assert kinds.thumbprint(b"k") == expected
    assert len(expected) == 16


def test_parse_date_reads_iso_dates_only():
    assert kinds.parse_date("2026-01-02", "d") == datetime.date(2026, 1, 2)
    with pytest.raises(kinds.MisuseError, match="d must be YYYY-MM-DD"):
        kinds.parse_date("02/01/2026", "d")
    with pytest.raises(kinds.MisuseError, match="x must be YYYY-MM-DD"):
        kinds.parse_date(None, "x")


# ---- grants ----------------------------------------------------------------


def test_the_version_is_two():
    assert grants.VERSION == 2


def test_an_attestation_signs_and_verifies_with_its_key():
    reply = grants.Attestation.create(KEY, **reply_fields())
    assert reply.key_id == kinds.thumbprint(KEY)
    assert reply.decision is kinds.Reply.APPROVE
    assert reply.verify(KEY) is True
    assert reply.verify("other") is False
    forged = grants.Attestation.from_dict({**reply.to_dict(), "scope": "v4:a"})
    assert forged.verify(KEY) is False
    unsigned = grants.Attestation(
        key_id=reply.key_id,
        **{**reply_fields(), "decision": kinds.Reply.APPROVE},
    )
    assert unsigned.verify(KEY) is False


def test_an_attestation_round_trips_its_stored_layout():
    reply = grants.Attestation.create(KEY, **reply_fields())
    record = reply.to_dict()
    assert record["schema_version"] == 2
    assert record["decision"] == "approve"
    assert "signature" not in reply.preimage()
    assert grants.Attestation.from_dict(record) == reply


def test_a_version_one_attestation_still_verifies():
    reply = grants.Attestation.create(KEY, **reply_fields(), schema_version=1)
    record = reply.to_dict()
    assert "schema_version" not in record
    assert record["escalation_id"] == "esc-1"
    assert record["question_hash"] == "d" * 64
    again = grants.Attestation.from_dict(record)
    assert again.schema_version == 1
    assert again.verify(KEY) is True


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"role": " "}, "attestation role is required"),
        ({"digests": {}}, "digests.question is required"),
        ({"expires": "2026-10-31"}, "expires is before the date"),
        ({"date": "soon"}, "date must be YYYY-MM-DD"),
    ],
)
def test_an_attestation_refuses_a_bad_field(change, message):
    with pytest.raises(kinds.MisuseError, match=message):
        grants.Attestation.create(KEY, **reply_fields(**change))


def test_an_attestation_expiring_on_its_date_is_accepted():
    reply = grants.Attestation.create(KEY, **reply_fields(expires="2026-11-01"))
    assert reply.expires == reply.date


def test_an_attestation_refuses_a_plain_text_decision_and_bad_records():
    with pytest.raises(kinds.MisuseError, match="decision must be a Reply"):
        grants.Attestation(key_id="k", **reply_fields())
    record = grants.Attestation.create(KEY, **reply_fields()).to_dict()
    with pytest.raises(kinds.MisuseError, match="unexpected or missing reply"):
        grants.Attestation.from_dict({**record, "extra": 1})
    with pytest.raises(kinds.MisuseError, match="unsupported record version 3"):
        grants.Attestation.from_dict({**record, "schema_version": 3})


def test_an_envelope_signs_and_round_trips():
    grant = grants.Envelope.create(KEY, **envelope_fields())
    assert grant.verify(KEY) is True
    assert grant.verify(SYSTEM_KEY) is False
    record = grant.to_dict()
    assert record["limits"] == {"train_challenger": 2}
    assert grants.Envelope.from_dict(record) == grant
    with pytest.raises(
        kinds.MisuseError, match="unexpected or missing envelope"
    ):
        grants.Envelope.from_dict({**record, "extra": 1})


def test_a_version_one_envelope_uses_the_old_keys():
    grant = grants.Envelope.create(KEY, **envelope_fields(), schema_version=1)
    record = grant.to_dict()
    assert record["envelope_id"] == "env-1"
    assert record["name_or_id"] == "A. Person"
    assert grants.Envelope.from_dict(record).verify(KEY) is True


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"statement": ""}, "envelope statement is required"),
        ({"limits": {}}, "permits at least one action"),
        ({"limits": {"promote": 1}}, "'promote' cannot be put in an envelope"),
        ({"limits": {"train_challenger": 0}}, "must be a positive integer"),
        ({"limits": {"train_challenger": True}}, "must be a positive integer"),
        ({"limits": {"train_challenger": 1.0}}, "must be a positive integer"),
        ({"expires": "2026-10-01"}, "expires is before the date"),
    ],
)
def test_an_envelope_refuses_a_bad_field(change, message):
    with pytest.raises(kinds.MisuseError, match=message):
        grants.Envelope.create(KEY, **envelope_fields(**change))


def test_store_and_load_move_between_layouts():
    layout = {"schema_version": 2, "identifier": "e", "attestor": "a"}
    stored = grants.store(layout, grants.GRANT)
    assert stored == layout
    old = grants.store({**layout, "schema_version": 1}, grants.GRANT)
    assert old == {"envelope_id": "e", "name_or_id": "a"}
    assert grants.load(old, grants.GRANT) == {**layout, "schema_version": 1}


# ---- chain -----------------------------------------------------------------


def test_create_writes_the_open_row(tmp_path):
    path = tmp_path / "d" / kinds.FILE_NAME
    book = make(path)
    row = book.rows[0]
    assert row["seq"] == 0
    assert row["kind"] == "open"
    assert row["prev"] == scmle.assurance.ledger.ledger.GENESIS
    body = row["body"]
    assert body["policy_hash"] == scmle.assurance.ledger.ledger.digest({"r": 1})
    assert body["system_keys"] == [kinds.thumbprint(SYSTEM_KEY)]
    assert json.loads(path.read_text().splitlines()[0]) == row
    with pytest.raises(kinds.MisuseError, match="exists; use load"):
        make(path)


def test_load_reads_back_and_checks_the_head(tmp_path):
    path = tmp_path / kinds.FILE_NAME
    book = make(path)
    book.commit(kinds.ALARM, {"batch": 1})
    again = chain.DeploymentLedger.load(path, expected_head=book.head)
    assert again.rows == book.rows
    with pytest.raises(kinds.MisuseError, match="does not match the recorded"):
        chain.DeploymentLedger.load(path, expected_head="x")


def test_load_refuses_missing_empty_and_garbled_files(tmp_path):
    with pytest.raises(kinds.MisuseError, match="does not exist"):
        chain.DeploymentLedger.load(tmp_path / "none")
    empty = tmp_path / "empty"
    empty.write_text("\n\n")
    with pytest.raises(kinds.MisuseError, match="ledger is empty"):
        chain.DeploymentLedger.load(empty)
    garbled = tmp_path / "bad"
    garbled.write_text("{")
    with pytest.raises(kinds.MisuseError, match="ledger row 0 is not JSON"):
        chain.DeploymentLedger.load(garbled)


def test_ingest_refuses_broken_rows():
    book = make()
    row = book.commit(kinds.ALARM, {})
    fresh = make()
    with pytest.raises(kinds.MisuseError, match="has wrong fields"):
        fresh.ingest({k: v for k, v in row.items() if k != "hash"})
    with pytest.raises(kinds.MisuseError, match="chain broken at row 1"):
        fresh.ingest({**row, "body": {"x": 1}})
    with pytest.raises(kinds.MisuseError, match="unknown ledger row kind"):
        fresh.commit("nonsense", {})
    with pytest.raises(kinds.MisuseError, match="exactly one 'open'"):
        fresh.commit(kinds.OPEN, {})
    content = {"seq": 1, "kind": kinds.ALARM, "body": [], "prev": fresh.head}
    listed = {**content, "hash": scmle.assurance.ledger.ledger.digest(content)}
    with pytest.raises(kinds.MisuseError, match="malformed alarm row 1"):
        fresh.ingest(listed)
    assert len(fresh.rows) == 1


def test_the_first_row_must_be_open():
    empty = chain.DeploymentLedger(None)
    with pytest.raises(kinds.MisuseError, match="first ledger row must be"):
        empty.commit(kinds.ALARM, {})


def test_verify_replays_memory_and_compares_the_file(tmp_path):
    make().verify()
    path = tmp_path / kinds.FILE_NAME
    book = make(path)
    book.verify()
    book.rows.append(dict(book.rows[0]))
    with pytest.raises(kinds.MisuseError, match="differs from memory"):
        book.verify()


# ---- rules -----------------------------------------------------------------


def test_an_unopened_ledger_has_genesis_and_no_sequence():
    state = rules.Oversight(None)
    assert state.head == scmle.assurance.ledger.ledger.GENESIS
    with pytest.raises(kinds.MisuseError, match="has no open row"):
        state.elond()


def test_apply_open_checks_version_and_policy_hash():
    state = rules.Oversight(None)
    body = make().rows[0]["body"]
    with pytest.raises(kinds.MisuseError, match="unsupported ledger version"):
        state.apply_open({**body, "ledger_version": 2})
    with pytest.raises(kinds.MisuseError, match="does not match its hash"):
        state.apply_open({**body, "policy": {"r": 2}})
    state.apply_open(body)
    assert state.system_keys == frozenset(body["system_keys"])


def test_challengers_register_then_replay_their_decision():
    book = make()
    book.commit(kinds.CHALLENGER_RESULT, {"name": "c1", "phase": "registered"})
    with pytest.raises(kinds.MisuseError, match="already registered"):
        book.commit(
            kinds.CHALLENGER_RESULT, {"name": "c1", "phase": "registered"}
        )
    with pytest.raises(kinds.MisuseError, match="registered or decided"):
        book.commit(kinds.CHALLENGER_RESULT, {"name": "c1", "phase": "maybe"})
    with pytest.raises(kinds.MisuseError, match="was not registered"):
        book.commit(kinds.CHALLENGER_RESULT, {"name": "c9", "phase": "decided"})
    twin = sequential_tests.Sequence("1/20")
    twin.register("c1")
    step = twin.decide("c1", 50.0)
    good = {
        "name": "c1",
        "phase": "decided",
        "e_value": 50.0,
        "level": str(step.level),
        "rejected": step.rejected,
        "wealth_after": str(step.wealth_after),
    }
    with pytest.raises(kinds.MisuseError, match="replay of e-LOND differs"):
        make_registered().commit(
            kinds.CHALLENGER_RESULT, {**good, "rejected": not step.rejected}
        )
    book.commit(kinds.CHALLENGER_RESULT, good)
    assert book.decided["c1"]["e_value"] == 50.0
    assert book.registered == ["c1"]


def make_registered():
    book = make()
    book.commit(kinds.CHALLENGER_RESULT, {"name": "c1", "phase": "registered"})
    return book


def test_escalations_open_and_time_out_once():
    book = make()
    identity = escalate(book)[0]
    assert identity == "esc-1"
    assert [row["id"] for row in book.pending()] == ["esc-1"]
    book.commit(kinds.ESCALATION, {"phase": "timeout", "id": identity})
    assert book.closed == {"esc-1": "timeout"}
    assert book.pending() == []
    with pytest.raises(kinds.MisuseError, match="is not pending"):
        book.commit(kinds.ESCALATION, {"phase": "timeout", "id": identity})
    with pytest.raises(kinds.MisuseError, match="must be open or timeout"):
        book.commit(kinds.ESCALATION, {"phase": "later", "id": identity})


def test_an_escalation_must_match_its_hash_row_and_fields():
    book = make()
    base = {
        "phase": "open",
        "id": "esc-1",
        "question": {"a": 1},
        "question_hash": scmle.assurance.ledger.ledger.digest({"a": 1}),
        "evidence": [],
        "proposed_default": "keep",
    }
    with pytest.raises(kinds.MisuseError, match="does not match its hash"):
        book.commit(kinds.ESCALATION, {**base, "question_hash": "x"})
    with pytest.raises(kinds.MisuseError, match="id is not its row"):
        book.commit(kinds.ESCALATION, {**base, "id": "esc-2"})
    with pytest.raises(kinds.MisuseError, match="needs proposed_default"):
        book.commit(
            kinds.ESCALATION,
            {k: v for k, v in base.items() if k != "proposed_default"},
        )


def test_an_attestation_closes_its_escalation_and_is_spent_once():
    book = make()
    identity, digest = escalate(book)
    with pytest.raises(kinds.MisuseError, match="only be spent after"):
        book.commit(kinds.ATTESTATION_USED, {"escalation_id": identity})
    wrong = grants.Attestation.create(KEY, **reply_fields(identity, "x" * 64))
    with pytest.raises(kinds.MisuseError, match="different question"):
        book.commit(kinds.ATTESTATION, wrong.to_dict())
    system = grants.Attestation.create(
        SYSTEM_KEY, **reply_fields(identity, digest)
    )
    with pytest.raises(kinds.MisuseError, match="key the system holds"):
        book.commit(kinds.ATTESTATION, system.to_dict())
    orphan = grants.Attestation.create(KEY, **reply_fields("esc-9", digest))
    with pytest.raises(kinds.MisuseError, match="answers no escalation"):
        book.commit(kinds.ATTESTATION, orphan.to_dict())
    good = grants.Attestation.create(KEY, **reply_fields(identity, digest))
    book.commit(kinds.ATTESTATION, good.to_dict())
    assert book.closed[identity] == "attestation"
    with pytest.raises(kinds.MisuseError, match="already closed"):
        book.commit(kinds.ATTESTATION, good.to_dict())
    book.commit(kinds.ATTESTATION_USED, {"escalation_id": identity})
    with pytest.raises(kinds.MisuseError, match="already spent"):
        book.commit(kinds.ATTESTATION_USED, {"escalation_id": identity})


def test_a_halt_is_released_only_with_a_recalibration():
    book = make()
    with pytest.raises(kinds.MisuseError, match="release without a halt"):
        book.commit(kinds.HALT, {"phase": "release", "recalibration": {"a": 1}})
    book.commit(kinds.HALT, {"phase": "enter"})
    assert book.halted is True
    with pytest.raises(kinds.MisuseError, match="passing recalibration"):
        book.commit(kinds.HALT, {"phase": "release"})
    with pytest.raises(kinds.MisuseError, match="enter or release"):
        book.commit(kinds.HALT, {"phase": "pause"})
    book.commit(kinds.HALT, {"phase": "release", "recalibration": {"a": 1}})
    assert book.halted is False


def test_an_envelope_counts_uses_until_spent_or_revoked():
    book = make()
    grant = grants.Envelope.create(KEY, **envelope_fields())
    book.commit(kinds.ENVELOPE, grant.to_dict())
    with pytest.raises(kinds.MisuseError, match="was already granted"):
        book.commit(kinds.ENVELOPE, grant.to_dict())
    use = {
        "envelope_id": "env-1",
        "action": "train_challenger",
        "scope": "v3:a",
    }
    with pytest.raises(kinds.MisuseError, match="outside the envelope's scope"):
        book.commit(kinds.ENVELOPE_USED, {**use, "scope": "v4:a"})
    with pytest.raises(kinds.MisuseError, match="does not cover 'promote'"):
        book.commit(kinds.ENVELOPE_USED, {**use, "action": "promote"})
    with pytest.raises(kinds.MisuseError, match="'env-9' is unknown"):
        book.commit(kinds.ENVELOPE_USED, {**use, "envelope_id": "env-9"})
    book.commit(kinds.ENVELOPE_USED, use)
    book.commit(kinds.ENVELOPE_USED, use)
    assert book.envelope_uses["env-1"] == {"train_challenger": 2}
    with pytest.raises(kinds.MisuseError, match="spent its 2 use"):
        book.commit(kinds.ENVELOPE_USED, use)
    book.commit(kinds.ENVELOPE_REVOKED, {"envelope_id": "env-1"})
    with pytest.raises(kinds.MisuseError, match="was revoked"):
        book.commit(kinds.ENVELOPE_USED, use)
    with pytest.raises(kinds.MisuseError, match="is revoked"):
        book.commit(kinds.ENVELOPE_REVOKED, {"envelope_id": "env-1"})
    with pytest.raises(kinds.MisuseError, match="is unknown"):
        book.commit(kinds.ENVELOPE_REVOKED, {"envelope_id": "env-2"})


def test_an_envelope_under_a_system_key_is_refused():
    book = make()
    grant = grants.Envelope.create(SYSTEM_KEY, **envelope_fields())
    with pytest.raises(kinds.MisuseError, match="key the system holds"):
        book.commit(kinds.ENVELOPE, grant.to_dict())


def test_the_controller_journal_applies_each_input_once():
    book = make()
    book.commit(kinds.CONTROLLER, {"phase": "tick"})
    intent = {"phase": "intent", "digest": "d" * 64, "what": "batch"}
    with pytest.raises(kinds.MisuseError, match="phase must be one of"):
        book.commit(kinds.CONTROLLER, {**intent, "phase": "maybe"})
    with pytest.raises(kinds.MisuseError, match="input must be one of"):
        book.commit(kinds.CONTROLLER, {**intent, "what": "model"})
    with pytest.raises(kinds.MisuseError, match="applied without an intent"):
        book.commit(kinds.CONTROLLER, {**intent, "phase": "applied"})
    book.commit(kinds.CONTROLLER, intent)
    with pytest.raises(kinds.MisuseError, match="already has an intent"):
        book.commit(kinds.CONTROLLER, intent)
    book.commit(kinds.CONTROLLER, {**intent, "phase": "applied"})
    with pytest.raises(kinds.MisuseError, match="dddddddddddd was applied"):
        book.commit(kinds.CONTROLLER, {**intent, "phase": "rejected"})
    other = {**intent, "digest": "e" * 64, "phase": "rejected"}
    book.commit(kinds.CONTROLLER, other)
    assert set(book.rejected) == {"e" * 64}
    assert set(book.consumed) == {"d" * 64}


def test_rollouts_start_once_and_check_assignments_and_decisions():
    book = make()
    with pytest.raises(kinds.MisuseError, match="lies in"):
        book.commit(
            kinds.ROLLOUT_EXPERIMENT, {"experiment_id": "x", "level": "1"}
        )
    with pytest.raises(kinds.MisuseError, match="lies in"):
        book.commit(
            kinds.ROLLOUT_EXPERIMENT, {"experiment_id": "x", "level": "0"}
        )
    book.commit(
        kinds.ROLLOUT_EXPERIMENT, {"experiment_id": "x", "level": "1/20"}
    )
    with pytest.raises(kinds.MisuseError, match="already started"):
        book.commit(
            kinds.ROLLOUT_EXPERIMENT, {"experiment_id": "x", "level": "1/20"}
        )
    batch = {
        "experiment_id": "x",
        "units": [1, 2],
        "arm": [0, 1],
        "probability": [0.5, 0.5],
    }
    book.commit(kinds.ROLLOUT_ASSIGNMENT, batch)
    with pytest.raises(kinds.MisuseError, match="differ in length"):
        book.commit(kinds.ROLLOUT_ASSIGNMENT, {**batch, "arm": [0]})
    with pytest.raises(kinds.MisuseError, match="arms are 0 or 1"):
        book.commit(kinds.ROLLOUT_ASSIGNMENT, {**batch, "arm": [0, 2]})
    with pytest.raises(kinds.MisuseError, match="arms are 0 or 1"):
        book.commit(
            kinds.ROLLOUT_ASSIGNMENT, {**batch, "probability": [0.5, 1.0]}
        )
    with pytest.raises(kinds.MisuseError, match="which was not started"):
        book.commit(kinds.ROLLOUT_ASSIGNMENT, {**batch, "experiment_id": "y"})
    book.commit(kinds.ROLLOUT_DECISION, {"experiment_id": "x"})
    with pytest.raises(kinds.MisuseError, match="decision for 'y'"):
        book.commit(kinds.ROLLOUT_DECISION, {"experiment_id": "y"})
