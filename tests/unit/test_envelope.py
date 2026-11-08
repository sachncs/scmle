"""Envelopes: authority granted in advance, bounded and on the record."""

import dataclasses
import datetime
import json

import pytest

from scmle.data.ledger import ledger as ledger_module
from scmle.governance import deployment_ledger as dl
from scmle.governance import envelope
from scmle.governance import never_list

PERSON_KEY = "person-key"
SYSTEM_KEY = "system-key"
SCOPE = "model-v1:stream-a"
TODAY = datetime.date(2026, 11, 4)
ACTION = envelope.EnvelopeAction.TRAIN_CHALLENGER


def make_ledger(path=None):
    return dl.DeploymentLedger.create(
        path,
        charter_hash="a" * 64,
        policy={"scope": SCOPE},
        alpha_promo="1/20",
        system_keys=[dl.key_fingerprint(SYSTEM_KEY)],
    )


def make_guard(ledger=None, today=TODAY, keys=True):
    guard = envelope.EnvelopeGuard(
        ledger or make_ledger(),
        today=lambda: today,
        verification_keys=(
            {dl.key_fingerprint(PERSON_KEY): PERSON_KEY} if keys else None
        ),
    )
    return guard


def signed(key=PERSON_KEY, **changes):
    fields = {
        "envelope_id": "e1",
        "limits": {ACTION.value: 2},
        "statement": "retrain a challenger when the alarm gates it",
        "role": "owner",
        "name_or_id": "P. Owner",
        "date": "2026-11-01",
        "scope": SCOPE,
        "expires": "2026-12-01",
    }
    fields.update(changes)
    return dl.Envelope.create(key, **fields)


def kinds(guard):
    return [row["kind"] for row in guard.ledger.rows]


# ------------------------------------------------------------ the dataclass


def test_the_actions_an_envelope_can_name_never_overlap_the_never_list():
    allowed = {item.value for item in envelope.EnvelopeAction}
    assert allowed == set(dl.ENVELOPE_ACTIONS)
    assert not allowed & {item.value for item in never_list.NeverAction}


@pytest.mark.parametrize(
    "action", [item.value for item in never_list.NeverAction] + ["unknown"]
)
def test_an_envelope_cannot_name_a_never_list_or_unknown_action(action):
    with pytest.raises(dl.DeploymentLedgerError, match="cannot be put"):
        signed(limits={action: 1})


@pytest.mark.parametrize("limit", [0, -1, 1.5, True, "2"])
def test_a_limit_is_a_positive_integer(limit):
    with pytest.raises(dl.DeploymentLedgerError, match="positive integer"):
        signed(limits={ACTION.value: limit})


def test_an_envelope_needs_an_action_a_name_and_ordered_dates():
    with pytest.raises(dl.DeploymentLedgerError, match="at least one"):
        signed(limits={})
    with pytest.raises(dl.DeploymentLedgerError, match="role is required"):
        signed(role=" ")
    with pytest.raises(dl.DeploymentLedgerError, match="before the date"):
        signed(date="2026-12-02")
    with pytest.raises(dl.DeploymentLedgerError, match="YYYY-MM-DD"):
        signed(expires="soon")


def test_the_signature_covers_every_field():
    grant = signed()
    assert grant.verify(PERSON_KEY)
    assert not grant.verify("another-key")
    for change in (
        {"limits": {ACTION.value: 99}},
        {"scope": "model-v2:stream-a"},
        {"expires": "2027-01-01"},
        {"statement": "anything"},
    ):
        assert not dataclasses.replace(grant, **change).verify(PERSON_KEY)
    assert not dataclasses.replace(grant, signature="").verify(PERSON_KEY)
    assert dl.Envelope.from_dict(grant.to_dict()) == grant


def test_from_dict_refuses_missing_or_extra_fields():
    record = signed().to_dict()
    with pytest.raises(dl.DeploymentLedgerError, match="fields"):
        dl.Envelope.from_dict({**record, "extra": 1})
    del record["scope"]
    with pytest.raises(dl.DeploymentLedgerError, match="fields"):
        dl.Envelope.from_dict(record)


# --------------------------------------------------------------- granting


def test_a_grant_is_chained_and_shows_what_is_left():
    guard = make_guard()
    guard.grant(signed())
    assert kinds(guard) == [dl.OPEN, dl.ENVELOPE]
    assert guard.remaining("e1") == {ACTION.value: 2}


def test_a_signature_that_does_not_verify_is_not_granted():
    guard = make_guard()
    with pytest.raises(envelope.EnvelopeRefused, match="signature"):
        guard.grant(signed(key="stranger"))
    forged = dataclasses.replace(signed(), scope="other")
    with pytest.raises(envelope.EnvelopeRefused, match="signature"):
        guard.grant(forged)
    assert kinds(guard) == [dl.OPEN]


def test_the_ledger_refuses_an_envelope_signed_with_a_system_key():
    guard = make_guard(keys=False)
    with pytest.raises(dl.DeploymentLedgerError, match="system holds"):
        guard.grant(signed(key=SYSTEM_KEY))


def test_an_id_is_granted_once():
    guard = make_guard()
    guard.grant(signed())
    with pytest.raises(dl.DeploymentLedgerError, match="already granted"):
        guard.grant(signed())


# -------------------------------------------------------------- spending


def test_a_use_is_chained_before_it_returns_and_spends_the_budget():
    guard = make_guard()
    guard.grant(signed())
    assert guard.authorize(ACTION, SCOPE, {"batch": 3}) == "e1"
    used = guard.ledger.rows[-1]
    assert used["kind"] == dl.ENVELOPE_USED
    assert used["body"]["detail"] == {"batch": 3}
    assert used["body"]["date"] == "2026-11-04"
    assert guard.remaining("e1") == {ACTION.value: 1}
    guard.authorize(ACTION, SCOPE)
    assert guard.remaining("e1") == {ACTION.value: 0}
    with pytest.raises(envelope.EnvelopeRefused, match="spent"):
        guard.authorize(ACTION, SCOPE)


def test_a_refusal_writes_nothing():
    guard = make_guard()
    with pytest.raises(envelope.EnvelopeRefused, match="no envelope was"):
        guard.authorize(ACTION, SCOPE)
    guard.grant(signed())
    before = len(guard.ledger.rows)
    with pytest.raises(envelope.EnvelopeRefused, match="different scope"):
        guard.authorize(ACTION, "model-v2:stream-a")
    assert len(guard.ledger.rows) == before


def test_the_dates_bound_the_authority():
    guard = make_guard(today=datetime.date(2026, 10, 31))
    guard.grant(signed())
    with pytest.raises(envelope.EnvelopeRefused, match="not valid today"):
        guard.authorize(ACTION, SCOPE)
    guard.today = lambda: datetime.date(2026, 12, 1)
    guard.authorize(ACTION, SCOPE)
    guard.today = lambda: datetime.date(2026, 12, 2)
    with pytest.raises(envelope.EnvelopeRefused, match="not valid today"):
        guard.authorize(ACTION, SCOPE)


def test_the_next_envelope_serves_when_the_first_is_spent():
    guard = make_guard()
    guard.grant(signed(limits={ACTION.value: 1}))
    guard.grant(signed(envelope_id="e2", limits={ACTION.value: 1}))
    assert guard.authorize(ACTION, SCOPE) == "e1"
    assert guard.authorize(ACTION, SCOPE) == "e2"


def test_a_revoked_envelope_authorises_nothing_and_anyone_may_revoke():
    guard = make_guard()
    guard.grant(signed())
    guard.revoke("e1", "model retired")
    assert guard.remaining("e1") == {}
    with pytest.raises(envelope.EnvelopeRefused, match="revoked"):
        guard.authorize(ACTION, SCOPE)
    with pytest.raises(dl.DeploymentLedgerError, match="is revoked"):
        guard.revoke("e1", "again")
    with pytest.raises(dl.DeploymentLedgerError, match="unknown"):
        guard.revoke("missing", "no such")


def test_a_key_that_is_no_longer_trusted_stops_the_authority():
    guard = make_guard()
    guard.grant(signed())
    guard.verification_keys = {}
    with pytest.raises(envelope.EnvelopeRefused, match="signature"):
        guard.authorize(ACTION, SCOPE)


def test_a_never_list_action_is_refused_even_if_it_were_passed_in():
    guard = make_guard()
    guard.grant(signed())

    class Impostor:
        value = never_list.NeverAction.MOVE_CHAMPION.value

    with pytest.raises(envelope.EnvelopeRefused, match="never list"):
        guard.authorize(Impostor(), SCOPE)  # type: ignore[arg-type]


# -------------------------------------------------- the ledger on its own


def test_the_ledger_refuses_a_use_the_guard_would_have_refused():
    ledger = make_ledger()
    guard = make_guard(ledger)
    guard.grant(signed(limits={ACTION.value: 1}))
    use = {"envelope_id": "e1", "action": ACTION.value, "scope": SCOPE}
    ledger.commit(dl.ENVELOPE_USED, use)
    with pytest.raises(dl.DeploymentLedgerError, match="spent"):
        ledger.commit(dl.ENVELOPE_USED, use)
    with pytest.raises(dl.DeploymentLedgerError, match="outside"):
        ledger.commit(dl.ENVELOPE_USED, {**use, "scope": "elsewhere"})
    with pytest.raises(dl.DeploymentLedgerError, match="does not cover"):
        ledger.commit(dl.ENVELOPE_USED, {**use, "action": "other"})
    with pytest.raises(dl.DeploymentLedgerError, match="unknown"):
        ledger.commit(dl.ENVELOPE_USED, {**use, "envelope_id": "nobody"})
    guard.revoke("e1", "stop")
    with pytest.raises(dl.DeploymentLedgerError, match="revoked"):
        ledger.commit(dl.ENVELOPE_USED, use)


def test_a_reloaded_ledger_replays_the_same_budget(tmp_path):
    path = tmp_path / dl.FILE_NAME
    ledger = make_ledger(path)
    guard = make_guard(ledger)
    guard.grant(signed())
    guard.authorize(ACTION, SCOPE)
    again = dl.DeploymentLedger.load(path)
    assert again.envelope_uses == {"e1": {ACTION.value: 1}}
    assert make_guard(again).remaining("e1") == {ACTION.value: 1}
    again.verify()


def test_a_forged_use_row_does_not_survive_a_reload(tmp_path):
    path = tmp_path / dl.FILE_NAME
    ledger = make_ledger(path)
    guard = make_guard(ledger)
    guard.grant(signed(limits={ACTION.value: 1}))
    guard.authorize(ACTION, SCOPE)
    row = ledger.rows[-1]
    path.write_text(
        path.read_text(encoding="utf-8") + chained_copy(ledger, row),
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(dl.DeploymentLedgerError, match="spent"):
        dl.DeploymentLedger.load(path)


def chained_copy(ledger, row):
    content = {
        "seq": len(ledger.rows),
        "kind": row["kind"],
        "body": row["body"],
        "prev": ledger.head,
    }
    chained = {**content, "hash": ledger_module.digest(content)}
    return json.dumps(chained, sort_keys=True) + "\n"


def test_the_controller_rows_apply_each_input_once():
    ledger = make_ledger()
    base = {"what": "batch", "digest": "d" * 64, "file": "b0.csv"}
    ledger.commit(
        dl.CONTROLLER, {**base, "phase": dl.INTENT, "batch_before": 0}
    )
    with pytest.raises(dl.DeploymentLedgerError, match="already has an intent"):
        ledger.commit(
            dl.CONTROLLER, {**base, "phase": dl.INTENT, "batch_before": 0}
        )
    ledger.commit(dl.CONTROLLER, {**base, "phase": dl.APPLIED})
    with pytest.raises(dl.DeploymentLedgerError, match="was applied"):
        ledger.commit(dl.CONTROLLER, {**base, "phase": dl.APPLIED})
    with pytest.raises(dl.DeploymentLedgerError, match="without an intent"):
        ledger.commit(
            dl.CONTROLLER,
            {"what": "labels", "digest": "e" * 64, "phase": dl.APPLIED},
        )
    with pytest.raises(dl.DeploymentLedgerError, match="phase must be"):
        ledger.commit(dl.CONTROLLER, {"phase": "other"})
    with pytest.raises(dl.DeploymentLedgerError, match="input must be"):
        ledger.commit(
            dl.CONTROLLER,
            {"phase": dl.INTENT, "what": "photo", "digest": "f" * 64},
        )
    ledger.commit(dl.CONTROLLER, {"phase": dl.TICK, "observed": 0})
