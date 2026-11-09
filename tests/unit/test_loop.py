import datetime
import json
import math
import os
import types

import numpy as np
import pandas as pd
import pytest

from benchmarks.qualification import streams
from scmle.contracts import qualification
from scmle.governance import deployment_ledger as dl
from scmle.governance import loop as loop_module
from scmle.governance import never_list

HUMAN_KEY = "human-key"
SYSTEM_KEY = "system-key"
TODAY = datetime.date(2026, 11, 4)
CHARTER = types.SimpleNamespace(charter_hash="a" * 64)
SCOPE = "model-v1:stream-a"


def spec(**changes):
    return streams.StreamSpec(**{"window": 0, **changes})


def make_policy(**changes):
    values = {
        "reference_risk": 0.10,
        "tolerance": 0.05,
        "scope": SCOPE,
        "initial_tau": 0.5,
        "min_matured": 5,
        "r_star": 0.6,
        "alarm_gates_retrain": True,
        "outcome_window": 0,
    }
    values.update(changes)
    return loop_module.LoopPolicy(**values)


def open_loop(tmp_path, policy=None, hooks=None, **kwargs):
    return loop_module.Loop.open(
        tmp_path / "deploy",
        CHARTER,
        policy or make_policy(),
        hooks=hooks,
        verification_keys={dl.key_fingerprint(HUMAN_KEY): HUMAN_KEY},
        system_keys=[SYSTEM_KEY],
        today=lambda: TODAY,
        **kwargs,
    )


def batch_frame(stream, index, **extra):
    size = stream.loss.shape[1]
    frame = pd.DataFrame(
        {
            "case_id": [f"b{index}-{i}" for i in range(size)],
            "propensity": stream.pi[index],
            "audit": stream.audit[index],
        }
    )
    for name, value in extra.items():
        frame[name] = value
    return frame


def feed(loop, stream, first=0, last=None, challengers=None, **extra):
    """Observes, matures arrived labels and steps for each batch."""
    last = stream.spec.batches if last is None else last
    result = None
    for index in range(first, last):
        frame = batch_frame(stream, index, **extra)
        size = len(frame)
        loop.observe(frame, np.zeros(size), ["approve"] * size)
        arrived = {}
        losses = {}
        for batch in range(index + 1):
            hit = (stream.arrival[batch] == index) & stream.selected[batch]
            for position in np.flatnonzero(hit):
                losses[f"b{batch}-{position}"] = stream.loss[batch, position]
        arrived = losses
        paired = {}
        if challengers:
            for name, matrix in challengers.items():
                paired[name] = {
                    key: matrix[
                        int(key[1:].split("-")[0]), int(key.split("-")[1])
                    ]
                    for key in arrived
                }
        loop.mature(arrived, challengers=paired or None)
        result = loop.step()
    return result


def rows(loop, kind):
    return [r for r in loop.ledger.rows if r["kind"] == kind]


def reply_to(escalation, key=HUMAN_KEY, **changes):
    fields = {
        "escalation_id": escalation.id,
        "question_hash": escalation.question_hash,
        "decision": "approve",
        "statement": "reviewed",
        "evidence": "ticket-1",
        "role": "owner",
        "name_or_id": "P. Owner",
        "date": "2026-11-01",
        "scope": SCOPE,
        "expires": "2026-12-01",
    }
    fields.update(changes)
    return dl.Attestation.create(key, **fields)


# ---------------------------------------------------------------- opening


def test_open_creates_the_ledger_and_reopens_the_same_loop(tmp_path):
    loop = open_loop(tmp_path)
    assert (tmp_path / "deploy" / dl.FILE_NAME).exists()
    again = open_loop(tmp_path)
    assert again.ledger.head == loop.ledger.head


def test_a_different_charter_or_policy_is_refused(tmp_path):
    open_loop(tmp_path)
    other = types.SimpleNamespace(charter_hash="b" * 64)
    with pytest.raises(dl.DeploymentLedgerError, match="charter"):
        loop_module.Loop.open(tmp_path / "deploy", other, make_policy())
    with pytest.raises(dl.DeploymentLedgerError, match="policy"):
        open_loop(tmp_path, make_policy(r_star=0.04))


def test_policy_validation():
    with pytest.raises(ValueError):
        make_policy(r_star=1.5)
    with pytest.raises(ValueError):
        make_policy(reference_risk=0.99, tolerance=0.5)
    with pytest.raises(ValueError, match="safe_default"):
        make_policy(safe_default=" ")


def test_state_that_does_not_match_the_ledger_starts_halted(tmp_path):
    loop = open_loop(tmp_path)
    stream = streams.simulate(spec(batches=3, seed=1))
    feed(loop, stream, 0, 2)
    state = tmp_path / "deploy" / loop_module.STATE_FILE
    saved = state.read_text()
    feed(loop, stream, 2, 3)
    state.write_text(saved)
    again = open_loop(tmp_path)
    assert again.halted
    assert again.halt_reason == loop_module.HALT_STATE
    routed = again.observe(
        batch_frame(stream, 0), np.zeros(50), ["approve"] * 50
    )
    assert routed.routing.coverage == 0.0


# ---------------------------------------------------------------- routing


def test_cases_are_routed_by_score_and_capacity(tmp_path):
    loop = open_loop(tmp_path, make_policy(c_max=0.2))
    stream = streams.simulate(spec(batches=1, seed=2))
    sizes = np.where(np.arange(50) < 20, 3, 1)
    outcome = loop.observe(
        batch_frame(stream, 0, set_size=sizes), np.zeros(50), ["approve"] * 50
    )
    status = outcome.routing.status.tolist()
    assert status.count("answered") == 30
    assert status.count("escalated") == 10
    assert status.count("safe_default") == 10
    assert outcome.actions.count("defer") == 10
    assert outcome.actions.count("escalate") == 10
    summary = rows(loop, dl.OBSERVE_SUMMARY)[-1]["body"]
    assert (summary["answered"], summary["safe_default"]) == (30, 10)


def test_everything_escalates_until_the_first_calibration(tmp_path):
    loop = open_loop(tmp_path, make_policy(initial_tau=None, c_max=1.0))
    stream = streams.simulate(spec(batches=1, seed=2))
    outcome = loop.observe(batch_frame(stream, 0), np.zeros(50), ["a"] * 50)
    assert outcome.routing.coverage == 0.0
    data = streams.draw_calibration(3000, 0.05, seed=1)
    result = loop.calibrate(data["score"], data["loss"])
    assert not result.everything_escalates
    assert (tmp_path / "deploy" / loop_module.RISK_COVERAGE_FILE).exists()
    assert loop.tau == result.tau


def test_observe_validates_alignment(tmp_path):
    loop = open_loop(tmp_path)
    stream = streams.simulate(spec(batches=1, seed=2))
    with pytest.raises(ValueError, match="align"):
        loop.observe(batch_frame(stream, 0), np.zeros(3), ["a"] * 50)


def test_out_of_support_high_impact_cases_are_not_answered(tmp_path):
    loop = open_loop(
        tmp_path, make_policy(high_impact_actions=("deny",), c_max=1.0)
    )
    stream = streams.simulate(spec(batches=1, seed=2))
    novel = np.arange(50) < 5
    decisions = ["deny"] * 25 + ["approve"] * 25
    outcome = loop.observe(
        batch_frame(stream, 0, novel=novel), np.zeros(50), decisions
    )
    assert outcome.routing.status[:5].tolist() == ["escalated"] * 5
    assert outcome.routing.status[5:].tolist() == ["answered"] * 45
    assert outcome.actions[:5] == ("escalate",) * 5
    assert (
        rows(loop, dl.OBSERVE_SUMMARY)[-1]["body"]["blocked_high_impact"] == 5
    )


# ---------------------------------------------------------------- L1


def test_a_stable_stream_raises_no_alarm_and_no_retrain(tmp_path):
    loop = open_loop(tmp_path)
    stream = streams.simulate(spec(batches=60, seed=3, base=0.10))
    result = feed(loop, stream)
    assert result.status != loop_module.HALTED
    assert loop.mechanism.value == "complete"
    assert not rows(loop, dl.ALARM) and not rows(loop, dl.RETRAIN)


def test_an_abrupt_rise_alarms_retrains_and_registers_the_challenger(tmp_path):
    calls = []

    def retrain():
        calls.append(1)
        return "challenger-1"

    loop = open_loop(tmp_path, hooks=loop_module.Hooks(retrain=retrain))
    stream = streams.simulate(
        spec(kind="abrupt", batches=80, change=30, delta=0.25, seed=4)
    )
    feed(loop, stream)
    alarms = [r for r in rows(loop, dl.ALARM) if r["body"]["rule"] == "L1"]
    assert alarms and alarms[0]["body"]["batch"] >= 30
    assert calls
    assert rows(loop, dl.RETRAIN)[0]["body"]["challenger"] == "challenger-1"
    registered = [
        r
        for r in rows(loop, dl.CHALLENGER_RESULT)
        if r["body"]["phase"] == dl.REGISTERED
    ]
    assert registered[0]["body"]["name"] == "challenger-1"


def test_covariate_drift_alone_does_not_retrain(tmp_path):
    loop = open_loop(tmp_path)
    stream = streams.simulate(
        spec(kind="covariate", batches=80, change=30, seed=5)
    )
    feed(loop, stream)
    assert not rows(loop, dl.RETRAIN)


def test_mnar_without_an_audit_slice_never_retrains_and_halts_on_an_alarm(
    tmp_path,
):
    loop = open_loop(tmp_path, hooks=loop_module.Hooks(retrain=lambda: "x"))
    config = spec(
        kind="abrupt",
        labels="mnar",
        batches=80,
        change=20,
        delta=0.3,
        epsilon=0.0,
        seed=6,
    )
    stream = streams.simulate(config)
    result = None
    for index in range(config.batches):
        frame = batch_frame(stream, index).drop(columns=["propensity", "audit"])
        loop.observe(frame, np.zeros(50), ["a"] * 50)
        arrived = {
            f"b{b}-{p}": stream.loss[b, p]
            for b in range(index + 1)
            for p in np.flatnonzero(
                (stream.arrival[b] == index) & stream.selected[b]
            )
        }
        loop.mature(arrived)
        result = loop.step()
    assert not rows(loop, dl.RETRAIN)
    assert not rows(loop, dl.CHALLENGER_RESULT)
    assert loop.mechanism.value == "unverified"
    assert result.mechanism.value == "unverified"
    assert loop.ledger.halted
    assert any(
        e.question["action"] == "l1_alarm_unverified_label_mechanism"
        for e in loop.pending_escalations()
    )
    label_file = json.loads(
        (tmp_path / "deploy" / loop_module.LABEL_MECHANISM_FILE).read_text()
    )
    assert label_file["actions_permitted"] is False


def test_mar_with_logged_propensity_still_acts(tmp_path):
    loop = open_loop(
        tmp_path,
        make_policy(epsilon=0.1),
        hooks=loop_module.Hooks(retrain=lambda: "c"),
    )
    config = spec(
        kind="abrupt",
        labels="mar",
        batches=300,
        change=60,
        delta=0.3,
        floor=0.1,
        seed=7,
    )
    feed(loop, streams.simulate(config))
    assert rows(loop, dl.RETRAIN)
    assert loop.mechanism.value == "logged_propensity"


# ---------------------------------------------------------------- L2, L3


def paired(advantages, seed=1):
    sequence = streams.challenger_sequence(advantages, 120, 100, 0.2, seed)
    stream = streams.simulate(spec(batches=120, size=100, seed=seed))
    object.__setattr__(stream, "loss", sequence.champion)
    return stream, sequence


def test_a_better_challenger_yields_a_promotion_proposal_in_order(tmp_path):
    loop = open_loop(
        tmp_path,
        make_policy(
            reference_risk=0.2,
            tolerance=0.3,
            challenger_budget=100,
            reply_timeout=1000,
        ),
    )
    stream, sequence = paired([0.0, 0.12])
    loop.register_challenger("null")
    loop.register_challenger("better")
    feed(
        loop,
        stream,
        challengers={
            "null": sequence.challengers[0],
            "better": sequence.challengers[1],
        },
    )
    decided = {
        r["body"]["name"]: r["body"]
        for r in rows(loop, dl.CHALLENGER_RESULT)
        if r["body"]["phase"] == dl.DECIDED
    }
    assert decided["better"]["rejected"] is True
    assert decided["null"]["rejected"] is False
    proposals = rows(loop, dl.PROMOTION_PROPOSAL)
    assert [p["body"]["name"] for p in proposals] == ["better"]
    order = [
        r["body"]["name"]
        for r in rows(loop, dl.CHALLENGER_RESULT)
        if r["body"]["phase"] == dl.DECIDED
    ]
    assert order == ["null", "better"]
    loop.ledger.verify()


def test_a_challenger_without_evidence_escalates_at_the_label_budget(tmp_path):
    loop = open_loop(
        tmp_path,
        make_policy(
            reference_risk=0.2,
            tolerance=0.3,
            challenger_budget=20,
            reply_timeout=1000,
        ),
    )
    stream, sequence = paired([0.0])
    loop.register_challenger("same")
    feed(loop, stream, challengers={"same": sequence.challengers[0]})
    assert not rows(loop, dl.PROMOTION_PROPOSAL)
    assert any(
        e.question["action"] == "challenger_evidence_insufficient"
        for e in loop.pending_escalations()
    )


def test_a_worse_new_champion_creates_a_rollback_proposal_and_tightens_tau(
    tmp_path,
):
    loop = open_loop(tmp_path, make_policy(reference_risk=0.2, tolerance=0.3))
    stream, sequence = paired([0.12])
    loop.watch_rollback("previous")
    feed(loop, stream, challengers={"previous": sequence.challengers[0]})
    assert len(rows(loop, dl.ROLLBACK_PROPOSAL)) == 1
    assert loop.tau == pytest.approx(0.55)
    assert loop.rollback_watch is None


def test_two_consecutive_rollbacks_escalate(tmp_path):
    loop = open_loop(tmp_path, make_policy(reference_risk=0.2, tolerance=0.3))
    for seed in (1, 2):
        stream, sequence = paired([0.12], seed=seed)
        loop.watch_rollback("previous")
        loop.pending_batches = []
        loop.batch = 0
        feed(loop, stream, challengers={"previous": sequence.challengers[0]})
    assert any(
        e.question["action"] == "two_consecutive_rollbacks"
        for e in loop.pending_escalations()
    )


# ------------------------------------------------------------------- halt


def test_selective_risk_above_target_halts_and_coverage_goes_to_zero(tmp_path):
    loop = open_loop(
        tmp_path, make_policy(reference_risk=0.5, tolerance=0.4, r_star=0.05)
    )
    stream = streams.simulate(spec(base=0.3, batches=60, seed=8))
    feed(loop, stream)
    assert loop.ledger.halted
    assert loop.halt_reason == loop_module.HALT_SELECTIVE
    after = loop.observe(batch_frame(stream, 0), np.zeros(50), ["a"] * 50)
    assert after.routing.coverage == 0.0
    assert any(
        e.question["action"] == "halt_selective_risk"
        for e in loop.pending_escalations()
    )


def test_halt_is_released_only_by_a_passing_recalibration(tmp_path):
    loop = open_loop(
        tmp_path, make_policy(reference_risk=0.5, tolerance=0.4, r_star=0.05)
    )
    stream = streams.simulate(spec(base=0.3, batches=60, seed=8))
    feed(loop, stream)
    bad = streams.draw_calibration(30, 0.05, seed=1)
    assert loop.calibrate(bad["score"], bad["loss"]).everything_escalates
    assert loop.ledger.halted
    good = streams.draw_calibration(3000, 0.05, seed=2)
    result = loop.calibrate(good["score"], good["loss"])
    assert not loop.ledger.halted and loop.tau == result.tau


def test_lowering_tau_after_the_first_calibration_needs_an_attestation(
    tmp_path,
):
    loop = open_loop(tmp_path, make_policy(initial_tau=None))
    data = streams.draw_calibration(3000, 0.05, seed=1)
    first = loop.calibrate(data["score"], data["loss"])
    loop.tau = min(1.0, first.tau + 0.2)
    with pytest.raises(never_list.AttestationRequired):
        loop.calibrate(data["score"], data["loss"])


def test_coverage_and_novelty_triggers_escalate(tmp_path):
    loop = open_loop(
        tmp_path, make_policy(coverage_target=0.9, novelty_bound=0.05)
    )
    stream = streams.simulate(spec(batches=40, seed=9))
    for index in range(40):
        frame = batch_frame(stream, index, novel=np.arange(50) < 20)
        loop.observe(frame, np.zeros(50), ["a"] * 50)
        ids = list(frame["case_id"])
        loop.mature(
            dict.fromkeys(ids, 0.0),
            covered={i: float(n % 2) for n, i in enumerate(ids)},
        )
        loop.step()
    actions = {e.question["action"] for e in loop.pending_escalations()}
    assert {
        "coverage_below_nominal",
        "out_of_support_rate_above_bound",
    } <= actions


# ---------------------------------------------------------------- reply


def pending_one(tmp_path):
    loop = open_loop(tmp_path)
    with pytest.raises(never_list.AttestationRequired):
        loop.move_champion(
            types.SimpleNamespace(promote=lambda *a, **k: "moved"), "m", 3, None
        )
    return loop, loop.pending_escalations()[0]


def test_a_valid_reply_is_recorded_and_attests_the_decision(tmp_path):
    loop, esc = pending_one(tmp_path)
    outcome = loop.reply(esc.id, reply_to(esc))
    assert outcome.tier is qualification.Tier.ATTESTED
    assert loop.pending_escalations() == []
    registry = types.SimpleNamespace(promote=lambda *a, **k: "moved")
    assert loop.move_champion(registry, "m", 3, None) == "moved"


def test_rejecting_or_undecided_replies_apply_the_safe_default(tmp_path):
    loop, esc = pending_one(tmp_path)
    outcome = loop.reply(esc.id, reply_to(esc, decision="cannot_decide"))
    assert outcome.tier is qualification.Tier.UNVERIFIED
    assert outcome.safe_default_applies
    with pytest.raises(never_list.AttestationRequired):
        loop.move_champion(
            types.SimpleNamespace(promote=lambda *a, **k: 1), "m", 3, None
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"question_hash": "f" * 64}, "different question"),
        ({"scope": "other"}, "scope"),
        ({"expires": "2026-11-02"}, "expired"),
        ({"date": "2026-11-10", "expires": "2026-12-10"}, "future"),
        ({"escalation_id": "esc-99"}, "different escalation"),
    ],
)
def test_replies_that_do_not_fit_are_refused(tmp_path, changes, message):
    loop, esc = pending_one(tmp_path)
    with pytest.raises(loop_module.ReplyRefused, match=message):
        loop.reply(esc.id, reply_to(esc, **changes))
    assert len(loop.pending_escalations()) == 1


def test_signature_key_and_system_key_checks(tmp_path):
    loop, esc = pending_one(tmp_path)
    with pytest.raises(loop_module.ReplyRefused, match="system holds"):
        loop.reply(esc.id, reply_to(esc, key=SYSTEM_KEY))
    with pytest.raises(loop_module.ReplyRefused, match="no verification key"):
        loop.reply(esc.id, reply_to(esc, key="unknown-person-key"))
    forged = reply_to(esc).to_dict()
    forged["statement"] = "changed after signing"
    with pytest.raises(loop_module.ReplyRefused, match="signature"):
        loop.reply(esc.id, dl.Attestation.from_dict(forged))
    with pytest.raises(loop_module.ReplyRefused, match="not a pending"):
        loop.reply("esc-0", reply_to(esc))


def test_unanswered_escalations_time_out_with_the_safe_default(tmp_path):
    loop = open_loop(tmp_path, make_policy(reply_timeout=3))
    with pytest.raises(never_list.AttestationRequired):
        loop.rewrite_ledger(lambda: None, {"op": "truncate"})
    stream = streams.simulate(spec(batches=6, seed=1))
    feed(loop, stream, 0, 6)
    assert loop.pending_escalations() == []
    closing = [
        r
        for r in rows(loop, dl.ESCALATION)
        if r["body"]["phase"] == dl.PHASE_TIMEOUT
    ]
    assert closing and closing[0]["body"]["tier"] == "unverified"


def test_a_tampered_ledger_halts_the_loop(tmp_path):
    loop = open_loop(tmp_path)
    stream = streams.simulate(spec(batches=4, seed=1))
    feed(loop, stream, 0, 3)
    path = tmp_path / "deploy" / dl.FILE_NAME
    lines = path.read_text().splitlines()
    path.write_text("\n".join([lines[0], *lines[2:]]) + "\n")
    result = loop.step()
    assert result.status == loop_module.HALTED
    assert loop.halt_reason == loop_module.HALT_LEDGER
    assert (
        loop.observe(
            batch_frame(stream, 0), np.zeros(50), ["a"] * 50
        ).routing.coverage
        == 0.0
    )


# ---------------------------------------------------------------- never list


def attested_loop(tmp_path):
    return open_loop(tmp_path)


NEVER_CASES = {
    "move_champion": lambda loop, calls: loop.move_champion(
        types.SimpleNamespace(promote=lambda *a, **k: calls.append("promote")),
        "m",
        2,
        None,
    ),
    "rollback": lambda loop, calls: loop.execute_rollback(
        lambda: calls.append("rb"), {"to": 1}
    ),
    "reopen_sealed": lambda loop, calls: loop.reopen_sealed(
        lambda: calls.append("c"), {"set": "C"}
    ),
    "change_charter": lambda loop, calls: loop.amend_policy(
        safe_default="approve"
    ),
    "rewrite_ledger": lambda loop, calls: loop.rewrite_ledger(
        lambda: calls.append("x"), {"op": "rm"}
    ),
    "relax_threshold": lambda loop, calls: loop.amend_policy(r_star=0.9),
    "act_on_unverified": lambda loop, calls: loop.act_on_unverified(
        "retrain", lambda: calls.append("r")
    ),
    "silence_alert": lambda loop, calls: loop.silence_alert(
        "slo", lambda: calls.append("s")
    ),
    "change_label_policy": lambda loop, calls: loop.change_label_policy(
        {"epsilon": 0.0}, lambda: calls.append("l")
    ),
    "delete_or_send": lambda loop, calls: loop.delete_or_send(
        "bundle", lambda: calls.append("d")
    ),
}


@pytest.mark.parametrize("name", sorted(NEVER_CASES))
def test_never_list_action_without_attestation_raises_and_escalates(
    tmp_path, name
):
    loop = attested_loop(tmp_path)
    calls = []
    before = len(rows(loop, dl.ESCALATION))
    with pytest.raises(never_list.AttestationRequired) as caught:
        NEVER_CASES[name](loop, calls)
    assert calls == []
    assert len(rows(loop, dl.ESCALATION)) == before + 1
    assert caught.value.escalation_id == loop.pending_escalations()[-1].id
    # asking again does not pile up a second identical escalation
    with pytest.raises(never_list.AttestationRequired):
        NEVER_CASES[name](loop, calls)
    assert len(rows(loop, dl.ESCALATION)) == before + 1


def test_high_impact_answer_is_blocked_in_routing_and_guarded_as_an_action(
    tmp_path,
):
    loop = attested_loop(tmp_path)
    with pytest.raises(never_list.AttestationRequired):
        loop.perform(
            never_list.NeverAction.ANSWER_HIGH_IMPACT,
            {"case": "c1"},
            lambda: "answered",
        )


def test_the_system_cannot_attest_for_itself(tmp_path):
    loop, esc = pending_one(tmp_path)
    assert not hasattr(loop, "sign") and not hasattr(loop, "attest")
    with pytest.raises(loop_module.ReplyRefused):
        loop.reply(esc.id, reply_to(esc, key=SYSTEM_KEY))
    environment_key = "env-key"
    os.environ["SCMLE_SIGNING_KEY"] = environment_key
    try:
        other = open_loop(tmp_path / "second")
        with pytest.raises(never_list.AttestationRequired):
            other.move_champion(
                types.SimpleNamespace(promote=lambda *a, **k: 1), "m", 3, None
            )
        esc2 = other.pending_escalations()[0]
        with pytest.raises(loop_module.ReplyRefused):
            other.reply(
                esc2.id,
                reply_to(esc2, key=environment_key),
                key=environment_key,
            )
    finally:
        del os.environ["SCMLE_SIGNING_KEY"]


def test_tightening_is_free_and_loosening_is_not(tmp_path):
    loop = open_loop(tmp_path)
    updated = loop.amend_policy(r_star=0.03, delta_promo=0.01)
    assert updated.r_star == 0.03
    with pytest.raises(never_list.AttestationRequired):
        loop.amend_policy(r_star=0.9)
    esc = loop.pending_escalations()[-1]
    loop.reply(esc.id, reply_to(esc))
    assert loop.amend_policy(r_star=0.9).r_star == 0.9
    again = open_loop(tmp_path, loop.policy)
    assert again.policy.r_star == 0.9
    with pytest.raises(ValueError, match="unknown"):
        loop.amend_policy(nothing=1)


def test_attestation_is_specific_to_its_exact_request(tmp_path):
    loop, esc = pending_one(tmp_path)
    loop.reply(esc.id, reply_to(esc))
    registry = types.SimpleNamespace(promote=lambda *a, **k: "moved")
    assert loop.move_champion(registry, "m", 3, None) == "moved"
    with pytest.raises(never_list.AttestationRequired):
        loop.move_champion(registry, "m", 4, None)


def test_an_expired_attestation_no_longer_allows(tmp_path):
    clock = {"today": TODAY}
    loop = loop_module.Loop.open(
        tmp_path / "d",
        CHARTER,
        make_policy(),
        verification_keys={dl.key_fingerprint(HUMAN_KEY): HUMAN_KEY},
        system_keys=[SYSTEM_KEY],
        today=lambda: clock["today"],
    )
    registry = types.SimpleNamespace(promote=lambda *a, **k: "moved")
    with pytest.raises(never_list.AttestationRequired):
        loop.move_champion(registry, "m", 3, None)
    esc = loop.pending_escalations()[0]
    loop.reply(esc.id, reply_to(esc))
    assert loop.move_champion(registry, "m", 3, None) == "moved"
    clock["today"] = datetime.date(2027, 1, 1)
    with pytest.raises(never_list.AttestationRequired):
        loop.move_champion(registry, "m", 3, None)


def single_use_loop(tmp_path):
    return loop_module.Loop.open(
        tmp_path / "d",
        CHARTER,
        make_policy(),
        verification_keys={dl.key_fingerprint(HUMAN_KEY): HUMAN_KEY},
        system_keys=[SYSTEM_KEY],
        today=lambda: TODAY,
        single_use_attestations=True,
    )


def test_a_single_use_attestation_allows_one_action_and_is_ledgered(tmp_path):
    loop = single_use_loop(tmp_path)
    registry = types.SimpleNamespace(promote=lambda *a, **k: "moved")
    with pytest.raises(never_list.AttestationRequired):
        loop.move_champion(registry, "m", 3, None)
    esc = loop.pending_escalations()[0]
    loop.reply(esc.id, reply_to(esc))
    assert loop.move_champion(registry, "m", 3, None) == "moved"
    assert len(rows(loop, dl.ATTESTATION_USED)) == 1
    with pytest.raises(never_list.AttestationRequired):
        loop.move_champion(registry, "m", 3, None)
    # asking again needs a new escalation and a new reply
    assert len(loop.pending_escalations()) == 1


def test_the_default_keeps_an_attestation_valid_until_it_expires(tmp_path):
    loop, esc = pending_one(tmp_path)
    loop.reply(esc.id, reply_to(esc))
    registry = types.SimpleNamespace(promote=lambda *a, **k: "moved")
    assert loop.move_champion(registry, "m", 3, None) == "moved"
    assert loop.move_champion(registry, "m", 3, None) == "moved"
    assert rows(loop, dl.ATTESTATION_USED) == []


def test_a_used_marker_survives_reopening_the_ledger(tmp_path):
    loop = single_use_loop(tmp_path)
    registry = types.SimpleNamespace(promote=lambda *a, **k: "moved")
    with pytest.raises(never_list.AttestationRequired):
        loop.move_champion(registry, "m", 3, None)
    loop.reply(
        loop.pending_escalations()[0].id,
        reply_to(loop.pending_escalations()[0]),
    )
    loop.move_champion(registry, "m", 3, None)
    again = single_use_loop(tmp_path)
    with pytest.raises(never_list.AttestationRequired):
        again.move_champion(registry, "m", 3, None)


def test_a_used_marker_for_an_unanswered_escalation_is_refused(tmp_path):
    loop = single_use_loop(tmp_path)
    with pytest.raises(dl.DeploymentLedgerError):
        loop.ledger.commit(dl.ATTESTATION_USED, {"escalation_id": "esc-9"})


def test_an_uncertified_alarm_escalates_instead_of_retraining(tmp_path):
    calls = []
    loop = open_loop(
        tmp_path,
        make_policy(alarm_gates_retrain=False),
        hooks=loop_module.Hooks(retrain=lambda: calls.append(1)),
    )
    stream = streams.simulate(
        spec(kind="abrupt", batches=80, change=30, delta=0.25, seed=4)
    )
    feed(loop, stream)
    assert rows(loop, dl.ALARM) and not rows(loop, dl.RETRAIN) and not calls
    assert any(
        e.question["action"] == "l1_alarm_investigate"
        for e in loop.pending_escalations()
    )


def test_a_rollback_proposal_never_lowers_an_infinite_tau(tmp_path):
    loop = open_loop(tmp_path)
    assert math.isinf(loop.tau)
    loop.rollback_watch = {
        "previous": "v1",
        "process": types.SimpleNamespace(crossed=True, e_value=50.0),
    }
    loop.decide_rollback()
    assert math.isinf(loop.tau)
