"""Unit tests of the loop submodules, each imported by its full name."""

import dataclasses
import datetime
import json
import math
import types

import numpy
import pandas
import pytest

from scmle.contracts import errors
from scmle.contracts import qualification
from scmle.governance import deployment_ledger
from scmle.governance import never_list
from scmle.governance.loop import alarms
from scmle.governance.loop import contest
from scmle.governance.loop import cycle
from scmle.governance.loop import desk
from scmle.governance.loop import guard
from scmle.governance.loop import intake
from scmle.governance.loop import memory
from scmle.governance.loop import risk
from scmle.governance.loop import terms
from scmle.monitoring import labels
from scmle.monitoring import sequential_tests

HUMAN_KEY = "human-key"
SYSTEM_KEY = "system-key"
TODAY = datetime.date(2026, 11, 4)
CHARTER = types.SimpleNamespace(charter_hash="a" * 64)
SCOPE = "model-v1:stream-a"


def make_policy(**changes):
    values = {
        "reference": 0.10,
        "tolerance": 0.05,
        "scope": SCOPE,
        "tau": 0.5,
        "matured": 3,
        "target": 0.6,
        "maturity": 0,
        "timeout": 2,
    }
    values.update(changes)
    return terms.Preregistration(**values)


def open_loop(tmp_path, policy=None, hooks=None):
    return cycle.Loop.open(
        tmp_path / "deploy",
        CHARTER,
        policy or make_policy(),
        hooks=hooks,
        verification_keys={deployment_ledger.thumbprint(HUMAN_KEY): HUMAN_KEY},
        system_keys=[SYSTEM_KEY],
        today=lambda: TODAY,
    )


def kinds_of(loop):
    return [row["kind"] for row in loop.ledger.rows]


def reply_to(pending, key=HUMAN_KEY, **changes):
    fields = {
        "escalation": pending.id,
        "digests": {"question": pending.question_hash},
        "decision": "approve",
        "statement": "reviewed",
        "evidence": "ticket-1",
        "role": "owner",
        "attestor": "P. Owner",
        "date": "2026-11-01",
        "scope": SCOPE,
        "expires": "2026-12-01",
    }
    fields.update(changes)
    return deployment_ledger.Attestation.create(key, **fields)


def frame(size=4, prefix="c"):
    return pandas.DataFrame({"case_id": [f"{prefix}{i}" for i in range(size)]})


def crossed():
    return types.SimpleNamespace(crossed=True, e_value=40.0)


# ---- memory ----------------------------------------------------------------


def test_opening_arms_monitors_and_saves_state(tmp_path):
    loop = open_loop(tmp_path)
    assert isinstance(loop, memory.Memory)
    assert loop.statefile == tmp_path / "deploy" / "loop_state.json"
    state = json.loads(loop.statefile.read_text())
    assert state["ledger_head"] == loop.ledger.head
    assert state["tau"] == 0.5
    assert state["batch"] == 0
    assert loop.test is None
    assert loop.halted is False
    assert loop.guard.observer == loop.save


def test_an_infinite_tau_is_saved_as_none(tmp_path):
    loop = open_loop(tmp_path, make_policy(tau=None))
    assert math.isinf(loop.tau)
    assert json.loads(loop.statefile.read_text())["tau"] is None


def test_restore_reads_back_the_saved_state(tmp_path):
    loop = open_loop(tmp_path, make_policy(coverage=0.9))
    loop.tau = 0.25
    loop.batch = 3
    loop.queue = ["x"]
    loop.save()
    again = open_loop(tmp_path, make_policy(coverage=0.9))
    assert (again.tau, again.batch, again.queue) == (0.25, 3, ["x"])
    assert again.test is not None
    assert again.frozen is False


def test_a_state_behind_the_ledger_halts(tmp_path):
    loop = open_loop(tmp_path)
    loop.ledger.commit(deployment_ledger.ALARM, {"batch": 0})
    again = open_loop(tmp_path)
    assert again.frozen is True
    assert again.reason == terms.HALT_STATE
    assert again.halted is True


def test_recheck_halts_in_memory_when_the_ledger_was_edited(tmp_path):
    loop = open_loop(tmp_path)
    assert loop.recheck() is True
    path = loop.ledger.path
    path.write_text(path.read_text().replace('"r', '"R', 1))
    assert loop.recheck() is False
    assert loop.reason == terms.HALT_LEDGER
    assert loop.halted is True


# ---- desk ------------------------------------------------------------------


def test_escalate_writes_once_per_question(tmp_path):
    loop = open_loop(tmp_path)
    assert isinstance(loop, desk.Desk)
    first = loop.escalate("q", {"a": 1}, evidence=("e.json",))
    assert first.id == f"esc-{len(loop.ledger.rows) - 1}"
    assert first.question == {"action": "q", "detail": {"a": 1}, "scope": SCOPE}
    assert first.evidence == ("e.json",)
    assert first.proposed_default == "defer"
    assert loop.escalate("q", {"a": 1}) is None
    other = loop.escalate(
        never_list.Forbidden.EDIT_LEDGER, {"a": 1}, proposed_default="keep"
    )
    assert other.question["action"] == "edit_ledger_or_register_late"
    assert other.proposed_default == "keep"
    assert [e.id for e in loop.survey()] == [first.id, other.id]


def test_suspend_halts_once_and_calls_the_incident_hook(tmp_path):
    reasons = []
    loop = open_loop(tmp_path, hooks=terms.Hooks(open_incident=reasons.append))
    loop.suspend("why", {"batch": 1})
    assert kinds_of(loop)[-2:] == ["halt", "incident"]
    assert loop.ledger.rows[-1]["body"] == {
        "status": "open",
        "reason": "why",
        "batch": 0,
    }
    assert loop.reason == "why"
    assert reasons == ["why"]
    size = len(loop.ledger.rows)
    loop.suspend("again", {})
    assert len(loop.ledger.rows) == size
    assert reasons == ["why"]


def test_reply_records_an_approval_as_attested(tmp_path):
    loop = open_loop(tmp_path)
    pending = loop.escalate("q", {})
    out = loop.reply(pending.id, reply_to(pending))
    assert out.tier is qualification.Tier.ATTESTED
    assert out.decision is deployment_ledger.Reply.APPROVE
    assert out.safe_default_applies is False
    assert kinds_of(loop)[-1] == "attestation"
    assert loop.survey() == []


def test_a_rejecting_reply_keeps_the_safe_default(tmp_path):
    loop = open_loop(tmp_path)
    pending = loop.escalate("q", {})
    out = loop.reply(pending.id, reply_to(pending, decision="reject"))
    assert out.tier is qualification.Tier.UNVERIFIED
    assert out.safe_default_applies is True


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"escalation": "esc-0"}, "answers a different escalation"),
        ({"digests": {"question": "x"}}, "different question"),
        ({"scope": "other"}, "different scope"),
        ({"key": "stranger"}, "no verification key"),
        ({"key": SYSTEM_KEY}, "key the system holds"),
        ({"date": "2026-11-05", "expires": "2026-11-06"}, "expired or dated"),
        ({"expires": "2026-11-03"}, "expired or dated"),
    ],
)
def test_reply_refuses_a_reply_that_does_not_fit(tmp_path, changes, message):
    loop = open_loop(tmp_path)
    pending = loop.escalate("q", {})
    key = changes.pop("key", HUMAN_KEY)
    with pytest.raises(terms.RefusalError, match=message):
        loop.reply(pending.id, reply_to(pending, key=key, **changes))


def test_reply_refuses_an_unknown_escalation_and_a_bad_signature(tmp_path):
    loop = open_loop(tmp_path)
    pending = loop.escalate("q", {})
    with pytest.raises(terms.RefusalError, match="not a pending escalation"):
        loop.reply("esc-99", reply_to(pending))
    forged = dataclasses.replace(reply_to(pending), statement="other")
    with pytest.raises(terms.RefusalError, match="signature does not verify"):
        loop.reply(pending.id, forged)
    with pytest.raises(terms.RefusalError, match="verification key is one"):
        loop.reply(pending.id, reply_to(pending), key=SYSTEM_KEY)


def test_expire_closes_escalations_after_the_timeout(tmp_path):
    loop = open_loop(tmp_path)
    pending = loop.escalate("q", {}, proposed_default="keep")
    loop.batch = 1
    assert loop.expire() is None
    assert loop.ledger.pending() != []
    loop.batch = 2
    loop.expire()
    assert loop.ledger.rows[-1]["body"] == {
        "phase": "timeout",
        "id": pending.id,
        "applied": "keep",
        "tier": "unverified",
    }
    assert loop.ledger.pending() == []


def test_perform_runs_only_with_an_attestation(tmp_path):
    loop = open_loop(tmp_path)
    calls = []
    with pytest.raises(never_list.UnattestedError):
        loop.perform(
            never_list.Forbidden.REOPEN_SEALED,
            {"s": 1},
            lambda: calls.append(1),
        )
    assert calls == []
    pending = loop.survey()[-1]
    loop.reply(pending.id, reply_to(pending))
    out = loop.perform(
        never_list.Forbidden.REOPEN_SEALED, {"s": 1}, lambda: "done"
    )
    assert out == "done"


# ---- risk ------------------------------------------------------------------


def test_estimate_and_centred_weigh_observed_cases(tmp_path):
    loop = open_loop(tmp_path)
    assert isinstance(loop, risk.Risk)
    record = {"propensity": None, "audit": None}
    losses = numpy.array([0.0, 1.0, 0.5, 0.5])
    observed = numpy.ones(4, dtype=bool)
    out = loop.estimate(record, losses, observed)
    assert out.mean == pytest.approx(0.5)
    assert out.mechanism is labels.Mechanism.COMPLETE
    part = loop.estimate(
        record, losses, observed, numpy.array([1, 1, 0, 0], bool)
    )
    assert part.mean == pytest.approx(0.5)
    signed = loop.centred(
        record,
        numpy.array([-0.5, 0.5, 1.0, 0.0]),
        observed,
        numpy.ones(4, bool),
    )
    assert signed.mean == pytest.approx(0.25)


def test_calibrate_sets_tau_and_writes_the_curve(tmp_path):
    loop = open_loop(tmp_path, make_policy(tau=None))
    score = numpy.linspace(0, 1, 200)
    loss = (score > 0.7).astype(float)
    result = loop.calibrate(score, loss)
    assert loop.tau == result.tau
    assert loop.calibrated is True
    stored = json.loads((loop.directory / terms.RISK_COVERAGE_FILE).read_text())
    assert set(stored) == {"curve", "threshold"}
    assert "calibration" in loop.ledger.rows[-1]["body"]


def test_performativity_needs_a_bypass_slice(tmp_path):
    loop = open_loop(tmp_path)
    with pytest.raises(errors.InputError, match="no bypass slice is declared"):
        loop.performativity(pandas.DataFrame(), pandas.DataFrame())


def test_label_mechanism_file_states_feedback_is_unverified(tmp_path):
    loop = open_loop(tmp_path)
    loop.declare()
    record = json.loads(
        (loop.directory / terms.LABEL_MECHANISM_FILE).read_text()
    )
    assert record["mechanism"] == "complete"
    assert record["feedback"] == labels.UNVERIFIED
    assert record["feedback_note"] == labels.FEEDBACK_UNVERIFIED
    assert record["bypass_rate"] == 0.0
    assert record["outcome_window"] == 0
    declared = open_loop(tmp_path / "b", make_policy(bypass=0.1))
    declared.declare()
    again = json.loads(
        (declared.directory / terms.LABEL_MECHANISM_FILE).read_text()
    )
    assert again["feedback"] == "bypass slice declared"
    assert again["feedback_note"] == ""


# ---- contest ---------------------------------------------------------------


def test_enroll_registers_and_queues_a_challenger(tmp_path):
    loop = open_loop(tmp_path)
    assert isinstance(loop, contest.Contest)
    assert loop.enroll("c1") == 1
    assert loop.enroll("c2") == 2
    assert loop.queue == ["c1", "c2"]
    assert loop.active["c1"]["batches"] == 0
    assert loop.ledger.registered == ["c1", "c2"]


def test_refit_calls_the_hook_or_records_a_skip(tmp_path):
    loop = open_loop(tmp_path, hooks=terms.Hooks(retrain=lambda: "fresh"))
    loop.refit(4)
    assert loop.ledger.rows[-2]["body"] == {
        "reason": "L1 alarm",
        "batch": 4,
        "challenger": "fresh",
    }
    assert loop.queue == ["fresh"]
    loop.refit(5)
    assert loop.ledger.rows[-1]["body"]["skipped"] == (
        "a challenger is already pending"
    )
    assert loop.ledger.rows[-1]["body"]["challenger"] is None


def test_propose_does_nothing_without_a_crossed_watch(tmp_path):
    loop = open_loop(tmp_path)
    assert loop.propose() is None
    loop.surveil("old")
    assert loop.watch["previous"] == "old"
    assert loop.propose() is None


def test_a_crossed_watch_proposes_a_rollback_and_raises_tau(tmp_path):
    loop = open_loop(tmp_path, make_policy(tau=0.98, step=0.05))
    loop.watch = {"previous": "old", "process": crossed()}
    loop.batch = 3
    assert loop.propose() is None
    assert loop.tau == 1.0
    assert loop.ledger.rows[-1]["body"] == {
        "previous": "old",
        "e_value": 40.0,
        "batch": 2,
        "tau_before": 0.98,
    }
    loop.watch = {"previous": "old", "process": crossed()}
    second = loop.propose()
    assert second.question["action"] == "two_consecutive_rollbacks"
    assert loop.rollbacks == 2


def test_drain_stops_at_a_challenger_without_evidence_or_budget(tmp_path):
    loop = open_loop(tmp_path, make_policy(budget=2))
    loop.enroll("c1")
    assert loop.drain() is None
    assert loop.queue == ["c1"]
    loop.active["c1"]["batches"] = 2
    newest = loop.drain()
    assert newest.question["action"] == "challenger_evidence_insufficient"
    assert loop.queue == []
    decided = loop.ledger.decided["c1"]
    assert decided["rejected"] is False
    assert decided["matured_batches"] == 2


def test_drain_promotes_a_challenger_with_enough_evidence(tmp_path):
    loop = open_loop(tmp_path)
    loop.enroll("c1")
    loop.active["c1"]["process"] = types.SimpleNamespace(e_value=1e6)
    assert loop.drain() is None
    assert kinds_of(loop)[-1] == "promotion_proposal"
    assert loop.ledger.rows[-1]["body"]["name"] == "c1"
    assert loop.rollbacks == 0


# ---- alarms ----------------------------------------------------------------


def test_a_naive_alarm_on_unverified_labels_halts(tmp_path):
    loop = open_loop(tmp_path)
    assert isinstance(loop, alarms.Alarms)
    loop.naive_flag = True
    newest = loop.alert(3, verified=False)
    assert newest.question["action"] == "l1_alarm_unverified_label_mechanism"
    assert loop.naive_flag is False
    assert loop.reason == terms.HALT_UNVERIFIED
    assert loop.ledger.halted is True
    alarm_row = next(r for r in loop.ledger.rows if r["kind"] == "alarm")
    assert alarm_row["body"]["rule"] == "L1_naive"


def test_an_l1_alarm_escalates_without_retrain_and_is_replaced(tmp_path):
    loop = open_loop(tmp_path)
    loop.alarm = types.SimpleNamespace(test=crossed())
    newest = loop.alert(2, verified=True)
    assert newest.question["action"] == "l1_alarm_investigate"
    assert isinstance(loop.alarm, sequential_tests.Tripwire)
    body = next(r for r in loop.ledger.rows if r["kind"] == "alarm")["body"]
    assert body["rule"] == "L1"
    assert body["e_value"] == 40.0


def test_an_l1_alarm_with_retrain_asks_for_a_challenger(tmp_path):
    loop = open_loop(
        tmp_path,
        make_policy(retrain=True),
        hooks=terms.Hooks(retrain=lambda: "c9"),
    )
    loop.alarm = types.SimpleNamespace(test=crossed(), to_dict=dict)
    assert loop.alert(2, verified=True) is None
    assert loop.queue == ["c9"]


def test_a_selective_alarm_halts_and_escalates(tmp_path):
    loop = open_loop(tmp_path)
    loop.selective = types.SimpleNamespace(test=crossed())
    newest = loop.alert(1, verified=True)
    assert newest.question["action"] == "halt_selective_risk"
    assert loop.reason == terms.HALT_SELECTIVE
    assert isinstance(loop.selective, sequential_tests.Tripwire)


def test_nothing_crossed_raises_no_alarm(tmp_path):
    loop = open_loop(tmp_path)
    assert loop.alert(0, verified=True) is None
    assert loop.alert(0, verified=False) is None
    assert loop.screen(0) is None
    assert "alarm" not in kinds_of(loop)


def test_screen_escalates_coverage_and_novelty(tmp_path):
    loop = open_loop(tmp_path, make_policy(coverage=0.9))
    loop.test = crossed()
    loop.novelty = crossed()
    newest = loop.screen(5)
    assert newest.question["action"] == "out_of_support_rate_above_bound"
    actions = [e.question["action"] for e in loop.survey()]
    assert actions == [
        "coverage_below_nominal",
        "out_of_support_rate_above_bound",
    ]
    assert isinstance(loop.test, sequential_tests.Mixture)
    assert isinstance(loop.novelty, sequential_tests.Mixture)


# ---- intake ----------------------------------------------------------------


def test_observe_refuses_misaligned_or_empty_batches(tmp_path):
    loop = open_loop(tmp_path)
    assert isinstance(loop, intake.Intake)
    with pytest.raises(errors.InputError, match="must align"):
        loop.observe(frame(2), [0], ["a", "a"])
    with pytest.raises(errors.InputError, match="must align"):
        loop.observe(frame(0), [], [])


def test_observe_routes_logs_and_advances_the_batch(tmp_path):
    loop = open_loop(tmp_path)
    out = loop.observe(frame(3), [0, 0, 0], ["go", "go", "go"])
    assert out.batch == 0
    assert out.actions == ("go", "go", "go")
    assert loop.batch == 1
    body = loop.ledger.rows[-1]["body"]
    assert body["cases"] == 3
    assert body["answered"] == 3
    assert body["tau"] == 0.5
    assert body["halted"] is False
    assert loop.batches[0]["ids"] == ["c0", "c1", "c2"]
    generated = loop.observe(
        pandas.DataFrame({"x": [1, 2]}), [0, 0], ["a", "b"]
    )
    assert generated.batch == 1
    assert loop.batches[1]["ids"] == ["b1-0", "b1-1"]


def test_a_halted_loop_escalates_every_case(tmp_path):
    loop = open_loop(tmp_path, make_policy(escalation=1.0))
    loop.suspend("why", {})
    out = loop.observe(frame(2), [0, 0], ["go", "go"])
    assert out.actions == ("escalate", "escalate")


def test_flag_reads_a_column_or_gives_false():
    data = pandas.DataFrame({"novel": [1, 0]})
    assert intake.Intake.flag(data, "novel").tolist() == [True, False]
    assert intake.Intake.flag(data, "audit").tolist() == [False, False]


def test_mature_matches_known_ids_only(tmp_path):
    loop = open_loop(tmp_path)
    loop.observe(frame(3), [0, 0, 0], ["go"] * 3)
    matched = loop.mature(
        {"c0": 0.1, "c1": 0.2, "zz": 0.3},
        covered={"c0": 1, "zz": 0},
        challengers={"k": {"c0": 0.0, "zz": 1.0}},
    )
    assert matched == 2
    record = loop.batches[0]
    assert record["labels"] == {"c0": 0.1, "c1": 0.2}
    assert record["covered"] == {"c0": 1.0}
    assert record["challenger_losses"] == {"k": {"c0": 0.0}}


def test_feed_needs_enough_matured_labels(tmp_path):
    loop = open_loop(tmp_path)
    loop.observe(frame(4), [0] * 4, ["go"] * 4)
    record = loop.batches[0]
    loop.mature({"c0": 0.1, "c1": 0.1})
    assert loop.feed(record) is False
    loop.mature({"c2": 0.1})
    assert loop.feed(record) is True
    assert loop.mechanism is not None


# ---- guard -----------------------------------------------------------------


def test_amend_policy_refuses_unknown_fields(tmp_path):
    loop = open_loop(tmp_path)
    assert isinstance(loop, guard.Gatekeeper)
    with pytest.raises(
        errors.InputError, match=r"unknown policy fields: \['x'\]"
    ):
        loop.amend(x=1)


def test_tightening_a_threshold_needs_no_attestation(tmp_path):
    loop = open_loop(tmp_path)
    updated = loop.amend(target=0.5, promotion={"delta": 0.1})
    assert updated.target == 0.5
    assert updated.promotion == {"alpha": "1/20", "delta": 0.1}
    body = loop.ledger.rows[-1]["body"]["policy_amendment"]
    assert body["target"] == 0.5
    assert body["schema_version"] == terms.VERSION
    assert loop.policy is updated


def test_loosening_or_changing_the_charter_needs_an_attestation(tmp_path):
    loop = open_loop(tmp_path)
    with pytest.raises(never_list.UnattestedError):
        loop.amend(target=0.7)
    assert loop.policy.target == 0.6
    question = loop.survey()[-1].question
    assert question["action"] == "relax_threshold"
    assert question["detail"] == {"field": "target", "from": 0.6, "to": 0.7}
    with pytest.raises(never_list.UnattestedError):
        loop.amend(fallback="approve")
    assert loop.survey()[-1].question["action"] == "change_charter"


def test_turning_on_the_coverage_test_is_amended_or_gated(tmp_path):
    loop = open_loop(tmp_path)
    with pytest.raises(never_list.UnattestedError):
        loop.amend(coverage=0.9)
    assert loop.policy.coverage is None
    question = loop.survey()[-1].question
    assert question["action"] == "change_charter"
    assert question["detail"] == {"field": "coverage", "from": None, "to": 0.9}


ACTION_FIRST = (
    "silence",
    "act_on_unverified",
    "delete_or_send",
    "change_label_policy",
)


@pytest.mark.parametrize(
    ("method", "detail", "action"),
    [
        ("silence", "alert-1", "suppress_own_alert"),
        ("rewrite", {"row": 3}, "edit_ledger_or_register_late"),
        ("reopen_sealed", {"set": "C"}, "reopen_sealed_set"),
        ("act_on_unverified", "retrain", "act_on_unverified_stream"),
        ("delete_or_send", "model", "delete_or_send_external"),
        ("change_label_policy", {"rate": 0.2}, "change_labelling_policy"),
        ("revert", {"to": 1}, "move_champion_or_roll_back"),
    ],
)
def test_each_never_list_action_needs_its_attestation(
    tmp_path, method, detail, action
):
    loop = open_loop(tmp_path)
    calls = []

    def do():
        calls.append(1)

    call = (detail, do) if method in ACTION_FIRST else (do, detail)
    with pytest.raises(never_list.UnattestedError):
        getattr(loop, method)(*call)
    assert calls == []
    assert loop.survey()[-1].question["action"] == action


def test_move_champion_through_an_attached_guard_skips_the_attestation(
    tmp_path,
):
    loop = open_loop(tmp_path)
    moved = []
    registry = types.SimpleNamespace(
        guard=loop.guard,
        promote=lambda m, v, approval, signing_key: (
            moved.append((m, v)) or "ok"
        ),
        attach=lambda g, s: moved.append(("attach", s)),
    )
    loop.intercept(registry)
    assert loop.move(registry, "m", 2, approval=None) == "ok"
    assert moved == [("attach", SCOPE), ("m", 2)]
    registry.guard = None
    with pytest.raises(never_list.UnattestedError):
        loop.move(registry, "m", 3, approval=None)


# ---- cycle -----------------------------------------------------------------


def test_open_refuses_another_charter_and_resume_reads_the_ledger(tmp_path):
    loop = open_loop(tmp_path)
    with pytest.raises(
        deployment_ledger.MisuseError, match="different charter"
    ):
        cycle.Loop.open(
            tmp_path / "deploy",
            types.SimpleNamespace(charter_hash="b" * 64),
            make_policy(),
        )
    with pytest.raises(deployment_ledger.MisuseError, match="different policy"):
        open_loop(tmp_path, make_policy(target=0.5))
    again = cycle.Loop.resume(tmp_path / "deploy", today=lambda: TODAY)
    assert again.ledger.head == loop.ledger.head
    assert again.policy == loop.policy


def test_step_without_labels_reports_insufficient_labels(tmp_path):
    loop = open_loop(tmp_path)
    loop.observe(frame(2), [0, 0], ["go", "go"])
    out = loop.step()
    assert out.status == terms.INSUFFICIENT_LABELS
    assert out.escalation is None
    assert out.events == ()
    assert loop.batches == []


def test_step_with_enough_labels_is_ok(tmp_path):
    loop = open_loop(tmp_path)
    loop.observe(frame(4), [0] * 4, ["go"] * 4)
    loop.mature({f"c{i}": 0.1 for i in range(4)})
    out = loop.step()
    assert out.status == terms.OK
    assert out.mechanism is labels.Mechanism.COMPLETE


def test_step_on_an_edited_ledger_halts_without_writing(tmp_path):
    loop = open_loop(tmp_path)
    path = loop.ledger.path
    path.write_text(path.read_text().replace('"r', '"R', 1))
    size = len(loop.ledger.rows)
    out = loop.step()
    assert out.status == terms.HALTED
    assert out.events == (("halt", terms.HALT_LEDGER),)
    assert len(loop.ledger.rows) == size


def test_decide_skips_the_queue_when_labels_are_unverified(tmp_path):
    loop = open_loop(tmp_path, make_policy(budget=0))
    loop.enroll("c1")
    loop.batch = 1
    loop.mechanism = labels.Mechanism.UNVERIFIED
    assert loop.decide() is None
    assert loop.queue == ["c1"]
    loop.mechanism = labels.Mechanism.COMPLETE
    newest = loop.decide()
    assert newest.question["action"] == "challenger_evidence_insufficient"
