"""Delayed labels into the F13 loop, the retraining policy, gated replacement.

The loop here keeps ``alarm_gates_retrain`` False, as the certified loop
does: its alarm escalates for investigation and trains nothing. Retraining
proposals come from the operations policy and travel the same escalation
path; a person's signed reply is what lets a recommended retrain run.
"""

import dataclasses
import json
import types

import numpy as np
import pandas as pd
import pytest

from scmle.governance import deployment_ledger as dl
from scmle.operations import continuous
from scmle.operations import signals
from tests import support_ops as ops
from tests import support_release as sr
from tests import support_runtime as rt
from tests.support_continuous import candidate_for
from tests.support_continuous import degraded
from tests.support_continuous import evaluation
from tests.support_continuous import kinds
from tests.support_continuous import loss_signals
from tests.support_continuous import make_loop
from tests.support_continuous import MODEL
from tests.support_continuous import release_world
from tests.support_continuous import reply_to
from tests.support_continuous import winning_challenger


def named(name, category, breached, value=0.5, samples=500):
    return signals.Signal(
        name,
        category,
        signals.Status.BREACHED if breached else signals.Status.OK,
        value,
        0.2,
        "max",
        samples,
    )


SPECS = [
    continuous.MetricSpec(
        "conversion", "business", "higher", margin=0.0, primary=True
    ),
    continuous.MetricSpec("latency_ms", "operational", "lower", margin=5.0),
]


def test_the_f13_alarm_escalates_for_investigation_and_trains_nothing(tmp_path):
    loop, ignored, ignored, called = degraded(tmp_path)
    assert kinds(loop, dl.ALARM) and not kinds(loop, dl.RETRAIN)
    assert not called and loop.policy.alarm_gates_retrain is False
    actions = [e.question["action"] for e in loop.pending_escalations()]
    assert "l1_alarm_investigate" in actions


def test_a_recommendation_reaches_a_person_and_trains_only_after_approval(
    tmp_path,
):
    loop, ev, ignored, called = degraded(tmp_path)
    decision = ev.review(loss_signals(0.4))
    assert decision.action == "recommend" and decision.escalation_id
    assert decision.alarm_gates_retrain is False and not called
    pending = {e.id: e for e in loop.pending_escalations()}
    question = pending[decision.escalation_id].question
    assert question["action"] == continuous.RETRAIN_QUESTION
    assert pending[decision.escalation_id].proposed_default == "no_retrain"
    with pytest.raises(ValueError, match="no approving reply"):
        ev.run_approved_retrain(decision.escalation_id)
    loop.reply(
        decision.escalation_id, reply_to(pending[decision.escalation_id])
    )
    assert ev.run_approved_retrain(decision.escalation_id) == "challenger-1"
    assert called == [1]
    registered = [
        r
        for r in kinds(loop, dl.CHALLENGER_RESULT)
        if r["body"]["phase"] == dl.REGISTERED
    ]
    assert registered[0]["body"]["name"] == "challenger-1"
    assert kinds(loop, dl.RETRAIN)[0]["body"]["mode"] == "recommend"


def test_a_rejecting_reply_does_not_start_a_retrain(tmp_path):
    loop, ev, ignored, called = degraded(tmp_path)
    decision = ev.review(loss_signals(0.4))
    target = next(
        e for e in loop.pending_escalations() if e.id == decision.escalation_id
    )
    loop.reply(target.id, reply_to(target, "reject"))
    with pytest.raises(ValueError):
        ev.run_approved_retrain(decision.escalation_id)
    assert not called


def test_trigger_mode_trains_a_challenger_at_once_and_still_escalates(tmp_path):
    loop, ev, ignored, called = degraded(
        tmp_path, policy=continuous.RetrainPolicy(mode="trigger")
    )
    decision = ev.review(loss_signals(0.4))
    assert (
        decision.action == "trigger" and decision.challenger == "challenger-1"
    )
    assert called == [1] and decision.escalation_id
    assert kinds(loop, dl.RETRAIN)[0]["body"]["mode"] == "trigger"
    assert loop.policy.alarm_gates_retrain is False
    assert "challenger-1" in loop.ledger.registered


def test_a_proposal_is_not_repeated_during_the_cooldown(tmp_path):
    loop, ev, ignored, ignored = degraded(tmp_path)
    first = ev.review(loss_signals(0.4))
    again = ev.review(loss_signals(0.4))
    assert first.action == "recommend" and again.action == "none"
    assert "cooldown" in again.held_back
    assert [e.question["action"] for e in loop.pending_escalations()].count(
        continuous.RETRAIN_QUESTION
    ) == 1


# ---- the policy on its own ---------------------------------------------


def judge(rows, **options):
    return continuous.judge_retraining(
        continuous.RetrainPolicy(**options.pop("policy", {})),
        rows,
        options.pop("alarm", False),
        options.pop("age", None),
        options.pop("since", None),
    )


def test_drift_alone_is_never_a_reason_to_retrain():
    drift = named("feature_psi_max", signals.Category.DATA_QUALITY, True)
    primary, support, held = judge([drift])
    assert primary == [] and support and held is None


def test_performance_with_too_few_outcomes_is_not_enough():
    few = named(
        "windowed_loss", signals.Category.MODEL_PERFORMANCE, True, samples=40
    )
    assert judge([few])[0] == []
    assert len(judge([few], policy={"min_outcomes": 30})[0]) == 1


def test_age_and_alarm_are_primary_reasons_when_declared():
    assert judge([], age=400, policy={"max_age_days": 365})[0]
    assert not judge([], age=100, policy={"max_age_days": 365})[0]
    assert not judge([], age=400)[0]
    assert judge([], alarm=True)[0] == ["the loop's L1 alarm fired"]


def test_open_infrastructure_or_data_faults_hold_a_proposal_back():
    loss = named("windowed_loss", signals.Category.MODEL_PERFORMANCE, True)
    infra = named("availability", signals.Category.INFRASTRUCTURE, True)
    data = named("invalid_request_rate", signals.Category.DATA_QUALITY, True)
    assert "infrastructure" in judge([loss, infra])[2]
    assert "data-quality" in judge([loss, data])[2]
    assert judge([loss])[2] is None
    assert "cooldown" in judge([loss], since=3)[2]
    assert judge([loss], since=30)[2] is None


def test_a_held_back_reason_is_recorded_and_nothing_is_escalated(tmp_path):
    loop = make_loop(tmp_path)
    ev = evaluation(tmp_path, loop)
    rows = [
        *loss_signals(0.4),
        named("availability", signals.Category.INFRASTRUCTURE, True),
    ]
    decision = ev.review(rows)
    assert decision.action == "none" and "infrastructure" in decision.held_back
    assert not loop.pending_escalations()
    assert ev.history.of_kind("retrain_review")


def test_policy_and_spec_validation():
    with pytest.raises(ValueError):
        continuous.RetrainPolicy(mode="auto")
    with pytest.raises(ValueError):
        continuous.RetrainPolicy(min_outcomes=0)
    with pytest.raises(ValueError):
        continuous.RetrainPolicy(max_age_days=0)
    with pytest.raises(ValueError):
        continuous.MetricSpec("x", "finance", "higher")
    with pytest.raises(ValueError):
        continuous.MetricSpec("x", "business", "bigger")


# ---- feeding -----------------------------------------------------------


def test_business_outcomes_become_bounded_losses_and_are_kept(tmp_path):
    loop = make_loop(tmp_path)
    ev = evaluation(tmp_path, loop)
    loop.observe(
        pd.DataFrame({"case_id": ["a", "b", "c"]}),
        np.zeros(3),
        ["approve"] * 3,
    )
    assert ev.feed_outcomes({"a": 0.0, "b": 50.0, "c": 400.0}, cap=100.0) == 3
    stored = loop.pending_batches[0]["labels"]
    assert stored == {"a": 0.0, "b": 0.5, "c": 1.0}
    kept = [
        json.loads(line)
        for line in (tmp_path / "ops" / continuous.LABELS_FILE)
        .read_text()
        .splitlines()
    ]
    assert kept[0]["kind"] == "business_outcomes"
    assert kept[0]["values"]["c"] == 400.0
    with pytest.raises(ValueError):
        ev.feed_outcomes({"a": -1.0}, cap=10.0)
    with pytest.raises(ValueError):
        ev.feed_outcomes({"a": 1.0}, cap=0.0)


# ---- history -----------------------------------------------------------


def test_datasets_are_versioned_by_content_and_preserved(tmp_path):
    ev = evaluation(tmp_path, make_loop(tmp_path))
    frame = ops.frame(50, seed=1)
    first = ev.record_training("m1", frame, {"auc": 0.9}, "initial")
    again = ev.record_training("m1b", frame.copy(), {"auc": 0.9}, "same data")
    other = ev.record_training("m2", ops.frame(50, seed=2), {"auc": 0.8}, "new")
    version = first["facts"]["dataset_version"]
    assert version == again["facts"]["dataset_version"]
    assert version != other["facts"]["dataset_version"]
    saved = tmp_path / "ops" / continuous.DATASET_DIRECTORY
    assert len(list(saved.iterdir())) == 2
    reloaded = continuous.load_dataset(tmp_path / "ops", version)
    assert continuous.dataset_version(reloaded) == version
    path = saved / (version.replace(":", "-") + ".csv")
    path.write_text(path.read_text().replace("basic", "other", 1))
    with pytest.raises(ValueError, match="has changed"):
        continuous.load_dataset(tmp_path / "ops", version)
    shuffled = frame.sample(frac=1.0, random_state=1)[list(frame.columns)[::-1]]
    assert (
        continuous.dataset_version(shuffled.reset_index(drop=True)) != version
    )
    assert len(ev.history.rows()) == 3


def test_the_history_chain_detects_edits(tmp_path):
    ev = evaluation(tmp_path, make_loop(tmp_path))
    ev.record_training("m", ops.frame(20), {}, "r", preserve=False)
    ev.record_training("n", ops.frame(21), {}, "r", preserve=False)
    path = ev.history.path
    text = path.read_text()
    path.write_text(text.replace('"rows": 20', '"rows": 99'))
    with pytest.raises(ValueError, match="broken"):
        ev.history.rows()
    path.write_text(text)
    assert len(ev.history.rows()) == 2
    lines = text.splitlines()
    path.write_text(lines[1] + "\n")
    with pytest.raises(ValueError):
        ev.history.rows()


# ---- challenger against champion, then replacement ---------------------


def test_a_challenger_that_wins_is_proposed_by_the_loop_in_order(tmp_path):
    loop, ev = winning_challenger(tmp_path)
    assert ev.promotion_proposals() == ["challenger-1"]
    assert ev.history.of_kind("promotion_proposal")
    decided = [
        r["body"]
        for r in kinds(loop, dl.CHALLENGER_RESULT)
        if r["body"]["phase"] == dl.DECIDED
    ]
    assert decided[0]["rejected"] is True
    loop.ledger.verify()


def test_a_new_model_deploys_only_after_the_proposal_the_gate_and_the_approval(
    tmp_path,
):
    ignored, ev = winning_challenger(tmp_path)
    release, plane, store = release_world(tmp_path)
    assert release.promote(
        MODEL, 1, plane, "staging", rt.OLD, signing_key=sr.KEY
    ).done
    good = candidate_for(store)
    unsigned = dataclasses.replace(
        good, model=dataclasses.replace(good.model, signed=False)
    )
    failed_scan = dataclasses.replace(
        good, scan=dataclasses.replace(good.scan, passed=False)
    )
    unapproved = dataclasses.replace(good, approval=None)
    self_approved = dataclasses.replace(good, approver=good.requested_by)
    for name, bad in {
        "unsigned": unsigned,
        "scan": failed_scan,
        "approval": unapproved,
        "same approver": self_approved,
    }.items():
        refused = ev.replace("challenger-1", release, plane, bad, sr.KEY)
        assert not refused.done and refused.reasons, name
        assert release.environment(MODEL, "staging")["version"] == 1, name
    no_proposal = ev.replace("someone-else", release, plane, good, sr.KEY)
    assert not no_proposal.done
    assert "no promotion proposal" in no_proposal.reasons[0]
    assert release.environment(MODEL, "staging")["version"] == 1
    done = ev.replace("challenger-1", release, plane, good, sr.KEY)
    assert done.done and done.outcome.version == 2
    assert release.environment(MODEL, "staging")["version"] == 2
    assert release.reconcile(MODEL, "staging", plane.observe()).consistent
    kinds_seen = [r["kind"] for r in ev.history.rows()]
    assert kinds_seen.count("replacement_refused") == 5
    assert kinds_seen.count("replacement") == 1


def test_production_replacement_needs_the_signed_staging_record(tmp_path):
    ignored, ev = winning_challenger(tmp_path)
    release, plane, store = release_world(tmp_path)
    prod = sr.candidate(
        "production",
        model=store.resolve(MODEL, 2),
        approval=rt.approval_for(2),
    )
    refused = ev.replace("challenger-1", release, plane, prod, sr.KEY)
    assert not refused.done
    assert any("staging deployment record" in r for r in refused.reasons)


def test_a_platform_that_does_not_confirm_leaves_the_registry_alone(tmp_path):
    ignored, ev = winning_challenger(tmp_path)
    release, ignored, store = release_world(tmp_path)
    plane = rt.MemoryPlane(never_settles=True)
    result = ev.replace(
        "challenger-1", release, plane, candidate_for(store), sr.KEY
    )
    assert not result.done and result.outcome is not None
    assert release.environment(MODEL, "staging") is None


def test_an_offline_comparison_that_keeps_the_champion_blocks_replacement(
    tmp_path,
):
    ignored, ev = winning_challenger(tmp_path)
    release, plane, store = release_world(tmp_path)
    keep = types.SimpleNamespace(decision="keep")
    result = ev.replace(
        "challenger-1",
        release,
        plane,
        candidate_for(store),
        sr.KEY,
        comparison=keep,
    )
    assert not result.done and "offline comparison" in result.reasons[0]


# ---- after promotion ---------------------------------------------------


def periods(seed=0, n=300, shift_conv=0.0, shift_latency=0.0):
    rng = np.random.default_rng(seed)
    before = {
        "conversion": rng.binomial(1, 0.10, n).astype(float),
        "latency_ms": rng.normal(120, 20, n),
    }
    after = {
        "conversion": rng.binomial(1, 0.10 + shift_conv, n).astype(float),
        "latency_ms": rng.normal(120 + shift_latency, 20, n),
    }
    return before, after


def test_a_clear_gain_is_reported_with_its_interval_not_a_point():
    before, after = periods(shift_conv=0.12)
    result = continuous.post_promotion_check(SPECS, before, after)
    assert result["overall"] == "helped"
    conversion = result["metrics"][0]
    low, high = conversion["interval"]
    assert 0 < low < high and conversion["confidence"] == 0.975
    assert "cannot separate the model" in result["statement"]
    assert "helped" in result["statement"]


def test_a_regression_in_any_selected_metric_is_harm():
    before, after = periods(shift_conv=0.12, shift_latency=40.0)
    result = continuous.post_promotion_check(SPECS, before, after)
    assert result["overall"] == "harmed"
    assert result["metrics"][1]["verdict"] == "worse"


def test_no_change_and_small_samples_are_inconclusive_not_helped():
    before, after = periods(seed=5)
    assert continuous.post_promotion_check(SPECS, before, after)["overall"] == (
        "inconclusive"
    )
    before, after = periods(n=10, shift_conv=0.5)
    small = continuous.post_promotion_check(SPECS, before, after)
    assert small["overall"] == "inconclusive"
    assert small["metrics"][0]["interval"] is None


def test_the_check_is_deterministic_and_validates_inputs(tmp_path):
    before, after = periods(shift_conv=0.1)
    first = continuous.post_promotion_check(SPECS, before, after, seed=4)
    assert first == continuous.post_promotion_check(
        SPECS, before, after, seed=4
    )
    with pytest.raises(ValueError):
        continuous.post_promotion_check([], before, after)
    with pytest.raises(ValueError):
        continuous.post_promotion_check(SPECS, {"conversion": [1.0]}, after)
    ev = evaluation(tmp_path, make_loop(tmp_path))
    recorded = ev.check_after_promotion(SPECS, before, after)
    assert (
        recorded["overall"]
        == ev.history.of_kind("post_promotion_check")[0]["facts"]["overall"]
    )


# ---- holdback ----------------------------------------------------------


def test_holdback_assignment_is_deterministic_and_hits_its_fraction():
    ids = [f"user-{k}" for k in range(20000)]
    first = continuous.holdback_assignment(ids, 0.1, "release-7")
    assert (
        first == continuous.holdback_assignment(ids, 0.1, "release-7")
    ).all()
    assert abs(first.mean() - 0.1) < 0.01
    other = continuous.holdback_assignment(ids, 0.1, "release-8")
    assert (first != other).any()
    # a unit's side does not depend on the other units in the call
    assert (
        first[17]
        == continuous.holdback_assignment(ids[17:18], 0.1, "release-7")[0]
    )


def test_holdback_assignment_rejects_a_fraction_outside_the_open_unit_interval():
    for bad in (0.0, 1.0, -0.2, 1.5):
        with pytest.raises(ValueError):
            continuous.holdback_assignment(["a"], bad, "s")


def test_a_concurrent_holdback_attributes_the_change_to_the_model():
    rng = np.random.default_rng(0)
    holdback = {
        "conversion": rng.binomial(1, 0.10, 400).astype(float),
        "latency_ms": rng.normal(120, 20, 400),
    }
    promoted = {
        "conversion": rng.binomial(1, 0.25, 3600).astype(float),
        "latency_ms": rng.normal(120, 20, 3600),
    }
    result = continuous.holdback_check(
        SPECS, holdback, promoted, expected_fraction=0.1
    )
    assert result["overall"] == "helped"
    assert result["design"] == "concurrent_holdback"
    assert "attributable" in result["statement"]
    assert "cannot separate the model" not in result["statement"]
    assert result["ratio_check"]["flag"] is False
    assert result["ratio_check"]["holdback_units"] == 400


def test_a_holdback_of_the_wrong_size_is_flagged_and_blocks_a_helped_verdict():
    rng = np.random.default_rng(1)
    holdback = {
        "conversion": rng.binomial(1, 0.10, 100).astype(float),
        "latency_ms": rng.normal(120, 20, 100),
    }
    promoted = {
        "conversion": rng.binomial(1, 0.30, 3900).astype(float),
        "latency_ms": rng.normal(120, 20, 3900),
    }
    result = continuous.holdback_check(
        SPECS, holdback, promoted, expected_fraction=0.2
    )
    assert result["ratio_check"]["flag"] is True
    assert result["overall"] == "inconclusive"
    assert "assignment" in result["statement"]
