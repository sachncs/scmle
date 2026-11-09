"""Closed-loop wiring: incidents on halt, the feedback test, guarded registry."""

import json

import numpy
import pandas
import pytest

from benchmarks.qualification import streams
from scmle.artifacts import bundle
from scmle.contracts import data
from scmle.contracts import tasks
from scmle.data.curation import preprocessing
from scmle.governance import deployment_ledger
from scmle.governance import lifecycle
from scmle.governance import never_list
import scmle.governance.loop
import scmle.governance.registry
import scmle.models.registry
from tests.unit import test_loop

APPROVAL_KEY = "promotion-review-key"
PLAN = data.FeaturePlan(("a", "b"), ())
MODELS = scmle.models.registry.ModelRegistry.with_builtins()


def halted_loop(tmp_path, hooks=None):
    loop = test_loop.open_loop(
        tmp_path,
        test_loop.make_policy(reference_risk=0.5, tolerance=0.4, r_star=0.05),
        hooks=hooks,
    )
    test_loop.feed(
        loop, streams.simulate(test_loop.spec(base=0.3, batches=60, seed=8))
    )
    return loop


def test_a_halt_writes_one_open_incident_row(tmp_path):
    loop = halted_loop(tmp_path)
    assert loop.ledger.halted
    incidents = test_loop.rows(loop, deployment_ledger.INCIDENT)
    assert len(incidents) == 1
    assert incidents[0]["body"]["status"] == "open"
    assert (
        incidents[0]["body"]["reason"] == scmle.governance.loop.HALT_SELECTIVE
    )
    loop.ledger.verify()


def test_the_open_incident_hook_runs_after_the_ledger_row(tmp_path):
    seen = []

    def hook(reason):
        text = (tmp_path / "deploy" / deployment_ledger.FILE_NAME).read_text()
        seen.append((reason, text.count('"kind": "incident"')))

    halted_loop(tmp_path, scmle.governance.loop.Hooks(open_incident=hook))
    assert seen == [(scmle.governance.loop.HALT_SELECTIVE, 1)]


def feedback_groups(gamma):
    stream = streams.feedback_stream(100, 100, gamma=gamma, bypass=0.3, seed=1)
    frame = pandas.DataFrame({"z": stream.next_z.reshape(-1)})
    flags = stream.bypass.reshape(-1)
    return frame[flags].iloc[:300], frame[~flags].iloc[:300]


def read_mechanism(loop):
    return json.loads(
        (
            loop.directory / scmle.governance.loop.LABEL_MECHANISM_FILE
        ).read_text()
    )


def test_feedback_cannot_be_tested_without_a_declared_bypass_slice(tmp_path):
    loop = test_loop.open_loop(tmp_path)
    bypass, model = feedback_groups(0.8)
    with pytest.raises(ValueError) as caught:
        loop.check_feedback(bypass, model)
    assert str(caught.value) == (
        "no bypass slice is declared (policy.bypass_rate is 0): "
        "feedback cannot be tested"
    )


def test_a_feedback_loop_is_recorded_escalated_and_survives_reopening(
    tmp_path,
):
    loop = test_loop.open_loop(tmp_path, test_loop.make_policy(bypass_rate=0.3))
    bypass, model = feedback_groups(0.8)
    check = loop.check_feedback(bypass, model)
    assert check.p_value is not None and check.p_value <= 0.05
    record = read_mechanism(loop)
    assert record["feedback"] == "loop_detected"
    assert record["feedback_test"]["n_bypass"] == 300
    assert any(
        e.question["action"] == "feedback_loop_detected"
        for e in loop.pending_escalations()
    )
    again = test_loop.open_loop(
        tmp_path, test_loop.make_policy(bypass_rate=0.3)
    )
    assert not again.halted
    assert again.feedback == loop.feedback


def test_no_feedback_loop_is_reported_without_an_escalation(tmp_path):
    loop = test_loop.open_loop(tmp_path, test_loop.make_policy(bypass_rate=0.3))
    bypass, model = feedback_groups(0.0)
    loop.check_feedback(bypass, model)
    assert read_mechanism(loop)["feedback"] == "no_loop_detected"
    assert not loop.pending_escalations()


# ------------------------------------------------------------ guarded registry


def traffic(rows=400, seed=0, flip=False):
    rng = numpy.random.default_rng(seed)
    frame = pandas.DataFrame(
        {"a": rng.normal(size=rows), "b": rng.normal(size=rows)}
    )
    signal = frame.a + 0.3 * frame.b + rng.normal(0, 0.5, rows)
    frame["y"] = ((-signal if flip else signal) > 0).astype(int)
    return frame


def bundle_of(frame):
    features = frame.drop(columns="y")
    fitted = preprocessing.build_pipeline(
        PLAN, MODELS.get("logistic_regression"), tasks.TaskType.BINARY, 0
    ).fit(features, frame.y)
    return bundle.ModelBundle.create(
        fitted,
        PLAN,
        features,
        "y",
        "binary_classification",
        "roc_auc",
        "logistic_regression",
    )


def approval_for(record):
    return scmle.governance.registry.PromotionApproval(
        model=record.model,
        version=record.version,
        content_id=record.content_id,
        reviewer="unit-reviewer",
        decision=scmle.governance.registry.PromotionDecision.APPROVED,
        reason="verified unit fixture",
        evidence=("unit-evidence.json",),
        approved_at="2026-10-20T00:00:00+00:00",
    ).sign(APPROVAL_KEY)


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    root = tmp_path_factory.mktemp("bundles")
    return (
        bundle_of(traffic()).save(root / "one"),
        bundle_of(traffic(seed=0, flip=True)).save(root / "two"),
    )


def guarded(tmp_path, exported):
    loop = test_loop.open_loop(tmp_path)
    store = scmle.governance.registry.ModelRegistryStore(tmp_path / "registry")
    loop.guard_registry(store)
    first = store.register("m", exported[0], trusted=True)
    second = store.register("m", exported[1], trusted=True)
    return loop, store, first, second


def approve_pending(loop):
    pending = loop.pending_escalations()
    assert len(pending) == 1
    loop.reply(pending[0].id, test_loop.reply_to(pending[0]))


def test_an_unattested_champion_move_is_refused_and_escalated(
    tmp_path, exported
):
    loop, store, *records = guarded(tmp_path, exported)
    first = records[0]
    with pytest.raises(never_list.AttestationRequired):
        store.promote(
            "m", 1, approval=approval_for(first), signing_key=APPROVAL_KEY
        )
    assert "champion" not in store.aliases("m")
    assert loop.pending_escalations()[0].question["detail"] == {
        "model": "m",
        "version": 1,
    }
    # The loop's saved state still matches the ledger the guard wrote to.
    assert not test_loop.open_loop(tmp_path).halted


def test_staging_aliases_stay_free(tmp_path, exported):
    store = guarded(tmp_path, exported)[1]
    store.promote("m", 2, alias=scmle.governance.registry.CHALLENGER)
    assert store.aliases("m") == {scmle.governance.registry.CHALLENGER: 2}


def test_promote_challenger_needs_the_attestation_and_the_approval(
    tmp_path, exported
):
    loop, store, *records = guarded(tmp_path, exported)
    second = records[1]
    store.promote("m", 2, alias=scmle.governance.registry.CHALLENGER)
    manager = lifecycle.LifecycleManager(store, "m", str(tmp_path / "work"))
    approval = approval_for(second)
    with pytest.raises(never_list.AttestationRequired):
        manager.promote_challenger(approval, signing_key=APPROVAL_KEY)
    approve_pending(loop)
    record = manager.promote_challenger(approval, signing_key=APPROVAL_KEY)
    assert record.version == 2 and store.resolve("m").version == 2


def test_a_rollback_needs_an_attestation_naming_the_rollback(
    tmp_path, exported
):
    loop, store, first, second = guarded(tmp_path, exported)
    # Put both versions in place without the guard, as an older deployment did.
    store.attach_guard(None, "stream")
    for record in (first, second):
        store.promote(
            "m",
            record.version,
            approval=approval_for(record),
            signing_key=APPROVAL_KEY,
        )
    loop.guard_registry(store)
    manager = lifecycle.LifecycleManager(store, "m", str(tmp_path / "work"))
    with pytest.raises(never_list.AttestationRequired):
        manager.check_rollback(
            traffic(1200, seed=20),
            "y",
            approval=approval_for(first),
            signing_key=APPROVAL_KEY,
        )
    assert store.resolve("m").version == 2
    approve_pending(loop)
    outcome = manager.check_rollback(
        traffic(1200, seed=20),
        "y",
        approval=approval_for(first),
        signing_key=APPROVAL_KEY,
    )
    assert outcome.rolled_back and store.resolve("m").version == 1


def test_a_guarded_loop_move_spends_a_single_use_attestation_once(
    tmp_path, exported
):
    loop = test_loop.open_loop(tmp_path, single_use_attestations=True)
    store = scmle.governance.registry.ModelRegistryStore(tmp_path / "registry")
    loop.guard_registry(store)
    record = store.register("m", exported[0], trusted=True)
    approval = approval_for(record)
    with pytest.raises(never_list.AttestationRequired):
        loop.move_champion(store, "m", 1, approval, APPROVAL_KEY)
    approve_pending(loop)
    loop.move_champion(store, "m", 1, approval, APPROVAL_KEY)
    assert store.resolve("m").version == 1
    assert len(test_loop.rows(loop, deployment_ledger.ATTESTATION_USED)) == 1


def test_a_refused_approval_does_not_spend_a_single_use_attestation(
    tmp_path, exported
):
    loop = test_loop.open_loop(tmp_path, single_use_attestations=True)
    store = scmle.governance.registry.ModelRegistryStore(tmp_path / "registry")
    loop.guard_registry(store)
    record = store.register("m", exported[0], trusted=True)
    approval = approval_for(record)
    with pytest.raises(never_list.AttestationRequired):
        store.promote("m", 1, approval=approval, signing_key=APPROVAL_KEY)
    approve_pending(loop)
    with pytest.raises(scmle.governance.registry.errors.ArtifactError):
        store.promote("m", 1, approval=None, signing_key=APPROVAL_KEY)
    assert test_loop.rows(loop, deployment_ledger.ATTESTATION_USED) == []
    store.promote("m", 1, approval=approval, signing_key=APPROVAL_KEY)
    assert store.resolve("m").version == 1
