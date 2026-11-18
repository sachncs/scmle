"""The governance and operations records write single-word keys (phase 5).

Each write-only record of the governance, deploy, monitoring, operations,
runners and workflow packages routes `to_dict` through its `KEYS` migration
(or, for the versioned model card, its `migrations`): the new keys are
written, and the downgrade gives back the version-1 layout, so a version-1
sample still reads where a reader exists. The self-hashing records (job,
workflow, alert id) and the signed promotion approval hash the version-1
layout, so the pinned digests below are the ones stored before the rename.
"""

import hashlib
import json
from pathlib import Path

import pytest

from scmle.contracts import observability
from scmle.contracts import records
from scmle.contracts import signing
from scmle.deploy import observation
from scmle.deploy import parity
from scmle.deploy import release
from scmle.governance import escalation
from scmle.governance import model_card
from scmle.governance import registry
from scmle.monitoring import abtest
from scmle.monitoring import alerts
from scmle.monitoring import slo
import scmle.monitoring.observability
from scmle.operations import notify
from scmle.operations import recovery
from scmle.operations import respond
from scmle.operations.continuous import policies
from scmle.runners import base
from scmle.runners import entrypoint
from scmle.workflow import definition

NOW = "2026-01-01T00:00:00+00:00"
DIGEST = "sha256:" + "a" * 64
KEY = "governance-records-key"
WORKFLOW_DIGEST = (
    "5d9f8fa8a008d0718bce869589c548ed551578730bc44eb57da3ba881bf3d7c0"
)
JOB_DIGEST = "7f9f25072040c409ad824e3df462c4e0b515190100bd1eca547332549dcaa35b"
ALERT_ID = "ef33f54b7fca6323d9d9bbdc79f220112d13d4e6ca6ec1e80592525134418bb0"


def seen() -> observation.Observation:
    return observation.Observation(
        provider="aws",
        service="serve",
        revision="serve:12",
        image_digest=DIGEST,
        ready=True,
        traffic=(observation.Traffic("serve:12", 100),),
        observed_at=NOW,
    )


def approval() -> registry.Sanction:
    return registry.Sanction(
        model="churn",
        version=3,
        content_id="9" * 64,
        reviewer="reviewer-1",
        decision=registry.Verdict.APPROVED,
        reason="evidence reviewed",
        evidence=("ticket-17",),
        approved_at=NOW,
    )


def card() -> model_card.Card:
    return model_card.Card(
        model_name="m",
        model_version="v1",
        model_fingerprint="sha256:x",
        task="binary_classification",
        metric="roc_auc",
        dataset_id="ds_1",
        snapshot_id="snap_1",
        intended_use="inference",
        out_of_scope=("elsewhere",),
        limitations=("review",),
        feature_names=("amount",),
        evaluation={},
        resources={},
        evidence_artifacts=("evaluation.json",),
        support_status="beta",
        generated_at=NOW,
    )


def target() -> slo.Slo:
    return slo.Slo(
        name="latency",
        metric_name="p95_latency_ms",
        kind=observability.Indicator.LATENCY,
        maximum=200.0,
        window_days=7,
        minimum_samples=2,
        owner="ml-on-call",
        runbook="https://runbooks.example/latency",
        alert_route="on-call",
    )


def assessment() -> slo.Assessment:
    return slo.Assessment(
        target=target(),
        observed_at=NOW,
        status=slo.Status.BREACHED,
        sample_count=3,
        in_window_count=2,
        observed_maximum=300.0,
        error_budget_remaining=0.0,
        reason="over",
    )


def alert() -> alerts.Alert:
    return alerts.Alert.from_evidence(
        route="ops",
        severity="high",
        title="breach",
        details={"k": 1},
        evidence={"e": 2},
        created_at=NOW,
    )


def delivery() -> notify.Delivery:
    return notify.Delivery(
        kind="pager",
        delivered=True,
        attempts=1,
        http_status=202,
        reason="delivered",
        idempotency_key="k-1",
    )


def job() -> base.Job:
    return base.Job(
        run_id="run-1",
        name="train",
        project={"task": "binary", "target": "y"},
        identity="role",
    )


class Verdict:
    """Stands in for a classification: only its record is written."""

    def to_dict(self):
        return {"verdict": "none"}


# record, the written keys it gains, the version-1 keys it loses
CASES = {
    "observation": (
        seen(),
        {"digests", "observed"},
        {"image_digest", "observed_at"},
    ),
    "parity": (
        parity.Parity(True, 1, 0.0, 0.01, ()),
        {"error"},
        {"max_probability_error"},
    ),
    "threshold": (
        escalation.Threshold(0.5, 0.9, 0.01, 0.02, 0.05, False, ()),
        {"target"},
        {"r_star"},
    ),
    "approval": (
        approval(),
        {"content", "approved"},
        {"content_id", "approved_at"},
    ),
    "abtest": (
        abtest.Comparison("auc", 0.7, 0.8, 0.1, (0.0, 0.2), 10, 12, "adopt"),
        {"scores", "rows"},
        {"control_score", "treatment_score", "control_rows", "treatment_rows"},
    ),
    "alert": (
        alert(),
        {"identifier", "digests", "created"},
        {"alert_id", "evidence_digest", "created_at"},
    ),
    "alert delivery": (
        alerts.Delivery("a" * 64, "ops", "delivered", 1, "ok", NOW),
        {"alert", "observed"},
        {"alert_id", "observed_at"},
    ),
    "observability": (
        scmle.monitoring.observability.Evaluation(True, NOW, (), (), (), ()),
        {"observed", "missing", "expired", "future", "breaches"},
        {
            "observed_at",
            "missing_signals",
            "expired_metrics",
            "future_metrics",
            "threshold_breaches",
        },
    ),
    "slo": (
        target(),
        {"metric", "window", "samples", "route"},
        {"metric_name", "window_days", "minimum_samples", "alert_route"},
    ),
    "assessment": (
        assessment(),
        {"observed", "samples", "windowed", "maximum", "remaining"},
        {
            "observed_at",
            "sample_count",
            "in_window_count",
            "observed_maximum",
            "error_budget_remaining",
            "incident_required",
        },
    ),
    "retrain": (
        policies.RetrainDecision("none", (), "cooldown", "e-1", None, False),
        {"withheld", "escalation", "retrain"},
        {"held_back", "escalation_id", "alarm_gates_retrain"},
    ),
    "notify": (
        delivery(),
        {"code", "idempotency"},
        {"http_status", "idempotency_key"},
    ),
    "measurement": (
        recovery.Measurement("service", 1.5, 0, "probe"),
        {"rto", "rpo"},
        {"rto_seconds", "rpo_lost"},
    ),
    "outcome": (
        recovery.Outcome("process_kill"),
        {"fault"},
        {"fault_id"},
    ),
    "response": (
        respond.Response(
            "i-1",
            Verdict(),
            (delivery(),),
            "t" * 64,
            "h" * 64,
        ),
        {"incident", "digests", "head"},
        {"incident_id", "telemetry_digest", "audit_head"},
    ),
    "job": (
        job(),
        {"identifier", "timeout", "digests"},
        {"run_id", "timeout_seconds", "project_digest"},
    ),
    "step": (
        definition.standard_workflow().step("train"),
        {"needs", "timeout", "budget", "concurrency", "expiry"},
        {
            "depends_on",
            "timeout_seconds",
            "cost_budget",
            "concurrency_group",
            "expires_seconds",
        },
    ),
    "workflow": (
        definition.standard_workflow(),
        {"concurrency", "budget", "limits", "parallel"},
        {"max_concurrent_runs", "cost_budget", "group_limits", "max_parallel"},
    ),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_each_record_writes_its_new_keys_and_downgrades_to_version_one(name):
    record, gained, lost = CASES[name]
    written = record.to_dict()
    assert gained <= set(written), name
    assert not lost & set(written), name
    json.dumps(written)
    restored = type(record).KEYS.downgrade(dict(written))
    assert not gained & set(restored), name
    assert lost <= set(restored), name


def test_the_assessment_names_the_incident_path_and_route_once():
    written = assessment().to_dict()
    assert written["escalate"] is True
    assert written["alert_route"] == "on-call"
    assert written["target"]["route"] == "on-call"


def test_the_workflow_steps_are_written_with_the_step_keys():
    written = definition.standard_workflow().to_dict()
    train = next(s for s in written["steps"] if s["name"] == "train")
    assert train["needs"] == ["ingest"] and train["concurrency"] == "train"
    restored = definition.Workflow.KEYS.downgrade(written)
    assert restored["steps"][0]["depends_on"] == []


def test_the_digests_and_the_alert_id_hash_the_version_one_layout():
    assert definition.standard_workflow().digest() == WORKFLOW_DIGEST
    assert job().digest() == JOB_DIGEST
    assert alert().alert_id == ALERT_ID
    anchored = job().anchor()
    text = json.dumps(anchored, sort_keys=True)
    assert hashlib.sha256(text.encode()).hexdigest() == JOB_DIGEST
    assert {"run_id", "timeout_seconds", "project_digest"} <= set(anchored)


def test_a_version_one_job_spec_still_reads():
    spec = job()
    old = base.Job.KEYS.downgrade(spec.to_dict())

    class Store:
        def __init__(self, document):
            self.document = document

        def get(self, key):
            return (json.dumps(self.document).encode(), None)

    for document in (old, spec.to_dict()):
        rebuilt = entrypoint.respec(Store(document), "run-1")
        assert rebuilt == spec
        assert rebuilt.digest() == JOB_DIGEST


def test_the_approval_signs_its_version_one_fields_and_reads_both_layouts():
    signed = approval().sign(KEY)
    old = {**signed.preimage(), "signature": signed.signature}
    assert {"content_id", "approved_at"} <= set(signed.preimage())
    assert signed.signature == signing.mac(signed.preimage(), KEY)
    for stored in (old, signed.to_dict()):
        held = registry.Sanction.load(stored)
        assert held == signed and held.verify(KEY)
        assert release.revive(stored) == signed


def test_a_version_one_observation_still_reads():
    current = seen()
    old = observation.Observation.KEYS.downgrade(current.to_dict())
    assert old["image_digest"] == DIGEST and old["observed_at"] == NOW
    assert release.rebuild(old) == current
    assert release.rebuild(current.to_dict()) == current


def test_a_version_one_delivery_journal_still_deduplicates(tmp_path: Path):
    journal = tmp_path / "journal.jsonl"
    item = alert()
    old = {
        "alert": alerts.Alert.KEYS.downgrade(item.to_dict()),
        "delivery": {"status": "delivered"},
    }
    journal.write_text(json.dumps(old) + "\n", encoding="utf-8")
    dispatcher = alerts.Dispatcher(
        {"ops": alerts.Jsonl(tmp_path / "a")}, journal
    )
    assert dispatcher.dispatch(item).status == "deduplicated"


def test_the_model_card_is_version_two_and_downgrades_to_version_one():
    written = card().to_dict()
    assert written[records.KEY] == 2
    assert written["digests"] == {"model": "sha256:x"}
    assert (written["model"], written["version"]) == ("m", "v1")
    assert (written["use"], written["excluded"]) == ("inference", ["elsewhere"])
    stored = dict(written)
    del stored[records.KEY]
    old = records.downgrade(model_card.Card.migrations, stored)
    assert old["model_fingerprint"] == "sha256:x"
    assert {
        "model_name",
        "model_version",
        "dataset_id",
        "snapshot_id",
        "intended_use",
        "out_of_scope",
        "feature_names",
        "evidence_artifacts",
        "support_status",
        "review_status",
        "generated_at",
    } <= set(old)
