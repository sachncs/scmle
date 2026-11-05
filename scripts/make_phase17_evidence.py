"""Generates the Phase 17 local evidence by running the real code.

    uv run python scripts/make_phase17_evidence.py plan/evidence/phase-17/local

Writes, under the output folder:

* ``observability.json``: an injected service failure (the local service is
  killed) and a controlled drift event, each with the signals, the
  classification, the webhook deliveries (to recording transports), the
  retained telemetry digest and the operator rows in the audit ledger;
  plus the Prometheus exposition and the exporter payloads;
* ``continuous-evaluation.json``: a controlled degradation, the retraining
  proposal on the escalation path, the challenger comparison, the gated
  replacement and the post-promotion check;
* ``recovery.json``: the fault catalogue, what was run against the local
  service and emulators with measured RTO and RPO, and what was not run;
* ``production-claim.json``: the claim checker on the committed
  ``plan/evidence/phase-17/{aws,gcp}`` folders (it refuses: they hold no
  signed records);
* ``scenario-tests.json``: the Phase 17 tests, run by pytest;
* ``summary.json``: what ran and what did not.

Nothing here is edited by hand. Nothing touches a cloud, a container or a
real pager. The webhook endpoints are recording transports; the cloud
exporter payloads are checked against shapes written from the API
references; the control plane in the replacement scenario is the scripted
one in ``tests/support_runtime.py``.
"""

import dataclasses
import datetime
import json
import os
from pathlib import Path
import platform
import subprocess  # nosec B404
import sys
import tempfile
import time
import xml.etree.ElementTree as ElementTree  # nosec B405 - our own report

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from scmle.contracts import observability  # noqa: E402
from scmle.deploy import demo  # noqa: E402
from scmle.governance import deployment_ledger as dl  # noqa: E402
from scmle.inference import parity  # noqa: E402
from scmle.operations import claim  # noqa: E402
from scmle.operations import classify  # noqa: E402
from scmle.operations import continuous  # noqa: E402
from scmle.operations import cost  # noqa: E402
from scmle.operations import exporters  # noqa: E402
from scmle.operations import faults  # noqa: E402
from scmle.operations import logs  # noqa: E402
from scmle.operations import metrics  # noqa: E402
from scmle.operations import notify  # noqa: E402
from scmle.operations import respond  # noqa: E402
from scmle.operations import signals  # noqa: E402
from scmle.security import audit  # noqa: E402
from tests import support_claim  # noqa: E402
from tests import support_continuous as sc  # noqa: E402
from tests import support_ops as ops  # noqa: E402
from tests import support_release as sr  # noqa: E402
from tests import support_runtime as rt  # noqa: E402

GOLDEN = ROOT / "deploy" / "local" / "golden"
TEST_FILES = (
    "tests/unit/test_operations_telemetry.py",
    "tests/unit/test_continuous_evaluation.py",
    "tests/unit/test_fault_catalogue.py",
    "tests/unit/test_recovery_local.py",
    "tests/unit/test_production_claim.py",
)
RAN = [
    "an injected service failure (SIGKILL of the local service) and a "
    "controlled drift event, classified, notified to recording webhooks, "
    "retained and answered by operator rows in a hash-chained audit ledger",
    "exporter payloads for CloudWatch and Cloud Monitoring checked against "
    "shapes written from the API references",
    "a controlled degradation through the F13 loop, a retraining "
    "recommendation on the escalation path, a person's reply, a challenger, "
    "the loop's promotion proposal, four seeded refusals of the gate and the "
    "replacement on a scripted ECS control plane",
    "eleven recovery faults against the real local service, the local "
    "object-store and metrics emulators, a real ledger writer and a real "
    "workflow executor, with measured RTO and RPO per stateful component",
    "the production-claim checker on the committed (empty) provider folders "
    "and on seeded incomplete trees",
]
NOT_RUN = [
    "any AWS or GCP account, staging environment or provider monitoring "
    "service: no CloudWatch or Cloud Monitoring metric was written, no "
    "alert was raised by a provider",
    "real pager or ticket endpoints: webhooks went to recording transports",
    "the cloud halves of every recovery fault, including the rollback to a "
    "managed revision and a zone or region loss; none ran on a provider, so "
    "no provider RTO or RPO was measured",
    "containers, Docker and an orchestrator's own restart and health logic",
    "machine crash, power loss, disk full and network partition",
    "a controlled degradation in staging, with labels that arrive after a "
    "real delay and real business outcomes",
    "a reviewer's sign-off by anyone other than the author; the production "
    "claim has not been made and is refused by the checker",
]
FEATURES = ("age", "balance", "plan")


def write_json(path: Path, payload: object) -> None:
    """Write sorted, indented JSON with a trailing newline."""
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


class Recorder:
    """A webhook transport that records requests and answers 200."""

    def __init__(self) -> None:
        """Starts empty."""
        self.requests: list[notify.Request] = []

    def __call__(self, request: notify.Request) -> int:
        """Records the request."""
        self.requests.append(request)
        return 200


class Probe:
    """A client of the real local service that feeds every monitor."""

    def __init__(self, base: str, port_holder: dict[str, object]) -> None:
        """Binds the client to a base URL and a mutable service holder."""
        self.base = base
        self.holder = port_holder
        self.registry = metrics.MetricsRegistry()
        self.service = metrics.ServiceMetrics(
            self.registry, "golden-1", str(port_holder["digest"])
        )
        self.lines: list[str] = []
        self.log = logs.StructuredLog(
            "serve",
            "golden-1",
            str(port_holder["digest"]),
            sink=self.lines.append,
        )
        self.watch = signals.SignalMonitor(
            signals.WindowPolicy(window_seconds=120.0, min_samples=30),
            signals.SignalThresholds(reference_loss=0.05, loss_tolerance=0.05),
            ops.profile(),
        )
        self.clock = 5_000_000.0
        self.rng = np.random.default_rng(17)

    def request(self, frame: pd.DataFrame) -> list[int]:
        """Sends one /predict request; a refused connection is a 503.

        Returns:
            The predictions (empty when the service did not answer).
        """
        trace = logs.new_trace_id()
        started = time.monotonic()
        records = frame.to_dict("records")
        try:
            status, body = demo.post_json(
                f"{self.base}/predict", {"records": records}, timeout=5.0
            )
        except OSError:
            status, body = 503, {}
        seconds = time.monotonic() - started
        self.clock += 0.5
        self.service.record_request("/predict", status, seconds)
        self.watch.record_request(self.clock, "/predict", status, seconds)
        self.log.request(
            trace,
            "/predict",
            status,
            seconds,
            service_request_id=body.get("request_id"),
            records=len(records),
        )
        if status == 200:
            self.watch.observe_features(self.clock, frame)
        return [int(p) for p in body.get("predictions", [])]

    def outcomes(self, predictions: list[int], flip: float) -> None:
        """Delayed labels: the prediction, wrong with probability ``flip``."""
        wrong = self.rng.random(len(predictions)) < flip
        truth = [
            p if not w else 1 - p
            for p, w in zip(predictions, wrong, strict=True)
        ]
        self.watch.observe_outcomes(
            self.clock, signals.zero_one_losses(predictions, truth)
        )


def serving_folder() -> Path:
    """Exports the golden service into a temporary folder."""
    folder = Path(tempfile.mkdtemp(prefix="phase17-service-")) / "service"
    golden = parity.load_golden(GOLDEN)
    parity.reference_bundle(golden.spec).export_service(
        folder, signing_key=demo.DEMO_SIGNING_KEY
    )
    return folder


def start(folder: Path) -> tuple[object, str]:
    """Starts the service and waits until it is ready."""
    port = demo.free_port()
    process = demo.start_service(folder, port)
    base = f"http://127.0.0.1:{port}"
    if demo.wait_ready(base, process, 60.0) is None:
        demo.stop_service(process)
        raise RuntimeError("the local service did not become ready")
    return process, base


def desk_for(
    scratch: Path, name: str, digest: str = "unset"
) -> tuple[respond.ResponseDesk, audit.AuditLog, Recorder, Recorder]:
    """A desk whose pager and ticket are recording transports."""
    pager, ticket = Recorder(), Recorder()
    senders = notify.Notifier(
        notify.WebhookSender(
            "pager", "https://pager.invalid/hook", "pager-secret", pager
        ),
        notify.WebhookSender(
            "ticket", "https://tickets.invalid/hook", "ticket-secret", ticket
        ),
    )
    log = audit.AuditLog.open(scratch / f"{name}-audit.jsonl")
    desk = respond.ResponseDesk(
        scratch / name,
        log,
        senders,
        "golden-1",
        digest,
        clock=time.time,
    )
    return desk, log, pager, ticket


def chain_verified(log: audit.AuditLog) -> bool:
    """Whether the ledger file matches the chain in memory."""
    try:
        log.ledger.verify()
    except Exception:  # the evidence must say it failed, not stop
        return False
    return True


def describe(
    response: respond.Response, log: audit.AuditLog, hooks: tuple[Recorder, ...]
) -> dict[str, object]:
    """A JSON-ready account of one handled window."""
    return {
        **response.to_dict(),
        "audit_rows": [r["body"] for r in log.rows("admin")],
        "audit_chain_verified": chain_verified(log),
        "webhook_requests": [
            {
                "kind": json.loads(r.body)["kind"],
                "severity": json.loads(r.body)["severity"],
                "signature_header_present": notify.SIGNATURE_HEADER
                in r.headers,
                "idempotency_key": r.headers[notify.IDEMPOTENCY_HEADER],
                "trace_id": json.loads(r.body)["trace_id"],
            }
            for hook in hooks
            for r in hook.requests
        ],
    }


def observability_scenarios(out: Path) -> dict[str, object]:
    """An injected service failure and a controlled drift event."""
    folder = serving_folder()
    scratch = Path(tempfile.mkdtemp(prefix="phase17-ops-"))
    process, base = start(folder)
    digest = demo.get_json(f"{base}/health")[1]["artifact_digest"]
    probe = Probe(base, {"digest": digest})
    desk, log, pager, ticket = desk_for(scratch, "ops", digest)
    result: dict[str, object] = {}
    try:
        predictions: list[int] = []
        for step in range(8):
            predictions += probe.request(ops.frame(20, seed=100 + step))
        probe.outcomes(predictions, 0.04)
        rows = probe.watch.evaluate(probe.clock)
        baseline = desk.handle(rows)
        result["healthy_baseline"] = {
            "verdict": baseline.classification.verdict.value,
            "signals": [r.to_dict() for r in rows],
        }

        # Injected failure: kill the service process, keep sending.
        process.kill()  # type: ignore[attr-defined]
        process.wait(timeout=15)  # type: ignore[attr-defined]
        for ignored in range(40):
            probe.request(ops.frame(5, seed=7))
        rows = probe.watch.evaluate(probe.clock)
        failure = desk.handle(rows)
        incident = failure.incident_id or ""
        desk.act(incident, "oncall-1", "acknowledged", "service process gone")
        restart_started = time.monotonic()
        process, base = start(folder)
        probe.base = base
        restart_seconds = time.monotonic() - restart_started
        desk.act(
            incident,
            "oncall-1",
            "resolved",
            f"restarted the service; ready after {restart_seconds:.1f}s",
        )
        result["injected_failure"] = {
            **describe(failure, log, (pager, ticket)),
            "fault": "kill -9 of the local service process",
            "restart_to_ready_seconds": round(restart_seconds, 2),
            "signals": [r.to_dict() for r in rows],
        }
        if (
            failure.classification.verdict
            is not classify.Verdict.INFRASTRUCTURE
        ):
            raise RuntimeError("the service failure was not classified")

        # Controlled drift: valid requests, shifted feature values.
        probe.clock += 200.0
        pager.requests.clear()
        ticket.requests.clear()
        for step in range(8):
            shifted = ops.frame(20, seed=300 + step, shift=25.0)
            predictions = probe.request(shifted)
        drift_rows = probe.watch.evaluate(probe.clock)
        drift = desk.handle(drift_rows)
        later = probe.request(ops.frame(30, seed=999, shift=25.0))
        probe.outcomes(later * 2, 0.4)
        mixed_rows = probe.watch.evaluate(probe.clock)
        mixed = desk.handle(mixed_rows)
        result["controlled_drift"] = {
            "drift_only_before_labels": {
                "verdict": drift.classification.verdict.value,
                "caveats": list(drift.classification.caveats),
                "signals": [r.to_dict() for r in drift_rows],
            },
            "after_delayed_labels": {
                "verdict": mixed.classification.verdict.value,
                "categories": [
                    c.value for c in mixed.classification.categories
                ],
                "note": "drift and a worse loss together are reported as "
                "mixed; the classifier does not choose a cause",
                "signals": [r.to_dict() for r in mixed_rows],
            },
        }
        result["performance_only"] = performance_only(scratch)
        result["telemetry_retained_windows"] = len(desk.retained())
        result["structured_log_sample"] = probe.lines[:2]
        result["every_log_line_has_trace_and_version"] = all(
            {"trace_id", "model_version", "artifact_digest"}
            <= set(json.loads(x))
            for x in probe.lines
        )
        (out / "metrics.prom").write_text(probe.registry.render())
        result["request_summary"] = probe.service.summary()
        result["exporters"] = exporter_payloads(out, rows)
        result["cost"] = cost.report(
            [
                cost.Usage.from_allocation("serve", "golden-1", 1.0, 2.0, 3600),
                cost.Usage.from_allocation("serve", "golden-2", 1.0, 2.0, 600),
            ],
            cost.Rates(
                0.000024, 0.0000025, 0.4, 0.12, source="declared for the demo"
            ),
        )
    finally:
        demo.stop_service(process)  # type: ignore[arg-type]
    result["passed"] = (
        failure.classification.verdict is classify.Verdict.INFRASTRUCTURE
        and drift.classification.verdict is classify.Verdict.DATA_QUALITY
        and mixed.classification.verdict is classify.Verdict.MIXED
        and bool(result["every_log_line_has_trace_and_version"])
        and bool(result["performance_only"]["passed"])  # type: ignore[index]
    )
    write_json(out / "observability.json", result)
    return result


def performance_only(scratch: Path) -> dict[str, object]:
    """Outcomes get worse with no drift and no outage: model performance."""
    desk, log, pager, ticket = desk_for(scratch, "perf")
    watch = ops.monitor(minimum=30)
    ops.healthy_traffic(watch, count=120, at=ops.START)
    watch.observe_features(ops.START, ops.frame(400, seed=21))
    watch.observe_outcomes(ops.START, [1.0] * 40 + [0.0] * 60)
    rows = watch.evaluate(ops.START + 20)
    response = desk.handle(rows)
    return {
        **describe(response, log, (pager, ticket)),
        "passed": response.classification.verdict
        is classify.Verdict.MODEL_PERFORMANCE,
    }


def exporter_payloads(
    out: Path, rows: tuple[signals.Signal, ...]
) -> dict[str, object]:
    """The CloudWatch and Cloud Monitoring payloads and their shape checks."""
    stamp = "2026-11-05T10:00:00+00:00"
    samples = signals.as_metric_samples(
        rows, stamp, {"model_version": "golden-1"}
    )
    samples += [
        observability.MetricSample(
            "heartbeat",
            observability.SignalKind.THROUGHPUT,
            1.0,
            "count",
            stamp,
            {"model_version": "golden-1"},
        )
    ]
    put = exporters.put_metric_data("Scmle/Serving", samples)
    emf = exporters.embedded_metric_format("Scmle/Serving", samples)
    resource = {
        "type": "generic_task",
        "labels": {
            "project_id": "demo-project",
            "location": "europe-west1",
            "namespace": "scmle",
            "job": "serve",
            "task_id": "revision-1",
        },
    }
    series = exporters.time_series(resource, samples)
    report = {
        "shapes_checked_against": "tests/fixtures/ops (written from the API "
        "references, not captured from an account)",
        "put_metric_data": put,
        "put_metric_data_problems": [
            exporters.put_metric_data_problems(p) for p in put
        ],
        "embedded_metric_format": emf,
        "embedded_metric_format_problems": [
            exporters.embedded_metric_problems(r) for r in emf
        ],
        "cloud_monitoring_time_series": series,
        "cloud_monitoring_problems": [
            exporters.time_series_problems(s) for s in series
        ],
    }
    write_json(out / "exporter-payloads.json", report)
    return {
        "payload_problems": sum(
            len(p)
            for key in (
                "put_metric_data_problems",
                "embedded_metric_format_problems",
                "cloud_monitoring_problems",
            )
            for p in report[key]  # type: ignore[attr-defined]
        )
    }


def continuous_scenarios(out: Path) -> dict[str, object]:
    """A controlled degradation, the proposal, the gate and the check."""
    scratch = Path(tempfile.mkdtemp(prefix="phase17-continuous-"))
    loop, ev, stream, called = sc.degraded(scratch / "recommend")
    result: dict[str, object] = {
        "stream": "simulated: champion error 5% for 15 batches, then 40%; "
        "100 cases per batch; labels arrive with the batch (no delay)",
        "alarm_gates_retrain": loop.policy.alarm_gates_retrain,
        "l1_alarm_rows": len(sc.kinds(loop, dl.ALARM)),
        "retrain_rows_after_alarm": len(sc.kinds(loop, dl.RETRAIN)),
        "pending_after_alarm": [
            e.question["action"] for e in loop.pending_escalations()
        ],
        "hook_calls_after_alarm": len(called),
    }
    decision = ev.review(sc.loss_signals(0.4))
    result["decision"] = decision.to_dict()
    target = next(
        e for e in loop.pending_escalations() if e.id == decision.escalation_id
    )
    result["escalation"] = {
        "question": dict(target.question),
        "proposed_default": target.proposed_default,
    }
    result["hook_calls_before_reply"] = len(called)
    loop.reply(target.id, sc.reply_to(target))
    result["challenger"] = ev.run_approved_retrain(target.id)
    stream.run(60, 0.40, challengers={"challenger-1": 0.05})
    result["promotion_proposals"] = ev.promotion_proposals()
    ev.record_training(
        "challenger-1",
        ops.frame(300, seed=5),
        {"error": 0.05},
        "recommended retrain approved by a person",
    )
    release, plane, store = sc.release_world(scratch / "release")
    release.promote(sc.MODEL, 1, plane, "staging", rt.OLD, signing_key=sr.KEY)
    good = sc.candidate_for(store)
    refusals = {}
    for name, bad in {
        "model unsigned": dataclasses.replace(
            good, model=dataclasses.replace(good.model, signed=False)
        ),
        "image scan failed": dataclasses.replace(
            good, scan=dataclasses.replace(good.scan, passed=False)
        ),
        "no approval": dataclasses.replace(good, approval=None),
        "approver is the requester": dataclasses.replace(
            good, approver=good.requested_by
        ),
    }.items():
        refused = ev.replace("challenger-1", release, plane, bad, sr.KEY)
        refusals[name] = {
            "refused": not refused.done,
            "reasons": list(refused.reasons),
            "staging_still_runs_version": release.environment(
                sc.MODEL, "staging"
            )["version"],
        }
    none = ev.replace("unproposed", release, plane, good, sr.KEY)
    refusals["no promotion proposal"] = {
        "refused": not none.done,
        "reasons": list(none.reasons),
    }
    done = ev.replace("challenger-1", release, plane, good, sr.KEY)
    result["refusals"] = refusals
    result["replacement"] = done.to_dict()
    result["reconcile_consistent"] = release.reconcile(
        sc.MODEL, "staging", plane.observe()
    ).consistent
    rng = np.random.default_rng(2)
    before = {
        "retained": rng.binomial(1, 0.60, 400).astype(float),
        "latency_ms": rng.normal(120, 20, 400),
    }
    after = {
        "retained": rng.binomial(1, 0.68, 400).astype(float),
        "latency_ms": rng.normal(121, 20, 400),
    }
    result["post_promotion_check"] = ev.check_after_promotion(
        [
            continuous.MetricSpec(
                "retained", "business", "higher", margin=0.0, primary=True
            ),
            continuous.MetricSpec(
                "latency_ms", "operational", "lower", margin=5.0
            ),
        ],
        before,
        after,
    )
    result["post_promotion_data"] = "synthetic periods declared in the script"
    trigger_loop, trigger_ev, ignored, trigger_calls = sc.degraded(
        scratch / "trigger", policy=continuous.RetrainPolicy(mode="trigger")
    )
    triggered = trigger_ev.review(sc.loss_signals(0.4))
    result["trigger_mode"] = {
        "decision": triggered.to_dict(),
        "hook_calls": len(trigger_calls),
        "escalation_still_opened": bool(triggered.escalation_id),
        "alarm_gates_retrain": trigger_loop.policy.alarm_gates_retrain,
    }
    loop.ledger.verify()
    result["ledger_verified"] = True
    result["history"] = [
        {"seq": r["seq"], "kind": r["kind"], "hash": r["hash"]}
        for r in ev.history.rows()
    ]
    result["passed"] = (
        result["alarm_gates_retrain"] is False
        and result["retrain_rows_after_alarm"] == 0
        and result["hook_calls_before_reply"] == 0
        and decision.action == "recommend"
        and result["promotion_proposals"] == ["challenger-1"]
        and all(r["refused"] for r in refusals.values())
        and done.done
        and bool(result["reconcile_consistent"])
        and trigger_calls == [1]
        and triggered.action == "trigger"
    )
    write_json(out / "continuous-evaluation.json", result)
    return result


def recovery_scenarios(out: Path) -> dict[str, object]:
    """Runs every local fault and writes the report."""
    report = faults.run_all()
    write_json(out / "recovery.json", report)
    return report


def production_claim(out: Path) -> dict[str, object]:
    """The checker on the committed folders and on seeded incomplete trees."""
    committed = claim.check_claim(
        ROOT / "plan" / "evidence" / "phase-17",
        "pipeline-key-for-this-report",
        "reviewer-key-for-this-report",
    )
    scratch = Path(tempfile.mkdtemp(prefix="phase17-claim-"))
    tree = support_claim.build(scratch / "synthetic")
    seeded: dict[str, bool] = {}
    keys = (support_claim.PIPELINE_KEY, support_claim.REVIEWER_KEY)
    for provider in claim.PROVIDERS:
        for kind in (*claim.KINDS, claim.SIGNOFF):
            path = tree / provider / f"{kind}.json"
            kept = path.read_bytes()
            path.unlink()
            seeded[f"{provider}: {kind}.json missing"] = not claim.check_claim(
                tree, *keys
            ).allowed
            path.write_bytes(kept)
    self_test = claim.check_claim(tree, *keys).allowed
    result = {
        "committed_folders": committed.to_dict(),
        "committed_folders_allow_the_claim": committed.allowed,
        "checker_self_test": {
            "note": "synthetic records signed with throwaway keys by "
            "tests/support_claim.py; this is a test of the checker, not "
            "evidence about any provider",
            "synthetic_complete_tree_accepted": self_test,
            "every_seeded_missing_record_refused": all(seeded.values()),
            "seeded": seeded,
        },
        "claim_text_matches_releasing_guide": not claim.claim_text_problems(
            ROOT / "docs" / "RELEASING.md"
        ),
        "claim_made": False,
    }
    write_json(out / "production-claim.json", result)
    return result


def scenario_tests(out: Path) -> dict[str, object]:
    """Run the Phase 17 tests and record each outcome."""
    with tempfile.TemporaryDirectory() as scratch:
        report = Path(scratch) / "junit.xml"
        completed = subprocess.run(  # nosec B603 - our own interpreter
            [
                sys.executable,
                "-m",
                "pytest",
                *TEST_FILES,
                "-q",
                f"--junitxml={report}",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        cases = []
        for case in ElementTree.parse(report).getroot().iter("testcase"):  # nosec B314
            failed = case.find("failure") is not None or (
                case.find("error") is not None
            )
            skipped = case.find("skipped") is not None
            cases.append(
                {
                    "id": f"{case.get('classname')}::{case.get('name')}",
                    "outcome": "failed"
                    if failed
                    else "skipped"
                    if skipped
                    else "passed",
                }
            )
    write_json(
        out / "scenario-tests.json",
        {
            "pytest_exit_status": completed.returncode,
            "tests": len(cases),
            "failed": sum(c["outcome"] == "failed" for c in cases),
            "skipped": sum(c["outcome"] == "skipped" for c in cases),
            "all": cases,
        },
    )
    return {"exit_status": completed.returncode, "tests": len(cases)}


def commit_hash() -> str:
    """The commit the evidence was generated at."""
    done = subprocess.run(  # nosec B603 B607
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return done.stdout.strip() or "unknown"


def main(arguments: list[str]) -> int:
    """Generate every file; exit 1 if anything that should pass did not."""
    out = Path(arguments[0] if arguments else "plan/evidence/phase-17/local")
    out.mkdir(parents=True, exist_ok=True)
    checks: dict[str, object] = {}
    checks["observability"] = observability_scenarios(out)["passed"]
    checks["continuous_evaluation"] = continuous_scenarios(out)["passed"]
    recovered = recovery_scenarios(out)
    checks["recovery"] = recovered["summary"]
    claimed = production_claim(out)
    self_test = claimed["checker_self_test"]
    checks["production_claim"] = {
        "committed_folders_allow_the_claim": claimed[
            "committed_folders_allow_the_claim"
        ],
        "claim_made": claimed["claim_made"],
    }
    checks["scenario_tests"] = scenario_tests(out)
    ok = (
        bool(checks["observability"])
        and bool(checks["continuous_evaluation"])
        and bool(recovered["all_local_passed"])
        and not claimed["committed_folders_allow_the_claim"]
        and bool(self_test["every_seeded_missing_record_refused"])  # type: ignore[index]
        and bool(claimed["claim_text_matches_releasing_guide"])
        and checks["scenario_tests"]["exit_status"] == 0  # type: ignore[index]
    )
    write_json(
        out / "summary.json",
        {
            "generated": datetime.datetime.now(datetime.UTC).isoformat(
                timespec="seconds"
            ),
            "commit": commit_hash(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "local_checks_passed": ok,
            "production_claim_made": False,
            "checks": checks,
            "ran": RAN,
            "not_run": NOT_RUN,
        },
    )
    print(
        "PHASE 17 LOCAL EVIDENCE PASSED"
        if ok
        else "PHASE 17 LOCAL EVIDENCE FAILED"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
