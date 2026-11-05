"""Generates the Phase 15 local evidence by running the real code.

    uv run python scripts/make_phase15_evidence.py plan/evidence/phase-15

Writes, under the output folder:

* ``object-store-conformance.json``: the one conformance suite run on the
  local, S3 and GCS backends (S3 and GCS over in-memory clients);
* ``scenario-tests.json``: the security, governed-data, training and
  orchestration tests, run by pytest, with each outcome;
* ``interrupted-job.json``, ``duplicate-run.json``, ``partial-failure.json``,
  ``missed-schedule.json``: the logs of those scenarios on the local runner
  and executor;
* ``lineage-trace.json``: a training read traced to its source snapshots;
* ``step-functions.json`` and ``cloud-workflows.json``: the emitted
  definitions with the result of their structural validators;
* ``summary.json``: what ran and what did not.

Nothing here is edited by hand. Nothing touches a cloud or a container; the
S3 and GCS clients are the in-memory fakes from ``tests/support_cloud.py``.
"""

from collections.abc import Callable
import datetime
import json
import os
from pathlib import Path
import platform
import subprocess  # nosec B404
import sys
import tempfile
import xml.etree.ElementTree as ElementTree  # nosec B405 - our own report

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from scmle.core import retry  # noqa: E402
from scmle.data.store import cloud  # noqa: E402
from scmle.data.store import conformance  # noqa: E402
from scmle.data.store import objectstore  # noqa: E402
from scmle.data.store import snapshot  # noqa: E402
from scmle.runners import base  # noqa: E402
from scmle.runners import local  # noqa: E402
from scmle.workflow import definition  # noqa: E402
from scmle.workflow import emit  # noqa: E402
from scmle.workflow import executor as executor_module  # noqa: E402
from scmle.workflow import schedule  # noqa: E402
from tests import support_cloud as fakes  # noqa: E402
from tests import support_runners as sr  # noqa: E402

TEST_FILES = (
    "tests/unit/test_security_tokens.py",
    "tests/unit/test_security_service.py",
    "tests/unit/test_object_store_conformance.py",
    "tests/unit/test_governed_data.py",
    "tests/unit/test_portable_training.py",
    "tests/integration/test_runner_project.py",
    "tests/unit/test_workflow.py",
)
SCENARIOS = {
    "security: invalid, revoked, expired credentials; expired keys": (
        "test_security_tokens",
        (
            "malformed_tokens",
            "expired_and_future",
            "revoked_tokens",
            "expired_key",
            "none_algorithm",
            "tampered_payload",
        ),
    ),
    "security: tampered artifacts": (
        "test_security_service",
        ("tampered_artifact",),
    ),
    "security: no unauthenticated backend": (
        "test_security_service",
        ("no_unauthenticated", "shipped_compose", "runtime_default_host"),
    ),
    "interrupted job": (
        "test_portable_training",
        ("interrupted_job", "spot_interruption", "preemption"),
    ),
    "duplicate invocation": (
        "test_workflow",
        ("repeated_invocation", "run_in_flight", "max_concurrent"),
    ),
    "partial failure": ("test_workflow", ("partial_failure",)),
    "missed schedule": (
        "test_workflow",
        ("missed_slots", "first_tick", "lost_its_state"),
    ),
    "cost budget stops a runaway job": (
        "test_portable_training",
        ("cost_budget", "budget_accumulates", "aws_budget", "gcp_budget"),
    ),
    "checkpoint recovery of incremental processing": (
        "test_governed_data",
        ("crash_between_output",),
    ),
}

RAN = [
    "conformance suite on a local directory and S3 and GCS client fakes",
    "interrupted job resume compared with a local reference",
    "duplicate invocation, partial failure and missed schedule, local executor",
    "source-snapshot lineage trace on a local store",
    "structural validation of the emitted Step Functions and Cloud Workflows",
    "unit and integration tests for the four work packages",
]
NOT_RUN = [
    "MinIO, the GCS emulator or any real bucket",
    "any AWS or GCP account: ECS tasks, Cloud Run jobs, Step Functions, "
    "Cloud Workflows, Scheduler, IAM",
    "Docker or any container",
    "the reproducibility-tolerance report across local, AWS and GCP",
    "TLS, workload identity, managed keys",
    "a cloud dataset ingested, validated, trained on and traced",
    "cost measured against a bill",
]


def write_json(path: Path, payload: object) -> None:
    """Write deterministic JSON."""
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def conformance_reports(out: Path) -> dict[str, bool]:
    """Run the one suite on the three backends."""
    with tempfile.TemporaryDirectory() as scratch:
        stores = {
            "local": objectstore.LocalObjectStore(Path(scratch) / "root"),
            "s3-in-memory-client": cloud.S3ObjectStore(
                fakes.FakeS3Client(), "bucket", "team/data"
            ),
            "gcs-in-memory-client": cloud.GcsObjectStore(
                fakes.FakeGcsClient(), "bucket", "team/data"
            ),
        }
        reports = {
            name: conformance.run_conformance(
                store, run_id="evidence"
            ).to_dict()
            for name, store in stores.items()
        }
    write_json(out / "object-store-conformance.json", reports)
    return {name: report["passed"] for name, report in reports.items()}


def scenario_tests(out: Path) -> dict[str, object]:
    """Run the scenario tests and record each outcome."""
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
    grouped = {
        scenario: [
            c
            for c in cases
            if module in c["id"] and any(word in c["id"] for word in words)
        ]
        for scenario, (module, words) in SCENARIOS.items()
    }
    write_json(
        out / "scenario-tests.json",
        {
            "pytest_exit_status": completed.returncode,
            "tests": len(cases),
            "failed": sum(c["outcome"] == "failed" for c in cases),
            "skipped": sum(c["outcome"] == "skipped" for c in cases),
            "scenarios": grouped,
            "all": cases,
        },
    )
    return {
        "exit_status": completed.returncode,
        "tests": len(cases),
        "scenarios_without_a_test": [s for s, t in grouped.items() if not t],
    }


def interruption_at(point: int | None) -> Callable[[int, int], bool] | None:
    """Interrupts attempt 1 right after the checkpoint of step `point`."""
    if point is None:
        return None
    return lambda attempt, step: attempt == 1 and step == point


def interrupted_job(out: Path) -> bool:
    """A job interrupted at step 3 resumes and matches the reference."""
    results = {}
    with tempfile.TemporaryDirectory() as scratch:
        for name, interrupt in (("reference", None), ("interrupted", 3)):
            clock = sr.FakeClock()
            store = objectstore.LocalObjectStore(Path(scratch) / name)
            runner = local.LocalRunner(
                store,
                sr.toy_workload(clock),
                serving_identity="serving",
                clock=base.Clock(clock.now, clock.sleep),
                poll_seconds=1.0,
                interruptions=interruption_at(interrupt),
            )
            spec = base.JobSpec(
                run_id=name,
                name="toy",
                project={"data": "d", "target": "y"},
                identity="training",
                retries=retry.RetryPolicy(3, 10.0, 2.0, 60.0),
            )
            manifest = runner.run(spec)
            results[name] = {
                "manifest": manifest.to_dict(),
                "result": json.loads(store.get(f"runs/{name}/result.json")[0]),
                "backoff_waits": [s for s in clock.sleeps if s >= 10],
            }
    same = results["reference"]["result"] == results["interrupted"]["result"]
    write_json(
        out / "interrupted-job.json",
        {**results, "interrupted_result_equals_reference": same},
    )
    return same


def make_workflow() -> definition.Workflow:
    """The standard pipeline with short timings."""
    flow = definition.standard_workflow(approvers=("ml-owner", "second"))
    import dataclasses

    steps = tuple(
        dataclasses.replace(
            s,
            retries=retry.RetryPolicy(3, 2.0, 2.0, 60.0),
            expires_seconds=7200.0,
        )
        for s in flow.steps
    )
    return definition.Workflow(flow.name, flow.version, steps, flow.schedule)


class Tick:
    """A clock for the workflow scenarios."""

    def __init__(self, now: float) -> None:
        """Starts the clock at `now` epoch seconds."""
        self.now = now
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        """The current time."""
        return self.now

    def sleep(self, seconds: float) -> None:
        """Advances the clock instead of waiting."""
        self.sleeps.append(seconds)
        self.now += seconds


def workflow_scenarios(out: Path) -> dict[str, bool]:
    """Duplicate invocation, partial failure and missed schedule logs."""
    keys = {"ml-owner": "owner-key-evidence-0001"}
    outcomes: dict[str, bool] = {}
    with tempfile.TemporaryDirectory() as scratch:
        root = Path(scratch)
        calls: list[str] = []

        def handlers(
            flow: definition.Workflow, fail: str | None = None
        ) -> dict[str, executor_module.Handler]:
            def make(name: str) -> executor_module.Handler:
                def run(
                    context: executor_module.StepContext,
                ) -> dict[str, str]:
                    calls.append(name)
                    if name == fail:
                        raise RuntimeError("injected failure")
                    return {"step": name}

                return run

            return {
                s.name: make(s.name)
                for s in flow.steps
                if s.kind is not definition.StepKind.APPROVAL
            }

        flow = make_workflow()
        clock = Tick(1_000_000.0)
        run = executor_module.LocalExecutor(
            flow,
            objectstore.LocalObjectStore(root / "dup"),
            handlers(flow),
            approver_keys=keys,
            clock=clock,
            sleep=clock.sleep,
        )
        first = run.start("request-1", "alice")
        signed = run.sign_approval(
            first["run_id"], "approve", "ml-owner", "approve", keys["ml-owner"]
        )
        run.submit_approval(signed)
        finished = run.resume(first["run_id"])
        before = list(calls)
        again = run.start("request-1", "alice")
        write_json(
            out / "duplicate-run.json",
            {
                "first_state_before_approval": first["state"],
                "state_after_approval": finished["state"],
                "run_id": first["run_id"],
                "second_invocation_run_id": again["run_id"],
                "second_invocation_state": again["state"],
                "handler_calls_before": before,
                "handler_calls_after_second_invocation": calls,
                "second_invocation_ran_nothing": calls == before,
            },
        )
        outcomes["duplicate invocation runs nothing"] = (
            calls == before and again["run_id"] == first["run_id"]
        )

        calls.clear()
        clock = Tick(1_000_000.0)
        failing = executor_module.LocalExecutor(
            flow,
            objectstore.LocalObjectStore(root / "partial"),
            handlers(flow, fail="evaluate"),
            approver_keys=keys,
            clock=clock,
            sleep=clock.sleep,
        )
        broken = failing.start("request-2", "alice")
        before = list(calls)
        failing.handlers.update(handlers(flow))
        healed = failing.resume(broken["run_id"])
        write_json(
            out / "partial-failure.json",
            {
                "failed_run_steps": broken["steps"],
                "failed_run_state": broken["state"],
                "calls_in_failed_run": before,
                "state_after_resume": healed["state"],
                "calls_in_resume": calls[len(before) :],
            },
        )
        outcomes["partial failure resumes only the rest"] = broken[
            "state"
        ] == "failed" and calls[len(before) :] == ["evaluate", "monitor"]

        log: dict[str, object] = {}
        for policy in definition.CatchUp:
            scheduled = definition.Workflow(
                f"wf-{policy.value}",
                flow.version,
                flow.steps,
                definition.Schedule(3600.0, 0.0, policy),
            )
            clock = Tick(5 * 3600.0 + 2000.0)
            store = objectstore.LocalObjectStore(root / policy.value)
            runner = executor_module.LocalExecutor(
                scheduled,
                store,
                handlers(scheduled),
                approver_keys=keys,
                clock=clock,
                sleep=clock.sleep,
            )
            scheduler = schedule.Scheduler(scheduled, store)
            scheduler.save(0.0, {})
            report = scheduler.tick(runner)
            quiet = scheduler.tick(runner)
            log[policy.value] = {
                "first_tick": report.to_dict(),
                "second_tick": quiet.to_dict(),
            }
        write_json(out / "missed-schedule.json", log)
        outcomes["missed schedule follows the declared policy"] = (
            len(log["run_all"]["first_tick"]["ran"]) == 5  # type: ignore[index]
            and len(log["latest_only"]["first_tick"]["ran"]) == 1  # type: ignore[index]
            and len(log["skip"]["first_tick"]["ran"]) == 0  # type: ignore[index]
        )
    return outcomes


def lineage(out: Path) -> bool:
    """Reads a raw and a cleaned dataset and traces the second to the first."""
    contract = snapshot.DataContract(
        "orders", "2", {"id": "int", "amount": "float"}, 86400.0, ("id",)
    )
    with tempfile.TemporaryDirectory() as scratch:
        store = objectstore.LocalObjectStore(Path(scratch) / "store")
        store.put("raw/orders.csv", b"id,amount\n1,10.5\n2,20.0\n2,20.0\n")
        store.put("clean/orders.csv", b"id,amount\n1,10.5\n2,20.0\n")
        reader = snapshot.SnapshotReader(store)
        ignored, raw = reader.read("raw/orders.csv", contract)
        ignored, clean = reader.read(
            "clean/orders.csv", contract, parents=[raw]
        )
        trace = reader.log.trace(clean.snapshot_id)
        write_json(
            out / "lineage-trace.json",
            {
                "trace": [record.to_dict() for record in trace],
                "reads": len(reader.log.records),
            },
        )
    return [r.snapshot_id for r in trace] == [
        clean.snapshot_id,
        raw.snapshot_id,
    ]


def emitted(out: Path) -> dict[str, list[str]]:
    """Emit both definitions and validate their structure."""
    flow = definition.Workflow(
        "scmle-pipeline",
        "1",
        make_workflow().steps,
        definition.Schedule(3600.0, 0.0),
    )
    aws = emit.to_step_functions(flow)
    gcp = emit.to_cloud_workflows(flow)
    problems = {
        "step_functions": emit.validate_step_functions(aws["definition"]),
        "cloud_workflows": emit.validate_cloud_workflows(gcp["definition"]),
    }
    write_json(
        out / "step-functions.json",
        {**aws, "structural_problems": problems["step_functions"]},
    )
    write_json(
        out / "cloud-workflows.json",
        {**gcp, "structural_problems": problems["cloud_workflows"]},
    )
    return problems


def commit_hash() -> str:
    """The commit the evidence was generated from."""
    done = subprocess.run(  # nosec B603 B607
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return done.stdout.strip() or "unknown"


def main(arguments: list[str]) -> int:
    """Generate every file; exit 1 if anything that should pass did not."""
    out = Path(arguments[0] if arguments else "plan/evidence/phase-15")
    out.mkdir(parents=True, exist_ok=True)
    checks: dict[str, object] = {
        "conformance_passed": conformance_reports(out),
        "interrupted_job_matches_reference": interrupted_job(out),
        "workflow_scenarios": workflow_scenarios(out),
        "lineage_traced": lineage(out),
        "emitted_structural_problems": emitted(out),
        "scenario_tests": scenario_tests(out),
    }
    ok = (
        all(checks["conformance_passed"].values())  # type: ignore[attr-defined]
        and checks["interrupted_job_matches_reference"]
        and all(checks["workflow_scenarios"].values())  # type: ignore[attr-defined]
        and checks["lineage_traced"]
        and not any(checks["emitted_structural_problems"].values())  # type: ignore[attr-defined]
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
            "checks": checks,
            "ran": RAN,
            "not_run": NOT_RUN,
        },
    )
    print(
        "PHASE 15 LOCAL EVIDENCE PASSED"
        if ok
        else "PHASE 15 LOCAL EVIDENCE FAILED"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
