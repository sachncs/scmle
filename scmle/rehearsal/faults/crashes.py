"""Scenarios that kill a process: the service, a job and an audit writer."""

import http
import json
import pathlib
import time
from typing import Any

import numpy

from scmle.contracts import errors
from scmle.governance import deployment_ledger
from scmle.operations import recovery
from scmle.rehearsal.faults import harness
from scmle.runtime import limits
from scmle.security import audit
from scmle.storage import objectstore

STATE_WRITER = """
import json, sys
from scmle import durability
n = 0
while True:
    body = json.dumps({"counter": n}).encode()
    durability.write_atomically(sys.argv[1], body)
    print(n, flush=True)
    n += 1
"""
LEDGER_WRITER = """
import sys
from scmle.security import audit
log = audit.Log.open(sys.argv[1])
n = 0
while True:
    log.admin("writer", "tick", n=n)
    print(n, flush=True)
    n += 1
"""


def stress(fleet: harness.Fleet, base: pathlib.Path) -> dict[str, Any]:
    """Kills an atomic state writer ten times and reads its file back.

    Each trial kills the writer after a number of acknowledged writes drawn
    from 5 to 60 with a fixed seed, so the points are random but the same on
    every run.

    Args:
        fleet: The fleet that owns the writer processes.
        base: Directory for the writers' state files.

    Returns:
        A dict with ``trials`` (always 10), ``acknowledged_writes_lost`` (the
        sum over trials of how far the stored counter is below the last
        acknowledged one), ``unreadable_files`` (trials whose file could not
        be read back) and ``read_seconds_max`` (the longest read time).
    """
    rng = numpy.random.default_rng(7)
    lost = 0
    torn = 0
    seconds = []
    for trial in range(10):
        path = base / f"state-{trial}.json"
        acknowledged = harness.interrupt(
            fleet, STATE_WRITER, path, int(rng.integers(5, 61))
        )
        started = time.monotonic()
        try:
            found = json.loads(path.read_text(encoding="utf-8"))["counter"]
        except (ValueError, KeyError, OSError):
            # handled: counted as torn trial
            torn += 1
            found = -1
        seconds.append(time.monotonic() - started)
        lost += max(0, acknowledged - found)
    return {
        "trials": 10,
        "acknowledged_writes_lost": lost,
        "unreadable_files": torn,
        "read_seconds_max": max(seconds),
    }


def process_kill() -> recovery.Outcome:
    """Kills the service process three times, restarting it each time.

    After the three kills it also runs `stress` on atomic state
    files.

    Returns:
        The recovery result of this fault: its checks, its RTO and RPO
        measurements and the details observed.

    Raises:
        ScmleError: If the service did not become ready.
    """
    with harness.stage("kill", "process_kill", "local subprocess") as (
        fleet,
        base,
        result,
    ):
        service = harness.export_service(base / "service", harness.specify())
        times = []
        same = []
        remaining = 3
        while remaining > 0:
            remaining -= 1
            first = fleet.start(service)
            if harness.boot(first) is None:
                raise errors.ScmleError("the service did not become ready")
            before = harness.predict(first.base)[1].get("predictions")
            killed = time.monotonic()
            first.process.kill()
            first.process.wait(timeout=15)
            gone = harness.predict(first.base)[0] == 0
            second = fleet.start(service)
            after = None
            while time.monotonic() - killed < recovery.scaled(
                harness.READY_SECONDS
            ):
                status, body = harness.predict(second.base)
                if status == http.HTTPStatus.OK:
                    after = body.get("predictions")
                    break
                time.sleep(0.1)
            times.append(time.monotonic() - killed)
            same.append(bool(gone) and after is not None and after == before)
            fleet.close()
        state = stress(fleet, base)
        result.checks = {
            "killed process stopped answering and the restart answers the "
            "same predictions": all(same),
            "no state file was ever unreadable": state["unreadable_files"] == 0,
        }
        result.measurements = [
            recovery.Measurement(
                "service",
                max(times),
                0 if all(same) else sum(not s for s in same),
                "SIGKILL to the first 200 from /predict of the restarted "
                "process, 3 trials, maximum; RPO is answers that differ from "
                "before the kill. " + harness.METHOD_CLOCK,
            ),
            recovery.Measurement(
                "registry_state",
                state["read_seconds_max"],
                state["acknowledged_writes_lost"],
                "10 trials: a child rewrites a state file atomically in a "
                "loop and prints each acknowledged counter; SIGKILL at a "
                "random point; the file is read back and its counter must "
                "not be below the last acknowledged one",
            ),
        ]
        result.detail = {
            "service_rto_seconds_per_trial": [round(t, 3) for t in times],
            "state_trials": state,
        }
    return result


# ---- jobs and ledgers --------------------------------------------------

# POSIX reports death by signal N as -N; Windows has no signals, so
# `os.kill(pid, N)` terminates the process with exit code N.
KILLED_RETURN_CODE = (
    int(limits.KILL_SIGNAL) if limits.WINDOWS else -int(limits.KILL_SIGNAL)
)

JOB_CHILD = """
import json, os, sys, time
from scmle.runtime import limits
from scmle.storage import objectstore
from scmle.workflow import definition
from scmle.workflow import executor

root, marker, kill = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
flow = definition.Workflow(
    "recovery-job",
    "1",
    (
        definition.Step("ingest", definition.StepKind.INGEST),
        definition.Step("train", definition.StepKind.TRAIN, ("ingest",)),
        definition.Step("evaluate", definition.StepKind.EVALUATE, ("train",)),
    ),
)

def handler(name):
    def run(context):
        with open(marker, "a") as handle:
            handle.write(name + "\\n")
            handle.flush()
            os.fsync(handle.fileno())
        if name == "train" and kill:
            os.kill(os.getpid(), limits.KILL_SIGNAL)
            time.sleep(60)  # delivery is asynchronous; do not return
        return {"step": name}
    return run

store = objectstore.LocalObjectStore(root)
runner = executor.Executor(
    flow,
    store,
    {n: handler(n) for n in ("ingest", "train", "evaluate")},
    lease_seconds=1.0,
)
record = runner.start("trigger-1")
print(json.dumps({"state": record["state"]}), flush=True)
"""


def job_interruption() -> recovery.Outcome:
    """A three-step workflow run kills itself in step two; a rerun resumes it.

    A second process started after the one-second lease runs the same trigger
    from the stored state.

    Returns:
        The recovery result of this fault: its checks, its RTO and RPO
        measurements and the details observed.
    """
    with harness.stage("job", "job_interruption", "local subprocess") as (
        fleet,
        base,
        result,
    ):
        marker = base / "marker.txt"
        state = base / "state"
        first = fleet.child(JOB_CHILD, str(state), str(marker), "1")
        first.wait(timeout=60)
        killed = first.returncode == KILLED_RETURN_CODE
        after_kill = marker.read_text(encoding="utf-8").split()
        started = time.monotonic()
        time.sleep(1.5)  # the lease is 1 s; a live run would hold it
        second = fleet.child(JOB_CHILD, str(state), str(marker), "0")
        output = (second.stdout.read() if second.stdout else "").strip()
        second.wait(timeout=60)
        elapsed = time.monotonic() - started
        final = marker.read_text(encoding="utf-8").split()
        finished = '"succeeded"' in output
        persisted = objectstore.LocalObjectStore(state)
        keys = persisted.list_objects("workflows/recovery-job/")
        repeated_finished = final.count("ingest") - 1
        result.checks = {
            "the first run was killed during train": killed
            and after_kill == ["ingest", "train"],
            "the continued run finished": finished,
            "the finished step was not repeated": final.count("ingest") == 1,
            "the interrupted step ran again once": final.count("train")
            == harness.INTERRUPTED_STEP_RUNS,
            "the remaining step ran once": final.count("evaluate") == 1,
        }
        result.measurements = [
            recovery.Measurement(
                "job_state",
                elapsed,
                max(0, repeated_finished),
                "a child runs a 3-step workflow and SIGKILLs itself in step "
                "2; after the 1 s lease a second process starts the same "
                "trigger; RTO includes the lease wait; RPO counts completed "
                "steps run again or lost",
            )
        ]
        result.detail = {
            "handler_calls": final,
            "state_objects": len(list(keys)),
            "lease_seconds": 1.0,
        }
    return result


def resilience() -> recovery.Outcome:
    """Kills an audit-log writer five times and reopens each log.

    Each reopened log must verify, hold every acknowledged row, continue after
    three more rows and verify again.

    Returns:
        The recovery result of this fault: its checks, its RTO and RPO
        measurements and the details observed.
    """
    with harness.stage(
        "ledger", "audit_log_continuity", "local subprocess"
    ) as (fleet, base, result):
        rng = numpy.random.default_rng(11)
        lost = 0
        broken = 0
        reload_seconds = []
        contiguous = True
        for trial in range(5):
            path = base / f"audit-{trial}.jsonl"
            acknowledged = harness.interrupt(
                fleet, LEDGER_WRITER, path, int(rng.integers(3, 61))
            )
            started = time.monotonic()
            try:
                reopened = audit.Log.open(path)
            except deployment_ledger.MisuseError:
                # handled: counted as broken trial
                broken += 1
                continue
            reload_seconds.append(time.monotonic() - started)
            ticks = [r["body"]["n"] for r in reopened.rows("admin")]
            lost += max(0, acknowledged + 1 - len(ticks))
            for extra in range(3):
                reopened.admin("operator", "after_restart", n=extra)
            again = audit.Log.open(path)
            again.ledger.verify()
            contiguous &= ticks == list(range(len(ticks)))
            contiguous &= [r["seq"] for r in again.ledger.rows] == list(
                range(len(again.ledger.rows))
            )
        result.checks = {
            "every killed ledger reopened with a verified chain": broken == 0,
            "recovered rows are contiguous and the chain continues": contiguous,
        }
        result.measurements = [
            recovery.Measurement(
                "ledger",
                max(reload_seconds) if reload_seconds else None,
                lost,
                "5 trials: a child appends audit rows and prints each "
                "acknowledged one; SIGKILL at a random point; the ledger is "
                "reopened (chain verified) and must hold at least every "
                "acknowledged row; then it is extended and verified again. "
                "Machine crash and power loss are not covered.",
            )
        ]
        result.detail = {"trials": 5, "torn_or_broken": broken}
    return result
