"""The scoring workers: daemon threads, and capacity left by hung calls."""

import concurrent.futures
import subprocess
import sys
import threading
import time

import pytest

from scmle.contracts import errors
from scmle.inference import guard
from scmle.serving.server import workers


def test_a_crew_runs_calls_and_reports_their_errors():
    crew = workers.Crew(2)
    try:
        assert crew.submit(lambda: 7).result(5) == 7
        failed = crew.submit(lambda: 1 / 0)
        with pytest.raises(ZeroDivisionError):
            failed.result(5)
        assert crew.submit(sum, [1, 2], start=3).result(5) == 6
    finally:
        crew.shutdown(wait=False, cancel_futures=True)


def test_a_crew_starts_no_more_threads_than_its_size():
    crew = workers.Crew(2)
    gate = threading.Event()
    try:
        held = [crew.submit(gate.wait, 20) for ignored in range(3)]
        assert len(crew.threads) == 2
        assert not held[2].running()
        gate.set()
        assert [future.result(5) for future in held] == [True, True, True]
    finally:
        gate.set()
        crew.shutdown(wait=False, cancel_futures=True)


def test_a_closed_crew_refuses_work_and_cancels_what_waits():
    crew = workers.Crew(1)
    gate = threading.Event()
    running = crew.submit(gate.wait, 20)
    waiting = crew.submit(lambda: 1)
    crew.shutdown(wait=False, cancel_futures=True)
    assert waiting.cancelled()
    with pytest.raises(errors.OrderError, match="shut down"):
        crew.submit(lambda: 1)
    gate.set()
    assert running.result(5) is True


def test_a_hung_call_does_not_hold_the_process_at_exit():
    # A ThreadPoolExecutor joins its workers at exit, so one call that
    # never returns kept a drained server process from ever exiting.
    code = (
        "import time\n"
        "from scmle.serving.server import workers\n"
        "crew = workers.Crew(1)\n"
        "crew.submit(time.sleep, 600)\n"
        "time.sleep(0.2)\n"
        "crew.shutdown(wait=False, cancel_futures=True)\n"
    )
    started = time.monotonic()
    done = subprocess.run([sys.executable, "-c", code], timeout=60, check=False)
    assert done.returncode == 0
    assert time.monotonic() - started < 30


def test_shutdown_with_wait_joins_the_threads():
    crew = workers.Crew(2)
    assert crew.submit(lambda: 1).result(5) == 1
    crew.shutdown(wait=True)
    assert not any(thread.is_alive() for thread in crew.threads)


def stall(admission, gates):
    """Times out one call on `admission`, which stays stuck on a gate."""
    gate = threading.Event()
    gates.append(gate)
    with pytest.raises(guard.DeadlineError):
        admission.run(lambda: gate.wait(30))


def test_a_stuck_worker_leaves_a_parallel_replica_ready_and_alive():
    admission = guard.Admission(2, 0.05)
    admission.pool = workers.Crew(2)
    capacity = workers.Capacity(admission, lambda: False, 0.2)
    readiness = guard.Readiness()
    for step in guard.STEPS:
        readiness.begin(step)
        readiness.complete(step)
    gates = []
    try:
        stall(admission, gates)
        found = capacity.assess(readiness)
        assert found["ready"] is True and found["reason"] is None
        assert found["workers"] == {"busy": 1, "stuck": 1, "misses": 1}
        time.sleep(0.3)
        assert capacity.alive() is True
        assert admission.run(lambda: 5) == 5
    finally:
        for gate in gates:
            gate.set()
        admission.close()


def test_every_worker_stuck_is_unready_then_wedged_after_the_patience():
    admission = guard.Admission(2, 0.05, strikes=5)
    admission.pool = workers.Crew(2)
    capacity = workers.Capacity(admission, lambda: False, 0.3)
    readiness = guard.Readiness()
    for step in guard.STEPS:
        readiness.begin(step)
        readiness.complete(step)
    gates = []
    try:
        stall(admission, gates)
        stall(admission, gates)
        found = capacity.assess(readiness)
        assert found["ready"] is False
        assert found["reason"] == "2 timed-out call(s) hold every worker"
        assert capacity.alive() is True
        time.sleep(0.4)
        assert capacity.alive() is False
        gates[0].set()
        deadline = time.monotonic() + 10
        while admission.state()["stuck"] == 2 and time.monotonic() < deadline:
            time.sleep(0.02)
        assert capacity.alive() is True
        assert capacity.assess(readiness)["ready"] is True
    finally:
        for gate in gates:
            gate.set()
        admission.close()


def test_a_stuck_call_blocks_a_serialized_model():
    admission = guard.Admission(4, 0.05, strikes=5)
    admission.pool = workers.Crew(4)
    capacity = workers.Capacity(admission, lambda: True, 0.2)
    readiness = guard.Readiness()
    for step in guard.STEPS:
        readiness.begin(step)
        readiness.complete(step)
    gates = []
    try:
        stall(admission, gates)
        found = capacity.assess(readiness)
        assert found["ready"] is False
        assert found["reason"] == (
            "1 timed-out call(s) hold the lock that serializes scoring"
        )
        time.sleep(0.3)
        assert capacity.alive() is False
    finally:
        for gate in gates:
            gate.set()
        admission.close()


def test_the_default_patience_is_the_strikes_times_the_timeout():
    admission = guard.Admission(2, 4.0)
    try:
        capacity = workers.Capacity(admission, lambda: False)
        assert capacity.patience == guard.STRIKES * 4.0
        with pytest.raises(errors.ConfigError, match="positive"):
            workers.Capacity(admission, lambda: False, 0.0)
    finally:
        admission.close()


def test_a_crew_is_an_executor():
    assert issubclass(workers.Crew, concurrent.futures.Executor)


def test_enlist_puts_the_admission_on_a_crew():
    capacity = workers.enlist(3, 2.0, lambda: False)
    try:
        assert isinstance(capacity.admission.pool, workers.Crew)
        assert capacity.admission.max_concurrent == 3
        assert capacity.admission.pool.size == 3
        assert capacity.patience == guard.STRIKES * 2.0
        assert capacity.admission.run(lambda: 4) == 4
    finally:
        capacity.admission.close()
