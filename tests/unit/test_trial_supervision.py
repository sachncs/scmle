"""Tests for hard time and memory limits enforced by process supervision."""

import os
import sys
import time

import numpy
import pytest

from scmle.contracts import tasks
from scmle.runtime import budget
from scmle.runtime import limits


def double(value):
    return value * 2


def spin(seconds):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        sum(range(1000))
    return "finished"


def hog(megabytes):
    block = numpy.ones(megabytes * 131072)  # touches every page
    time.sleep(5)
    return block.size


def explode(value):
    raise ValueError("boom")


def test_a_trial_within_limits_returns_its_value():
    supervisor = limits.Supervisor(30.0, 4.0)
    assert supervisor.run(double, 21) == 42


def test_a_runaway_trial_is_killed_at_the_ceiling():
    # Enforces docs/WHEN_THINGS_GO_WRONG.md:222.
    # Enforces website/src/learn/when-things-go-wrong.html:112.
    started = time.monotonic()
    supervisor = limits.Supervisor(1.0, None)
    with pytest.raises(limits.TrialError, match="hard time ceiling"):
        supervisor.run(spin, 60)
    assert time.monotonic() - started < 15


def test_a_memory_hog_is_killed_at_the_limit():
    supervisor = limits.Supervisor(60.0, 0.4)
    with pytest.raises(limits.TrialError, match="memory limit"):
        supervisor.run(hog, 800)


def test_an_exception_in_the_trial_is_reported_not_lost():
    with pytest.raises(limits.TrialError, match="ValueError: boom"):
        limits.Supervisor(30.0, None).run(explode, 1)


def test_no_trial_gets_more_than_the_trial_window():
    # The first-trial whole-budget rule let a run take twice its budget;
    # every trial now ends when the trial window closes, with no grace.
    late = limits.Governor(
        tasks.Resources(time_budget_seconds=10), time.monotonic() - 100
    )
    assert late.deadline() == 0.0
    assert late.skip_reason(first=True) is not None


def test_a_trial_is_killed_within_the_slack_of_its_ceiling():
    started = time.monotonic()
    with pytest.raises(limits.TrialError, match="hard time ceiling"):
        limits.Supervisor(1.0, None).run(spin, 30)
    elapsed = time.monotonic() - started
    assert elapsed <= 1.0 + budget.START_SECONDS + limits.SLACK_SECONDS


def test_the_cpu_limit_caps_math_threads_in_the_main_process():
    # Profiling fits run in the main process; without the cap they used every
    # core whatever `cpu_limit` said (C7 defect).
    import sklearn.ensemble  # noqa: F401  # loads the OpenMP runtime
    import threadpoolctl

    before = {
        p["filepath"]: p["num_threads"] for p in threadpoolctl.threadpool_info()
    }
    with limits.Governor(tasks.Resources(cpu_limit=1)).cap():
        pools = threadpoolctl.threadpool_info()
        assert pools and all(pool["num_threads"] == 1 for pool in pools)
    with limits.Governor(tasks.Resources()).cap():
        after = {
            p["filepath"]: p["num_threads"]
            for p in threadpoolctl.threadpool_info()
        }
        assert after == before


def test_governor_modes_describe_their_enforcement():
    # Enforces docs/ARCHITECTURE.md:90.
    hard = limits.Governor(tasks.Resources(time_budget_seconds=100))
    assert 99 <= hard.deadline() <= 100 and hard.hard
    assert "killed" in hard.report()["time_budget"]["enforcement"]
    assert "killed" in hard.report()["memory"]["enforcement"]
    soft = limits.Governor(tasks.Resources(enforcement="cooperative"))
    assert not soft.hard
    assert "cooperative" in soft.report()["memory"]["enforcement"]


def read_thread_setting(value):
    return os.environ.get(limits.ACCELERATE_THREADS)


def large_products(value):
    import numpy

    matrix = numpy.random.default_rng(0).normal(size=(3000, 3000))
    vector = numpy.ones(3000)
    return float(
        sum((matrix @ vector).sum() for repeat in range(3) if repeat >= 0)
    )


@pytest.mark.skipif(sys.platform != "darwin", reason="Accelerate is macOS only")
def test_trial_processes_run_with_single_threaded_accelerate_and_survive():
    supervisor = limits.Supervisor(120.0, None)

    assert supervisor.run(read_thread_setting, 1) == "1"
    assert isinstance(supervisor.run(large_products, 1), float)


def test_the_environment_is_restored_after_starting_a_trial():
    before = os.environ.get(limits.ACCELERATE_THREADS)

    limits.Supervisor(30.0, None).run(double, 2)

    assert os.environ.get(limits.ACCELERATE_THREADS) == before
