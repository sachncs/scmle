"""Tests for hard time and memory limits enforced by process supervision."""

import time

import numpy as np
import pytest

from mml.contracts import tasks
from mml.core import limits


def double(value):
    return value * 2


def spin(seconds):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        sum(range(1000))
    return "finished"


def hog(megabytes):
    block = np.ones(megabytes * 131072)  # touches every page
    time.sleep(5)
    return block.size


def explode(value):
    raise ValueError("boom")


def test_a_trial_within_limits_returns_its_value():
    supervisor = limits.TrialSupervisor(30.0, 4.0)
    assert supervisor.run(double, 21) == 42


def test_a_runaway_trial_is_killed_at_the_ceiling():
    started = time.monotonic()
    supervisor = limits.TrialSupervisor(1.0, None)
    with pytest.raises(limits.TrialLimitError, match="hard time ceiling"):
        supervisor.run(spin, 60)
    assert time.monotonic() - started < 15


def test_a_memory_hog_is_killed_at_the_limit():
    supervisor = limits.TrialSupervisor(60.0, 0.4)
    with pytest.raises(limits.TrialLimitError, match="memory limit"):
        supervisor.run(hog, 800)


def test_an_exception_in_the_trial_is_reported_not_lost():
    with pytest.raises(limits.TrialLimitError, match="ValueError: boom"):
        limits.TrialSupervisor(30.0, None).run(explode, 1)


def test_governor_modes_describe_their_enforcement():
    hard = limits.ResourceGovernor(tasks.ResourceSpec(time_budget_seconds=100))
    assert 100 <= hard.trial_ceiling_seconds() <= 110 and hard.hard
    assert "killed" in hard.report()["time_budget"]["enforcement"]
    assert "killed" in hard.report()["memory"]["enforcement"]
    soft = limits.ResourceGovernor(
        tasks.ResourceSpec(enforcement="cooperative")
    )
    assert not soft.hard
    assert "cooperative" in soft.report()["memory"]["enforcement"]
