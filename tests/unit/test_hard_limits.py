"""Tests for the hard time ceiling and memory cap around a trial."""

import sys
import time

import numpy as np
import pytest

from mml.core import limits

LINUX_ONLY = pytest.mark.skipif(
    not limits.HARD_MEMORY_SUPPORTED, reason="needs Linux address-space caps"
)


@pytest.mark.skipif(sys.platform == "win32", reason="needs SIGALRM")
def test_time_ceiling_interrupts_a_running_block():
    started = time.monotonic()
    with (
        pytest.raises(limits.TrialLimitError, match="hard time ceiling"),
        limits.hard_limits(0.3, None),
    ):
        while True:
            sum(range(1000))
    assert time.monotonic() - started < 5


@pytest.mark.skipif(sys.platform == "win32", reason="needs SIGALRM")
def test_a_block_that_finishes_in_time_is_untouched_and_state_restored():
    with limits.hard_limits(5.0, None):
        value = sum(range(100))
    assert value == 4950
    with limits.hard_limits(None, None):
        time.sleep(0.01)


@LINUX_ONLY
def test_memory_cap_fails_an_oversized_allocation():
    with pytest.raises(MemoryError), limits.hard_limits(None, 0.05):
        np.ones(50_000_000)
    np.ones(1_000_000)


def test_governor_ceiling_and_report_state_the_enforcement():
    from mml.contracts import tasks

    governor = limits.ResourceGovernor(
        tasks.ResourceSpec(time_budget_seconds=100)
    )
    assert 100 <= governor.trial_ceiling_seconds() <= 110
    report = governor.report()
    assert "hard ceiling" in report["time_budget"]["enforcement"]
    assert "cooperative" not in report["time_budget"]["enforcement"]
