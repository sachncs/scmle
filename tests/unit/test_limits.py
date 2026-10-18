"""Tests for resource limit enforcement and reporting."""

import time

from scml.contracts import tasks
from scml.core import limits


def governor(**kwargs):
    return limits.ResourceGovernor(tasks.ResourceSpec(**kwargs))


def test_cpu_limit_caps_workers_and_splits_threads():
    assert governor(workers=8, cpu_limit=2).workers == 2
    assert governor(workers=2, cpu_limit=8).workers == 2
    assert governor(workers=4, cpu_limit=8).thread_limit == 2
    assert governor(workers=4).thread_limit == 1
    assert governor(workers=1).thread_limit is None
    assert governor(workers=1, cpu_limit=3).thread_limit == 3


def test_first_trial_always_runs_then_budget_and_memory_apply(monkeypatch):
    tight = limits.ResourceGovernor(
        tasks.ResourceSpec(time_budget_seconds=1),
        started_at=time.monotonic() - 5,
    )
    assert tight.skip_reason(first=True) is None
    assert "time budget" in tight.skip_reason(first=False)
    roomy = governor(time_budget_seconds=1000)
    assert roomy.skip_reason(first=False) is None
    monkeypatch.setattr(limits, "peak_memory_gb", lambda: 99.0)
    assert "memory limit" in roomy.skip_reason(first=False)
    assert roomy.skip_reason(first=True) is None


def test_peak_memory_is_positive_and_plausible():
    assert 0.01 < limits.peak_memory_gb() < 1024


def test_report_states_enforcement_honestly(monkeypatch):
    monkeypatch.setattr(limits, "peak_memory_gb", lambda: 1.5)
    report = governor(workers=4, cpu_limit=2, memory_limit_gb=1.0).report()
    assert (
        report["cpu"]["workers"] == 2
        and "capped" in report["cpu"]["enforcement"]
    )
    assert report["memory"]["within_limit"] is False
    assert "no trial starts" in report["memory"]["enforcement"]
    assert "hard ceiling" in report["time_budget"]["enforcement"]
    assert "no built-in model" in report["gpu"]["enforcement"]
    assert governor().report()["cpu"]["enforcement"] == "no cap configured"
