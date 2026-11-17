"""Tests for resource limit enforcement and reporting."""

import time

import pytest

from scmle.contracts import errors
from scmle.contracts import tasks
from scmle.core import budget
from scmle.core import limits


def governor(**kwargs):
    return limits.Governor(tasks.Resources(**kwargs))


def test_cpu_limit_caps_workers_and_splits_threads():
    assert governor(workers=8, cpu_limit=2).workers == 2
    assert governor(workers=2, cpu_limit=8).workers == 2
    assert governor(workers=4, cpu_limit=8).threadcap == 2
    assert governor(workers=4).threadcap == 1
    assert governor(workers=1).threadcap is None
    assert governor(workers=1, cpu_limit=3).threadcap == 3


def test_the_trial_window_and_memory_gate_trials(monkeypatch):
    # Enforces README.md:138. Time applies to the first trial too (the run
    # was admitted only if one fits); memory spares it so a result exists.
    tight = limits.Governor(
        tasks.Resources(time_budget_seconds=1),
        started_at=time.monotonic() - 5,
    )
    assert "trial window" in tight.skip_reason(first=True)
    assert "trial window" in tight.skip_reason(first=False)
    roomy = governor(time_budget_seconds=1000)
    assert roomy.skip_reason(first=False) is None
    monkeypatch.setattr(limits, "peak_memory_gb", lambda: 99.0)
    assert "memory limit" in roomy.skip_reason(first=False)
    assert roomy.skip_reason(first=True) is None


def test_peak_memory_is_positive_and_plausible():
    assert 0.01 < limits.peak_memory_gb() < 1024


def test_report_states_enforcement_honestly(monkeypatch):
    # Enforces docs/ARCHITECTURE.md:59.
    # Enforces README.md:138.
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


def test_until_splits_the_budget_into_a_trial_window_and_a_finish():
    started = time.monotonic()
    timed = limits.Governor(
        tasks.Resources(time_budget_seconds=100), started_at=started
    )
    timed.reserve.parts = {"refit": 10.0, "diagnostics": 6.0, "package": 4.0}
    assert timed.until("finish") == started + 100
    assert timed.until("trials") == started + 80
    with pytest.raises(errors.InputError, match="phase"):
        timed.until("lunch")


def test_a_trial_ceiling_ends_at_the_trial_window_with_no_grace():
    started = time.monotonic()
    timed = limits.Governor(
        tasks.Resources(time_budget_seconds=100), started_at=started
    )
    timed.reserve.parts = {"refit": 30.0}
    assert 69 < timed.deadline() <= 70
    late = limits.Governor(
        tasks.Resources(time_budget_seconds=10), time.monotonic() - 100
    )
    assert late.deadline() == 0.0
    assert "trial window" in late.skip_reason(first=True)
    assert not hasattr(limits, "GRACE_SHARE")


def test_the_slack_is_the_poll_interval_plus_the_kill_time():
    assert limits.SLACK_SECONDS == limits.POLL_SECONDS + limits.KILL_SECONDS
    assert limits.SLACK_SECONDS < 1.0


def test_the_reserve_follows_the_measured_cost_models():
    # MIC (diagnosis 2026-11-17): 1132 rows, 111 columns, 8 classes; the
    # bootstrap took 3.6 s, the other diagnostics at most 1.7 s.
    reserve = budget.reservation(1132, 111, 8, 60)
    assert set(reserve) == {"refit", "diagnostics", "package"}
    assert 3.6 * 0.85 < budget.bootstrap(226, 111, 8) < 3.6 * 1.15
    assert reserve["diagnostics"] > 3.6 + 1.7
    assert all(value > 0 for value in reserve.values())
    larger = budget.reservation(1132, 111, 8, 600)
    assert larger["refit"] > reserve["refit"]


def test_the_preparation_estimate_grows_with_the_table():
    small = budget.preparation(500, 5, 2, 60)
    wide = budget.preparation(1132, 111, 8, 60)
    assert 0 < small < wide
    assert (
        budget.cheapest(52035) > budget.cheapest(1000) >= (budget.START_SECONDS)
    )


def test_a_run_that_cannot_fit_is_refused_with_the_smallest_budget():
    tight = budget.Reserve(5)
    tight.provision(50_000, 100, 10)
    with pytest.raises(errors.BlockedError, match="smallest budget") as caught:
        tight.admit()
    smallest = caught.value.details["smallest_budget_seconds"]
    assert smallest > 5 and caught.value.details["estimate_seconds"] > 5
    roomy = budget.Reserve(smallest)
    roomy.provision(50_000, 100, 10)
    roomy.admit()


def test_overdue_says_why_work_cannot_end_before_the_finish():
    timed = limits.Governor(tasks.Resources(time_budget_seconds=100))
    assert timed.overdue(10.0) is None
    assert "before the run's end" in timed.overdue(200.0)
    assert "before the run's end" in timed.overdue(60.0, keep=50.0)
