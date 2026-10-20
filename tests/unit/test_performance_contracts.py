"""Tests for explicit performance measurements and release budgets."""

import pytest

from scml.contracts import performance

OBSERVED = "2026-10-21T00:00:00+00:00"


def observation(
    operation: performance.PerformanceOperation,
    elapsed: float = 1.0,
    memory: float | None = 1.0,
) -> performance.PerformanceObservation:
    """Build one deterministic performance observation."""
    return performance.PerformanceObservation(
        operation=operation,
        elapsed_seconds=elapsed,
        peak_memory_gb=memory,
        sample_count=10,
        machine="test-machine",
        observed_at=OBSERVED,
    )


def test_budget_requires_every_declared_operation_and_passes_complete_input():
    """A green budget requires measurements for every requested operation."""
    budget = performance.PerformanceBudget(
        {
            performance.PerformanceOperation.INGEST: 2.0,
            performance.PerformanceOperation.INFERENCE: 2.0,
        },
        max_peak_memory_gb=2.0,
    )

    result = budget.evaluate(
        (
            observation(performance.PerformanceOperation.INGEST),
            observation(performance.PerformanceOperation.INFERENCE),
        )
    )

    assert result.passed
    assert result.violations == ()
    assert len(result.to_dict()["observations"]) == 2


def test_budget_fails_closed_for_missing_and_exceeded_observations():
    """Missing evidence and observed regressions are both release failures."""
    budget = performance.PerformanceBudget(
        {
            "ingest": 1.0,
            performance.PerformanceOperation.FIT: 2.0,
        },
        max_peak_memory_gb=1.0,
    )

    result = budget.evaluate(
        (observation(performance.PerformanceOperation.INGEST, 1.1, 1.2),)
    )

    assert not result.passed
    assert any("ingest took" in item for item in result.violations)
    assert any("ingest used" in item for item in result.violations)
    assert any(
        "missing performance observation: fit" in item
        for item in result.violations
    )


def test_memory_budget_fails_closed_when_memory_was_not_measured():
    """A memory ceiling cannot pass without a peak-memory observation."""
    budget = performance.PerformanceBudget(
        {performance.PerformanceOperation.LOAD: 2.0},
        max_peak_memory_gb=1.0,
    )

    result = budget.evaluate(
        (observation(performance.PerformanceOperation.LOAD, memory=None),)
    )

    assert not result.passed
    assert result.violations == ("missing peak memory observation: load",)


def test_measurements_are_unique_and_json_round_trip():
    """Duplicate operation records cannot hide which measurement was used."""
    first = observation(performance.PerformanceOperation.PACKAGE)
    second = observation(performance.PerformanceOperation.PACKAGE, 0.5)
    with pytest.raises(ValueError, match="unique per operation"):
        performance.PerformanceBudget({"package": 2.0}).evaluate(
            (first, second)
        )

    restored = performance.PerformanceObservation.from_dict(first.to_dict())
    assert restored == first


def test_invalid_measurement_and_budget_values_fail_at_the_boundary():
    """NaN, negative, naive, and empty values cannot become evidence."""
    with pytest.raises(ValueError, match="elapsed_seconds"):
        observation(performance.PerformanceOperation.LOAD, float("nan"))
    with pytest.raises(ValueError, match="positive"):
        performance.PerformanceBudget({"load": 0.0})
    with pytest.raises(ValueError, match="timezone"):
        performance.PerformanceObservation(
            performance.PerformanceOperation.LOAD,
            1.0,
            None,
            1,
            "machine",
            "2026-10-21T00:00:00",
        )


def test_result_and_budget_are_deterministic():
    """Serialized policy evidence has stable operation ordering."""
    budget = performance.PerformanceBudget(
        {
            performance.PerformanceOperation.INFERENCE: 2.0,
            performance.PerformanceOperation.INGEST: 1.0,
        }
    )

    assert list(budget.to_dict()["max_seconds"]) == ["inference", "ingest"]
