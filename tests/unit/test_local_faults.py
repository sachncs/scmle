"""Tests for the controlled local reliability fault executor."""

from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

import pytest

from scmle.contracts import reliability
import scmle.governance.reliability
from scmle.rehearsal import local_faults


def test_local_executor_passes_every_declared_scenario(tmp_path):
    """Every local scenario must satisfy its complete evidence contract."""
    root = tmp_path / "local-faults"
    executor = local_faults.Executor(root)
    registry = reliability.Registry.prerelease()
    harness = scmle.governance.reliability.Harness(
        registry,
        root / "evidence",
        execution_id="local-fault-run",
        reviewer="local-release-rehearsal",
        recorded_at="2026-10-20T12:00:00+00:00",
    )

    for scenario in reliability.Contingency:
        plan = harness.run(scenario, executor.execute)

        assert plan.verification_status is reliability.Verification.PASSED
        assert all(
            evidence.observed_outcome
            is reliability.Outcome.SAFE_OUTCOME_OBSERVED
            for evidence in plan.evidence
        )
        assert all(
            Path(url2pathname(urlparse(evidence.artifact_uri).path)).exists()
            for evidence in plan.evidence
        )


@pytest.mark.parametrize("scenario", tuple(reliability.Contingency))
def test_local_executor_returns_declared_evidence_only(tmp_path, scenario):
    """A scenario cannot pass by omitting or inventing evidence types."""
    executor = local_faults.Executor(tmp_path / scenario.value)
    artifacts = executor.execute(scenario)
    plan = next(
        plan for plan in reliability.drills() if plan.scenario is scenario
    )

    assert tuple(item.evidence_type for item in artifacts) == (
        plan.evidence_requirements
    )


def test_local_executor_rejects_unknown_scenario(tmp_path):
    """The executor validates the typed scenario boundary."""
    executor = local_faults.Executor(tmp_path)

    with pytest.raises(TypeError, match="Contingency"):
        executor.execute("process_death")  # type: ignore[arg-type]


def test_local_resource_injectors_do_not_fill_or_allocate_the_host(tmp_path):
    """Disk and memory exercises remain bounded controlled failures."""
    executor = local_faults.Executor(tmp_path)

    executor.execute(reliability.Contingency.DISK_FULL)
    executor.execute(reliability.Contingency.MEMORY_EXHAUSTION)

    files = tuple(path for path in tmp_path.rglob("*") if path.is_file())
    assert all(path.stat().st_size < 1_000_000 for path in files)
