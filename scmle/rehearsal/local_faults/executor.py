"""The executor that dispatches each fault scenario to its handler."""

from typing import TYPE_CHECKING

from scmle.contracts import errors
from scmle.contracts import reliability
import scmle.governance.reliability
from scmle.rehearsal.local_faults import interruptions
from scmle.rehearsal.local_faults import provenance
from scmle.rehearsal.local_faults import resources

if TYPE_CHECKING:
    from collections.abc import Callable


class Executor(
    interruptions.Interruptions, resources.Resources, provenance.Provenance
):
    """Execute bounded, reproducible fault scenarios in a local workspace."""

    __slots__ = ()

    def execute(
        self, scenario: reliability.Contingency
    ) -> tuple[scmle.governance.reliability.Emission, ...]:
        """Executes one scenario and checks its evidence against its plan.

        Args:
            scenario: The fault scenario to rehearse.

        Returns:
            The artifacts the scenario's handler produced, in its order;
            their evidence types are exactly those its default plan requires,
            each once.

        Raises:
            RuntimeError: If the scenario's handler raises any exception.
            TypeError: If `scenario` is not a `Contingency`.
            ValueError: If the evidence types produced differ from those the
                plan requires or repeat.
        """
        if not isinstance(scenario, reliability.Contingency):
            raise errors.KindError("scenario must be Contingency")
        handlers: dict[
            reliability.Contingency,
            Callable[[], tuple[scmle.governance.reliability.Emission, ...]],
        ] = {
            reliability.Contingency.PROCESS_DEATH: self.run_process_death,
            reliability.Contingency.STALE_CHECKPOINT: (
                self.run_stale_checkpoint
            ),
            reliability.Contingency.DISK_FULL: self.run_disk_full,
            reliability.Contingency.CORRUPT_ARTIFACT: (
                self.run_corrupt_artifact
            ),
            reliability.Contingency.WORKER_LOSS: self.run_worker_loss,
            reliability.Contingency.TIMEOUT: self.run_timeout,
            reliability.Contingency.MEMORY_EXHAUSTION: (
                self.run_memory_exhaustion
            ),
            reliability.Contingency.PARTIAL_WRITE: self.run_partial_write,
            reliability.Contingency.DEPENDENCY_MISMATCH: (
                self.run_dependency_mismatch
            ),
            reliability.Contingency.INVALID_SIGNATURE: (
                self.run_invalid_signature
            ),
            reliability.Contingency.RETRY_EXHAUSTION: (
                self.run_retry_exhaustion
            ),
            reliability.Contingency.DUPLICATE_EVENT: (self.run_duplicate_event),
            reliability.Contingency.INCOMPLETE_LINEAGE: (
                self.run_incomplete_lineage
            ),
            reliability.Contingency.REJECTED_APPROVAL: (
                self.run_rejected_approval
            ),
            reliability.Contingency.CHANGED_SNAPSHOT: (
                self.run_changed_snapshot
            ),
        }
        try:
            artifacts = handlers[scenario]()
        except Exception as error:
            raise RuntimeError(
                f"local fault scenario failed: {scenario.value}"
            ) from error
        expected = next(
            plan.requirements
            for plan in reliability.drills()
            if plan.scenario is scenario
        )
        names = tuple(item.evidence_type for item in artifacts)
        if set(names) != set(expected) or len(names) != len(set(names)):
            raise ValueError(
                f"scenario {scenario.value} returned evidence {names}; "
                f"expected {expected}"
            )
        return artifacts
