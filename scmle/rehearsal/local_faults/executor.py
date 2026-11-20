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
            reliability.Contingency.PROCESS_DEATH: self.kill,
            reliability.Contingency.STALE_CHECKPOINT: (self.rebind),
            reliability.Contingency.DISK_FULL: self.fill,
            reliability.Contingency.CORRUPT_ARTIFACT: (self.corrupt),
            reliability.Contingency.WORKER_LOSS: self.lose,
            reliability.Contingency.TIMEOUT: self.expire,
            reliability.Contingency.MEMORY_EXHAUSTION: (self.starve),
            reliability.Contingency.PARTIAL_WRITE: self.truncate,
            reliability.Contingency.DEPENDENCY_MISMATCH: (self.mismatch),
            reliability.Contingency.INVALID_SIGNATURE: (self.forge),
            reliability.Contingency.RETRY_EXHAUSTION: (self.exhaust),
            reliability.Contingency.DUPLICATE_EVENT: (self.duplicate),
            reliability.Contingency.INCOMPLETE_LINEAGE: (self.orphan),
            reliability.Contingency.REJECTED_APPROVAL: (self.reject),
            reliability.Contingency.CHANGED_SNAPSHOT: (self.mutate),
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
        names = tuple(item.kind for item in artifacts)
        if set(names) != set(expected) or len(names) != len(set(names)):
            raise ValueError(
                f"scenario {scenario.value} returned evidence {names}; "
                f"expected {expected}"
            )
        return artifacts
