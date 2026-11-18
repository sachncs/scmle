"""Where a run and its steps are."""

import enum


class Progress(enum.StrEnum):
    """Where a run is.

    Attributes:
        RUNNING: Steps are being executed.
        WAITING_APPROVAL: Stopped at an approval step with no decision yet.
        SUCCEEDED: Every step succeeded.
        FAILED: A step failed.
        BUDGET_STOPPED: The run's cost budget was passed.
        REJECTED: An approver rejected the run.
        EXPIRED: An approval was not decided in time.
    """

    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    BUDGET_STOPPED = "budget_stopped"
    REJECTED = "rejected"
    EXPIRED = "expired"


FINISHED = frozenset(
    {
        Progress.SUCCEEDED,
        Progress.FAILED,
        Progress.BUDGET_STOPPED,
        Progress.REJECTED,
        Progress.EXPIRED,
    }
)

TABLE = {
    Progress.RUNNING: frozenset(Progress),
    Progress.WAITING_APPROVAL: frozenset({Progress.RUNNING}),
    Progress.FAILED: frozenset({Progress.RUNNING}),
    Progress.SUCCEEDED: frozenset(),
    Progress.BUDGET_STOPPED: frozenset(),
    Progress.REJECTED: frozenset(),
    Progress.EXPIRED: frozenset(),
}
"""A run's transition table on the shared `Machine` (`scmle.runtime.state`).

A drive enters `RUNNING` and ends in any state; `RUNNING` again continues a
run whose executor died. Only a waiting run (after a decision) or a failed
one (from its failed steps) runs again; the other ends are settled."""


class Phase(enum.StrEnum):
    """Where a step is.

    Attributes:
        PENDING: Not started.
        RUNNING: Executing.
        SUCCEEDED: Ended successfully.
        FAILED: Ended with an error after its retries.
        SKIPPED: Not run because an upstream step did not succeed or the run's
            budget was passed.
        WAITING: An approval step waiting for a decision.
        REJECTED: An approval step that was rejected.
        EXPIRED: An approval step that was not decided in time.
    """

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"
    WAITING = "waiting"
    REJECTED = "rejected"
    EXPIRED = "expired"


UNMET = frozenset({Phase.FAILED, Phase.SKIPPED, Phase.REJECTED, Phase.EXPIRED})
"""Step states that leave a dependent step unable to run: it is skipped."""
