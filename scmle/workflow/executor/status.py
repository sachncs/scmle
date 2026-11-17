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
