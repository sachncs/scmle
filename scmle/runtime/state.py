"""Run state machine. Every transition emits an event."""

import enum

from scmle.contracts import errors
from scmle.contracts import events


class Progress(enum.StrEnum):
    """Lifecycle states of a run."""

    CREATED = "CREATED"
    VALIDATING = "VALIDATING"
    PROFILING = "PROFILING"
    PLANNING = "PLANNING"
    TRAINING = "TRAINING"
    SELECTING = "SELECTING"
    TESTING = "TESTING"
    PACKAGING = "PACKAGING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    BLOCKED = "BLOCKED"


FORWARD_ORDER = (
    Progress.CREATED,
    Progress.VALIDATING,
    Progress.PROFILING,
    Progress.PLANNING,
    Progress.TRAINING,
    Progress.SELECTING,
    Progress.TESTING,
    Progress.PACKAGING,
    Progress.COMPLETED,
)
ABORT_STATES = frozenset(
    {Progress.FAILED, Progress.CANCELLED, Progress.BLOCKED}
)
TERMINAL_STATES = ABORT_STATES | {Progress.COMPLETED}


class TransitionError(errors.ScmleError):
    """A state change violated the lifecycle order."""

    category = "state"


class Tracker:
    """Tracks the current run state and records every transition."""

    def __init__(
        self,
        run_id: str,
        log: events.Log,
        config_hash: str = "",
        reason: str = "run created",
    ) -> None:
        """Initializes the machine in `CREATED` and emits the first event.

        Args:
            run_id: Identifier attached to every event.
            log: Destination for transition events.
            config_hash: Configuration hash attached to every event.
            reason: Why the machine was created (for example a resume).
        """
        self.run_id = run_id
        self.log = log
        self.config_hash = config_hash
        self.state = Progress.CREATED
        self.emit(self.state, reason)

    def emit(
        self,
        state: Progress,
        reason: str,
        artifacts: tuple[str, ...] = (),
    ) -> None:
        """Emits the event for entering `state`.

        Args:
            state: The state being entered.
            reason: Why the state is entered.
            artifacts: Paths of the artifacts the state produced.
        """
        self.log.emit(
            events.Event(
                run_id=self.run_id,
                kind=f"state.{state.value}",
                reason=reason,
                config_hash=self.config_hash,
                artifacts=artifacts,
            )
        )

    def transition(
        self,
        target: Progress,
        reason: str = "",
        artifacts: tuple[str, ...] = (),
    ) -> None:
        """Moves to `target`.

        Args:
            target: The next state.
            reason: Why the transition happens.
            artifacts: Run-relative artifact paths relevant to the change.

        Raises:
            TransitionError: If the machine is in a terminal state, or
                `target` is neither the immediate successor of the current
                state nor an abort state (an abort is allowed from any
                non-terminal state).
        """
        if self.state in TERMINAL_STATES:
            raise TransitionError(
                f"{self.state} is terminal; cannot go to {target}"
            )
        if target not in ABORT_STATES:
            expected = FORWARD_ORDER.index(self.state) + 1
            if FORWARD_ORDER.index(target) != expected:
                raise TransitionError(
                    f"{self.state} -> {target} is not a valid step"
                )
        self.state = target
        self.emit(target, reason, artifacts)
