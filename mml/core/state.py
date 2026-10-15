"""Run state machine. Every transition emits an event."""

import enum

from mml.contracts import errors
from mml.contracts import events


class RunState(enum.StrEnum):
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
    RunState.CREATED,
    RunState.VALIDATING,
    RunState.PROFILING,
    RunState.PLANNING,
    RunState.TRAINING,
    RunState.SELECTING,
    RunState.TESTING,
    RunState.PACKAGING,
    RunState.COMPLETED,
)
ABORT_STATES = frozenset(
    {RunState.FAILED, RunState.CANCELLED, RunState.BLOCKED}
)
TERMINAL_STATES = ABORT_STATES | {RunState.COMPLETED}


class InvalidTransitionError(errors.MagicMLError):
    """A state change violated the lifecycle order."""

    category = "state"


class StateMachine:
    """Tracks the current run state and records every transition."""

    def __init__(
        self,
        run_id: str,
        log: events.EventLog,
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
        self.state = RunState.CREATED
        self.emit(self.state, reason)

    def emit(
        self,
        state: RunState,
        reason: str,
        artifacts: tuple[str, ...] = (),
    ) -> None:
        """Emits the event for entering `state`."""
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
        target: RunState,
        reason: str = "",
        artifacts: tuple[str, ...] = (),
    ) -> None:
        """Moves to `target`.

        Args:
            target: The next state.
            reason: Why the transition happens.
            artifacts: Run-relative artifact paths relevant to the change.

        Raises:
            InvalidTransitionError: If the machine is terminal or `target`
                is not the immediate successor (aborts are always allowed).
        """
        if self.state in TERMINAL_STATES:
            raise InvalidTransitionError(
                f"{self.state} is terminal; cannot go to {target}"
            )
        if target not in ABORT_STATES:
            expected = FORWARD_ORDER.index(self.state) + 1
            if FORWARD_ORDER.index(target) != expected:
                raise InvalidTransitionError(
                    f"{self.state} -> {target} is not a valid step"
                )
        self.state = target
        self.emit(target, reason, artifacts)
