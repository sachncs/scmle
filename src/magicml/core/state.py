"""Run state machine. Every transition emits an event."""

from __future__ import annotations

from enum import StrEnum

from magicml.contracts.errors import MLForgeError
from magicml.contracts.events import Event, EventLog


class RunState(StrEnum):
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


_FORWARD = [
    RunState.CREATED,
    RunState.VALIDATING,
    RunState.PROFILING,
    RunState.PLANNING,
    RunState.TRAINING,
    RunState.SELECTING,
    RunState.TESTING,
    RunState.PACKAGING,
    RunState.COMPLETED,
]
TERMINAL = {RunState.COMPLETED, RunState.FAILED, RunState.CANCELLED, RunState.BLOCKED}
_ABORT = {RunState.FAILED, RunState.CANCELLED, RunState.BLOCKED}


class InvalidTransition(MLForgeError):
    pass


class StateMachine:
    def __init__(self, run_id: str, log: EventLog, config_hash: str = "") -> None:
        self.run_id = run_id
        self.log = log
        self.config_hash = config_hash
        self.state = RunState.CREATED
        self._emit(self.state, "run created")

    def _emit(self, state: RunState, reason: str, artifacts: tuple[str, ...] = ()) -> None:
        self.log.emit(
            Event(
                run_id=self.run_id,
                kind=f"state.{state.value}",
                reason=reason,
                config_hash=self.config_hash,
                artifacts=artifacts,
            )
        )

    def transition(
        self, to: RunState, reason: str = "", artifacts: tuple[str, ...] = ()
    ) -> None:
        if self.state in TERMINAL:
            raise InvalidTransition(f"{self.state} is terminal; cannot go to {to}")
        if to not in _ABORT:
            if _FORWARD.index(to) != _FORWARD.index(self.state) + 1:
                raise InvalidTransition(f"{self.state} -> {to} is not a valid step")
        self.state = to
        self._emit(to, reason, artifacts)
