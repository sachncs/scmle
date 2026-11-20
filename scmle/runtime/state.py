"""The state machine every run moves through, driven by a transition table.

A `Machine` holds one state and a table, given as data, of the states each
state may move to; a state with no entry or an empty one is terminal. The
tabular run (`TABLE`, through `Tracker`, which emits an event per
transition), the modality lifecycle (`modality_lifecycle.state.TABLE`) and
the workflow executor (`workflow.executor.status.TABLE`) are three tables
on this one machine; `chain` builds the forward tables.
"""

from collections.abc import Hashable
from collections.abc import Mapping
from collections.abc import Sequence
import enum
import itertools
from typing import Generic
from typing import TypeVar

from scmle.contracts import errors
from scmle.contracts import events

StateT = TypeVar("StateT", bound=Hashable)


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


def chain(
    order: Sequence[StateT], aborts: frozenset[StateT]
) -> dict[StateT, frozenset[StateT]]:
    """Builds the table of a forward lifecycle.

    Args:
        order: The states in order, the last one terminal.
        aborts: States any non-terminal state may move to; each is terminal.

    Returns:
        Each state of `order` moving to its successor or to an abort state,
        and the last state and every abort state moving nowhere.
    """
    table = {
        current: frozenset({following, *aborts})
        for current, following in itertools.pairwise(order)
    }
    return table | dict.fromkeys((order[-1], *aborts), frozenset())


TABLE = chain(FORWARD_ORDER, ABORT_STATES)
"""The tabular run's table: one state forward, or an abort from any state
that is not terminal."""


class TransitionError(errors.ScmleError):
    """A state change violated the lifecycle order."""

    category = "state"


class Machine(Generic[StateT]):
    """One state, moved only along a transition table.

    Attributes:
        table: The states each state may move to; a state with no entry, or
            an empty one, is terminal.
        state: The current state.
    """

    def __init__(
        self, table: Mapping[StateT, frozenset[StateT]], state: StateT
    ) -> None:
        """Starts the machine in a state.

        Args:
            table: The transition table.
            state: The state it starts in.
        """
        self.table = table
        self.state = state

    def transition(
        self,
        target: StateT,
        reason: str = "",
        artifacts: tuple[str, ...] = (),
    ) -> None:
        """Moves to `target`, then enters it (`enter`).

        Args:
            target: The next state.
            reason: Why the transition happens.
            artifacts: Paths of the artifacts relevant to the change.

        Raises:
            TransitionError: If the current state is terminal, or the table
                does not let it move to `target`.
        """
        allowed = self.table.get(self.state, frozenset())
        if not allowed:
            raise TransitionError(
                f"{self.state} is terminal; cannot go to {target}"
            )
        if target not in allowed:
            raise TransitionError(
                f"{self.state} -> {target} is not a valid step"
            )
        self.state = target
        self.enter(target, reason, artifacts)

    def enter(
        self, state: StateT, reason: str, artifacts: tuple[str, ...]
    ) -> None:
        """Records entering a state; a plain machine records nothing.

        Args:
            state: The state entered.
            reason: Why it was entered.
            artifacts: Paths of the artifacts relevant to the change.
        """


class Tracker(Machine[Progress]):
    """The tabular run's machine (`TABLE`); every transition emits an event.

    Attributes:
        run: Identifier attached to every event.
        log: Destination for transition events.
        config: Configuration hash attached to every event.
    """

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
        super().__init__(TABLE, Progress.CREATED)
        self.run = run_id
        self.log = log
        self.config = config_hash
        self.enter(self.state, reason, ())

    def enter(
        self, state: Progress, reason: str, artifacts: tuple[str, ...]
    ) -> None:
        """Emits the event for entering `state`.

        Args:
            state: The state being entered.
            reason: Why the state is entered.
            artifacts: Paths of the artifacts the state produced.
        """
        self.log.emit(
            events.Event(
                run_id=self.run,
                kind=f"state.{state.value}",
                reason=reason,
                config_hash=self.config,
                artifacts=artifacts,
            )
        )
