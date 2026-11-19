"""Tests for the run state machine."""

import pytest

from scmle.contracts import events
from scmle.runtime import state


def test_happy_path_emits_event_per_transition():
    log = events.Log()
    machine = state.Tracker("r1", log, "abc")
    for step in state.FORWARD_ORDER[1:]:
        machine.transition(step, "ok")
    assert machine.state is state.Progress.COMPLETED
    assert len(log.events) == len(state.FORWARD_ORDER)
    assert all(e.run_id == "r1" and e.config_hash == "abc" for e in log.events)


def test_cannot_skip_states():
    machine = state.Tracker("r1", events.Log())
    with pytest.raises(state.TransitionError):
        machine.transition(state.Progress.TRAINING)


def test_abort_then_terminal():
    machine = state.Tracker("r1", events.Log())
    machine.transition(state.Progress.VALIDATING)
    machine.transition(state.Progress.BLOCKED, "ambiguous target")
    with pytest.raises(state.TransitionError):
        machine.transition(state.Progress.PROFILING)


def test_a_tracker_is_the_machine_over_the_lifecycle_table():
    machine = state.Tracker("r1", events.Log())
    assert isinstance(machine, state.Machine)
    assert machine.table is state.TABLE
    assert state.chain(state.FORWARD_ORDER, state.ABORT_STATES) == state.TABLE
    assert state.TABLE[state.Progress.CREATED] == frozenset(
        {state.Progress.VALIDATING, *state.ABORT_STATES}
    )
    assert all(not state.TABLE[end] for end in state.TERMINAL_STATES)


def test_a_machine_moves_only_along_its_table():
    table = {
        "a": frozenset({"b"}),
        "b": frozenset({"a", "c"}),
        "c": frozenset(),
    }
    machine = state.Machine(table, "a")
    machine.transition("b")
    machine.transition("a")
    with pytest.raises(state.TransitionError, match=r"^a -> c is not"):
        machine.transition("c")
    machine.transition("b")
    machine.transition("c")
    assert machine.state == "c"
    with pytest.raises(state.TransitionError, match=r"^c is terminal"):
        machine.transition("a")


def test_a_state_missing_from_the_table_is_terminal():
    machine = state.Machine({"a": frozenset({"b"})}, "a")
    machine.transition("b")
    with pytest.raises(state.TransitionError, match="terminal"):
        machine.transition("a")
