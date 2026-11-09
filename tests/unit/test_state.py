"""Tests for the run state machine."""

import pytest

from scmle.contracts import events
from scmle.core import state


def test_happy_path_emits_event_per_transition():
    log = events.Log()
    machine = state.Tracker("r1", log, "abc")
    for step in state.FORWARD_ORDER[1:]:
        machine.transition(step, "ok")
    assert machine.state is state.RunState.COMPLETED
    assert len(log.events) == len(state.FORWARD_ORDER)
    assert all(e.run_id == "r1" and e.config_hash == "abc" for e in log.events)


def test_cannot_skip_states():
    machine = state.Tracker("r1", events.Log())
    with pytest.raises(state.InvalidTransitionError):
        machine.transition(state.RunState.TRAINING)


def test_abort_then_terminal():
    machine = state.Tracker("r1", events.Log())
    machine.transition(state.RunState.VALIDATING)
    machine.transition(state.RunState.BLOCKED, "ambiguous target")
    with pytest.raises(state.InvalidTransitionError):
        machine.transition(state.RunState.PROFILING)
