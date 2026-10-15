import pytest

from magicml.contracts.events import EventLog
from magicml.core.state import InvalidTransition, RunState, StateMachine


def test_happy_path_emits_event_per_transition():
    log = EventLog()
    sm = StateMachine("r1", log, "abc")
    for s in list(RunState)[1:9]:
        sm.transition(s, "ok")
    assert sm.state is RunState.COMPLETED
    assert len(log.events) == 9
    assert all(e.run_id == "r1" and e.config_hash == "abc" for e in log.events)


def test_cannot_skip_states():
    sm = StateMachine("r1", EventLog())
    with pytest.raises(InvalidTransition):
        sm.transition(RunState.TRAINING)


def test_abort_and_terminal():
    sm = StateMachine("r1", EventLog())
    sm.transition(RunState.VALIDATING)
    sm.transition(RunState.BLOCKED, "ambiguous target")
    with pytest.raises(InvalidTransition):
        sm.transition(RunState.PROFILING)
