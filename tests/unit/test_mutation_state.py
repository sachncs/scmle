"""Exact events and refusals of the run state machine."""

import pytest

from scmle.contracts import events
from scmle.core import state


def fields(event: events.Event) -> tuple:
    return (
        event.run_id,
        event.kind,
        event.reason,
        event.config_hash,
        tuple(event.artifacts),
    )


def test_creation_emits_the_default_reason() -> None:
    log = events.Log()
    state.Tracker("r1", log, "abc")
    assert [fields(e) for e in log.events] == [
        ("r1", "state.CREATED", "run created", "abc", ())
    ]


def test_transition_records_reason_and_artifacts() -> None:
    log = events.Log()
    machine = state.Tracker("r1", log, "abc", reason="resumed")
    machine.transition(state.Progress.VALIDATING, "checked", ("a.json",))
    machine.transition(state.Progress.PROFILING)
    assert [fields(e) for e in log.events] == [
        ("r1", "state.CREATED", "resumed", "abc", ()),
        ("r1", "state.VALIDATING", "checked", "abc", ("a.json",)),
        ("r1", "state.PROFILING", "", "abc", ()),
    ]


def test_a_skipped_state_is_refused_by_name() -> None:
    machine = state.Tracker("r1", events.Log())
    with pytest.raises(
        state.TransitionError,
        match=r"^CREATED -> TRAINING is not a valid step$",
    ):
        machine.transition(state.Progress.TRAINING)


def test_a_terminal_state_is_refused_by_name() -> None:
    machine = state.Tracker("r1", events.Log())
    machine.transition(state.Progress.CANCELLED)
    with pytest.raises(
        state.TransitionError,
        match=r"^CANCELLED is terminal; cannot go to FAILED$",
    ):
        machine.transition(state.Progress.FAILED)
