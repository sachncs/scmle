"""The pipeline engine on fake stages: order, state, review, until, aborts."""

import contextlib
import json

import pytest

from scmle.contracts import errors
from scmle.contracts import events
from scmle.pipeline import engine
from scmle.runtime import state


class Regulator:
    def cap(self):
        return contextlib.nullcontext()


class Board:
    """Stands in for the run context: a cancel flag and a thread cap."""

    def __init__(self):
        self.cancelled = False
        self.ran = []
        self.regulator = Regulator()

    def check_cancelled(self):
        if self.cancelled:
            raise errors.CancelledError("cancelled")


def stage(name, progress, needs=(), effect=None):
    def execute(self, context):
        context.ran.append(name)
        if effect is not None:
            effect(context)

    return type(
        name,
        (),
        {
            "state": progress,
            "requires": needs,
            "describe": lambda self, context: f"enter {name}",
            "execute": execute,
        },
    )


def ignore(gate, context):
    """A reviewer that approves every gate."""


def chain(effect=None):
    first = stage("First", state.Progress.VALIDATING)
    second = stage("Second", state.Progress.VALIDATING, (first,))
    third = stage("Third", state.Progress.PROFILING, (first,), effect)
    return first(), second(), third()


def machine(tmp_path):
    log = events.Log(tmp_path / "events.jsonl")
    return state.Tracker("r1", log), tmp_path / "events.jsonl"


def kinds(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line)["kind"] for line in handle]


def test_stages_run_in_dependency_order_and_share_a_state(tmp_path):
    board = Board()
    tracker, path = machine(tmp_path)
    reviewed = []
    pipeline = engine.Pipeline(tuple(reversed(chain())))
    pipeline.advance(
        board, tracker, lambda gate, context: reviewed.append(gate)
    )
    assert board.ran == ["First", "Second", "Third"]
    assert kinds(path) == [
        "state.CREATED",
        "state.VALIDATING",
        "state.PROFILING",
    ]
    assert reviewed == [state.Progress.VALIDATING, state.Progress.PROFILING]
    assert set(pipeline.timings) == {"First", "Second", "Third"}
    assert all(lap.seconds >= 0 for lap in pipeline.timings.values())


def test_a_missing_need_a_duplicate_and_a_cycle_are_refused():
    first, second, third = chain()
    with pytest.raises(errors.ConfigError, match="requires"):
        engine.Pipeline((second, third))
    with pytest.raises(errors.ConfigError, match="more than once"):
        engine.Pipeline((first, first))
    left = stage("Left", state.Progress.VALIDATING)
    right = stage("Right", state.Progress.VALIDATING, (left,))
    left.requires = (right,)
    with pytest.raises(errors.ConfigError, match="cycle"):
        engine.Pipeline((left(), right()))


def test_until_runs_only_the_stage_and_what_it_needs():
    board = Board()
    first, second, third = chain()
    engine.Pipeline((first, second, third)).advance(board, until=type(third))
    assert board.ran == ["First", "Third"]
    with pytest.raises(errors.ConfigError, match="no Board"):
        engine.Pipeline((first,)).select(Board)


def test_cancellation_is_checked_before_each_stage(tmp_path):
    def cancel(context):
        context.cancelled = True

    board = Board()
    tracker, path = machine(tmp_path)
    first = stage("First", state.Progress.VALIDATING, (), cancel)
    second = stage("Second", state.Progress.PROFILING, (first,))
    pipeline = engine.Pipeline((first(), second()))
    with pytest.raises(errors.CancelledError) as caught:
        pipeline.run(board, tracker, ignore, lambda: {"run_dir": "d", "n": 1})
    assert board.ran == ["First"]
    assert caught.value.details == {"run_dir": "d", "n": 1}
    assert kinds(path)[-1] == "state.CANCELLED"


@pytest.mark.parametrize(
    ("raised", "escaped", "final"),
    [
        (errors.BlockedError("no"), errors.BlockedError, "state.BLOCKED"),
        (errors.InputError("bad"), errors.InputError, "state.FAILED"),
        (ValueError("defect"), errors.ScmleError, "state.FAILED"),
        (KeyboardInterrupt(), errors.CancelledError, "state.CANCELLED"),
    ],
)
def test_each_failure_ends_in_its_terminal_state(
    tmp_path, raised, escaped, final
):
    def fail(context):
        raise raised

    tracker, path = machine(tmp_path)
    pipeline = engine.Pipeline(chain(fail))
    with pytest.raises(escaped) as caught:
        pipeline.run(Board(), tracker, ignore, lambda: {"run_dir": "d"})
    assert caught.value.details["run_dir"] == "d"
    assert kinds(path)[-1] == final


def test_a_complete_run_ends_completed_with_its_artifacts(tmp_path):
    tracker, path = machine(tmp_path)
    lifecycle = state.FORWARD_ORDER[1:-1]
    stages = [
        stage(f"S{index}", progress)()
        for index, progress in enumerate(lifecycle)
    ]
    engine.Pipeline(stages).run(Board(), tracker, ignore, dict, ("model/",))
    with path.open(encoding="utf-8") as handle:
        last = [json.loads(line) for line in handle][-1]
    assert last["kind"] == "state.COMPLETED"
    assert last["artifacts"] == ["model/"]
