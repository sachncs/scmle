"""A stored failure keeps its type, message, traceback and causes (ERR-19).

Records used to keep only `"Type: message"`; the child process of a
supervised trial lost its traceback entirely at the process boundary.
"""

import json

import pytest

from scmle.contracts import errors
from scmle.pipeline import trials
from scmle.runtime import limits


def chained():
    """Raises a ValueError caused by a ZeroDivisionError."""
    try:
        1 / 0  # noqa: B018 -- the point is to raise
    except ZeroDivisionError as error:
        raise ValueError("bad input") from error


def explode(argument):
    """Fails inside a supervised child process."""
    raise RuntimeError(f"child failed on {argument}")


def test_a_failure_keeps_its_cause_and_traceback():
    with pytest.raises(ValueError) as caught:
        chained()
    failure = errors.Failure.of(caught.value)
    assert failure.kind == "ValueError" and failure.message == "bad input"
    assert failure.text == "ValueError: bad input"
    assert failure.causes == ("ZeroDivisionError: division by zero",)
    assert "chained" in failure.trace and "1 / 0" in failure.trace
    assert json.loads(json.dumps(failure.dump())) == {
        "kind": "ValueError",
        "message": "bad input",
        "trace": failure.trace,
        "causes": ["ZeroDivisionError: division by zero"],
    }


def test_the_trace_is_bounded_to_its_innermost_part():
    error = ValueError("x" * (errors.TRACE * 2))
    failure = errors.Failure.of(error)
    assert len(failure.trace) <= errors.TRACE
    assert failure.trace.rstrip().endswith("x")


def test_a_child_process_failure_reaches_the_parent_with_its_frames():
    supervisor = limits.Supervisor(30.0, None)
    with pytest.raises(limits.TrialError) as caught:
        supervisor.run(explode, "row 7")
    error = caught.value
    assert str(error) == "RuntimeError: child failed on row 7"
    remote = error.details["failure"]
    assert remote["kind"] == "RuntimeError" and "explode" in remote["trace"]
    failure = errors.Failure.of(error)
    assert failure.causes[0] == "RuntimeError: child failed on row 7"
    assert "-- in the child process --" in failure.trace
    assert "explode" in failure.trace


def test_a_lost_trial_records_its_failure():
    class Job:
        trial, model, params, strategy = "t1", "m", {}, "default"

    class Metric:
        name = "accuracy"

    class Payload:
        job, metric = Job(), Metric()

    with pytest.raises(ValueError) as caught:
        chained()
    trial = trials.lose(Payload(), caught.value)
    assert trial.error == "WorkerLost: ValueError: bad input"
    assert trial.failure["causes"] == ["ZeroDivisionError: division by zero"]
    assert json.loads(trial.to_dict()["failure"])["kind"] == "ValueError"
