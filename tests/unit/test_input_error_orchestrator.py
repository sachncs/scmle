"""A stage's input error fails the run as input, not as a defect (ERR-20).

Before the raises moved to `errors.InputError`, a stage's `ValueError` fell
to the orchestrator's catch-all: the run failed as `internal: ValueError(..)`
and escaped wrapped in a plain `ScmleError`. An `InputError` is an
`ScmleError`, so it now takes the categorised branch: the run fails as
`input: ...` and the error itself escapes. A plain `ValueError`, a defect,
still takes the catch-all.
"""

import json
import pathlib

import pytest
import support

from scmle.contracts import errors
from scmle.pipeline import orchestrator
import scmle.pipeline.stages


def failing(frame, tmp_path, error):
    """Runs the stock stages with a first stage that raises `error`."""

    first, *rest = scmle.pipeline.stages.sequence()

    class Failing(type(first)):
        def execute(self, context):
            raise error

    stages = (Failing(), *rest)
    spec = support.make_project(frame, "y").spec
    engine = orchestrator.Orchestrator(spec, frame, stages=stages)
    with pytest.raises(errors.ScmleError) as caught:
        engine.run(tmp_path)
    path = pathlib.Path(caught.value.details["run_dir"], "events.jsonl")
    with path.open(encoding="utf-8") as handle:
        last = [json.loads(line) for line in handle][-1]
    return caught.value, last


@pytest.mark.parametrize(
    "error",
    [
        errors.InputError("bad column"),
        errors.KindError("bad column"),
        errors.OrderError("bad column"),
    ],
)
def test_an_input_error_fails_the_run_as_input(binary_frame, tmp_path, error):
    raised, last = failing(binary_frame, tmp_path, error)
    assert raised is error and raised.category == "input"
    assert last["kind"] == "state.FAILED"
    assert "input: bad column" in json.dumps(last)
    assert "recoverable_trials" in raised.details


def test_a_plain_value_error_still_fails_as_internal(binary_frame, tmp_path):
    raised, last = failing(binary_frame, tmp_path, ValueError("defect"))
    assert type(raised) is errors.ScmleError
    assert isinstance(raised.__cause__, ValueError)
    assert "internal: ValueError" in json.dumps(last)
