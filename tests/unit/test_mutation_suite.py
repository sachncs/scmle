"""How the diagnostic suite times the model and shares the time left."""

import dataclasses

import pandas
import pytest

from scmle.contracts import tasks
from scmle.evaluation import diagnostics
import scmle.pipeline.suite
from tests.unit.test_diagnostics import Clock
from tests.unit.test_diagnostics import make_inputs

ROWS = 2 * scmle.pipeline.suite.PROBE_ROWS
STEP = 2.0


class Model:
    """Predicts nothing, taking `STEP` seconds of the test clock per call."""

    def __init__(self, clock: Clock) -> None:
        self.clock = clock

    def predict(self, features):
        self.clock.now += STEP
        return [0] * len(features)


def inputs(clock: Clock) -> diagnostics.Input:
    frame = pandas.DataFrame({"signal": [0.0] * ROWS})
    return dataclasses.replace(
        make_inputs(tasks.Task.BINARY), estimator=Model(clock), features=frame
    )


def probe(name: str, cost: float, seen: list) -> diagnostics.Diagnostic:
    class Probe(diagnostics.Diagnostic):
        fixed = cost
        passes = 0.0

        def compute(self, data):
            seen.append((name, data.pace, data.deadline))
            return {}

    Probe.name = name
    return Probe()


@pytest.fixture
def clock(monkeypatch) -> Clock:
    found = Clock()
    monkeypatch.setattr(diagnostics, "clock", found)
    return found


def test_the_gauge_scales_the_probe_to_every_row(clock) -> None:
    found = scmle.pipeline.suite.gauge(inputs(clock))
    probed = scmle.pipeline.suite.PROBE_ROWS
    assert found == pytest.approx(STEP * ROWS / probed)


def test_the_suite_times_the_model_while_time_is_left(clock) -> None:
    seen = []
    scmle.pipeline.suite.run_suite(
        inputs(clock), (probe("a", 0.0, seen),), deadline=clock.now + 100.0
    )
    assert seen[0][1] == pytest.approx(STEP * 2)


def test_the_suite_does_not_time_the_model_at_the_deadline(clock) -> None:
    seen = []
    scmle.pipeline.suite.run_suite(
        inputs(clock), (probe("a", 0.0, seen),), deadline=clock.now
    )
    assert seen == [("a", None, clock.now)]


def test_each_diagnostic_gets_its_share_of_the_time_left(clock) -> None:
    seen = []
    start = clock.now
    left = 100.0 + STEP
    scmle.pipeline.suite.run_suite(
        inputs(clock),
        (probe("a", 1.0, seen), probe("b", 3.0, seen)),
        deadline=start + left,
    )
    # After the timing call, 100 s are left; "a" wants a quarter of them.
    assert seen[0] == ("a", pytest.approx(STEP * 2), start + STEP + 25.0)
