"""Exact refusal of a scenario that is not a contingency."""

import pytest

from scmle.contracts import errors
from scmle.rehearsal.local_faults import executor


def test_a_scenario_must_be_a_contingency(tmp_path):
    runner = executor.Executor(tmp_path / "faults")

    with pytest.raises(errors.KindError) as caught:
        runner.execute("process_death")

    assert str(caught.value) == "scenario must be Contingency"
