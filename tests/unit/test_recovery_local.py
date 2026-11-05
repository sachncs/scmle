"""Every local fault, run against the real local service and emulators.

These start real subprocesses (the generated ``serve.py``, a ledger writer, a
workflow executor) and kill them, so they take a while. Each scenario owns
what it starts; the last assertion is that nothing was left running.
"""

import subprocess

import pytest

from scmle.operations import faults
from scmle.operations import recovery


@pytest.fixture(scope="module", autouse=True)
def nothing_left_behind():
    yield
    listing = subprocess.run(
        ["ps", "-axo", "command"], capture_output=True, text=True, check=False
    ).stdout
    assert "scmle-recovery-" not in listing, "a scenario left a process behind"


@pytest.mark.parametrize("fault_id", sorted(faults.SCENARIOS))
def test_each_local_fault_recovers_within_its_objectives(fault_id):
    result = recovery.Runner({fault_id: faults.SCENARIOS[fault_id]}).run(
        [fault_id]
    )
    found = next(r for r in result["results"] if r["fault_id"] == fault_id)
    assert found["status"] == recovery.PASSED, found["problems"]
    assert found["measurements"], "a recovery without a measurement"
    assert found["detail"]["processes_still_alive_after_close"] == 0
    for measured in found["measurements"]:
        assert measured["rto_seconds"] is not None
        assert measured["rpo_lost"] == 0
        assert measured["method"]


def test_the_local_scenarios_are_exactly_the_runnable_faults():
    runnable = {f.id for f in recovery.CATALOGUE if f.local}
    assert set(faults.SCENARIOS) == runnable


def test_the_wrong_key_is_never_printed_by_the_scenario_report():
    result = faults.invalid_credentials()
    result.judge()
    text = str(result.to_dict())
    assert faults.WRONG_KEY not in text and faults.KEY not in text
