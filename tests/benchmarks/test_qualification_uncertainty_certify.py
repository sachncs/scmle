"""The F12 certificates are current, honest and cheap to re-check."""

import json
import math

from benchmarks.qualification import uncertainty_certify as cert
from scmle.contracts import tasks
from scmle.evaluation import task_gate


def stored() -> dict:
    return json.loads(cert.STORE_PATH.read_text(encoding="utf-8"))


def test_certificates_match_the_current_code() -> None:
    assert cert.problems() == []


def test_missing_and_stale_certificates_are_reported() -> None:
    assert cert.problems({"gates": {}}) == [
        f"{name}: no certificate" for name in cert.GATES
    ]
    document = stored()
    document["gates"]["G1"]["code_hash"] = "0" * 64
    assert cert.problems(document) == [
        "G1: stale (code changed since measurement)"
    ]


def test_certification_seeds_do_not_overlap_development_seeds() -> None:
    document = stored()
    for name, entry in document["gates"].items():
        assert entry["seeds"][0] == 0, name
        assert entry["seeds"][1] < cert.DEV_SEED_BASE, name
    assert document["dev_seed_base"] == cert.DEV_SEED_BASE


def test_gate_status_file_agrees_with_the_certificates() -> None:
    gates = stored()["gates"]
    for task in tasks.TaskType:
        if task_gate.status(task) != task_gate.PASSED:
            continue
        needed = task_gate.load()[task.value]["gates"]
        assert needed and all(gates[g]["passed"] for g in needed), task


def test_misses_are_published_not_hidden() -> None:
    gates = stored()["gates"]
    missed = [name for name, entry in gates.items() if not entry["passed"]]
    assert set(missed) == {"G3", "G5"}
    assert (
        gates["G5"]["familywise_false_flag"] > gates["G5"]["false_flag_bound"]
    )
    assert stored()["not_measured"]


def test_a_small_g1_run_has_the_expected_shape() -> None:
    result = cert.gate_g1(40, cert.DEV_SEED_BASE)
    assert result["design"]["beta"] == [451, 50]
    assert 0.89 < result["mean"] < 0.91
    assert math.isfinite(result["ks_p"])


def test_a_small_g13_run_reports_both_calibrations() -> None:
    result = cert.gate_g13(20, cert.DEV_SEED_BASE)
    assert {"entity_wise", "row_wise"} <= set(result)
