"""The causal certificates are current, honest and cheap to re-check."""

import json

from benchmarks.qualification import f10_certify
from scmle.data.identification import certified


def stored() -> dict:
    return json.loads(f10_certify.STORE_PATH.read_text(encoding="utf-8"))


def test_certificates_match_the_current_code():
    assert f10_certify.problems() == []


def test_missing_and_stale_certificates_are_reported():
    assert f10_certify.problems({"gates": {}}) == [
        f"{name}: no certificate" for name in f10_certify.GATES
    ]
    document = stored()
    document["gates"]["selector"]["code_hash"] = "0" * 64
    assert f10_certify.problems(document) == [
        "selector: stale (code changed since measurement)"
    ]


def test_certification_seeds_do_not_overlap_development_seeds():
    document = stored()
    for name, entry in document["gates"].items():
        assert entry["seeds"][0] == 0, name
        assert entry["seeds"][1] < f10_certify.DEV_SEED_BASE, name
    assert document["dev_seed_base"] == f10_certify.DEV_SEED_BASE


def test_what_was_not_measured_is_listed():
    assert stored()["not_measured"] == list(f10_certify.NOT_MEASURED)
    assert any("NSW" in line for line in f10_certify.NOT_MEASURED)


def test_the_package_counts_only_passing_current_gates():
    document = stored()
    passed = {k for k, v in document["gates"].items() if v["passed"]}
    assert certified.certified_gates() == passed
    assert set(certified.NOT_MEASURED).isdisjoint(passed)


def test_a_changed_method_makes_its_gate_uncertified(tmp_path):
    document = stored()
    for entry in document["gates"].values():
        entry["method_hash"] = "0" * 64
    path = tmp_path / "f10.json"
    path.write_text(json.dumps(document))
    assert certified.certified_gates(path) == set()
    assert certified.certified_gates(tmp_path / "missing.json") == set()


def test_rate_and_floor_helpers():
    row = f10_certify.rate(95, 100)
    assert row["wilson"][0] < 0.95 < row["wilson"][1]
    assert f10_certify.alarm_bound(0.01, 2000) > 0.01
    grid = [0.0, 0.1, 0.2]
    curve = f10_certify.power_curve(
        grid, {0.0: (5, 1000), 0.1: (50, 100), 0.2: (100, 100)}
    )
    assert curve["delta_min"] == 0.2


def test_a_small_selector_gate_has_the_expected_shape():
    result = f10_certify.gate_selector(30, f10_certify.DEV_SEED_BASE)
    assert result["refuses_when_none_eligible"]["hits"] == 30
    assert result["accepts_when_eligible"]["hits"] == 30
