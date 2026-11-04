"""The F10 certificates are current, honest and cheap to re-check."""

import json

from benchmarks.qualification import f10_certify as cert
from scmle.data.identification import certified


def stored() -> dict:
    return json.loads(cert.STORE_PATH.read_text(encoding="utf-8"))


def test_certificates_match_the_current_code():
    assert cert.problems() == []


def test_missing_and_stale_certificates_are_reported():
    assert cert.problems({"gates": {}}) == [
        f"{name}: no certificate" for name in cert.GATES
    ]
    document = stored()
    document["gates"]["selector"]["code_hash"] = "0" * 64
    assert cert.problems(document) == [
        "selector: stale (code changed since measurement)"
    ]


def test_certification_seeds_do_not_overlap_development_seeds():
    document = stored()
    for name, entry in document["gates"].items():
        assert entry["seeds"][0] == 0, name
        assert entry["seeds"][1] < cert.DEV_SEED_BASE, name
    assert document["dev_seed_base"] == cert.DEV_SEED_BASE


def test_what_was_not_measured_is_listed():
    assert stored()["not_measured"] == list(cert.NOT_MEASURED)
    assert any("NSW" in line for line in cert.NOT_MEASURED)


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
    row = cert.rate(95, 100)
    assert row["wilson"][0] < 0.95 < row["wilson"][1]
    assert cert.alarm_bound(0.01, 2000) > 0.01
    grid = [0.0, 0.1, 0.2]
    curve = cert.power_curve(
        grid, {0.0: (5, 1000), 0.1: (50, 100), 0.2: (100, 100)}
    )
    assert curve["delta_min"] == 0.2


def test_a_small_selector_gate_has_the_expected_shape():
    result = cert.gate_selector(30, cert.DEV_SEED_BASE)
    assert result["refuses_when_none_eligible"]["hits"] == 30
    assert result["accepts_when_eligible"]["hits"] == 30
