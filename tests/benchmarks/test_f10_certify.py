"""The causal certificates are current, honest and cheap to re-check."""

import json

import pytest
from scipy import stats

from benchmarks import acceptance
from benchmarks.qualification import f10_certify
from scmle.causal.identification import certified


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
    assert certified.certificated() == passed
    assert set(certified.NOT_MEASURED).isdisjoint(passed)


def test_a_changed_method_makes_its_gate_uncertified(tmp_path):
    document = stored()
    for entry in document["gates"].values():
        entry["method_hash"] = "0" * 64
    path = tmp_path / "f10.json"
    path.write_text(json.dumps(document))
    assert certified.certificated(path) == set()
    assert certified.certificated(tmp_path / "missing.json") == set()


def test_rate_and_floor_helpers():
    row = f10_certify.rate(95, 100)
    assert row["exact"][0] < 0.95 < row["exact"][1]
    grid = [0.0, 0.1, 0.2]
    curve = f10_certify.power_curve(
        grid, {0.0: (5, 1000), 0.1: (50, 100), 0.2: (100, 100)}
    )
    assert curve["delta_min"] == 0.2


def test_every_rate_is_decided_by_its_exact_bound_at_alpha():
    alpha = acceptance.load()["alpha"].value
    assert alpha == f10_certify.ALPHA
    row = f10_certify.rate(15, 1000)
    assert row["exact"][1] == pytest.approx(stats.beta.ppf(1 - alpha, 16, 985))
    # battery_rd's no-violation count passed `0.01 + 2 SE`; its bound does not
    assert not f10_certify.meets(f10_certify.FALSIFY, row)
    # a placebo-in-space size of 0.04 over 125 runs passed `0.05 + 2 SE`
    assert not f10_certify.meets(
        f10_certify.FALSE_ALARM, f10_certify.rate(5, 125)
    )
    assert f10_certify.meets(f10_certify.FALSE_ALARM, f10_certify.rate(0, 300))
    power = f10_certify.rate(382, 500)
    assert power["exact"][0] == pytest.approx(stats.beta.ppf(alpha, 382, 119))
    assert not f10_certify.meets(f10_certify.POWER, power)
    assert f10_certify.meets(f10_certify.COVERAGE, f10_certify.rate(1890, 2000))


def test_the_thresholds_are_the_acceptance_variables():
    variables = acceptance.load()
    assert variables["interval_coverage_min"].rule == f10_certify.COVERAGE
    assert variables["causal_power_min"].rule == f10_certify.POWER
    assert variables["causal_selection_min"].rule == f10_certify.SELECTION
    assert variables["causal_false_alarm_max"].rule == f10_certify.FALSE_ALARM
    assert acceptance.sized(0.01, variables) == f10_certify.FALSIFY
    assert acceptance.sized(0.05, variables) == f10_certify.SYNTH_SIZE


def test_each_gate_runs_the_count_it_states():
    document = stored()
    assert document["terms"] == f10_certify.terms()
    stated = {name: entry[1] for name, entry in f10_certify.RUNNERS.items()}
    assert stated["size_synth"] == 500
    assert stated["sensitivity_did"] == 500
    assert stated["sensitivity_bound"] == 1000
    assert stated["false_confirm"] == 300
    for name, entry in document["gates"].items():
        assert entry["seeds"] == [0, stated[name]], name
    gates = document["gates"]
    assert gates["size_synth"]["size"]["n"] == 500
    assert gates["battery_rd"]["false_alarm"]["n"] == 2000
    assert gates["overlap"]["replicates"] == 2000
    assert gates["sensitivity_bound"]["replicates"] == 1000
    assert gates["sensitivity_did"]["m1_coverage"]["n"] == 500
    assert gates["false_confirm"]["replicates"] == 300


def test_the_stored_verdicts_follow_from_the_stored_counts():
    for name, entry in stored()["gates"].items():
        assert entry["passed"] is f10_certify.verdict(name, entry), name


def test_a_verdict_that_does_not_follow_from_its_counts_is_reported():
    document = stored()
    entry = document["gates"]["battery_rd"]
    entry["passed"] = not entry["passed"]
    assert f10_certify.problems(document) == [
        "battery_rd: passed does not follow from its counts"
    ]
    document = stored()
    document["terms"] = {**document["terms"], "alpha": 0.1}
    assert f10_certify.problems(document) == [
        "terms: rule or replicate counts changed since measurement"
    ]


def test_a_small_selector_gate_has_the_expected_shape():
    result = f10_certify.gate_selector(30, f10_certify.DEV_SEED_BASE)
    assert result["refuses_when_none_eligible"]["hits"] == 30
    assert result["accepts_when_eligible"]["hits"] == 30
