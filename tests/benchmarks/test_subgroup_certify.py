"""The subgroup certificate is current, honest and wired into --check."""

import json

import pytest
from scipy import stats

from benchmarks import acceptance
from benchmarks.qualification import __main__
from benchmarks.qualification import subgroup_certify
import scmle.causal.identification.certified


def stored() -> dict:
    return json.loads(subgroup_certify.STORE_PATH.read_text(encoding="utf-8"))


def test_the_certificate_matches_the_current_code():
    assert subgroup_certify.problems() == []


def test_a_missing_or_stale_certificate_is_reported():
    assert subgroup_certify.problems({"gates": {}}) == [
        f"{scmle.causal.identification.certified.SUBGROUP_GATE}: no certificate"
    ]
    document = stored()
    document["gates"][scmle.causal.identification.certified.SUBGROUP_GATE][
        "method_hash"
    ] = "0" * 64
    assert subgroup_certify.problems(document) == [
        f"{scmle.causal.identification.certified.SUBGROUP_GATE}: stale (method code changed since measurement)"
    ]


def test_the_seeds_do_not_overlap_development_seeds():
    entry = stored()["gates"][
        scmle.causal.identification.certified.SUBGROUP_GATE
    ]
    assert (
        entry["seeds"][0] == 0
        and entry["seeds"][1] < subgroup_certify.DEV_SEED_BASE
    )


def test_a_missed_floor_is_listed_as_descriptive():
    document = stored()
    document["gates"][scmle.causal.identification.certified.SUBGROUP_GATE][
        "passed"
    ] = False
    assert subgroup_certify.descriptive_only(document) == [
        scmle.causal.identification.certified.SUBGROUP_GATE
    ]


def test_the_default_check_includes_the_subgroup_certificate(capsys):
    __main__.check_certificates()
    report = json.loads(capsys.readouterr().out)
    assert report["subgroup_problems"] == []
    assert "subgroup_descriptive_only" in report


def test_error_rates_are_decided_by_the_exact_upper_bound_at_alpha():
    alpha = acceptance.load()["alpha"].value
    assert alpha == subgroup_certify.ALPHA
    row = subgroup_certify.rate(111, 2000)
    assert row["exact"][1] == pytest.approx(
        stats.beta.ppf(1 - alpha, 112, 1889)
    )
    # 0.0555: its bound 0.0647 is above the level 0.05 but within the size
    # ceiling 0.05 (1 + size_margin); a family at twice its level is not
    assert subgroup_certify.controlled(row)
    assert subgroup_certify.controlled(subgroup_certify.rate(70, 2000))
    assert not subgroup_certify.controlled(subgroup_certify.rate(200, 2000))


def test_power_is_decided_by_the_exact_lower_bound_at_alpha():
    alpha = acceptance.load()["alpha"].value
    row = subgroup_certify.rate(382, 500)
    assert row["exact"][0] == pytest.approx(stats.beta.ppf(alpha, 382, 119))
    assert not subgroup_certify.powered(row)
    assert subgroup_certify.powered(subgroup_certify.rate(493, 500))


def test_the_stored_verdict_follows_from_the_stored_counts():
    entry = stored()["gates"][
        scmle.causal.identification.certified.SUBGROUP_GATE
    ]
    assert entry["terms"] == subgroup_certify.terms()
    assert entry["replicates"] == subgroup_certify.NULL_REPLICATES
    assert entry["power_replicates"] == subgroup_certify.POWER_REPLICATES
    assert entry["seeds"] == [0, subgroup_certify.NULL_REPLICATES]
    sizes = [
        int(k)
        for k, row in entry["power_at_effect_half_sd"].items()
        if subgroup_certify.powered(row)
    ]
    assert entry["min_certified_rows"] == (min(sizes) if sizes else None)
    expected = (
        subgroup_certify.controlled(entry["familywise_error_null"])
        and subgroup_certify.controlled(entry["familywise_error_partial_null"])
        and bool(sizes)
    )
    assert entry["passed"] is expected


def test_changed_terms_make_the_certificate_stale():
    document = stored()
    entry = document["gates"][
        scmle.causal.identification.certified.SUBGROUP_GATE
    ]
    entry["terms"] = {**entry["terms"], "familywise_ceiling": 0.06}
    gate = scmle.causal.identification.certified.SUBGROUP_GATE
    reason = "rule or replicate count changed since measurement"
    assert subgroup_certify.problems(document) == [f"{gate}: stale ({reason})"]
