"""The subgroup certificate is current, honest and wired into --check."""

import json

from benchmarks.qualification import __main__ as qualification
from benchmarks.qualification import subgroup_certify as cert


def stored() -> dict:
    return json.loads(cert.STORE_PATH.read_text(encoding="utf-8"))


def test_the_certificate_matches_the_current_code():
    assert cert.problems() == []


def test_a_missing_or_stale_certificate_is_reported():
    assert cert.problems({"gates": {}}) == [f"{cert.GATE}: no certificate"]
    document = stored()
    document["gates"][cert.GATE]["method_hash"] = "0" * 64
    assert cert.problems(document) == [
        f"{cert.GATE}: stale (method code changed since measurement)"
    ]


def test_the_seeds_do_not_overlap_development_seeds():
    entry = stored()["gates"][cert.GATE]
    assert entry["seeds"][0] == 0 and entry["seeds"][1] < cert.DEV_SEED_BASE


def test_a_missed_floor_is_listed_as_descriptive():
    document = stored()
    document["gates"][cert.GATE]["passed"] = False
    assert cert.descriptive_only(document) == [cert.GATE]


def test_the_default_check_includes_the_subgroup_certificate(capsys):
    qualification.check_certificates()
    report = json.loads(capsys.readouterr().out)
    assert report["subgroup_problems"] == []
    assert "subgroup_descriptive_only" in report
