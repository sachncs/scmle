"""The subgroup certificate is current, honest and wired into --check."""

import json

from benchmarks.qualification import __main__
from benchmarks.qualification import subgroup_certify
import scmle.data.identification.certified


def stored() -> dict:
    return json.loads(subgroup_certify.STORE_PATH.read_text(encoding="utf-8"))


def test_the_certificate_matches_the_current_code():
    assert subgroup_certify.problems() == []


def test_a_missing_or_stale_certificate_is_reported():
    assert subgroup_certify.problems({"gates": {}}) == [
        f"{scmle.data.identification.certified.SUBGROUP_GATE}: no certificate"
    ]
    document = stored()
    document["gates"][scmle.data.identification.certified.SUBGROUP_GATE][
        "method_hash"
    ] = "0" * 64
    assert subgroup_certify.problems(document) == [
        f"{scmle.data.identification.certified.SUBGROUP_GATE}: stale (method code changed since measurement)"
    ]


def test_the_seeds_do_not_overlap_development_seeds():
    entry = stored()["gates"][scmle.data.identification.certified.SUBGROUP_GATE]
    assert (
        entry["seeds"][0] == 0
        and entry["seeds"][1] < subgroup_certify.DEV_SEED_BASE
    )


def test_a_missed_floor_is_listed_as_descriptive():
    document = stored()
    document["gates"][scmle.data.identification.certified.SUBGROUP_GATE][
        "passed"
    ] = False
    assert subgroup_certify.descriptive_only(document) == [
        scmle.data.identification.certified.SUBGROUP_GATE
    ]


def test_the_default_check_includes_the_subgroup_certificate(capsys):
    __main__.check_certificates()
    report = json.loads(capsys.readouterr().out)
    assert report["subgroup_problems"] == []
    assert "subgroup_descriptive_only" in report
