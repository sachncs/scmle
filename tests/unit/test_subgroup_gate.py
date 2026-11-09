"""The subgroup family gates or only describes, by its certificate."""

import json

from scmle.data.identification import certified
from scmle.data.identification import run
from tests.unit.test_identification_run import BUDGET
from tests.unit.test_identification_run import GATES
from tests.unit.test_identification_run import NAMED
from tests.unit.test_identification_subgroups import go
from tests.unit.test_identification_subgroups import subgroup_spec
from tests.unit.test_identification_subgroups import world

EFFECTS = {"g1": 2.0, "g2": 0.0, "g3": 0.0}


def certificate(path, **entry):
    body = {
        "method_hash": certified.subgroup_method_hash(run.subgroup_report),
        "passed": True,
        "min_certified_rows": 200,
    }
    body.update(entry)
    path.write_text(json.dumps({"gates": {certified.SUBGROUP_GATE: body}}))
    return path


def test_without_a_certificate_the_family_is_descriptive(tmp_path):
    gate = certified.subgroup_gate(run.subgroup_report, tmp_path / "none.json")
    assert gate.status == certified.DESCRIPTIVE
    assert "no subgroup certificate" in gate.reason


def test_a_passing_current_certificate_makes_the_family_gating(tmp_path):
    gate = certified.subgroup_gate(
        run.subgroup_report, certificate(tmp_path / "c.json")
    )
    assert gate.status == certified.GATING and gate.min_rows == 200


def test_a_failed_or_stale_certificate_is_descriptive(tmp_path):
    missed = certificate(tmp_path / "m.json", passed=False)
    stale = certificate(tmp_path / "s.json", method_hash="0" * 64)
    assert (
        certified.subgroup_gate(run.subgroup_report, missed).status
        == certified.DESCRIPTIVE
    )
    gate = certified.subgroup_gate(run.subgroup_report, stale)
    assert gate.status == certified.DESCRIPTIVE and "stale" in gate.reason


def test_the_report_carries_the_gate_and_the_certified_size():
    spec = subgroup_spec(EFFECTS)
    x, c = world(1500, 1, EFFECTS), world(1500, 2, EFFECTS)
    out = go(spec, x, c)
    found = out.documents["identification_report.json"]["subgroups"]
    assert found["gate"]["status"] == certified.DESCRIPTIVE
    summary = out.documents["causal_claims.json"]["subgroups"]
    assert summary["gate"] == certified.DESCRIPTIVE
    assert "descriptive" in found["statement_gate"]


def test_naming_the_gate_among_the_certified_ones_makes_it_gating():
    spec = subgroup_spec(EFFECTS)
    x, c = world(1500, 1, EFFECTS), world(1500, 2, EFFECTS)
    out = run.identify(
        spec,
        x,
        c,
        BUDGET,
        attestors=NAMED,
        certified=(*GATES, certified.SUBGROUP_GATE),
        seed=0,
    )
    found = out.documents["identification_report.json"]["subgroups"]
    assert found["gate"]["status"] == certified.GATING
    sizes = {r["value"]: r["certified_size"] for r in found["hypotheses"]}
    assert sizes == {"g1": True, "g2": True, "g3": True}  # about 500 rows each


def test_the_gate_never_changes_the_tier():
    spec = subgroup_spec(EFFECTS)
    x, c = world(1500, 1, EFFECTS), world(1500, 2, EFFECTS)
    plain = go(spec, x, c)
    gated = run.identify(
        spec,
        x,
        c,
        BUDGET,
        attestors=NAMED,
        certified=(*GATES, certified.SUBGROUP_GATE),
        seed=0,
    )
    assert plain.decision == gated.decision
