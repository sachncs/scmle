"""The scorecard has one row per claim, all from evidence (plan section 5)."""

import json

import pytest

from benchmarks import scorecard
from benchmarks.evidence import coverage
from benchmarks.evidence import faults
from benchmarks.evidence import guarantees
from benchmarks.evidence import mutation
from scmle.contracts import errors


def test_every_claim_has_one_row_in_order():
    found = scorecard.rows()
    assert [r.claim.split()[0] for r in found] == [
        f"C{n}" for n in range(1, 15)
    ]
    assert {r.verdict for r in found} <= {"pass", "fail", "not measured"}


@pytest.mark.parametrize(
    ("verdicts", "expected"),
    [
        (["pass", "pass"], "pass"),
        (["pass", "not measured"], "not measured"),
        (["not measured", "fail", "pass"], "fail"),
        (["pass", "fail"], "fail"),
        ([], "not measured"),
    ],
)
def test_a_claim_passes_only_when_every_clause_passes(verdicts, expected):
    clauses = [scorecard.Clause("c", "v", "e", v) for v in verdicts]
    assert scorecard.Row("C0 x", tuple(clauses)).verdict == expected


def test_every_clause_of_the_plan_is_listed():
    found = {r.claim.split()[0]: r for r in scorecard.rows()}
    # Plan section 3: C12 has three clauses, C3 six, C6 three, C7 six, C1
    # three; one measured clause must not stand for the others.
    counts = {"C1": 3, "C3": 6, "C6": 3, "C7": 6, "C12": 3}
    for claim, count in counts.items():
        assert len(found[claim].clauses) == count, claim


def test_a_measured_clause_cannot_pass_a_claim_alone(monkeypatch):
    monkeypatch.setattr(guarantees, "summary", lambda: ("all linked", "pass"))
    monkeypatch.setattr(coverage, "summary", lambda: ("covered", "pass"))
    found = {r.claim.split()[0]: r for r in scorecard.rows()}
    assert found["C12"].verdict == "not measured"
    assert found["C3"].verdict == "not measured"
    assert "not measured: " in found["C12"].value
    monkeypatch.setattr(guarantees, "summary", lambda: ("1/2 linked", "fail"))
    found = {r.claim.split()[0]: r for r in scorecard.rows()}
    assert found["C12"].verdict == "fail"


def test_unreadable_evidence_is_not_measured(monkeypatch):
    def broken():
        raise errors.InputError("no deciding counts")

    monkeypatch.setattr(guarantees, "summary", broken)
    clause = scorecard.measured("every guarantee linked", guarantees)
    assert clause.verdict == "not measured"
    assert "no deciding counts" in clause.value


def test_measured_rows_come_from_their_evidence():
    found = {r.claim.split()[0]: r for r in scorecard.rows()}
    stems = {
        "C1": "faults",
        "C2": "leakage",
        "C3": "coverage",
        "C4": "quality",
        "C5": "reading",
        "C6": "platforms",
        "C8": "bypass",
        "C7": "budget",
        "C12": "guarantees",
        "C14": "refusal",
    }
    for claim, stem in stems.items():
        assert f"benchmarks/evidence/{stem}.json" in found[claim].evidence
    assert faults.summary()[0] in found["C1"].value
    assert "functions over" in found["C10"].value
    counts = json.loads(scorecard.STRUCTURE.read_text(encoding="utf-8"))
    over = sum(
        counts[k] for k in ("modules_over", "classes_over", "functions_over")
    )
    if over or mutation.summary()[1] == "fail":
        assert found["C10"].verdict == "fail"
    else:
        # Change cost, no test that cannot fail and CI minutes are not
        # measured, so C10 cannot pass yet.
        assert found["C10"].verdict == "not measured"


def test_the_table_renders_every_row():
    text = scorecard.render(scorecard.rows())
    claims = [line for line in text.splitlines() if line[:3] == "| C"]
    assert len(claims) == 15  # the header row and one row per claim
    assert scorecard.main([]) == 0
