"""The scorecard has one row per claim, all from evidence (plan section 5)."""

from benchmarks import scorecard


def test_every_claim_has_one_row_in_order():
    found = scorecard.rows()
    assert [r.claim.split()[0] for r in found] == [
        f"C{n}" for n in range(1, 15)
    ]
    assert {r.verdict for r in found} <= {"pass", "fail", "not measured"}


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
        assert found[claim].evidence == f"benchmarks/evidence/{stem}.json"
    assert "injected faults" in found["C1"].value
    assert "functions over" in found["C10"].value
    assert found["C10"].verdict == "fail"


def test_the_table_renders_every_row():
    text = scorecard.render(scorecard.rows())
    claims = [line for line in text.splitlines() if line[:3] == "| C"]
    assert len(claims) == 15  # the header row and one row per claim
    assert scorecard.main([]) == 0
