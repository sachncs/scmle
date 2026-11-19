"""Claim C1: custody and admissibility faults block at their own gates."""

from benchmarks.evidence import faults


def test_the_quarantine_site_reaches_its_gate_through_the_services():
    """The stand-in context once lacked `services` (AttributeError)."""
    record = faults.attempt("admissibility.quarantine_all", "fault")
    assert (record["outcome"], record.get("by")) == (
        "blocked",
        "admissibility.quarantine_all",
    )
    override = faults.attempt("admissibility.quarantine_all", "override")
    assert override["outcome"] == "override"


def test_a_foreign_reuse_journal_blocks_a_real_charter_run():
    record = faults.attempt("confirm.reuse_journal", "fault", "real")
    assert (record["outcome"], record.get("by"), record["level"]) == (
        "blocked",
        "confirm.reuse_journal",
        "run",
    )
    control = faults.attempt("confirm.reuse_journal", "control", "real")
    assert control["outcome"] == "passed"
