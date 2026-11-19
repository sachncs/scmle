"""Single-word keys on the write-only assurance records (phase 5, F1).

Each record writes its new keys, and its `migrations` move the layout the
previous writer produced (spelled out here by hand) to the current one and
back. Records that carry a version say so; the framing draft's digest is
computed on its version-1 layout, so a registered draft keeps its hash.
"""

from fractions import Fraction
import hashlib

import pandas
import pytest

from scmle.assurance import coverage_slices
from scmle.assurance.admissibility import reconciliation
from scmle.assurance.admissibility import rules
import scmle.assurance.admissibility.screen.report
from scmle.assurance.detectors import duplicates
from scmle.assurance.detectors import labels
from scmle.assurance.detectors import missingness
from scmle.assurance.framing import baseline
from scmle.assurance.framing import claim
from scmle.assurance.framing import draft
from scmle.assurance.framing import feasibility
from scmle.assurance.framing import questions
from scmle.assurance.framing import refusal
from scmle.assurance.ingest import budget
from scmle.assurance.ingest import grain
from scmle.assurance.ingest import joins
from scmle.assurance.ingest import profile
from scmle.assurance.ingest import reconcile
from scmle.assurance.ingest.entities import resolution
from scmle.assurance.ingest.synthesize import selection
from scmle.assurance.ledger import confirm
from scmle.assurance.robustness import coverage
from scmle.assurance.robustness import forward
from scmle.assurance.robustness import shift
import scmle.assurance.robustness.report
from scmle.assurance.robustness.stability import estimation
from scmle.assurance.robustness.stability import permutation
from scmle.contracts import records
from scmle.contracts import signing
from tests.unit.test_framing_baseline_mutants import guard
from tests.unit.test_framing_baseline_mutants import run
from tests.unit.test_framing_baseline_mutants import world
from tests.unit.test_framing_draft import MODEL
from tests.unit.test_framing_draft import request
from tests.unit.test_framing_draft import table


def moves(kind, old, new):
    """Asserts that `kind`'s single step maps `old` to `new` and back.

    Args:
        kind: A record class with a one-step `migrations`.
        old: The layout the previous writer produced.
        new: The layout the current writer produces, without a version.
    """
    assert len(kind.migrations) == 1
    assert records.upgrade(kind.migrations, dict(old), 1) == new
    assert records.downgrade(kind.migrations, dict(new)) == old


def bare(layout):
    """The layout without its version key."""
    return {k: v for k, v in layout.items() if k != records.KEY}


def test_a_flag_gate_writes_rows():
    gate = coverage_slices.Gate("gating", "certified", 100)
    new = {"status": "gating", "reason": "certified", "rows": 100}
    assert gate.to_dict() == new
    moves(
        coverage_slices.Gate,
        {"status": "gating", "reason": "certified", "min_rows": 100},
        new,
    )


def test_duplicates_group_the_exact_counts():
    found = duplicates.Duplicates(10, 1, 0.1, (0.0, 0.4), 2, 0.01, 3, 1)
    grouped = {
        "rows": 10,
        "cut": 0.01,
        "groups": 1,
        "near": 2,
        "conflicts": 3,
        "exact": {"extra": 1, "rate": 0.1, "interval": (0.0, 0.4)},
    }
    text = {"near": 0, "threshold": duplicates.DEFAULT_SIMILARITY}
    new = {**grouped, "text": text}
    assert found.to_dict() == new
    old = {
        "rows": 10,
        "exact_extra": 1,
        "exact_rate": 0.1,
        "exact_interval": (0.0, 0.4),
        "near_extra": 2,
        "cut": 0.01,
        "label_conflicts": 3,
        "groups": 1,
    }
    first, second = duplicates.Duplicates.migrations
    assert first.upgrade(dict(old)) == grouped
    assert first.downgrade(dict(grouped)) == old
    # Version 3 gathers the MinHash text fields under `text`.
    middle = {**grouped, "similar": 0, "similarity": text["threshold"]}
    assert second.upgrade(dict(middle)) == new
    assert second.downgrade(dict(new)) == middle
    migrations = duplicates.Duplicates.migrations
    assert records.upgrade(migrations, dict(middle), 2) == new
    assert records.downgrade(migrations, dict(new), 2) == middle


def test_label_noise_and_missingness_write_single_words():
    rate = labels.Rate(0.1, (0.0, 0.2), "detector_only", 0.95, "u", "c", 0)
    noise = labels.Noise((1,), (1, 2), (1,), 0.5, 5, 6, 0.3, rate, (0.5,))
    written = noise.to_dict()
    assert {"candidates", "checked", "received", "bound"} <= set(written)
    assert written["candidates"] == (1,) and written["bound"] == 0.3
    old = records.downgrade(labels.Noise.migrations, written)
    assert old["confident_learning"] == (1,)
    assert (old["rows_checked"], old["input_rows"]) == (5, 6)
    assert old["sampling_bound"] == 0.3
    gaps = missingness.Missingness(5, 6, None, (), True, 0.05, ())
    assert gaps.to_dict() == {
        "checked": 5,
        "received": 6,
        "little": None,
        "indicators": (),
        "rejected": True,
        "alpha": 0.05,
        "skipped": (),
        "tier": "unverified",
    }
    old = records.downgrade(missingness.Missingness.migrations, gaps.to_dict())
    assert old["mcar_rejected"] is True and old["input_rows"] == 6


def test_reconciliation_results_write_residual_at_format_three():
    item = reconciliation.Item("rows", 900.0, 1000.0, "ledger")
    verdict = reconciliation.settle(item, 0.2)
    written = verdict.to_dict()
    assert written["residual"] == verdict.residual
    assert "residual_gap" not in written
    assert (
        1 + len(reconciliation.Verdict.migrations)
        == reconciliation.FORMAT_VERSION
    )
    old = records.downgrade(reconciliation.Verdict.migrations, written, 2)
    assert old["residual_gap"] == verdict.residual
    report = reconcile.Report(reconciliation.reconcile([item], 0.2), ("x",))
    document = report.to_dict()
    assert document["unreconciled"] == ["x"]
    assert document["format_version"] == reconciliation.FORMAT_VERSION
    assert "not_reconciled" not in document
    moves(
        reconcile.Report,
        {"not_reconciled": ["x"], "passed": False},
        {"unreconciled": ["x"], "passed": False},
    )


def test_the_admissibility_table_nests_the_dag_digest():
    table_ = rules.Admissibility("causal", rules.Mode.STRICT, (), "d", ("z",))
    current = {
        "format_version": 3,
        "kind": "causal",
        "mode": "strict",
        "adjustment": ["z"],
        "decisions": [],
        "digests": {"dag": "d"},
    }
    assert table_.to_dict() == current
    assert 1 + len(rules.Admissibility.migrations) == rules.FORMAT_VERSION
    second = {
        "format_version": 3,
        "kind": "causal",
        "mode": "strict",
        "dag_hash": "d",
        "adjustment_set": ["z"],
        "decisions": [],
    }
    assert records.upgrade(rules.Admissibility.migrations, second, 2) == (
        current
    )


def test_an_abstaining_screen_keeps_its_rationale():
    screen = scmle.assurance.admissibility.screen.report.Screen(
        (), 3, 0, 0, 0.05, {}, {}, (), ("a",), "too few rows"
    )
    assert screen.rationale == "too few rows"
    assert "(too few rows)" in screen.warning
    assert "rationale" not in screen.to_dict()


def test_the_baseline_report_nests_its_registration_digest():
    y, a, b, cluster = world(30)
    evaluator = guard()
    outcome = run(evaluator, y, a, b, cluster)
    document = outcome.to_dict()
    assert document[records.KEY] == 1 + len(baseline.Outcome.migrations) == 2
    digest = evaluator.registration.digest
    assert document["digests"] == {"registration": digest}
    assert set(document["selective_labels"]["missing"]) == {
        "control",
        "treated",
    }
    old = records.downgrade(baseline.Outcome.migrations, bare(document))
    assert old["registration_hash"] == digest
    assert "missing_rate_b0" in old["selective_labels"]
    stored = {**old, records.KEY: 1}
    assert records.load("Outcome", baseline.Outcome.migrations, stored) == (
        document
    )


def test_a_refused_baseline_report_is_written_at_version_two():
    found = baseline.screen([True] * 20 + [False] * 20, [0, 1] * 20)
    registration = guard().registration
    document = claim.refused(registration, found.to_dict())
    assert document[records.KEY] == 2
    assert document["digests"] == {"registration": registration.digest}
    assert "registration_hash" not in document


def test_the_framing_draft_hash_is_anchored_at_version_one():
    frame = table()
    result = draft.frame(request(frame), frame, cost_model=MODEL)
    document = result.to_dict()
    assert document[records.KEY] == 2
    assert document["pending"] == list(result.pending)
    assert "open_items" not in document
    anchor = result.anchor()
    assert anchor[records.KEY] == 1
    assert anchor["open_items"] == list(result.pending)
    assert "pending" not in anchor
    # The hash a version-1 writer produced: its own document, with the
    # charter in the charter's version-1 layout.
    written = {
        **{k: v for k, v in document.items() if k != "pending"},
        records.KEY: 1,
        "open_items": document["pending"],
        "charter": result.charter.anchor(),
    }
    assert (
        result.sha == hashlib.sha256(signing.canonicalise(written)).hexdigest()
    )


def test_the_framing_records_write_single_words():
    verdict = feasibility.Verdict(
        "feasible", "confirmatory", 1.0, (0.5, 2.0), 0.1, 1.5, 10, 20.0, "r"
    )
    moves(
        feasibility.Verdict,
        {
            "verdict": "feasible",
            "scope": "confirmatory",
            "s_d": 1.0,
            "s_d_interval": [0.5, 2.0],
            "rho": 0.1,
            "design_effect": 1.5,
            "n_required": 10,
            "n_effective": 20.0,
            "reason": "r",
        },
        verdict.to_dict(),
    )
    assert verdict.to_dict()["interval"] == [0.5, 2.0]
    question = questions.Question("s", "f", 0.0, 1.0, 0.5, ("a", "b"))
    moves(
        questions.Question,
        {
            "slot": "s",
            "form": "f",
            "low": 0.0,
            "high": 1.0,
            "tipping_value": 0.5,
            "verdicts": ["a", "b"],
        },
        question.to_dict(),
    )
    assert question.to_dict()["tipping"] == 0.5
    calibration = refusal.Calibration(0.2, 0.05, 40, ())
    moves(
        refusal.Calibration,
        {"tau": 0.2, "epsilon": 0.05, "n_calibration": 40, "risk_coverage": []},
        calibration.to_dict(),
    )
    assert calibration.to_dict()["calibration"] == 40


def test_the_ingest_budget_uses_the_charter_spelling():
    limits = budget.Budget(1, 2, 3.0, 4)
    new = {"queries": 1, "scanned": 2, "seconds": 3.0, "features": 4}
    assert limits.to_dict() == new
    moves(
        budget.Budget,
        {
            "max_queries": 1,
            "max_rows_scanned": 2,
            "max_seconds": 3.0,
            "max_features": 4,
        },
        new,
    )


def test_entity_resolution_nests_its_thresholds():
    result = resolution.isolate(3, 2, 1.0, 4)
    written = result.to_dict()
    assert written["thresholds"] == {
        "group": result.group,
        "linkage": result.t_match,
    }
    assert (written["reduction"], written["skipped"]) == (1.0, 4)
    old = records.downgrade(resolution.Resolution.migrations, written)
    assert (old["t_group"], old["t_match"]) == (result.group, result.t_match)
    assert (old["reduction_ratio"], old["skipped_blocks"]) == (1.0, 4)
    assert records.upgrade(resolution.Resolution.migrations, old, 1) == written


def test_a_grain_scan_writes_tolerances_and_its_keys_at_version_two():
    key = grain.Candidate(("a",), 0.0, 0.1, 5, 10, True)
    scan = grain.Scan("t", 10, [key], {}, 2, False, 0.0, 0.05)
    written = scan.to_dict()
    assert written["keys"] == [
        {
            "columns": ["a"],
            "g3": 0.0,
            "nulls": 0.1,
            "distinct": 5,
            "rows": 10,
            "small": True,
        }
    ]
    assert written["levels"] == 2
    assert written["tolerances"] == {"key": 0.0, "null": 0.05}
    old = records.downgrade(grain.Scan.migrations, written)
    assert old["keys"][0]["null_rate"] == 0.1 and old["keys"][0]["small_n"]
    assert (old["levels_done"], old["eps_key"], old["eps_null"]) == (
        2,
        0.0,
        0.05,
    )
    assert records.upgrade(grain.Scan.migrations, old, 1) == written
    unit = grain.Verdict("unknown", "u", ("a",), "why")
    assert unit.to_dict()["columns"] == ["a"]
    assert "unit_columns" not in unit.to_dict()


def test_a_link_nests_its_columns():
    link = joins.Link(
        "c",
        ("a",),
        "p",
        ("b",),
        1.0,
        3,
        "many_to_one",
        False,
        True,
        0.5,
        "accepted",
        "w",
    )
    new = {
        "child": "c",
        "parent": "p",
        "containment": 1.0,
        "distinct": 3,
        "cardinality": "many_to_one",
        "declared": False,
        "named": True,
        "score": 0.5,
        "decision": "accepted",
        "why": "w",
        "joinable": True,
        "columns": {"child": ["a"], "parent": ["b"]},
    }
    assert link.to_dict() == new
    moves(
        joins.Link,
        {
            "child": "c",
            "child_columns": ["a"],
            "parent": "p",
            "parent_columns": ["b"],
            "containment": 1.0,
            "child_distinct": 3,
            "cardinality": "many_to_one",
            "declared": False,
            "name_support": True,
            "score": 0.5,
            "decision": "accepted",
            "why": "w",
            "joinable": True,
        },
        new,
    )


def test_aggregates_and_the_feature_set_write_single_words():
    column = profile.Aggregates("a", "number", 3, 2, "1", "2")
    moves(
        profile.Aggregates,
        {
            "name": "a",
            "kind": "number",
            "non_null": 3,
            "distinct": 2,
            "minimum": "1",
            "maximum": "2",
        },
        column.to_dict(),
    )
    kept = selection.Selection(pandas.DataFrame(), ({"kept": True},), 1)
    moves(
        selection.Selection,
        {
            "declared_family_size": 1,
            "proposed": 1,
            "features": [{"kept": True}],
        },
        kept.to_dict(),
    )


def confirmation(**changes):
    """A confirmation record with every optional part filled."""
    fields = {
        "outcome": confirm.NO_SURVIVORS,
        "reason": "r",
        "claims": (),
        "checks": (),
        "rule": "bh",
        "confirm": Fraction(1, 20),
        "survivors_sha256": "s",
        "ledger_head": "l",
        "head": "c",
        "control": None,
        "second": {"family": []},
    }
    return confirm.Confirmation(**{**fields, **changes})


def test_the_claims_file_groups_digests_and_heads_at_version_two():
    written = confirmation().to_dict()
    assert written[records.KEY] == 2
    assert written["alpha"] == "1/20"
    assert written["digests"] == {"survivors": "s"}
    assert written["heads"] == {"ledger": "l", "custody": "c"}
    assert written["second"] == {"family": []}
    old = records.downgrade(confirm.Confirmation.migrations, bare(written))
    assert old["alpha_confirm"] == "1/20"
    assert old["survivors_sha256"] == "s"
    assert (old["ledger_head"], old["custody_head"]) == ("l", "c")
    assert old["stage_two"] == {"family": []}
    # A version-1 file carries no version key and still loads.
    assert records.load(
        "Confirmation", confirm.Confirmation.migrations, old
    ) == (written)
    plain = confirmation(second=None, head=None).to_dict()
    assert "second" not in plain and plain["heads"]["custody"] is None


def test_the_robustness_records_write_single_words():
    cell = coverage.Cell(
        "c", "l", 1, 10, 0.1, (0.0, 0.4), 0.2, shift.Status.PASS
    )
    assert cell.to_dict()["deployment"] == 0.2
    assert "deployment_share" not in cell.to_dict()
    finding = forward.Finding("f", 0.3, None, forward.INCONCLUSIVE, None)
    assert finding.to_dict()["effects"] == {"observed": 0.3, "forward": None}
    old = records.downgrade(forward.Finding.migrations, finding.to_dict())
    assert (old["effect_x"], old["effect_f"]) == (0.3, None)
    drift = shift.Drift("c", "numeric", 0.1, 0.5, 0.5, 0.1, False, True)
    assert drift.to_dict()["sparse"] is True
    moves(
        estimation.Thresholds,
        {"s_min": 0.8, "band": 0.5, "q_min": 0.8},
        {"agreement": 0.8, "band": 0.5, "exclusion": 0.8},
    )
    assert estimation.Thresholds().to_dict() == {
        "agreement": 0.8,
        "band": 0.5,
        "exclusion": 0.8,
    }
    curve = permutation.Curve(10, 0.3, 0.5, 0.1, 0.2)
    new = {
        "permutations": 10,
        "median": 0.3,
        "significant": 0.5,
        "pvalues": {"median": 0.1, "share": 0.2},
    }
    assert curve.to_dict() == new
    moves(
        permutation.Curve,
        {
            "permutations": 10,
            "median_abs_effect": 0.3,
            "share_significant": 0.5,
            "p_median": 0.1,
            "p_share": 0.2,
        },
        new,
    )


@pytest.mark.parametrize("version", [1, 2])
def test_robustness_reports_of_either_version_are_accepted(version):
    assert scmle.assurance.robustness.report.SCHEMA_VERSION == 2
    scmle.assurance.robustness.report.require(
        {"schema_version": version}, (), "report"
    )
