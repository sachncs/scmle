"""Reconciliation gate: every gap needs a named, quantified explanation."""

import pytest

from scmle.contracts.qualification import Decision
from scmle.contracts.qualification import Tier
from scmle.data.admissibility import reconciliation
from scmle.data.detectors import certificates


def item(extract, source=1000.0, explanations=(), name="rows"):
    return reconciliation.Item(
        name, extract, source, "finance ledger", explanations
    )


def test_gap_is_the_relative_absolute_difference():
    assert item(950.0).gap == pytest.approx(0.05)
    assert item(1050.0).gap == pytest.approx(0.05)
    assert item(500.0, 2000.0).gap == pytest.approx(0.75)
    assert item(-90.0, -100.0).gap == pytest.approx(0.1)  # |source| in divisor


def test_an_exact_match_passes_without_an_explanation():
    result = reconciliation.reconcile_item(item(1000.0))
    assert result.status is reconciliation.Status.MATCHED
    assert result.tier is Tier.ATTESTED
    assert result.check.decision is Decision.PASS


def test_a_gap_exactly_at_the_tolerance_matches_and_above_it_does_not():
    assert (
        reconciliation.reconcile_item(item(950.0), 0.05).status
        is reconciliation.Status.MATCHED
    )
    over = reconciliation.reconcile_item(
        item(949.9), 0.05, certificates.GateStatus.GATING
    )
    assert over.status is reconciliation.Status.UNEXPLAINED
    assert over.check.decision is Decision.BLOCK


def test_an_unexplained_gap_fails_the_gate_and_blocks_when_certified():
    result = reconciliation.reconcile_item(
        item(900.0), gate=certificates.GateStatus.GATING
    )
    assert result.status is reconciliation.Status.UNEXPLAINED
    assert result.check.decision is Decision.BLOCK
    assert result.tier is Tier.UNVERIFIED
    # not certified: described, never blocking
    soft = reconciliation.reconcile_item(item(900.0))
    assert soft.check.decision is Decision.WARN


def test_an_explained_gap_passes_with_the_explanation_recorded():
    why = reconciliation.Explanation(
        reconciliation.Mechanism.FILTER,
        "test accounts removed",
        -100.0,
        "j.doe",
    )
    result = reconciliation.reconcile_item(
        item(900.0, explanations=(why,)), gate=certificates.GateStatus.GATING
    )
    assert result.status is reconciliation.Status.EXPLAINED
    assert result.residual_gap == pytest.approx(0.0)
    assert result.tier is Tier.ATTESTED
    assert result.check.decision is Decision.PASS
    assert result.to_dict()["explanations"][0]["mechanism"] == "filter"


def test_an_explanation_must_cover_the_gap_within_the_tolerance():
    partial = reconciliation.Explanation(
        reconciliation.Mechanism.DEDUPE, "duplicate orders", -60.0
    )
    result = reconciliation.reconcile_item(
        item(900.0, explanations=(partial,)),
        gate=certificates.GateStatus.GATING,
    )
    assert result.status is reconciliation.Status.UNEXPLAINED
    assert result.residual_gap == pytest.approx(0.04)
    lenient = reconciliation.reconcile_item(
        item(900.0, explanations=(partial,)), 0.04
    )
    assert (
        lenient.status is reconciliation.Status.EXPLAINED
    )  # residual == tolerance


def test_several_mechanisms_add_up_with_their_signs():
    late = reconciliation.Explanation(
        reconciliation.Mechanism.LATE_ARRIVAL, "batch after cutoff", -70.0
    )
    dup = reconciliation.Explanation(
        reconciliation.Mechanism.DEDUPE, "double posted", 20.0
    )
    filt = reconciliation.Explanation(
        reconciliation.Mechanism.FILTER, "voided", -50.0
    )
    result = reconciliation.reconcile_item(
        item(900.0, explanations=(late, dup, filt))
    )
    assert result.status is reconciliation.Status.EXPLAINED
    # sum of amounts is -100, the observed difference
    over = reconciliation.reconcile_item(item(900.0, explanations=(late, filt)))
    assert over.status is reconciliation.Status.UNEXPLAINED


def test_an_explanation_that_overshoots_is_not_accepted():
    big = reconciliation.Explanation(
        reconciliation.Mechanism.FILTER, "everything", -400.0
    )
    assert (
        reconciliation.reconcile_item(item(900.0, explanations=(big,))).status
        is reconciliation.Status.UNEXPLAINED
    )


def test_unattested_explanations_leave_the_tier_unverified():
    why = reconciliation.Explanation(
        reconciliation.Mechanism.FILTER, "voided", -100.0
    )
    result = reconciliation.reconcile_item(item(900.0, explanations=(why,)))
    assert result.status is reconciliation.Status.EXPLAINED
    assert result.tier is Tier.UNVERIFIED


def test_inputs_are_validated():
    with pytest.raises(ValueError, match="non-zero"):
        reconciliation.Item("n", 1.0, 0.0, "s")
    with pytest.raises(ValueError, match="source_name"):
        reconciliation.Item("n", 1.0, 1.0, " ")
    with pytest.raises(ValueError):
        reconciliation.Item("n", float("nan"), 1.0, "s")
    with pytest.raises(ValueError, match="detail"):
        reconciliation.Explanation(reconciliation.Mechanism.FILTER, " ", 1.0)
    with pytest.raises(ValueError, match="Mechanism"):
        reconciliation.Explanation("filter", "x", 1.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        reconciliation.reconcile_item(item(1.0), tolerance=1.5)
    with pytest.raises(ValueError, match="at least one"):
        reconciliation.reconcile([])
    with pytest.raises(ValueError, match="unique"):
        reconciliation.reconcile([item(1000.0), item(1000.0)])


def test_the_dataset_gate_fails_on_any_unexplained_item():
    why = reconciliation.Explanation(
        reconciliation.Mechanism.FILTER, "voided", -100.0, "j.doe"
    )
    good = reconciliation.reconcile(
        [item(1000.0, name="a"), item(900.0, explanations=(why,), name="b")]
    )
    assert good.passed
    bad = reconciliation.reconcile(
        [item(1000.0, name="a"), item(800.0, name="b")]
    )
    assert not bad.passed
    assert bad.to_dict()["passed"] is False
    assert len(bad.checks) == 2
    assert {c.seed_path[2] for c in bad.checks} == {0, 1}
