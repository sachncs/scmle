"""Pins for reconciliation behaviours the main suite left unchecked."""

import pytest

from scmle.contracts import qualification
from scmle.data.admissibility import reconciliation
from scmle.data.detectors import certificates


def explanation(amount=-100.0, attested_by="j.doe", detail="cut-off"):
    return reconciliation.Explanation(
        reconciliation.Mechanism.FILTER, detail, amount, attested_by
    )


def make_item(extract=900.0, source=1000.0, explanations=(), name="rows"):
    return reconciliation.Item(
        name, extract, source, "finance ledger", explanations
    )


def test_constants_and_mechanism_values_are_the_filed_names():
    assert reconciliation.DETECTOR_ID == "f4-reconciliation"
    assert reconciliation.DEFECT_TYPE == "unexplained_gap"
    assert reconciliation.FORMAT_VERSION == 1
    assert [m.value for m in reconciliation.Mechanism] == [
        "filter",
        "late_arrival",
        "dedupe",
    ]
    assert [s.value for s in reconciliation.Status] == [
        "matched",
        "explained",
        "unexplained",
    ]


def test_explanation_round_trips_and_names_each_rejected_field():
    original = explanation()
    record = original.to_dict()
    assert record == {
        "mechanism": "filter",
        "detail": "cut-off",
        "amount": -100.0,
        "attested_by": "j.doe",
    }
    assert reconciliation.Explanation.from_dict(record) == original
    with pytest.raises(ValueError, match=r"^mechanism must be a Mechanism$"):
        reconciliation.Explanation("filter", "d", 1.0)
    with pytest.raises(ValueError, match=r"^detail must not be empty$"):
        explanation(detail="  ")
    with pytest.raises(ValueError, match=r"^attested_by must not be empty$"):
        explanation(attested_by=" ")
    with pytest.raises(ValueError, match=r"^amount must be a finite number$"):
        explanation(amount=float("nan"))
    assert explanation(attested_by=None).attested_by is None


def test_explanation_from_dict_rejects_unknown_and_missing_fields():
    record = explanation().to_dict()
    with pytest.raises(ValueError, match="unexpected field"):
        reconciliation.Explanation.from_dict({**record, "extra": 1})
    del record["detail"]
    with pytest.raises(ValueError, match="missing field"):
        reconciliation.Explanation.from_dict(record)
    with pytest.raises(ValueError):
        reconciliation.Explanation.from_dict(
            {**explanation().to_dict(), "mechanism": "magic"}
        )


def test_item_names_the_field_it_rejects():
    with pytest.raises(ValueError, match=r"^name must not be empty$"):
        reconciliation.Item(" ", 1.0, 1.0, "s")
    with pytest.raises(ValueError, match=r"^source_name must not be empty$"):
        reconciliation.Item("n", 1.0, 1.0, " ")
    with pytest.raises(ValueError, match=r"^extract must be a finite number$"):
        reconciliation.Item("n", float("inf"), 1.0, "s")
    with pytest.raises(ValueError, match=r"^source must be a finite number$"):
        reconciliation.Item("n", 1.0, float("nan"), "s")
    with pytest.raises(
        ValueError, match=r"^source must be non-zero to form a gap ratio$"
    ):
        reconciliation.Item("n", 1.0, 0.0, "s")
    with pytest.raises(ValueError, match="explanations must be a tuple"):
        reconciliation.Item("n", 1.0, 1.0, "s", [explanation()])
    assert reconciliation.Item("n", 0.0, -2.0, "s").gap == 1.0


def test_an_explained_verdict_serialises_every_field():
    item = make_item(explanations=(explanation(),))
    verdict = reconciliation.reconcile_item(item, 0.0)
    assert verdict.to_dict() == {
        "name": "rows",
        "extract": 900.0,
        "source": 1000.0,
        "source_name": "finance ledger",
        "gap": pytest.approx(0.1),
        "residual_gap": 0.0,
        "status": "explained",
        "tier": "attested",
        "explanations": [explanation().to_dict()],
    }


def test_the_check_records_the_statistic_interval_and_seed():
    item = make_item(explanations=(explanation(-40.0),))
    verdict = reconciliation.reconcile_item(
        item,
        0.02,
        certificates.GateStatus.GATING,
        qualification.SetId.ALL,
        (3, "x", 4),
    )
    check = verdict.check
    assert check.check_id == "reconciliation.rows"
    assert check.null_hypothesis == (
        "the extract equals finance ledger within 0.02, or a named "
        "mechanism explains the gap"
    )
    assert check.statistic == pytest.approx(0.1)
    assert check.effect == pytest.approx(0.1)
    assert check.p_value is None
    assert check.n == 1
    assert check.interval == (pytest.approx(0.06), pytest.approx(0.1))
    assert check.seed_path == (3, "x", 4)
    assert check.decision is qualification.Decision.BLOCK
    assert verdict.residual_gap == pytest.approx(0.06)


def test_the_interval_is_ordered_when_the_residual_exceeds_the_gap():
    item = make_item(explanations=(explanation(+100.0),))
    verdict = reconciliation.reconcile_item(item, 0.0)
    low, high = verdict.check.interval
    assert low == pytest.approx(0.1)
    assert high == pytest.approx(0.2)


def test_default_arguments_describe_and_record_a_zero_seed():
    verdict = reconciliation.reconcile_item(make_item())
    assert verdict.check.seed_path == (0, "f4-reconciliation", 0)
    assert verdict.check.set_id is qualification.SetId.ALL
    assert verdict.check.decision is qualification.Decision.WARN


def test_an_explanation_must_close_the_gap_within_the_tolerance():
    partial = make_item(explanations=(explanation(-60.0),))
    assert (
        reconciliation.reconcile_item(partial, 0.039).status
        is reconciliation.Status.UNEXPLAINED
    )
    assert (
        reconciliation.reconcile_item(partial, 0.0405).status
        is reconciliation.Status.EXPLAINED
    )
    boundary = make_item(explanations=(explanation(-50.0),))
    assert (
        reconciliation.reconcile_item(boundary, 0.05).status
        is reconciliation.Status.EXPLAINED
    )


def test_amounts_are_summed_and_every_attester_is_required():
    two = make_item(explanations=(explanation(-60.0), explanation(-40.0, None)))
    verdict = reconciliation.reconcile_item(two)
    assert verdict.status is reconciliation.Status.EXPLAINED
    assert verdict.residual_gap == 0.0
    assert verdict.tier is qualification.Tier.UNVERIFIED
    attested = make_item(
        explanations=(explanation(-60.0), explanation(-40.0, "a"))
    )
    assert (
        reconciliation.reconcile_item(attested).tier
        is qualification.Tier.ATTESTED
    )


def test_an_unexplained_gap_is_unverified_and_blocks_only_when_gating():
    verdict = reconciliation.reconcile_item(
        make_item(), 0.0, certificates.GateStatus.GATING
    )
    assert verdict.tier is qualification.Tier.UNVERIFIED
    assert verdict.check.decision is qualification.Decision.BLOCK
    matched = reconciliation.reconcile_item(
        make_item(1000.0), 0.0, certificates.GateStatus.GATING
    )
    assert matched.check.decision is qualification.Decision.PASS
    assert matched.tier is qualification.Tier.ATTESTED


def test_tolerance_must_be_a_probability():
    with pytest.raises(
        ValueError, match=r"^tolerance must be within \[0, 1\]$"
    ):
        reconciliation.reconcile_item(make_item(), 1.5)
    with pytest.raises(
        ValueError, match=r"^tolerance must be within \[0, 1\]$"
    ):
        reconciliation.reconcile_item(make_item(), -0.1)
    assert reconciliation.reconcile_item(make_item(), 1.0).status is (
        reconciliation.Status.MATCHED
    )


def test_reconcile_numbers_the_seed_paths_and_reports_the_document():
    items = [make_item(name="a"), make_item(1000.0, name="b")]
    result = reconciliation.reconcile(
        items, 0.05, certificates.GateStatus.GATING, qualification.SetId.ALL
    )
    assert [c.seed_path for c in result.checks] == [
        (0, "f4-reconciliation", 0),
        (0, "f4-reconciliation", 1),
    ]
    assert result.passed is False
    document = result.to_dict()
    assert document["format_version"] == 1
    assert document["tolerance"] == 0.05
    assert document["passed"] is False
    assert [r["name"] for r in document["results"]] == ["a", "b"]
    assert [c.decision for c in result.checks] == [
        qualification.Decision.BLOCK,
        qualification.Decision.PASS,
    ]
    ok = reconciliation.reconcile([make_item(1000.0)])
    assert ok.passed is True
    assert ok.to_dict()["passed"] is True


def test_reconcile_rejects_empty_and_duplicate_input_with_its_reason():
    with pytest.raises(ValueError, match=r"^at least one item is required$"):
        reconciliation.reconcile([])
    with pytest.raises(ValueError, match=r"^item names must be unique$"):
        reconciliation.reconcile([make_item(), make_item()])
