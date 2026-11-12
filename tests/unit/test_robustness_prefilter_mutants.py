"""Targeted tests for the stability stage's records, notes and checks."""

import types

from scmle.contracts import qualification
from scmle.core import randomness
from scmle.data.detectors import certificates
from scmle.data.robustness import prefilter
from scmle.data.robustness import stability
from tests.unit.test_robustness_prefilter import run


def make_row(column, rejected):
    return types.SimpleNamespace(
        column=column,
        statistic=1.5,
        p_value=0.25,
        effect=0.5,
        rejected=rejected,
    )


def test_feature_checks_put_rejected_columns_first_and_keep_their_seed_index():
    shift_result = types.SimpleNamespace(
        features=(
            make_row("a", False),
            make_row("b", True),
            make_row("c", False),
        )
    )

    checks = prefilter.feature_checks(shift_result, rows=40, seed=7)

    assert [c.check_id for c in checks] == [
        "f6-shift-feature:b",
        "f6-shift-feature:a",
        "f6-shift-feature:c",
    ]
    assert [c.decision for c in checks] == [
        qualification.Decision.WARN,
        qualification.Decision.PASS,
        qualification.Decision.PASS,
    ]
    assert [c.seed_path for c in checks] == [
        randomness.seed_path(7, prefilter.FEATURE_SHIFT_ID, 1),
        randomness.seed_path(7, prefilter.FEATURE_SHIFT_ID, 0),
        randomness.seed_path(7, prefilter.FEATURE_SHIFT_ID, 2),
    ]
    assert all(
        c.n == 40 and c.set_id is qualification.SetId.EXPLORATION
        for c in checks
    )


def test_feature_checks_are_capped():
    rows = tuple(make_row(f"c{i}", False) for i in range(300))
    checks = prefilter.feature_checks(
        types.SimpleNamespace(features=rows), rows=1, seed=0
    )
    assert len(checks) == prefilter.MAX_FEATURE_ROWS


def test_the_applied_stage_records_every_field_and_the_fragile_reason():
    result, stage = run(True)

    record = result.confirmation.stage_two["rule"]
    found = stage.found
    assert record["name"] == prefilter.FILTER_ID
    assert record["requested"] is True
    assert record["applied"] is True
    assert record["gate"] == certificates.GateStatus.GATING.value
    assert record["thresholds"] == stability.DEFAULT_RULE.to_dict()
    assert record["assessed"] == [f.name for f in found.findings]
    assert record["fragile"] == sorted(found.fragile) == ["o"]
    assert record["note"] == stage.note
    assert set(record) == {
        "name",
        "requested",
        "applied",
        "gate",
        "thresholds",
        "assessed",
        "fragile",
        "note",
    }
    reasons = result.confirmation.stage_two["withdrawn"]
    assert len(reasons) == 1
    (reason,) = reasons.values()
    by_name = {f.name: f for f in found.findings}
    assert reason == f"fragile across the declared grid: {by_name['o'].reason}"
    assert "1 of 2 survivors withdrawn as fragile" in stage.note


def test_a_not_applied_note_appends_the_rule_note_in_parentheses():
    loose = stability.Thresholds(s_min=0.5, band=0.9, q_min=0.5)
    stage = run(True, rule=loose)[1]

    found = stage.found
    assert found.rule_note and not found.store_note
    assert stage.note.endswith(
        f"fragile verdicts are warnings ({found.rule_note})"
    )
