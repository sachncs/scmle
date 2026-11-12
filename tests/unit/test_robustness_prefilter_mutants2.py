"""The second stage's records and notes, whatever the gate status."""

import types

import pytest

from scmle.contracts import qualification
from scmle.core import randomness
from scmle.data.detectors import certificates
from scmle.data.robustness import prefilter
from scmle.data.robustness import shift
from scmle.data.robustness import stability


def make_check(check_id, set_id=qualification.SetId.EXPLORATION):
    return qualification.Check(
        check_id=check_id,
        null_hypothesis="h",
        statistic=0.5,
        p_value=0.5,
        interval=None,
        effect=0.1,
        n=10,
        decision=qualification.Decision.PASS,
        set_id=set_id,
        seed_path=(1, check_id, 0),
        tier=qualification.Tier.UNVERIFIED,
    )


def make_finding(name, status, check=None, reason="why"):
    return types.SimpleNamespace(
        name=name, status=status, check=check, reason=reason
    )


def make_found(
    findings=(),
    gate=certificates.GateStatus.GATING,
    c2st=None,
    mmd=None,
    features=(),
    rule_note="",
    store_note="",
):
    return types.SimpleNamespace(
        findings=tuple(findings),
        stability_gate=gate,
        fragile=tuple(
            f.name for f in findings if f.status == stability.FRAGILE
        ),
        shift=types.SimpleNamespace(c2st=c2st, mmd=mmd, features=features),
        rule=stability.DEFAULT_RULE,
        rule_note=rule_note,
        store_note=store_note,
    )


def make_hypothesis(column, identifier):
    return types.SimpleNamespace(
        spec=types.SimpleNamespace(column=column), hypothesis_id=identifier
    )


def stage_with(found, requested, rows=40, seed=3):
    seen = []

    def compute(columns):
        seen.append(columns)
        return found

    return prefilter.StabilityStage(compute, requested, rows, seed), seen


def test_the_constants_are_exact():
    assert prefilter.FILTER_ID == "f6-stability-filter"
    assert prefilter.FEATURE_SHIFT_ID == "f6-shift-feature"
    assert prefilter.MAX_FEATURE_ROWS == 200


def test_a_feature_check_records_the_row_and_the_context():
    row = types.SimpleNamespace(
        column="age", statistic=2.5, p_value=0.125, effect=0.75, rejected=True
    )

    (check,) = prefilter.feature_checks(
        types.SimpleNamespace(features=(row,)), rows=33, seed=9
    )

    assert check.check_id == "f6-shift-feature:age"
    assert check.null_hypothesis == (
        "'age' has the same distribution in the exploration rows and the "
        "reference sample"
    )
    assert check.statistic == 2.5
    assert check.p_value == 0.125
    assert check.interval is None
    assert check.effect == 0.75
    assert check.n == 33
    assert check.tier is qualification.Tier.UNVERIFIED
    assert check.seed_path == randomness.seed_path(9, "f6-shift-feature", 0)


def test_the_cap_keeps_the_first_rejected_columns():
    rows = tuple(
        types.SimpleNamespace(
            column=f"c{i}",
            statistic=1.0,
            p_value=0.5,
            effect=0.0,
            rejected=i >= 250,
        )
        for i in range(300)
    )

    checks = prefilter.feature_checks(
        types.SimpleNamespace(features=rows), rows=1, seed=0
    )

    assert len(checks) == 200
    assert checks[0].check_id == "f6-shift-feature:c250"
    assert checks[49].check_id == "f6-shift-feature:c299"
    assert checks[50].check_id == "f6-shift-feature:c0"
    assert checks[-1].check_id == "f6-shift-feature:c149"


def test_diagnostics_are_ordered_and_keep_only_exploration_checks():
    first = make_check("s1")
    outside = make_check("s2", qualification.SetId.CONFIRMATION)
    c2st = make_check("c2st")
    mmd = make_check("mmd")
    found = make_found(
        findings=(
            make_finding("a", stability.STABLE, first),
            make_finding("b", stability.STABLE, None),
            make_finding("c", stability.STABLE, outside),
        ),
        c2st=c2st,
        mmd=mmd,
        features=(
            types.SimpleNamespace(
                column="x",
                statistic=1.0,
                p_value=0.5,
                effect=0.0,
                rejected=False,
            ),
        ),
    )

    rows = prefilter.diagnostic_rows(found, rows=7, seed=2)

    assert [source for source, check in rows] == [
        stability.DETECTOR_ID,
        shift.C2ST_ID,
        shift.MMD_ID,
        "f6-shift-feature",
    ]
    assert rows[0][1] is first
    assert rows[1][1] is c2st
    assert rows[2][1] is mmd
    assert rows[3][1].check_id == "f6-shift-feature:x"
    assert rows[3][1].n == 7


def test_diagnostics_skip_missing_shift_checks():
    found = make_found(mmd=make_check("mmd"))

    rows = prefilter.diagnostic_rows(found, rows=1, seed=0)

    assert [source for source, check in rows] == [shift.MMD_ID]


def test_an_unrequested_stage_only_describes():
    found = make_found(
        findings=(make_finding("o", stability.FRAGILE),),
        gate=certificates.GateStatus.GATING,
    )
    stage, seen = stage_with(found, requested=False)

    result = stage([make_hypothesis("o", "h1"), make_hypothesis(None, "h2")])

    assert seen == [("o",)]
    assert stage.found is found
    assert stage.applied is False
    assert stage.note == (
        "not applied: the charter does not ask for fragile survivors to be "
        "withdrawn"
    )
    assert dict(result.reasons) == {}
    assert result.accounted is False
    assert result.rule["requested"] is False
    assert result.rule["applied"] is False


@pytest.mark.parametrize(
    ("rule_note", "store_note", "suffix"),
    [
        ("", "", ""),
        ("rule says no", "", " (rule says no)"),
        ("", "store says no", " (store says no)"),
        ("rule says no", "store says no", " (rule says no)"),
    ],
)
def test_a_requested_but_descriptive_stage_explains_itself(
    rule_note, store_note, suffix
):
    found = make_found(
        findings=(make_finding("o", stability.FRAGILE),),
        gate=certificates.GateStatus.DESCRIPTIVE,
        rule_note=rule_note,
        store_note=store_note,
    )
    stage = stage_with(found, requested=True)[0]

    result = stage([make_hypothesis("o", "h1")])

    assert stage.applied is False
    assert stage.note == (
        "requested but not applied: the stability rule has no current "
        "passing certificate for these thresholds, so fragile verdicts are "
        "warnings" + suffix
    )
    assert dict(result.reasons) == {}
    assert result.accounted is True
    assert result.rule["gate"] == "descriptive"
    assert result.rule["fragile"] == ["o"]
    assert result.rule["note"] == stage.note


def test_a_gating_requested_stage_withdraws_the_fragile_survivors():
    found = make_found(
        findings=(
            make_finding("o", stability.FRAGILE, reason="moves"),
            make_finding("s", stability.STABLE),
            make_finding("p", stability.FRAGILE, reason="flips"),
        ),
        c2st=make_check("c2st"),
        features=(
            types.SimpleNamespace(
                column="x",
                statistic=1.0,
                p_value=0.5,
                effect=0.0,
                rejected=False,
            ),
        ),
    )
    stage = stage_with(found, requested=True, rows=12, seed=5)[0]

    result = stage(
        [
            make_hypothesis("s", "h1"),
            make_hypothesis("o", "h2"),
            make_hypothesis("p", "h3"),
            make_hypothesis("zz", "h4"),
        ]
    )

    assert stage.applied is True
    assert dict(result.reasons) == {
        "h2": "fragile across the declared grid: moves",
        "h3": "fragile across the declared grid: flips",
    }
    assert stage.note == (
        "applied before C was opened: 2 of 4 survivors withdrawn as fragile; "
        "the verdicts read the exploration set only"
    )
    assert result.accounted is True
    assert dict(result.rule) == {
        "name": "f6-stability-filter",
        "requested": True,
        "applied": True,
        "gate": "gating",
        "thresholds": stability.DEFAULT_RULE.to_dict(),
        "assessed": ["o", "s", "p"],
        "fragile": ["o", "p"],
        "note": stage.note,
    }
    assert result.diagnostics == prefilter.diagnostic_rows(found, 12, 5)
    assert len(result.diagnostics) == 2
    feature = result.diagnostics[1][1]
    assert feature.n == 12
    assert feature.seed_path == randomness.seed_path(5, "f6-shift-feature", 0)


def test_survivors_without_a_column_are_not_assessed():
    stage, seen = stage_with(make_found(), requested=True)

    stage([make_hypothesis("a", "h1"), make_hypothesis("", "h2")])

    assert seen == [("a",)]
