"""The stability filter as a pre-registered second stage of the confirmation."""

import json

import numpy
import pandas

from scmle.data.detectors import certificates
from scmle.data.ledger import confirm
from scmle.data.ledger import governed
from scmle.data.robustness import prefilter
from scmle.data.robustness import qualify
from scmle.data.robustness import report
from scmle.data.robustness import stability
from tests.unit.test_charter_contracts import make_charter
from tests.unit.test_ledger_governed import source_for
from tests.unit.test_robustness_qualify import RICH

SEED = 1


def world(seed, rows=500):
    """A stable effect (s), an outlier-driven one (o) and noise (n)."""
    rng = numpy.random.default_rng(seed)
    s, o = rng.normal(size=rows), rng.normal(size=rows)
    y = 0.6 * s + rng.normal(size=rows)
    hit = rng.choice(rows, 25, replace=False)
    side = rng.choice([-1.0, 1.0], 25)
    o[hit] = side * 8
    y[hit] = side * 6 + 0.5 * rng.normal(size=25)
    frame = pandas.DataFrame({"s": s, "o": o, "n": rng.normal(size=rows)})
    return frame, pandas.Series(y)


def stage_for(features, target, requested, reference=None, rule=None, **extra):
    def compute(columns):
        return qualify.run(
            features,
            target,
            survivors=list(columns),
            multiclass=False,
            choices=RICH,
            seed=SEED,
            permutations=20,
            reference=reference,
            candidates=list(features.columns),
            **({} if rule is None else {"rule": rule}),
            **extra,
        )

    return prefilter.StabilityStage(compute, requested, len(features), SEED)


def run(requested, tmp_path=None, reference=None, rule=None):
    features, target = world(SEED)
    cfeatures, ctarget = world(100 + SEED)
    stage = stage_for(features, target, requested, reference, rule)
    result = governed.run_governed(
        charter=make_charter(),
        features=features,
        target=target,
        classification=False,
        source=source_for(cfeatures, ctarget),
        run_dir=tmp_path,
        second_stage=stage,
    )
    return result, stage


def tiers(result):
    return {
        result.ledger.hypotheses[c.claim_id].spec.column: c.tier.value
        for c in result.confirmation.claims
    }


def test_a_fragile_planted_effect_is_not_tested_on_c(tmp_path):
    result, stage = run(True, tmp_path)

    assert stage.found.stability_gate is certificates.GateStatus.GATING
    assert {f.name: f.status for f in stage.found.findings} == {
        "s": stability.STABLE,
        "o": stability.FRAGILE,
    }
    assert tiers(result) == {"s": "confirmed", "o": "unverified"}
    tested = {c.check_id.split(":")[0] for c in result.confirmation.checks}
    fragile_id = next(
        h.hypothesis_id
        for h in result.ledger.survivors()
        if h.spec.column == "o"
    )
    assert fragile_id not in tested
    stage_two = result.confirmation.stage_two
    assert set(stage_two["withdrawn"]) == {fragile_id}
    assert stage_two["rule"]["applied"] is True
    assert stage_two["rule"]["thresholds"] == stability.DEFAULT_RULE.to_dict()
    assert stage.applied and stage.note.startswith("applied before C")
    claims = json.loads((tmp_path / confirm.CLAIMS_FILE).read_text())
    assert claims["stage_two"]["family_sha256"] == stage_two["family_sha256"]


def test_without_the_charter_request_the_fragile_effect_is_still_tested():
    result, stage = run(False)

    assert result.confirmation.stage_two is None
    assert not stage.applied
    assert stage.note.startswith("not applied: the charter does not ask")
    tested = {c.check_id.split(":")[0] for c in result.confirmation.checks}
    assert len(tested) == 2
    assert tiers(result)["s"] == "confirmed"
    assert report.stability_report(stage.found, stage.note)[
        "confirmation_filter"
    ].startswith("not applied")


def test_stability_results_are_wealth_free_ledger_diagnostics():
    result = run(False)[0]

    ledger = result.ledger
    ids = {c.check_id for c in ledger.diagnostics}
    assert {"f6-stability:s", "f6-stability:o"} <= ids
    assert all(
        c.set_id.value == "X" and c.tier.value == "unverified"
        for c in ledger.diagnostics
    )


def test_shift_results_are_diagnostics_when_a_reference_exists(tmp_path):
    features, target = world(SEED)
    reference = features.copy()
    reference["s"] = reference["s"] + 1.5
    cfeatures, ctarget = world(100 + SEED)
    stage = stage_for(features, target, False, reference=reference)
    result = governed.run_governed(
        charter=make_charter(),
        features=features,
        target=target,
        classification=False,
        source=source_for(cfeatures, ctarget),
        run_dir=tmp_path,
        second_stage=stage,
    )

    ids = {c.check_id for c in result.ledger.diagnostics}
    assert {"f6-shift-feature:s", "f6-shift-feature:n"} <= ids
    assert "f6-shift-c2st" in ids and "f6-shift-mmd" in ids
    row = next(
        c
        for c in result.ledger.diagnostics
        if c.check_id == "f6-shift-feature:s"
    )
    assert row.decision.value == "warn" and row.p_value < 0.01
    loaded = type(result.ledger).load(
        tmp_path / "hypothesis_ledger.jsonl", expected_head=result.ledger.head
    )
    assert [c.check_id for c in loaded.diagnostics] == [
        c.check_id for c in result.ledger.diagnostics
    ]
    assert loaded.wealth == result.ledger.wealth


def test_diagnostics_spend_no_wealth():
    plain = governed.run_governed(
        charter=make_charter(),
        features=world(SEED)[0],
        target=world(SEED)[1],
        classification=False,
        source=None,
        run_dir=None,
    )
    result = run(False)[0]
    assert result.ledger.wealth == plain.ledger.wealth
    assert len(result.ledger.diagnostics) > len(plain.ledger.diagnostics)


def test_thresholds_other_than_the_certified_rule_never_withdraw():
    loose = stability.Rule(s_min=0.5, band=0.9, q_min=0.5)
    result, stage = run(True, rule=loose)

    assert stage.found.stability_gate is certificates.GateStatus.DESCRIPTIVE
    assert "differ from the certified" in stage.found.rule_note
    assert not stage.applied
    assert stage.note.startswith("requested but not applied")
    assert result.confirmation.stage_two["withdrawn"] == {}
    assert (
        result.confirmation.stage_two["rule"]["thresholds"] == loose.to_dict()
    )
    document = report.stability_report(stage.found, stage.note)
    assert document["rule"] == loose.to_dict()
    assert document["certificate"]["note"] == stage.found.rule_note


def test_a_missing_certificate_means_a_warning_not_a_withdrawal(tmp_path):
    features, target = world(SEED)
    cfeatures, ctarget = world(100 + SEED)
    stage = stage_for(features, target, True, directory=tmp_path)
    result = governed.run_governed(
        charter=make_charter(),
        features=features,
        target=target,
        classification=False,
        source=source_for(cfeatures, ctarget),
        run_dir=None,
        second_stage=stage,
    )
    assert stage.found.stability_gate is certificates.GateStatus.DESCRIPTIVE
    assert not stage.applied and "no current passing certificate" in stage.note
    assert result.confirmation.stage_two["withdrawn"] == {}
