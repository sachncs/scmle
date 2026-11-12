"""Exact documents, wording and branches of the identification runner."""

import hashlib
import json

import numpy
import pandas
import pytest

from scmle.contracts import qualification
import scmle.data.admissibility.dag
from scmle.data.charter import causal
from scmle.data.identification import aipw
from scmle.data.identification import battery
from scmle.data.identification import certified
from scmle.data.identification import did
from scmle.data.identification import iv
from scmle.data.identification import rd
from scmle.data.identification import run
from scmle.data.identification import select
from scmle.data.identification import sensitivity
from scmle.data.identification import synth
from scmle.data.identification import tier
from scmle.data.ledger import budget
from tests.unit.test_identification_run import did_frame
from tests.unit.test_identification_run import observational

BUDGET = budget.ErrorBudget.create(0.05, 0.1)
COVARIATES = "unconfoundedness_given:a,b"
ASSUMPTIONS = (COVARIATES, "positivity", "no_interference")
UNCHECKED = (
    "the adjustment set is an unchecked attestation (no DAG): the tier "
    "cannot exceed attested"
)


def exposure_note(name):
    return (
        f"negative-control exposure {name} is only supported for "
        "backdoor_aipw and was not run"
    )


def block(**changes) -> causal.Question:
    values = {
        "treatment": "t",
        "outcome": "y",
        "estimand": "ATE",
        "assumptions": ASSUMPTIONS,
        "strategies": ("backdoor_aipw",),
    }
    values.update(changes)
    return causal.Question(**values)


class Stub(run.Estimator):
    """An adapter with a fixed estimate; the rows' size flags weakness."""

    name = "stub"

    def __init__(self, *args, se=0.1, estimate=1.0):
        super().__init__(*args)
        self.se = se
        self.value = estimate
        self.calls = []

    def estimate(self, c, level):
        self.calls.append((len(c), level))
        self.design["overlap_on_c"] = len(c)
        self.weak = self.weak or len(c) < 100
        value = self.value
        if "shift" in c.columns:
            value = float(c["shift"].iloc[0])
        if "bad" in c.columns and c["bad"].iloc[0] > 0:
            raise ValueError("cannot estimate here")
        return run.Est(value, self.se, value - 1, value + 1, len(c), {})


def stub_for(spec, **kw):
    frame = pandas.DataFrame({"y": [0.0], "t": [0]})
    return Stub(spec, frame, 0, "linear", 0.05, **kw)


def strata_frame(**sizes):
    parts = [
        pandas.DataFrame({"region": name, "y": 0.0, "t": 0}, index=range(n))
        for name, n in sizes.items()
    ]
    return pandas.concat(parts, ignore_index=True)


# constants


def test_the_gate_tables_and_documents_are_pinned():
    assert run.DOCUMENTS == (
        "identification_plan.json",
        "identification_report.json",
        "falsification_report.json",
        "sensitivity_report.json",
        "causal_claims.json",
    )
    assert run.INTERVAL_GATES == {
        "backdoor_aipw": "coverage_aipw",
        "randomized_diff": "coverage_randomized",
        "did": "coverage_did",
        "iv": "coverage_iv",
        "rd": "coverage_rd",
        "synth": "size_synth",
    }
    assert run.DESIGN_GATES == {
        "backdoor_aipw": ("overlap",),
        "iv": ("iv_strength",),
        "rd": ("density_rd",),
    }
    assert run.BATTERY_GATES == {
        "backdoor_aipw": "battery_nco",
        "randomized_diff": "battery_nco",
        "did": "battery_pretrend",
        "iv": "battery_iv",
        "rd": "battery_rd",
        "synth": "battery_synth",
    }
    assert run.SENSITIVITY_GATES == {
        "backdoor_aipw": "sensitivity_bound",
        "did": "sensitivity_did",
    }
    assert run.MIN_STRATUM_ROWS == 50
    assert run.LIMITS == (
        "confirmed means: under the attested assumptions the interval has "
        "its measured coverage and the battery would have flagged a "
        "violation of the certified size; it does not say the assumptions "
        "are true, and a confounder the negative controls do not share "
        "passes undetected"
    )
    assert {k: v.name for k, v in run.ADAPTERS.items()} == {
        k: k for k in select.STRATEGIES
    }
    assert set(run.ADAPTERS) == set(select.STRATEGIES)
    assert run.ADAPTERS["backdoor_aipw"] is run.BackdoorEstimator
    assert run.ADAPTERS["synth"] is run.SynthEstimator


# Est, digest, wald_from


@pytest.mark.parametrize(
    ("low", "high", "excludes"),
    [
        (0.1, 2.0, True),
        (-2.0, -0.1, True),
        (-1.0, 1.0, False),
        (0.0, 2.0, False),
        (-2.0, 0.0, False),
        (-1.0, -0.5, True),
        (0.5, 1.0, True),
    ],
)
def test_an_interval_excludes_zero_only_when_strictly_one_sided(
    low, high, excludes
):
    assert run.Est(0.0, 1.0, low, high, 10).excludes_zero() is excludes


def test_an_estimate_has_its_own_empty_detail():
    first = run.Est(0.0, 1.0, -1.0, 1.0, 10)
    second = run.Est(0.0, 1.0, -1.0, 1.0, 10)
    assert first.detail == {}
    assert first.detail is not second.detail
    assert (first.estimate, first.se, first.low, first.high, first.n) == (
        0.0,
        1.0,
        -1.0,
        1.0,
        10,
    )


def test_the_digest_is_the_sha256_of_the_sorted_compact_json():
    document = {"b": 1, "a": [1, 2], "c": {"z": None, "y": "é"}}
    text = '{"a":[1,2],"b":1,"c":{"y":"\\u00e9","z":null}}'
    assert run.digest(document) == hashlib.sha256(text.encode()).hexdigest()


def test_the_digest_writes_other_values_as_text():
    value = pandas.Timestamp("2026-01-02")
    text = json.dumps({"k": str(value)}, separators=(",", ":"))
    assert run.digest({"k": value}) == hashlib.sha256(text.encode()).hexdigest()
    assert run.digest({"a": 1}) != run.digest({"a": 2})


def test_a_wald_test_is_built_from_a_usable_standard_error_only():
    est = run.Est(2.0, 1.0, 0.0, 4.0, 10)
    assert run.wald_from(est, "n", "k") == battery.wald_test("n", "k", 2.0, 1.0)
    for se in (float("nan"), 0.0, -1.0, float("inf")):
        assert run.wald_from(run.Est(2.0, se, 0, 1, 1), "n", "k") is None
    tiny = run.wald_from(run.Est(1.0, 1e-9, 0, 1, 1), "n", "k")
    assert tiny is not None
    assert tiny.name == "n"
    assert tiny.kind == "k"


# the base adapter


def test_the_base_adapter_stores_its_settings_and_does_nothing_by_default():
    spec = block()
    frame = pandas.DataFrame({"y": [0.0], "t": [0]})
    adapter = run.Estimator(spec, frame, 7, "boosted", 0.02)
    assert adapter.spec is spec
    assert adapter.x is frame
    assert adapter.seed == 7
    assert adapter.learner == "boosted"
    assert adapter.alpha_falsify == 0.02
    assert adapter.design == {}
    assert adapter.refusal is None
    assert adapter.weak is False
    assert adapter.single_unit is False
    assert adapter.notes == []
    assert adapter.prepare() is None
    assert adapter.controls(frame, 0.95) == []
    assert adapter.sensitivity(frame, 0.95) == {
        "holds": None,
        "note": "no sensitivity analysis for this strategy",
    }
    other = run.Estimator(spec, frame, 7, "boosted", 0.02)
    assert other.design is not adapter.design
    assert other.notes is not adapter.notes
    with pytest.raises(NotImplementedError):
        adapter.estimate(frame, 0.95)
    with pytest.raises(NotImplementedError):
        adapter.estimate_for(frame, 0.95)


# alphas, assemble, refusal_outcome


def test_the_confirmation_level_is_split_four_to_one_exactly():
    primary, falsify = run.alphas(BUDGET)
    assert isinstance(primary, float)
    assert isinstance(falsify, float)
    assert primary == pytest.approx(0.04, abs=1e-12)
    assert falsify == pytest.approx(0.01, abs=1e-12)
    wide = budget.ErrorBudget.create(0.1, 0.1)
    assert run.alphas(wide) == pytest.approx((0.08, 0.02), abs=1e-12)


def make_assignment(tier_value=qualification.Tier.ATTESTED):
    return tier.Assignment(
        tier_value,
        "identified",
        {"estimated_on_c": True, "replication": False},
        ("one", "two"),
        "downgrade_tier",
    )


def test_the_claim_file_records_the_decision_and_hashes_the_others():
    spec = block()
    plan = select.select(spec)
    out = run.assemble(
        plan,
        spec,
        {"r": 1},
        {"f": 2},
        {"s": 3},
        make_assignment(),
        "backdoor_aipw",
        ("positivity", "no_interference"),
        {"positivity": "A, 2026"},
        ("b", "a"),
    )
    assert list(out.documents) == [
        "identification_plan.json",
        "identification_report.json",
        "falsification_report.json",
        "sensitivity_report.json",
        "causal_claims.json",
    ]
    assert out.documents["identification_plan.json"] == {
        **plan.to_dict(),
        "causal_spec": spec.to_dict(),
    }
    assert out.documents["identification_report.json"] == {"r": 1}
    assert out.documents["falsification_report.json"] == {"f": 2}
    assert out.documents["sensitivity_report.json"] == {"s": 3}
    claims = out.documents["causal_claims.json"]
    assert claims == {
        "status": "identified",
        "strategy": "backdoor_aipw",
        "estimand": "ATE",
        "tier": "attested",
        "conditions": {"estimated_on_c": True, "replication": False},
        "reasons": ["one", "two"],
        "action": "downgrade_tier",
        "assumptions": {"positivity": "A, 2026", "no_interference": ""},
        "certified_gates_used": ["a", "b"],
        "hashes": {
            name: run.digest(out.documents[name]) for name in run.DOCUMENTS[:4]
        },
        "limits": run.LIMITS,
    }
    assert out.hashes == {
        **claims["hashes"],
        "causal_claims.json": run.digest(claims),
    }
    assert out.decision == make_assignment()


def test_the_claim_file_without_a_strategy_or_tier_writes_null():
    spec = block()
    plan = select.select(spec)
    decision = tier.Assignment(None, "not_identified", {}, (), "refuse_claim")
    out = run.assemble(plan, spec, {}, {}, {}, decision, "", ())
    claims = out.documents["causal_claims.json"]
    assert claims["strategy"] is None
    assert claims["tier"] is None
    assert claims["assumptions"] == {}
    assert claims["certified_gates_used"] == []
    assert "subgroups" not in claims
    assert "attestations" not in claims


def test_subgroup_and_attestation_summaries_are_copied_into_the_claim_file():
    spec = block()
    plan = select.select(spec)
    subgroups = {
        "rule": "holm",
        "alpha": 0.04,
        "family_size": 2,
        "rejected": ("g=1",),
        "gate": {"status": "gating", "reason": "x"},
        "hypotheses": ["not copied"],
    }
    out = run.assemble(
        plan,
        spec,
        {},
        {},
        {},
        make_assignment(),
        "backdoor_aipw",
        (),
        None,
        (),
        subgroups,
        {"applied": ["x"]},
    )
    claims = out.documents["causal_claims.json"]
    assert claims["subgroups"] == {
        "rule": "holm",
        "alpha": 0.04,
        "family_size": 2,
        "rejected": ["g=1"],
        "gate": "gating",
    }
    assert claims["attestations"] == {"applied": ["x"]}
    assert claims["hashes"] == {
        name: run.digest(out.documents[name]) for name in run.DOCUMENTS[:4]
    }


def test_a_refusal_has_five_documents_and_no_tier():
    spec = block(assumptions=("positivity",))
    plan = select.select(spec)
    out = run.refusal_outcome(plan, spec)
    assert set(out.documents) == set(run.DOCUMENTS)
    report = out.documents["identification_report.json"]
    assert report == dict(select.refusal(plan, spec))
    assert "reason" not in report
    assert "dag" not in report
    assert out.documents["falsification_report.json"] == {}
    assert out.documents["sensitivity_report.json"] == {}
    claims = out.documents["causal_claims.json"]
    assert claims["status"] == "not_identified"
    assert claims["tier"] is None
    assert claims["action"] == "refuse_claim"
    assert claims["strategy"] is None
    assert claims["assumptions"] == {}
    assert claims["certified_gates_used"] == []
    assert out.decision.tier is None
    assert out.decision.status == "not_identified"


def test_a_refusal_carries_its_reason_and_the_dag_findings():
    spec = block(assumptions=("positivity",))
    plan = select.select(spec)
    out = run.refusal_outcome(plan, spec, "why not", {"hash": "abc"})
    report = out.documents["identification_report.json"]
    assert report["reason"] == "why not"
    assert report["dag"] == {"hash": "abc"}
    empty = run.refusal_outcome(plan, spec, None, {})
    assert empty.documents["identification_report.json"]["dag"] == {}
    assert "reason" not in empty.documents["identification_report.json"]


# replication


def test_replication_tests_each_declared_stratum_column():
    spec = block(strata=("region",))
    adapter = stub_for(spec)
    frame = strata_frame(a=60, b=80)
    out = run.replication(adapter, frame, 0.95, 1.0, 0.05)
    assert out["passed"] is True
    column = out["by_column"]["region"]
    assert set(column["strata"]) == {"region=a", "region=b"}
    assert column["strata"]["region=a"] == [1.0, 0.1]
    assert column["alpha"] == 0.05
    assert column["pooled_sign"] == 1
    assert adapter.calls == [(60, 0.95), (80, 0.95)]


def test_replication_skips_absent_columns_and_thin_or_failing_strata():
    spec = block(strata=("missing", "region"))
    adapter = stub_for(spec)
    out = run.replication(adapter, strata_frame(a=60, b=49), 0.95, 1.0, 0.05)
    assert out["by_column"]["missing"] == {"skipped": "column not in C"}
    assert out["by_column"]["region"] == {
        "skipped": "fewer than two usable strata"
    }
    assert out["passed"] is None
    assert adapter.calls == [(60, 0.95)]
    only = run.replication(adapter, strata_frame(a=50, b=50), 0.95, 1.0, 0.05)
    assert only["by_column"]["region"]["strata"].keys() == {
        "region=a",
        "region=b",
    }


def test_a_stratum_the_estimator_cannot_handle_is_not_usable():
    spec = block(strata=("region",))
    adapter = stub_for(spec)
    frame = strata_frame(a=60, b=60, c=60)
    frame["bad"] = (frame["region"] == "c").astype(float)
    out = run.replication(adapter, frame, 0.95, 1.0, 0.05)
    assert set(out["by_column"]["region"]["strata"]) == {"region=a", "region=b"}


def test_strata_without_a_usable_standard_error_are_dropped():
    spec = block(strata=("region",))
    frame = strata_frame(a=60, b=60)
    for se in (0.0, float("nan"), -1.0):
        adapter = stub_for(spec, se=se)
        out = run.replication(adapter, frame, 0.95, 1.0, 0.05)
        assert out["by_column"]["region"] == {
            "skipped": "fewer than two usable strata"
        }
        assert out["passed"] is None
    tiny = stub_for(spec, se=1e-9)
    kept = run.replication(tiny, frame, 0.95, 1.0, 0.05)
    assert len(kept["by_column"]["region"]["strata"]) == 2


def test_replication_passes_only_when_every_usable_column_does():
    spec = block(strata=("region", "other"))
    frame = strata_frame(a=60, b=60)
    frame["other"] = ["p", "q"] * 60
    adapter = stub_for(spec)
    both = run.replication(adapter, frame, 0.95, 1.0, 0.05)
    assert both["passed"] is True
    opposite = run.replication(adapter, frame, 0.95, -1.0, 0.05)
    assert opposite["passed"] is False
    assert opposite["by_column"]["region"]["same_sign"] is False
    # first column fails, second passes: the failure must stick
    frame["shift"] = [1.0] * 60 + [-1.0] * 60
    mixed = run.replication(adapter, frame, 0.95, 1.0, 0.05)
    assert mixed["passed"] is False
    # first column passes, second fails
    swapped = frame.assign(region=frame["other"], other=frame["region"])
    again = run.replication(
        stub_for(block(strata=("region", "other"))),
        swapped,
        0.95,
        1.0,
        0.05,
    )
    assert again["passed"] is False


def test_replication_restores_design_and_weak_even_when_estimates_change_them():
    spec = block(strata=("region",))
    adapter = stub_for(spec)
    adapter.design = {"keep": 1}
    adapter.weak = False
    run.replication(adapter, strata_frame(a=60, b=60), 0.95, 1.0, 0.05)
    assert adapter.design == {"keep": 1}
    assert adapter.weak is False
    adapter.weak = True
    run.replication(adapter, strata_frame(a=60, b=60), 0.95, 1.0, 0.05)
    assert adapter.weak is True


# subgroups


def subgroup_block(**changes):
    values = {
        "subgroups": (
            causal.Subgroup("region", ("a", "b")),
            causal.Subgroup("absent", ("z",)),
        )
    }
    values.update(changes)
    return block(**values)


def test_the_subgroup_report_has_one_row_per_declared_value():
    spec = subgroup_block()
    adapter = stub_for(spec)
    adapter.design = {"keep": 1}
    frame = strata_frame(a=60, b=40, c=70)
    report = run.subgroup_report(adapter, frame, 0.95, spec.subgroups)
    rows = report["hypotheses"]
    assert [(r["column"], r["value"]) for r in rows] == [
        ("region", "a"),
        ("region", "b"),
        ("absent", "z"),
    ]
    first, second, third = rows
    assert first["skipped"] is None
    assert (first["estimate"], first["se"], first["n"]) == (1.0, 0.1, 60)
    assert (first["low"], first["high"]) == (0.0, 2.0)
    assert first["p_value"] == pytest.approx(
        battery.wald_test("region=a", "subgroup", 1.0, 0.1).p_value
    )
    assert second["skipped"] == "fewer than 50 rows"
    assert second["p_value"] == 1.0
    assert "estimate" not in second
    assert third["skipped"] == "column not in C"
    assert third["p_value"] == 1.0
    assert report["rule"] == "holm"
    assert report["alpha"] == pytest.approx(0.05)
    assert report["family_size"] == 3
    assert report["statement"] == "FWER <= alpha under arbitrary dependence"
    assert first["adjusted_p"] == pytest.approx(3 * first["p_value"])
    assert [r["adjusted_p"] for r in rows[1:]] == [1.0, 1.0]
    assert [r["rejected"] for r in rows] == [True, False, False]
    assert report["rejected"] == ["region=a"]
    assert adapter.design == {"keep": 1}


def test_a_strong_subgroup_is_rejected_and_named():
    spec = block(subgroups=(causal.Subgroup("region", ("a", "b")),))
    adapter = stub_for(spec, se=0.01, estimate=5.0)
    frame = strata_frame(a=60, b=60)
    report = run.subgroup_report(adapter, frame, 0.9, spec.subgroups)
    assert report["alpha"] == pytest.approx(0.1)
    assert report["rejected"] == ["region=a", "region=b"]
    assert all(r["rejected"] is True for r in report["hypotheses"])
    assert all(r["adjusted_p"] < 0.1 for r in report["hypotheses"])


def test_a_subgroup_value_is_matched_as_text_and_errors_are_recorded():
    spec = block(subgroups=(causal.Subgroup("region", ("1", "bad")),))
    frame = pandas.DataFrame(
        {"region": [1] * 60 + [2] * 60, "y": 0.0, "t": 0, "shift": 2.0}
    )
    adapter = stub_for(spec)
    report = run.subgroup_report(adapter, frame, 0.95, spec.subgroups)
    assert report["hypotheses"][0]["n"] == 60
    assert report["hypotheses"][0]["estimate"] == 2.0
    assert report["hypotheses"][1]["skipped"] == "fewer than 50 rows"
    failing = pandas.DataFrame(
        {"region": ["x"] * 60, "y": 0.0, "t": 0, "bad": 1.0}
    )
    spec = block(subgroups=(causal.Subgroup("region", ("x",)),))
    out = run.subgroup_report(stub_for(spec), failing, 0.95, spec.subgroups)
    row = out["hypotheses"][0]
    assert row["skipped"] == "cannot estimate here"
    assert row["p_value"] == 1.0
    assert "estimate" not in row


def test_a_subgroup_without_a_standard_error_stays_in_the_family():
    spec = block(subgroups=(causal.Subgroup("region", ("a",)),))
    for se in (0.0, float("nan")):
        adapter = stub_for(spec, se=se)
        report = run.subgroup_report(
            adapter, strata_frame(a=60), 0.95, spec.subgroups
        )
        row = report["hypotheses"][0]
        assert row["skipped"] == "no usable standard error"
        assert row["p_value"] == 1.0
        assert report["family_size"] == 1


def label(named, status_path=None):
    found = {"hypotheses": [{"n": 100}, {"n": 10}, {"n": None}, {}]}
    run.label_subgroups(found, named)
    return found


def test_naming_the_subgroup_gate_makes_the_family_gating():
    found = label((certified.SUBGROUP_GATE, "other"))
    gate = found["gate"]
    assert gate["status"] == "gating"
    assert gate["reason"] == "named among the certified gates"
    assert found["statement_gate"].startswith(
        "gating: the family-wise error and the power at the certified size "
        "met their pre-registered floors"
    )
    assert found["statement_gate"].endswith(
        "(a result for a subgroup under `min_rows` is still descriptive)"
    )
    flags = [row["certified_size"] for row in found["hypotheses"]]
    shipped = certified.subgroup_gate(run.subgroup_report).min_rows
    if shipped is None:
        assert flags == [True, True, False, False]
    else:
        assert flags == [shipped <= 100, shipped <= 10, False, False]


def test_leaving_the_subgroup_gate_out_keeps_the_family_descriptive():
    found = label(("coverage_aipw",))
    assert found["gate"]["status"] == "descriptive"
    assert found["gate"]["reason"] == (
        "the caller named the certified gates and this is not one"
    )
    assert found["statement_gate"] == (
        "descriptive: the family-wise error and power of the subgroup tests "
        "are not certified, so read a rejected level as exploratory; "
        "the caller named the certified gates and this is not one"
    )
    assert [r["certified_size"] for r in found["hypotheses"]] == [False] * 4
    empty = label(())
    assert empty["gate"]["status"] == "descriptive"


def test_the_minimum_rows_come_from_the_shipped_certificate():
    found = label((certified.SUBGROUP_GATE,))
    shipped = certified.subgroup_gate(run.subgroup_report)
    assert found["gate"]["min_rows"] == shipped.min_rows
    unnamed = label(None)
    assert unnamed["gate"] == shipped.to_dict()
    if shipped.status == "descriptive":
        assert unnamed["statement_gate"].endswith(shipped.reason)


def test_subgroup_results_need_declared_subgroups_and_several_units():
    spec = subgroup_block()
    adapter = stub_for(spec)
    frame = strata_frame(a=60, b=60)
    out = run.subgroup_results(adapter, spec, frame, 0.95, ())
    assert out is not None
    assert out["gate"]["status"] == "descriptive"
    assert out["family_size"] == 3
    plain = run.subgroup_results(adapter, block(), frame, 0.95, ())
    assert plain is None
    single = stub_for(spec)
    single.single_unit = True
    assert run.subgroup_results(single, spec, frame, 0.95, ()) is None


def test_subgroup_estimation_does_not_leave_the_instrument_weak():
    spec = subgroup_block()
    adapter = stub_for(spec)
    adapter.weak = False
    run.subgroup_results(adapter, spec, strata_frame(a=60, b=60), 0.95, ())
    assert adapter.weak is False
    adapter.weak = True
    run.subgroup_results(adapter, spec, strata_frame(a=60, b=60), 0.95, ())
    assert adapter.weak is True


# small helpers


def test_sensitivity_of_a_single_treated_unit_is_the_placebo_note():
    adapter = stub_for(block())
    adapter.single_unit = True
    frame = strata_frame(a=3)
    assert run.sensitivity_of(adapter, frame, 0.95) == {
        "holds": None,
        "note": "single treated unit: inference is the placebo permutation",
    }
    adapter.single_unit = False
    assert run.sensitivity_of(adapter, frame, 0.95) == {
        "holds": None,
        "note": "no sensitivity analysis for this strategy",
    }


def test_the_closing_notes_name_unsupported_exposures_then_the_unchecked_set():
    spec = block(
        negative_controls=(
            causal.NegativeControl("exposure", "e1"),
            causal.NegativeControl("outcome", "o1"),
            causal.NegativeControl("exposure", "e2"),
        )
    )
    assert run.closing_notes(spec, "did", checked=True) == [
        exposure_note("e1"),
        exposure_note("e2"),
    ]
    assert run.closing_notes(spec, "backdoor_aipw", checked=True) == []
    assert run.closing_notes(spec, "backdoor_aipw", checked=False) == [
        UNCHECKED
    ]
    both = run.closing_notes(spec, "iv", checked=False)
    assert len(both) == 3
    assert both[-1].startswith("the adjustment set is an unchecked")


def test_the_verdict_states_the_hypothesis_in_words():
    assert run.verdict_of(block()) == {
        "hypothesis": "ATE of t on y = 0",
        "labelled": "causal, under the attested assumptions",
    }


def test_the_needed_assumptions_replace_unconfoundedness_by_the_charter_words():
    spec = block(
        assumptions=(
            "positivity",
            COVARIATES,
            "no_interference",
            "unconfoundedness_given:c",
        )
    )
    assert run.needed_assumptions("backdoor_aipw", spec) == (
        COVARIATES,
        "unconfoundedness_given:c",
        "positivity",
        "no_interference",
    )
    assert run.needed_assumptions("rd", spec) == (
        "continuity_at_cutoff",
        "no_manipulation",
    )


# the back-door adapter


def with_exposure(frame, seed):
    rng = numpy.random.default_rng(seed)
    return frame.assign(z=(rng.uniform(size=len(frame)) < 0.4).astype(float))


def backdoor_spec(**changes):
    values = {
        "negative_controls": (
            causal.NegativeControl("outcome", "nco"),
            causal.NegativeControl("exposure", "z"),
        ),
        "sensitivity": causal.Confounding("strongest_observed", 1.0),
    }
    values.update(changes)
    return block(**values)


def prepared(spec=None, rows=1500, seed=1):
    spec = spec or backdoor_spec()
    x = with_exposure(observational(rows, seed), 50)
    adapter = run.BackdoorEstimator(spec, x, 3, "linear", 0.01)
    adapter.prepare()
    return adapter, x


def confirm_rows():
    return with_exposure(observational(2500, 2), 51)


def test_the_backdoor_adapter_reads_its_covariates_and_starts_unfitted():
    adapter = run.BackdoorEstimator(
        block(), observational(100, 1), 0, "linear", 0.01
    )
    assert adapter.name == "backdoor_aipw"
    assert adapter.covariates == ("a", "b")
    assert adapter.nuisances is None
    none = block(
        assumptions=(
            "unconfoundedness_given:none",
            "positivity",
            "no_interference",
        )
    )
    assert (
        run.BackdoorEstimator(
            none, observational(100, 1), 0, "linear", 0.01
        ).covariates
        == ()
    )
    bare = block(assumptions=("positivity",))
    assert (
        run.BackdoorEstimator(
            bare, observational(100, 1), 0, "linear", 0.01
        ).covariates
        == ()
    )


def test_preparing_records_overlap_digest_and_learner_from_x():
    adapter, x = prepared()
    expected = aipw.fit_nuisances(x, ("a", "b"), "t", "y", 3, "linear")
    assert set(adapter.design) == {"overlap_on_x", "nuisance_digest", "learner"}
    assert adapter.design["nuisance_digest"] == expected.digest
    assert adapter.design["learner"] == "linear"
    e = expected.predict(x)[0]
    matrix = expected.scaler.transform(expected.design.matrix(x))
    report = aipw.overlap_report(e, aipw.binary(x["t"], "t"), matrix)
    assert adapter.design["overlap_on_x"] == report
    assert adapter.refusal is None
    assert adapter.nuisances.digest == expected.digest


def overlap_failing(n, seed):
    rng = numpy.random.default_rng(seed)
    a = rng.standard_normal(n)
    t = (rng.uniform(size=n) < 1 / (1 + numpy.exp(-4.0 * a))).astype(float)
    return pandas.DataFrame(
        {
            "a": a,
            "b": rng.standard_normal(n),
            "t": t,
            "y": 2 * t + a + rng.standard_normal(n),
        }
    )


def test_a_failed_overlap_refuses_unless_trimming_is_declared():
    x = overlap_failing(3000, 0)
    strict = run.BackdoorEstimator(block(), x, 0, "linear", 0.01)
    strict.prepare()
    assert strict.design["overlap_on_x"]["passed"] is False
    assert strict.refusal == (
        "the overlap gate failed on X and the charter declares no trimming rule"
    )
    trimmed = run.BackdoorEstimator(block(trim=0.05), x, 0, "linear", 0.01)
    trimmed.prepare()
    assert trimmed.design["overlap_on_x"]["passed"] is False
    assert trimmed.refusal is None


def test_the_backdoor_estimate_is_the_frozen_aipw_estimate_on_c():
    adapter = prepared(rows=1500)[0]
    c = confirm_rows()
    spec = adapter.spec
    out = adapter.estimate(c, 0.9)
    direct = aipw.estimate(
        adapter.nuisances, c, "t", "y", "ATE", 0.9, spec.trim, spec.min_share
    )
    assert (out.estimate, out.se, out.low, out.high, out.n) == (
        direct.estimate,
        direct.se,
        direct.low,
        direct.high,
        direct.n,
    )
    assert out.detail == {"population": direct.population}
    assert adapter.design["overlap_on_c"] == direct.overlap
    assert adapter.design["population"] == direct.population
    wide = adapter.estimate(c, 0.99)
    assert wide.high - wide.low > out.high - out.low


def test_the_backdoor_estimate_needs_a_prepared_adapter():
    adapter = run.BackdoorEstimator(
        block(), observational(100, 1), 0, "linear", 0.01
    )
    with pytest.raises(ValueError, match=r"prepare\(\) must run first"):
        adapter.estimate(observational(100, 2), 0.95)


def test_the_backdoor_estimate_for_another_outcome_refits_on_x():
    adapter, x = prepared()
    c = confirm_rows()
    spec = adapter.spec
    same = adapter.estimate_for(c, 0.95)
    main = adapter.estimate(c, 0.95)
    assert (same.estimate, same.se, same.low, same.high, same.n) == (
        main.estimate,
        main.se,
        main.low,
        main.high,
        main.n,
    )
    assert same.detail == {}
    other = adapter.estimate_for(c, 0.9, "nco")
    fitted = aipw.fit_nuisances(x, ("a", "b"), "t", "nco", 3, "linear")
    direct = aipw.estimate(
        fitted, c, "t", "nco", "ATE", 0.9, spec.trim, spec.min_share
    )
    assert (other.estimate, other.se, other.low, other.high, other.n) == (
        direct.estimate,
        direct.se,
        direct.low,
        direct.high,
        direct.n,
    )
    assert other.estimate != main.estimate


def test_the_backdoor_controls_test_each_negative_outcome_and_exposure():
    adapter, x = prepared()
    c = confirm_rows()
    tests = adapter.controls(c, 0.95)
    assert [(t.name, t.kind) for t in tests] == [
        ("negative_outcome:nco", "negative_outcome"),
        ("negative_exposure:z", "negative_exposure"),
    ]
    spec = adapter.spec
    fitted = aipw.fit_nuisances(x, ("a", "b"), "z", "y", 3, "linear")
    found = aipw.estimate(
        fitted, c, "z", "y", "ATE", 0.95, spec.trim, spec.min_share
    )
    assert tests[1] == battery.wald_test(
        "negative_exposure:z", "negative_exposure", found.estimate, found.se
    )
    outcome = adapter.estimate_for(c, 0.95, "nco")
    assert tests[0] == battery.wald_test(
        "negative_outcome:nco", "negative_outcome", outcome.estimate, outcome.se
    )
    only = backdoor_spec(negative_controls=())
    quiet = prepared(only)[0]
    assert quiet.controls(c, 0.95) == []


def test_the_backdoor_sensitivity_wording_when_it_cannot_run():
    adapter = prepared(backdoor_spec(sensitivity=None))[0]
    assert adapter.sensitivity(confirm_rows(), 0.95) == {
        "holds": None,
        "note": "no sensitivity block in the charter",
    }
    none = backdoor_spec(
        assumptions=(
            "unconfoundedness_given:none",
            "positivity",
            "no_interference",
        )
    )
    bare = run.BackdoorEstimator(none, observational(100, 1), 0, "linear", 0.01)
    assert bare.sensitivity(observational(100, 2), 0.95) == {
        "holds": None,
        "note": "benchmarking needs a covariate",
    }
    unprepared = run.BackdoorEstimator(
        backdoor_spec(), observational(100, 1), 0, "linear", 0.01
    )
    with pytest.raises(ValueError, match=r"prepare\(\) must run first"):
        unprepared.sensitivity(observational(100, 2), 0.95)


def test_the_backdoor_sensitivity_is_the_linear_summary_on_c():
    spec = backdoor_spec(sensitivity=causal.Confounding("fixed", 0.3))
    adapter = prepared(spec)[0]
    c = confirm_rows()
    out = adapter.sensitivity(c, 0.9)
    matrix = adapter.nuisances.design.matrix(c)
    names = [f"x{j}" for j in range(matrix.shape[1])]
    frame = pandas.DataFrame(matrix, columns=names)
    frame["t"] = c["t"].to_numpy(dtype=float)
    frame["y"] = c["y"].to_numpy(dtype=float)
    expected = sensitivity.assess(
        frame, "t", "y", tuple(names), "fixed", 0.3, 0.9
    ).to_dict()
    expected["scale"] = (
        "linear-model coefficient of the treatment on the rows of C; a "
        "description of how fragile that summary is, not of the AIPW "
        "estimate itself"
    )
    assert out == expected
    other = adapter.sensitivity(c, 0.99)
    assert other != out


# the randomized adapter


def test_the_randomized_adapter_is_the_difference_in_means():
    spec = block(
        estimand="ATE",
        strategies=("randomized_diff",),
        negative_controls=(
            causal.NegativeControl("exposure", "z"),
            causal.NegativeControl("outcome", "nco"),
        ),
    )
    x = observational(300, 1)
    adapter = run.RandomizedEstimator(spec, x, 0, "linear", 0.01)
    assert adapter.name == "randomized_diff"
    c = observational(600, 2)
    direct = aipw.difference_in_means(c, "t", "y", 0.9)
    out = adapter.estimate(c, 0.9)
    assert (out.estimate, out.se, out.low, out.high, out.n) == (
        direct.estimate,
        direct.se,
        direct.low,
        direct.high,
        direct.n,
    )
    assert out.detail == {}
    other = aipw.difference_in_means(c, "t", "nco", 0.9)
    assert adapter.estimate_for(c, 0.9, "nco").estimate == other.estimate
    assert adapter.estimate_for(c, 0.9).estimate == direct.estimate
    tests = adapter.controls(c, 0.9)
    assert tests == [
        battery.wald_test(
            "negative_outcome:nco", "negative_outcome", other.estimate, other.se
        )
    ]
    assert adapter.sensitivity(c, 0.9) == {
        "holds": True,
        "note": "randomized assignment: no unmeasured confounding to bound",
        "applicable": False,
    }


# the difference-in-differences adapter


def did_spec(**changes):
    values = {
        "treatment": "g",
        "outcome": "y",
        "estimand": "ATT_DID",
        "unit": "id",
        "time": "t",
        "adoption": "g",
        "assumptions": (
            "parallel_trends",
            "no_anticipation",
            "no_interference",
        ),
        "strategies": ("did",),
        "negative_controls": (
            causal.NegativeControl("outcome", "nco"),
            causal.NegativeControl("exposure", "z"),
        ),
        "sensitivity": causal.Confounding(m_max=1.0),
    }
    values.update(changes)
    return causal.Question(**values)


def early_adopters(frame):
    return frame.assign(g=frame["g"].where(frame["g"].isna(), 1.0))


def test_the_did_adapter_needs_all_three_panel_columns():
    frame = did_frame(1, 100)
    ok = run.DidEstimator(did_spec(), frame, 0, "linear", 0.01)
    assert ok.name == "did"
    assert ok.columns() == ("id", "t", "g")
    for missing in ("unit", "time", "adoption"):
        adapter = run.DidEstimator(
            did_spec(**{missing: None}), frame, 0, "linear", 0.01
        )
        with pytest.raises(
            ValueError, match="did needs unit, time and adoption"
        ):
            adapter.columns()


def test_the_did_design_is_read_from_x_at_the_default_level():
    frame = did_frame(1, 200)
    adapter = run.DidEstimator(did_spec(), frame, 0, "linear", 0.01)
    adapter.prepare()
    fit = did.estimate(frame, "id", "t", "g", "y")
    assert adapter.design == {
        "groups": fit.groups,
        "n_never_treated": fit.n_never,
        "events": list(fit.events),
        "pretrend_p_on_x": fit.pretrend_p,
        "pre_periods_support_sensitivity": True,
    }
    early = early_adopters(frame)
    other = run.DidEstimator(did_spec(), early, 0, "linear", 0.01)
    other.prepare()
    assert other.design["pre_periods_support_sensitivity"] is False
    assert other.design["pretrend_p_on_x"] is None


def test_the_did_estimate_is_the_overall_att_with_its_event_study():
    adapter = run.DidEstimator(did_spec(), did_frame(1, 200), 0, "linear", 0.01)
    c = did_frame(2, 200)
    out = adapter.estimate(c, 0.9)
    fit = did.estimate(c, "id", "t", "g", "y", 0.9)
    assert (out.estimate, out.se, out.low, out.high) == (
        fit.estimate,
        fit.se,
        fit.low,
        fit.high,
    )
    assert out.n == len(c)
    assert out.detail == {
        "pretrend_p": fit.pretrend_p,
        "event_study": dict(zip(fit.events, fit.coefficients, strict=True)),
    }
    other = adapter.estimate_for(c, 0.9, "nco")
    direct = did.estimate(c, "id", "t", "g", "nco", 0.9)
    assert (other.estimate, other.se, other.low, other.high, other.n) == (
        direct.estimate,
        direct.se,
        direct.low,
        direct.high,
        len(c),
    )
    assert other.detail == {}
    assert adapter.estimate_for(c, 0.9).estimate == fit.estimate


def test_the_did_controls_start_with_the_pretrend_wald_test():
    adapter = run.DidEstimator(did_spec(), did_frame(1, 200), 0, "linear", 0.01)
    c = did_frame(2, 200)
    tests = adapter.controls(c, 0.9)
    fit = did.estimate(c, "id", "t", "g", "y", 0.9)
    assert tests[0] == battery.Test("pretrend_wald", "placebo", fit.pretrend_p)
    direct = did.estimate(c, "id", "t", "g", "nco", 0.9)
    assert tests[1:] == [
        battery.wald_test(
            "negative_outcome:nco",
            "negative_outcome",
            direct.estimate,
            direct.se,
        )
    ]
    early = adapter.controls(early_adopters(c), 0.9)
    assert [t.name for t in early] == ["negative_outcome:nco"]


def test_the_did_sensitivity_widens_the_interval_at_m_max():
    adapter = run.DidEstimator(did_spec(), did_frame(1, 200), 0, "linear", 0.01)
    c = did_frame(2, 200)
    out = adapter.sensitivity(c, 0.9)
    fit = did.estimate(c, "id", "t", "g", "y", 0.9)
    low, high = fit.sensitivity(1.0)
    assert out == {
        "holds": bool(low > 0 or high < 0),
        "m_max": 1.0,
        "low": low,
        "high": high,
        "breakdown_m": fit.breakdown(),
        "max_pre_change": fit.max_pre_change,
        "note": "simplified relative-magnitudes bound, not HonestDiD",
    }
    wide = run.DidEstimator(
        did_spec(sensitivity=causal.Confounding(m_max=50.0)),
        did_frame(1, 200),
        0,
        "linear",
        0.01,
    )
    assert wide.sensitivity(c, 0.9)["holds"] is False
    assert wide.sensitivity(c, 0.9)["m_max"] == 50.0


def test_the_did_sensitivity_says_when_it_cannot_run():
    adapter = run.DidEstimator(
        did_spec(sensitivity=None), did_frame(1, 100), 0, "linear", 0.01
    )
    assert adapter.sensitivity(did_frame(2, 100), 0.9) == {
        "holds": None,
        "note": "no sensitivity block in the charter",
    }
    full = run.DidEstimator(did_spec(), did_frame(1, 100), 0, "linear", 0.01)
    assert full.sensitivity(early_adopters(did_frame(2, 100)), 0.9) == {
        "holds": None,
        "note": "fewer than two pre-periods",
    }


# the instrumental-variable adapter


def iv_world(n, seed, strength=1.0):
    rng = numpy.random.default_rng(seed)
    z = rng.standard_normal(n)
    u = rng.standard_normal(n)
    d = strength * z + u + rng.standard_normal(n)
    y = 2 * d + 1.5 * u + rng.standard_normal(n)
    return pandas.DataFrame(
        {"z": z, "d": d, "y": y, "nco": u + rng.standard_normal(n)}
    )


def iv_spec(**changes):
    values = {
        "treatment": "d",
        "outcome": "y",
        "estimand": "LATE",
        "instrument": "z",
        "assumptions": (
            "instrument_relevance",
            "exclusion",
            "independence",
            "monotonicity",
        ),
        "strategies": ("iv",),
        "negative_controls": (
            causal.NegativeControl("exposure", "z"),
            causal.NegativeControl("outcome", "nco"),
        ),
    }
    values.update(changes)
    return causal.Question(**values)


def test_the_iv_adapter_uses_the_declared_covariates_as_controls():
    frame = iv_world(100, 1)
    adapter = run.IvEstimator(iv_spec(), frame, 0, "linear", 0.01)
    assert adapter.name == "iv"
    assert adapter.controls_columns() == ()
    spec = iv_spec(
        assumptions=(*iv_spec().assumptions, "unconfoundedness_given:nco")
    )
    controlled = run.IvEstimator(spec, frame, 0, "linear", 0.01)
    assert controlled.controls_columns() == ("nco",)


def test_the_iv_fit_needs_an_instrument_and_passes_controls_and_outcome():
    frame = iv_world(300, 1)
    bare = run.IvEstimator(iv_spec(instrument=None), frame, 0, "linear", 0.01)
    with pytest.raises(ValueError, match="iv needs an instrument"):
        bare.fit(frame, 0.95)
    spec = iv_spec(
        assumptions=(*iv_spec().assumptions, "unconfoundedness_given:nco")
    )
    adapter = run.IvEstimator(spec, frame, 0, "linear", 0.01)
    fit = adapter.fit(frame, 0.9)
    direct = iv.estimate(frame, "d", "y", "z", ("nco",), 0.9)
    assert (fit.estimate, fit.se, fit.f_stat) == (
        direct.estimate,
        direct.se,
        direct.f_stat,
    )
    other = adapter.fit(frame, 0.9, "nco")
    assert (
        other.estimate
        == iv.estimate(frame, "d", "nco", "z", ("nco",), 0.9).estimate
    )


def test_the_iv_design_records_the_first_stage_on_x():
    frame = iv_world(1500, 1)
    adapter = run.IvEstimator(iv_spec(), frame, 0, "linear", 0.01)
    adapter.prepare()
    fit = iv.estimate(frame, "d", "y", "z", (), 0.95)
    assert adapter.weak is False
    assert adapter.design == {
        "first_stage_F_on_x": fit.f_stat,
        "threshold": iv.F_THRESHOLD,
        "weak_on_x": False,
        "overidentification": "not applicable: one instrument, just identified",
    }
    weak = run.IvEstimator(
        iv_spec(), iv_world(1500, 2, strength=0.02), 0, "linear", 0.01
    )
    weak.prepare()
    assert weak.weak is True
    assert weak.design["weak_on_x"] is True


def test_the_iv_estimate_marks_a_weak_confirmation_set_and_keeps_weakness():
    strong = iv_world(1500, 1)
    faint = iv_world(1500, 2, strength=0.02)
    adapter = run.IvEstimator(iv_spec(), strong, 0, "linear", 0.01)
    adapter.prepare()
    out = adapter.estimate(strong, 0.9)
    fit = iv.estimate(strong, "d", "y", "z", (), 0.9)
    assert (out.estimate, out.se, out.low, out.high, out.n) == (
        fit.estimate,
        fit.se,
        fit.low,
        fit.high,
        fit.n,
    )
    assert out.detail == {
        "headline": fit.headline,
        "ar_kind": fit.ar_kind,
        "estimand": "LATE",
    }
    assert adapter.weak is False
    assert adapter.design["first_stage_F_on_c"] == fit.f_stat
    adapter.estimate(faint, 0.9)
    assert adapter.weak is True
    adapter.estimate(strong, 0.9)
    assert adapter.weak is True


def test_the_iv_estimate_for_another_outcome_gives_the_two_stage_interval():
    frame = iv_world(1500, 1)
    adapter = run.IvEstimator(iv_spec(), frame, 0, "linear", 0.01)
    out = adapter.estimate_for(frame, 0.9, "nco")
    fit = iv.estimate(frame, "d", "nco", "z", (), 0.9)
    assert (out.estimate, out.se, out.low, out.high, out.n) == (
        fit.estimate,
        fit.se,
        fit.wald_low,
        fit.wald_high,
        fit.n,
    )
    assert out.detail == {}
    assert (
        adapter.estimate_for(frame, 0.9).estimate
        == iv.estimate(frame, "d", "y", "z", (), 0.9).estimate
    )
    tests = adapter.controls(frame, 0.9)
    assert tests == [
        battery.wald_test(
            "negative_outcome:nco", "negative_outcome", fit.estimate, fit.se
        )
    ]


# the regression-discontinuity adapter


def rd_world(n, seed):
    rng = numpy.random.default_rng(seed)
    x = rng.uniform(-1, 1, n)
    return pandas.DataFrame(
        {
            "x": x,
            "y": 2 * x + (x >= 0) + rng.normal(scale=0.5, size=n),
            "cov": 0.5 * x + rng.normal(scale=0.5, size=n),
        }
    )


def rd_spec(**changes):
    values = {
        "treatment": "d",
        "outcome": "y",
        "estimand": "LATE_RD",
        "running_variable": "x",
        "cutoff": 0.0,
        "assumptions": ("continuity_at_cutoff", "no_manipulation"),
        "strategies": ("rd",),
        "negative_controls": (
            causal.NegativeControl("outcome", "cov"),
            causal.NegativeControl("exposure", "z"),
        ),
    }
    values.update(changes)
    return causal.Question(**values)


def test_the_rd_adapter_balances_on_the_negative_control_outcomes():
    adapter = run.RdEstimator(rd_spec(), rd_world(100, 1), 0, "linear", 0.01)
    assert adapter.name == "rd"
    assert adapter.bandwidth == 0.0
    assert adapter.balance_columns() == ("cov",)
    none = run.RdEstimator(
        rd_spec(negative_controls=()), rd_world(100, 1), 0, "linear", 0.01
    )
    assert none.balance_columns() == ()


def test_the_rd_design_fixes_the_bandwidth_on_x():
    frame = rd_world(2000, 1)
    adapter = run.RdEstimator(rd_spec(), frame, 0, "linear", 0.01)
    adapter.prepare()
    fit = rd.estimate(frame, "x", "y", 0.0, covariates=("cov",))
    assert adapter.bandwidth == fit.bandwidth > 0
    assert adapter.design == {
        "bandwidth": fit.bandwidth,
        "density_p_on_x": fit.density_p,
        "alpha_density": 0.01,
        "n_left_in_bandwidth": fit.n_left,
        "n_right_in_bandwidth": fit.n_right,
    }
    assert adapter.refusal is None


def test_the_rd_density_check_refuses_below_the_level_only():
    frame = rd_world(2000, 1)
    frame.loc[
        frame.x.between(-0.2, 0) & (numpy.arange(2000) % 2 == 0), "x"
    ] *= -1
    fit = rd.estimate(frame, "x", "y", 0.0, covariates=("cov",))
    refusing = run.RdEstimator(rd_spec(), frame, 0, "linear", fit.density_p * 2)
    refusing.prepare()
    assert refusing.refusal == (
        "the density check rejects at the cutoff on X: possible manipulation "
        "of the running variable"
    )
    exact = run.RdEstimator(rd_spec(), frame, 0, "linear", fit.density_p)
    exact.prepare()
    assert exact.refusal is None


def test_the_rd_adapter_needs_a_running_variable_and_cutoff():
    frame = rd_world(100, 1)
    for change in ({"running_variable": None}, {"cutoff": None}):
        adapter = run.RdEstimator(rd_spec(**change), frame, 0, "linear", 0.01)
        with pytest.raises(
            ValueError, match="rd needs a running variable and cutoff"
        ):
            adapter.prepare()
        with pytest.raises(
            ValueError, match="rd needs a running variable and cutoff"
        ):
            adapter.fit(frame, 0.95)


def test_a_cutoff_of_zero_is_a_valid_cutoff():
    frame = rd_world(1500, 1)
    adapter = run.RdEstimator(rd_spec(cutoff=0.0), frame, 0, "linear", 0.01)
    adapter.prepare()
    assert adapter.bandwidth > 0


def test_the_rd_estimate_is_bias_corrected_at_the_frozen_bandwidth():
    x, c = rd_world(1500, 1), rd_world(1500, 2)
    adapter = run.RdEstimator(rd_spec(), x, 0, "linear", 0.01)
    adapter.prepare()
    out = adapter.estimate(c, 0.9)
    fit = rd.estimate(c, "x", "y", 0.0, 0.9, adapter.bandwidth, ())
    assert (out.estimate, out.se, out.low, out.high) == (
        fit.bias_corrected,
        fit.se,
        fit.low,
        fit.high,
    )
    assert out.n == fit.n_left + fit.n_right
    assert out.detail == {
        "conventional": fit.estimate,
        "bandwidth": adapter.bandwidth,
    }
    other = adapter.estimate_for(c, 0.9, "cov")
    direct = rd.estimate(c, "x", "cov", 0.0, 0.9, adapter.bandwidth, ())
    assert (other.estimate, other.se, other.low, other.high, other.n) == (
        direct.bias_corrected,
        direct.se,
        direct.low,
        direct.high,
        direct.n_left + direct.n_right,
    )
    assert other.detail == {}
    assert adapter.estimate_for(c, 0.9).estimate == fit.bias_corrected


def test_the_rd_controls_are_the_placebo_cutoffs_and_covariate_balance():
    x, c = rd_world(1500, 1), rd_world(1500, 2)
    adapter = run.RdEstimator(rd_spec(), x, 0, "linear", 0.01)
    adapter.prepare()
    tests = adapter.controls(c, 0.9)
    fit = rd.estimate(c, "x", "y", 0.0, 0.9, adapter.bandwidth, ("cov",))
    expected = [
        battery.Test(
            f"placebo_cutoff:{name}",
            "placebo",
            float(row["p"]),
            float(row["bias_corrected"]),
            float(row["se"]),
        )
        for name, row in fit.placebos.items()
    ] + [
        battery.Test(
            f"covariate_balance:{name}",
            "negative_outcome",
            float(row["p"]),
            float(row["bias_corrected"]),
            float(row["se"]),
        )
        for name, row in fit.balance.items()
    ]
    assert expected
    assert tests == expected
    assert any(t.name == "covariate_balance:cov" for t in tests)


def test_the_rd_sensitivity_is_descriptive_and_compares_conclusions():
    x, c = rd_world(1500, 1), rd_world(1500, 2)
    adapter = run.RdEstimator(rd_spec(), x, 0, "linear", 0.01)
    adapter.prepare()
    out = adapter.sensitivity(c, 0.9)
    fit = rd.estimate(c, "x", "y", 0.0, 0.9, adapter.bandwidth, ())
    stable = [
        (row["low"] <= 0 <= row["high"]) == (fit.low <= 0 <= fit.high)
        for row in fit.sensitivity.values()
    ]
    assert out == {
        "holds": True,
        "applicable": False,
        "note": "RD has no unmeasured-confounding bound; the bandwidth "
        "sensitivity is descriptive",
        "bandwidth_sensitivity": fit.sensitivity,
        "same_conclusion_at_all_multiples": all(stable),
    }
    assert isinstance(out["same_conclusion_at_all_multiples"], bool)


# the synthetic-control adapter


def synthetic_panel(donors=24, periods=16, first=12, seed=3):
    rng = numpy.random.default_rng(seed)
    factors = rng.standard_normal((periods, 2)).cumsum(axis=0) * 0.5
    loadings = rng.uniform(0, 1, (donors + 1, 2))
    rows = []
    for u in range(donors + 1):
        for t in range(periods):
            y = factors[t] @ loadings[u] + rng.normal(scale=0.1)
            treated = float(u == 0 and t >= first)
            rows.append((u, t, y + 3.0 * treated, treated))
    return pandas.DataFrame(rows, columns=["u", "t", "y", "d"])


def synth_spec(**changes):
    values = {
        "treatment": "d",
        "outcome": "y",
        "estimand": "ATT_SC",
        "unit": "u",
        "time": "t",
        "assumptions": ("synthetic_hull", "no_interference"),
        "strategies": ("synth",),
    }
    values.update(changes)
    return causal.Question(**values)


def test_the_synthetic_adapter_runs_on_x_and_marks_a_single_unit():
    panel = synthetic_panel()
    adapter = run.SynthEstimator(synth_spec(), panel, 0, "linear", 0.01)
    assert adapter.name == "synth"
    assert adapter.single_unit is True
    out = adapter.estimate(panel.iloc[:0], 0.95)
    fit = synth.estimate(panel, "u", "t", "y", 0, 12, 0.05)
    assert out.estimate == fit.effect
    assert numpy.isnan(out.se)
    assert numpy.isnan(out.low)
    assert numpy.isnan(out.high)
    assert out.n == len(panel)
    assert out.detail == {"p_value": fit.p_value, "ratio": fit.ratio}
    assert adapter.design == {
        "donors": fit.donors,
        "min_attainable_p": fit.min_p,
        "pre_rmspe": fit.pre_rmspe,
        "weights": fit.weights,
    }


def test_the_synthetic_adapter_checks_the_donor_count_at_one_minus_level():
    panel = synthetic_panel()
    adapter = run.SynthEstimator(synth_spec(), panel, 0, "linear", 0.01)
    adapter.estimate(panel, 0.95)
    with pytest.raises(ValueError, match="donors"):
        adapter.estimate(panel, 0.99)


def test_the_synthetic_adapter_refuses_a_panel_it_cannot_use():
    panel = synthetic_panel()
    for change in ({"unit": None}, {"time": None}):
        adapter = run.SynthEstimator(
            synth_spec(**change), panel, 0, "linear", 0.01
        )
        with pytest.raises(ValueError, match="synth needs unit and time"):
            adapter.estimate(panel, 0.95)
    two = panel.copy()
    two.loc[(two.u == 1) & (two.t >= 12), "d"] = 1.0
    adapter = run.SynthEstimator(synth_spec(), two, 0, "linear", 0.01)
    with pytest.raises(
        ValueError, match="synthetic control needs exactly one treated unit"
    ):
        adapter.estimate(two, 0.95)
    none = panel.assign(d=0.0)
    adapter = run.SynthEstimator(synth_spec(), none, 0, "linear", 0.01)
    with pytest.raises(ValueError, match="exactly one treated unit"):
        adapter.estimate(none, 0.95)
    with pytest.raises(NotImplementedError):
        adapter.estimate_for(panel, 0.95, "y")


# identify and design_only, with scripted adapters


def scripted(**attributes):
    """An adapter class whose hooks are set by class attributes."""

    class Scripted(run.Estimator):
        name = "scripted"
        prepare_error = None
        refusal_text = None
        estimate_error = None
        controls_error = None
        found = run.Est(1.5, 0.2, 1.1, 1.9, 40, {"extra": 1})
        tests = ()
        sens = {"holds": True}
        weak_after = False
        single = False
        seen = []

        def __init__(self, *args):
            super().__init__(*args)
            self.single_unit = self.single
            type(self).seen = [args[2:], 0]

        def prepare(self):
            if self.prepare_error:
                raise ValueError(self.prepare_error)
            self.design = {"prepared": True}
            self.refusal = self.refusal_text

        def estimate(self, c, level):
            type(self).seen[1] += 1
            type(self).level = level
            if self.estimate_error:
                raise ValueError(self.estimate_error)
            self.weak = self.weak_after
            return self.found

        def controls(self, c, level):
            if self.controls_error:
                raise ValueError(self.controls_error)
            return list(self.tests)

        def sensitivity(self, c, level):
            return dict(self.sens)

    for key, value in attributes.items():
        setattr(Scripted, key, value)
    return Scripted


def install(monkeypatch, strategy="backdoor_aipw", **attributes):
    adapter = scripted(**attributes)
    monkeypatch.setitem(run.ADAPTERS, strategy, adapter)
    return adapter


def record_decisions(monkeypatch):
    seen = []
    decide = tier.decide

    def recording(facts):
        seen.append(facts)
        return decide(facts)

    monkeypatch.setattr(run.tier, "decide", recording)
    return seen


def frames():
    x = pandas.DataFrame({"a": [0.0, 1.0], "b": [1.0, 0.0], "t": [0, 1]})
    c = pandas.DataFrame(
        {
            "a": numpy.linspace(0, 1, 120),
            "region": ["n", "s"] * 60,
            "t": [0, 1] * 60,
            "y": 0.0,
        }
    )
    return x, c


def test_identify_runs_the_adapter_and_documents_every_step(monkeypatch):
    adapter = install(
        monkeypatch,
        tests=(battery.Test("p1", "placebo", 0.5),),
        sens={"holds": True, "note": "ok"},
    )
    decisions = record_decisions(monkeypatch)
    x, c = frames()
    spec = block(
        strata=("region",),
        negative_controls=(causal.NegativeControl("outcome", "nco"),),
    )
    out = run.identify(
        spec,
        x,
        c,
        BUDGET,
        {"positivity": "A"},
        seed=9,
        learner="boosted",
        certified=("coverage_aipw", "overlap"),
    )
    assert adapter.seen[0] == (9, "boosted", pytest.approx(0.01))
    assert adapter.level == pytest.approx(0.96)
    report = out.documents["identification_report.json"]
    assert report["status"] == "identified"
    assert report["strategy"] == "backdoor_aipw"
    assert report["estimand"] == "ATE"
    assert report["level"] == pytest.approx(0.96)
    assert report["alpha_primary"] == pytest.approx(0.04)
    assert report["alpha_falsify"] == pytest.approx(0.01)
    assert report["estimate"] == {
        "point": 1.5,
        "se": 0.2,
        "low": 1.1,
        "high": 1.9,
        "n": 40,
        "detail": {"extra": 1},
    }
    assert report["design"] == {"prepared": True}
    assert report["multiverse_not_run"] == []
    assert report["verdict"] == {
        "hypothesis": "ATE of t on y = 0",
        "labelled": "causal, under the attested assumptions",
        "interval_excludes_zero": True,
    }
    assert "dag" not in report
    assert "subgroups" not in report
    assert report["notes"] == [UNCHECKED]
    falsification = out.documents["falsification_report.json"]
    assert falsification["gate"] == "battery_nco"
    assert falsification["ran"] is True
    assert falsification["passed"] is True
    assert falsification["alpha"] == pytest.approx(0.01)
    assert falsification["tests"] == [
        {
            "name": "p1",
            "kind": "placebo",
            "p_value": 0.5,
            "estimate": None,
            "se": None,
        }
    ]
    replication = falsification["replication"]
    assert set(replication["by_column"]) == {"region"}
    assert replication["passed"] is True
    assert out.documents["sensitivity_report.json"] == {
        "holds": True,
        "note": "ok",
    }
    facts = decisions[0]
    assert facts == tier.Inputs(
        identified=True,
        needed=(COVARIATES, "positivity", "no_interference"),
        attestors={"positivity": "A"},
        estimated_on_c=True,
        design_passed=True,
        design_gates=("overlap",),
        interval_gate="coverage_aipw",
        battery_ran=True,
        battery_passed=True,
        negative_control_declared=True,
        battery_gate="battery_nco",
        replication_passed=True,
        sensitivity_holds=True,
        sensitivity_gate="sensitivity_bound",
        single_unit=False,
        weak_instrument=False,
        certified=("coverage_aipw", "overlap"),
        adjustment_checked=False,
    )
    claims = out.documents["causal_claims.json"]
    assert claims["certified_gates_used"] == ["coverage_aipw", "overlap"]
    assert claims["assumptions"] == {
        COVARIATES: "",
        "positivity": "A",
        "no_interference": "",
    }
    assert claims["strategy"] == "backdoor_aipw"


def test_the_verdict_records_whether_the_interval_excludes_zero(monkeypatch):
    x, c = frames()
    cases = [
        (run.Est(1.0, 0.1, -1.0, 1.0, 5, {}), False),
        (run.Est(-1.0, 0.1, -2.0, -0.5, 5, {}), True),
        (run.Est(0.0, 0.1, 0.0, 1.0, 5, {}), False),
        (run.Est(1.0, float("nan"), float("nan"), float("nan"), 5, {}), None),
    ]
    for found, expected in cases:
        install(monkeypatch, found=found)
        out = run.identify(block(), x, c, BUDGET, certified=())
        report = out.documents["identification_report.json"]
        assert report["verdict"]["interval_excludes_zero"] is expected


def test_a_design_error_gives_a_refusal_with_the_error_text(monkeypatch):
    adapter = install(monkeypatch, prepare_error="boom")
    x, c = frames()
    out = run.identify(
        block(), x, c, BUDGET, certified=(), dag_checks={"hash": "h"}
    )
    report = out.documents["identification_report.json"]
    assert report["reason"] == "design could not be assessed: boom"
    assert report["dag"] == {"hash": "h"}
    assert report["estimate"] is None
    assert adapter.seen[1] == 0
    assert out.documents["causal_claims.json"]["tier"] is None


def test_a_refusal_from_the_design_stops_before_the_estimate(monkeypatch):
    adapter = install(monkeypatch, refusal_text="not here")
    x, c = frames()
    out = run.identify(block(), x, c, BUDGET, certified=())
    assert out.documents["identification_report.json"]["reason"] == "not here"
    assert adapter.seen[1] == 0
    assert out.documents["causal_claims.json"]["status"] == "not_identified"


def test_an_estimate_error_gives_a_refusal_with_the_error_text(monkeypatch):
    install(monkeypatch, estimate_error="too thin")
    x, c = frames()
    out = run.identify(
        block(), x, c, BUDGET, certified=(), dag_checks={"hash": "h"}
    )
    report = out.documents["identification_report.json"]
    assert report["reason"] == "too thin"
    assert report["dag"] == {"hash": "h"}
    assert out.documents["causal_claims.json"]["tier"] is None


def test_a_controls_error_is_a_note_and_the_battery_did_not_run(monkeypatch):
    install(monkeypatch, controls_error="no controls")
    decisions = record_decisions(monkeypatch)
    x, c = frames()
    out = run.identify(block(), x, c, BUDGET, certified=())
    report = out.documents["identification_report.json"]
    assert report["notes"][0] == "controls could not run: no controls"
    assert len(report["notes"]) == 2
    assert decisions[0].battery_ran is False
    assert decisions[0].battery_passed is False
    assert decisions[0].negative_control_declared is False
    assert out.documents["falsification_report.json"]["ran"] is False


def test_a_single_treated_unit_is_neither_replicated_nor_subgrouped(
    monkeypatch,
):
    install(monkeypatch, single=True)
    decisions = record_decisions(monkeypatch)
    x, c = frames()
    spec = block(
        strata=("region",),
        subgroups=(causal.Subgroup("region", ("n",)),),
    )
    out = run.identify(spec, x, c, BUDGET, certified=())
    falsification = out.documents["falsification_report.json"]
    assert falsification["replication"] == {"by_column": {}, "passed": None}
    assert "subgroups" not in out.documents["identification_report.json"]
    assert decisions[0].single_unit is True
    assert decisions[0].estimated_on_c is False
    assert decisions[0].replication_passed is None
    assert out.documents["sensitivity_report.json"] == {
        "holds": None,
        "note": "single treated unit: inference is the placebo permutation",
    }


def test_without_strata_replication_is_not_attempted(monkeypatch):
    install(monkeypatch)
    x, c = frames()
    out = run.identify(block(), x, c, BUDGET, certified=())
    falsification = out.documents["falsification_report.json"]
    assert falsification["replication"] == {"by_column": {}, "passed": None}


def test_the_weak_flag_of_the_adapter_reaches_the_tier(monkeypatch):
    install(monkeypatch, weak_after=True)
    decisions = record_decisions(monkeypatch)
    x, c = frames()
    run.identify(block(), x, c, BUDGET, certified=())
    assert decisions[0].weak_instrument is True


def test_declared_subgroups_are_estimated_and_reported(monkeypatch):
    install(monkeypatch)
    x, c = frames()
    spec = block(subgroups=(causal.Subgroup("region", ("n", "s")),))
    out = run.identify(spec, x, c, BUDGET, certified=("subgroup_holm",))
    report = out.documents["identification_report.json"]
    assert report["subgroups"]["gate"]["status"] == "gating"
    claims = out.documents["causal_claims.json"]
    assert claims["subgroups"]["gate"] == "gating"
    assert claims["subgroups"]["family_size"] == 2
    assert claims["subgroups"]["alpha"] == pytest.approx(0.04)


def test_unnamed_gates_come_from_the_shipped_certificate_sorted(monkeypatch):
    install(monkeypatch)
    monkeypatch.setattr(
        certified, "certified_gates", lambda: {"zeta", "alpha", "mid"}
    )
    decisions = record_decisions(monkeypatch)
    x, c = frames()
    spec = block(subgroups=(causal.Subgroup("region", ("n",)),))
    out = run.identify(spec, x, c, BUDGET)
    assert decisions[0].certified == ("alpha", "mid", "zeta")
    claims = out.documents["causal_claims.json"]
    assert claims["certified_gates_used"] == ["alpha", "mid", "zeta"]
    shipped = certified.subgroup_gate(run.subgroup_report)
    group = out.documents["identification_report.json"]["subgroups"]
    assert group["gate"]["status"] == shipped.status


def test_a_declared_dag_checks_the_set_and_the_reports_carry_the_extras(
    monkeypatch,
):
    install(monkeypatch)
    decisions = record_decisions(monkeypatch)
    x, c = frames()
    graph = scmle_dag(
        ("a", "t"), ("a", "y"), ("b", "t"), ("b", "y"), ("t", "y")
    )
    out = run.identify(
        block(),
        x,
        c,
        BUDGET,
        certified=(),
        dag=graph,
        dag_checks={"hash": "h"},
        attestation_report={"applied": []},
    )
    assert decisions[0].adjustment_checked is True
    report = out.documents["identification_report.json"]
    assert report["notes"] == []
    assert report["dag"] == {"hash": "h"}
    claims = out.documents["causal_claims.json"]
    assert claims["attestations"] == {"applied": []}
    plain = run.identify(block(), x, c, BUDGET, certified=())
    assert "dag" not in plain.documents["identification_report.json"]
    assert "attestations" not in plain.documents["causal_claims.json"]


def scmle_dag(*edges):
    return scmle.data.admissibility.dag.Dag.build(source="test", edges=edges)


def test_other_strategies_count_as_checked_and_use_their_own_gates(
    monkeypatch,
):
    install(monkeypatch, strategy="randomized_diff")
    decisions = record_decisions(monkeypatch)
    x, c = frames()
    spec = block(
        strategies=("randomized_diff",),
        assumptions=("randomized_assignment", "no_interference"),
    )
    out = run.identify(spec, x, c, BUDGET, certified=())
    facts = decisions[0]
    assert facts.adjustment_checked is True
    assert facts.interval_gate == "coverage_randomized"
    assert facts.battery_gate == "battery_nco"
    assert facts.design_gates == ()
    assert facts.sensitivity_gate is None
    assert facts.needed == ("randomized_assignment", "no_interference")
    assert out.documents["identification_report.json"]["notes"] == []
    assert out.documents["causal_claims.json"]["strategy"] == "randomized_diff"


def test_the_multiverse_that_was_not_run_is_listed(monkeypatch):
    install(monkeypatch)
    x, c = frames()
    spec = block(
        strategies=("backdoor_aipw", "randomized_diff"),
        assumptions=(*ASSUMPTIONS, "randomized_assignment"),
    )
    out = run.identify(spec, x, c, BUDGET, certified=())
    report = out.documents["identification_report.json"]
    assert report["multiverse_not_run"] == ["randomized_diff"]


def test_without_an_eligible_strategy_identify_refuses_with_the_dag_findings():
    x, c = frames()
    spec = block(assumptions=("positivity",))
    out = run.identify(spec, x, c, BUDGET, dag_checks={"hash": "h"})
    assert out.documents["identification_report.json"]["dag"] == {"hash": "h"}
    assert out.documents["causal_claims.json"]["status"] == "not_identified"


def test_design_only_documents_a_design_with_no_estimate(monkeypatch):
    adapter = install(monkeypatch)
    x = frames()[0]
    spec = block(strategies=("backdoor_aipw", "randomized_diff"))
    out = run.design_only(
        spec,
        x,
        BUDGET,
        seed=4,
        learner="boosted",
        dag_checks={"hash": "h"},
    )
    assert adapter.seen[0] == (4, "boosted", pytest.approx(0.01))
    assert run.NOT_ESTIMATED == (
        "no estimate was computed on the confirmation set: it was not "
        "available to the run, or the claim could not be registered, so the "
        "claim is unverified. `identify` computes the estimate on the single "
        "opening of C when the ledger hands it the released rows "
        "(`scmle.core.causal_claim`)."
    )
    assert out.documents["identification_report.json"] == {
        "status": "identified",
        "strategy": "backdoor_aipw",
        "estimand": "ATE",
        "estimate": None,
        "design": {"prepared": True},
        "multiverse_not_run": [],
        "notes": [run.NOT_ESTIMATED],
        "dag": {"hash": "h"},
    }
    assert out.documents["falsification_report.json"] == {
        "ran": False,
        "note": run.NOT_ESTIMATED,
    }
    assert out.documents["sensitivity_report.json"] == {
        "holds": None,
        "note": run.NOT_ESTIMATED,
    }
    assert out.decision == tier.Assignment(
        qualification.Tier.UNVERIFIED,
        "identified",
        dict.fromkeys(tier.CONDITIONS, False),
        (run.NOT_ESTIMATED,),
        "downgrade_tier",
    )
    claims = out.documents["causal_claims.json"]
    assert claims["tier"] == "unverified"
    assert claims["status"] == "identified"
    assert claims["assumptions"] == {
        COVARIATES: "",
        "positivity": "",
        "no_interference": "",
    }
    assert claims["conditions"] == dict.fromkeys(tier.CONDITIONS, False)
    assert claims["action"] == "downgrade_tier"
    assert adapter.seen[1] == 0


def test_design_only_without_a_dag_check_has_no_dag_entry(monkeypatch):
    install(monkeypatch)
    x = frames()[0]
    out = run.design_only(block(), x, BUDGET)
    assert "dag" not in out.documents["identification_report.json"]


def test_design_only_refuses_like_identify(monkeypatch):
    x = frames()[0]
    none = run.design_only(
        block(assumptions=("positivity",)), x, BUDGET, dag_checks={"hash": "h"}
    )
    assert none.documents["identification_report.json"]["dag"] == {"hash": "h"}
    assert none.documents["causal_claims.json"]["status"] == "not_identified"
    install(monkeypatch, prepare_error="boom")
    broken = run.design_only(block(), x, BUDGET, dag_checks={"hash": "h"})
    report = broken.documents["identification_report.json"]
    assert report["reason"] == "design could not be assessed: boom"
    assert report["dag"] == {"hash": "h"}
    install(monkeypatch, refusal_text="no overlap")
    refused = run.design_only(block(), x, BUDGET, dag_checks={"hash": "h"})
    report = refused.documents["identification_report.json"]
    assert report["reason"] == "no overlap"
    assert report["dag"] == {"hash": "h"}


# further pins found by mutation testing


def thin_overlap_rows(n, seed):
    frame = overlap_failing(n, seed)
    rng = numpy.random.default_rng(seed + 100)
    reach = 1 / (1 + numpy.exp(-4.0 * frame["a"]))
    return frame.assign(z=(rng.uniform(size=n) < reach).astype(float))


def test_the_declared_minimum_share_reaches_every_backdoor_estimate():
    x, c = thin_overlap_rows(3000, 0), thin_overlap_rows(3000, 1)
    spec = block(
        trim=0.2,
        min_share=0.99,
        negative_controls=(causal.NegativeControl("exposure", "z"),),
    )
    adapter = run.BackdoorEstimator(spec, x, 0, "linear", 0.01)
    adapter.prepare()
    assert adapter.refusal is None
    with pytest.raises(ValueError, match=r"declared minimum share 0\.99"):
        adapter.estimate(c, 0.95)
    with pytest.raises(ValueError, match=r"declared minimum share 0\.99"):
        adapter.estimate_for(c, 0.95)
    with pytest.raises(ValueError, match=r"declared minimum share 0\.99"):
        adapter.controls(c, 0.95)
    lenient = block(
        trim=0.2,
        min_share=0.1,
        negative_controls=(causal.NegativeControl("exposure", "z"),),
    )
    ok = run.BackdoorEstimator(lenient, x, 0, "linear", 0.01)
    ok.prepare()
    assert ok.estimate(c, 0.95).n < len(c)
    assert ok.estimate_for(c, 0.95).n < len(c)
    assert len(ok.controls(c, 0.95)) == 1


def test_the_boosted_learner_and_seed_reach_every_nuisance_fit():
    x = with_exposure(observational(800, 1), 50)
    c = with_exposure(observational(800, 2), 51)
    spec = backdoor_spec(
        sensitivity=None,
        negative_controls=(
            causal.NegativeControl("outcome", "nco"),
            causal.NegativeControl("exposure", "z"),
        ),
    )
    adapter = run.BackdoorEstimator(spec, x, 5, "boosted", 0.01)
    adapter.prepare()
    direct = aipw.fit_nuisances(x, ("a", "b"), "t", "y", 5, "boosted")
    assert adapter.design["nuisance_digest"] == direct.digest
    assert adapter.design["learner"] == "boosted"
    assert (
        direct.digest
        != aipw.fit_nuisances(x, ("a", "b"), "t", "y", 0, "boosted").digest
    )
    other = aipw.fit_nuisances(x, ("a", "b"), "t", "nco", 5, "boosted")
    found = aipw.estimate(other, c, "t", "nco", "ATE", 0.9, 0.0, 0.5)
    got = adapter.estimate_for(c, 0.9, "nco")
    assert (got.estimate, got.se) == (found.estimate, found.se)
    exposure = aipw.fit_nuisances(x, ("a", "b"), "z", "y", 5, "boosted")
    expected = aipw.estimate(exposure, c, "z", "y", "ATE", 0.9, 0.0, 0.5)
    tests = adapter.controls(c, 0.9)
    assert tests[1] == battery.wald_test(
        "negative_exposure:z",
        "negative_exposure",
        expected.estimate,
        expected.se,
    )


def test_the_randomized_estimate_for_another_outcome_keeps_level_and_counts():
    spec = block(strategies=("randomized_diff",))
    adapter = run.RandomizedEstimator(
        spec, observational(300, 1), 0, "linear", 0.01
    )
    c = observational(600, 2)
    out = adapter.estimate_for(c, 0.8, "nco")
    direct = aipw.difference_in_means(c, "t", "nco", 0.8)
    assert (out.estimate, out.se, out.low, out.high, out.n) == (
        direct.estimate,
        direct.se,
        direct.low,
        direct.high,
        direct.n,
    )
    assert out.detail == {}


def test_the_backdoor_adapter_names_the_missing_preparation_exactly():
    adapter = run.BackdoorEstimator(
        backdoor_spec(), observational(100, 1), 0, "linear", 0.01
    )
    with pytest.raises(ValueError, match=r"^prepare\(\) must run first$"):
        adapter.estimate(observational(100, 2), 0.95)
    with pytest.raises(ValueError, match=r"^prepare\(\) must run first$"):
        adapter.sensitivity(observational(100, 2), 0.95)


def test_the_adapters_name_their_missing_declarations_exactly():
    frame = did_frame(1, 50)
    did_adapter = run.DidEstimator(
        did_spec(unit=None), frame, 0, "linear", 0.01
    )
    with pytest.raises(
        ValueError, match=r"^did needs unit, time and adoption$"
    ):
        did_adapter.columns()
    iv_adapter = run.IvEstimator(
        iv_spec(instrument=None), iv_world(50, 1), 0, "linear", 0.01
    )
    with pytest.raises(ValueError, match=r"^iv needs an instrument$"):
        iv_adapter.fit(iv_world(50, 1), 0.95)
    rd_adapter = run.RdEstimator(
        rd_spec(cutoff=None), rd_world(50, 1), 0, "linear", 0.01
    )
    message = r"^rd needs a running variable and cutoff$"
    with pytest.raises(ValueError, match=message):
        rd_adapter.prepare()
    with pytest.raises(ValueError, match=message):
        rd_adapter.fit(rd_world(50, 1), 0.95)
    panel = synthetic_panel()
    synth_adapter = run.SynthEstimator(
        synth_spec(time=None), panel, 0, "linear", 0.01
    )
    with pytest.raises(ValueError, match=r"^synth needs unit and time$"):
        synth_adapter.estimate(panel, 0.95)
    two = panel.copy()
    two.loc[(two.u == 1) & (two.t >= 12), "d"] = 1.0
    many = run.SynthEstimator(synth_spec(), two, 0, "linear", 0.01)
    with pytest.raises(
        ValueError, match=r"^synthetic control needs exactly one treated unit$"
    ):
        many.estimate(two, 0.95)


def test_the_synthetic_interval_limits_are_not_a_number():
    panel = synthetic_panel()
    adapter = run.SynthEstimator(synth_spec(), panel, 0, "linear", 0.01)
    out = adapter.estimate(panel, 0.95)
    assert out.se != out.se
    assert out.low != out.low
    assert out.high != out.high


def test_the_rd_design_reads_the_balance_covariates_from_x():
    frame = rd_world(500, 1).drop(columns="cov")
    adapter = run.RdEstimator(rd_spec(), frame, 0, "linear", 0.01)
    with pytest.raises(KeyError, match="cov"):
        adapter.prepare()


def test_the_rd_bandwidth_sensitivity_is_compared_with_the_headline_on_a_null():
    rng = numpy.random.default_rng(4)
    n = 3000
    running = rng.uniform(-1, 1, n)
    frame = pandas.DataFrame(
        {
            "x": running,
            "y": 0.3 * running + rng.normal(scale=0.5, size=n),
            "cov": rng.normal(size=n),
        }
    )
    other = frame.assign(
        y=0.3 * frame["x"] + rng.normal(scale=0.5, size=n),
    )
    adapter = run.RdEstimator(rd_spec(), frame, 0, "linear", 0.01)
    adapter.prepare()
    out = adapter.sensitivity(other, 0.95)
    fit = rd.estimate(other, "x", "y", 0.0, 0.95, adapter.bandwidth, ())
    stable = [
        (row["low"] <= 0 <= row["high"]) == (fit.low <= 0 <= fit.high)
        for row in fit.sensitivity.values()
    ]
    assert fit.low <= 0 <= fit.high
    assert out["same_conclusion_at_all_multiples"] is all(stable)
    flipped = frame.assign(y=frame["y"] + 2.0 * (frame["x"] >= 0))
    jump = adapter.sensitivity(flipped, 0.95)
    refit = rd.estimate(flipped, "x", "y", 0.0, 0.95, adapter.bandwidth, ())
    assert jump["bandwidth_sensitivity"] == refit.sensitivity
    assert refit.low > 0


def small_effect_panel(tau):
    return did_frame(2, 400, tau=tau)


def test_the_did_sensitivity_holds_only_when_the_widened_interval_avoids_zero():
    spec = did_spec(sensitivity=causal.Confounding(m_max=0.01))
    adapter = run.DidEstimator(spec, did_frame(1, 100), 0, "linear", 0.01)
    positive = adapter.sensitivity(small_effect_panel(0.5), 0.95)
    assert 0 < positive["low"] < 1
    assert positive["holds"] is True
    negative = adapter.sensitivity(small_effect_panel(-0.5), 0.95)
    assert -1 < negative["high"] < 0
    assert negative["holds"] is True
    null = adapter.sensitivity(small_effect_panel(0.0), 0.95)
    assert null["low"] < 0 < null["high"] < 1
    assert null["holds"] is False
    mirror = adapter.sensitivity(
        small_effect_panel(0.0).assign(y=lambda f: -f["y"]), 0.95
    )
    assert -1 < mirror["low"] < 0 < mirror["high"]
    assert mirror["holds"] is False


def test_replication_goes_on_after_a_thin_or_failing_stratum_or_column():
    spec = block(strata=("region", "other"))
    thin = strata_frame(a=10, b=60, c=60)
    thin["other"] = ["p", "q"] * 65
    out = run.replication(stub_for(spec), thin, 0.95, 1.0, 0.05)
    assert set(out["by_column"]["region"]["strata"]) == {
        "region=b",
        "region=c",
    }
    failing = strata_frame(a=60, b=60, c=60)
    failing["bad"] = (failing["region"] == "a").astype(float)
    failing["other"] = ["p", "q"] * 90
    out = run.replication(stub_for(spec), failing, 0.95, 1.0, 0.05)
    assert set(out["by_column"]["region"]["strata"]) == {
        "region=b",
        "region=c",
    }
    skipped = strata_frame(a=10, b=10, r=120)
    skipped["other"] = ["p", "q"] * 70
    out = run.replication(stub_for(spec), skipped, 0.95, 1.0, 0.05)
    assert out["by_column"]["region"] == {
        "skipped": "fewer than two usable strata"
    }
    assert set(out["by_column"]["other"]["strata"]) == {"other=p", "other=q"}


def test_the_replication_level_is_the_one_asked_for():
    spec = block(strata=("region",))
    frame = strata_frame(a=60, b=60)
    out = run.replication(stub_for(spec), frame, 0.95, 1.0, 0.01)
    assert out["by_column"]["region"]["alpha"] == 0.01


def test_a_subgroup_of_exactly_the_minimum_rows_is_estimated():
    spec = block(subgroups=(causal.Subgroup("region", ("a", "b")),))
    adapter = stub_for(spec)
    report = run.subgroup_report(
        adapter, strata_frame(a=50, b=49), 0.93, spec.subgroups
    )
    first, second = report["hypotheses"]
    assert first["skipped"] is None
    assert first["n"] == 50
    assert second["skipped"] == "fewer than 50 rows"
    assert adapter.calls == [(50, 0.93)]


def test_the_subgroup_family_level_is_the_exact_complement_of_the_level():
    spec = block(subgroups=(causal.Subgroup("region", ("a",)),))
    report = run.subgroup_report(
        stub_for(spec), strata_frame(a=60), 0.9877, spec.subgroups
    )
    assert report["alpha"] == pytest.approx(0.0123, abs=1e-12)


def write_certificate(path, rows, passed=True):
    entry = {
        "method_hash": certified.subgroup_method_hash(run.subgroup_report),
        "passed": passed,
        "min_certified_rows": rows,
    }
    path.write_text(json.dumps({"gates": {"subgroup_holm": entry}}))


def test_the_certified_minimum_size_decides_which_subgroups_gate(
    monkeypatch, tmp_path
):
    store = tmp_path / "subgroups.json"
    write_certificate(store, 80)
    monkeypatch.setattr(certified, "SUBGROUP_STORE_PATH", store)
    found = {"hypotheses": [{"n": 100}, {"n": 80}, {"n": 79}, {"n": None}]}
    run.label_subgroups(found, ("subgroup_holm",))
    assert found["gate"]["status"] == "gating"
    assert found["gate"]["min_rows"] == 80
    assert [r["certified_size"] for r in found["hypotheses"]] == [
        True,
        True,
        False,
        False,
    ]
    unnamed = {"hypotheses": [{"n": 80}, {"n": 79}]}
    run.label_subgroups(unnamed, None)
    assert unnamed["gate"] == {
        "status": "gating",
        "reason": "family-wise error and power met their pre-registered floors",
        "min_rows": 80,
    }
    assert [r["certified_size"] for r in unnamed["hypotheses"]] == [True, False]
    left_out = {"hypotheses": [{"n": 80}]}
    run.label_subgroups(left_out, ("other",))
    assert left_out["gate"]["status"] == "descriptive"
    assert left_out["gate"]["min_rows"] == 80
    assert left_out["hypotheses"][0]["certified_size"] is False


def test_a_refusal_from_the_design_keeps_the_dag_findings(monkeypatch):
    install(monkeypatch, refusal_text="not here")
    x, c = frames()
    out = run.identify(
        block(), x, c, BUDGET, certified=(), dag_checks={"hash": "h"}
    )
    assert out.documents["identification_report.json"]["dag"] == {"hash": "h"}


def test_an_exposure_control_is_only_a_note_for_other_strategies(monkeypatch):
    install(monkeypatch)
    x, c = frames()
    spec = block(
        negative_controls=(causal.NegativeControl("exposure", "z"),),
    )
    out = run.identify(spec, x, c, BUDGET, certified=())
    assert out.documents["identification_report.json"]["notes"] == [UNCHECKED]
    install(monkeypatch, strategy="iv")
    instrumental = block(
        estimand="LATE",
        strategies=("iv",),
        instrument="z",
        assumptions=(
            "instrument_relevance",
            "exclusion",
            "independence",
            "monotonicity",
        ),
        negative_controls=(causal.NegativeControl("exposure", "z"),),
    )
    other = run.identify(instrumental, x, c, BUDGET, certified=())
    assert other.documents["identification_report.json"]["notes"] == [
        exposure_note("z")
    ]


def test_design_only_asks_the_declared_dag_about_the_adjustment_set(
    monkeypatch,
):
    install(monkeypatch)
    x = frames()[0]
    bad = scmle_dag(
        ("a", "t"),
        ("a", "y"),
        ("b", "t"),
        ("b", "y"),
        ("t", "y"),
        ("u", "t"),
        ("u", "y"),
    )
    refused = run.design_only(block(), x, BUDGET, dag=bad)
    assert refused.documents["causal_claims.json"]["status"] == "not_identified"
    declared = block(dag="dag.json")
    missing = run.design_only(declared, x, BUDGET)
    assert missing.documents["causal_claims.json"]["status"] == "not_identified"
    good = scmle_dag(("a", "t"), ("a", "y"), ("b", "t"), ("b", "y"), ("t", "y"))
    kept = run.design_only(declared, x, BUDGET, dag=good)
    assert kept.documents["causal_claims.json"]["status"] == "identified"
    assert kept.documents["causal_claims.json"]["strategy"] == "backdoor_aipw"


def test_the_family_level_keeps_the_denominator_limit_of_a_billion():
    spec = block(subgroups=(causal.Subgroup("region", ("a",)),))
    report = run.subgroup_report(
        stub_for(spec), strata_frame(a=60), 0.9894368340493388, spec.subgroups
    )
    assert report["alpha"] == 0.0105631659506612


def test_the_backdoor_sensitivity_reads_boolean_columns_as_numbers():
    spec = backdoor_spec(sensitivity=causal.Confounding("fixed", 0.3))
    adapter = prepared(spec)[0]
    c = confirm_rows()
    flagged = c.assign(t=c["t"].astype(bool))
    assert adapter.sensitivity(flagged, 0.9) == adapter.sensitivity(c, 0.9)
