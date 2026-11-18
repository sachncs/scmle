"""Select, estimate, falsify and tier one chartered causal claim."""

import pandas
import pytest

from scmle.assurance.charter import causal
from scmle.assurance.ledger import budget
from scmle.causal.identification import battery
from scmle.causal.identification import select
from scmle.causal.identification import tier
from scmle.causal.identification.run import adjustment
from scmle.causal.identification.run import analysis
from scmle.causal.identification.run import documents
from scmle.causal.identification.run import estimator
from scmle.causal.identification.run import exogeneity
from scmle.causal.identification.run import panel
from tests.unit.test_identification_run import observational

BUDGET = budget.ErrorBudget.create(0.05, 0.1)
ASSUMPTIONS = ("unconfoundedness_given:a,b", "positivity", "no_interference")


def block(**changes):
    values = {
        "treatment": "t",
        "outcome": "y",
        "estimand": "ATE",
        "assumptions": ASSUMPTIONS,
        "strategies": ("backdoor_aipw",),
    }
    values.update(changes)
    return causal.Question(**values)


class Stub(estimator.Estimator):
    """A fixed estimate; settings come from class attributes."""

    name = "backdoor_aipw"
    prepare_error = None
    refuse_with = None
    estimate_error = None
    controls_error = None
    tests = ()

    def prepare(self):
        if self.prepare_error:
            raise ValueError(self.prepare_error)
        self.design = {"stub": True}
        self.refusal = self.refuse_with

    def estimate(self, c, level):
        if self.estimate_error:
            raise ValueError(self.estimate_error)
        return estimator.Est(1.0, 0.1, 0.8, 1.2, len(c), {"k": 1})

    def controls(self, c, level):
        if self.controls_error:
            raise ValueError(self.controls_error)
        return list(self.tests)


@pytest.fixture
def stubbed(monkeypatch):
    monkeypatch.setitem(analysis.ADAPTERS, "backdoor_aipw", Stub)
    return Stub


def test_the_gate_tables_are_pinned():
    assert analysis.INTERVAL_GATES == {
        "backdoor_aipw": "coverage_aipw",
        "randomized_diff": "coverage_randomized",
        "did": "coverage_did",
        "iv": "coverage_iv",
        "rd": "coverage_rd",
        "synth": "size_synth",
    }
    assert analysis.DESIGN_GATES == {
        "backdoor_aipw": ("overlap",),
        "iv": ("iv_strength",),
        "rd": ("density_rd",),
    }
    assert analysis.BATTERY_GATES == {
        "backdoor_aipw": "battery_nco",
        "randomized_diff": "battery_nco",
        "did": "battery_pretrend",
        "iv": "battery_iv",
        "rd": "battery_rd",
        "synth": "battery_synth",
    }
    assert analysis.SENSITIVITY_GATES == {
        "backdoor_aipw": "sensitivity_bound",
        "did": "sensitivity_did",
    }


def test_the_adapter_registry_maps_each_strategy_to_its_class():
    assert {
        "backdoor_aipw": adjustment.Backdoor,
        "randomized_diff": adjustment.RandomizedEstimator,
        "did": panel.DiD,
        "iv": exogeneity.Instrument,
        "rd": exogeneity.RdEstimator,
        "synth": panel.Synth,
    } == analysis.ADAPTERS
    assert set(analysis.ADAPTERS) == set(select.STRATEGIES)


def test_the_alphas_split_four_to_one():
    primary, falsify = analysis.alphas(BUDGET)
    assert primary == pytest.approx(0.04, abs=1e-12)
    assert falsify == pytest.approx(0.01, abs=1e-12)
    assert analysis.alphas(
        budget.ErrorBudget.create(0.1, 0.1)
    ) == pytest.approx((0.08, 0.02), abs=1e-12)


def test_a_single_unit_fragility_is_the_placebo_note():
    adapter = Stub(block(), pandas.DataFrame(), 0, "linear", 0.01)
    assert analysis.fragility(adapter, None, 0.9) == {
        "holds": None,
        "note": "no sensitivity analysis for this strategy",
    }
    adapter.single_unit = True
    assert analysis.fragility(adapter, None, 0.9) == {
        "holds": None,
        "note": "single treated unit: inference is the placebo permutation",
    }


def test_caveats_name_unsupported_exposures_then_the_unchecked_set():
    spec = block(
        controls=(
            causal.Control("exposure", "e1"),
            causal.Control("outcome", "o"),
            causal.Control("exposure", "e2"),
        )
    )
    unchecked = (
        "the adjustment set is an unchecked attestation (no DAG): the tier "
        "cannot exceed attested"
    )
    assert analysis.caveats(spec, "iv", checked=False) == [
        (
            "negative-control exposure e1 is only supported for backdoor_aipw "
            "and was not run"
        ),
        (
            "negative-control exposure e2 is only supported for backdoor_aipw "
            "and was not run"
        ),
        unchecked,
    ]
    assert analysis.caveats(spec, "backdoor_aipw", checked=False) == [unchecked]
    assert analysis.caveats(spec, "backdoor_aipw", checked=True) == []


def test_a_design_returns_the_prepared_adapter(stubbed):
    spec = block()
    plan = select.select(spec)
    adapter = analysis.design(
        plan, spec, "backdoor_aipw", "X", 3, "boosted", 0.02, None
    )
    assert isinstance(adapter, Stub)
    assert (adapter.x, adapter.seed, adapter.learner) == ("X", 3, "boosted")
    assert adapter.alpha_falsify == 0.02
    assert adapter.design == {"stub": True}


def test_a_design_error_or_refusal_returns_the_refusal(stubbed, monkeypatch):
    spec = block()
    plan = select.select(spec)
    monkeypatch.setattr(Stub, "prepare_error", "broken")
    out = analysis.design(
        plan, spec, "backdoor_aipw", "X", 0, "linear", 0.01, {"d": 1}
    )
    report = out.documents["identification_report.json"]
    assert report["reason"] == "design could not be assessed: broken"
    assert report["dag"] == {"d": 1}
    monkeypatch.setattr(Stub, "prepare_error", None)
    monkeypatch.setattr(Stub, "refuse_with", "no overlap")
    out = analysis.design(
        plan, spec, "backdoor_aipw", "X", 0, "linear", 0.01, None
    )
    assert isinstance(out, documents.Outcome)
    assert out.documents["identification_report.json"]["reason"] == "no overlap"


def trial(spec=None, adapter=None):
    spec = spec or block()
    plan = select.select(spec)
    adapter = adapter or Stub(spec, pandas.DataFrame(), 0, "linear", 0.01)
    primary = estimator.Est(1.0, 0.1, 0.8, 1.2, 10, {"k": 1})
    return analysis.Trial(
        plan, spec, "backdoor_aipw", adapter, primary, 0.96, 0.04, 0.01
    )


def test_a_falsification_runs_the_battery_without_strata():
    found = trial()
    found.adapter.tests = (battery.wald("n", "negative_outcome", 0.0, 1.0),)
    report, rep, notes = analysis.falsify(found, pandas.DataFrame())
    assert report.ran is True and report.passed is True
    assert rep == {"by_column": {}, "passed": None}
    assert notes == []


def test_a_falsification_notes_failing_controls():
    found = trial()
    found.adapter.controls_error = "no nco"
    report, ignored, notes = analysis.falsify(found, pandas.DataFrame())
    assert notes == ["controls could not run: no nco"]
    assert report.ran is False


def test_a_falsification_replicates_over_declared_strata():
    spec = block(strata=("region",))
    found = trial(spec, Stub(spec, pandas.DataFrame(), 0, "linear", 0.01))
    rows = pandas.DataFrame({"region": ["a"] * 50 + ["b"] * 50})
    ignored, rep, ignored = analysis.falsify(found, rows)
    assert rep["passed"] is True
    found.adapter.single_unit = True
    ignored, rep, ignored = analysis.falsify(found, rows)
    assert rep == {"by_column": {}, "passed": None}


def test_a_grade_is_the_tier_decision_of_the_facts(monkeypatch):
    seen = []
    monkeypatch.setattr(tier, "decide", lambda facts: seen.append(facts) or "D")
    found = trial(block(controls=(causal.Control("outcome", "nco"),)))
    found.adapter.weak = True
    report = battery.run_battery([], 0.01)
    decision, needed, checked = analysis.grade(
        found, report, {"passed": False}, {"holds": True}, {"p": "A"}, ("g",)
    )
    assert decision == "D"
    assert needed == documents.prerequisites("backdoor_aipw", found.spec)
    assert checked is False
    facts = seen[0]
    assert facts.identified is True
    assert facts.attestors == {"p": "A"}
    assert facts.estimated_on_c is True
    assert facts.design_gates == ("overlap",)
    assert facts.interval_gate == "coverage_aipw"
    assert facts.battery_ran is False
    assert facts.negative_control_declared is True
    assert facts.battery_gate == "battery_nco"
    assert facts.replication_passed is False
    assert facts.sensitivity_holds is True
    assert facts.sensitivity_gate == "sensitivity_bound"
    assert facts.weak_instrument is True
    assert facts.certified == ("g",)
    assert facts.adjustment_checked is False


def test_a_chronicle_reports_the_estimate_and_verdict():
    found = trial()
    report = analysis.chronicle(found, ["n"], {"h": 1}, {"s": 2})
    assert report["status"] == select.IDENTIFIED
    assert report["strategy"] == "backdoor_aipw"
    assert report["level"] == 0.96
    assert report["alpha_primary"] == 0.04
    assert report["alpha_falsify"] == 0.01
    assert report["estimate"] == {
        "point": 1.0,
        "se": 0.1,
        "low": 0.8,
        "high": 1.2,
        "n": 10,
        "detail": {"k": 1},
    }
    assert report["verdict"]["interval_excludes_zero"] is True
    assert report["verdict"]["hypothesis"] == "ATE of t on y = 0"
    assert report["notes"] == ["n"]
    assert report["dag"] == {"h": 1}
    assert report["subgroups"] == {"s": 2}
    plain = analysis.chronicle(found, [], None, None)
    assert "dag" not in plain and "subgroups" not in plain


def test_a_chronicle_of_an_interval_without_limits_has_no_verdict():
    found = trial()
    nan = float("nan")
    found = analysis.Trial(
        found.plan,
        found.spec,
        found.strategy,
        found.adapter,
        estimator.Est(1.0, nan, nan, nan, 3),
        0.96,
        0.04,
        0.01,
    )
    assert (
        analysis.chronicle(found, [], None, None)["verdict"][
            "interval_excludes_zero"
        ]
        is None
    )


def test_an_identification_without_a_strategy_is_refused():
    out = analysis.identify(
        block(assumptions=("positivity",)),
        None,
        None,
        BUDGET,
        dag_checks={"x": 1},
    )
    assert out.documents["causal_claims.json"]["status"] == "not_identified"
    assert out.documents["identification_report.json"]["dag"] == {"x": 1}


def test_an_identification_refuses_an_estimate_error(stubbed, monkeypatch):
    monkeypatch.setattr(Stub, "estimate_error", "thin arm")
    out = analysis.identify(
        block(), "X", pandas.DataFrame(), BUDGET, certified=()
    )
    assert out.documents["identification_report.json"]["reason"] == "thin arm"


def test_an_identification_estimates_at_the_primary_level(stubbed):
    c = pandas.DataFrame({"a": range(7)})
    out = analysis.identify(
        block(),
        "X",
        c,
        BUDGET,
        attestors={"positivity": "A"},
        certified=(),
        attestation_report={"ok": True},
    )
    report = out.documents["identification_report.json"]
    assert report["level"] == pytest.approx(0.96)
    assert report["estimate"]["n"] == 7
    claims = out.documents["causal_claims.json"]
    assert claims["attestations"] == {"ok": True}
    assert claims["assumptions"]["positivity"] == "A"
    falsification = out.documents["falsification_report.json"]
    assert falsification["gate"] == "battery_nco"
    assert falsification["replication"] == {"by_column": {}, "passed": None}
    assert report["notes"][-1].startswith("the adjustment set is an unchecked")


def test_a_design_only_run_never_estimates():
    assert analysis.NOT_ESTIMATED.startswith("no estimate was computed")
    out = analysis.design_only(
        block(), observational(800, 1), BUDGET, dag_checks={"d": 2}
    )
    report = out.documents["identification_report.json"]
    assert report["estimate"] is None
    assert report["notes"] == [analysis.NOT_ESTIMATED]
    assert report["dag"] == {"d": 2}
    assert report["strategy"] == "backdoor_aipw"
    claims = out.documents["causal_claims.json"]
    assert claims["tier"] == "unverified"
    assert claims["conditions"] == dict.fromkeys(tier.CONDITIONS, False)
    assert claims["reasons"] == [analysis.NOT_ESTIMATED]
    assert claims["action"] == "downgrade_tier"
    assert out.documents["falsification_report.json"] == {
        "ran": False,
        "note": analysis.NOT_ESTIMATED,
    }
    assert out.documents["sensitivity_report.json"] == {
        "holds": None,
        "note": analysis.NOT_ESTIMATED,
    }


def test_a_design_only_run_refuses_unidentified_or_failed_designs(
    stubbed, monkeypatch
):
    refused = analysis.design_only(
        block(assumptions=("positivity",)), None, BUDGET
    )
    assert refused.documents["causal_claims.json"]["status"] == "not_identified"
    monkeypatch.setattr(Stub, "refuse_with", "no")
    failed = analysis.design_only(block(), "X", BUDGET)
    assert failed.documents["identification_report.json"]["reason"] == "no"
