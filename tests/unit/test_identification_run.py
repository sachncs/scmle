"""End to end: refusal, tiers, determinism and the held-out-only world."""

import json

import numpy as np
import pandas as pd
import pytest

from scmle.data.charter import causal
from scmle.data.identification import run
from scmle.data.ledger import budget as budget_module

BUDGET = budget_module.ErrorBudget.create(0.05, 0.1)
COVARIATES = "unconfoundedness_given:a,b"
ASSUMPTIONS = (COVARIATES, "positivity", "no_interference")
GATES = (
    "coverage_aipw",
    "overlap",
    "battery_nco",
    "sensitivity_bound",
    "coverage_did",
    "battery_pretrend",
    "sensitivity_did",
    "coverage_iv",
    "iv_strength",
    "coverage_rd",
    "density_rd",
    "battery_rd",
)
NAMED = dict.fromkeys(
    (
        "positivity",
        "no_interference",
        COVARIATES,
        "parallel_trends",
        "no_anticipation",
        "instrument_relevance",
        "exclusion",
        "independence",
        "monotonicity",
        "continuity_at_cutoff",
        "no_manipulation",
        "synthetic_hull",
    ),
    "A. Analyst, 2026-11-04",
)


def observational(n: int, seed: int, tau: float = 2.0, hidden: float = 0.0):
    """Confounded by observed a, b; optional hidden u hitting both outcomes."""
    rng = np.random.default_rng(seed)
    a, b = rng.standard_normal(n), rng.standard_normal(n)
    u = rng.standard_normal(n)
    e = 1 / (1 + np.exp(-(0.8 * a + 0.4 * b + hidden * u)))
    t = (rng.uniform(size=n) < e).astype(float)
    base = 1.0 * a - 0.5 * b + hidden * u
    y = tau * t + base + rng.standard_normal(n)
    nco = 0.7 * a - 0.3 * b + hidden * u + rng.standard_normal(n)
    region = np.where(rng.uniform(size=n) < 0.5, "n", "s")
    return pd.DataFrame(
        {"a": a, "b": b, "t": t, "y": y, "nco": nco, "region": region}
    )


def block(**changes) -> causal.CausalSpec:
    values = {
        "treatment": "t",
        "outcome": "y",
        "estimand": "ATE",
        "assumptions": ASSUMPTIONS,
        "strategies": ("backdoor_aipw",),
        "negative_controls": (causal.NegativeControl("outcome", "nco"),),
        "strata": ("region",),
        "sensitivity": causal.SensitivitySpec("strongest_observed", 1.0),
    }
    values.update(changes)
    return causal.CausalSpec(**values)


def go(spec, x, c, **kw):
    kw.setdefault("attestors", NAMED)
    kw.setdefault("certified", GATES)
    return run.identify(spec, x, c, BUDGET, **kw)


def test_no_eligible_strategy_prints_no_estimate():
    spec = block(assumptions=("positivity",))
    out = go(spec, observational(500, 1), observational(500, 2))
    claims = out.documents["causal_claims.json"]
    report = out.documents["identification_report.json"]
    assert claims["status"] == "not_identified" and claims["tier"] is None
    assert report["estimate"] is None and report["action"] == "refuse_claim"
    assert "associational" in report["note"]
    assert set(out.documents) == set(run.DOCUMENTS)


def test_an_identified_run_without_a_dag_is_attested_with_a_reason():
    out = go(block(), observational(3000, 1), observational(4000, 2))
    report = out.documents["identification_report.json"]
    claims = out.documents["causal_claims.json"]
    est = report["estimate"]
    assert est["low"] < 2.0 < est["high"]
    assert claims["tier"] == "attested"
    assert any("unchecked attestation" in r for r in claims["reasons"])
    assert (
        claims["hashes"]["identification_report.json"]
        == out.hashes["identification_report.json"]
    )


def test_with_a_checked_dag_and_certificates_the_claim_is_confirmed():
    from scmle.data.admissibility import dag as dag_module

    graph = dag_module.Dag.build(
        source="test",
        edges=(
            ("a", "t"),
            ("a", "y"),
            ("b", "t"),
            ("b", "y"),
            ("t", "y"),
            ("a", "nco"),
            ("b", "nco"),
        ),
    )
    spec = block(sensitivity=causal.SensitivitySpec("strongest_observed", 0.2))
    out = go(spec, observational(3000, 3), observational(5000, 4), dag=graph)
    claims = out.documents["causal_claims.json"]
    assert claims["tier"] == "confirmed", claims["reasons"]
    assert all(claims["conditions"].values())
    assert "does not say the assumptions are true" in claims["limits"]


def test_uncertified_gates_keep_the_claim_at_attested():
    out = go(
        block(), observational(2000, 5), observational(3000, 6), certified=()
    )
    assert out.documents["causal_claims.json"]["tier"] == "attested"


def test_the_held_out_estimate_alone_is_not_confirmed():
    # unobserved confounding, significant on C, no negative control, no strata
    spec = block(negative_controls=(), strata=(), sensitivity=None)
    x, c = (
        observational(3000, 7, hidden=1.2),
        observational(5000, 8, hidden=1.2),
    )
    out = go(spec, x, c)
    est = out.documents["identification_report.json"]["estimate"]
    assert est["low"] > 0
    assert out.documents["causal_claims.json"]["tier"] != "confirmed"


def test_a_shared_hidden_confounder_is_flagged_by_the_negative_control():
    x, c = (
        observational(4000, 9, hidden=1.5),
        observational(8000, 10, hidden=1.5),
    )
    out = go(block(), x, c)
    fals = out.documents["falsification_report.json"]
    assert fals["rejected"] == ["negative_outcome:nco"]
    assert out.documents["causal_claims.json"]["tier"] == "unverified"


def test_missing_attestors_make_the_claim_unverified():
    out = go(
        block(), observational(1500, 11), observational(2000, 12), attestors={}
    )
    assert out.documents["causal_claims.json"]["tier"] == "unverified"


def test_the_documents_are_deterministic_and_json_ready():
    first = go(block(), observational(1500, 13), observational(2000, 14))
    second = go(block(), observational(1500, 13), observational(2000, 14))
    assert first.hashes == second.hashes
    json.dumps(first.documents, allow_nan=False, default=str)


def test_overlap_failure_without_trimming_refuses_and_with_trimming_runs():
    rng = np.random.default_rng(0)

    def world(n):
        a = rng.standard_normal(n)
        e = 1 / (1 + np.exp(-4.0 * a))
        t = (rng.uniform(size=n) < e).astype(float)
        y = 2 * t + a + rng.standard_normal(n)
        return pd.DataFrame(
            {"a": a, "b": rng.standard_normal(n), "t": t, "y": y}
        )

    spec = block(negative_controls=(), strata=(), sensitivity=None)
    refused = go(spec, world(3000), world(3000))
    assert refused.documents["causal_claims.json"]["status"] == "not_identified"
    assert (
        "overlap gate failed"
        in refused.documents["identification_report.json"]["reason"]
    )
    trimmed = go(
        block(trim=0.05, negative_controls=(), strata=(), sensitivity=None),
        world(3000),
        world(3000),
    )
    assert (
        trimmed.documents["identification_report.json"]["estimate"] is not None
    )
    assert "estimand changed" in json.dumps(
        trimmed.documents["identification_report.json"]
    )


def test_randomized_difference_runs_and_is_labelled_not_applicable_for_sensitivity():
    spec = block(
        estimand="ATE",
        strategies=("randomized_diff",),
        assumptions=("randomized_assignment", "no_interference"),
    )
    x = observational(500, 1, hidden=0.0)
    c = x.assign(t=(np.arange(len(x)) % 2).astype(float))
    out = go(spec, x, c, attestors=NAMED | {"randomized_assignment": "R, 2026"})
    assert out.documents["sensitivity_report.json"]["applicable"] is False
    assert (
        out.documents["identification_report.json"]["strategy"]
        == "randomized_diff"
    )


def did_frame(seed, n=120, tau=1.5, periods=5):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n):
        g = 3.0 if u < n // 2 else np.nan
        level = rng.normal()
        region = "n" if u % 2 else "s"
        for t in range(periods):
            y = level + 0.3 * t + rng.normal(scale=0.7)
            nco = level + 0.3 * t + rng.normal(scale=0.7)
            if g == g and t >= g:
                y += tau
            rows.append((u, t, g, y, nco, region))
    return pd.DataFrame(rows, columns=["id", "t", "g", "y", "nco", "region"])


def test_difference_in_differences_runs_through_the_same_documents():
    spec = causal.CausalSpec(
        treatment="g",
        outcome="y",
        estimand="ATT_DID",
        unit="id",
        time="t",
        adoption="g",
        assumptions=("parallel_trends", "no_anticipation", "no_interference"),
        strategies=("did",),
        negative_controls=(causal.NegativeControl("outcome", "nco"),),
        strata=("region",),
        sensitivity=causal.SensitivitySpec(m_max=1.0),
    )
    out = go(spec, did_frame(1, 400), did_frame(2, 400))
    est = out.documents["identification_report.json"]["estimate"]
    assert est["low"] < 1.5 < est["high"]
    tests = [
        t["name"] for t in out.documents["falsification_report.json"]["tests"]
    ]
    assert "pretrend_wald" in tests
    assert out.documents["sensitivity_report.json"]["holds"] is True
    assert out.documents["causal_claims.json"]["tier"] in {
        "attested",
        "confirmed",
    }


def test_instrument_and_discontinuity_and_synthetic_control_run():
    rng = np.random.default_rng(3)

    def iv_world(n):
        z = rng.standard_normal(n)
        u = rng.standard_normal(n)
        d = z + u + rng.standard_normal(n)
        y = 2 * d + 1.5 * u + rng.standard_normal(n)
        return pd.DataFrame({"z": z, "d": d, "y": y})

    iv_spec = causal.CausalSpec(
        treatment="d",
        outcome="y",
        estimand="LATE",
        instrument="z",
        assumptions=(
            "instrument_relevance",
            "exclusion",
            "independence",
            "monotonicity",
        ),
        strategies=("iv",),
    )
    out = go(iv_spec, iv_world(3000), iv_world(3000))
    assert (
        out.documents["identification_report.json"]["estimate"]["low"]
        < 2
        < out.documents["identification_report.json"]["estimate"]["high"]
    )
    assert out.documents["causal_claims.json"]["tier"] == "attested"

    def rd_world(n):
        x = rng.uniform(-1, 1, n)
        y = 2 * x + (x >= 0) + rng.normal(scale=0.5, size=n)
        return pd.DataFrame({"x": x, "y": y})

    rd_spec = causal.CausalSpec(
        treatment="d",
        outcome="y",
        estimand="LATE_RD",
        running_variable="x",
        cutoff=0.0,
        assumptions=("continuity_at_cutoff", "no_manipulation"),
        strategies=("rd",),
    )
    out = go(rd_spec, rd_world(3000), rd_world(3000))
    assert (
        out.documents["identification_report.json"]["design"]["bandwidth"] > 0
    )
    manip = rd_world(3000)
    manip.loc[manip.x.between(-0.2, 0) & (np.arange(3000) % 2 == 0), "x"] *= -1
    refused = go(rd_spec, manip, rd_world(3000))
    assert (
        "density" in refused.documents["identification_report.json"]["reason"]
    )

    rows = []
    factors = rng.standard_normal((16, 2)).cumsum(axis=0) * 0.5
    loadings = rng.uniform(0, 1, (25, 2))
    for u in range(25):
        for t in range(16):
            y = factors[t] @ loadings[u] + rng.normal(scale=0.1)
            treated = float(u == 0 and t >= 12)
            rows.append((u, t, y + 3.0 * treated, treated))
    panel = pd.DataFrame(rows, columns=["u", "t", "y", "d"])
    sc_spec = causal.CausalSpec(
        treatment="d",
        outcome="y",
        estimand="ATT_SC",
        unit="u",
        time="t",
        assumptions=("synthetic_hull", "no_interference"),
        strategies=("synth",),
    )
    out = go(sc_spec, panel, panel.iloc[:0])
    assert (
        out.documents["identification_report.json"]["estimate"]["detail"][
            "p_value"
        ]
        <= 0.05
    )
    assert out.documents["causal_claims.json"]["tier"] == "attested"
    assert any(
        "nothing to hold out" in r
        for r in out.documents["causal_claims.json"]["reasons"]
    )


def test_c_is_only_used_after_the_design_step():
    # the adapter receives X alone in prepare(); C never reaches it
    import inspect

    assert "c" not in inspect.signature(run.Adapter.prepare).parameters
    assert "c" not in inspect.signature(run.Adapter.__init__).parameters


def test_alphas_split_the_confirmation_level_four_to_one():
    primary, falsify = run.alphas(BUDGET)
    assert primary == pytest.approx(0.04) and falsify == pytest.approx(0.01)
