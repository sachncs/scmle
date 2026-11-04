"""Rollout design and the promote, hold and rollback rule."""

from fractions import Fraction
import math

import numpy as np
import pytest

from scmle.data.charter import decision as charter
from scmle.decision import design
from scmle.decision import rollout


def make_plan(**overrides):
    values = {
        "unit": "user",
        "primary": "conversion",
        "assignment_prob": 0.5,
        "delta_min": 0.01,
        "tolerance": 0.01,
        "n_max": 40000,
        "alpha_promote": Fraction(1, 50),
        "alpha_rollback": Fraction(1, 100),
        "guardrails": (charter.Guardrail("bounce", margin=0.05),),
    }
    values.update(overrides)
    return charter.RolloutPlan(**values)


def traffic(n, seed, lift=0.04, bounce_lift=0.0, base=0.10):
    rng = np.random.default_rng(seed)
    arm = (rng.random(n) < 0.5).astype(float)
    primary = (rng.random(n) < base + lift * arm).astype(float)
    bounce = (rng.random(n) < 0.2 + bounce_lift * arm).astype(float)
    return arm, primary, {"bounce": bounce}


def test_assignment_is_reproducible_and_close_to_the_probability():
    ids = [f"user-{i}" for i in range(20000)]
    first = design.assign(ids, "salt", 0.3)
    assert (first == design.assign(ids, "salt", 0.3)).all()
    assert not (first == design.assign(ids, "other", 0.3)).all()
    assert first.mean() == pytest.approx(0.3, abs=0.01)
    one = design.assign(["u1"], "s", 0.5)
    assert one[0] in (0.0, 1.0)
    with pytest.raises(ValueError):
        design.assign(ids, " ", 0.3)
    with pytest.raises(ValueError):
        design.assign(ids, "s", 1.0)


def test_the_fixed_horizon_size_uses_the_closed_form_and_the_design_effect():
    base = design.fixed_horizon_rows(0.10, 0.02, alpha=0.05, power=0.8)
    assert base == 2 * 3841 or abs(base - 7682) < 400
    assert design.fixed_horizon_rows(
        0.10, 0.02, alpha=0.05, power=0.8, design_effect=2.0
    ) == pytest.approx(2 * base, abs=2)
    with pytest.raises(ValueError):
        design.fixed_horizon_rows(
            0.1, 0.02, alpha=0.05, power=0.8, design_effect=0.5
        )


def test_the_exact_binomial_check_and_the_anytime_check_agree_on_a_bad_split():
    assert design.srm_p_value(500, 1000, 0.5) == pytest.approx(1.0)
    assert design.srm_p_value(560, 1000, 0.5) < 0.001
    assert design.srm_rejects(5200, 10000, 0.5, 0.001) is False
    assert design.srm_rejects(5600, 10000, 0.5, 0.001) is True
    with pytest.raises(ValueError):
        design.srm_p_value(11, 10, 0.5)


def test_the_srm_bayes_factor_is_a_martingale_under_correct_assignment():
    n = 12
    total = 0.0
    for k in range(n + 1):
        weight = math.comb(n, k) * 0.3**k * 0.7 ** (n - k)
        total += weight * math.exp(design.srm_log_bayes_factor(k, n, 0.3))
    assert total == pytest.approx(1.0, abs=1e-9)


def test_a_clear_win_is_promoted_with_a_bound_above_the_minimum_effect():
    plan = make_plan()
    run = rollout.Rollout(plan, "e1", 0.02)
    arm, primary, guards = traffic(40000, 0)
    decision = run.observe(arm, primary, guards)
    assert decision.status == rollout.PROMOTE
    assert decision.tier == "confirmed"
    assert decision.lower_bound > plan.delta_min
    assert decision.units < 40000
    again = run.observe(arm, primary, guards)
    assert again.units == decision.units and again.status == rollout.PROMOTE


def test_batch_size_does_not_change_the_decision_time():
    plan = make_plan()
    arm, primary, guards = traffic(40000, 1)
    whole = rollout.Rollout(plan, "a", 0.02).observe(arm, primary, guards)
    run = rollout.Rollout(plan, "b", 0.02)
    last = None
    for i in range(0, 40000, 997):
        last = run.observe(
            arm[i : i + 997],
            primary[i : i + 997],
            {"bounce": guards["bounce"][i : i + 997]},
        )
        if last.status != rollout.HOLD:
            break
    assert last.status == whole.status and last.units == whole.units


def test_an_effect_below_the_minimum_ends_in_hold_final():
    plan = make_plan(n_max=8000)
    run = rollout.Rollout(plan, "e", 0.02)
    arm, primary, guards = traffic(8000, 2, lift=0.003)
    decision = run.observe(arm, primary, guards)
    assert (
        decision.status == rollout.HOLD_FINAL and decision.tier == "unverified"
    )
    assert decision.units == 8000


def test_a_harmful_guardrail_blocks_promotion():
    plan = make_plan(n_max=30000)
    run = rollout.Rollout(plan, "e", 0.02)
    arm, primary, guards = traffic(30000, 3, bounce_lift=0.08)
    assert run.observe(arm, primary, guards).status == rollout.HOLD_FINAL


def test_a_lower_is_worse_guardrail_is_read_in_the_other_direction():
    guard = charter.Guardrail("rating", 0.05, higher_is_worse=False)
    plan = make_plan(guardrails=(guard,), n_max=30000)
    arm, primary, _ = traffic(30000, 4)
    rng = np.random.default_rng(44)
    worse = (rng.random(30000) < 0.8 - 0.10 * arm).astype(float)
    better = (rng.random(30000) < 0.8 + 0.0 * arm).astype(float)
    assert (
        rollout.Rollout(plan, "x", 0.02)
        .observe(arm, primary, {"rating": worse})
        .status
        == rollout.HOLD_FINAL
    )
    assert (
        rollout.Rollout(plan, "y", 0.02)
        .observe(arm, primary, {"rating": better})
        .status
        == rollout.PROMOTE
    )


def test_a_broken_allocation_halts_inference():
    plan = make_plan(n_max=20000)
    rng = np.random.default_rng(5)
    arm = (rng.random(20000) < 0.56).astype(float)
    primary = (rng.random(20000) < 0.10 + 0.04 * arm).astype(float)
    bounce = (rng.random(20000) < 0.2).astype(float)
    decision = rollout.Rollout(plan, "e", 0.02).observe(
        arm, primary, {"bounce": bounce}
    )
    assert decision.status == rollout.HOLD_INVALID
    assert decision.tier == "unverified"


def test_a_missing_guardrail_series_is_refused():
    run = rollout.Rollout(make_plan(), "e", 0.02)
    with pytest.raises(ValueError, match="missing guardrail"):
        run.observe([1.0, 0.0], [1.0, 0.0], {})
    with pytest.raises(ValueError):
        rollout.Rollout(make_plan(), "e", 1.0)


def test_the_boundary_null_is_promoted_rarely_under_peeking():
    plan = make_plan(guardrails=(), n_max=2500, delta_min=0.02)
    hits = 0
    runs = 150
    for seed in range(runs):
        rng = np.random.default_rng(1000 + seed)
        arm = (rng.random(2500) < 0.5).astype(float)
        primary = (rng.random(2500) < 0.10 + 0.02 * arm).astype(float)
        decision = rollout.Rollout(plan, "e", 0.05).observe(arm, primary)
        hits += decision.status == rollout.PROMOTE
    assert hits <= 0.05 * runs + 3 * math.sqrt(0.05 * 0.95 * runs)


def test_a_worse_champion_is_rolled_back_and_a_fine_one_is_kept():
    rng = np.random.default_rng(6)
    arm = (rng.random(30000) < 0.5).astype(float)
    losing = (rng.random(30000) < 0.20 - 0.05 * arm).astype(float)
    monitor = rollout.RollbackMonitor(0.01, 0.01)
    assert monitor.observe(arm, losing) == rollout.ROLLBACK
    assert monitor.done_at < 30000
    assert monitor.observe(arm, losing) == rollout.ROLLBACK
    fine = (rng.random(30000) < 0.20 + 0.0 * arm).astype(float)
    assert (
        rollout.RollbackMonitor(0.01, 0.01).observe(arm, fine) == rollout.HOLD
    )


def test_the_decision_log_chain_detects_edits_and_round_trips(tmp_path):
    log = rollout.DecisionLog()
    log.append("assign", {"experiment": "e1", "level": "1/100"})
    log.append("decision", {"status": "promote"})
    assert log.verify()
    path = tmp_path / "rollout_decisions.jsonl"
    log.write(path)
    again = rollout.DecisionLog.read(path)
    assert again.rows == log.rows and again.verify()
    again.rows[0]["body"]["level"] = "1/10"
    assert not again.verify()


def test_the_simulated_stopping_time_is_reported_with_its_crossed_share():
    plan = make_plan(guardrails=(), n_max=4000, delta_min=0.02)
    found = design.simulate_stopping(plan, 0.10, 0.08, runs=200, seed=1)
    assert found["crossed_share"] > 0.9
    assert 0 < found["median_units"] < 4000
    none = design.simulate_stopping(plan, 0.10, 0.0, runs=100, seed=1)
    assert none["crossed_share"] == 0.0 and none["mean_units"] is None
