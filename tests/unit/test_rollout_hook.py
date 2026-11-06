"""The decision layer's rollout through LifecycleManager and the ledger (F11)."""

from fractions import Fraction
import types

import numpy as np
import pytest
from test_decision_rollout import make_plan
from test_decision_rollout import traffic
from test_lifecycle_extras import approval_for
from test_lifecycle_extras import APPROVAL_KEY
from test_lifecycle_extras import bundle_of
from test_lifecycle_extras import traffic as rows

from scmle.contracts import errors
from scmle.data.ledger import budget as budget_module
from scmle.decision import rollout
from scmle.governance import deployment_ledger
from scmle.governance import lifecycle
from scmle.governance import registry as registry_module
from scmle.governance import rollout_hook


def staged(tmp_path):
    good = bundle_of("logistic_regression", rows())
    other = bundle_of("logistic_regression", rows(seed=1))
    store = registry_module.ModelRegistryStore(tmp_path / "registry")
    manager = lifecycle.LifecycleManager(store, "m", str(tmp_path / "work"))
    first = store.register("m", good.save(tmp_path / "e1"), trusted=True)
    store.promote(
        "m",
        first.version,
        approval=approval_for(first),
        signing_key=APPROVAL_KEY,
    )
    second = store.register("m", other.save(tmp_path / "e2"), trusted=True)
    store.promote("m", second.version, registry_module.CHALLENGER)
    return store, manager, second


def ledger_for():
    return deployment_ledger.DeploymentLedger.create(
        None, charter_hash="c" * 64, policy={"p": 1}, alpha_promo="1/20"
    )


def hook_for(tmp_path, **plan):
    store, manager, challenger = staged(tmp_path)
    ledger = ledger_for()
    hook = rollout_hook.RolloutHook(manager, make_plan(**plan), ledger)
    return types.SimpleNamespace(
        store=store,
        manager=manager,
        challenger=challenger,
        ledger=ledger,
        hook=hook,
    )


def test_a_started_experiment_takes_its_level_from_the_decision_budget(
    tmp_path,
):
    env = hook_for(tmp_path)
    ledger = env.ledger
    hook = env.hook

    first = hook.start("exp-1")
    second_hook = hook.budget
    assert first == pytest.approx(0.02 * 0.5)
    assert second_hook.assigned == (("exp-1", Fraction(1, 100)),)
    row = next(r for r in ledger.rows if r["kind"] == "rollout_experiment")
    assert row["body"]["experiment_id"] == "exp-1"
    assert row["body"]["level"] == "1/100"
    assert hook.log.rows[-1]["kind"] == "experiment_started"
    with pytest.raises(ValueError):
        hook.start("exp-1")


def test_the_level_is_spent_even_without_a_promotion(tmp_path):
    env = hook_for(tmp_path)
    hook = env.hook
    hook.start("exp-1")
    arm, primary, guards = traffic(500, 1, lift=0.0)
    hook.observe(arm, primary, guards)

    hook.start("exp-2")

    assert [name for name, level in hook.budget.assigned] == ["exp-1", "exp-2"]
    assert hook.budget.spent == Fraction(1, 100) + Fraction(1, 300)


def test_a_promotion_decision_goes_through_the_manager_and_needs_approval(
    tmp_path,
):
    env = hook_for(tmp_path)
    store = env.store
    challenger = env.challenger
    ledger = env.ledger
    hook = env.hook
    hook.start("exp-1")
    arm, primary, guards = traffic(40000, 3, lift=0.05)

    decision = hook.observe(arm, primary, guards)

    assert decision.status == rollout.PROMOTE
    kinds = [r["kind"] for r in ledger.rows]
    assert "promotion_proposal" in kinds
    assert store.resolve("m").version == 1  # a proposal moves nothing
    record = hook.promote(approval_for(challenger), signing_key=APPROVAL_KEY)
    assert record.version == challenger.version
    assert store.resolve("m").version == challenger.version
    assert hook.budget.monitored == (f"m:v{challenger.version}",)
    ledger.verify()
    assert hook.log.verify()


def test_a_hold_cannot_be_promoted(tmp_path):
    env = hook_for(tmp_path)
    challenger = env.challenger
    hook = env.hook
    hook.start("exp-1")
    arm, primary, guards = traffic(300, 4, lift=0.0)
    hook.observe(arm, primary, guards)

    with pytest.raises(errors.ScmleError, match="promote"):
        hook.promote(approval_for(challenger), signing_key=APPROVAL_KEY)


def test_a_rollback_decision_restores_the_previous_champion(tmp_path):
    env = hook_for(tmp_path, n_max=5000, tolerance=0.01)
    store = env.store
    challenger = env.challenger
    ledger = env.ledger
    hook = env.hook
    hook.start("exp-1")
    arm, primary, guards = traffic(40000, 5, lift=0.05)
    hook.observe(arm, primary, guards)
    hook.promote(approval_for(challenger), signing_key=APPROVAL_KEY)
    rng = np.random.default_rng(6)
    n = 20000
    watch_arm = (rng.random(n) < 0.5).astype(float)
    outcome = (rng.random(n) < 0.30 - 0.15 * watch_arm).astype(float)

    status = hook.watch(watch_arm, outcome)

    assert status == rollout.ROLLBACK
    assert "rollback_proposal" in [r["kind"] for r in ledger.rows]
    assert store.resolve("m").version == challenger.version
    restored = hook.roll_back(
        approval_for(store.resolve("m", 1)), signing_key=APPROVAL_KEY
    )
    assert restored == 1
    assert store.resolve("m").version == 1
    ledger.verify()


def test_no_rollback_without_a_rollback_decision(tmp_path):
    env = hook_for(tmp_path)
    store = env.store
    challenger = env.challenger
    hook = env.hook
    hook.start("exp-1")
    arm, primary, guards = traffic(40000, 5, lift=0.05)
    hook.observe(arm, primary, guards)
    hook.promote(approval_for(challenger), signing_key=APPROVAL_KEY)

    with pytest.raises(errors.ScmleError, match="rollback"):
        hook.roll_back(approval_for(store.resolve("m", 1)), APPROVAL_KEY)


def test_the_budget_survives_in_the_ledger_rows(tmp_path):
    env = hook_for(tmp_path)
    ledger = env.ledger
    hook = env.hook
    hook.start("exp-1")
    row = next(r for r in ledger.rows if r["kind"] == "rollout_experiment")
    restored = budget_module.DecisionBudget.from_dict(row["body"]["budget"])
    assert restored == hook.budget
