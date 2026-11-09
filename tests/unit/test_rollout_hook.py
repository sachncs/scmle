"""The decision layer's rollout through LifecycleManager and the ledger."""

from fractions import Fraction
import json
import types

import numpy
import pytest
from test_decision_rollout import make_plan
from test_decision_rollout import traffic
import test_lifecycle_extras
from test_lifecycle_extras import approval_for
from test_lifecycle_extras import APPROVAL_KEY
from test_lifecycle_extras import bundle_of

from scmle.contracts import errors
from scmle.data.ledger import budget
from scmle.decision import rollout
from scmle.governance import deployment_ledger
from scmle.governance import lifecycle
from scmle.governance import registry
from scmle.governance import rollout_hook


def staged(tmp_path):
    good = bundle_of("logistic_regression", test_lifecycle_extras.traffic())
    other = bundle_of(
        "logistic_regression", test_lifecycle_extras.traffic(seed=1)
    )
    store = registry.ModelRegistryStore(tmp_path / "registry")
    manager = lifecycle.LifecycleManager(store, "m", str(tmp_path / "work"))
    first = store.register("m", good.save(tmp_path / "e1"), trusted=True)
    store.promote(
        "m",
        first.version,
        approval=approval_for(first),
        signing_key=APPROVAL_KEY,
    )
    second = store.register("m", other.save(tmp_path / "e2"), trusted=True)
    store.promote("m", second.version, registry.CHALLENGER)
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


def test_assignment_rows_hold_arm_and_probability_per_unit(tmp_path):
    env = hook_for(tmp_path, assignment_prob=0.3)
    ledger = env.ledger
    hook = env.hook
    hook.start("exp-1")
    ids = [f"user-{i}" for i in range(50)]

    arms = hook.assign(ids)

    rows_ = [r for r in ledger.rows if r["kind"] == "rollout_assignment"]
    assert len(rows_) == 1
    body = rows_[0]["body"]
    assert body["experiment_id"] == "exp-1"
    assert body["units"] == ids
    assert body["arm"] == [int(a) for a in arms]
    expected = [0.3 if a else 0.7 for a in arms]
    assert body["probability"] == pytest.approx(expected)
    ledger.verify()


def test_an_assignment_for_an_unstarted_experiment_is_refused(tmp_path):
    env = hook_for(tmp_path)
    ledger = env.ledger
    hook = env.hook
    with pytest.raises(deployment_ledger.DeploymentLedgerError):
        ledger.commit(
            "rollout_assignment",
            {
                "experiment_id": "ghost",
                "first_unit": 0,
                "units": ["a"],
                "arm": [1],
                "probability": [0.5],
            },
        )
    with pytest.raises(errors.ScmleError):
        hook.assign(["a"])


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
    rng = numpy.random.default_rng(6)
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
    restored = budget.DecisionBudget.from_dict(row["body"]["budget"])
    assert restored == hook.budget


def broken_champion(tmp_path):
    """A registry whose new champion is clearly worse than the previous."""
    good = bundle_of("logistic_regression", test_lifecycle_extras.traffic())
    broken = bundle_of(
        "logistic_regression", test_lifecycle_extras.traffic(seed=0, flip=True)
    )
    store = registry.ModelRegistryStore(tmp_path / "registry")
    manager = lifecycle.LifecycleManager(store, "m", str(tmp_path / "work"))
    for index, model in enumerate((good, broken), start=1):
        record = store.register(
            "m", model.save(tmp_path / f"export{index}"), trusted=True
        )
        store.promote(
            "m",
            record.version,
            approval=approval_for(record),
            signing_key=APPROVAL_KEY,
        )
    return store, manager


def audit_rows(store, action):
    lines = (store.root / registry.AUDIT_FILE).read_text().splitlines()
    found = [json.loads(line) for line in lines]
    return [r for r in found if r["action"] == action]


def test_binding_a_hook_makes_the_bootstrap_rules_descriptive(tmp_path):
    ignored, manager = broken_champion(tmp_path)
    assert manager.rollout_decides is False
    rollout_hook.RolloutHook(manager, make_plan(), ledger_for())
    assert manager.rollout_decides is True


def test_the_bootstrap_rollback_check_never_moves_the_alias_once_a_hook_is_bound(
    tmp_path,
):
    store, manager = broken_champion(tmp_path)
    ledger = ledger_for()
    hook = rollout_hook.RolloutHook(manager, make_plan(), ledger)

    outcome = hook.check_rollback(
        test_lifecycle_extras.traffic(1200, seed=20),
        "y",
        approval=approval_for(store.resolve("m", 1)),
        signing_key=APPROVAL_KEY,
    )

    assert outcome.rolled_back is False and outcome.restored_version is None
    assert "descriptive" in outcome.reason
    assert outcome.comparison is not None and outcome.comparison.interval[1] < 0
    assert store.resolve("m").version == 2
    checked = audit_rows(store, "lifecycle.rollback_check")[-1]
    assert checked["descriptive"] is True and checked["rolled_back"] is False
    assert checked["rollback_required"] is True
    assert not audit_rows(store, "lifecycle.rollback_pending_approval")
    diagnostic = hook.log.rows[-1]
    assert diagnostic["kind"] == "bootstrap_diagnostic"
    assert diagnostic["body"]["stage"] == "rollback"
    assert diagnostic["body"]["descriptive"] is True
    assert diagnostic["body"]["decides"] is False
    assert hook.log.verify()


def test_the_hook_is_the_only_decider_of_promotion_and_rollback(tmp_path):
    env = hook_for(tmp_path, n_max=5000, tolerance=0.01)
    store, manager, hook = env.store, env.manager, env.hook
    challenger = env.challenger
    hook.start("exp-1")
    arm, primary, guards = traffic(40000, 5, lift=0.05)
    hook.observe(arm, primary, guards)
    hook.promote(approval_for(challenger), signing_key=APPROVAL_KEY)
    fresh = test_lifecycle_extras.traffic(1200, seed=20)

    bootstrap = manager.check_rollback(
        fresh,
        "y",
        approval=approval_for(store.resolve("m", 1)),
        signing_key=APPROVAL_KEY,
    )
    assert bootstrap.rolled_back is False
    assert store.resolve("m").version == challenger.version
    with pytest.raises(errors.ScmleError, match="rollback"):
        hook.roll_back(approval_for(store.resolve("m", 1)), APPROVAL_KEY)

    rng = numpy.random.default_rng(6)
    watch_arm = (rng.random(20000) < 0.5).astype(float)
    outcome = (rng.random(20000) < 0.30 - 0.15 * watch_arm).astype(float)
    assert hook.watch(watch_arm, outcome) == rollout.ROLLBACK
    assert store.resolve("m").version == challenger.version
    hook.roll_back(
        approval_for(store.resolve("m", 1)), signing_key=APPROVAL_KEY
    )
    assert store.resolve("m").version == 1
