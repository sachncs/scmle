"""Shadow evaluation, A/B analysis and rollback with small real bundles."""

import numpy as np
import pandas as pd
import pytest

from scml.artifacts import bundle as bundle_module
from scml.contracts import data as data_contracts
from scml.contracts import errors
from scml.contracts import tasks
from scml.data.curation import preprocessing
from scml.governance import lifecycle
from scml.governance import registry as registry_module
from scml.models import registry as models_module
from scml.monitoring import abtest
from scml.monitoring import shadow

PLAN = data_contracts.FeaturePlan(("a", "b"), ())
MODELS = models_module.ModelRegistry.with_builtins()


def traffic(rows=600, seed=0, flip=False):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {"a": rng.normal(size=rows), "b": rng.normal(size=rows)}
    )
    signal = frame.a + 0.3 * frame.b + rng.normal(0, 0.5, rows)
    frame["y"] = ((-signal if flip else signal) > 0).astype(int)
    return frame


def bundle_of(name, frame):
    features = frame.drop(columns="y")
    fitted = preprocessing.build_pipeline(
        PLAN, MODELS.get(name), tasks.TaskType.BINARY, 0
    ).fit(features, frame.y)
    return bundle_module.ModelBundle.create(
        fitted, PLAN, features, "y", "binary_classification", "roc_auc", name
    )


def approval_for(record):
    return registry_module.PromotionApproval(
        model=record.model,
        version=record.version,
        content_id=record.content_id,
        reviewer="unit-reviewer",
        decision=registry_module.PromotionDecision.APPROVED,
        reason="verified unit fixture",
        evidence=("unit-evidence.json",),
        approved_at="2026-10-20T00:00:00+00:00",
    )


@pytest.fixture(scope="module")
def good():
    return bundle_of("logistic_regression", traffic())


@pytest.fixture(scope="module")
def weak():
    return bundle_of("decision_tree", traffic(120, seed=5))


def test_shadow_serves_the_champion_and_measures_disagreement(good, weak):
    evaluator = shadow.ShadowEvaluator(good, weak)
    rows = traffic(400, seed=2).drop(columns="y")

    served = evaluator.observe(rows)

    assert (served == good.predict(rows)).all()
    agreement = evaluator.agreement()
    assert agreement.rows == 400 and 0.0 < agreement.disagreement_rate < 1.0
    assert agreement.mean_probability_gap > 0


def test_shadow_settles_with_the_known_outcomes(good, weak):
    evaluator = shadow.ShadowEvaluator(good, weak)
    fresh = traffic(500, seed=3)
    evaluator.observe(fresh.drop(columns="y"))

    comparison = evaluator.settle(fresh.y)

    assert comparison.decision == "keep"  # The weak challenger is not better.
    assert comparison.champion_score > comparison.challenger_score


def test_an_empty_shadow_has_nothing_to_report(good, weak):
    with pytest.raises(errors.ConfigError, match="nothing"):
        shadow.ShadowEvaluator(good, weak).agreement()


def arm_of(model, frame):
    features = frame.drop(columns="y")
    return abtest.Arm(
        np.asarray(model.predict(features)),
        frame.y.to_numpy(),
        np.asarray(model.predict_proba(features)),
        np.asarray(model.classes),
    )


def test_a_clearly_better_treatment_is_adopted(good, weak):
    control = arm_of(weak, traffic(1500, seed=10))
    treatment = arm_of(good, traffic(1500, seed=11))

    result = abtest.compare_arms(
        tasks.TaskType.BINARY, "roc_auc", control, treatment
    )

    assert result.decision == "adopt" and result.interval[0] > 0


def test_a_clearly_worse_treatment_is_kept_out(good, weak):
    control = arm_of(good, traffic(1500, seed=12))
    treatment = arm_of(weak, traffic(1500, seed=13))

    result = abtest.compare_arms(
        tasks.TaskType.BINARY, "roc_auc", control, treatment
    )

    assert result.decision == "keep" and result.interval[1] < 0


def test_identical_models_are_inconclusive(good):
    control = arm_of(good, traffic(1200, seed=14))
    treatment = arm_of(good, traffic(1200, seed=15))

    result = abtest.compare_arms(
        tasks.TaskType.BINARY, "roc_auc", control, treatment
    )

    assert result.decision == "inconclusive"


def test_too_few_requests_never_decide(good, weak):
    control = arm_of(weak, traffic(60, seed=16))
    treatment = arm_of(good, traffic(60, seed=17))

    result = abtest.compare_arms(
        tasks.TaskType.BINARY, "roc_auc", control, treatment
    )

    assert result.decision == "inconclusive"


def registry_with(tmp_path, first, second):
    store = registry_module.ModelRegistryStore(tmp_path / "registry")
    manager = lifecycle.LifecycleManager(store, "m", str(tmp_path / "work"))
    for index, model in enumerate((first, second), start=1):
        directory = model.save(tmp_path / f"export{index}")
        record = store.register("m", directory, trusted=True)
        store.promote("m", record.version, approval=approval_for(record))
    return store, manager


def test_a_clearly_worse_new_champion_is_rolled_back(good, tmp_path):
    broken = bundle_of("logistic_regression", traffic(seed=0, flip=True))
    store, manager = registry_with(tmp_path, good, broken)
    fresh = traffic(1200, seed=20)

    outcome = manager.check_rollback(
        fresh, "y", approval=approval_for(store.resolve("m", 1))
    )

    assert outcome.rolled_back and outcome.restored_version == 1
    assert store.resolve("m").version == 1
    actions = [
        line
        for line in (store.root / registry_module.AUDIT_FILE)
        .read_text()
        .splitlines()
        if "rollback_check" in line
    ]
    assert actions and '"rolled_back": true' in actions[0]


def test_a_clearly_worse_new_champion_waits_for_approval(good, tmp_path):
    broken = bundle_of("logistic_regression", traffic(seed=0, flip=True))
    store, manager = registry_with(tmp_path, good, broken)

    outcome = manager.check_rollback(traffic(1200, seed=20), "y")

    assert not outcome.rolled_back
    assert outcome.restored_version is None
    assert "requires independent approval" in outcome.reason
    assert store.resolve("m").version == 2


def test_a_new_champion_that_holds_up_stays(good, tmp_path):
    same = bundle_of("logistic_regression", traffic(seed=1))
    store, manager = registry_with(tmp_path, good, same)

    outcome = manager.check_rollback(
        traffic(1200, seed=21),
        "y",
        lifecycle.RollbackPolicy(tolerance=0.02),
    )

    assert not outcome.rolled_back and store.resolve("m").version == 2


def test_nothing_to_roll_back_to_or_too_little_evidence(good, tmp_path):
    store = registry_module.ModelRegistryStore(tmp_path / "registry")
    manager = lifecycle.LifecycleManager(store, "m", str(tmp_path / "work"))
    record = store.register("m", good.save(tmp_path / "e"), trusted=True)
    store.promote("m", record.version, approval=approval_for(record))

    assert "no previous" in manager.check_rollback(traffic(300), "y").reason
    second = store.register(
        "m",
        bundle_of("decision_tree", traffic(100, seed=9)).save(tmp_path / "f"),
        trusted=True,
    )
    store.promote("m", second.version, approval=approval_for(second))
    assert "fewer than" in manager.check_rollback(traffic(20), "y").reason
