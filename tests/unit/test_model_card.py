"""Tests for generated model-card evidence."""

import json

from scmle.governance import model_card


def test_model_card_requires_review_and_preserves_evaluation_context():
    card = model_card.Card(
        model="logistic_regression",
        version="run-1",
        fingerprint="sha256:model-manifest",
        task="binary_classification",
        metric="roc_auc",
        dataset="ds_1",
        snapshot="snap_1",
        use="Inference for the declared binary classification task.",
        excluded=("Not a causal claim.",),
        limitations=("Human review is required.",),
        features=("amount",),
        evaluation={"test_metrics": {"roc_auc": 0.91}},
        resources={"model_mb": 1.2},
        artifacts=("evaluation.json",),
        support="tabular_beta_library_inference",
    )

    record = card.to_dict()

    assert record["review"] == "machine_generated_unreviewed"
    assert record["evaluation"]["test_metrics"]["roc_auc"] == 0.91
    assert json.loads(json.dumps(record)) == record


def build(evaluation):
    import types

    manifest = types.SimpleNamespace(identifier="ds_1", snapshot="snap_1")
    plan = types.SimpleNamespace(features=("amount",), text=(), extras=())
    profile = types.SimpleNamespace(plan=plan)
    bundle = types.SimpleNamespace(
        info=types.SimpleNamespace(model="m", version="v1", metric="roc_auc"),
        schema=types.SimpleNamespace(task="binary_classification"),
    )
    return model_card.Card.assemble(
        manifest, profile, bundle, evaluation, "sha256:x"
    )


def test_a_missing_release_evaluation_is_stated_not_read_as_passed():
    # ERR-23: a missing evaluation used to read as "not blocked".
    unknown = build({"test_metrics": {}})
    assert any("status is unknown" in item for item in unknown.limitations)
    blocked = build({"release_evaluation": {"blocked": True}})
    assert any("is blocked" in item for item in blocked.limitations)
    passed = build({"release_evaluation": {"blocked": False}})
    assert not any(
        "unknown" in item or "is blocked" in item for item in passed.limitations
    )
