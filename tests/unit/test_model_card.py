"""Tests for generated model-card evidence."""

import json

from scmle.governance import model_card


def test_model_card_requires_review_and_preserves_evaluation_context():
    card = model_card.ModelCard(
        model_name="logistic_regression",
        model_version="run-1",
        model_fingerprint="sha256:model-manifest",
        task="binary_classification",
        metric="roc_auc",
        dataset_id="ds_1",
        snapshot_id="snap_1",
        intended_use="Inference for the declared binary classification task.",
        out_of_scope=("Not a causal claim.",),
        limitations=("Human review is required.",),
        feature_names=("amount",),
        evaluation={"test_metrics": {"roc_auc": 0.91}},
        resources={"model_mb": 1.2},
        evidence_artifacts=("evaluation.json",),
        support_status="tabular_beta_library_inference",
    )

    record = card.to_dict()

    assert record["review_status"] == "machine_generated_unreviewed"
    assert record["evaluation"]["test_metrics"]["roc_auc"] == 0.91
    assert json.loads(json.dumps(record)) == record
