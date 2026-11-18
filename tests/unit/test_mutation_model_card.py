"""The model card's limitations, evaluation copy and record, word for word."""

from scmle.governance import model_card
from tests.unit.test_model_card import build


def test_the_fixed_limitations_are_stated_word_for_word() -> None:
    assert model_card.caveat({"blocked": False}) == (
        "This machine-generated card requires human review before release.",
        (
            "Evaluation results describe the declared snapshot and split; "
            "they do not establish production-population performance."
        ),
        "Associational explanations are not causal claims.",
    )


def test_the_evaluation_copies_the_cross_validation_summary() -> None:
    card = build(
        {
            "selected_cv_score": 0.8,
            "selected_cv_std": 0.02,
            "test_metrics": {"roc_auc": 0.9},
            "cv_confidence": [0.75, 0.85],
            "release_evaluation": {"blocked": False},
        }
    )
    assert dict(card.evaluation) == {
        "selected_cv_score": 0.8,
        "selected_cv_std": 0.02,
        "test_metrics": {"roc_auc": 0.9},
        "cv_confidence": [0.75, 0.85],
        "release_evaluation": {"blocked": False},
    }


def test_the_record_holds_every_field_under_its_name() -> None:
    card = build({"release_evaluation": {"blocked": False}})
    record = card.to_dict()
    assert record == {
        "schema_version": 1,
        "model_name": "m",
        "model_version": "v1",
        "model_fingerprint": "sha256:x",
        "task": "binary_classification",
        "metric": "roc_auc",
        "dataset_id": "ds_1",
        "snapshot_id": "snap_1",
        "intended_use": card.intended_use,
        "out_of_scope": list(model_card.EXCLUSIONS),
        "limitations": list(card.limitations),
        "feature_names": ["amount"],
        "evaluation": dict(card.evaluation),
        "resources": {},
        "evidence_artifacts": list(model_card.ARTIFACTS),
        "support_status": "tabular_beta_library_inference",
        "review_status": "machine_generated_unreviewed",
        "generated_at": card.generated_at,
    }
