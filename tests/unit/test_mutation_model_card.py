"""The model card's limitations, evaluation copy and record, word for word."""

import dataclasses

import pytest

from scmle.contracts import errors
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
        "schema_version": 2,
        "model": "m",
        "version": "v1",
        "digests": {"model": "sha256:x"},
        "task": "binary_classification",
        "metric": "roc_auc",
        "dataset": "ds_1",
        "snapshot": "snap_1",
        "use": card.intended_use,
        "excluded": list(model_card.EXCLUSIONS),
        "limitations": list(card.limitations),
        "features": ["amount"],
        "evaluation": dict(card.evaluation),
        "resources": {},
        "artifacts": list(model_card.ARTIFACTS),
        "support": "tabular_beta_library_inference",
        "review": "machine_generated_unreviewed",
        "generated": card.generated_at,
    }


def test_a_blank_field_and_no_limitations_are_refused_word_for_word() -> None:
    card = build({"test_metrics": {}})
    with pytest.raises(
        errors.InputError, match=r"^model_name must not be empty$"
    ):
        dataclasses.replace(card, model_name=" ")
    with pytest.raises(
        errors.InputError, match=r"^limitations must not be empty$"
    ):
        dataclasses.replace(card, limitations=())
