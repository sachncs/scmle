"""Machine-generated model cards tied to run and bundle evidence."""

from collections.abc import Mapping
import dataclasses
import datetime
from typing import Any

import scmle.artifacts.bundle
from scmle.contracts import data


@dataclasses.dataclass(frozen=True, slots=True)
class ModelCard:
    """Reviewable context for one packaged model."""

    model_name: str
    model_version: str
    model_fingerprint: str
    task: str
    metric: str
    dataset_id: str
    snapshot_id: str
    intended_use: str
    out_of_scope: tuple[str, ...]
    limitations: tuple[str, ...]
    feature_names: tuple[str, ...]
    evaluation: Mapping[str, Any]
    resources: Mapping[str, Any]
    evidence_artifacts: tuple[str, ...]
    support_status: str
    review_status: str = "machine_generated_unreviewed"
    schema_version: int = 1
    generated_at: str = dataclasses.field(
        default_factory=lambda: datetime.datetime.now(datetime.UTC).isoformat()
    )

    def __post_init__(self) -> None:
        """Checks the fields required for a useful review record.

        Raises:
            ValueError: If a required text field is blank, the schema version is
                below 1, or the out-of-scope or limitations list is empty.
        """
        for name in (
            "model_name",
            "model_version",
            "model_fingerprint",
            "task",
            "metric",
            "dataset_id",
            "snapshot_id",
            "intended_use",
            "support_status",
            "review_status",
            "generated_at",
        ):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must not be empty")
        if self.schema_version < 1:
            raise ValueError("schema_version must be positive")
        if not self.out_of_scope:
            raise ValueError("out_of_scope must not be empty")
        if not self.limitations:
            raise ValueError("limitations must not be empty")

    @classmethod
    def from_run(
        cls,
        manifest: data.DatasetManifest,
        profile: data.DatasetProfile,
        bundle: scmle.artifacts.bundle.Bundle,
        evaluation: Mapping[str, Any],
        model_fingerprint: str,
    ) -> "ModelCard":
        """Builds a card from the exact data, model, and evaluation records.

        Args:
            manifest: The dataset manifest, supplying the dataset and snapshot
                identifiers.
            profile: The dataset profile, supplying the feature names.
            bundle: The packaged model bundle, supplying the model identity,
                task and metric.
            evaluation: The evaluation document; its cross-validation scores,
                test metrics, release evaluation and resources are copied in.
            model_fingerprint: Digest identifying the packaged model.

        Returns:
            A card with fixed intended-use, out-of-scope and evidence-file
            texts and the review status ``machine_generated_unreviewed``. Its
            feature names are the profile's features, text columns and extra
            inputs without repeats. Its limitations state that the card needs
            human review, and, when the release evaluation is blocked, that
            it is blocked.
        """
        release = evaluation.get("release_evaluation", {})
        limitations = [
            "This machine-generated card requires human review before release.",
            (
                "Evaluation results describe the declared snapshot and split; "
                "they do not establish production-population performance."
            ),
            "Associational explanations are not causal claims.",
        ]
        if release.get("blocked", False):
            limitations.append(
                "The release evaluation is blocked until explicit success "
                "criteria and reviewed evidence are present."
            )
        return cls(
            model_name=bundle.info.model_name,
            model_version=bundle.info.model_version,
            model_fingerprint=model_fingerprint,
            task=bundle.schema.task,
            metric=bundle.info.metric,
            dataset_id=manifest.dataset_id,
            snapshot_id=manifest.snapshot_id,
            intended_use=(
                f"Inference for the declared {bundle.schema.task} task using "
                "the packaged schema and preprocessing pipeline."
            ),
            out_of_scope=(
                (
                    "It is not evidence of causal effect or fairness outside "
                    "the declared evaluation slices."
                ),
                (
                    "It must not be used with inputs that fail the packaged "
                    "schema."
                ),
                "It is not a production approval without independent review.",
            ),
            limitations=tuple(limitations),
            feature_names=tuple(
                dict.fromkeys(
                    (
                        *profile.feature_plan.features,
                        *profile.feature_plan.text,
                        *profile.feature_plan.extra_inputs,
                    )
                )
            ),
            evaluation={
                "selected_cv_score": evaluation.get("selected_cv_score"),
                "selected_cv_std": evaluation.get("selected_cv_std"),
                "test_metrics": evaluation.get("test_metrics", {}),
                "cv_confidence": evaluation.get("cv_confidence"),
                "release_evaluation": release,
            },
            resources=dict(evaluation.get("resources", {})),
            evidence_artifacts=(
                "dataset_card.json",
                "dataset_manifest.json",
                "evaluation.json",
                "explanations/permutation_importance.json",
                "explanations/permutation_importance.html",
                "explanations/partial_dependence.json",
                "explanations/partial_dependence.html",
                "explanations/pairwise_interaction.json",
                "explanations/pairwise_interaction.html",
                "provenance.json",
                "sbom.json",
            ),
            support_status="tabular_beta_library_inference",
        )

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-compatible model card."""
        return {
            "schema_version": self.schema_version,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "model_fingerprint": self.model_fingerprint,
            "task": self.task,
            "metric": self.metric,
            "dataset_id": self.dataset_id,
            "snapshot_id": self.snapshot_id,
            "intended_use": self.intended_use,
            "out_of_scope": list(self.out_of_scope),
            "limitations": list(self.limitations),
            "feature_names": list(self.feature_names),
            "evaluation": dict(self.evaluation),
            "resources": dict(self.resources),
            "evidence_artifacts": list(self.evidence_artifacts),
            "support_status": self.support_status,
            "review_status": self.review_status,
            "generated_at": self.generated_at,
        }
