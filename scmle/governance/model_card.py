"""Machine-generated model cards tied to run and bundle evidence."""

from collections.abc import Mapping
import dataclasses
import datetime
from typing import Any
from typing import ClassVar

import scmle.artifacts.bundle
from scmle.contracts import data
from scmle.contracts import errors
from scmle.contracts import records

EXCLUSIONS = (
    (
        "It is not evidence of causal effect or fairness outside "
        "the declared evaluation slices."
    ),
    "It must not be used with inputs that fail the packaged schema.",
    "It is not a production approval without independent review.",
)
"""The out-of-scope uses every machine-generated card states."""
ARTIFACTS = (
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
)
"""The evidence files a card points a reviewer to."""


def caveat(release: object) -> tuple[str, ...]:
    """Returns the limitations a card states for a release evaluation.

    Args:
        release: The evaluation's `release_evaluation` entry, if any.

    Returns:
        The fixed limitations (human review, snapshot scope, associational
        explanations) and, when the release evaluation is missing or
        blocked, a statement that its status is unknown or that it is
        blocked.
    """
    limitations = [
        "This machine-generated card requires human review before release.",
        (
            "Evaluation results describe the declared snapshot and split; "
            "they do not establish production-population performance."
        ),
        "Associational explanations are not causal claims.",
    ]
    if not isinstance(release, dict) or "blocked" not in release:
        # ERR-23: a missing evaluation read as "not blocked".
        limitations.append(
            "No release evaluation was recorded, so the release status is "
            "unknown; treat the model as not approved for release."
        )
    elif release["blocked"]:
        limitations.append(
            "The release evaluation is blocked until explicit success "
            "criteria and reviewed evidence are present."
        )
    return tuple(limitations)


@dataclasses.dataclass(frozen=True, slots=True)
class Card:
    """Reviewable context for one packaged model.

    Attributes:
        migrations: The written layout's version steps, oldest first;
            version 2 writes `model`, `version`, `digests.model`, `dataset`,
            `snapshot`, `use`, `excluded`, `features`, `artifacts`,
            `support`, `review` and `generated`, the `Datasheet` spelling.
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = (
        records.nest(
            {
                "model_name": "model",
                "model_version": "version",
                "dataset_id": "dataset",
                "snapshot_id": "snapshot",
                "intended_use": "use",
                "out_of_scope": "excluded",
                "feature_names": "features",
                "evidence_artifacts": "artifacts",
                "support_status": "support",
                "review_status": "review",
                "generated_at": "generated",
            },
            groups={"digests": {"model_fingerprint": "model"}},
        ),
    )

    model: str
    version: str
    fingerprint: str
    task: str
    metric: str
    dataset: str
    snapshot: str
    use: str
    excluded: tuple[str, ...]
    limitations: tuple[str, ...]
    features: tuple[str, ...]
    evaluation: Mapping[str, Any]
    resources: Mapping[str, Any]
    artifacts: tuple[str, ...]
    support: str
    status: str = "machine_generated_unreviewed"
    schema_version: int = records.ORIGIN + 1
    generated: str = dataclasses.field(
        default_factory=lambda: datetime.datetime.now(datetime.UTC).isoformat()
    )

    def __post_init__(self) -> None:
        """Checks the fields required for a useful review record.

        Raises:
            ValueError: If a required text field is blank, the schema version is
                below 1, or the out-of-scope or limitations list is empty.
        """
        for name in (
            "model",
            "version",
            "fingerprint",
            "task",
            "metric",
            "dataset",
            "snapshot",
            "use",
            "support",
            "status",
            "generated",
        ):
            if not getattr(self, name).strip():
                raise errors.InputError(f"{name} must not be empty")
        if self.schema_version < 1:
            raise errors.InputError("schema_version must be positive")
        if not self.excluded:
            raise errors.InputError("out_of_scope must not be empty")
        if not self.limitations:
            raise errors.InputError("limitations must not be empty")

    @classmethod
    def assemble(
        cls,
        manifest: data.Manifest,
        profile: data.Portrait,
        bundle: scmle.artifacts.bundle.Bundle,
        evaluation: Mapping[str, Any],
        model_fingerprint: str,
    ) -> "Card":
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
            human review, and, when the release evaluation is blocked or
            missing, that it is blocked or that its status is unknown.
        """
        release = evaluation.get("release_evaluation")
        return cls(
            model=bundle.info.model,
            version=bundle.info.version,
            fingerprint=model_fingerprint,
            task=bundle.schema.task,
            metric=bundle.info.metric,
            dataset=manifest.identifier,
            snapshot=manifest.snapshot,
            use=(
                f"Inference for the declared {bundle.schema.task} task using "
                "the packaged schema and preprocessing pipeline."
            ),
            excluded=EXCLUSIONS,
            limitations=caveat(release),
            features=tuple(
                dict.fromkeys(
                    (
                        *profile.plan.features,
                        *profile.plan.text,
                        *profile.plan.extras,
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
            artifacts=ARTIFACTS,
            support="tabular_beta_library_inference",
        )

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-compatible model card.

        Returns:
            The card in the current written layout, with `schema_version`
            set to the current version.
        """
        layout = {
            "model_name": self.model,
            "model_version": self.version,
            "model_fingerprint": self.fingerprint,
            "task": self.task,
            "metric": self.metric,
            "dataset_id": self.dataset,
            "snapshot_id": self.snapshot,
            "intended_use": self.use,
            "out_of_scope": list(self.excluded),
            "limitations": list(self.limitations),
            "feature_names": list(self.features),
            "evaluation": dict(self.evaluation),
            "resources": dict(self.resources),
            "evidence_artifacts": list(self.artifacts),
            "support_status": self.support,
            "review_status": self.status,
            "generated_at": self.generated,
        }
        return {
            records.KEY: records.ORIGIN + len(self.migrations),
            **records.upgrade(self.migrations, layout, records.ORIGIN),
        }
