"""User-facing recommendations derived from the dataset profile.

The planner decides which experiments to run; recommendations tell the
person what they might change about the data or the project specification.
They are advice, never automatic actions.
"""

import abc
import dataclasses
from typing import Any

from scmle.contracts import data as data_contracts
from scmle.contracts import tasks

LEAKAGE_MARKER = "leakage"
SMALL_DATA_ROWS = 500
MISSING_SHARE = 0.30
DUPLICATE_SHARE = 0.10
SEVERE_IMBALANCE = 10.0


@dataclasses.dataclass(frozen=True)
class Recommendation:
    """One piece of advice.

    Attributes:
        topic: Short area name (for example `imbalance`).
        message: What to consider and why.
        setting: The specification field that would change, if any.
    """

    topic: str
    message: str
    setting: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-serializable mapping."""
        return dataclasses.asdict(self)


class RecommendationRule(abc.ABC):
    """Produces recommendations from one aspect of the profile."""

    @abc.abstractmethod
    def advise(
        self, profile: data_contracts.DatasetProfile, spec: tasks.ProjectSpec
    ) -> list[Recommendation]:
        """Returns the recommendations that apply, possibly none.

        Args:
            profile: Profile of the development partition.
            spec: The project specification in force.
        """


class ImbalanceAdvice(RecommendationRule):
    """Suggests decision and metric choices for rare positive classes."""

    def advise(
        self, profile: data_contracts.DatasetProfile, spec: tasks.ProjectSpec
    ) -> list[Recommendation]:
        """Advises on a severely imbalanced binary target."""
        ratio = float(profile.target_summary.get("imbalance_ratio", 0.0))
        binary = profile.target_summary.get("task") == str(
            tasks.TaskType.BINARY
        )
        if not binary or ratio < SEVERE_IMBALANCE:
            return []
        advice = [
            Recommendation(
                "imbalance",
                f"the target is imbalanced ({ratio:.1f}:1); accuracy hides "
                "failures on the rare class, so rank models by pr_auc or "
                "log_loss rather than accuracy",
                "metric",
            )
        ]
        if spec.decision.threshold_objective == "none":
            advice.append(
                Recommendation(
                    "imbalance",
                    "choose a decision threshold from your real costs "
                    "instead of the default 0.5",
                    "decision.threshold_objective",
                )
            )
        return advice


class MissingnessAdvice(RecommendationRule):
    """Flags columns that are mostly missing."""

    def advise(
        self, profile: data_contracts.DatasetProfile, spec: tasks.ProjectSpec
    ) -> list[Recommendation]:
        """Advises on heavy missingness."""
        overall = float(
            profile.dataset_statistics.get("missingness", {}).get(
                "overall_rate", 0.0
            )
        )
        heavy = [f for f in profile.quality_findings if "% missing" in f]
        if overall < MISSING_SHARE and not heavy:
            return []
        return [
            Recommendation(
                "missing values",
                "some data is missing at a high rate; check whether it is "
                "missing for a reason that will also hold at prediction "
                "time, and whether the source can be improved",
                "data.excluded_columns",
            )
        ]


class LeakageAdvice(RecommendationRule):
    """Points at leakage hypotheses raised by the profile."""

    def advise(
        self, profile: data_contracts.DatasetProfile, spec: tasks.ProjectSpec
    ) -> list[Recommendation]:
        """Advises when leakage hypotheses exist and are not blocking."""
        leaks = [r for r in profile.risk_findings if LEAKAGE_MARKER in r]
        if not leaks:
            return []
        advice = [
            Recommendation(
                "leakage",
                "possible leakage: "
                + "; ".join(leaks)
                + ". List columns that are only known after the outcome in "
                "data.unavailable_at_prediction",
                "data.unavailable_at_prediction",
            )
        ]
        if spec.leakage_policy != "block":
            advice.append(
                Recommendation(
                    "leakage",
                    "set leakage_policy='block' to refuse to train until "
                    "the hypotheses are resolved",
                    "leakage_policy",
                )
            )
        return advice


class DuplicateAdvice(RecommendationRule):
    """Flags heavily duplicated rows."""

    def advise(
        self, profile: data_contracts.DatasetProfile, spec: tasks.ProjectSpec
    ) -> list[Recommendation]:
        """Advises when many rows are exact duplicates."""
        share = float(
            profile.dataset_statistics.get("quality", {}).get(
                "duplicate_share", 0.0
            )
        )
        if share < DUPLICATE_SHARE or spec.data.group_column:
            return []
        return [
            Recommendation(
                "duplicates",
                f"{share:.0%} of rows are exact duplicates; duplicates can "
                "appear in both training and validation folds and inflate "
                "scores. Deduplicate, or set data.group_column if rows "
                "belong to the same entity",
                "data.group_column",
            )
        ]


class SmallDataAdvice(RecommendationRule):
    """Warns that estimates from few rows are noisy."""

    def advise(
        self, profile: data_contracts.DatasetProfile, spec: tasks.ProjectSpec
    ) -> list[Recommendation]:
        """Advises on small development sets."""
        rows = int(profile.dataset_statistics.get("shape", {}).get("rows", 0))
        if rows == 0 or rows >= SMALL_DATA_ROWS:
            return []
        return [
            Recommendation(
                "small data",
                f"only {rows} development rows: scores are noisy, so treat "
                "small differences between models as ties and prefer the "
                "'one_standard_error' selection rule",
                "selection.uncertainty",
            )
        ]


def default_rules() -> tuple[RecommendationRule, ...]:
    """Returns the standard recommendation rules."""
    return (
        ImbalanceAdvice(),
        MissingnessAdvice(),
        LeakageAdvice(),
        DuplicateAdvice(),
        SmallDataAdvice(),
    )


def recommend(
    profile: data_contracts.DatasetProfile,
    spec: tasks.ProjectSpec,
    rules: tuple[RecommendationRule, ...] | None = None,
) -> tuple[Recommendation, ...]:
    """Collects the recommendations of every rule.

    Args:
        profile: Profile of the development partition.
        spec: The project specification in force.
        rules: Rules to apply; the standard ones by default.

    Returns:
        All applicable recommendations, in rule order.
    """
    found: list[Recommendation] = []
    for rule in rules if rules is not None else default_rules():
        found.extend(rule.advise(profile, spec))
    return tuple(found)
