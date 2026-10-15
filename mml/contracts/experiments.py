"""Experiment plan and trial result contracts."""

import dataclasses
import enum
from typing import Any

DEFAULT_STRATEGY = "default"


class TrialStatus(enum.StrEnum):
    """Terminal status of a trial."""

    OK = "ok"
    FAILED = "failed"
    SKIPPED = "skipped"
    PRUNED = "pruned"


class DecisionAction(enum.StrEnum):
    """What the planner did with a technique."""

    INCLUDE = "include"
    REJECT = "reject"
    DEFER = "defer"


@dataclasses.dataclass(frozen=True, slots=True)
class Decision:
    """One auditable planner decision.

    Attributes:
        technique: Model or technique the decision is about.
        action: Whether it was included, rejected or deferred.
        reason: Why.
    """

    technique: str
    action: DecisionAction
    reason: str

    def to_dict(self) -> dict[str, str]:
        """Returns a JSON-serializable mapping."""
        return {
            "technique": self.technique,
            "action": str(self.action),
            "reason": self.reason,
        }


@dataclasses.dataclass(frozen=True, slots=True)
class ExperimentPlan:
    """Declarative, auditable description of the proposed workflow.

    Attributes:
        plan_id: Deterministic identifier.
        task: Resolved task type value.
        metric: Primary metric name.
        validation_strategy: Resolved split strategy name.
        candidate_models: Models that will be trained, in execution order.
        optimization_budget_seconds: Wall-clock budget.
        decisions: Included / rejected / deferred techniques with reasons.
        warnings: Risks surfaced during planning.
    """

    plan_id: str
    task: str
    metric: str
    validation_strategy: str
    candidate_models: tuple[str, ...]
    optimization_budget_seconds: int
    decisions: tuple[Decision, ...] = ()
    warnings: tuple[str, ...] = ()

    def summary(self) -> str:
        """Returns a human-readable multi-line summary."""
        lines = [
            f"Plan {self.plan_id}: {self.task}, metric={self.metric}, "
            f"validation={self.validation_strategy}, "
            f"budget={self.optimization_budget_seconds}s",
            "Candidates: " + ", ".join(self.candidate_models),
        ]
        lines += [
            f"  [{d.action}] {d.technique}: {d.reason}" for d in self.decisions
        ]
        lines += [f"  WARNING: {warning}" for warning in self.warnings]
        return "\n".join(lines)

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> ExperimentPlan:
        """Rebuilds a plan from `to_dict` output."""
        return cls(
            plan_id=record["plan_id"],
            task=record["task"],
            metric=record["metric"],
            validation_strategy=record["validation_strategy"],
            candidate_models=tuple(record["candidate_models"]),
            optimization_budget_seconds=record["optimization_budget_seconds"],
            decisions=tuple(
                Decision(
                    d["technique"], DecisionAction(d["action"]), d["reason"]
                )
                for d in record["decisions"]
            ),
            warnings=tuple(record["warnings"]),
        )

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-serializable mapping."""
        return {
            "plan_id": self.plan_id,
            "task": self.task,
            "metric": self.metric,
            "validation_strategy": self.validation_strategy,
            "candidate_models": list(self.candidate_models),
            "optimization_budget_seconds": self.optimization_budget_seconds,
            "decisions": [d.to_dict() for d in self.decisions],
            "warnings": list(self.warnings),
        }


@dataclasses.dataclass(slots=True)
class TrialResult:
    """Outcome of evaluating one complete candidate pipeline.

    Attributes:
        trial_id: Identifier within the run.
        model_name: Candidate model name.
        status: Terminal status.
        metric: Metric name the scores refer to.
        score_mean: Mean cross-validation score (natural units).
        score_std: Standard deviation across folds.
        fold_scores: Per-fold scores.
        fit_seconds: Wall-clock cost.
        error: Failure or skip reason, empty on success.
        params: Hyperparameter overrides (empty for default configuration).
        strategy: Data strategy the candidate was built with.
    """

    trial_id: str
    model_name: str
    status: TrialStatus
    metric: str = ""
    score_mean: float = float("nan")
    score_std: float = float("nan")
    fold_scores: list[float] = dataclasses.field(default_factory=list)
    fit_seconds: float = 0.0
    error: str = ""
    params: dict[str, Any] = dataclasses.field(default_factory=dict)
    strategy: str = DEFAULT_STRATEGY

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-serializable mapping."""
        record = dataclasses.asdict(self)
        record["status"] = str(self.status)
        return record


@dataclasses.dataclass(frozen=True, slots=True)
class HistoricalRun:
    """What is remembered about a past run, for meta-learning.

    Attributes:
        run_id: Run identifier.
        dataset_id: Dataset identity (content derived).
        data_hash: Hash of the dataset contents.
        task: Task type value.
        metric: Optimized metric name.
        fingerprint: Numeric dataset fingerprint.
        selected_model: Winning model name.
        selected_strategy: Winning data strategy.
        cv_score: Cross-validation score of the winner.
    """

    run_id: str
    dataset_id: str
    data_hash: str
    task: str
    metric: str
    fingerprint: dict[str, float]
    selected_model: str
    selected_strategy: str
    cv_score: float
