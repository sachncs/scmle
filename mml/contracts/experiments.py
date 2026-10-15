"""Plan and trial contracts."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Decision:
    """One auditable planner decision (included, rejected, or deferred technique)."""

    technique: str
    action: str  # include | reject | defer
    reason: str


@dataclass(frozen=True)
class ExperimentPlan:
    plan_id: str
    task: str
    metric: str
    validation_strategy: str
    candidate_models: tuple[str, ...]
    optimization_budget_seconds: int
    decisions: tuple[Decision, ...] = ()
    warnings: tuple[str, ...] = ()

    def summary(self) -> str:
        lines = [
            f"Plan {self.plan_id}: {self.task}, metric={self.metric}, "
            f"validation={self.validation_strategy}, budget={self.optimization_budget_seconds}s",
            "Candidates: " + ", ".join(self.candidate_models),
        ]
        for d in self.decisions:
            lines.append(f"  [{d.action}] {d.technique}: {d.reason}")
        for w in self.warnings:
            lines.append(f"  WARNING: {w}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "plan_id": self.plan_id,
            "task": self.task,
            "metric": self.metric,
            "validation_strategy": self.validation_strategy,
            "candidate_models": list(self.candidate_models),
            "optimization_budget_seconds": self.optimization_budget_seconds,
            "decisions": [vars(d) for d in self.decisions],
            "warnings": list(self.warnings),
        }


@dataclass
class TrialResult:
    trial_id: str
    model_name: str
    status: str  # ok | failed | skipped
    metric: str = ""
    score_mean: float = float("nan")  # natural units
    score_std: float = float("nan")
    fold_scores: list[float] = field(default_factory=list)
    fit_seconds: float = 0.0
    error: str = ""

    def to_dict(self) -> dict:
        return dict(vars(self))
