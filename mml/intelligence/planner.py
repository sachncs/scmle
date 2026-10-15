"""Candidate-plan generation.

The planner decides which experiments are worth trying and records why; the evaluation
layer decides whether they help. Nothing here asserts a technique will improve performance.
"""

from __future__ import annotations

import hashlib

from mml.contracts.data import DatasetProfile
from mml.contracts.errors import ConfigError
from mml.contracts.experiments import Decision, ExperimentPlan
from mml.contracts.tasks import ProjectSpec, TaskType
from mml.data.splitting import resolve_strategy
from mml.evaluation.metrics import resolve_metric
from mml.models.registry import candidates

FAST_BUDGET_SECONDS = 300


def make_plan(
    spec: ProjectSpec,
    profile: DatasetProfile,
    task: TaskType,
    allowed_models: tuple[str, ...] | None = None,
) -> ExperimentPlan:
    metric = resolve_metric(spec.metric, task)
    strategy = resolve_strategy(spec, task)
    ok, rejected = candidates(task, profile.manifest.row_count)
    decisions: list[Decision] = []
    warnings: list[str] = list(profile.risk_findings)

    for name, why in sorted(rejected.items()):
        decisions.append(Decision(name, "reject", why))

    chosen = list(ok)
    if spec.resources.time_budget_seconds < FAST_BUDGET_SECONDS:
        slow = [
            m for m in chosen if m.capabilities.cost >= 3 and m.name != "hist_gradient_boosting"
        ]
        chosen = [m for m in chosen if m not in slow]
        for m in slow:
            decisions.append(
                Decision(
                    m.name, "reject", f"time budget < {FAST_BUDGET_SECONDS}s: fast models only"
                )
            )

    if allowed_models is not None:
        unknown = set(allowed_models) - {m.name for m in ok}
        if unknown:
            raise ConfigError(f"approved models not compatible with this plan: {sorted(unknown)}")
        for m in [m for m in chosen if m.name not in allowed_models]:
            decisions.append(Decision(m.name, "reject", "not in approved plan"))
        chosen = [m for m in chosen if m.name in allowed_models]

    for m in chosen:
        decisions.append(
            Decision(m.name, "include", f"{m.family} candidate (cost {m.capabilities.cost})")
        )

    if task.is_classification and profile.target_summary.get("imbalance_ratio", 1) >= 10:
        decisions.append(
            Decision(
                "class_weighting", "defer", "severe imbalance; scheduled for the M3 strategy search"
            )
        )
        decisions.append(
            Decision(
                "resampling_inside_cv",
                "defer",
                "severe imbalance; scheduled for the M3 strategy search",
            )
        )
    decisions.append(
        Decision("hyperparameter_tuning", "defer", "v0.1 compares default-configured pipelines")
    )

    names = tuple(m.name for m in chosen)
    if not names:
        raise ConfigError("planner produced no candidate models")
    pid = hashlib.sha256(
        f"{spec.config_hash()}|{profile.manifest.data_hash}|{names}".encode()
    ).hexdigest()[:10]
    return ExperimentPlan(
        plan_id=f"plan_{pid}",
        task=task.value,
        metric=metric.name,
        validation_strategy=strategy,
        candidate_models=names,
        optimization_budget_seconds=spec.resources.time_budget_seconds,
        decisions=tuple(decisions),
        warnings=tuple(warnings),
    )
