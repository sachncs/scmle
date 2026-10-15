"""Candidate-plan generation through composable planning rules.

The planner decides which experiments are worth trying and records why; the
evaluation layer decides whether they help. Nothing here asserts that a
technique will improve performance. Each `PlanningRule` is an independent,
polymorphic policy applied in order to the candidate list.
"""

import abc
import dataclasses
import hashlib

from mml.contracts import data as data_contracts
from mml.contracts import errors
from mml.contracts import experiments
from mml.contracts import models
from mml.contracts import tasks
from mml.data import splitting
from mml.data import strategies
from mml.evaluation import metrics
from mml.models import registry as registry_module

FAST_BUDGET_SECONDS = 300
FAST_EXCEPTION = "hist_gradient_boosting"

Action = experiments.DecisionAction


@dataclasses.dataclass(frozen=True, slots=True)
class PlanningContext:
    """Facts available to planning rules.

    Attributes:
        spec: Project specification.
        profile: Dataset profile.
        task: Resolved task type.
        approved_models: Models a human approved, or None for no restriction.
    """

    spec: tasks.ProjectSpec
    profile: data_contracts.DatasetProfile
    task: tasks.TaskType
    approved_models: tuple[str, ...] | None = None


class PlanningRule(abc.ABC):
    """A policy that filters candidates and records its decisions."""

    @abc.abstractmethod
    def apply(
        self,
        context: PlanningContext,
        candidates: list[models.ModelAdapter],
        decisions: list[experiments.Decision],
    ) -> list[models.ModelAdapter]:
        """Applies the rule.

        Args:
            context: Planning facts.
            candidates: Candidates surviving earlier rules.
            decisions: Decision log to append to.

        Returns:
            The candidates that remain.

        Raises:
            ConfigError: If the rule's input is inconsistent.
        """


class FastBudgetRule(PlanningRule):
    """Prefers fast models when the time budget is small."""

    def apply(
        self,
        context: PlanningContext,
        candidates: list[models.ModelAdapter],
        decisions: list[experiments.Decision],
    ) -> list[models.ModelAdapter]:
        """Drops expensive models under a tight budget."""
        if context.spec.resources.time_budget_seconds >= FAST_BUDGET_SECONDS:
            return candidates
        kept = []
        for adapter in candidates:
            expensive = adapter.capabilities.cost >= 3
            if expensive and adapter.name != FAST_EXCEPTION:
                decisions.append(
                    experiments.Decision(
                        adapter.name,
                        Action.REJECT,
                        f"time budget < {FAST_BUDGET_SECONDS}s: "
                        "fast models only",
                    )
                )
            else:
                kept.append(adapter)
        return kept


class ApprovedModelsRule(PlanningRule):
    """Restricts candidates to a human-approved subset."""

    def apply(
        self,
        context: PlanningContext,
        candidates: list[models.ModelAdapter],
        decisions: list[experiments.Decision],
    ) -> list[models.ModelAdapter]:
        """Keeps only approved models; rejects unknown approvals."""
        approved = context.approved_models
        if approved is None:
            return candidates
        unknown = set(approved) - {a.name for a in candidates}
        if unknown:
            raise errors.ConfigError(
                f"approved models not compatible with this plan: "
                f"{sorted(unknown)}"
            )
        for adapter in candidates:
            if adapter.name not in approved:
                decisions.append(
                    experiments.Decision(
                        adapter.name, Action.REJECT, "not in approved plan"
                    )
                )
        return [a for a in candidates if a.name in approved]


class StrategyRule(PlanningRule):
    """Records which data strategies will be evaluated and which cannot."""

    def apply(
        self,
        context: PlanningContext,
        candidates: list[models.ModelAdapter],
        decisions: list[experiments.Decision],
    ) -> list[models.ModelAdapter]:
        """Documents strategy applicability; leaves candidates unchanged."""
        spec = context.spec.strategies
        for name in spec.names:
            strategy = strategies.get_strategy(name)
            reasons = {
                a.name: strategy.applicability(context.profile, context.task, a)
                for a in candidates
            }
            applicable = sorted(n for n, r in reasons.items() if r is None)
            if applicable and spec.top_models > 0:
                decisions.append(
                    experiments.Decision(
                        name,
                        Action.INCLUDE,
                        f"{strategy.description}; evaluated on the top "
                        f"{spec.top_models} screened models where applicable "
                        f"(applies to: {', '.join(applicable)})",
                    )
                )
            else:
                why = (
                    "strategy search disabled (strategies.top_models=0)"
                    if applicable
                    else next(
                        (r for r in reasons.values() if r),
                        "no candidate models",
                    )
                )
                decisions.append(
                    experiments.Decision(
                        name, Action.REJECT, f"not applicable: {why}"
                    )
                )
        return candidates


class TuningRule(PlanningRule):
    """Records whether hyperparameter search will run after screening."""

    def apply(
        self,
        context: PlanningContext,
        candidates: list[models.ModelAdapter],
        decisions: list[experiments.Decision],
    ) -> list[models.ModelAdapter]:
        """Documents the search decision; leaves candidates unchanged."""
        search = context.spec.search
        tunable = [a.name for a in candidates if a.search_space is not None]
        if search.top_models == 0 or not tunable:
            decisions.append(
                experiments.Decision(
                    "hyperparameter_tuning",
                    Action.DEFER,
                    "disabled (search.top_models=0) or no tunable candidate",
                )
            )
        else:
            decisions.append(
                experiments.Decision(
                    "hyperparameter_tuning",
                    Action.INCLUDE,
                    f"{search.optimizer} search on the top "
                    f"{search.top_models} of {len(tunable)} tunable models, "
                    f"{search.configs_per_model} configs each, "
                    f"{search.prune_fraction:.0%} pruned after the first rung",
                )
            )
        return candidates


def default_rules() -> tuple[PlanningRule, ...]:
    """Returns the standard rule chain."""
    return (
        FastBudgetRule(),
        ApprovedModelsRule(),
        StrategyRule(),
        TuningRule(),
    )


class Planner:
    """Builds an `ExperimentPlan` from a profile."""

    def __init__(
        self,
        registry: registry_module.ModelRegistry,
        catalog: metrics.MetricCatalog = metrics.CATALOG,
        rules: tuple[PlanningRule, ...] | None = None,
    ) -> None:
        """Initializes the planner.

        Args:
            registry: Source of candidate adapters.
            catalog: Metric catalog used to resolve the primary metric.
            rules: Rule chain; defaults to `default_rules()`.
        """
        self.registry = registry
        self.catalog = catalog
        self.rules = rules if rules is not None else default_rules()

    def plan(self, context: PlanningContext) -> experiments.ExperimentPlan:
        """Creates the plan.

        Args:
            context: Planning facts.

        Returns:
            A deterministic, auditable plan.

        Raises:
            ConfigError: If no candidate model remains.
        """
        spec, profile, task = context.spec, context.profile, context.task
        metric = self.catalog.resolve(spec.metric, task)
        compatible, rejected = self.registry.candidates(
            task, profile.manifest.row_count
        )
        decisions = [
            experiments.Decision(name, Action.REJECT, reason)
            for name, reason in sorted(rejected.items())
        ]
        candidates = compatible
        for rule in self.rules:
            candidates = rule.apply(context, candidates, decisions)
        if not candidates:
            raise errors.ConfigError("planner produced no candidate models")
        decisions += [
            experiments.Decision(
                a.name,
                Action.INCLUDE,
                f"{a.family} candidate (cost {a.capabilities.cost})",
            )
            for a in candidates
        ]
        names = tuple(a.name for a in candidates)
        identity = f"{spec.config_hash()}|{profile.manifest.data_hash}|{names}"
        plan_id = hashlib.sha256(identity.encode()).hexdigest()[:10]
        return experiments.ExperimentPlan(
            plan_id=f"plan_{plan_id}",
            task=task.value,
            metric=metric.name,
            validation_strategy=splitting.resolve_strategy_name(spec, task),
            candidate_models=names,
            optimization_budget_seconds=spec.resources.time_budget_seconds,
            decisions=tuple(decisions),
            warnings=profile.risk_findings,
        )
