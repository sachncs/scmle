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
from mml.data.splitting import splitters
from mml.evaluation import metrics
from mml.intelligence import knowledge
from mml.models import registry as registry_module
from mml.sampling import strategies

FAST_BUDGET_SECONDS = tasks.FAST_BUDGET_SECONDS
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
        history: Past runs available for meta-learning.
    """

    spec: tasks.ProjectSpec
    profile: data_contracts.DatasetProfile
    task: tasks.TaskType
    approved_models: tuple[str, ...] | None = None
    history: tuple[experiments.HistoricalRun, ...] = ()


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


class MemoryRule(PlanningRule):
    """Rejects models whose estimated training memory exceeds the limit.

    The estimate is `rows x features x 8 bytes x memory_factor`, where the
    factor is each adapter's declared memory characteristic.
    """

    def apply(
        self,
        context: PlanningContext,
        candidates: list[models.ModelAdapter],
        decisions: list[experiments.Decision],
    ) -> list[models.ModelAdapter]:
        """Drops candidates that would not fit in `memory_limit_gb`."""
        rows = context.profile.manifest.row_count
        width = max(len(context.profile.feature_plan.features), 1)
        limit = context.spec.resources.memory_limit_gb
        base_gb = rows * width * 8 / 1024**3
        kept = []
        for adapter in candidates:
            estimate = base_gb * adapter.capabilities.memory_factor
            if estimate > limit:
                decisions.append(
                    experiments.Decision(
                        adapter.name,
                        Action.REJECT,
                        f"estimated training memory {estimate:.2f} GB exceeds "
                        f"the {limit:g} GB limit",
                    )
                )
            else:
                kept.append(adapter)
        return kept


class GpuRule(PlanningRule):
    """Honours the GPU policy: `required` keeps only GPU-capable models."""

    def apply(
        self,
        context: PlanningContext,
        candidates: list[models.ModelAdapter],
        decisions: list[experiments.Decision],
    ) -> list[models.ModelAdapter]:
        """Filters for GPU capability when a GPU is required.

        Raises:
            BlockedError: If a GPU is required but no candidate can use one.
        """
        if context.spec.resources.gpu != "required":
            return candidates
        capable = [a for a in candidates if a.capabilities.gpu]
        if not capable:
            raise errors.BlockedError(
                "resources.gpu='required' but no compatible model can use a "
                "GPU; set gpu='auto' or install a GPU-capable adapter"
            )
        for adapter in candidates:
            if adapter not in capable:
                decisions.append(
                    experiments.Decision(
                        adapter.name, Action.REJECT, "gpu required"
                    )
                )
        return capable


class SharedProcessRule(PlanningRule):
    """Rejects models needing their own process when trials share one."""

    def apply(
        self,
        context: PlanningContext,
        candidates: list[models.ModelAdapter],
        decisions: list[experiments.Decision],
    ) -> list[models.ModelAdapter]:
        """Drops isolated-process models under cooperative enforcement."""
        if context.spec.resources.enforcement == "hard":
            return candidates
        kept = []
        for adapter in candidates:
            if adapter.capabilities.isolated_process:
                decisions.append(
                    experiments.Decision(
                        adapter.name,
                        Action.REJECT,
                        "its native runtime cannot share a process with "
                        "other models; use resources.enforcement='hard'",
                    )
                )
            else:
                kept.append(adapter)
        return kept


class FastBudgetRule(PlanningRule):
    """Prefers fast models when the time budget is small."""

    def apply(
        self,
        context: PlanningContext,
        candidates: list[models.ModelAdapter],
        decisions: list[experiments.Decision],
    ) -> list[models.ModelAdapter]:
        """Drops expensive models and shrinks the search on a tight budget."""
        if not context.spec.fast_budget:
            return candidates
        search = context.spec.effective_search()
        decisions.append(
            experiments.Decision(
                "reduce_search_complexity",
                Action.INCLUDE,
                f"time budget < {FAST_BUDGET_SECONDS}s: tune at most "
                f"{search.top_models} model(s) with "
                f"{search.configs_per_model} config(s) each and try strategies "
                f"on at most {context.spec.effective_strategies().top_models} "
                "model(s)",
            )
        )
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


class SerializableRule(PlanningRule):
    """Rejects models that cannot be persisted in a bundle."""

    def apply(
        self,
        context: PlanningContext,
        candidates: list[models.ModelAdapter],
        decisions: list[experiments.Decision],
    ) -> list[models.ModelAdapter]:
        """Drops adapters whose models are not serializable."""
        kept = []
        for adapter in candidates:
            if adapter.capabilities.serializable:
                kept.append(adapter)
            else:
                decisions.append(
                    experiments.Decision(
                        adapter.name,
                        Action.REJECT,
                        "model cannot be serialized into a bundle",
                    )
                )
        return kept


class HighCardinalityRule(PlanningRule):
    """Plans for categorical features with many levels.

    Native-categorical models are searched first, and regularized target
    encoding is evaluated as a data strategy; neither is assumed to help.
    """

    def apply(
        self,
        context: PlanningContext,
        candidates: list[models.ModelAdapter],
        decisions: list[experiments.Decision],
    ) -> list[models.ModelAdapter]:
        """Prefers native-categorical models when such features exist."""
        wide = context.profile.feature_plan.high_cardinality
        if not wide:
            return candidates
        native = [
            a
            for a in candidates
            if a.capabilities.native_categorical
            and a.categorical_params([0])
            and a.family != "baseline"
        ]
        if native:
            decisions.append(
                experiments.Decision(
                    "native_categorical_boosting",
                    Action.INCLUDE,
                    f"{len(wide)} high-cardinality categorical feature(s) "
                    f"({', '.join(wide)}); searched first: "
                    + ", ".join(a.name for a in native),
                )
            )
        else:
            decisions.append(
                experiments.Decision(
                    "native_categorical_boosting",
                    Action.DEFER,
                    "no compatible model handles categoricals natively",
                )
            )
        strategies_spec = context.spec.effective_strategies()
        enabled = (
            "target_encoding" in strategies_spec.names
            and strategies_spec.top_models > 0
        )
        decisions.append(
            experiments.Decision(
                "regularized_target_encoding",
                Action.INCLUDE if enabled else Action.DEFER,
                "evaluated as the 'target_encoding' strategy on the original "
                "validation folds"
                if enabled
                else "strategy search disabled or 'target_encoding' not listed",
            )
        )
        baselines = [a for a in candidates if a.family == "baseline"]
        rest = [a for a in candidates if a not in baselines + native]
        return baselines + native + rest


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


class HistoryRule(PlanningRule):
    """Searches models that won on similar past datasets first.

    Only the order changes; every candidate is still evaluated, and runs on
    the same dataset are excluded from the evidence.
    """

    def apply(
        self,
        context: PlanningContext,
        candidates: list[models.ModelAdapter],
        decisions: list[experiments.Decision],
    ) -> list[models.ModelAdapter]:
        """Moves historically successful models ahead of the others."""
        manifest = context.profile.manifest
        learner = knowledge.MetaLearner(list(context.history))
        recommendations = learner.recommend(
            knowledge.fingerprint(context.profile, context.task),
            context.task,
            manifest.dataset_id,
            manifest.data_hash,
        )
        weights = dict(recommendations)
        baselines = [a for a in candidates if a.family == "baseline"]
        preferred = sorted(
            (a for a in candidates if a.name in weights and a not in baselines),
            key=lambda a: -weights[a.name],
        )
        if not preferred:
            decisions.append(
                experiments.Decision(
                    "meta_learning",
                    Action.DEFER,
                    "no comparable history on other datasets",
                )
            )
            return candidates
        rest = [a for a in candidates if a not in baselines + preferred]
        decisions.append(
            experiments.Decision(
                "meta_learning",
                Action.INCLUDE,
                "searched first, from similar past datasets: "
                + ", ".join(
                    f"{a.name} ({weights[a.name]:.2f})" for a in preferred
                ),
            )
        )
        return baselines + preferred + rest


class StrategyRule(PlanningRule):
    """Records which data strategies will be evaluated and which cannot."""

    def apply(
        self,
        context: PlanningContext,
        candidates: list[models.ModelAdapter],
        decisions: list[experiments.Decision],
    ) -> list[models.ModelAdapter]:
        """Documents strategy applicability; leaves candidates unchanged."""
        spec = context.spec.effective_strategies()
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
        search = context.spec.effective_search()
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


class EnsembleRule(PlanningRule):
    """Records whether a stacking ensemble will be assembled."""

    def apply(
        self,
        context: PlanningContext,
        candidates: list[models.ModelAdapter],
        decisions: list[experiments.Decision],
    ) -> list[models.ModelAdapter]:
        """Documents the ensemble decision; leaves candidates unchanged."""
        size = context.spec.ensemble.size
        distinct = sum(
            1
            for a in candidates
            if a.family not in ("baseline", "ensemble")
            and not a.capabilities.isolated_process
        )
        if size < 2 or distinct < 2:
            reason = (
                "disabled (ensemble.size < 2)"
                if size < 2
                else "fewer than two eligible distinct models"
            )
            action = Action.DEFER
        else:
            reason = (
                f"stack the best of up to {size} distinct models; adopted "
                "only if it beats its best member in cross-validation"
            )
            action = Action.INCLUDE
        decisions.append(experiments.Decision("stacking", action, reason))
        return candidates


def default_rules() -> tuple[PlanningRule, ...]:
    """Returns the standard rule chain."""
    return (
        GpuRule(),
        SharedProcessRule(),
        MemoryRule(),
        SerializableRule(),
        FastBudgetRule(),
        ApprovedModelsRule(),
        HighCardinalityRule(),
        HistoryRule(),
        StrategyRule(),
        TuningRule(),
        EnsembleRule(),
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
        metric = self.catalog.resolve(spec.primary_metric, task)
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
        strategy = splitters.resolve_strategy_name(spec, task)
        resolved = tasks.ProjectSpec.model_validate(
            spec.model_dump()
            | {
                "metric": metric.name,
                "validation": spec.validation.model_dump()
                | {"strategy": strategy},
            }
        )
        identity = (
            f"{resolved.config_hash()}|{profile.manifest.data_hash}|{names}"
        )
        plan_id = hashlib.sha256(identity.encode()).hexdigest()[:10]
        return experiments.ExperimentPlan(
            plan_id=f"plan_{plan_id}",
            task=task.value,
            metric=metric.name,
            validation_strategy=strategy,
            candidate_models=names,
            optimization_budget_seconds=spec.resources.time_budget_seconds,
            decisions=tuple(decisions),
            warnings=profile.risk_findings,
        )
