"""Candidate-plan generation through composable planning rules.

The planner decides which experiments are worth trying and records why; the
evaluation layer decides whether they help. Nothing here asserts that a
technique will improve performance. Each `Refiner` is an independent,
polymorphic policy applied in order to the candidate list.
"""

import hashlib

from scmle.contracts import errors
from scmle.contracts import experiments
from scmle.contracts import models
from scmle.contracts import tasks
from scmle.data.splitting import splitters
from scmle.evaluation import metrics
import scmle.models.registry
from scmle.pipeline import knowledge
from scmle.pipeline.rules import Budget
from scmle.pipeline.rules import Cardinality
from scmle.pipeline.rules import Exclusion
from scmle.pipeline.rules import Memory
from scmle.pipeline.rules import Persistence
from scmle.pipeline.rules import Refiner
from scmle.pipeline.rules import Situation
from scmle.pipeline.rules import Streaming
from scmle.preparation import strategies

__all__ = [
    "MIN_DISTINCT_MODELS",
    "MIN_ENSEMBLE_SIZE",
    "Approval",
    "Budget",
    "Cardinality",
    "Ensemble",
    "Exclusion",
    "GpuFilter",
    "History",
    "Memory",
    "MonotonicFilter",
    "Persistence",
    "Planner",
    "Refiner",
    "Situation",
    "Strategy",
    "Streaming",
    "Tuning",
    "ruleset",
]

MIN_ENSEMBLE_SIZE = 2
MIN_DISTINCT_MODELS = 2


class GpuFilter(Refiner):
    """Honours the GPU policy: `required` keeps only GPU-capable models."""

    def apply(
        self,
        context: Situation,
        candidates: list[models.Model],
        decisions: list[experiments.Decision],
    ) -> list[models.Model]:
        """Filters for GPU capability when a GPU is required.

        Args:
            context: Planning facts.
            candidates: Candidates surviving earlier rules.
            decisions: Decision log; a rejection is appended for each model that
                cannot use a GPU.

        Returns:
            All candidates unless `resources.gpu` is `required`; then only those
            that can use a GPU.

        Raises:
            BlockedError: If a GPU is required but no candidate can use one.
        """
        if context.spec.resources.gpu != "required":
            return candidates
        capable = [a for a in candidates if a.accelerated()]
        if not capable:
            raise errors.BlockedError(
                "resources.gpu='required' but no compatible model can use a "
                "GPU; set gpu='auto' or install a GPU-capable adapter"
            )
        decisions.extend(
            experiments.Decision(
                adapter.name,
                experiments.Edict.REJECT,
                "gpu required",
            )
            for adapter in candidates
            if adapter not in capable
        )
        return capable


class MonotonicFilter(Refiner):
    """Keeps only models that can enforce the requested monotonic signs."""

    def apply(
        self,
        context: Situation,
        candidates: list[models.Model],
        decisions: list[experiments.Decision],
    ) -> list[models.Model]:
        """Drops models that cannot enforce monotonic constraints.

        Args:
            context: Planning facts.
            candidates: Candidates surviving earlier rules.
            decisions: Decision log; a rejection is appended for each model
                dropped.

        Returns:
            The candidates unchanged if no constraints are requested; otherwise
            those that can enforce them, plus baselines.

        Raises:
            BlockedError: For multiclass tasks (not enforceable) or when no
                model that can enforce the constraints is available.
        """
        if not context.spec.data.monotonic_constraints:
            return candidates
        if context.task is tasks.Task.MULTICLASS:
            raise errors.BlockedError(
                "monotonic constraints are supported for binary "
                "classification and regression, not multiclass"
            )
        kept = []
        for adapter in candidates:
            if adapter.monotonic_param or adapter.family == "baseline":
                kept.append(adapter)
            else:
                decisions.append(
                    experiments.Decision(
                        adapter.name,
                        experiments.Edict.REJECT,
                        "cannot enforce monotonic constraints",
                    )
                )
        if not any(a.monotonic_param for a in kept):
            raise errors.BlockedError(
                "no available model can enforce the monotonic constraints; "
                "install the 'boosting' extra or relax the constraints"
            )
        return kept


class Approval(Refiner):
    """Restricts candidates to a human-approved subset."""

    def apply(
        self,
        context: Situation,
        candidates: list[models.Model],
        decisions: list[experiments.Decision],
    ) -> list[models.Model]:
        """Keeps only approved models; refuses unknown approvals.

        Args:
            context: Planning facts.
            candidates: Candidates surviving earlier rules.
            decisions: Decision log; a rejection is appended for each model not
                approved.

        Returns:
            The candidates unchanged if no approval list was given; otherwise
            the approved ones.

        Raises:
            ConfigError: If an approved model is not among the candidates.
        """
        approved = context.approved_models
        if approved is None:
            return candidates
        unknown = set(approved) - {a.name for a in candidates}
        if unknown:
            raise errors.ConfigError(
                f"approved models not compatible with this plan: "
                f"{sorted(unknown)}"
            )
        decisions.extend(
            experiments.Decision(
                adapter.name,
                experiments.Edict.REJECT,
                "not in approved plan",
            )
            for adapter in candidates
            if adapter.name not in approved
        )
        return [a for a in candidates if a.name in approved]


class History(Refiner):
    """Searches models that won on similar past datasets first.

    Only the order changes; every candidate is still evaluated, and runs on
    the same dataset are excluded from the evidence.
    """

    def apply(
        self,
        context: Situation,
        candidates: list[models.Model],
        decisions: list[experiments.Decision],
    ) -> list[models.Model]:
        """Moves historically successful models ahead of the others.

        Args:
            context: Planning facts, including the history.
            candidates: Candidates surviving earlier rules.
            decisions: Decision log; records whether comparable history existed.

        Returns:
            The candidates reordered as baselines, models that won on similar
            past datasets (best first), then the rest; unchanged if there is no
            comparable history.
        """
        manifest = context.profile.manifest
        learner = knowledge.Recommender(list(context.history))
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
                    experiments.Edict.DEFER,
                    "no comparable history on other datasets",
                )
            )
            return candidates
        rest = [a for a in candidates if a not in baselines + preferred]
        decisions.append(
            experiments.Decision(
                "meta_learning",
                experiments.Edict.INCLUDE,
                "searched first, from similar past datasets: "
                + ", ".join(
                    f"{a.name} ({weights[a.name]:.2f})" for a in preferred
                ),
            )
        )
        return baselines + preferred + rest


class Strategy(Refiner):
    """Records which data strategies will be evaluated and which cannot."""

    def apply(
        self,
        context: Situation,
        candidates: list[models.Model],
        decisions: list[experiments.Decision],
    ) -> list[models.Model]:
        """Documents strategy applicability; leaves candidates unchanged.

        Args:
            context: Planning facts.
            candidates: Candidates surviving earlier rules.
            decisions: Decision log; one entry for each configured strategy.

        Returns:
            The candidates, unchanged.
        """
        spec = context.spec.effective()
        for name in spec.names:
            strategy = strategies.lookup(name)
            reasons = {
                a.name: strategy.applicability(context.profile, context.task, a)
                for a in candidates
            }
            applicable = sorted(n for n, r in reasons.items() if r is None)
            if applicable and spec.top_models > 0:
                decisions.append(
                    experiments.Decision(
                        name,
                        experiments.Edict.INCLUDE,
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
                        (
                            reasons[a.name]
                            for a in sorted(
                                candidates, key=lambda a: a.family == "baseline"
                            )
                            if reasons[a.name]
                        ),
                        "no candidate models",
                    )
                )
                decisions.append(
                    experiments.Decision(
                        name,
                        experiments.Edict.REJECT,
                        f"not applicable: {why}",
                    )
                )
        return candidates


class Tuning(Refiner):
    """Records whether hyperparameter search will run after screening."""

    def apply(
        self,
        context: Situation,
        candidates: list[models.Model],
        decisions: list[experiments.Decision],
    ) -> list[models.Model]:
        """Documents the search decision; leaves candidates unchanged.

        Args:
            context: Planning facts.
            candidates: Candidates surviving earlier rules.
            decisions: Decision log; one entry for hyperparameter tuning.

        Returns:
            The candidates, unchanged.
        """
        search = context.spec.budgeted()
        tunable = [a.name for a in candidates if a.search_space is not None]
        if search.top_models == 0 or not tunable:
            decisions.append(
                experiments.Decision(
                    "hyperparameter_tuning",
                    experiments.Edict.DEFER,
                    "disabled (search.top_models=0) or no tunable candidate",
                )
            )
        else:
            decisions.append(
                experiments.Decision(
                    "hyperparameter_tuning",
                    experiments.Edict.INCLUDE,
                    f"{search.optimizer} search on the top "
                    f"{search.top_models} of {len(tunable)} tunable models, "
                    f"{search.configs_per_model} configs each, "
                    f"{search.prune_fraction:.0%} pruned after the first rung",
                )
            )
        return candidates


class Ensemble(Refiner):
    """Records whether a stacking ensemble will be assembled."""

    def apply(
        self,
        context: Situation,
        candidates: list[models.Model],
        decisions: list[experiments.Decision],
    ) -> list[models.Model]:
        """Documents the ensemble decision; leaves candidates unchanged.

        Args:
            context: Planning facts.
            candidates: Candidates surviving earlier rules.
            decisions: Decision log; one entry each for stacking and blending.

        Returns:
            The candidates, unchanged.
        """
        size = context.spec.ensemble.size
        distinct = sum(
            1 for a in candidates if a.family not in ("baseline", "ensemble")
        )
        if size < MIN_ENSEMBLE_SIZE or distinct < MIN_DISTINCT_MODELS:
            reason = (
                "disabled (ensemble.size < 2)"
                if size < MIN_ENSEMBLE_SIZE
                else "fewer than two eligible distinct models"
            )
            action = experiments.Edict.DEFER
        else:
            reason = (
                f"stack the best of up to {size} distinct models; adopted "
                "only if it beats its best member in cross-validation"
            )
            action = experiments.Edict.INCLUDE
        reason_blend = reason.replace("stack the best", "blend the best")
        decisions.append(experiments.Decision("stacking", action, reason))
        decisions.append(experiments.Decision("blending", action, reason_blend))
        return candidates


def ruleset() -> tuple[Refiner, ...]:
    """Returns the standard rule chain.

    Returns:
        One instance of each rule, in the order they are applied.
    """
    return (
        Exclusion(),
        MonotonicFilter(),
        GpuFilter(),
        Memory(),
        Persistence(),
        Streaming(),
        Budget(),
        Approval(),
        Cardinality(),
        History(),
        Strategy(),
        Tuning(),
        Ensemble(),
    )


class Planner:
    """Builds an `Plan` from a profile.

    Attributes:
        registry: Source of candidate adapters.
        catalog: Metric catalog used to resolve the primary metric.
        rules: The rule chain.
    """

    def __init__(
        self,
        registry: scmle.models.registry.Catalog,
        catalog: metrics.Repertoire = metrics.CATALOG,
        rules: tuple[Refiner, ...] | None = None,
    ) -> None:
        """Initializes the planner.

        Args:
            registry: Source of candidate adapters.
            catalog: Metric catalog used to resolve the primary metric.
            rules: Rule chain; defaults to `ruleset()`.
        """
        self.registry = registry
        self.catalog = catalog
        self.rules = rules if rules is not None else ruleset()

    def plan(self, context: Situation) -> experiments.Plan:
        """Builds the experiment plan.

        Models the registry rejects are logged first, then each rule is applied
        in order, and the surviving candidates are logged as included.

        Args:
            context: Planning facts.

        Returns:
            A deterministic, auditable plan whose id is derived from the
            resolved configuration, the data hash and the candidate names.

        Raises:
            ConfigError: If no candidate model remains.
        """
        spec, profile, task = context.spec, context.profile, context.task
        metric = self.catalog.resolve(spec.primary_metric, task)
        compatible, rejected = self.registry.candidates(
            task,
            profile.manifest.row_count,
            tuple(spec.models.experimental_models),
        )
        decisions = [
            experiments.Decision(name, experiments.Edict.REJECT, reason)
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
                experiments.Edict.INCLUDE,
                (
                    f"{'experimental-only ' if not a.auto_candidate else ''}"
                    f"{a.family} candidate (cost {a.capabilities.cost})"
                ),
            )
            for a in candidates
        ]
        if profile.admonition:
            decisions.append(
                experiments.Decision(
                    "leakage_screen",
                    experiments.Edict.DEFER,
                    profile.admonition,
                )
            )
        names = tuple(a.name for a in candidates)
        strategy = splitters.scheme(spec, task)
        resolved = tasks.Config.model_validate(
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
        return experiments.Plan(
            plan_id=f"plan_{plan_id}",
            task=task.value,
            metric=metric.name,
            validation_strategy=strategy,
            candidate_models=names,
            optimization_budget_seconds=spec.resources.time_budget_seconds,
            decisions=tuple(decisions),
            warnings=profile.risk_findings,
        )
