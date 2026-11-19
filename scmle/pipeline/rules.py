"""Planning facts, the rule interface and the resource and data rules.

A `Refiner` filters the candidate models and records why. The rules here
look at the specification and the data (memory, budget, streaming,
persistence, cardinality, exclusions); `planner` holds the rest of the chain
and the `Planner` that applies it.
"""

import abc
import dataclasses

from scmle.contracts import data
from scmle.contracts import experiments
from scmle.contracts import models
from scmle.contracts import tasks

EXPENSIVE_COST = 3
FAST_EXCEPTIONS = frozenset(
    {
        "hist_gradient_boosting",
        "multi_hist_gradient_boosting",
        "ordinal_hist_gradient_boosting",
        "pointwise_ranker",
        "uplift_t_learner",
    }
)
"""Boosted learners cheap enough to keep on a tight budget, one per task."""


@dataclasses.dataclass(frozen=True, slots=True)
class Situation:
    """Facts available to planning rules.

    Attributes:
        spec: Project specification.
        profile: Dataset profile.
        task: Resolved task type.
        approved_models: Models a human approved, or None for no restriction.
        history: Past runs available for meta-learning.
    """

    spec: tasks.Config
    profile: data.Portrait
    task: tasks.Task
    approved_models: tuple[str, ...] | None = None
    history: tuple[experiments.Precedent, ...] = ()


class Refiner(abc.ABC):
    """A policy that filters candidates and records its decisions."""

    @abc.abstractmethod
    def apply(
        self,
        context: Situation,
        candidates: list[models.Model],
        decisions: list[experiments.Decision],
    ) -> list[models.Model]:
        """Applies the rule.

        Args:
            context: Planning facts.
            candidates: Candidates surviving earlier rules.
            decisions: Decision log to append to.

        Returns:
            The candidates that remain, in the order they will be searched.

        Raises:
            ConfigError: If the rule's input is inconsistent. A rule may also
                raise `BlockedError` when it leaves no workable candidate.
        """


class Memory(Refiner):
    """Rejects models whose estimated training memory exceeds the limit.

    The estimate is `rows x features x 8 bytes x memory_factor`, where the
    factor is each adapter's declared memory characteristic.
    """

    def apply(
        self,
        context: Situation,
        candidates: list[models.Model],
        decisions: list[experiments.Decision],
    ) -> list[models.Model]:
        """Drops candidates that would not fit in `memory_limit_gb`.

        Args:
            context: Planning facts.
            candidates: Candidates surviving earlier rules.
            decisions: Decision log; a rejection is appended for each model
                dropped.

        Returns:
            The candidates whose estimated training memory is within the limit.
        """
        rows = context.profile.manifest.rows
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
                        experiments.Edict.REJECT,
                        f"estimated training memory {estimate:.2f} GB exceeds "
                        f"the {limit:g} GB limit",
                    )
                )
            else:
                kept.append(adapter)
        return kept


class Budget(Refiner):
    """Prefers fast models when the time budget is small."""

    def apply(
        self,
        context: Situation,
        candidates: list[models.Model],
        decisions: list[experiments.Decision],
    ) -> list[models.Model]:
        """Drops expensive models and shrinks the search on a tight budget.

        Args:
            context: Planning facts.
            candidates: Candidates surviving earlier rules.
            decisions: Decision log; the reduced search and each rejection are
                appended.

        Returns:
            All candidates if the budget is not tight; otherwise those with a
            cost below 3 or listed in `FAST_EXCEPTIONS`, the baselines first,
            then the `FAST_EXCEPTIONS` and then the rest in their order, so a
            trial window that closes early skips a weak model, not the
            boosted learner.
        """
        if not context.spec.brief:
            return candidates
        search = context.spec.budgeted()
        decisions.append(
            experiments.Decision(
                "reduce_search_complexity",
                experiments.Edict.INCLUDE,
                f"time budget < {tasks.FAST_BUDGET_SECONDS}s: tune at most "
                f"{search.top_models} model(s) with "
                f"{search.configs_per_model} config(s) each and try strategies "
                f"on at most {context.spec.effective().top_models} "
                "model(s)",
            )
        )
        kept = []
        for adapter in candidates:
            expensive = adapter.capabilities.cost >= EXPENSIVE_COST
            if expensive and adapter.name not in FAST_EXCEPTIONS:
                decisions.append(
                    experiments.Decision(
                        adapter.name,
                        experiments.Edict.REJECT,
                        f"time budget < {tasks.FAST_BUDGET_SECONDS}s: "
                        "fast models only",
                    )
                )
            else:
                kept.append(adapter)
        first = [a for a in kept if a.family == "baseline"]
        first += [a for a in kept if a.name in FAST_EXCEPTIONS]
        return first + [a for a in kept if a not in first]


class Streaming(Refiner):
    """Keeps only models that can learn from batches when streaming."""

    def apply(
        self,
        context: Situation,
        candidates: list[models.Model],
        decisions: list[experiments.Decision],
    ) -> list[models.Model]:
        """Drops non-incremental models when training on a stream.

        When not streaming, models built only for streaming are dropped instead.

        Args:
            context: Planning facts.
            candidates: Candidates surviving earlier rules.
            decisions: Decision log; a rejection is appended for each model
                dropped.

        Returns:
            When streaming, the incremental models and the dummy baseline;
            otherwise the models that are not stream-only.
        """
        if context.spec.data.training != "stream":
            kept = []
            for adapter in candidates:
                if adapter.capabilities.stream_only:
                    decisions.append(
                        experiments.Decision(
                            adapter.name,
                            experiments.Edict.REJECT,
                            "built for data larger than memory; set "
                            "memory_budget_gb with training='stream'",
                        )
                    )
                else:
                    kept.append(adapter)
            return kept
        kept = []
        for adapter in candidates:
            if adapter.capabilities.incremental or adapter.name == "dummy":
                kept.append(adapter)
            else:
                decisions.append(
                    experiments.Decision(
                        adapter.name,
                        experiments.Edict.REJECT,
                        "cannot learn incrementally, so it cannot train on "
                        "a stream larger than memory",
                    )
                )
        return kept


class Persistence(Refiner):
    """Rejects models that cannot be persisted in a bundle."""

    def apply(
        self,
        context: Situation,  # noqa: ARG002 -- Refiner.apply interface
        candidates: list[models.Model],
        decisions: list[experiments.Decision],
    ) -> list[models.Model]:
        """Drops adapters whose models are not serializable.

        Args:
            context: Planning facts. Unused.
            candidates: Candidates surviving earlier rules.
            decisions: Decision log; a rejection is appended for each model
                dropped.

        Returns:
            The candidates that can be persisted in a bundle.
        """
        kept = []
        for adapter in candidates:
            if adapter.capabilities.serializable:
                kept.append(adapter)
            else:
                decisions.append(
                    experiments.Decision(
                        adapter.name,
                        experiments.Edict.REJECT,
                        "model cannot be serialized into a bundle",
                    )
                )
        return kept


class Cardinality(Refiner):
    """Plans for categorical features with many levels.

    Native-categorical models are searched first, and regularized target
    encoding is evaluated as a data strategy; neither is assumed to help.
    """

    def apply(
        self,
        context: Situation,
        candidates: list[models.Model],
        decisions: list[experiments.Decision],
    ) -> list[models.Model]:
        """Prefers native-categorical models when such features exist.

        Args:
            context: Planning facts.
            candidates: Candidates surviving earlier rules.
            decisions: Decision log; records the native-categorical and
                target-encoding decisions.

        Returns:
            The candidates unchanged if there are no high-cardinality features;
            otherwise the same candidates reordered as baselines, native
            categorical models, then the rest.
        """
        wide = context.profile.feature_plan.high_cardinality
        if not wide:
            return candidates
        native = [
            a
            for a in candidates
            if a.capabilities.native_categorical
            and a.nominals([0])
            and a.family != "baseline"
        ]
        if native:
            decisions.append(
                experiments.Decision(
                    "native_categorical_boosting",
                    experiments.Edict.INCLUDE,
                    f"{len(wide)} high-cardinality categorical feature(s) "
                    f"({', '.join(wide)}); searched first: "
                    + ", ".join(a.name for a in native),
                )
            )
        else:
            decisions.append(
                experiments.Decision(
                    "native_categorical_boosting",
                    experiments.Edict.DEFER,
                    "no compatible model handles categoricals natively",
                )
            )
        strategies_spec = context.spec.effective()
        enabled = (
            "target_encoding" in strategies_spec.names
            and strategies_spec.top_models > 0
        )
        decisions.append(
            experiments.Decision(
                "regularized_target_encoding",
                experiments.Edict.INCLUDE
                if enabled
                else experiments.Edict.DEFER,
                "evaluated as the 'target_encoding' strategy on the original "
                "validation folds"
                if enabled
                else "strategy search disabled or 'target_encoding' not listed",
            )
        )
        baselines = [a for a in candidates if a.family == "baseline"]
        rest = [a for a in candidates if a not in baselines + native]
        return baselines + native + rest


class Exclusion(Refiner):
    """Removes the models and families the specification excludes."""

    def apply(
        self,
        context: Situation,
        candidates: list[models.Model],
        decisions: list[experiments.Decision],
    ) -> list[models.Model]:
        """Drops excluded models, recording why.

        Args:
            context: Planning facts.
            candidates: Candidates surviving earlier rules.
            decisions: Decision log; a rejection is appended for each model
                dropped.

        Returns:
            The candidates whose name and family are not excluded.
        """
        spec = context.spec.models
        kept = []
        for adapter in candidates:
            if adapter.name in spec.exclude_models:
                reason = "excluded by models.exclude_models"
            elif adapter.family in spec.exclude_families:
                reason = f"family '{adapter.family}' is excluded"
            else:
                kept.append(adapter)
                continue
            decisions.append(
                experiments.Decision(
                    adapter.name, experiments.Edict.REJECT, reason
                )
            )
        return kept
