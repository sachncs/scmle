"""The selection stage and its Pareto frontier."""

import dataclasses
from typing import Any
from typing import ClassVar

import scmle.assurance.framing.claim
from scmle.contracts import errors
from scmle.contracts import experiments
from scmle.contracts import models
from scmle.contracts import records
from scmle.evaluation import resources
from scmle.pipeline import products
import scmle.pipeline.context
from scmle.pipeline.stages import cost
from scmle.pipeline.stages import exploration
from scmle.pipeline.stages import interface
from scmle.pipeline.stages import profiling
from scmle.pipeline.stages import training
from scmle.preparation import strategies
import scmle.runtime.state


@dataclasses.dataclass(frozen=True)
class Choice(products.Kept):
    """The selected trial and the refits made to choose it.

    Attributes:
        selected: The winning trial.
        refits: Candidates refit on the development data, by trial id.
        pareto_front: The measured front, when the Pareto option is on.
        pending_ledger: A charter run's confirmation still to open, if any.
        outcomes: The evidence-based decisions so far.
        skipped: The work reduced or skipped for the budget so far.
    """

    selected: experiments.Trial | None
    refits: dict[str, models.Estimator]
    pareto_front: list[records.Layout] | None
    pending_ledger: Any
    outcomes: list[experiments.Decision]
    skipped: list[dict[str, str]]


class SelectStage(interface.Stage):
    """Applies the selection policy and the serving constraints."""

    requires: ClassVar[tuple[type, ...]] = (training.TrainStage,)
    makes: ClassVar[type[products.Product] | None] = Choice
    state: ClassVar[scmle.runtime.state.Progress] = (
        scmle.runtime.state.Progress.SELECTING
    )

    def describe(self, context: scmle.pipeline.context.Context) -> str:
        """Names the policy.

        Args:
            context: The run context, holding the selection policy.

        Returns:
            The selection policy's description.
        """
        return context.policy.description

    def announce(self, context: scmle.pipeline.context.Context) -> None:
        """Emits the `selection.frozen` event for the selected trial.

        Args:
            context: The run context, holding the selected trial.
        """
        selected = scmle.pipeline.context.require(context.selected, "selected")
        context.emit(
            "selection.frozen",
            f"{selected.model_name}/{selected.strategy}",
            (f"trials/{selected.trial_id}.json",),
            {
                "trial_id": selected.trial_id,
                "params": selected.params,
                "cv_score": selected.score_mean,
            },
        )

    def limits(
        self, context: scmle.pipeline.context.Context
    ) -> resources.Constraints:
        """Builds the serving limits from the specification.

        Args:
            context: The run context, supplying the selection settings.

        Returns:
            The model-size and latency limits.
        """
        spec = context.spec.selection
        return resources.Constraints(
            spec.max_model_mb,
            spec.max_p95_latency_ms,
            spec.latency_safety_factor,
        )

    def select_on_front(
        self,
        context: scmle.pipeline.context.Context,
        checked: list[experiments.Trial],
        limits: resources.Constraints,
    ) -> None:
        """Measures every checked candidate and picks from the Pareto front.

        Among candidates that meet the limits and score within
        `pareto_tolerance` of the best, the fastest wins (smaller size breaks
        ties). The whole front is reported so the trade-off is visible.

        Args:
            context: The run context; receives the front and the selection.
            checked: The best-ranked trials to measure.
            limits: The serving limits.

        Raises:
            BlockedError: If no checked candidate satisfies the limits.
        """
        split = scmle.pipeline.context.require(context.split, "split")
        task = scmle.pipeline.context.require(context.task, "task")
        plan = scmle.pipeline.context.require(
            context.profile, "profile"
        ).feature_plan
        metric = scmle.pipeline.context.require(context.metric, "metric")
        sign = 1.0 if metric.greater_is_better else -1.0
        rows = []
        for trial in checked:
            if cost.late(context, trial):
                continue
            candidate = strategies.lookup(trial.strategy).build(
                plan,
                context.registry.get(trial.model_name),
                task,
                context.spec.validation.seed,
                trial.params,
            )
            candidate.fit(split.dev_features, split.dev_target)
            usage = resources.measure(candidate, split.dev_features)
            rows.append(
                Candidate(trial, candidate, usage, limits.violations(usage))
            )
        if not rows:
            context.selected = cost.affordable(context, checked)
            self.announce(context)
            return
        feasible = [r for r in rows if not r.broken]
        if not feasible:
            raise errors.BlockedError(
                f"none of the top {len(checked)} candidates satisfies the "
                "serving constraints; relax the limits or allow more candidates"
            )
        best = max(sign * r.trial.score_mean for r in feasible)
        near = [
            r
            for r in feasible
            if best - sign * r.trial.score_mean
            <= context.spec.selection.pareto_tolerance
        ]
        chosen = min(
            near,
            key=lambda r: (r.usage.p95_latency_ms, r.usage.model_mb),
        )
        context.pareto_front = frontier(rows, chosen, sign)
        trial = chosen.trial
        context.enter(
            f"{trial.model_name}/{trial.strategy}",
            experiments.Edict.INCLUDE,
            f"fastest within {context.spec.selection.pareto_tolerance} of the "
            f"best score ({chosen.usage.p95_latency_ms:.1f} ms p95)",
        )
        context.refits[trial.trial_id] = chosen.candidate
        context.selected = trial
        self.announce(context)

    def execute(self, context: scmle.pipeline.context.Context) -> None:
        """Chooses the model, then opens a deferred confirmation.

        If the confirmation was deferred, it is completed here once the model
        is frozen, unless the coverage claim still waits for calibration.
        Raises `BlockedError` (through `choose`) if no checked candidate
        satisfies the constraints.

        Args:
            context: The run context, holding any pending ledger.
        """
        self.choose(context)
        pending = context.pending_ledger
        if pending is None:
            return
        selected = scmle.pipeline.context.require(context.selected, "selected")
        split = scmle.pipeline.context.require(context.split, "split")
        data = context.spec.data
        subject = None
        if data.baseline_column and data.baseline_subject == "selected_model":
            subject = self.codify(context, pending, selected, split)
        if pending.holds_for_coverage:
            context.pending_ledger = dataclasses.replace(
                pending, subject=subject
            )
            return
        context.pending_ledger = None
        profiling.ProfileStage().conclude(context, pending, subject)

    def codify(
        self,
        context: scmle.pipeline.context.Context,
        pending: "exploration.Escrow",
        selected: experiments.Trial,
        split: Any,
    ) -> scmle.assurance.framing.claim.Selection:
        """Returns the selected model as the rule the baseline claim is about.

        Args:
            context: The run context; the refit pipeline is cached in it.
            pending: The deferred ledger, supplying the view and charter.
            selected: The selected trial.
            split: The split whose development rows the model is refit on if
                no refit exists.

        Returns:
            The selected model with its positive class (the last class in
            sorted order), the cost model's threshold and a description of how
            it was fitted.
        """
        pipeline = context.refits.get(selected.trial_id)
        if pipeline is None:
            pipeline = strategies.lookup(selected.strategy).build(
                scmle.pipeline.context.require(
                    context.profile, "profile"
                ).feature_plan,
                context.registry.get(selected.model_name),
                scmle.pipeline.context.require(context.task, "task"),
                context.spec.validation.seed,
                selected.params,
            )
            pipeline.fit(split.dev_features, split.dev_target)
            context.refits[selected.trial_id] = pipeline
        model = scmle.pipeline.context.require(
            pending.charter.costs, "cost_model"
        )
        classes = sorted(pending.view.target.dropna().unique(), key=str)
        return scmle.assurance.framing.claim.Selection(
            pipeline,
            classes[-1],
            float(model.threshold),
            {
                "model": selected.model_name,
                "strategy": selected.strategy,
                "params": selected.params,
                "trial_id": selected.trial_id,
                "split": split.manifest.fingerprint,
                "seed": context.spec.validation.seed,
            },
        )

    def choose(self, context: scmle.pipeline.context.Context) -> None:
        """Chooses the best trial that satisfies the serving constraints.

        Without serving limits and without the Pareto option the top-ranked
        trial whose refit ends before the run's end is taken unmeasured
        (`affordable`). Candidates whose refit cannot end in time are not
        measured; if none can, the choice falls back to `affordable`. With
        the Pareto option the choice is made on the front
        (`select_on_front`). Otherwise candidates are walked in policy order,
        up to `max_candidates_checked`; each is refit and measured on this
        machine, and the first one within every limit wins.

        Args:
            context: The run context; receives the selected trial.

        Raises:
            BlockedError: If no checked candidate satisfies the constraints.
        """
        ranked = context.policy.rank(
            context.trials,
            scmle.pipeline.context.require(context.metric, "metric"),
            context.registry,
        )
        limits = self.limits(context)
        spec = context.spec.selection
        if not limits.active and not spec.pareto:
            context.selected = cost.affordable(context, ranked)
            self.announce(context)
            return
        split = scmle.pipeline.context.require(context.split, "split")
        task = scmle.pipeline.context.require(context.task, "task")
        plan = scmle.pipeline.context.require(
            context.profile, "profile"
        ).feature_plan
        checked = ranked[: context.spec.selection.max_candidates_checked]
        if spec.pareto:
            self.select_on_front(context, checked, limits)
            return
        measured = 0
        for trial in checked:
            if cost.late(context, trial):
                continue
            measured += 1
            candidate = strategies.lookup(trial.strategy).build(
                plan,
                context.registry.get(trial.model_name),
                task,
                context.spec.validation.seed,
                trial.params,
            )
            candidate.fit(split.dev_features, split.dev_target)
            usage = resources.measure(candidate, split.dev_features)
            broken = limits.violations(usage)
            label = f"{trial.model_name}/{trial.strategy}"
            if broken:
                context.enter(
                    label, experiments.Edict.REJECT, "; ".join(broken)
                )
                continue
            context.enter(
                label,
                experiments.Edict.INCLUDE,
                f"within serving limits ({usage.model_mb:.2f} MB, p95 "
                f"{usage.p95_latency_ms:.1f} ms on {usage.machine})",
            )
            context.refits[trial.trial_id] = candidate
            context.selected = trial
            self.announce(context)
            return
        if not measured:
            context.selected = cost.affordable(context, checked)
            self.announce(context)
            return
        raise errors.BlockedError(
            f"none of the top {len(checked)} candidates satisfies the "
            "serving constraints; relax the limits or allow more candidates"
        )


@dataclasses.dataclass
class Candidate:
    """A candidate refit on the development rows and measured for serving.

    Attributes:
        trial: The trial it came from.
        candidate: The fitted pipeline.
        usage: Its measured size and latency.
        broken: Serving limits it violates.
    """

    trial: experiments.Trial
    candidate: Any
    usage: resources.Usage
    broken: list[str]


def frontier(
    rows: list[Candidate], chosen: Candidate, sign: float
) -> list[dict[str, Any]]:
    """Reports score, latency and size of each measured candidate.

    The report also marks which candidates another one dominates.

    Args:
        rows: The measured candidates.
        chosen: The candidate that was selected.
        sign: 1 if a higher score is better, -1 if a lower score is better.

    Returns:
        One record per candidate, in order: model, strategy, score (as
        recorded, not sign-adjusted), p95 latency, size, limit violations,
        whether another candidate is at least as good on all three and better
        on one, and whether it was chosen.
    """
    points = [
        (
            sign * r.trial.score_mean,
            r.usage.p95_latency_ms,
            r.usage.model_mb,
        )
        for r in rows
    ]

    def dominated(index: int) -> bool:
        a = points[index]
        return any(
            b[0] >= a[0]
            and b[1] <= a[1]
            and b[2] <= a[2]
            and (b[0] > a[0] or b[1] < a[1] or b[2] < a[2])
            for other, b in enumerate(points)
            if other != index
        )

    return [
        {
            "model": r.trial.model_name,
            "strategy": r.trial.strategy,
            "score": r.trial.score_mean,
            "p95_latency_ms": r.usage.p95_latency_ms,
            "model_mb": r.usage.model_mb,
            "violations": list(r.broken),
            "dominated": dominated(index),
            "chosen": r is chosen,
        }
        for index, r in enumerate(rows)
    ]
