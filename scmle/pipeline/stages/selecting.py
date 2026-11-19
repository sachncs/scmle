"""The selection stage and its Pareto frontier."""

import dataclasses
from typing import Any
from typing import ClassVar

import scmle.assurance.framing.claim
from scmle.contracts import errors
from scmle.contracts import experiments
from scmle.contracts import models
from scmle.contracts import records
from scmle.data.splitting import splitters
from scmle.evaluation import resources
from scmle.pipeline import products
from scmle.pipeline import strategy_search
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
    pending_ledger: "exploration.Escrow | None"
    outcomes: list[experiments.Decision]
    skipped: list[dict[str, str]]


@dataclasses.dataclass(frozen=True)
class Pick:
    """The trial a selection rule chose, before the confirmation opens.

    Attributes:
        trial: The selected trial.
        front: The measured Pareto front, when the choice was made on it.
    """

    trial: experiments.Trial
    front: list[records.Layout] | None = None


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
        return context.services.policy.description

    def announce(
        self,
        context: scmle.pipeline.context.Context,
        selected: experiments.Trial,
    ) -> Pick:
        """Emits the `selection.frozen` event for the selected trial.

        Args:
            context: The run context whose journal receives the event.
            selected: The selected trial.

        Returns:
            The selected trial, as picked without a front.
        """
        context.journal.emit(
            "selection.frozen",
            f"{selected.model_name}/{selected.strategy}",
            (f"trials/{selected.trial_id}.json",),
            {
                "trial_id": selected.trial_id,
                "params": selected.params,
                "cv_score": selected.score_mean,
            },
        )
        return Pick(selected)

    def limits(
        self, context: scmle.pipeline.context.Context
    ) -> resources.Constraints:
        """Builds the serving limits from the specification.

        Args:
            context: The run context, supplying the selection settings.

        Returns:
            The model-size and latency limits.
        """
        spec = context.services.spec.selection
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
        refits: dict[str, models.Estimator],
    ) -> Pick:
        """Measures every checked candidate and picks from the Pareto front.

        Among candidates that meet the limits and score within
        `pareto_tolerance` of the best, the fastest wins (smaller size breaks
        ties). The whole front is reported so the trade-off is visible.

        Args:
            context: The run context, holding the split and the profile.
            checked: The best-ranked trials to measure.
            limits: The serving limits.
            refits: Receives the chosen candidate's refit, by trial id.

        Returns:
            The chosen trial and the measured front; without a measured
            candidate, the affordable one and no front.

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
                context.services.registry.get(trial.model_name),
                task,
                context.services.spec.validation.seed,
                trial.params,
            )
            candidate.fit(split.dev_features, split.dev_target)
            usage = resources.measure(candidate, split.dev_features)
            rows.append(
                Candidate(trial, candidate, usage, limits.violations(usage))
            )
        if not rows:
            return self.announce(context, cost.affordable(context, checked))
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
            <= context.services.spec.selection.pareto_tolerance
        ]
        chosen = min(
            near,
            key=lambda r: (r.usage.p95_latency_ms, r.usage.model_mb),
        )
        trial = chosen.trial
        context.journal.enter(
            f"{trial.model_name}/{trial.strategy}",
            experiments.Edict.INCLUDE,
            "fastest within "
            f"{context.services.spec.selection.pareto_tolerance} of the "
            f"best score ({chosen.usage.p95_latency_ms:.1f} ms p95)",
        )
        refits[trial.trial_id] = chosen.candidate
        self.announce(context, trial)
        return Pick(trial, frontier(rows, chosen, sign))

    def execute(self, context: scmle.pipeline.context.Context) -> Choice:
        """Chooses the model, then opens a deferred confirmation.

        If the confirmation was deferred, it is completed here once the model
        is frozen, unless the coverage claim still waits for calibration.
        Raises `BlockedError` (through `choose`) if no checked candidate
        satisfies the constraints.

        Args:
            context: The run context, holding any pending ledger.

        Returns:
            The selected trial, the refits made on the way, the front and
            the ledger still pending.
        """
        refits = dict(context.refits)
        pick = self.choose(context, refits)
        pending = context.pending_ledger
        if pending is not None:
            split = scmle.pipeline.context.require(context.split, "split")
            data = context.services.spec.data
            subject = None
            if (
                data.baseline_column
                and data.baseline_subject == "selected_model"
            ):
                subject = codify(context, pending, pick.trial, split, refits)
            if pending.holds_for_coverage:
                pending = dataclasses.replace(pending, subject=subject)
            else:
                profiling.ProfileStage().conclude(context, pending, subject)
                pending = None
        with context.journal.lock:
            return Choice(
                pick.trial,
                refits,
                pick.front,
                pending,
                list(context.journal.outcomes),
                list(context.journal.skipped),
            )

    def choose(
        self,
        context: scmle.pipeline.context.Context,
        refits: dict[str, models.Estimator],
    ) -> Pick:
        """Chooses the best trial that satisfies the serving constraints.

        Only the trials that earned their place compete (`earn`). Without
        serving limits and without the Pareto option the top-ranked trial
        whose refit ends in time is taken unmeasured (`affordable`).
        Candidates whose refit cannot end in time are not measured; if none
        can, the choice falls back to `affordable`. With the Pareto option
        the choice is made on the front (`select_on_front`). Otherwise the
        first `max_candidates_checked` in policy order are refit and measured
        on this machine, and the first one within every limit wins.

        Args:
            context: The run context, holding the trials, split and profile.
            refits: Receives the refit of a measured choice, by trial id.

        Returns:
            The selected trial, with the front when it was chosen on one.

        Raises:
            BlockedError: If no checked candidate satisfies the constraints.
        """
        metric = scmle.pipeline.context.require(context.metric, "metric")
        earned = earn(context, list(context.journal.trials))
        ranked = context.services.policy.rank(
            earned, metric, context.services.registry
        )
        limits = self.limits(context)
        spec = context.services.spec.selection
        if not limits.active and not spec.pareto:
            return self.announce(context, cost.affordable(context, ranked))
        checked = ranked[: spec.max_candidates_checked]
        if spec.pareto:
            return self.select_on_front(context, checked, limits, refits)
        measured = 0
        for trial in checked:
            if cost.late(context, trial):
                continue
            measured += 1
            candidate = self.measure(context, trial, limits)
            if candidate is not None:
                refits[trial.trial_id] = candidate
                return self.announce(context, trial)
        if not measured:
            return self.announce(context, cost.affordable(context, checked))
        raise errors.BlockedError(
            f"none of the top {len(checked)} candidates satisfies the "
            "serving constraints; relax the limits or allow more candidates"
        )

    def measure(
        self,
        context: scmle.pipeline.context.Context,
        trial: experiments.Trial,
        limits: resources.Constraints,
    ) -> models.Estimator | None:
        """Refits a candidate on the development rows and checks its cost.

        The verdict is entered in the journal: included with its measured
        size and latency, or rejected with the limits it broke.

        Args:
            context: The run context, holding the split, task and profile.
            trial: The candidate's trial.
            limits: The serving limits.

        Returns:
            The refit when it is within every limit, else None.
        """
        split = scmle.pipeline.context.require(context.split, "split")
        task = scmle.pipeline.context.require(context.task, "task")
        plan = scmle.pipeline.context.require(
            context.profile, "profile"
        ).feature_plan
        candidate = strategies.lookup(trial.strategy).build(
            plan,
            context.services.registry.get(trial.model_name),
            task,
            context.services.spec.validation.seed,
            trial.params,
        )
        candidate.fit(split.dev_features, split.dev_target)
        usage = resources.measure(candidate, split.dev_features)
        broken = limits.violations(usage)
        label = f"{trial.model_name}/{trial.strategy}"
        if broken:
            context.journal.enter(
                label, experiments.Edict.REJECT, "; ".join(broken)
            )
            return None
        context.journal.enter(
            label,
            experiments.Edict.INCLUDE,
            f"within serving limits ({usage.model_mb:.2f} MB, p95 "
            f"{usage.p95_latency_ms:.1f} ms on {usage.machine})",
        )
        return candidate


def earn(
    context: scmle.pipeline.context.Context,
    trials: list[experiments.Trial],
) -> list[experiments.Trial]:
    """Keeps the default pipelines and the variants that beat theirs.

    A variant (a data strategy, tuned hyperparameters or an ensemble) stays
    only when the shared folds make it credibly better than its reference
    (`strategy_search.odds` at least `strategy_search.CONFIDENCE`): its
    model's default pipeline, or the best default pipeline for an ensemble.
    Each one dropped is rejected on record.

    Args:
        context: The run context, holding the metric and the journal.
        trials: Every trial of the run.

    Returns:
        The trials that compete, in their order; all of them when there is
        no successful default pipeline or no successful variant.
    """
    metric = scmle.pipeline.context.require(context.metric, "metric")
    sign = 1 if metric.greater_is_better else -1
    ok = [t for t in trials if t.status is experiments.Termination.OK]
    defaults = {t.model_name: t for t in ok if strategy_search.plain(t)}
    variants = {id(t) for t in ok if not strategy_search.plain(t)}
    if not defaults or not variants:
        return trials
    best = max(defaults.values(), key=lambda t: sign * t.score_mean)
    needed = strategy_search.CONFIDENCE
    kept = []
    for trial in trials:
        reference = defaults.get(trial.model_name, best)
        chance = (
            strategy_search.odds(trial, reference, sign)
            if id(trial) in variants
            else 1.0
        )
        if chance >= needed:
            kept.append(trial)
            continue
        context.journal.enter(
            f"{trial.model_name}/{trial.strategy}",
            experiments.Edict.REJECT,
            f"{trial.trial_id}: cv {metric.name} {trial.score_mean:.4f} vs "
            f"{reference.model_name} default {reference.score_mean:.4f}; "
            f"probability better {chance:.3f}, needed {needed:.3f}",
        )
    return kept


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


def codify(
    context: scmle.pipeline.context.Context,
    pending: "exploration.Escrow",
    selected: experiments.Trial,
    split: splitters.Split,
    refits: dict[str, models.Estimator],
) -> scmle.assurance.framing.claim.Selection:
    """Returns the selected model as the rule the baseline claim is about.

    Args:
        context: The run context, holding the profile and services.
        pending: The deferred ledger, supplying the view and charter.
        selected: The selected trial.
        split: The split whose development rows the model is refit on if
            no refit exists.
        refits: The refits so far, by trial id; receives this one.

    Returns:
        The selected model with its positive class (the last class in
        sorted order), the cost model's threshold and a description of how
        it was fitted.
    """
    pipeline = refits.get(selected.trial_id)
    if pipeline is None:
        pipeline = strategies.lookup(selected.strategy).build(
            scmle.pipeline.context.require(
                context.profile, "profile"
            ).feature_plan,
            context.services.registry.get(selected.model_name),
            scmle.pipeline.context.require(context.task, "task"),
            context.services.spec.validation.seed,
            selected.params,
        )
        pipeline.fit(split.dev_features, split.dev_target)
        refits[selected.trial_id] = pipeline
    model = scmle.pipeline.context.require(pending.charter.costs, "cost_model")
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
            "seed": context.services.spec.validation.seed,
        },
    )


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
