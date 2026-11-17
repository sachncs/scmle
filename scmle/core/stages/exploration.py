"""The claim ledger of a charter run."""

import dataclasses
from typing import Any

from scmle.assurance.admissibility import qualify
from scmle.assurance.charter import contracts
from scmle.assurance.charter import sealing
import scmle.assurance.framing.claim
from scmle.assurance.ledger import confirm
from scmle.assurance.ledger import governed
import scmle.assurance.ledger.budget
from scmle.contracts import errors
from scmle.contracts import tasks
from scmle.core import causal_claim
import scmle.core.context
from scmle.core.stages import admission
from scmle.core.stages import scoring
from scmle.data.robustness import prefilter


@dataclasses.dataclass(frozen=True, slots=True)
class Escrow:
    """A charter run's exploration, finished, with `C` still sealed.

    Attributes:
        explored: The exploration, control and second stage (from `X` only).
        task: The resolved task.
        charter: The verified charter.
        view: The exploration rows.
        structure: Group, time and baseline columns (never candidates).
        graph: The declared DAG, or None.
        found: The qualification of the exploration rows.
        stage: The stability stage.
        signed: Whether the charter signature was verified.
        baseline: The baseline claim about the logistic rule, or None.
        causal: The primary causal claim, or None.
        subject: The rule the baseline claim is about once the model is
            selected, or None.
        holds_for_coverage: Whether `C` stays sealed until the bundle and its
            calibration exist, so the coverage claim can be tested on it.
    """

    explored: governed.Explored
    task: tasks.Task
    charter: contracts.AnalysisCharter
    view: Any
    structure: tuple[str, ...]
    graph: Any
    found: qualify.Qualification
    stage: Any
    signed: bool
    baseline: confirm.Docket | None
    causal: confirm.Docket | None
    subject: Any = None
    holds_for_coverage: bool = False


class Exploration(admission.Admission):
    """The claim ledger of a charter run, from exploration to conclusion.

    Part of `ProfileStage`, which inherits these methods unchanged;
    abstract, never run on its own.
    """

    def run_claim_ledger(
        self, context: scmle.core.context.Context, task: tasks.Task
    ) -> None:
        """Explores `X` under the ledger and confirms on `C`, now or later.

        Runs only for a charter run. Exploration reads the exploration view; the
        sealed confirmation set is handed to the confirmation stage as an
        argument and is opened at most once, after the survivors are frozen in
        the ledger. Tasks other than binary, multiclass and regression are
        recorded as skipped: their findings stay unconfirmed. When the baseline
        claim is about the selected model, or the coverage claim needs the
        calibrated model, the confirmation is deferred until the model is
        frozen (the pending ledger is kept on the context).

        Args:
            context: The run context, holding the custody outcome and vault.
            task: The resolved task.
        """
        outcome = context.custody
        vault = context.custody_vault
        if outcome is None or vault is None:
            return
        if task not in tasks.STANDARD_TASKS:
            context.emit(
                "ledger.skipped",
                f"the claim ledger does not cover {task.value} tasks; "
                "findings stay unconfirmed",
            )
            return
        spec = context.spec
        verified = sealing.read_verified(str(spec.charter_path))
        charter = verified.charter
        data = spec.data
        view = outcome.view
        run_dir = context.store.directory if context.store is not None else None
        structure = tuple(
            c
            for c in (data.group_column, data.time_column, data.baseline_column)
            if c
        )
        block = charter.estimand.causal
        graph = causal_claim.unseal(block.dag) if block is not None else None
        found = self.qualify_exploration(
            context, task, charter, view, structure, graph
        )
        rules = charter.rules
        stage = prefilter.Prefilter(
            lambda columns: self.stress(context, task, charter, view, columns),
            requested=bool(rules and rules.stability_filter),
            rows=len(view.features),
            seed=charter.seed,
        )
        # Declarations are validated before the expensive exploration.
        baseline = self.baseline_runner(
            context, task, charter, view, structure, signed=verified.signed
        )
        causal = self.hypothesise(context, charter, view, graph, found)
        explored = governed.scout(
            charter=charter,
            features=view.features,
            target=view.target,
            classification=task.is_classification,
            run_dir=run_dir,
            exclude=structure + tuple(found.quarantined),
            strata=governed.strata_labels(
                view.features, data.group_column, data.time_column
            ),
            rule=governing(
                "ledger_rule",
                spec.ledger_rule,
                None if rules is None else rules.ledger_rule,
                "e-lond",
            ),
            policy=governing(
                "failure_policy",
                spec.confirmation_failure_policy,
                None if rules is None else rules.failure_policy,
                "unconfirmed",
            ),
            diagnostics=found.checks,
            second_stage=stage,
        )
        pending = Escrow(
            explored,
            task,
            charter,
            view,
            structure,
            graph,
            found,
            stage,
            verified.signed,
            baseline,
            causal,
            holds_for_coverage=scoring.coverage_on_confirmation(charter, task),
        )
        selected = bool(
            data.baseline_column and data.baseline_subject == "selected_model"
        )
        if selected or pending.holds_for_coverage:
            context.pending_ledger = pending
            context.emit(
                "ledger.deferred",
                "the confirmation set stays sealed until "
                + (
                    "the model and its calibration are frozen, so the "
                    "coverage claim can be tested on it"
                    if pending.holds_for_coverage
                    else "the model selection is frozen, so the baseline "
                    "claim can be about that model"
                ),
            )
            return
        self.conclude(context, pending)

    def conclude(
        self,
        context: scmle.core.context.Context,
        pending: "Escrow",
        subject: scmle.assurance.framing.claim.Rule | None = None,
        extra: confirm.Docket | None = None,
    ) -> None:
        """Registers the claims, opens `C` once and writes the reports.

        Args:
            context: The run context, holding the custody vault whose set `C`
                is opened.
            pending: The exploration, finished on `X` alone.
            subject: The rule the baseline claim is about, when it is not
                the logistic rule frozen on `X`.
            extra: A further registered claim tested in the same opening
                (the coverage claim).
        """
        charter = pending.charter
        view = pending.view
        rules = charter.rules
        baseline = pending.baseline
        if subject is not None:
            baseline = self.baseline_runner(
                context,
                pending.task,
                charter,
                view,
                pending.structure,
                signed=pending.signed,
                subject=subject,
            )
        runner = confirm.merge(baseline, pending.causal, extra)
        vault = scmle.core.context.require(
            context.custody_vault, "custody vault"
        )
        run = governed.finalise(
            pending.explored,
            source=vault.sets.get("C"),
            confirmation_rule=scmle.assurance.ledger.budget.Correction(
                governing(
                    "confirmation_rule",
                    context.spec.confirmation_rule,
                    None if rules is None else rules.confirmation_rule,
                    "by",
                )
            ),
            policy=governing(
                "failure_policy",
                context.spec.confirmation_failure_policy,
                None if rules is None else rules.failure_policy,
                "unconfirmed",
            ),
            claim_runner=runner,
            # The record of spent confirmation sets lives beside the run
            # directories, so a second run of the same charter on the same
            # data is refused before it reopens C (C8 charter_rerun).
            registry=None
            if context.store is None
            else context.store.directory.parent,
        )
        self.publish(context, run, runner)
        self.audit(context, run, pending.found)
        context.emit(
            "ledger.completed",
            run.confirmation.reason,
            (
                "hypothesis_ledger.jsonl",
                "negative_control.json",
                "confirmed_claims.json",
            ),
            {
                **run.summary(),
                "rules": "chartered" if rules else "not chartered",
            },
        )
        self.assess_robustness(context, pending.stage)
        self.ground(context, charter, view)
        self.identify(context, charter, view, run, pending.graph, pending.found)


def governing(
    name: str, requested: str, chartered: str | None, default: str
) -> str:
    """Returns the rule a charter run uses: the charter's when it fixes one.

    A run specification that names a different, non-default value than the
    charter does is a conflict and is refused; the charter is what was signed.

    Args:
        name: Field name, for the message.
        requested: Value in the run specification.
        chartered: Value in the charter's rules, or None without rules.
        default: The specification's default.

    Returns:
        The charter's value when it fixes one, otherwise `requested`.

    Raises:
        ConfigError: If the specification contradicts the charter.
    """
    if chartered is None:
        return requested
    if requested not in (chartered, default):
        raise errors.ConfigError(
            f"the run specification sets {name}={requested!r} but the "
            f"charter fixes {chartered!r}; change the specification or "
            "re-sign the charter"
        )
    return chartered
