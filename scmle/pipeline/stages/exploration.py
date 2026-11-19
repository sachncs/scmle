"""The claim ledger of a charter run."""

import dataclasses
from typing import Any

from scmle.assurance import coverage_confirm
from scmle.assurance.admissibility import qualify
from scmle.assurance.charter import contracts
from scmle.assurance.charter import sealing
import scmle.assurance.charter.rules
import scmle.assurance.custody.pipeline
import scmle.assurance.custody.sealed
import scmle.assurance.framing.claim
from scmle.assurance.ledger import confirm
from scmle.assurance.ledger import governed
import scmle.assurance.ledger.budget
from scmle.assurance.robustness import prefilter
from scmle.causal import claim
from scmle.contracts import errors
from scmle.contracts import tasks
from scmle.pipeline import engine
from scmle.pipeline import products
from scmle.pipeline.stages import admission


def coverage_on_confirmation(
    charter: contracts.AnalysisCharter, task: tasks.Task
) -> bool:
    """Returns whether the charter asks for the coverage claim on `C`.

    Args:
        charter: The analysis charter.
        task: The resolved task.

    Returns:
        True if the charter's uncertainty block asks for it and the task
        has a primary coverage choice.
    """
    block = charter.uncertainty
    return (
        block is not None
        and block.confirmatory
        and coverage_confirm.principal(task.value) is not None
    )


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


@dataclasses.dataclass(frozen=True)
class Custody(products.Derived):
    """The confirmation set of a charter run, carved out and sealed.

    Derived: the vault holds rows that are never written beside the stage
    record. Whoever holds the custody holds the vault: `engine.Board.close`
    closes it once the custody is applied, and `Impounding.sequester`
    closes it on every failure before that.

    Attributes:
        outcome: The custody outcome: the split restricted to `X`, the
            exploration view, the manifest and the feasibility verdict.
        vault: The vault holding the sealed confirmation set.
        column: The entity column whose rows are not independent, when the
            charter predicts known entities' future rows, else None.
    """

    outcome: scmle.assurance.custody.pipeline.Outcome
    vault: scmle.assurance.custody.sealed.Vault
    column: str | None = None

    def summary(self) -> object:
        """What the custody was derived from.

        Returns:
            The manifest's hash and scope, the set sizes and the entity
            column.
        """
        manifest = self.outcome.manifest
        return {
            "manifest": manifest.sha256,
            "scope": manifest.scope,
            "exploration": len(manifest.sets.exploration),
            "confirmation": len(manifest.sets.confirmation),
            "embargo": len(manifest.sets.embargo),
            "column": self.column,
        }

    def close(self) -> None:
        """Closes the vault holding the confirmation set."""
        self.vault.close()


@dataclasses.dataclass(frozen=True)
class Claims(products.Derived):
    """What the claim ledger left for the later stages.

    Derived: it holds the exploration rows, which are never written beside
    the stage record.

    Attributes:
        pending: The exploration whose opening of `C` waits until the model
            (and its calibration) is frozen, or None when the confirmation
            already ran or the ledger did not run.
    """

    pending: Escrow | None = None

    def summary(self) -> object:
        """What the claims were derived from.

        Returns:
            Whether a confirmation waits, and under which charter.
        """
        if self.pending is None:
            return {"pending": False}
        return {
            "pending": True,
            "charter": self.pending.charter.charter_hash,
            "coverage": self.pending.holds_for_coverage,
        }


class Exploration(admission.Admission):
    """The claim ledger of a charter run, from exploration to conclusion.

    Part of `ProfileStage`, which inherits these methods unchanged;
    abstract, never run on its own.
    """

    def ledger(self, board: engine.Board, task: tasks.Task) -> Claims:
        """Explores `X` under the ledger and confirms on `C`, now or later.

        Runs only for a charter run. Exploration reads the exploration view; the
        sealed confirmation set is handed to the confirmation stage as an
        argument and is opened at most once, after the survivors are frozen in
        the ledger. Tasks other than binary, multiclass and regression are
        recorded as skipped: their findings stay unconfirmed. When the baseline
        claim is about the selected model, or the coverage claim needs the
        calibrated model, the confirmation is deferred until the model is
        frozen (the pending ledger is the product).

        Args:
            board: The run's board, holding the custody outcome and vault.
            task: The resolved task.

        Returns:
            The claims, pending only when the confirmation waits.
        """
        custody = board.find(Custody)
        if custody is None:
            return Claims()
        outcome = custody.outcome
        if task not in tasks.STANDARD_TASKS:
            board.journal.emit(
                "ledger.skipped",
                f"the claim ledger does not cover {task.value} tasks; "
                "findings stay unconfirmed",
            )
            return Claims()
        spec = board.services.spec
        verified = sealing.read_verified(str(spec.charter_path))
        charter = verified.charter
        data = spec.data
        view = outcome.view
        structure = tuple(
            c
            for c in (data.group_column, data.time_column, data.baseline_column)
            if c
        )
        block = charter.estimand.causal
        graph = claim.unseal(block.dag) if block is not None else None
        found = self.qualify_exploration(
            board, task, charter, view, structure, graph
        )
        rules = charter.rules
        stage = prefilter.Prefilter(
            lambda columns: self.stress(board, task, charter, view, columns),
            requested=bool(rules and rules.stability),
            rows=len(view.features),
            seed=charter.seed,
        )
        # Declarations are validated before the expensive exploration.
        baseline = self.baseline_runner(
            board, task, charter, view, structure, signed=verified.signed
        )
        causal = self.hypothesise(board, charter, view, graph, found)
        explored = self.explore(board, task, charter, (structure, found, stage))
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
            holds_for_coverage=coverage_on_confirmation(charter, task),
        )
        if self.defer(board, pending):
            return Claims(pending)
        self.conclude(board, pending)
        return Claims()

    def explore(
        self,
        board: engine.Board,
        task: tasks.Task,
        charter: Any,
        setting: tuple[tuple[str, ...], Any, prefilter.Prefilter],
    ) -> Any:
        """Explores the exploration view under the charter's claim ledger.

        Args:
            board: The run's board, holding the custody outcome.
            task: The resolved task.
            charter: The verified charter.
            setting: The structure columns, the qualification findings (its
                quarantined columns are kept out too) and the second-stage
                stability prefilter.

        Returns:
            The exploration, its ledger and its survivors.
        """
        structure, found, stage = setting
        spec = board.services.spec
        data = spec.data
        view = board.get(Custody).outcome.view
        rules = charter.rules
        return governed.scout(
            charter=charter,
            features=view.features,
            target=view.target,
            classification=task.is_classification,
            run_dir=board.services.store.directory
            if board.services.store is not None
            else None,
            exclude=structure + tuple(found.quarantined),
            strata=governed.strata_labels(
                view.features, data.group_column, data.time_column
            ),
            rule=governing(
                "ledger_rule",
                spec.ledger_rule,
                None if rules is None else rules.ledger,
                "e-lond",
            ),
            policy=governing(
                "failure_policy",
                spec.confirmation_failure_policy,
                None if rules is None else rules.failure,
                "unconfirmed",
            ),
            diagnostics=found.checks,
            second_stage=stage,
        )

    def defer(self, board: engine.Board, pending: "Escrow") -> bool:
        """Keeps the confirmation sealed until the model is frozen, if needed.

        Args:
            board: The run's board, supplying the data spec and journal.
            pending: The explored ledger awaiting confirmation.

        Returns:
            Whether the confirmation waits: the baseline claim is about the
            selected model, or the coverage claim needs the calibrated model.
        """
        data = board.services.spec.data
        selected = bool(
            data.baseline_column and data.baseline_subject == "selected_model"
        )
        if selected or pending.holds_for_coverage:
            board.journal.emit(
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
            return True
        return False

    def conclude(
        self,
        board: engine.Board,
        pending: "Escrow",
        subject: scmle.assurance.framing.claim.Rule | None = None,
        extra: confirm.Docket | None = None,
    ) -> None:
        """Registers the claims, opens `C` once and writes the reports.

        Under the charter's `thresholdout` failure policy the confirmation
        also runs the reuse session `governed.replication` builds from `X`.

        Args:
            board: The run's board, holding the custody vault whose set `C`
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
                board,
                pending.task,
                charter,
                view,
                pending.structure,
                signed=pending.signed,
                subject=subject,
            )
        runner = confirm.merge(baseline, pending.causal, extra)
        vault = board.get(Custody).vault
        policy, correction = govern(board, rules)
        # A charter that pre-registers `thresholdout` gets its reuse session:
        # the survivors' agreement asked again on the same opening of C.
        reuse = (
            governed.replication(pending.explored, view.features, view.target)
            if policy == confirm.THRESHOLDOUT
            else None
        )
        run = governed.finalise(
            pending.explored,
            source=vault.sets.get("C"),
            confirmation_rule=correction,
            policy=policy,
            claim_runner=runner,
            reuse=reuse,
            # The record of spent confirmation sets lives beside the run
            # directories, so a second run of the same charter on the same
            # data is refused before it reopens C (C8 charter_rerun).
            registry=None
            if board.services.store is None
            else board.services.store.directory.parent,
        )
        self.publish(board, run, runner)
        board.journal.emit(
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
        self.assess_robustness(board, pending.stage)
        self.ground(board, charter, view)
        self.identify(board, charter, view, run, pending.graph, pending.found)


def govern(
    board: engine.Board,
    rules: scmle.assurance.charter.rules.Procedure | None,
) -> tuple[str, scmle.assurance.ledger.budget.Correction]:
    """Reads the failure policy and the confirmation rule of the run.

    Args:
        board: The run's board, whose spec may set either.
        rules: The charter's pre-registered rules, or None.

    Returns:
        The failure policy and the multiplicity correction, each from
        `governing` (the charter's choice wins).
    """
    policy = governing(
        "failure_policy",
        board.services.spec.confirmation_failure_policy,
        None if rules is None else rules.failure,
        "unconfirmed",
    )
    correction = scmle.assurance.ledger.budget.Correction(
        governing(
            "confirmation_rule",
            board.services.spec.confirmation_rule,
            None if rules is None else rules.confirmation,
            "by",
        )
    )
    return policy, correction


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
