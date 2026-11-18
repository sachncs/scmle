"""The charter-governed exploration and confirmation of one run.

This is the single entry point the pipeline calls. It takes exploration data
and, separately, the sealed confirmation set as an argument:

1. open the ledger (a new file or the run's existing one);
2. explore: one registered hypothesis per usable feature, tested under the
   online rule on `X`;
3. negative control: re-run the whole exploration `B` times with the target
   permuted within groups or time blocks, record the discoveries, and gate on
   the 95th percentile;
4. optionally, a second stage: a rule computed from `X` alone withdraws some
   survivors from the family (see "Second stage" in `confirm`);
5. confirm: freeze the survivors and test the family once on `C`.

The module imports nothing from custody; the caller hands `C` over as
`source`, so the code that selects survivors cannot reach it.
"""

from collections.abc import Callable
from collections.abc import Sequence
import dataclasses
from fractions import Fraction
import json
from pathlib import Path
from typing import Any

import numpy
import pandas

from scmle.assurance.ledger import budget
from scmle.assurance.ledger import confirm
from scmle.assurance.ledger import controls
from scmle.assurance.ledger import explore
from scmle.assurance.ledger import online_fdr
from scmle.assurance.ledger import usage
import scmle.assurance.ledger.budget
import scmle.assurance.ledger.ledger
import scmle.assurance.ledger.online_fdr
from scmle.contracts import qualification

CONTROL_FILE = "negative_control.json"
DEFAULT_BLOCKS = 20


@dataclasses.dataclass(frozen=True, slots=True)
class Governance:
    """What a governed exploration and confirmation produced.

    Attributes:
        ledger: The ledger, sealed by the confirmation.
        plan: The exploration plan.
        control: The negative control.
        confirmation: The confirmation outcome.
        scope: `strata` or `global`, how the control permuted the target.
    """

    ledger: scmle.assurance.ledger.ledger.Ledger
    plan: explore.Plan
    control: controls.Control
    confirmation: confirm.Confirmation
    scope: str

    def summary(self) -> dict[str, Any]:
        """Collects counts and hashes for the run event.

        Returns:
            A JSON-ready mapping with the online rule name, the numbers of
            hypotheses, tested hypotheses, hypotheses rejected on `X`,
            skipped features and confirmed claims, the confirmation outcome,
            the negative control figures (exceeds, observed, 95th
            percentile, scope), the ledger head, the survivors hash and the
            number of survivors withdrawn by the second stage (None when no
            withdrawal was accounted).
        """
        results = self.ledger.results.values()
        return {
            "rule": self.ledger.header["rule"]["rule"],
            "hypotheses": len(self.ledger.hypotheses),
            "tested": sum(1 for r in results if r.check is not None),
            "rejected_on_x": sum(1 for r in results if r.rejected),
            "skipped_features": len(self.plan.skipped),
            "confirmed": len(self.confirmation.confirmed),
            "outcome": self.confirmation.outcome,
            "control_exceeds": self.control.exceeds,
            "control_observed": self.control.observed,
            "control_p95": self.control.p95,
            "control_scope": self.scope,
            "ledger_head": self.ledger.head,
            "survivors_sha256": self.confirmation.survivors_sha256,
            "withdrawn": (
                None
                if self.confirmation.stage_two is None
                else len(self.confirmation.stage_two["withdrawn"])
            ),
        }


def strata_labels(
    features: pandas.DataFrame,
    group_column: str | None,
    time_column: str | None,
    blocks: int = DEFAULT_BLOCKS,
) -> numpy.ndarray | None:
    """Builds the labels within which the target may be permuted.

    Groups win over time: rows of one entity stay together. With only a time
    column the rows are cut, in time order, into contiguous blocks of
    ``max(2, n // blocks)`` rows, so about `blocks` blocks (a few more when
    the division leaves a remainder). A named column that is not among the
    features is ignored.
    Blocks must be short against the timescale of the drift: on random-walk
    data 10 blocks let the control be exceeded in about 7.7% of null worlds
    and 20 blocks in about 3.7% (see `benchmarks.qualification.ledger_checks`).
    With neither, the rows are taken as exchangeable and None is returned
    (a global permutation).

    Args:
        features: Exploration features, holding the structure columns.
        group_column: Entity column, if any.
        time_column: Time column, if any.
        blocks: Number of contiguous time blocks.

    Returns:
        One label per row (the group as text, or an integer block number),
        or None for exchangeable rows.
    """
    if group_column and group_column in features.columns:
        return features[group_column].astype(str).to_numpy()
    if time_column and time_column in features.columns:
        order = pandas.to_numeric(features[time_column], errors="coerce")
        size = max(2, len(features) // max(blocks, 1))
        return controls.block_labels(order.to_numpy(dtype=float), size)
    return None


@dataclasses.dataclass(frozen=True, slots=True)
class Explored:
    """A governed run after exploration, control and second stage.

    Everything here was computed from `X` alone. `C` is still sealed;
    `finalise` opens it.

    Attributes:
        ledger: The ledger, not yet sealed by a confirmation.
        plan: The exploration plan.
        control: The negative control.
        withdrawal: The second-stage withdrawal, or None.
        scope: `strata` or `global`, how the control permuted the target.
        charter: The governing charter.
        classification: Whether the target holds classes.
        classes: The target's classes in coding order, or None.
        run_dir: Directory for the artifacts, or None.
        data: Fingerprint of the exploration rows and target
            (`usage.fingerprint`), or None when nothing is recorded.
    """

    ledger: scmle.assurance.ledger.ledger.Ledger
    plan: explore.Plan
    control: controls.Control
    withdrawal: confirm.Withdrawal | None
    scope: str
    charter: Any
    classification: bool
    classes: tuple[Any, ...] | None
    run_dir: str | Path | None
    data: str | None = None


def run_governed(
    *,
    charter: Any,
    features: pandas.DataFrame,
    target: pandas.Series,
    classification: bool,
    source: confirm.Source | None,
    run_dir: str | Path | None,
    exclude: tuple[str, ...] = (),
    strata: numpy.ndarray | None = None,
    rule: str = scmle.assurance.ledger.online_fdr.ELOND,
    confirmation_rule: budget.Correction = (
        scmle.assurance.ledger.budget.Correction.BY
    ),
    policy: str = confirm.UNCONFIRMED,
    diagnostics: Sequence[tuple[str, qualification.Check]] = (),
    second_stage: Callable[
        [Sequence[scmle.assurance.ledger.ledger.Hypothesis]], confirm.Withdrawal
    ]
    | None = None,
    claim_runner: confirm.Docket | None = None,
) -> Governance:
    """Explores `X` under the ledger, runs the control and confirms on `C`.

    Args:
        charter: The governing `AnalysisCharter`.
        features: Exploration features (never confirmation rows).
        target: Exploration target.
        classification: Whether the target holds classes.
        source: The sealed confirmation set, or None if none was carved.
        run_dir: Directory for the artifacts, or None to keep them in memory.
        exclude: Structure columns (group, time) that are not candidates.
        strata: Permutation labels for the control (see `strata_labels`).
        rule: Online rule for exploration.
        confirmation_rule: Multiplicity rule for the confirmation.
        policy: Failure policy of the confirmation.
        diagnostics: (source, check) pairs from detectors run on `X`; they
            are logged once, before exploration, and spend no wealth (they
            are skipped when the ledger already holds diagnostics).
        second_stage: Pre-registered withdrawal rule. It receives the
            survivors (rejected hypotheses, in order) after the negative
            control and must compute its answer from `X` alone; the closure
            the caller builds holds no `C`. Its diagnostics are logged
            without wealth before the survivors are frozen, and the
            withdrawn survivors are not tested on `C`.
        claim_runner: Claims registered in the ledger and evaluated on the
            same opening of `C` (see `confirm.confirm`).

    Returns:
        The governed run; artifacts are written when `run_dir` is given.
        Errors from the steps it calls (a `ConfigError` for an unavailable
        policy, a `LedgerError`, a `LeakageError` for a second confirmation)
        propagate.
    """
    explored = scout(
        charter=charter,
        features=features,
        target=target,
        classification=classification,
        run_dir=run_dir,
        exclude=exclude,
        strata=strata,
        rule=rule,
        policy=policy,
        diagnostics=diagnostics,
        second_stage=second_stage,
    )
    return finalise(
        explored,
        source=source,
        confirmation_rule=confirmation_rule,
        policy=policy,
        claim_runner=claim_runner,
    )


def survey(
    charter: Any,
    run_dir: str | Path | None,
    rule: str,
    diagnostics: Sequence[tuple[str, qualification.Check]],
    plan: explore.Plan,
    features: pandas.DataFrame,
    codes: pandas.Series,
    strata: numpy.ndarray | None,
) -> tuple[scmle.assurance.ledger.ledger.Ledger, controls.Control]:
    """Opens the ledger, then runs the exploration and its negative control.

    Args:
        charter: The governing `AnalysisCharter`.
        run_dir: Directory of the ledger, or None to keep it in memory.
        rule: Online rule for exploration.
        diagnostics: (source, check) pairs from detectors run on `X`; they
            are logged only when the ledger holds no diagnostics yet.
        plan: The exploration plan.
        features: Exploration features.
        codes: The target, coded as numbers.
        strata: Permutation labels for the permuted-target control.

    Returns:
        The ledger, holding the diagnostics, the results and the control,
        and the negative control.

    Raises:
        LedgerError: If the ledger holds no online rule configuration.
    """
    if run_dir is None:
        ledger = scmle.assurance.ledger.ledger.Ledger.create(
            None,
            charter_hash=charter.charter_hash,
            config=online_fdr.Settings(
                rule, charter.budget.exploration.initial
            ),
            alpha_confirm=charter.budget.confirmation.alpha_confirm,
            seed=charter.seed,
        )
    else:
        ledger = scmle.assurance.ledger.ledger.Ledger.open(
            run_dir, charter, rule=rule
        )
    if diagnostics and not ledger.diagnostics:
        for source_name, found in diagnostics:
            ledger.record_diagnostic(found, source_name)
    config = ledger.config
    if config is None:  # pragma: no cover - open() always sets it
        raise scmle.assurance.ledger.ledger.LedgerError("ledger has no rule")
    results = explore.run_exploration(ledger, features, codes, plan)
    observed = sum(1 for r in results if r.rejected)

    def discoveries(frame: pandas.DataFrame, shuffled: pandas.Series) -> int:
        return explore.tally(config, plan, frame, shuffled)

    control = controls.placebo(
        discoveries,
        features,
        codes,
        observed=observed,
        b=charter.budget.permutations,
        seed=charter.seed,
        strata=strata,
    )
    ledger.negatives(control.to_dict())
    return ledger, control


def withdraw(
    ledger: scmle.assurance.ledger.ledger.Ledger,
    second_stage: Callable[
        [Sequence[scmle.assurance.ledger.ledger.Hypothesis]], confirm.Withdrawal
    ]
    | None,
) -> confirm.Withdrawal | None:
    """Applies the pre-registered second stage to the survivors.

    Args:
        ledger: The ledger after exploration; the withdrawal's diagnostics
            are recorded in it.
        second_stage: Pre-registered withdrawal rule, or None.

    Returns:
        The withdrawal, or None when there is no second stage.
    """
    if second_stage is None:
        return None
    withdrawal = second_stage(ledger.survivors())
    for source_name, found in withdrawal.diagnostics:
        ledger.record_diagnostic(found, source_name)
    return withdrawal


def scout(
    *,
    charter: Any,
    features: pandas.DataFrame,
    target: pandas.Series,
    classification: bool,
    run_dir: str | Path | None,
    exclude: tuple[str, ...] = (),
    strata: numpy.ndarray | None = None,
    rule: str = scmle.assurance.ledger.online_fdr.ELOND,
    policy: str = confirm.UNCONFIRMED,
    diagnostics: Sequence[tuple[str, qualification.Check]] = (),
    second_stage: Callable[
        [Sequence[scmle.assurance.ledger.ledger.Hypothesis]], confirm.Withdrawal
    ]
    | None = None,
) -> Explored:
    """Runs the part of `run_governed` that precedes the opening of `C`.

    The arguments are those of `run_governed` without the confirmation ones.
    A caller that needs something computed after exploration (a selected
    model) before the registered claims are fixed calls this first and
    `finalise` later; `C` is not touched in between.

    Args:
        charter: The governing `AnalysisCharter`.
        features: Exploration features (never confirmation rows).
        target: Exploration target.
        classification: Whether the target holds classes.
        run_dir: Directory for the artifacts, or None to keep them in memory.
        exclude: Structure columns (group, time) that are not candidates.
        strata: Permutation labels for the control.
        rule: Online rule for exploration.
        policy: Failure policy of the confirmation; checked before anything
            else is done.
        diagnostics: (source, check) pairs from detectors run on `X`; they
            are logged once, before exploration, and spend no wealth (they
            are skipped when the ledger already holds diagnostics).
        second_stage: Pre-registered withdrawal rule that receives the
            survivors and computes its answer from `X` alone, or None.

    Returns:
        The exploration state: the ledger with its results and negative
        control recorded, the exploration plan, the control, the second-stage
        withdrawal (None if there was none) and what `finalise`
        needs to open `C`.

    Raises:
        LedgerError: If the ledger holds no online rule configuration. An
            unavailable `policy` is refused by `confirm.enforce` with a
            `ConfigError`.
    """
    confirm.enforce(policy)
    kind = explore.target_kind(target, classification)
    codes = explore.vectorize(target, classification)
    classes = (
        tuple(sorted(target.dropna().unique(), key=str))
        if classification
        else None
    )
    plan = explore.hypothesize(features, kind, exclude)
    ledger, control = survey(
        charter, run_dir, rule, diagnostics, plan, features, codes, strata
    )
    return Explored(
        ledger=ledger,
        plan=plan,
        control=control,
        withdrawal=withdraw(ledger, second_stage),
        scope="global" if strata is None else "strata",
        charter=charter,
        classification=classification,
        classes=classes,
        run_dir=run_dir,
        data=None if run_dir is None else usage.fingerprint(features, target),
    )


def finalise(
    explored: Explored,
    *,
    source: confirm.Source | None,
    confirmation_rule: budget.Correction = (
        scmle.assurance.ledger.budget.Correction.BY
    ),
    policy: str = confirm.UNCONFIRMED,
    claim_runner: confirm.Docket | None = None,
    registry: str | Path | None = None,
) -> Governance:
    """Freezes the survivors and tests once on `C` (see `run_governed`).

    Args:
        explored: The state returned by `scout`.
        source: The sealed confirmation set, or None if none was carved.
        confirmation_rule: Multiplicity rule for the confirmation.
        policy: Failure policy of the confirmation.
        claim_runner: Claims registered in the ledger and evaluated on the
            same opening of `C`.
        registry: Folder that keeps, across runs, the record of spent
            confirmation sets (`usage`); the pipeline passes the folder
            holding its run directories. None records nothing.

    Returns:
        The governed run. When the exploration has a run directory, the
        claims file and the negative control record are written there after
        the confirmation returns; a second confirmation raises
        `LeakageError` and writes nothing, and so does a confirmation of
        a set another run recorded in `registry` already spent under the
        same charter on the same data (`usage`). A withdrawal that is not
        accounted is not passed to the confirmation.
    """
    charter = explored.charter
    classification = explored.classification
    classes = explored.classes
    withdrawal = explored.withdrawal
    alpha = Fraction(charter.budget.confirmation.alpha_confirm)
    confirmation = confirm.confirm(
        explored.ledger,
        source,
        alpha_confirm=alpha,
        rule=confirmation_rule,
        policy=policy,
        encode=lambda series: explore.vectorize(
            series, classification, classes
        ).to_numpy(dtype=float),
        withdrawal=withdrawal if withdrawal and withdrawal.accounted else None,
        claim_runner=claim_runner,
        register=spend(explored, source, registry),
    )
    if explored.run_dir is not None:
        folder = Path(explored.run_dir)
        confirmation.write(folder / confirm.CLAIMS_FILE)
        (folder / CONTROL_FILE).write_text(
            json.dumps(explored.control.to_dict(), indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
    return Governance(
        ledger=explored.ledger,
        plan=explored.plan,
        control=explored.control,
        confirmation=confirmation,
        scope=explored.scope,
    )


def spend(
    explored: Explored,
    source: confirm.Source | None,
    registry: str | Path | None,
) -> usage.Register | None:
    """The cross-run record of the confirmation set this run would open.

    Args:
        explored: The state returned by `scout`.
        source: The sealed confirmation set, or None.
        registry: Folder that keeps the records, or None.

    Returns:
        A register in `registry`, keyed by the charter hash, the exploration
        data and the confirmation row ids, naming the run by its directory;
        None without a registry, a run directory or `C`.
    """
    if (
        registry is None
        or explored.run_dir is None
        or explored.data is None
        or source is None
    ):
        return None
    folder = Path(explored.run_dir).resolve()
    charter = explored.charter.charter_hash
    return usage.Register(
        registry,
        usage.key(charter, explored.data, getattr(source, "ids", ())),
        folder.name,
        {"charter_hash": charter, "data": explored.data},
    )
