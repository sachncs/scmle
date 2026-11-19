"""Leakage screen with multiplicity control, for runs with or without a charter.

This replaces the fixed-cutoff profiler heuristics (an association of 0.98 or
a missingness score of 0.9, with no error control) by a test whose false-alarm
behaviour is stated and measured. It reads the training rows it is given (a
run without a charter has no custody, so those are the development rows; a
charter run passes the exploration set `X`) and never the final test set.

Procedure, for every candidate column `j` (the family has `m` members):

1. Single-feature skill. A cross-fitted (out-of-fold) ridge smoother of the
   outcome on one column (a piecewise-linear basis on `KNOTS` quantile knots
   for a numeric column, a level per category, and an indicator for missing
   values) gives a held-out ROC AUC (macro one-vs-rest for more than two
   classes) or R^2. The p-value is `(b + 1) / (B + 1)`, `b` counting
   permutations of the outcome whose skill is at least the observed one
   (Phipson and Smyth, 2010, Stat. Appl. Genet. Mol. Biol. 9(1)). The outcome
   is permuted within groups or within consecutive time blocks when they are
   given. The folds do not depend on the outcome, so the observed statistic
   and every permuted one are produced by the same procedure and the p-value
   is exact under exchangeability within the permutation strata.
2. Missingness, for a column with gaps, by two tests. (a) The missing-value
   indicator against the outcome (quartile bins for a continuous outcome)
   with Cramer's V. Without groups or time blocks the p-value is Fisher's
   exact test (2x2 table with an expected count below 5) or Pearson's
   chi-square; with them it is the permutation scheme above, because a
   chi-square p-value ignores dependence between rows of one group.
   (b) Record purity: the share of the commonest outcome class among the
   rows where the value is recorded, against the same permutation null. This
   sees a value captured only for some outcomes even when most rows are blank
   (a Titanic body number is recorded for 9 percent of passengers, all of
   whom died), which V, measured over all rows, cannot.
2b. Echo (`forecast`): a column read as the target itself, with no fit:
   the share of rows where it spells the class, or `1 - SSE / SST` for a
   number. Formed only when the two marginals allow it to pass the ceiling,
   so a legitimate predictor on its own scale raises no hypothesis; it finds
   a field recorded after the outcome (the label with a tenth garbled, the
   number plus noise) whose skill stays under the skill ceiling.
2c. Records (`records`), with analytic p-values. Target mean: a numeric
   column on the target's scale, grouped by its own levels and by its runs
   along the most telling other column's order, tested by the lower tail of
   `F(G, nu)` for being a more precise group mean of the target than that
   mean's sampling noise allows a column that never read the outcomes.
   Duplication: every row (not the subsample) is counted, and the records
   seen twice are tested against the most a sample from any population
   holds given those seen once and three times (`E[f2]^2 <= 1.5 E[f1]
   E[f3]`), so a table's natural duplicates set its own allowance. Both
   ceilings are that honest limit (`NATURAL`).
3. Multiplicity. All `m` p-values (skill, echo, missingness and record
   tests) form one family adjusted by Benjamini and Yekutieli (2001, Ann.
   Statist. 29:1165) at level `alpha`, exactly
   (`scmle.assurance.ledger.multiplicity`).
   The number of permutations is raised until the smallest reachable p-value
   `1 / (B + 1)` can pass the rank-1 threshold `alpha / (m * H_m)`; if that
   needs more than `MAX_PERMUTATIONS` the screen says so and flags nothing.
   A column whose p-value can no longer fall below the largest threshold
   `alpha / H_m` stops early; the step-up decisions are identical to those of
   the full run, because such a p-value can never be rejected and sorts last.
   A hypothesis whose observed statistic is not above its ceiling cannot be
   a suspect (step 4), so it draws no permutations and enters the step with
   p = 1 (`sift`): a larger p-value is still valid, so the control below is
   kept, and a wide table without a strong column needs no permutation work.
4. Suspect. A column is a *suspect* only when its test is rejected by step 3
   AND the one-sided lower confidence bound of its effect size is above the
   pre-registered ceiling (`skill_ceiling` or `association_ceiling`). The
   bound is a bootstrap of the out-of-fold predictions (or of the rows for
   V), clustered by group, and an exact Clopper-Pearson bound for the purity
   share. Significance alone is not leakage: strong legitimate predictors are
   significant.

What is guaranteed
------------------
Conditional on the model and fold choices above, and on exchangeability of
the outcome within the permutation strata, the Benjamini-Yekutieli step
controls the false discovery rate of the *significant* list at `alpha` times
the share of true nulls, under arbitrary dependence between columns. Under
the global null (no column is associated with the outcome) this is the
probability of at least one false alarm, at most `alpha`. The suspect list is
a subset of the significant list, so its false discoveries cannot exceed those
of the significant list, which is why a suspect may block a run without a
certificate (the certificates, `screen_certificates.json` from
`benchmarks.qualification.screen_certify`, are evidence about recall and
false flags on real tables). Without gates (`screen` called directly) every
suspect is reported as a warning with its p-values.

What is not guaranteed
----------------------
The screen cannot distinguish a legitimate strong predictor from a leak
without provenance: a suspect is a suspicion, never proof. Bootstrap bounds on
cross-fitted scores are approximate (folds share training rows), so no exact
coverage is claimed. Rows are subsampled to `MAX_ROWS`, which costs power for
effects below the ceiling, not size. The Fisher and chi-square p-values used
without groups or time blocks assume independent rows. Skill is measured one
column at a time with a smooth basis (a date as days, free text by its
commonest words), and jointly for each pair of two-level columns while
there are no more pairs than columns, so a leak spread over more columns,
or over two columns with more levels, is not seen. Wide tables pay in
power: the more columns, the more permutations a rejection needs. A run
without a charter has no custody, so the training rows are all the screen
has; it says nothing about rows it never saw.

In a run (`screen_table`) every suspect may block: a suspect is significant
after the Benjamini-Yekutieli step over permutation p-values, whose
false-alarm control holds for any rows that can be shuffled within their
groups or time blocks, and its effect bound exceeds the pre-registered
ceiling. What a clean result means is measured on the run's own rows: the
probe screens a copy with one planted column per leak type (`plants`), with
the same settings, structure and budget, and the report records which plants
were found, so a run states the kinds of leak the screen would have missed
there. The published certificates are evidence about the screen, reported in
the notes; they decide nothing.
"""

from collections.abc import Hashable
from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses

import numpy
import pandas

from scmle.assurance.admissibility import leakage
from scmle.assurance.admissibility import plants
from scmle.assurance.admissibility.screen.association import chisquare
from scmle.assurance.admissibility.screen.association import cramer
from scmle.assurance.admissibility.screen.association import dominance
from scmle.assurance.admissibility.screen.association import purity_statistic
from scmle.assurance.admissibility.screen.association import resample
from scmle.assurance.admissibility.screen.fitting import aucs
from scmle.assurance.admissibility.screen.fitting import basis
from scmle.assurance.admissibility.screen.fitting import concordance
from scmle.assurance.admissibility.screen.fitting import crossfit
from scmle.assurance.admissibility.screen.fitting import encode
from scmle.assurance.admissibility.screen.fitting import enough
from scmle.assurance.admissibility.screen.fitting import margin
from scmle.assurance.admissibility.screen.fitting import oof_fit
from scmle.assurance.admissibility.screen.fitting import pair
from scmle.assurance.admissibility.screen.fitting import pairs
from scmle.assurance.admissibility.screen.fitting import permute
from scmle.assurance.admissibility.screen.fitting import predict
from scmle.assurance.admissibility.screen.fitting import Problem
from scmle.assurance.admissibility.screen.fitting import replicate
from scmle.assurance.admissibility.screen.fitting import skill_bound
from scmle.assurance.admissibility.screen.fitting import skill_statistic
from scmle.assurance.admissibility.screen.fitting import stratify
from scmle.assurance.admissibility.screen.fitting import words
from scmle.assurance.admissibility.screen.hypotheses import caps
from scmle.assurance.admissibility.screen.hypotheses import floor
from scmle.assurance.admissibility.screen.hypotheses import forecast
from scmle.assurance.admissibility.screen.hypotheses import gauge
from scmle.assurance.admissibility.screen.hypotheses import hedge
from scmle.assurance.admissibility.screen.hypotheses import hypothesize
from scmle.assurance.admissibility.screen.hypotheses import Pending
from scmle.assurance.admissibility.screen.hypotheses import propose
from scmle.assurance.admissibility.screen.hypotheses import reflect
from scmle.assurance.admissibility.screen.hypotheses import sift
from scmle.assurance.admissibility.screen.hypotheses import spell
from scmle.assurance.admissibility.screen.hypotheses import tested
from scmle.assurance.admissibility.screen.permutation import draw_permutations
from scmle.assurance.admissibility.screen.permutation import resamples
from scmle.assurance.admissibility.screen.planning import admit
from scmle.assurance.admissibility.screen.planning import arrange
from scmle.assurance.admissibility.screen.planning import bound
from scmle.assurance.admissibility.screen.planning import bundled
from scmle.assurance.admissibility.screen.planning import confine
from scmle.assurance.admissibility.screen.planning import decline
from scmle.assurance.admissibility.screen.planning import Demand
from scmle.assurance.admissibility.screen.planning import disclose
from scmle.assurance.admissibility.screen.planning import foresee
from scmle.assurance.admissibility.screen.planning import forgo
from scmle.assurance.admissibility.screen.planning import load_store
from scmle.assurance.admissibility.screen.planning import measure
from scmle.assurance.admissibility.screen.planning import ready
from scmle.assurance.admissibility.screen.planning import reckon
from scmle.assurance.admissibility.screen.planning import subsample
from scmle.assurance.admissibility.screen.records import dated
from scmle.assurance.admissibility.screen.records import declared
from scmle.assurance.admissibility.screen.records import duplication
from scmle.assurance.admissibility.screen.records import encodings
from scmle.assurance.admissibility.screen.report import abstain
from scmle.assurance.admissibility.screen.report import conclude
from scmle.assurance.admissibility.screen.report import exceed
from scmle.assurance.admissibility.screen.report import faint
from scmle.assurance.admissibility.screen.report import Finding
from scmle.assurance.admissibility.screen.report import halt
from scmle.assurance.admissibility.screen.report import Screen
from scmle.assurance.admissibility.screen.report import tally
from scmle.assurance.admissibility.screen.report import throttle
from scmle.assurance.admissibility.screen.settings import allowance
from scmle.assurance.admissibility.screen.settings import BATCH
from scmle.assurance.admissibility.screen.settings import BINARY_CLASSES
from scmle.assurance.admissibility.screen.settings import BOUND_SECONDS
from scmle.assurance.admissibility.screen.settings import COLUMN_SECONDS
from scmle.assurance.admissibility.screen.settings import cost
from scmle.assurance.admissibility.screen.settings import DATED
from scmle.assurance.admissibility.screen.settings import DATED_DEFECT
from scmle.assurance.admissibility.screen.settings import DATED_ID
from scmle.assurance.admissibility.screen.settings import DECLARED
from scmle.assurance.admissibility.screen.settings import DETECTOR_ID
from scmle.assurance.admissibility.screen.settings import DETECTORS
from scmle.assurance.admissibility.screen.settings import DRAW_SECONDS
from scmle.assurance.admissibility.screen.settings import DUPLICATION
from scmle.assurance.admissibility.screen.settings import ECHO
from scmle.assurance.admissibility.screen.settings import ENCODING
from scmle.assurance.admissibility.screen.settings import family
from scmle.assurance.admissibility.screen.settings import FIT_SECONDS
from scmle.assurance.admissibility.screen.settings import FITS_PER_SECOND
from scmle.assurance.admissibility.screen.settings import HELD
from scmle.assurance.admissibility.screen.settings import HYPOTHESES
from scmle.assurance.admissibility.screen.settings import KNOTS
from scmle.assurance.admissibility.screen.settings import LATE
from scmle.assurance.admissibility.screen.settings import MAX_CERTIFIED_CLASSES
from scmle.assurance.admissibility.screen.settings import MAX_PERMUTATIONS
from scmle.assurance.admissibility.screen.settings import MAX_ROWS
from scmle.assurance.admissibility.screen.settings import MECHANISM_DEFECTS
from scmle.assurance.admissibility.screen.settings import MIN_CLASSES
from scmle.assurance.admissibility.screen.settings import MISSING
from scmle.assurance.admissibility.screen.settings import NATURAL
from scmle.assurance.admissibility.screen.settings import PAIR_LEVELS
from scmle.assurance.admissibility.screen.settings import prepare
from scmle.assurance.admissibility.screen.settings import PURITY
from scmle.assurance.admissibility.screen.settings import READ_SECONDS
from scmle.assurance.admissibility.screen.settings import RECORD_SECONDS
from scmle.assurance.admissibility.screen.settings import RIDGE
from scmle.assurance.admissibility.screen.settings import ROWS
from scmle.assurance.admissibility.screen.settings import SCAN_SECONDS
from scmle.assurance.admissibility.screen.settings import SCREEN_SHARE
from scmle.assurance.admissibility.screen.settings import share
from scmle.assurance.admissibility.screen.settings import SKILL
from scmle.assurance.admissibility.screen.settings import Statistic
from scmle.assurance.admissibility.screen.settings import STORE_PATH
from scmle.assurance.admissibility.screen.settings import TIME_BLOCKS
from scmle.assurance.admissibility.screen.settings import TOLERANCE
from scmle.assurance.admissibility.screen.settings import WAY_SECONDS
from scmle.assurance.admissibility.screen.settings import WORD
from scmle.assurance.admissibility.screen.settings import WORDS
from scmle.assurance.detectors import certificates
from scmle.assurance.detectors import trace
from scmle.assurance.ledger import budget
from scmle.contracts import errors

__all__ = [
    "BATCH",
    "BINARY_CLASSES",
    "BOUND_SECONDS",
    "COLUMN_SECONDS",
    "DATED",
    "DATED_DEFECT",
    "DATED_ID",
    "DECLARED",
    "DETECTORS",
    "DETECTOR_ID",
    "DRAW_SECONDS",
    "DUPLICATION",
    "ECHO",
    "ENCODING",
    "FITS_PER_SECOND",
    "FIT_SECONDS",
    "HELD",
    "HYPOTHESES",
    "KNOTS",
    "LATE",
    "MAX_CERTIFIED_CLASSES",
    "MAX_PERMUTATIONS",
    "MAX_ROWS",
    "MECHANISM_DEFECTS",
    "MIN_CLASSES",
    "MISSING",
    "NATURAL",
    "PAIR_LEVELS",
    "PURITY",
    "READ_SECONDS",
    "RECORD_SECONDS",
    "RIDGE",
    "ROWS",
    "SCAN_SECONDS",
    "SCREEN_SHARE",
    "SKILL",
    "STORE_PATH",
    "TIME_BLOCKS",
    "TOLERANCE",
    "WAY_SECONDS",
    "WORD",
    "WORDS",
    "Demand",
    "Finding",
    "Pending",
    "Problem",
    "Screen",
    "Statistic",
    "abstain",
    "admit",
    "allowance",
    "arrange",
    "aucs",
    "basis",
    "bound",
    "bundled",
    "caps",
    "chisquare",
    "conclude",
    "concordance",
    "confine",
    "cost",
    "cramer",
    "crossfit",
    "dated",
    "declared",
    "decline",
    "disclose",
    "dominance",
    "draw_permutations",
    "duplication",
    "encode",
    "encodings",
    "enough",
    "exceed",
    "faint",
    "family",
    "floor",
    "forecast",
    "foresee",
    "forgo",
    "gauge",
    "halt",
    "hedge",
    "hypothesize",
    "load_store",
    "margin",
    "measure",
    "oof_fit",
    "pair",
    "pairs",
    "permute",
    "predict",
    "prepare",
    "propose",
    "purity_statistic",
    "ready",
    "reckon",
    "reflect",
    "replicate",
    "resample",
    "resamples",
    "share",
    "sift",
    "skill_bound",
    "skill_statistic",
    "spell",
    "stratify",
    "subsample",
    "tally",
    "tested",
    "throttle",
    "words",
]


def screen_table(
    frame: pandas.DataFrame,
    target: str,
    task: str,
    features: Sequence[Hashable],
    seed: int,
    group_column: str | None = None,
    time_column: str | None = None,
    time_budget_seconds: float | None = None,
    cutoff: str | None = None,
    known: Mapping[str, str] | None = None,
) -> Screen:
    """Screens training data with the certified settings.

    Groups and a time column structure the rows as `arrange` describes.
    Every suspect may block; `probe` records what a clean result means on
    these rows (see the module docstring). With a declared prediction time,
    every date column with values after it is late (`records.dated`), and
    every column declared known only after the outcome is declared
    (`records.declared`); those readings need no budget and run even when
    the scan is declined.

    Args:
        frame: Training rows including the target and any structure columns.
        target: Target column.
        task: `classification` or `regression`.
        features: Candidate columns.
        seed: Run seed for every random stream.
        group_column: Entity column, or None. Ignored when it is not a
            column of `frame`.
        time_column: Time column, or None. Ignored when it is not a column
            of `frame` or when `group_column` is used.
        time_budget_seconds: The project's declared time budget; the real
            scan and the probe together may take `share` of it, predicted
            from the table's shape before proposing (`foresee`) and from
            the proposed family before drawing (`reckon`), and a scan
            predicted not to fit is declined with the reason recorded. None
            means no cap beyond `MAX_PERMUTATIONS`.
        cutoff: The declared prediction time (`Dataset.cutoff`), or None.
        known: When each column becomes known (`Dataset.known`), or None.

    Returns:
        The report, using `CERTIFIED` settings with `seed`. A screen that was
        declined says so in `unchecked` and `warning`. A `features` entry
        that is not a candidate column, or a missing `target`, raises as in
        `screen`.
    """
    cfg = dataclasses.replace(leakage.CERTIFIED, run_seed=seed)
    present = [c for c in features if c in frame.columns]
    late = (dated(frame, present, cutoff), declared(present, known))
    rows, grouping, block_size = arrange(frame, group_column, time_column)
    limit = share(time_budget_seconds)
    report = admit(
        rows, target, task, cfg, features, grouping, block_size, limit
    )
    if report is None:
        report = screen(
            rows,
            target,
            task,
            cfg,
            features,
            grouping,
            block_size,
            dict.fromkeys(MECHANISM_DEFECTS, certificates.Authority.GATING),
            limit=limit,
        )
    classes = int(frame[target].nunique()) if len(frame) else 0
    if limit is not None and report.rationale.startswith("work limit"):
        unrun = "probe not run: the screen itself did not run"
        notes = (*report.notes, evidence(task, classes), unrun)
        return stamp(dataclasses.replace(report, notes=notes), late, cutoff)
    left = None if limit is None else limit - max(report.seconds, SCAN_SECONDS)
    found = probe(
        rows, target, task, list(features), cfg, grouping, block_size, left
    )
    report = disclose(report, found, evidence(task, classes), left, limit)
    return stamp(report, late, cutoff)


def stamp(
    report: Screen,
    late: tuple[Sequence[Finding], Sequence[Finding]],
    cutoff: str | None,
) -> Screen:
    """Adds the findings of the user's declarations to a report.

    Args:
        report: The screen's report.
        late: The findings of `records.dated` and of `records.declared`.
        cutoff: The declared prediction time, or None.

    Returns:
        The report with each declaration's findings, its mechanism gating at
        ceiling `LATE`, and a note: with a cutoff, how many date columns were
        read and the certificate; with columns declared known after the
        outcome, which. Unchanged when neither was declared.
    """
    dates, told = late
    findings, mechanisms, notes = [*report.findings], [], [*report.notes]
    if cutoff is not None:
        store = bundled()
        state = (
            "missing"
            if store is None
            else store.state(DATED_ID, dated_digest()).value
        )
        findings += dates
        mechanisms.append(DATED)
        notes.append(
            f"prediction time {cutoff} declared: {len(dates)} date column(s) "
            f"hold values after it; published certificate for dated "
            f"columns: {state}"
        )
    if told:
        findings += told
        mechanisms.append(DECLARED)
        notes.append(
            "declared known only after the outcome: "
            + ", ".join(f.feature for f in told)
        )
    if not mechanisms:
        return report
    return dataclasses.replace(
        report,
        findings=tuple(findings),
        gates={
            **report.gates,
            **dict.fromkeys(mechanisms, certificates.Authority.GATING),
        },
        ceilings={**report.ceilings, **dict.fromkeys(mechanisms, LATE)},
        notes=tuple(notes),
    )


def dated_digest() -> str:
    """Hash of the code the dated check depends on, its certificate's key.

    Returns:
        The traced hash of `records.dated`.
    """
    return trace.digest(dated)


def probe(
    rows: pandas.DataFrame,
    target: str,
    task: str,
    features: list[Hashable],
    cfg: leakage.Config,
    groups: pandas.Series | None,
    block_size: int | None,
    limit: float | None = None,
) -> dict[str, bool]:
    """Finds whether the screen catches each planted leak on these rows.

    The planted copy is screened with the same settings and structure as
    the real one, when its predicted seconds (`foresee`, then `reckon`)
    fit `limit`. Each plant is found when it is a suspect by the mechanism
    its type belongs to; the plants are named apart from every column
    (`plants.names`), so a user column named like a leak type stays.

    Args:
        rows: The rows to screen, in screening order, with the target.
        target: Target column.
        task: `classification` or `regression`.
        features: Candidate columns.
        cfg: Settings of the real screen.
        groups: Group label per row, or None.
        block_size: Rows per time block, or None.
        limit: Seconds left of the screen's share after the real scan, or
            None for no cap.

    Returns:
        Leak type to whether its plant was found; empty when the planted
        copy did not fit `limit` (nothing was drawn).
    """
    if limit is not None and limit < SCAN_SECONDS:
        return {}
    source = rows[[*features, target]]
    labels = plants.names(source.columns)
    planted = plants.table(source, target, cfg.run_seed, task == "regression")
    names = [*features, *labels.values()]
    if limit is not None:
        before = foresee(planted, target, task, cfg, names, groups, block_size)
        if before is not None and before > limit:
            return {}
    report = screen(
        planted,
        target,
        task,
        cfg,
        names,
        groups,
        block_size,
        None,
        apart=tuple(labels.values()),
        counted=plants.copy(planted, cfg.run_seed),
        limit=limit,
    )
    if report.unchecked:
        return {}
    owner = {d: m for m, ds in MECHANISM_DEFECTS.items() for d in ds}
    where = {**labels, plants.DUPLICATES: ROWS}
    return {
        defect: any(
            f.suspect
            and f.feature == where[defect]
            and f.mechanism == owner[defect]
            for f in report.findings
        )
        for defect in plants.KINDS
    }


def evidence(task: str, n_classes: int) -> str:
    """States the published certificate for this kind of target.

    Args:
        task: `classification` or `regression`.
        n_classes: Number of classes of the target.

    Returns:
        One sentence naming the kind and whether its certificate is current,
        stale or missing.
    """
    kind = family(task, n_classes)
    if kind is None:
        return "no published certificate covers this kind of target"
    store = bundled()
    state = (
        "missing"
        if store is None
        else store.state(DETECTORS[kind], code_digest()).value
    )
    return f"published certificate for {kind} targets: {state}"


def code_digest() -> str:
    """Hash of the code the screen's decisions depend on.

    The key of its certificate: `screen` and every function, class and
    constant of the package it reaches, the plants it is measured on and the
    certified settings (`trace.digest`). Run wiring the certificate
    does not measure, such as `screen_table`, is outside the hash, so it can
    change without staling the certificate.

    Returns:
        The traced hash.
    """
    return trace.digest(screen, plants.plant, leakage.CERTIFIED)


def scan(
    frame: pandas.DataFrame,
    target: str,
    task: str,
    cfg: leakage.Config,
    names: Sequence[Hashable],
    groups: numpy.ndarray | None,
    block_size: int | None,
    gates: Mapping[str, certificates.Authority],
    time_budget_seconds: float | None = None,
    apart: Sequence[Hashable] = (),
    extra: Sequence[Pending] = (),
    limit: float | None = None,
) -> Screen:
    """Runs the screen on rows that are already chosen.

    Args:
        frame: The chosen rows, including the target.
        target: Target column.
        task: `classification` or `regression`.
        cfg: Settings.
        names: Candidate columns.
        groups: Group label per row, or None.
        block_size: Consecutive rows per time block, or None.
        gates: Gate status per mechanism.
        time_budget_seconds: Declared time budget; None for no cap.
        apart: Columns left out of every other column's rest (`rival`).
        extra: Hypotheses formed on all rows (duplication), for a family.
        limit: Seconds the scan may take (`planning.confine`), or None.

    Returns:
        The report; one that stopped (an invalid outcome, too few rows, no
        column to test, a limit) says why and lists `unchecked` columns.
    """
    n = len(frame)
    try:
        problem = encode(frame[target], task, cfg, groups, block_size)
    except ValueError as error:
        # handled: returned as failed finding/problem/refusal with error text
        return abstain(cfg, gates, n, f"screen not run: {error}", names)
    if not enough(problem.y, task):
        return abstain(
            cfg, gates, n, "screen not run: too few rows of a class", names
        )
    alpha = budget.exact(cfg.alpha, "alpha")
    labels = tally(frame[target], task)
    structured = groups is not None or block_size is not None
    found = propose(
        frame, target, names, problem, cfg, *labels, structured=structured
    )
    pending = found[0] + encodings(frame, names, problem, cfg)
    pending += list(extra) if pending else []
    family, notes = len(pending), found[1]
    if family == 0:
        return abstain(
            cfg, gates, n, "screen not run: no column could be tested", names
        )
    total = max(cfg.permutations, resamples(family, alpha))
    declined, size = tested(pending, names), (n, family, total)
    if sift(pending, cfg) and total > MAX_PERMUTATIONS:
        return exceed(cfg, gates, size, MAX_PERMUTATIONS, declined)
    demand = reckon(pending, problem, frame, names, cfg)
    overrun = throttle(
        cfg, gates, size, pending, problem, declined, time_budget_seconds
    ) or confine(cfg, demand, limit)
    if overrun is not None:
        return overrun
    strata = stratify(n, groups, block_size)
    draw_permutations(
        pending, problem.y, labels[0], strata, total, alpha, cfg.run_seed
    )
    plan = (total, alpha)
    findings = conclude(
        pending, plan, problem, frame, labels, cfg, groups, gates, apart
    )
    notes = (*notes, *faint(n, cfg))
    report = Screen(
        tuple(findings), n, family, total, cfg.alpha, gates, caps(cfg), notes
    )
    return dataclasses.replace(report, seconds=demand.seconds)


def screen(
    frame: pandas.DataFrame,
    target: str,
    task: str,
    cfg: leakage.Config,
    features: Sequence[Hashable] | None = None,
    groups: pandas.Series | None = None,
    block_size: int | None = None,
    gates: Mapping[str, certificates.Authority] | None = None,
    time_budget_seconds: float | None = None,
    apart: Sequence[Hashable] = (),
    counted: pandas.DataFrame | None = None,
    limit: float | None = None,
) -> Screen:
    """Screens the candidate columns of training data for leakage suspicion.

    Args:
        frame: Training (or exploration) rows including the target. For time
            blocks the rows must be in time order.
        target: Name of the target column; it must be a column of `frame`.
        task: `classification` or `regression`.
        cfg: Settings; `alpha` is the Benjamini-Yekutieli level and the
            ceilings are pre-registered, never tuned on this data.
        features: Columns to test; defaults to all but the target.
        groups: Group label per row; permutations and folds keep groups
            whole.
        block_size: Consecutive rows per time block; exclusive with `groups`.
        gates: Gate status per mechanism; None means every mechanism is
            descriptive. `screen_table` sets it from the probe.
        time_budget_seconds: Declared time budget; bounds the work through
            `allowance`. None means no cap beyond `MAX_PERMUTATIONS`.
        apart: Columns never counted in another column's rest (`rival`):
            the probe's plants, each judged against the real columns only.
        counted: The rows the duplication test counts, with the same
            columns; None counts `frame` (the probe passes its copied rows).
        limit: Seconds the scan may take (`scan`); None for no cap.

    Returns:
        The report. Nothing is raised for a degenerate target or too few
        rows: the report is empty and says why.

    Raises:
        ValueError: If both `groups` and `block_size` are given, `groups`
            differs in length from `frame`, or `features` names a column
            that is missing or is the target.
    """
    if groups is not None and block_size is not None:
        raise errors.InputError("use groups or block_size, not both")
    if groups is not None and len(groups) != len(frame):
        raise errors.InputError("groups and frame must have the same length")
    names = [c for c in frame.columns if c != target]
    chosen = names if features is None else list(features)
    unknown = [c for c in chosen if c not in names]
    if unknown:
        raise errors.InputError(f"not feature columns: {unknown}")
    chosen_gates = dict.fromkeys(
        MECHANISM_DEFECTS, certificates.Authority.DESCRIPTIVE
    )
    chosen_gates.update(gates or {})
    group_values = None if groups is None else groups.to_numpy()
    rows, blocks = subsample(len(frame), group_values, block_size, cfg.run_seed)
    sample = frame.iloc[rows].reset_index(drop=True)
    sample_groups = None if group_values is None else group_values[rows]
    copied = duplication(
        frame if counted is None else counted, chosen, target, cfg
    )
    return scan(
        sample,
        target,
        task,
        cfg,
        chosen,
        sample_groups,
        blocks,
        chosen_gates,
        time_budget_seconds,
        apart,
        () if copied is None else (copied,),
        limit,
    )
