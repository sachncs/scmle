"""Findings and reports of the screen, and the refusals it can return."""

from collections.abc import Hashable
from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
from fractions import Fraction
import math
from typing import Any

import numpy
import pandas

from scmle.assurance.admissibility import leakage
from scmle.assurance.admissibility.screen.fitting import margin
from scmle.assurance.admissibility.screen.fitting import Problem
from scmle.assurance.admissibility.screen.hypotheses import caps
from scmle.assurance.admissibility.screen.hypotheses import floor
from scmle.assurance.admissibility.screen.hypotheses import Pending
from scmle.assurance.admissibility.screen.hypotheses import reading
from scmle.assurance.admissibility.screen.hypotheses import rival
from scmle.assurance.admissibility.screen.settings import allowance
from scmle.assurance.admissibility.screen.settings import BINARY_CLASSES
from scmle.assurance.admissibility.screen.settings import DATED
from scmle.assurance.admissibility.screen.settings import DECLARED
from scmle.assurance.admissibility.screen.settings import SKILL
from scmle.assurance.admissibility.screen.settings import WORDS
from scmle.assurance.detectors import certificates
from scmle.assurance.ledger import multiplicity
import scmle.assurance.ledger.budget
from scmle.runtime import randomness


@dataclasses.dataclass(frozen=True, slots=True)
class Finding:
    """The screen's result for one column and one mechanism.

    Attributes:
        feature: Column name.
        mechanism: `skill` (cross-fitted AUC or R^2), `missingness`
            (Cramer's V) or `record_purity` (share of the commonest class
            among recorded values).
        statistic: Observed effect size.
        p_value: Permutation (or analytic) p-value. When `complete` is False
            the run stopped early and this is the estimate from the
            permutations drawn so far, which lies above the largest
            threshold; it is 1 when no permutation was drawn because the
            observed statistic is not above the ceiling (`sift`).
        adjusted_p: Benjamini-Yekutieli adjusted p-value (1.0 when stopped
            early or never drawn).
        significant: Rejected by the Benjamini-Yekutieli step.
        lower: One-sided lower confidence bound of the effect size,
            computed for significant columns only.
        suspect: Significant and the bound exceeds the ceiling.
        complete: Whether all permutations were drawn.
        gate: Whether this suspicion may block: `gating` in a run's screen
            (`screen_table`), `descriptive` when the caller gave no gates.
    """

    feature: str
    mechanism: str
    statistic: float
    p_value: float
    adjusted_p: float
    significant: bool
    lower: float | None
    suspect: bool
    complete: bool
    gate: certificates.Authority

    @property
    def blocking(self) -> bool:
        """Whether this suspicion may stop a run (certified mechanism)."""
        return self.suspect and self.gate is certificates.Authority.GATING

    def describe(self, ceiling: float) -> str:
        """Writes one sentence for a profile: what was found, how to read it.

        Args:
            ceiling: The ceiling the lower bound exceeded.

        Returns:
            A sentence naming the column, the mechanism, the statistic,
            the bound, both p-values and whether the mechanism is certified,
            and stating that it is a suspicion and not proof.
        """
        what = WORDS[self.mechanism]
        if self.mechanism == DATED:
            return (
                f"feature '{self.feature}' is dated after the declared "
                f"prediction time: {what} {self.statistic:.3f} (the honest "
                f"share is {ceiling:.0f}; certified, may block). Its values "
                "were not known when the prediction is made; declare it "
                "unavailable at prediction (a date scheduled in advance, such "
                "as a due date, reads the same: give it as a lead time)"
            )
        if self.mechanism == DECLARED:
            return (
                f"feature '{self.feature}' is {what}: it cannot be read when "
                "the prediction is made (certified by the declaration, may "
                "block); exclude it (unavailable_at_prediction) or correct "
                "the declaration"
            )
        bound = "n/a" if self.lower is None else f"{self.lower:.3f}"
        status = (
            "certified, may block"
            if self.blocking
            else "uncertified mechanism: warning only"
        )
        return (
            f"feature '{self.feature}' is a leakage suspect: {what} "
            f"{self.statistic:.3f} (lower bound {bound} above the ceiling "
            f"{ceiling:.2f}; permutation p={self.p_value:.2g}, "
            f"Benjamini-Yekutieli adjusted p={self.adjusted_p:.2g}; "
            f"{status}). This is a suspicion, not proof: a strong legitimate "
            "predictor looks the same without provenance; review before "
            "trusting"
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Screen:
    """Result of the leakage screen over all candidate columns.

    Attributes:
        findings: One per tested column and mechanism.
        rows: Rows used.
        family: Number of hypotheses adjusted together.
        permutations: Permutations planned per hypothesis (0 if none were
            run); a hypothesis that stopped early drew fewer.
        alpha: Level of the Benjamini-Yekutieli step.
        gates: Gate status per mechanism.
        ceilings: Pre-registered ceilings by mechanism.
        notes: Plain statements of what was not done.
        unchecked: Columns the screen declined to test; they are unchecked,
            not clean.
        rationale: Why they were declined, for the warning.
        probe: Leak type to whether its plant was found on these rows; empty
            when no probe ran.
        seconds: The predicted seconds of the scan (`planning.reckon`), what
            it takes of a time budget's share; 0 when it drew nothing.
    """

    findings: tuple[Finding, ...]
    rows: int
    family: int
    permutations: int
    alpha: float
    gates: Mapping[str, certificates.Authority]
    ceilings: Mapping[str, float]
    notes: tuple[str, ...] = ()
    unchecked: tuple[str, ...] = ()
    rationale: str = ""
    probe: Mapping[str, bool] = dataclasses.field(default_factory=dict)
    seconds: float = 0.0

    @property
    def warning(self) -> str:
        """The loud statement of an incomplete screen, or an empty string."""
        if not self.unchecked:
            return ""
        return (
            f"leakage screen NOT RUN for {len(self.unchecked)} columns "
            f"({self.rationale}): these columns are unchecked, not "
            f"clean; {self.remedy}"
        )

    @property
    def remedy(self) -> str:
        """What would let the screen run, given why it did not."""
        if "too few rows" in self.rationale:
            return "add rows, or attest the columns"
        if self.rationale.startswith("screen not run"):
            return "fix the target or the columns, or attest them"
        return "raise --budget or screen fewer columns"

    @property
    def suspects(self) -> tuple[Finding, ...]:
        """Suspects, certified or not."""
        return tuple(f for f in self.findings if f.suspect)

    @property
    def blocking(self) -> tuple[Finding, ...]:
        """Suspects whose mechanism may block a run."""
        return tuple(f for f in self.findings if f.blocking)

    def sentences(self) -> tuple[str, ...]:
        """Writes one sentence per suspect.

        Returns:
            The `describe` sentence of every suspect, in finding order.
        """
        return tuple(
            f.describe(self.ceilings[f.mechanism]) for f in self.suspects
        )

    def to_dict(self) -> dict[str, Any]:
        """Converts the report to a JSON-ready document.

        A suspicion is not a finding, and the document says so.

        Returns:
            A dict with a note, the rule, level, rows, family size,
            permutations, gates, ceilings, notes, unchecked columns,
            warning, suspect and blocking column names (each once, though
            a column may be suspect by several mechanisms), and the findings
            that are significant or suspect (an infinite lower bound is
            written as None).
        """
        return {
            "note": "statistical suspicion only; not proof of leakage",
            "rule": "benjamini-yekutieli",
            "alpha": self.alpha,
            "rows": self.rows,
            "family": self.family,
            "permutations": self.permutations,
            "gates": {k: v.value for k, v in self.gates.items()},
            "probe": dict(self.probe),
            "ceilings": dict(self.ceilings),
            "notes": list(self.notes),
            "unchecked": list(self.unchecked),
            "warning": self.warning,
            "suspects": list(dict.fromkeys(f.feature for f in self.suspects)),
            "blocking": list(dict.fromkeys(f.feature for f in self.blocking)),
            "findings": [
                {
                    "feature": f.feature,
                    "mechanism": f.mechanism,
                    "statistic": (
                        f.statistic if math.isfinite(f.statistic) else None
                    ),
                    "p_value": f.p_value,
                    "adjusted_p": f.adjusted_p,
                    "significant": f.significant,
                    "lower_bound": (
                        f.lower
                        if f.lower is not None and math.isfinite(f.lower)
                        else None
                    ),
                    "suspect": f.suspect,
                    "complete": f.complete,
                }
                for f in self.findings
                if f.significant or f.suspect
            ],
        }


def abstain(
    cfg: leakage.Config,
    gates: Mapping[str, certificates.Authority],
    rows: int,
    note: str,
    names: Sequence[Hashable] = (),
) -> Screen:
    """Builds a report that tested nothing, and says why.

    The candidate columns are listed as unchecked, never as clean, so a
    screen that did not run is surfaced and `block` refuses it (ERR-2).

    Args:
        cfg: Settings (`alpha` and the ceilings are recorded).
        gates: Gate status per mechanism.
        rows: Rows the screen saw.
        note: Why nothing was tested.
        names: The candidate columns that went untested.

    Returns:
        A report with no findings, a family of 0, `note` as its only note,
        and `names` as its unchecked columns with `note` as the reason.
    """
    return Screen(
        (),
        rows,
        0,
        0,
        cfg.alpha,
        gates,
        caps(cfg),
        (note,),
        tuple(str(n) for n in names),
        note,
    )


def halt(
    cfg: leakage.Config,
    gates: Mapping[str, certificates.Authority],
    rows: int,
    family: int,
    note: str,
    declined: tuple[str, ...],
    limit: str,
) -> Screen:
    """Builds the report of a screen that exceeded a limit.

    Args:
        cfg: Settings (`alpha` and the ceilings are recorded).
        gates: Gate status per mechanism.
        rows: Rows the screen saw.
        family: Number of hypotheses that were to be adjusted together.
        note: Why the screen stopped.
        declined: The candidate columns that were not checked.
        limit: Name of the limit that was exceeded.

    Returns:
        A report with no findings, no permutations, and `note` as its only
        note.
    """
    return Screen(
        (),
        rows,
        family,
        0,
        cfg.alpha,
        gates,
        caps(cfg),
        (note,),
        declined,
        limit,
    )


def tally(
    outcome: pandas.Series, task: str
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Codes the binned outcome by class and counts the rows of each.

    Args:
        outcome: The target column of the chosen rows.
        task: `classification` or `regression` (binned by quartile).

    Returns:
        The class code per row and the rows per class.
    """
    classes = leakage.bucket(outcome, task)
    class_codes = pandas.factorize(classes, sort=True)[0].astype("int64")
    return class_codes, numpy.bincount(class_codes).astype("float64")


def exceed(
    cfg: leakage.Config,
    gates: Mapping[str, certificates.Authority],
    size: tuple[int, int, int],
    limit: int,
    declined: tuple[str, ...],
) -> Screen:
    """Builds the report of a screen that needs too many permutations.

    Args:
        cfg: Settings (`alpha` and the ceilings are recorded).
        gates: Gate status per mechanism.
        size: Rows the screen saw, hypotheses to adjust together, and the
            permutations they need.
        limit: The most permutations allowed (`MAX_PERMUTATIONS`).
        declined: The candidate columns that were not checked.

    Returns:
        A report with no findings that says the limit was exceeded.
    """
    n, family, total = size
    return halt(
        cfg,
        gates,
        n,
        family,
        (
            f"{family} hypotheses need {total} permutations to resolve "
            f"the Benjamini-Yekutieli threshold; the limit is "
            f"{limit}. Nothing was flagged (power, not size, "
            "is lost): screen fewer columns"
        ),
        declined,
        "permutation limit",
    )


def throttle(
    cfg: leakage.Config,
    gates: Mapping[str, certificates.Authority],
    size: tuple[int, int, int],
    pending: list[Pending],
    problem: Problem,
    declined: tuple[str, ...],
    time_budget_seconds: float | None,
) -> Screen | None:
    """Refuses a scan whose cross-fitted fits exceed the budget's cap.

    Args:
        cfg: Settings (`alpha` and the ceilings are recorded).
        gates: Gate status per mechanism.
        size: Rows the screen saw, hypotheses to adjust together, and the
            permutations each needs.
        pending: The hypotheses.
        problem: The encoded outcome.
        declined: The candidate columns that would not be checked.
        time_budget_seconds: Declared time budget, or None for no cap.

    Returns:
        A report with no findings that names the work limit, or None when
        the work fits under `allowance`.
    """
    n, family, total = size
    skill_columns = sum(
        1
        for h in pending
        if h.mechanism == SKILL and h.statistic is not None and h.active
    )
    work = (
        total
        * skill_columns
        * (problem.classes if problem.classes > BINARY_CLASSES else 1)
    )
    cap = allowance(time_budget_seconds)
    if cap is not None and work > cap:
        return halt(
            cfg,
            gates,
            n,
            family,
            (
                f"{family} hypotheses over {max(problem.classes, 1)} classes "
                f"need about {work} cross-fitted fits ({total} permutations "
                f"for {skill_columns} columns); the limit is {cap}. "
                "Nothing was flagged (power, not size, is lost): screen "
                "fewer columns"
            ),
            declined,
            f"work limit for a {time_budget_seconds:g} s time budget",
        )
    return None


def conclude(
    pending: list[Pending],
    plan: tuple[int, Fraction],
    problem: Problem,
    frame: pandas.DataFrame,
    classes: tuple[numpy.ndarray, numpy.ndarray],
    cfg: leakage.Config,
    groups: numpy.ndarray | None,
    gates: Mapping[str, certificates.Authority],
    apart: Sequence[Hashable] = (),
) -> list[Finding]:
    """Adjusts the drawn p-values together and bounds the significant ones.

    A significant hypothesis is a suspect when its bound passes its
    ceiling; a skill hypothesis must also outrun the other columns
    (`rival`): the lower confidence bound of its paired skill difference
    over theirs (`margin`) must be at least zero, so a tie counts.

    Args:
        pending: The hypotheses, after their permutations were drawn.
        plan: Permutations planned per hypothesis, and the exact level of
            the Benjamini-Yekutieli step.
        problem: The encoded outcome with its folds.
        frame: The chosen rows.
        classes: Class code of the binned outcome per row, and rows per
            class.
        cfg: Settings.
        groups: Group label per row, or None.
        gates: Gate status per mechanism.
        apart: Columns left out of every other column's rest.

    Returns:
        One finding per hypothesis, in order. A hypothesis that stopped
        early keeps its estimated p-value and an adjusted p-value of 1.
    """
    total, alpha = plan
    ps = [
        h.p_value if h.p_value is not None else (h.exceed + 1) / (h.drawn + 1)
        for h in pending
    ]
    complete = [h.statistic is None or h.drawn == total for h in pending]
    used = [p if done else 1.0 for p, done in zip(ps, complete, strict=True)]
    adjustment = multiplicity.adjust(
        used, scmle.assurance.ledger.budget.Correction.BY, alpha
    )
    findings = []
    for k, h in enumerate(pending):
        significant = adjustment.rejected[k]
        bound: float | None = None
        suspect = False
        if significant:
            bound, suspect = weigh(
                h, pending, problem, frame, classes, cfg, groups, apart
            )
        findings.append(
            Finding(
                h.feature,
                h.mechanism,
                h.observed,
                ps[k],
                adjustment.adjusted[k] if complete[k] else 1.0,
                significant,
                bound,
                suspect,
                complete[k],
                gates[h.mechanism],
            )
        )
    return findings


def weigh(
    h: Pending,
    pending: Sequence[Pending],
    problem: Problem,
    frame: pandas.DataFrame,
    classes: tuple[numpy.ndarray, numpy.ndarray],
    cfg: leakage.Config,
    groups: numpy.ndarray | None,
    apart: Sequence[Hashable] = (),
) -> tuple[float, bool]:
    """Bounds one hypothesis and says whether it can be a suspect.

    Neither the bound nor a skill hypothesis's margin over the rest of its
    table reads a permutation, so the answer is the same before and after
    the permutations; `conclude` asks it of a significant hypothesis.

    Args:
        h: The hypothesis.
        pending: The hypotheses of the family.
        problem: The encoded outcome with its folds.
        frame: The chosen rows.
        classes: Class code of the binned outcome per row, and rows per
            class.
        cfg: Settings.
        groups: Group label per row, or None.
        apart: Columns left out of every other column's rest.

    Returns:
        The lower bound of its effect, and whether the bound passes its
        ceiling and, for skill, the column outruns the rest (`margin` at
        least zero).
    """
    class_codes, totals = classes
    bound = floor(h, problem, frame, class_codes, totals, cfg, groups)
    suspect = bound > caps(cfg)[h.mechanism]
    rest = None
    if suspect and h.mechanism == SKILL and h.data is not None:
        rest = rival(h, pending, problem, apart)
    if rest is not None and h.data is not None:
        path = f"leakage.screen.margin.{h.feature}"
        rng = randomness.spawn(cfg.seed, path, 0)
        mine = reading(h, h.data, frame, problem)
        suspect = margin(problem, mine, rest, cfg, groups, rng) >= 0
    return bound, suspect


def faint(rows: int, cfg: leakage.Config) -> tuple[str, ...]:
    """States the low power of a scan on fewer rows than `min_rows`.

    Args:
        rows: Rows the scan saw.
        cfg: Settings (`min_rows`).

    Returns:
        One note below `min_rows`, none otherwise.
    """
    if rows >= cfg.minimum:
        return ()
    note = (
        f"low power: {rows} rows, fewer than {cfg.minimum}; the probe "
        "states which planted leaks these rows hide"
    )
    return (note,)
