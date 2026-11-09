"""Conditional-independence tests that can only falsify a DAG.

A DAG implies conditional independences. Each one is tested on the
exploration set. A small p-value is evidence against the graph and is
flagged; a large one means only that the data did not contradict it: the test
has limited power, Markov-equivalent graphs imply the same independences, and
nothing here confirms a graph. Rejected implications are reported, never
corrected automatically. Every test returns a `Check` for the hypothesis
ledger, and the DAG tests never gate a run.

Tests:

* Continuous data: Fisher's z test of a partial correlation. With sample
  partial correlation r given a set S of size k from n rows,
  `z = atanh(r)` and `sqrt(n - k - 3) * |z|` is compared with a standard
  normal (Kalisch and Buehlmann, 2007, JMLR 8, section 2, around eq. 3; the
  formula was read in the paper). It assumes joint normality and linearity.
* Discrete data: the sum over strata of the Pearson chi-square statistics of
  x against y within each stratum of S, with the degrees of freedom summed
  (the standard stratified chi-square; strata where either variable is
  constant are skipped and counted).

Implied independences are the local Markov statements: each observed node is
independent of its observed non-descendants that are not parents, given its
parents (Lauritzen, 1996, Graphical Models). They are generated only when all
the parents are observed.
"""

from collections.abc import Sequence
import dataclasses
import enum
import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from scmle.contracts import qualification
from scmle.core import randomness
from scmle.data.admissibility import dag as dag_module

MAX_DISCRETE_LEVELS = 10
COMPONENT = "dag.ci"
FINITE_CAP = 1e308


class CIKind(enum.StrEnum):
    """Which test was used."""

    FISHER_Z = "fisher_z"
    STRATIFIED_CHI2 = "stratified_chi2"


class Verdict(enum.StrEnum):
    """What a test can say about an implied independence."""

    FALSIFIED = "falsified"
    NOT_FALSIFIED = "not_falsified"
    SKIPPED = "skipped"


@dataclasses.dataclass(frozen=True, slots=True)
class Statement:
    """An implied independence: x is independent of y given `given`.

    Attributes:
        x: First variable.
        y: Second variable.
        given: Conditioning variables, empty for a marginal independence.
    """

    x: str
    y: str
    given: tuple[str, ...]

    def text(self) -> str:
        """Writes the statement in words.

        Returns:
            A sentence naming `x`, `y` and, if any, the conditioning set.
        """
        if not self.given:
            return f"{self.x} is independent of {self.y}"
        return (
            f"{self.x} is independent of {self.y} given {', '.join(self.given)}"
        )


@dataclasses.dataclass(frozen=True, slots=True)
class CITest:
    """Result of one conditional-independence test.

    Attributes:
        kind: Test used.
        statistic: z (Fisher) or the summed chi-square.
        p_value: Two-sided (Fisher) or upper-tail (chi-square) p-value.
        effect: Partial correlation (Fisher) or Cramer's V of the pooled
            table (chi-square).
        n: Rows used.
        df: Degrees of freedom of the chi-square, 0 for Fisher.
        skipped_strata: Strata dropped as degenerate (chi-square only).
    """

    kind: CIKind
    statistic: float
    p_value: float
    effect: float
    n: int
    df: int
    skipped_strata: int


def residuals(values: np.ndarray, given: np.ndarray) -> np.ndarray:
    """Computes least-squares residuals on a set of columns and an intercept.

    Args:
        values: The response, one entry per row.
        given: Regressors, one row per entry of `values`.

    Returns:
        The response minus its least-squares fit, one entry per row.
    """
    design = np.column_stack([np.ones(len(values)), given])
    solution = np.linalg.lstsq(design, values, rcond=None)
    return values - design @ solution[0]


def partial_correlation(
    frame: pd.DataFrame, x: str, y: str, given: Sequence[str]
) -> float:
    """Computes the sample partial correlation of two columns given others.

    Args:
        frame: Numeric data holding all named columns.
        x: First variable.
        y: Second variable.
        given: Conditioning variables.

    Returns:
        The correlation of the residuals of `x` and `y` after regressing
        each on `given`, clipped to [-1, 1].

    Raises:
        ValueError: If a column has no spread after conditioning.
    """
    data = frame[[x, y, *given]].to_numpy(dtype="float64")
    conditioning = data[:, 2:]
    rx = residuals(data[:, 0], conditioning)
    ry = residuals(data[:, 1], conditioning)
    sx = float(np.sqrt(rx @ rx))
    sy = float(np.sqrt(ry @ ry))
    if sx == 0.0 or sy == 0.0:
        raise ValueError("a variable has no spread given the conditioning set")
    return float(max(-1.0, min(1.0, (rx @ ry) / (sx * sy))))


def fisher_z(
    frame: pd.DataFrame, x: str, y: str, given: Sequence[str]
) -> CITest:
    """Fisher's z test of zero partial correlation.

    Args:
        frame: Complete numeric data (no missing values).
        x: First variable.
        y: Second variable.
        given: Conditioning variables.

    Returns:
        The test. When the partial correlation is exactly +-1 the p-value is
        0.

    Raises:
        ValueError: If `n - len(given) - 3` is not positive, or a variable
            has no spread given the conditioning set.
    """
    n = len(frame)
    dof = n - len(given) - 3
    if dof <= 0:
        raise ValueError("too few rows for this conditioning set")
    r = partial_correlation(frame, x, y, given)
    if abs(r) >= 1.0:
        return CITest(CIKind.FISHER_Z, math.inf, 0.0, r, n, 0, 0)
    z = math.sqrt(dof) * math.atanh(r)
    p = float(2.0 * stats.norm.sf(abs(z)))
    return CITest(CIKind.FISHER_Z, z, p, r, n, 0, 0)


def stratified_chi2(
    frame: pd.DataFrame, x: str, y: str, given: Sequence[str]
) -> CITest:
    """Chi-square test of independence summed over strata of `given`.

    Args:
        frame: Complete discrete data.
        x: First variable.
        y: Second variable.
        given: Conditioning variables; their joint values define strata.

    Returns:
        The test. With no usable stratum the p-value is 1 and `df` is 0.
    """
    keys = [frame[c] for c in given] or [pd.Series(0, index=frame.index)]
    total = 0.0
    df = 0
    skipped = 0
    pooled = pd.crosstab(frame[x], frame[y])
    groups = frame.groupby(keys, sort=True, observed=True)
    for rows in (part[1] for part in groups):
        table = pd.crosstab(rows[x], rows[y])
        if table.shape[0] < 2 or table.shape[1] < 2:
            skipped += 1
            continue
        statistic = stats.chi2_contingency(table, correction=False)[0]
        total += float(statistic)
        df += (table.shape[0] - 1) * (table.shape[1] - 1)
    p = float(stats.chi2.sf(total, df)) if df > 0 else 1.0
    smaller = min(pooled.shape) - 1
    effect = math.sqrt(total / (len(frame) * smaller)) if smaller > 0 else 0.0
    return CITest(
        CIKind.STRATIFIED_CHI2, total, p, effect, len(frame), df, skipped
    )


def is_discrete(series: pd.Series) -> bool:
    """Decides whether a column is treated as categorical.

    Args:
        series: The column.

    Returns:
        True for categorical, boolean and non-numeric dtypes.
    """
    if isinstance(series.dtype, pd.CategoricalDtype):
        return True
    if pd.api.types.is_bool_dtype(series):
        return True
    return not pd.api.types.is_numeric_dtype(series)


def run_test(
    frame: pd.DataFrame, x: str, y: str, given: Sequence[str]
) -> CITest:
    """Choose Fisher's z for numeric data and chi-square for categories.

    Args:
        frame: Data holding the named columns.
        x: First variable.
        y: Second variable.
        given: Conditioning variables.

    Returns:
        The result of the chosen test.

    Raises:
        ValueError: On a mix of numeric and categorical columns, a column
            with more than `MAX_DISCRETE_LEVELS` levels for the chi-square,
            missing values, or an input the chosen test rejects (too few
            rows, or no spread).
    """
    names = [x, y, *given]
    sub = frame[names]
    if sub.isna().any().any():
        raise ValueError("conditional-independence tests need complete rows")
    flags = [is_discrete(sub[c]) for c in names]
    if all(flags):
        if any(sub[c].nunique() > MAX_DISCRETE_LEVELS for c in names):
            raise ValueError("too many levels for the chi-square test")
        return stratified_chi2(sub, x, y, given)
    if not any(flags):
        return fisher_z(sub.astype("float64"), x, y, given)
    raise ValueError("mixed numeric and categorical columns are unsupported")


def implied_statements(
    graph: dag_module.Dag,
) -> tuple[list[Statement], list[str]]:
    """List the local Markov statements of the observed part of a graph.

    Args:
        graph: The DAG hypothesis.

    Returns:
        (statements, skipped): statements to test, and the nodes skipped
        because a parent is latent.
    """
    statements: list[Statement] = []
    skipped: list[str] = []
    seen: set[tuple[frozenset[str], tuple[str, ...]]] = set()
    for node in graph.observed:
        parents = graph.parents(node)
        if not parents <= set(graph.observed):
            skipped.append(node)
            continue
        banned = {node} | parents | graph.descendants(node)
        for other in graph.observed:
            if other in banned:
                continue
            given = tuple(sorted(parents))
            key = (frozenset({node, other}), given)
            if key in seen:
                continue
            seen.add(key)
            statements.append(Statement(node, other, given))
    return statements, skipped


@dataclasses.dataclass(frozen=True, slots=True)
class Outcome:
    """One implied independence and what the data said.

    Attributes:
        statement: The implication.
        verdict: `falsified`, `not_falsified` or `skipped`.
        test: The test, None when skipped.
        reason: Why it was skipped, empty otherwise.
        check: Ledger record, None when skipped.
    """

    statement: Statement
    verdict: Verdict
    test: CITest | None
    reason: str
    check: qualification.Check | None


@dataclasses.dataclass(frozen=True, slots=True)
class FalsificationReport:
    """Result of testing every implication of a DAG on `X`.

    The report has no notion of confirmation: the best a graph can do is
    survive.

    Attributes:
        dag_hash: Hash of the graph under test.
        alpha: Family-wise level for falsifying any implication.
        level: Per-test level after Bonferroni correction.
        outcomes: One entry per statement.
        skipped_nodes: Nodes without statements because a parent is latent.
    """

    dag_hash: str
    alpha: float
    level: float
    outcomes: tuple[Outcome, ...]
    skipped_nodes: tuple[str, ...]

    @property
    def falsified(self) -> tuple[Outcome, ...]:
        """Implications the data rejected."""
        return tuple(o for o in self.outcomes if o.verdict is Verdict.FALSIFIED)

    @property
    def checks(self) -> tuple[qualification.Check, ...]:
        """Every ledger record, in statement order."""
        return tuple(o.check for o in self.outcomes if o.check is not None)


def falsify(
    graph: dag_module.Dag,
    frame: pd.DataFrame,
    alpha: float = 0.05,
    run_seed: int = 0,
    set_id: qualification.SetId = qualification.SetId.EXPLORATION,
) -> FalsificationReport:
    """Tests each implied independence of `graph` on the data.

    The per-test level is `alpha` divided by the number of testable
    statements (Bonferroni), so the chance of falsifying a true graph in any
    test is at most `alpha` under the test's assumptions. Pass `X` only.

    Args:
        graph: The DAG hypothesis.
        frame: Exploration-set data holding the graph's observed nodes.
        alpha: Family-wise level in (0, 1).
        run_seed: Run seed recorded in each check's seed path.
        set_id: Set the frame belongs to.

    Returns:
        The report. A statement that cannot be tested is `skipped` with a
        reason; it is never counted as passed.

    Raises:
        ValueError: If `alpha` is not a finite number in (0, 1) or an
            observed node is not a column of `frame`.
    """
    level_in = qualification.finite(alpha, "alpha")
    if not 0.0 < level_in < 1.0:
        raise ValueError("alpha must be within (0, 1)")
    missing = [n for n in graph.observed if n not in frame.columns]
    if missing:
        raise ValueError(f"observed nodes not in the data: {missing}")
    statements, skipped_nodes = implied_statements(graph)
    level = level_in / max(1, len(statements))
    outcomes: list[Outcome] = []
    for index, statement in enumerate(statements):
        try:
            result = run_test(frame, statement.x, statement.y, statement.given)
        except ValueError as error:
            outcomes.append(
                Outcome(statement, Verdict.SKIPPED, None, str(error), None)
            )
            continue
        falsified = result.p_value <= level
        statistic = (
            result.statistic if math.isfinite(result.statistic) else FINITE_CAP
        )
        check = qualification.Check(
            check_id=f"dag.ci.{statement.x}.{statement.y}|"
            + ",".join(statement.given),
            null_hypothesis=statement.text(),
            statistic=statistic,
            p_value=result.p_value,
            interval=None,
            effect=result.effect,
            n=result.n,
            decision=(
                qualification.Decision.WARN
                if falsified
                else qualification.Decision.PASS
            ),
            set_id=set_id,
            seed_path=randomness.seed_path(run_seed, COMPONENT, index),
            tier=qualification.Tier.UNVERIFIED,
        )
        verdict = Verdict.FALSIFIED if falsified else Verdict.NOT_FALSIFIED
        outcomes.append(Outcome(statement, verdict, result, "", check))
    return FalsificationReport(
        graph.digest(),
        level_in,
        level,
        tuple(outcomes),
        tuple(skipped_nodes),
    )


def describe(report: FalsificationReport) -> dict[str, Any]:
    """Summarise a report without claiming confirmation.

    Args:
        report: The falsification report.

    Returns:
        A JSON-ready dict with the DAG hash, status, family-wise and
        per-test levels, counts of falsified, not-falsified and skipped
        statements, and a note.
    """
    return {
        "dag_hash": report.dag_hash,
        "status": dag_module.STATUS,
        "alpha": report.alpha,
        "per_test_level": report.level,
        "falsified": len(report.falsified),
        "not_falsified": sum(
            o.verdict is Verdict.NOT_FALSIFIED for o in report.outcomes
        ),
        "skipped": sum(o.verdict is Verdict.SKIPPED for o in report.outcomes),
        "note": "tests can falsify an implication, never confirm the graph",
    }
