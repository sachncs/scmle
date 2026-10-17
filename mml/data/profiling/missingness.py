"""Does a value being missing say something about the outcome?

A column that is missing for a reason (a test not ordered, a field customers
skip when they plan to leave) carries signal in the gap itself. Columns whose
gaps are unrelated to the target do not, and flagging them only adds noise.
Each column with enough gaps and enough values is tested on development rows:
a chi-square test for class labels, a rank test for numeric targets, with a
Bonferroni correction across columns.
"""

import dataclasses
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

MIN_MISSING_RATE = 0.02
MIN_PRESENT_ROWS = 30
MIN_MISSING_ROWS = 10
ALPHA = 0.01
MIN_EFFECT = 0.05


@dataclasses.dataclass(frozen=True)
class MissingnessFinding:
    """Association between one column's gaps and the target.

    Attributes:
        column: Column name.
        missing_rate: Share of development rows with the value missing.
        test: Name of the statistical test used.
        p_value: Probability of an association this strong by chance.
        effect: Effect size in [0, 1] (Cramer's V or rank-biserial).
        informative: Whether the association passes both the corrected
            significance level and the minimum effect.
    """

    column: str
    missing_rate: float
    test: str
    p_value: float
    effect: float
    informative: bool

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready record."""
        return dataclasses.asdict(self)


def class_association(
    gaps: pd.Series, target: pd.Series
) -> tuple[str, float, float]:
    """Chi-square test and Cramer's V of gaps against class labels."""
    table = pd.crosstab(gaps, target)
    result = stats.chi2_contingency(table)
    size = table.to_numpy().sum()
    smaller = min(table.shape) - 1
    effect = (
        float(np.sqrt(result.statistic / (size * smaller))) if smaller else 0.0
    )
    return "chi-square", float(result.pvalue), effect


def value_association(
    gaps: pd.Series, target: pd.Series
) -> tuple[str, float, float]:
    """Mann-Whitney test and rank-biserial effect of gaps on a number."""
    missing = target[gaps].to_numpy(dtype=float)
    present = target[~gaps].to_numpy(dtype=float)
    result = stats.mannwhitneyu(missing, present, alternative="two-sided")
    effect = abs(2.0 * result.statistic / (len(missing) * len(present)) - 1.0)
    return "mann-whitney", float(result.pvalue), float(effect)


def assess(
    frame: pd.DataFrame,
    columns: list[str],
    target: pd.Series,
    classification: bool,
) -> list[MissingnessFinding]:
    """Tests every column whose gaps are numerous enough to judge.

    Args:
        frame: Development rows.
        columns: Feature columns to examine.
        target: Target aligned with `frame`.
        classification: Whether the target holds class labels.

    Returns:
        One finding per examined column, most informative first.
    """
    candidates = []
    for column in columns:
        gaps = frame[column].isna()
        missing, present = int(gaps.sum()), int((~gaps).sum())
        if (
            missing < MIN_MISSING_ROWS
            or present < MIN_PRESENT_ROWS
            or missing / len(frame) < MIN_MISSING_RATE
        ):
            continue
        candidates.append((column, gaps))
    threshold = ALPHA / max(len(candidates), 1)
    findings = []
    for column, gaps in candidates:
        test, p_value, effect = (
            class_association(gaps, target)
            if classification
            else value_association(gaps, target)
        )
        findings.append(
            MissingnessFinding(
                column,
                float(gaps.mean()),
                test,
                p_value,
                effect,
                bool(p_value < threshold and effect >= MIN_EFFECT),
            )
        )
    return sorted(findings, key=lambda f: (-f.effect, f.column))
