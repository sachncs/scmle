"""Outlier triage: what is it, not how far out is it.

A distance score only ranks candidates. The class of a row comes from
provenance the caller supplies (the field records):

* `error`: a value outside the field's declared instrument range, or a row
  that violates a declared cross-field contract. Such rows are classed even
  when their distance score is unremarkable.
* `population`: a value outside the range that defines the target
  population.
* `real_extreme`: a ranked candidate that lies inside the instrument range,
  breaks no contract and whose column is attested as able to produce real
  extremes.
* `unclassified`: any other ranked candidate; it is kept, never dropped.

Rows that are neither candidates nor violations get `none`. Without
provenance every candidate is `unclassified` and the result is `unverified`.
Candidates are ranked with the maximum robust z-score over columns
(median and normal-consistent MAD); an Isolation Forest ranking is offered
for multivariate ordering. The candidate cutoff only selects whom to look at
and never assigns a class.
"""

from collections.abc import Callable
from collections.abc import Mapping
import dataclasses
from typing import Any

import numpy as np
import pandas as pd
from sklearn import ensemble

from scmle.core import randomness

MAD_SCALE = 1.4826
CANDIDATE_Z = 6.0
CLASSES = ("error", "population", "real_extreme", "unclassified", "none")


@dataclasses.dataclass(frozen=True)
class FieldProvenance:
    """What is known about one numeric field.

    Attributes:
        instrument_range: (low, high) the instrument or write path can
            produce; None if unknown. Bounds are inclusive.
        population_range: (low, high) that defines the target population;
            None if unknown.
        extremes_possible: True if a named source attests that genuine
            extreme values occur; None if unknown.
    """

    instrument_range: tuple[float, float] | None = None
    population_range: tuple[float, float] | None = None
    extremes_possible: bool | None = None


Contract = Callable[[pd.DataFrame], pd.Series]


@dataclasses.dataclass(frozen=True)
class TriageResult:
    """Class of every row.

    Attributes:
        classes: One of `CLASSES` per row, aligned with the frame.
        scores: Maximum robust z-score per row (ranking only).
        candidates: Rows whose score reached the cutoff.
        tier: `attested` if any provenance was used, else `unverified`.
    """

    classes: pd.Series
    scores: pd.Series
    candidates: pd.Series
    tier: str

    def counts(self) -> dict[str, int]:
        """Counts the rows of each class.

        Returns:
            Class -> number of rows, for every name in `CLASSES`.
        """
        counted = self.classes.value_counts()
        return {name: int(counted.get(name, 0)) for name in CLASSES}


def robust_z(series: pd.Series) -> pd.Series:
    """Computes the absolute robust z-score of a series.

    Args:
        series: Numeric values.

    Returns:
        `|x - median| / (1.4826 * MAD)` per value, or all zeros where the
        spread is zero or not finite. Missing values score zero.
    """
    values = series.astype(float)
    centre = float(values.median())
    mad = MAD_SCALE * float((values - centre).abs().median())
    if not np.isfinite(mad) or mad == 0.0:
        return pd.Series(0.0, index=series.index)
    return ((values - centre).abs() / mad).fillna(0.0)


def row_scores(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Computes the maximum robust z-score per row over `columns`.

    Args:
        frame: Table.
        columns: Numeric columns to score.

    Returns:
        The largest score of each row, or zeros when `columns` is empty.
    """
    if not columns:
        return pd.Series(0.0, index=frame.index)
    table = pd.concat([robust_z(frame[c]) for c in columns], axis=1)
    return table.max(axis=1)


def isolation_rank(
    frame: pd.DataFrame, columns: list[str], seed: int
) -> pd.Series:
    """Scores rows with an Isolation Forest, for ordering.

    Missing values are filled with the column median.

    Args:
        frame: Table.
        columns: Numeric columns to use.
        seed: Run seed from which the forest's seed is derived.

    Returns:
        The anomaly score of each row (higher is more isolated).
    """
    block = frame[columns].astype(float)
    block = block.fillna(block.median())
    forest = ensemble.IsolationForest(
        random_state=randomness.derive_int_seed(seed, "outliers.forest", 0)
    ).fit(block)
    return pd.Series(-forest.score_samples(block), index=frame.index)


def outside(series: pd.Series, bounds: tuple[float, float]) -> pd.Series:
    """Flags values that lie outside inclusive bounds.

    Args:
        series: Numeric values.
        bounds: Low and high limits, both allowed.

    Returns:
        True where a non-missing value is below the low or above the high
        limit.
    """
    low, high = bounds
    values = series.astype(float)
    return ((values < low) | (values > high)) & values.notna()


def triage(
    frame: pd.DataFrame,
    provenance: Mapping[str, FieldProvenance] | None = None,
    contracts: Mapping[str, Contract] | None = None,
    cutoff: float = CANDIDATE_Z,
) -> TriageResult:
    """Classes every row by provenance.

    Args:
        frame: Exploration rows.
        provenance: Field name -> provenance; fields absent are unknown.
        contracts: Name -> function returning a boolean Series that is True
            where a cross-field contract is violated.
        cutoff: Robust-z value from which a row is a ranked candidate.

    Returns:
        The triage. A numeric column without provenance still ranks
        candidates but cannot produce `error`, `population` or
        `real_extreme`.

    Raises:
        ValueError: If `provenance` names a column the frame lacks, or a
            contract does not return a boolean Series.
    """
    known = dict(provenance or {})
    numeric = [
        str(c)
        for c in frame.columns
        if pd.api.types.is_numeric_dtype(frame[c])
        and not pd.api.types.is_bool_dtype(frame[c])
    ]
    scores = row_scores(frame, numeric)
    candidate = scores >= cutoff
    error = pd.Series(False, index=frame.index)
    population = pd.Series(False, index=frame.index)
    permitted = pd.Series(False, index=frame.index)
    for name, info in known.items():
        if name not in frame.columns:
            raise ValueError(f"provenance names unknown column '{name}'")
        if info.instrument_range is not None:
            error |= outside(frame[name], info.instrument_range)
        if info.population_range is not None:
            population |= outside(frame[name], info.population_range)
        if info.extremes_possible:
            z = robust_z(frame[name])
            permitted |= z >= cutoff
    for check in (contracts or {}).values():
        violated = check(frame)
        if not isinstance(violated, pd.Series) or violated.dtype != bool:
            raise ValueError("a contract must return a boolean Series")
        error |= violated.reindex(frame.index, fill_value=False)
    classes = pd.Series("none", index=frame.index, dtype=object)
    classes[candidate] = "unclassified"
    classes[candidate & permitted & ~error & ~population] = "real_extreme"
    classes[population] = "population"
    classes[error] = "error"
    tier = "attested" if known or contracts else "unverified"
    return TriageResult(classes, scores, candidate, tier)


def error_flags(
    frame: pd.DataFrame,
    provenance: Mapping[str, FieldProvenance],
    contracts: Mapping[str, Contract] | None = None,
) -> pd.Series:
    """Flags the rows classed `error` by provenance.

    Args:
        frame: Exploration rows.
        provenance: Field name -> provenance.
        contracts: Name -> function returning a boolean Series that is True
            where a cross-field contract is violated.

    Returns:
        A boolean per row on the index of `frame`.
    """
    result = triage(frame, provenance, contracts)
    return pd.Series(
        (result.classes == "error").to_numpy(dtype=bool), index=frame.index
    )


def summary(result: TriageResult) -> dict[str, Any]:
    """Summarises a triage result.

    Args:
        result: The triage to summarise.

    Returns:
        A JSON-ready mapping of the rows per class and the tier.
    """
    return {"counts": result.counts(), "tier": result.tier}
