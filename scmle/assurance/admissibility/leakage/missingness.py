"""The test of whether being recorded is associated with the target."""

import dataclasses
import math

import numpy
import pandas
from scipy import stats

from scmle.assurance.admissibility.leakage.settings import Config
from scmle.assurance.admissibility.leakage.settings import EXPECTED_FISHER
from scmle.assurance.admissibility.leakage.settings import MIN_CLASSES
from scmle.assurance.admissibility.leakage.settings import MISSING
from scmle.assurance.admissibility.leakage.settings import TARGET_BINS
from scmle.assurance.admissibility.leakage.skill import decide
from scmle.assurance.detectors import certificates
from scmle.contracts import errors
from scmle.contracts import qualification
from scmle.runtime import randomness


def cramers_v(table: numpy.ndarray) -> float:
    """Computes Cramer's V of a contingency table (no bias correction).

    Args:
        table: Counts, rows by columns.

    Returns:
        V in [0, 1]; 0 when the table is empty or has a single row or
        column.
    """
    total = float(table.sum())
    smaller = min(table.shape) - 1
    if total == 0.0 or smaller <= 0:
        return 0.0
    statistic = float(stats.chi2_contingency(table, correction=False)[0])  # pyright: ignore[reportArgumentType]  # scipy.stats result types are untyped tuples
    return math.sqrt(statistic / (total * smaller))


def bucket(target: pandas.Series, task: str) -> pandas.Series:
    """Bins the target for the missingness test.

    Args:
        target: The outcome.
        task: `classification` or `regression`.

    Returns:
        The classes as objects for classification; quartile bins (duplicate
        edges dropped) for a continuous target.
    """
    if task == "classification":
        return target.astype("object")
    return pandas.qcut(target, TARGET_BINS, duplicates="drop").astype("object")


@dataclasses.dataclass(frozen=True, slots=True)
class Missingness:
    """Result of testing whether a feature's missingness depends on the target.

    Attributes:
        feature: Feature name.
        method: `fisher` or `chi2`.
        p_value: Test p-value.
        cramers_v: Strength of association.
        n_missing: Rows with the feature missing.
        n: Rows.
        suspect: There are at least `min_rows` rows, p <= alpha and V is at
            least the association ceiling.
        check: Ledger record.
    """

    feature: str
    method: str
    p_value: float
    cramers_v: float
    n_missing: int
    n: int
    suspect: bool
    check: qualification.Check


def missingness_test(
    feature: pandas.Series,
    target: pandas.Series,
    task: str,
    cfg: Config,
    gate: certificates.Authority = certificates.Authority.DESCRIPTIVE,
    set_id: qualification.Subset = qualification.Subset.EXPLORATION,
) -> Missingness | None:
    """Tests whether recording a value depends on the outcome.

    Args:
        feature: The feature.
        target: The outcome.
        task: `classification` or `regression` (a regression target is
            binned to quartiles).
        cfg: Settings.
        gate: Whether the detector has a current certificate.
        set_id: Set the rows belong to.

    Returns:
        The test, or None when the feature is never or always missing, or
        the target has a single level (after binning). Fisher's exact test
        is used for a 2x2 table with an expected count below 5, otherwise
        the chi-square without continuity correction.

    Raises:
        ValueError: If `feature` and `target` differ in length.
    """
    if len(feature) != len(target):
        raise errors.InputError("feature and target must have the same length")
    missing = feature.isna().to_numpy()
    if missing.all() or not missing.any():
        return None
    classes = bucket(target, task)
    table = pandas.crosstab(missing, classes).to_numpy()
    if table.shape[1] < MIN_CLASSES:
        return None
    if table.shape == (2, 2) and (
        stats.contingency.expected_freq(table).min() < EXPECTED_FISHER
    ):
        method = "fisher"
        p_value = float(stats.fisher_exact(table).pvalue)  # pyright: ignore[reportAttributeAccessIssue]  # scipy.stats result types are untyped tuples
    else:
        method = "chi2"
        p_value = float(stats.chi2_contingency(table, correction=False)[1])  # pyright: ignore[reportArgumentType]  # scipy.stats result types are untyped tuples
    v = cramers_v(table)
    powered = len(feature) >= cfg.min_rows
    suspect = powered and p_value <= cfg.alpha and v >= cfg.association_ceiling
    path = f"{MISSING}.{feature.name}"
    check = qualification.Check(
        check_id=path,
        null_hypothesis=(
            "whether the value is recorded is independent of the outcome"
        ),
        statistic=v,
        p_value=p_value,
        interval=None,
        effect=v,
        n=len(feature),
        decision=decide(suspect, not powered, gate),
        set_id=set_id,
        seed_path=randomness.seed_path(cfg.run_seed, path, 0),
        tier=qualification.Tier.UNVERIFIED,
    )
    return Missingness(
        str(feature.name),
        method,
        p_value,
        v,
        int(missing.sum()),
        len(feature),
        suspect,
        check,
    )
