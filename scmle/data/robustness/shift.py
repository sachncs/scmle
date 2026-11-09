"""Two-sample shift tests between the exploration set and a reference.

Three tests, each returning `Check` records:

* `c2st`: a classifier two-sample test (Lopez-Paz and Oquab, 2017, ICLR).
  A gradient-boosting classifier is fitted on half of the pooled rows and
  scored on the other half. Given the fitted classifier, the held-out side
  labels are exchangeable under the null that both samples share one
  distribution, so the permutation p-value of the held-out AUC is exact with
  no refitting. The AUC carries a DeLong interval (DeLong, DeLong and
  Clarke-Pearson, 1988, Biometrics 44:837-845), which treats the fitted
  classifier as fixed.
* `mmd_test`: the unbiased squared maximum mean discrepancy with a Gaussian
  kernel (Gretton et al., 2012, JMLR 13:723-773), the bandwidth fixed by the
  median heuristic on the pooled sample before any permutation (the pooled
  sample does not change when labels are permuted, so the heuristic keeps the
  permutation test exact), and a permutation p-value.
* `feature_tests`: Kolmogorov-Smirnov for numeric columns and chi-square for
  categorical ones, with Benjamini-Hochberg adjustment over the family.

Permutation p-values are ``(1 + #{T_b >= T}) / (B + 1)``, exact under
exchangeability of the side labels. Where rows come in clusters or blocks,
pass `strata` so labels are permuted only within a stratum; that is valid
only when each stratum holds rows of both samples. Two-sample tests detect
covariate shift, not label or concept shift, and a non-rejection is a
failure to detect, never evidence of equivalence.

Operating characteristics (size and power at a planted shift) are on file in
`benchmarks/qualification/f6_certificates.json`; until a current certificate
passes, decisions are demoted to warnings.
"""

import dataclasses
import enum
import math
import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import distance
from sklearn import ensemble
from sklearn import metrics

from scmle.contracts import qualification
from scmle.core import randomness
from scmle.data.detectors import certificates
from scmle.data.ledger import controls

C2ST_ID = "f6-shift-c2st"
MMD_ID = "f6-shift-mmd"
DEFECT_TYPE = "mean_shift_auc060"
ALPHA = 0.05
PERMUTATIONS = 199
MIN_ROWS = 50
MAX_MMD_ROWS = 1000
MAX_ONE_HOT = 20
HOLDOUT = 0.5
CHUNK = 100
TIE_TOLERANCE = 1e-12
LOW_EXPECTED = 5.0


class Status(enum.StrEnum):
    """Outcome of a shift check against a deployment sample."""

    PASS = "pass"  # nosec B105 - check outcome, not a credential
    FAIL = "fail"
    UNVERIFIABLE = "unverifiable"


def permutation_p_value(observed: float, null: np.ndarray) -> float:
    """Computes ``(1 + #{null >= observed}) / (B + 1)``.

    A draw counts as at least as large as `observed` when it is within
    `TIE_TOLERANCE` below it.

    Args:
        observed: The statistic on the real labels.
        null: The statistic on the B permuted labellings.

    Returns:
        The permutation p-value, in (0, 1].
    """
    draws = np.asarray(null, dtype=float)
    hits = int(np.sum(draws >= observed - TIE_TOLERANCE))
    return (1.0 + hits) / (draws.size + 1.0)


def auc_delong(
    scores: np.ndarray, side: np.ndarray, confidence: float = 0.95
) -> tuple[float, tuple[float, float]]:
    """ROC AUC with the DeLong interval, from midranks.

    Args:
        scores: Classifier scores.
        side: 1 for the positive class, 0 for the negative class.
        confidence: Level of the two-sided interval, in (0, 1).

    Returns:
        (AUC, (low, high)) clipped to [0, 1]. The interval is degenerate when
        the estimated variance is zero.

    Raises:
        ValueError: If a class has fewer than two rows or the level is bad.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be within (0, 1)")
    values = np.asarray(scores, dtype=float)
    labels = np.asarray(side)
    positive = values[labels == 1]
    negative = values[labels == 0]
    m, n = len(positive), len(negative)
    if min(m, n) < 2:
        raise ValueError("each class needs at least two rows")
    pooled = stats.rankdata(np.r_[positive, negative])
    v10 = (pooled[:m] - stats.rankdata(positive)) / n
    v01 = 1.0 - (pooled[m:] - stats.rankdata(negative)) / m
    auc = float(v10.mean())
    variance = float(v10.var(ddof=1) / m + v01.var(ddof=1) / n)
    z = float(stats.norm.ppf(1.0 - (1.0 - confidence) / 2.0))
    half = z * math.sqrt(max(variance, 0.0))
    return auc, (max(0.0, auc - half), min(1.0, auc + half))


def pooled_codes(sample: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Stacks the two tables into one numeric table.

    Args:
        sample: Exploration rows.
        reference: Deployment sample; it must hold every column of
            `sample`.

    Returns:
        A float table with the sample rows first and then the reference rows
        (index reset), the columns of `sample` in order. Non-numeric columns
        are replaced by the integer codes of their categories (-1 for a
        missing value).
    """
    columns = list(sample.columns)
    both = pd.concat([sample[columns], reference[columns]], ignore_index=True)
    for name in columns:
        if not pd.api.types.is_numeric_dtype(both[name]):
            both[name] = both[name].astype("category").cat.codes
    return both.astype(float)


def side_labels(sample_rows: int, reference_rows: int) -> np.ndarray:
    """Labels the pooled rows by side.

    Args:
        sample_rows: Number of sample rows.
        reference_rows: Number of reference rows.

    Returns:
        A float array of length ``sample_rows + reference_rows``: 0 for each
        sample row, then 1 for each reference row.
    """
    return np.r_[np.zeros(sample_rows), np.ones(reference_rows)]


def holdout_split(
    side: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Splits the pooled rows into train and test, stratified by side.

    Within each side the rows are shuffled and the first
    ``round(rows * (1 - HOLDOUT))`` go to the train part.

    Args:
        side: The side label of each pooled row, 0 or 1.
        rng: Random generator that shuffles the rows.

    Returns:
        (train, test) arrays of row positions, each sorted ascending.
    """
    train: list[int] = []
    test: list[int] = []
    for label in (0.0, 1.0):
        rows = rng.permutation(np.flatnonzero(side == label))
        cut = round(len(rows) * (1.0 - HOLDOUT))
        train.extend(rows[:cut].tolist())
        test.extend(rows[cut:].tolist())
    return np.array(sorted(train)), np.array(sorted(test))


def decide(
    rejected: bool,
    requested: qualification.Decision,
    gate: certificates.GateStatus,
) -> qualification.Decision:
    """Chooses the decision for a test outcome.

    Args:
        rejected: Whether the test fires.
        requested: The action asked for when it fires.
        gate: Whether a current certificate lets the check act.

    Returns:
        `Decision.PASS` when the test does not fire; otherwise `requested`,
        limited to a warning when the gate is not gating.
    """
    if not rejected:
        return qualification.Decision.PASS
    return certificates.demote(requested, gate)


def c2st(
    sample: pd.DataFrame,
    reference: pd.DataFrame,
    seed: int,
    *,
    permutations: int = PERMUTATIONS,
    alpha: float = ALPHA,
    delta: float = 0.0,
    strata: np.ndarray | None = None,
    gate: certificates.GateStatus = certificates.GateStatus.DESCRIPTIVE,
    on_shift: qualification.Decision = qualification.Decision.WARN,
) -> qualification.Check:
    """Classifier two-sample test with a permutation p-value.

    Args:
        sample: Exploration rows.
        reference: Deployment sample, same columns.
        seed: Run seed, which fixes the split, the classifier and the
            permutations.
        permutations: Number of label permutations B.
        alpha: Level of the test.
        delta: Minimum excess AUC worth acting on (from the charter): the
            decision fires only when the AUC lower bound exceeds
            ``0.5 + delta`` as well as ``p <= alpha``.
        strata: One label per pooled row (sample rows, then reference rows);
            labels are permuted within a stratum.
        gate: Whether a current certificate lets the check act.
        on_shift: Action requested when the test fires.

    Returns:
        A check whose statistic is the held-out AUC, effect the excess over
        0.5, interval the DeLong interval and p-value the permutation value.
        Differing columns or too few rows raise `ValueError` (see
        `check_pair`).
    """
    check_pair(sample, reference)
    both = pooled_codes(sample, reference)
    side = side_labels(len(sample), len(reference))
    rng = randomness.component_rng(seed, C2ST_ID, 0)
    train, test = holdout_split(side, rng)
    model = ensemble.HistGradientBoostingClassifier(
        random_state=randomness.derive_int_seed(seed, C2ST_ID, 1)
    )
    model.fit(both.iloc[train], side[train])
    scores = model.predict_proba(both.iloc[test])[:, 1]
    held = side[test]
    observed = float(metrics.roc_auc_score(held, scores))
    labels = None if strata is None else np.asarray(strata)[test]
    null = np.array(
        [
            metrics.roc_auc_score(
                controls.permute_within(held, labels, rng), scores
            )
            for draw in range(permutations)
        ]
    )
    p_value = permutation_p_value(observed, null)
    auc, interval = auc_delong(scores, held)
    fires = p_value <= alpha and interval[0] > 0.5 + delta
    return qualification.Check(
        check_id=C2ST_ID,
        null_hypothesis="exploration rows and the reference sample come "
        "from one distribution",
        statistic=auc,
        p_value=p_value,
        interval=interval,
        effect=auc - 0.5,
        n=len(test),
        decision=decide(fires, on_shift, gate),
        set_id=qualification.SetId.EXPLORATION,
        seed_path=randomness.seed_path(seed, C2ST_ID, 0),
        tier=qualification.Tier.UNVERIFIED,
    )


def check_pair(sample: pd.DataFrame, reference: pd.DataFrame) -> None:
    """Requires the same columns and enough rows on both sides.

    Args:
        sample: Exploration rows.
        reference: Deployment sample.

    Raises:
        ValueError: If the two tables have different column sets, or either
            has fewer than `MIN_ROWS` rows.
    """
    if set(sample.columns) != set(reference.columns):
        raise ValueError("sample and reference must have the same columns")
    if min(len(sample), len(reference)) < MIN_ROWS:
        raise ValueError(f"each side needs at least {MIN_ROWS} rows")


def design_matrix(both: pd.DataFrame, rows: int) -> np.ndarray:
    """Numeric design for the kernel: standardised, categories one-hot.

    Missing numbers are filled with the pooled median. Categorical columns
    with more than `MAX_ONE_HOT` levels are left out.

    Args:
        both: The pooled rows. Boolean columns are treated as categorical.
        rows: Row count of the matrix returned when no column is usable.

    Returns:
        A float matrix with one row per pooled row: each numeric column
        standardised to mean 0 and unit spread (left unscaled when constant),
        then one indicator column per level of each kept categorical column.
        A single zero column of `rows` rows when no column is kept.
    """
    parts = []
    for name in both.columns:
        column = both[name]
        if pd.api.types.is_numeric_dtype(column) and not (
            pd.api.types.is_bool_dtype(column)
        ):
            values = column.to_numpy(dtype=float)
            filled = np.where(np.isfinite(values), values, np.nanmedian(values))
            spread = float(filled.std())
            parts.append(
                ((filled - filled.mean()) / (spread if spread > 0 else 1.0))[
                    :, None
                ]
            )
        elif column.nunique(dropna=False) <= MAX_ONE_HOT:
            parts.append(
                pd.get_dummies(column.astype(str)).to_numpy(dtype=float)
            )
    if not parts:
        return np.zeros((rows, 1))
    return np.hstack(parts)


def median_bandwidth(design: np.ndarray) -> float:
    """Computes the median-heuristic kernel bandwidth.

    Args:
        design: The pooled design matrix, one row per observation.

    Returns:
        The median pairwise Euclidean distance of the rows, or 1 when that
        median is zero or there are fewer than two rows.
    """
    pairwise = distance.pdist(design)
    middle = float(np.median(pairwise)) if pairwise.size else 0.0
    return middle if middle > 0.0 else 1.0


def gaussian_kernel(design: np.ndarray, bandwidth: float) -> np.ndarray:
    """Computes ``exp(-|a - b|^2 / (2 h^2))`` for every pair of rows.

    Args:
        design: The pooled design matrix, one row per observation.
        bandwidth: The bandwidth h.

    Returns:
        The symmetric kernel matrix, with ones on the diagonal.
    """
    squared = distance.cdist(design, design, "sqeuclidean")
    return np.exp(-squared / (2.0 * bandwidth * bandwidth))


def mmd_statistics(kernel: np.ndarray, first: np.ndarray) -> np.ndarray:
    """Unbiased squared MMD for each column of 0/1 membership indicators.

    Args:
        kernel: Symmetric kernel matrix of the pooled rows.
        first: Matrix with one column per labelling; 1 marks the first
            sample. Every column must hold the same two group sizes.

    Returns:
        A float array with one statistic per column of `first`.
    """
    off = kernel - np.diag(np.diag(kernel))
    second = 1.0 - first
    m = first.sum(axis=0)
    r = second.sum(axis=0)
    xx = (first * (off @ first)).sum(axis=0)
    yy = (second * (off @ second)).sum(axis=0)
    xy = (first * (off @ second)).sum(axis=0)
    return xx / (m * (m - 1.0)) + yy / (r * (r - 1.0)) - 2.0 * xy / (m * r)


def mmd_test(
    sample: pd.DataFrame,
    reference: pd.DataFrame,
    seed: int,
    *,
    permutations: int = PERMUTATIONS,
    alpha: float = ALPHA,
    strata: np.ndarray | None = None,
    gate: certificates.GateStatus = certificates.GateStatus.DESCRIPTIVE,
    on_shift: qualification.Decision = qualification.Decision.WARN,
) -> qualification.Check:
    """Kernel MMD two-sample test with a permutation p-value.

    At most `MAX_MMD_ROWS` rows per side are used (a seeded subsample), which
    bounds the kernel matrix.

    Args:
        sample: Exploration rows.
        reference: Deployment sample, same columns.
        seed: Run seed.
        permutations: Number of label permutations B.
        alpha: Level of the test.
        strata: One label per pooled row; labels are permuted within them.
        gate: Whether a current certificate lets the check act.
        on_shift: Action requested when the test fires.

    Returns:
        A check whose statistic is the unbiased squared MMD, effect the MMD
        (square root of its positive part) and p-value the permutation value.
        Differing columns or too few rows raise `ValueError` (see
        `check_pair`).
    """
    check_pair(sample, reference)
    rng = randomness.component_rng(seed, MMD_ID, 0)
    first = np.sort(
        rng.choice(len(sample), min(len(sample), MAX_MMD_ROWS), replace=False)
    )
    second = np.sort(
        rng.choice(
            len(reference), min(len(reference), MAX_MMD_ROWS), replace=False
        )
    )
    both = pd.concat(
        [sample.iloc[first], reference.iloc[second]], ignore_index=True
    )
    design = design_matrix(both, len(both))
    kernel = gaussian_kernel(design, median_bandwidth(design))
    labels = (
        None
        if strata is None
        else np.r_[
            np.asarray(strata)[first],
            np.asarray(strata)[len(sample) + second],
        ]
    )
    indicator = np.r_[np.ones(len(first)), np.zeros(len(second))]
    observed = float(mmd_statistics(kernel, indicator[:, None])[0])
    null = np.empty(permutations)
    for start in range(0, permutations, CHUNK):
        stop = min(start + CHUNK, permutations)
        block = np.column_stack(
            [
                controls.permute_within(indicator, labels, rng)
                for draw in range(start, stop)
            ]
        )
        null[start:stop] = mmd_statistics(kernel, block)
    p_value = permutation_p_value(observed, null)
    return qualification.Check(
        check_id=MMD_ID,
        null_hypothesis="exploration rows and the reference sample come "
        "from one distribution",
        statistic=observed,
        p_value=p_value,
        interval=None,
        effect=math.sqrt(max(observed, 0.0)),
        n=len(both),
        decision=decide(p_value <= alpha, on_shift, gate),
        set_id=qualification.SetId.EXPLORATION,
        seed_path=randomness.seed_path(seed, MMD_ID, 0),
        tier=qualification.Tier.UNVERIFIED,
    )


@dataclasses.dataclass(frozen=True, slots=True)
class FeatureShift:
    """One column compared between the sample and the reference.

    Attributes:
        column: Column name.
        kind: `numeric` (Kolmogorov-Smirnov) or `categorical` (chi-square).
        statistic: KS distance or chi-square statistic.
        p_value: Unadjusted p-value.
        adjusted_p_value: Benjamini-Hochberg adjusted over all columns.
        effect: KS distance or Cramer's V.
        rejected: Whether the adjusted p-value is at most the level.
        low_counts: True when a chi-square expected count is below 5, which
            makes the approximation poor.
    """

    column: str
    kind: str
    statistic: float
    p_value: float
    adjusted_p_value: float
    effect: float
    rejected: bool
    low_counts: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Converts the record to a JSON-ready dictionary.

        Returns:
            The fields by name.
        """
        return dataclasses.asdict(self)


def categorical_test(
    left: pd.Series, right: pd.Series
) -> tuple[float, float, float, bool]:
    """Tests whether two samples share one categorical distribution.

    Args:
        left: The sample values.
        right: The reference values.

    Returns:
        (chi-square statistic, p-value, Cramer's V capped at 1, flag that is
        True when an expected count is below `LOW_EXPECTED`). With fewer
        than two distinct levels the result is (0, 1, 0, False).
    """
    table = pd.concat(
        [
            left.astype(str).value_counts(),
            right.astype(str).value_counts(),
        ],
        axis=1,
        sort=True,
    ).fillna(0.0)
    counts = table.to_numpy(dtype=float).T
    if counts.shape[1] < 2:
        return 0.0, 1.0, 0.0, False
    result = stats.chi2_contingency(counts, correction=False)
    total = float(counts.sum())
    v = math.sqrt(float(result.statistic) / total) if total > 0 else 0.0
    return (
        float(result.statistic),
        float(result.pvalue),
        min(v, 1.0),
        bool((result.expected_freq < LOW_EXPECTED).any()),
    )


def feature_tests(
    sample: pd.DataFrame, reference: pd.DataFrame, alpha: float = ALPHA
) -> tuple[FeatureShift, ...]:
    """KS or chi-square per shared column, adjusted by Benjamini-Hochberg.

    Args:
        sample: Exploration rows.
        reference: Deployment sample.
        alpha: Level applied to the adjusted p-values.

    Returns:
        One record per shared column, in column order. Columns that are
        empty on either side get p = 1.
    """
    rows = []
    for name in sample.columns:
        if name not in reference.columns:
            continue
        left, right = sample[name], reference[name]
        numeric = pd.api.types.is_numeric_dtype(
            left
        ) and not pd.api.types.is_bool_dtype(left)
        if numeric:
            a, b = left.dropna(), right.dropna()
            if len(a) == 0 or len(b) == 0:
                rows.append((str(name), "numeric", 0.0, 1.0, 0.0, False))
                continue
            result = stats.ks_2samp(a, b)
            rows.append(
                (
                    str(name),
                    "numeric",
                    float(result.statistic),
                    float(result.pvalue),
                    float(result.statistic),
                    False,
                )
            )
        else:
            statistic, p, v, low = categorical_test(left, right)
            rows.append((str(name), "categorical", statistic, p, v, low))
    if not rows:
        return ()
    adjusted = stats.false_discovery_control([r[3] for r in rows], method="bh")
    return tuple(
        FeatureShift(
            column=name,
            kind=kind,
            statistic=statistic,
            p_value=p,
            adjusted_p_value=float(q),
            effect=effect,
            rejected=bool(q <= alpha),
            low_counts=low,
        )
        for (name, kind, statistic, p, effect, low), q in zip(
            rows, adjusted, strict=True
        )
    )


@dataclasses.dataclass(frozen=True, slots=True)
class ShiftResult:
    """The overall verdict of a comparison with a deployment sample.

    Attributes:
        status: `pass` (no shift detected, not proof of none), `fail`
            (the classifier test rejected) or `unverifiable`.
        reason: One sentence saying why.
        c2st: The classifier test, or None when unverifiable.
        mmd: The kernel test, or None when unverifiable.
        features: Per-column tests.
        agreement: Whether the two overall tests reached the same decision,
            or None when unverifiable.
    """

    status: Status
    reason: str
    c2st: qualification.Check | None
    mmd: qualification.Check | None
    features: tuple[FeatureShift, ...]
    agreement: bool | None

    def to_dict(self) -> dict[str, Any]:
        """Converts the record to a JSON-ready dictionary.

        Returns:
            The fields by name, with `status` as its string value, the
            checks as dictionaries (or None) and `features` as a list.
        """
        return {
            "status": self.status.value,
            "reason": self.reason,
            "c2st": None if self.c2st is None else self.c2st.to_dict(),
            "mmd": None if self.mmd is None else self.mmd.to_dict(),
            "agreement": self.agreement,
            "features": [f.to_dict() for f in self.features],
        }


def unverifiable(reason: str) -> ShiftResult:
    """Builds a result that says nothing could be checked, and why.

    Args:
        reason: One sentence saying why.

    Returns:
        An `unverifiable` result with no checks and no column tests.
    """
    return ShiftResult(Status.UNVERIFIABLE, reason, None, None, (), None)


def assess(
    sample: pd.DataFrame,
    reference: pd.DataFrame | None,
    seed: int,
    *,
    permutations: int = PERMUTATIONS,
    alpha: float = ALPHA,
    delta: float = 0.0,
    c2st_gate: certificates.GateStatus = certificates.GateStatus.DESCRIPTIVE,
    mmd_gate: certificates.GateStatus = certificates.GateStatus.DESCRIPTIVE,
) -> ShiftResult:
    """Compares exploration rows with a deployment sample.

    Without a reference the result is `unverifiable`, never `pass`. The status
    follows the classifier test; the kernel test is a second, independent
    statistic and their agreement is reported.

    Args:
        sample: Exploration rows.
        reference: Deployment sample, or None when the charter supplies none.
        seed: Run seed.
        permutations: Permutations for both tests.
        alpha: Level of the tests.
        delta: Minimum excess AUC worth acting on.
        c2st_gate: Gate status of the classifier test.
        mmd_gate: Gate status of the kernel test.

    Returns:
        The verdict with its checks. It is `unverifiable` when there is no
        reference, no shared column, differing columns or too few rows;
        otherwise `fail` when the classifier test fires and `pass` if not.
    """
    if reference is None:
        return unverifiable("no deployment sample was supplied")
    shared = [c for c in sample.columns if c in reference.columns]
    if not shared:
        return unverifiable("the reference shares no feature columns")
    left, right = sample[shared], reference[shared]
    try:
        check_pair(left, right)
    except ValueError as error:
        return unverifiable(str(error))
    first = c2st(
        left,
        right,
        seed,
        permutations=permutations,
        alpha=alpha,
        delta=delta,
        gate=c2st_gate,
    )
    second = mmd_test(
        left, right, seed, permutations=permutations, alpha=alpha, gate=mmd_gate
    )
    fired = first.decision is not qualification.Decision.PASS
    agree = (first.p_value is not None and first.p_value <= alpha) == (
        second.p_value is not None and second.p_value <= alpha
    )
    return ShiftResult(
        Status.FAIL if fired else Status.PASS,
        "the classifier can tell the two samples apart"
        if fired
        else "no shift detected (not evidence that none exists)",
        first,
        second,
        feature_tests(left, right, alpha),
        agree,
    )


def code_digest() -> str:
    """Hashes this module's source.

    Returns:
        The code hash that keys this module's certificates.
    """
    return certificates.code_hash(sys.modules[__name__])
