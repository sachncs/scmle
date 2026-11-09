"""Missingness: what the gaps say, and how far a conclusion survives them.

Mechanism is argued from evidence, never asserted:

* MCAR is tested with Little's chi-square test (own implementation; the
  project has no statsmodels dependency).
* MAR is probed with an indicator model: can the other columns predict that a
  value is missing? Out-of-fold AUC is compared with a permutation null.
* MNAR cannot be tested from the data. `tipping_point` reports the shift of
  the missing values at which a conclusion flips, over a declared range.
* The per-column screen against the target stays in
  `scmle.data.profiling.missingness` and is reused here unchanged.

References:
    Little, R. J. A. (1988). A test of missing completely at random for
    multivariate data with missing values. JASA 83(404), 1198-1202. The
    statistic and its degrees of freedom (sum of observed counts per pattern
    minus the number of variables) were checked against R package
    documentation, not the paper text (unverified against the paper itself).
    Rubin, D. B. (1976). Inference and missing data. Biometrika 63(3).
    Dempster, Laird and Rubin (1977), EM algorithm, JRSS-B 39(1).

Limits: Little's test assumes multivariate normality and low power is well
known; a non-rejection does not establish MCAR. The indicator model is
linear, so it misses nonlinear dependence. Rows beyond `MAX_ROWS` are
sampled and the sampling cap is recorded.
"""

from collections.abc import Callable
from collections.abc import Sequence
import dataclasses
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn import impute
from sklearn import model_selection

from scmle.core import randomness
from scmle.data.profiling import missingness as screen

MAX_ROWS = 5000
MAX_COLUMNS = 40
MAX_INDICATOR_COLUMNS = 8
MIN_ROWS = 30
MIN_MISSING = 5
EM_MAX_ITER = 500
EM_TOLERANCE = 1e-9
RIDGE = 1e-8
ALPHA = 0.01
PERMUTATIONS = 199
FOLDS = 5
STRATEGIES = ("mean", "median", "knn")
HOLDOUT_SHARE = 0.1


@dataclasses.dataclass(frozen=True)
class EmResult:
    """Maximum-likelihood mean and covariance under a normal model.

    Attributes:
        mean: Estimated mean vector.
        covariance: Estimated covariance (divisor n).
        iterations: EM iterations used.
        converged: Whether the change fell below the tolerance.
    """

    mean: np.ndarray
    covariance: np.ndarray
    iterations: int
    converged: bool


@dataclasses.dataclass(frozen=True)
class LittleResult:
    """Outcome of Little's MCAR test.

    Attributes:
        statistic: Chi-square statistic d2.
        df: Degrees of freedom, sum of observed counts minus variables.
        p_value: Upper-tail chi-square probability.
        patterns: Number of distinct missingness patterns used.
        rows: Rows used.
        columns: Columns used.
        converged: Whether EM converged.
    """

    statistic: float
    df: int
    p_value: float
    patterns: int
    rows: int
    columns: tuple[str, ...]
    converged: bool


@dataclasses.dataclass(frozen=True)
class IndicatorResult:
    """Can the other columns predict that this column is missing?

    Attributes:
        column: Column whose missing indicator was modelled.
        auc: Out-of-fold area under the ROC curve.
        p_value: Permutation p-value for AUC above its null.
        missing: Number of missing values.
    """

    column: str
    auc: float
    p_value: float
    missing: int


@dataclasses.dataclass(frozen=True)
class MissingnessReport:
    """Evidence about the missingness mechanism of a table.

    Attributes:
        rows_checked: Rows used after the cap.
        input_rows: Rows given.
        little: Little's test, or None when it was not run.
        indicators: One result per modelled column.
        mcar_rejected: Whether either test rejects MCAR after a split of
            `alpha` (half to Little, half Bonferroni over the indicators).
        alpha: Family level.
        skipped: Reasons something was not run ("skipped, over budget").
        tier: Always `unverified`: no provenance is supplied.
    """

    rows_checked: int
    input_rows: int
    little: LittleResult | None
    indicators: tuple[IndicatorResult, ...]
    mcar_rejected: bool
    alpha: float
    skipped: tuple[str, ...]
    tier: str = "unverified"

    def to_dict(self) -> dict[str, Any]:
        """Returns the report as a JSON-ready record.

        Returns:
            A mapping of every field to its value.
        """
        return dataclasses.asdict(self)


def numeric_columns(frame: pd.DataFrame) -> list[Any]:
    """Finds the numeric, non-boolean columns that are not constant.

    Args:
        frame: Any table.

    Returns:
        The column names, in table order.
    """
    names: list[Any] = []
    for name in frame.columns:
        column = frame[name]
        if not pd.api.types.is_numeric_dtype(column):
            continue
        if pd.api.types.is_bool_dtype(column):
            continue
        if column.nunique(dropna=True) < 2:
            continue
        names.append(name)
    return names


def pattern_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Counts of each missingness pattern, most frequent first.

    Args:
        frame: Any table.

    Returns:
        One row per pattern with boolean column flags and a `count`.
    """
    flags = frame.isna()
    counts = flags.value_counts().rename("count").reset_index()
    return counts.sort_values("count", ascending=False, kind="stable")


def em_normal(
    values: np.ndarray,
    max_iter: int = EM_MAX_ITER,
    tolerance: float = EM_TOLERANCE,
) -> EmResult:
    """EM estimate of the mean and covariance with missing entries.

    Standard EM for the multivariate normal (Dempster, Laird and Rubin
    1977): the E-step fills each pattern with its conditional mean and adds
    the conditional covariance; the M-step recomputes the moments.

    Args:
        values: Array of shape (n, p) with NaN for missing entries.
        max_iter: Iteration cap.
        tolerance: Stop when the largest change in the mean or covariance,
            relative to the covariance scale, is below this.

    Returns:
        The estimated mean and covariance, the iterations used and whether
        the tolerance was reached.

    Raises:
        ValueError: If a column is entirely missing or `values` is not 2-D.
    """
    if values.ndim != 2:
        raise ValueError("values must be two-dimensional")
    n, p = values.shape
    seen = ~np.isnan(values)
    if not seen.any(axis=0).all():
        raise ValueError("every column needs at least one observed value")
    mean = np.nanmean(values, axis=0)
    variance = np.nanvar(values, axis=0)
    covariance = np.diag(np.where(variance > 0, variance, 1.0))
    groups = row_patterns(seen)
    converged = False
    iterations = 0
    for step in range(1, max_iter + 1):
        iterations = step
        total = np.zeros(p)
        second = np.zeros((p, p))
        for observed, rows in groups:
            filled, extra = expect_block(
                values[np.ix_(rows, observed)], observed, mean, covariance, p
            )
            total += filled.sum(axis=0)
            second += filled.T @ filled + len(rows) * extra
        new_mean = total / n
        new_cov = second / n - np.outer(new_mean, new_mean)
        new_cov = (new_cov + new_cov.T) / 2.0 + RIDGE * np.eye(p)
        scale = max(float(np.abs(new_cov).max()), 1.0)
        change = max(
            float(np.abs(new_mean - mean).max()),
            float(np.abs(new_cov - covariance).max()),
        )
        mean, covariance = new_mean, new_cov
        if change <= tolerance * scale:
            converged = True
            break
    return EmResult(mean, covariance, iterations, converged)


def row_patterns(seen: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Groups row positions by which columns are observed.

    Args:
        seen: Boolean array (n, p), True where a value is observed.

    Returns:
        One pair per missingness pattern: the positions of the observed
        columns and the positions of the rows with that pattern.
    """
    keys: dict[bytes, list[int]] = {}
    for row, flags in enumerate(seen):
        keys.setdefault(flags.tobytes(), []).append(row)
    groups = []
    for rows in keys.values():
        observed = np.flatnonzero(seen[rows[0]])
        groups.append((observed, np.array(rows)))
    return groups


def expect_block(
    block: np.ndarray,
    observed: np.ndarray,
    mean: np.ndarray,
    covariance: np.ndarray,
    p: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Runs the E-step for rows sharing one missingness pattern.

    Args:
        block: The observed values of those rows, shape (rows, observed).
        observed: Positions of the observed columns.
        mean: Current mean estimate, length p.
        covariance: Current covariance estimate, shape (p, p).
        p: Number of columns.

    Returns:
        The rows with missing entries replaced by their conditional means,
        shape (rows, p), and the conditional covariance of the missing
        entries, shape (p, p), which is zero outside them.
    """
    rows = block.shape[0]
    filled = np.tile(mean, (rows, 1))
    extra = np.zeros((p, p))
    if len(observed) == 0:
        extra[:, :] = covariance
        return filled, extra
    filled[:, observed] = block
    hidden = np.setdiff1d(np.arange(p), observed)
    if len(hidden) == 0:
        return filled, extra
    s_oo = covariance[np.ix_(observed, observed)]
    s_ho = covariance[np.ix_(hidden, observed)]
    slope = np.linalg.solve(s_oo, s_ho.T).T
    filled[:, hidden] = mean[hidden] + (block - mean[observed]) @ slope.T
    extra[np.ix_(hidden, hidden)] = (
        covariance[np.ix_(hidden, hidden)] - slope @ s_ho.T
    )
    return filled, extra


def little_statistic(values: np.ndarray, fit: EmResult) -> tuple[float, int]:
    """Little's d2 and its degrees of freedom for given ML estimates.

    d2 = sum over patterns j of n_j (ybar_j - mu_j)' S_j^{-1} (ybar_j - mu_j),
    where mu_j and S_j are the ML mean and covariance restricted to the
    variables observed in pattern j. df = sum_j k_j - p.

    Args:
        values: Array (n, p) with NaN for missing entries.
        fit: ML mean and covariance of `values`.

    Returns:
        The statistic d2 and its degrees of freedom.
    """
    seen = ~np.isnan(values)
    p = values.shape[1]
    statistic = 0.0
    observed_total = 0
    for observed, rows in row_patterns(seen):
        if len(observed) == 0:
            continue
        gap = values[np.ix_(rows, observed)].mean(axis=0) - fit.mean[observed]
        s_oo = fit.covariance[np.ix_(observed, observed)]
        statistic += len(rows) * float(gap @ np.linalg.solve(s_oo, gap))
        observed_total += len(observed)
    return statistic, observed_total - p


def normal_scores(values: np.ndarray) -> np.ndarray:
    """Rank-based inverse-normal scores per column, NaN kept.

    A monotone transform of the observed values leaves the missingness
    mechanism unchanged, and makes the normal model of Little's test less
    sensitive to skew.

    Args:
        values: Array (n, p) with NaN for missing entries.

    Returns:
        An array of the same shape holding the inverse-normal score of each
        observed value's rank within its column, NaN where missing.
    """
    out = np.full_like(values, np.nan, dtype=float)
    for j in range(values.shape[1]):
        column = values[:, j]
        seen = ~np.isnan(column)
        ranks = stats.rankdata(column[seen])
        out[seen, j] = stats.norm.ppf((ranks - 0.5) / seen.sum())
    return out


def little_mcar_test(
    frame: pd.DataFrame,
    columns: Sequence[Any] | None = None,
    transform: str = "normal_scores",
) -> LittleResult | None:
    """Little's test of missing completely at random.

    Args:
        frame: Table.
        columns: Numeric columns to use; defaults to all non-constant,
            non-boolean numeric ones.
        transform: `normal_scores` or `none`.

    Returns:
        The result, or None when fewer than two columns are used, no value is
        missing (a non-finite value counts as missing, and rows with every
        value missing are dropped first), or the degrees of freedom are not
        positive.

    Raises:
        ValueError: On an unknown transform or non-numeric column.
    """
    if transform not in ("normal_scores", "none"):
        raise ValueError("transform must be 'normal_scores' or 'none'")
    names = list(columns) if columns is not None else numeric_columns(frame)
    for name in names:
        if not pd.api.types.is_numeric_dtype(frame[name]):
            raise ValueError(f"column '{name}' is not numeric")
    values = frame[names].to_numpy(dtype=float, copy=True)
    values[~np.isfinite(values)] = np.nan
    values = values[~np.isnan(values).all(axis=1)]
    if len(names) < 2 or not np.isnan(values).any():
        return None
    if transform == "normal_scores":
        values = normal_scores(values)
    fit = em_normal(values)
    statistic, df = little_statistic(values, fit)
    if df <= 0:
        return None
    return LittleResult(
        statistic=statistic,
        df=df,
        p_value=float(stats.chi2.sf(statistic, df)),
        patterns=len(row_patterns(~np.isnan(values))),
        rows=len(values),
        columns=tuple(str(n) for n in names),
        converged=fit.converged,
    )


def auc_from_ranks(scores: np.ndarray, positive: np.ndarray) -> float:
    """Computes the area under the ROC curve via the rank-sum identity.

    Args:
        scores: Score of each row.
        positive: Boolean per row, True for the positive class. Both classes
            must be present.

    Returns:
        The area under the curve, with ties given average ranks.
    """
    ranks = stats.rankdata(scores)
    n_pos = int(positive.sum())
    n_neg = len(positive) - n_pos
    total = float(ranks[positive].sum()) - n_pos * (n_pos + 1) / 2.0
    return total / (n_pos * n_neg)


def oof_linear_scores(
    design: np.ndarray, target: np.ndarray, folds: list[np.ndarray]
) -> np.ndarray:
    """Computes out-of-fold least-squares scores of a 0/1 target.

    Args:
        design: Design matrix, one row per observation.
        target: The 0/1 target as floats.
        folds: Row positions of each held-out fold.

    Returns:
        For each row, the prediction of a model fitted on the other folds.
    """
    scores = np.zeros(len(target))
    for held in folds:
        fit = np.setdiff1d(np.arange(len(target)), held)
        coefficients = np.linalg.lstsq(design[fit], target[fit], rcond=None)[0]
        scores[held] = design[held] @ coefficients
    return scores


def indicator_model_test(
    frame: pd.DataFrame,
    column: str,
    seed: int = 0,
    permutations: int = PERMUTATIONS,
    folds: int = FOLDS,
) -> IndicatorResult | None:
    """Tests whether other columns predict that `column` is missing.

    A least-squares linear probability model on the standardized, mean-filled
    other numeric columns gives out-of-fold scores; their AUC is compared
    with the AUC obtained after permuting the indicator and refitting with
    the same folds. The p-value is (1 + #{null >= observed}) / (B + 1).

    Args:
        frame: Table.
        column: Column to model.
        seed: Run seed for folds and permutations.
        permutations: Permutation count B, at least 1.
        folds: Cross-validation folds, at least 2.

    Returns:
        The result, or None when there are too few missing or observed
        values or no other numeric column.

    Raises:
        ValueError: If `permutations` is below 1 or `folds` is below 2.
    """
    if permutations < 1 or folds < 2:
        raise ValueError("need permutations >= 1 and folds >= 2")
    flag = frame[column].isna().to_numpy()
    others = [c for c in numeric_columns(frame) if c != column]
    missing = int(flag.sum())
    if not others or missing < MIN_MISSING or len(flag) - missing < MIN_ROWS:
        return None
    block = frame[others].to_numpy(dtype=float, copy=True)
    block[~np.isfinite(block)] = np.nan
    block = np.where(np.isnan(block), np.nanmean(block, axis=0), block)
    spread = block.std(axis=0)
    block = (block - block.mean(axis=0)) / np.where(spread > 0, spread, 1.0)
    design = np.column_stack([np.ones(len(block)), block])
    rng = randomness.component_rng(seed, f"detectors.indicator.{column}", 0)
    order = rng.permutation(len(flag))
    parts = [np.sort(part) for part in np.array_split(order, folds)]
    target = flag.astype(float)
    observed = auc_from_ranks(oof_linear_scores(design, target, parts), flag)
    beaten = 0
    shuffles = [rng.permutation(flag) for count in range(permutations)]
    for shuffled in shuffles:
        null = auc_from_ranks(
            oof_linear_scores(design, shuffled.astype(float), parts), shuffled
        )
        beaten += int(null >= observed)
    return IndicatorResult(
        column=str(column),
        auc=float(observed),
        p_value=(1 + beaten) / (permutations + 1),
        missing=missing,
    )


def assess_mechanism(
    frame: pd.DataFrame,
    seed: int = 0,
    alpha: float = ALPHA,
    permutations: int = PERMUTATIONS,
) -> MissingnessReport:
    """Collects the evidence about the missingness mechanism.

    Args:
        frame: Exploration rows only.
        seed: Run seed.
        alpha: Family level, in (0, 1).
        permutations: Permutations per indicator model.

    Returns:
        The report. `mcar_rejected` is True when Little's test rejects at
        `alpha / 2` or any indicator model rejects at `alpha / 2` divided by
        the number of models. Anything not run is listed in `skipped`.

    Raises:
        ValueError: If `alpha` is not strictly between 0 and 1.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be within (0, 1)")
    skipped: list[str] = []
    input_rows = len(frame)
    if input_rows > MAX_ROWS:
        rng = randomness.component_rng(seed, "detectors.missingness.cap", 0)
        keep = np.sort(rng.choice(input_rows, MAX_ROWS, replace=False))
        frame = frame.iloc[keep]
        skipped.append(f"rows sampled to {MAX_ROWS} of {input_rows}")
    numeric = numeric_columns(frame)
    if len(numeric) > MAX_COLUMNS:
        skipped.append(
            f"skipped, over budget: {len(numeric)} numeric columns exceed "
            f"{MAX_COLUMNS}"
        )
        numeric = []
    little = little_mcar_test(frame, numeric) if numeric else None
    gaps = sorted(
        (c for c in numeric if frame[c].isna().any()),
        key=lambda c: (-int(frame[c].isna().to_numpy().sum()), c),
    )
    if len(gaps) > MAX_INDICATOR_COLUMNS:
        skipped.append(
            f"skipped, over budget: indicator models for "
            f"{len(gaps) - MAX_INDICATOR_COLUMNS} columns"
        )
        gaps = gaps[:MAX_INDICATOR_COLUMNS]
    results = []
    for name in gaps:
        found = indicator_model_test(frame, name, seed, permutations)
        if found is not None:
            results.append(found)
    rejected = bool(
        (little is not None and little.p_value < alpha / 2.0)
        or any(r.p_value < alpha / 2.0 / len(results) for r in results)
    )
    return MissingnessReport(
        rows_checked=len(frame),
        input_rows=input_rows,
        little=little,
        indicators=tuple(results),
        mcar_rejected=rejected,
        alpha=alpha,
        skipped=tuple(skipped),
    )


def informative_columns(
    frame: pd.DataFrame,
    columns: list[str],
    target: pd.Series,
    classification: bool,
) -> list[str]:
    """Finds the columns whose gaps are associated with the target.

    This is the cheap screen. It delegates to
    `scmle.data.profiling.missingness.assess`, whose thresholds stay the
    policy defaults so behaviour is unchanged.

    Args:
        frame: Table.
        columns: Columns to screen.
        target: The target values.
        classification: Whether the target is categorical.

    Returns:
        The names of the columns the screen finds informative.
    """
    findings = screen.assess(frame, columns, target, classification)
    return [f.column for f in findings if f.informative]


@dataclasses.dataclass(frozen=True)
class TippingPoint:
    """Where a conclusion flips when missing values are shifted by delta.

    Attributes:
        low: Lower end of the declared delta range.
        high: Upper end of the declared delta range.
        flip_delta: Smallest-magnitude delta in the range at which the
            conclusion differs from the delta = 0 conclusion, or None.
        inside_range: Whether a flip exists inside the range.
        baseline: The conclusion at delta = 0.
        steps: Grid size used.
    """

    low: float
    high: float
    flip_delta: float | None
    inside_range: bool
    baseline: bool
    steps: int


def tipping_point(
    observed: np.ndarray,
    n_missing: int,
    conclusion: Callable[[np.ndarray], bool],
    low: float,
    high: float,
    steps: int = 201,
) -> TippingPoint:
    """Tipping-point (delta-adjustment) sensitivity analysis.

    Missing values are imputed at the observed mean and shifted by delta;
    the conclusion is re-evaluated on the completed data over a grid of
    deltas spanning the charter-declared range. MNAR is untestable, so this
    reports sensitivity, never a mechanism.

    Args:
        observed: Observed values of the column.
        n_missing: Number of missing values, at least 1.
        conclusion: Maps the completed values to a boolean conclusion.
        low: Lower end of the declared range (<= 0 <= high not required).
        high: Upper end of the declared range, above `low`.
        steps: Grid points over `[low, high]`, at least 2.

    Returns:
        The tipping point.

    Raises:
        ValueError: On an empty sample, bad range or too few steps.
    """
    values = np.asarray(observed, dtype=float)
    if values.size == 0 or n_missing < 1:
        raise ValueError("need observed values and at least one missing")
    if not low < high or steps < 2:
        raise ValueError("need low < high and steps >= 2")
    centre = float(values.mean())
    filler = np.full(n_missing, centre)
    baseline = bool(conclusion(np.concatenate([values, filler])))
    grid = np.linspace(low, high, steps)
    flips = [
        float(d)
        for d in grid
        if bool(conclusion(np.concatenate([values, filler + d]))) != baseline
    ]
    flip = min(flips, key=abs) if flips else None
    return TippingPoint(low, high, flip, flip is not None, baseline, steps)


def mean_tipping_delta(
    observed: np.ndarray, n_missing: int, null_value: float
) -> float:
    """Computes the delta at which the completed mean equals `null_value`.

    With the missing values imputed at the observed mean m and shifted by
    delta, the completed mean is (sum + k (m + delta)) / (n + k); setting it
    to `null_value` gives delta = null_value (n + k) / k - sum / k - m.

    Args:
        observed: Observed values of the column.
        n_missing: Number of missing values `k`, at least 1.
        null_value: The value of the mean at which the conclusion flips.

    Returns:
        The shift of the imputed values at which the completed mean equals
        `null_value`.
    """
    values = np.asarray(observed, dtype=float)
    n, k = values.size, n_missing
    centre = float(values.mean())
    return float(null_value * (n + k) / k - values.sum() / k - centre)


def draw_holdout(
    train: np.ndarray, seed: int, fold: int, share: float
) -> tuple[np.ndarray, np.ndarray]:
    """Picks observed cells of `train` to hide for scoring.

    Args:
        train: Array (n, p) with NaN for gaps.
        seed: Run seed.
        fold: Fold number, which varies the draw.
        share: Probability that each observed cell is chosen.

    Returns:
        The row positions and the column positions of the chosen cells.
    """
    rng = randomness.component_rng(seed, "detectors.imputation.holdout", fold)
    rows, cols = np.nonzero(~np.isnan(train))
    chosen = rng.random(len(rows)) < share
    return rows[chosen], cols[chosen]


def make_imputer(strategy: str) -> Any:
    """Builds an unfitted imputer for a strategy name.

    Args:
        strategy: One of `STRATEGIES`.

    Returns:
        A scikit-learn imputer that keeps all-missing columns.

    Raises:
        ValueError: If `strategy` is not one of `STRATEGIES`.
    """
    if strategy in ("mean", "median"):
        return impute.SimpleImputer(strategy=strategy, keep_empty_features=True)
    if strategy == "knn":
        return impute.KNNImputer(n_neighbors=5, keep_empty_features=True)
    raise ValueError(f"strategy must be one of {STRATEGIES}")


def fold_imputers(
    values: np.ndarray, strategy: str, seed: int, folds: int = FOLDS
) -> list[tuple[np.ndarray, np.ndarray, Any]]:
    """Fits one imputer per fold on that fold's training rows only.

    Args:
        values: Array (n, p) with NaN for gaps.
        strategy: One of `STRATEGIES`.
        seed: Run seed for the fold split.
        folds: Number of folds, at least 2.

    Returns:
        (train rows, held-out rows, fitted imputer) per fold. The fitted
        state depends only on the train rows.
    """
    splitter = model_selection.KFold(
        folds,
        shuffle=True,
        random_state=randomness.derive_int_seed(seed, "detectors.folds", 0),
    )
    return [
        (train, held, fit_one(values, strategy, train))
        for train, held in splitter.split(values)
    ]


def fit_one(values: np.ndarray, strategy: str, train: np.ndarray) -> Any:
    """Fits an imputer on the rows `train` of `values` and nothing else.

    Args:
        values: Array (n, p) with NaN for gaps.
        strategy: One of `STRATEGIES`.
        train: Positions of the rows to fit on.

    Returns:
        The fitted imputer.
    """
    imputer = make_imputer(strategy)
    imputer.fit(values[train])
    return imputer


def imputation_errors(
    values: np.ndarray,
    seed: int = 0,
    folds: int = FOLDS,
    share: float = HOLDOUT_SHARE,
) -> dict[str, float]:
    """Measures held-out imputation error per strategy, fitted in the folds.

    Within each fold a share of observed cells of the held-out rows is
    hidden; the imputer, fitted on the training rows, fills them; the error
    is the RMSE in units of the column standard deviation.

    Args:
        values: Array (n, p) with NaN for gaps.
        seed: Run seed.
        folds: Folds, at least 2.
        share: Share of held-out observed cells to hide, in (0, 1).

    Returns:
        Strategy name -> mean standardized RMSE over folds (NaN when no
        cell could be hidden in any fold).

    Raises:
        ValueError: If `share` is not strictly between 0 and 1.
    """
    if not 0.0 < share < 1.0:
        raise ValueError("share must be within (0, 1)")
    spread = np.nanstd(values, axis=0)
    spread = np.where(spread > 0, spread, 1.0)
    errors: dict[str, float] = {}
    for strategy in STRATEGIES:
        scores = []
        for index, fold in enumerate(
            fold_imputers(values, strategy, seed, folds)
        ):
            held, imputer = fold[1], fold[2]
            block = values[held].copy()
            rows, cols = draw_holdout(block, seed, index, share)
            if len(rows) == 0:
                continue
            truth = block[rows, cols].copy()
            block[rows, cols] = np.nan
            filled = imputer.transform(block)
            diff = (filled[rows, cols] - truth) / spread[cols]
            scores.append(float(np.sqrt(np.mean(diff**2))))
        errors[strategy] = float(np.mean(scores)) if scores else float("nan")
    return errors


def select_imputer(
    values: np.ndarray, seed: int = 0, folds: int = FOLDS
) -> str:
    """Selects the imputation strategy with the smallest held-out error.

    Args:
        values: Array (n, p) with NaN for gaps.
        seed: Run seed.
        folds: Folds, at least 2.

    Returns:
        The name of the strategy; ties go to the one listed first in
        `STRATEGIES`.

    Raises:
        ValueError: If no strategy has a finite error.
    """
    errors = imputation_errors(values, seed, folds)
    finite = {k: v for k, v in errors.items() if np.isfinite(v)}
    if not finite:
        raise ValueError("no strategy could be scored")
    return min(STRATEGIES, key=lambda s: finite.get(s, float("inf")))
