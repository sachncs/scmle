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
3. Multiplicity. All `m` p-values (skill tests and missingness tests) form one
   family adjusted by Benjamini and Yekutieli (2001, Ann. Statist. 29:1165)
   at level `alpha`, in exact arithmetic (`scmle.data.ledger.multiplicity`).
   The number of permutations is raised until the smallest reachable p-value
   `1 / (B + 1)` can pass the rank-1 threshold `alpha / (m * H_m)`; if that
   needs more than `MAX_PERMUTATIONS` the screen says so and flags nothing.
   A column whose p-value can no longer fall below the largest threshold
   `alpha / H_m` stops early; the step-up decisions are identical to those of
   the full run, because such a p-value can never be rejected and sorts last.
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
of the significant list. Gating follows the certificate rule: only
a mechanism whose leak types have a current, passing certificate
(`screen_certificates.json`, `benchmarks.qualification.screen_certify`) may
block a run; anything else is reported as a warning with its p-values.

What is not guaranteed
----------------------
The screen cannot distinguish a legitimate strong predictor from a leak
without provenance: a suspect is a suspicion, never proof. Bootstrap bounds on
cross-fitted scores are approximate (folds share training rows), so no exact
coverage is claimed. Rows are subsampled to `MAX_ROWS`, which costs power for
effects below the ceiling, not size. The Fisher and chi-square p-values used
without groups or time blocks assume independent rows. Skill is measured one
column at a time with a smooth basis, so a leak that needs several columns
together is not seen. Candidate columns of kind datetime or text are not
screened. Wide tables pay in power: the more columns, the more permutations a
rejection needs. A run without a charter has no custody, so the training
rows are all the screen has; it says nothing about rows it never saw.
"""

from collections.abc import Callable
from collections.abc import Hashable
from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
from fractions import Fraction
import functools
import math
from pathlib import Path
import sys
from typing import Any

import numpy
import pandas
from scipy import stats

from scmle.core import randomness
from scmle.data.admissibility import leakage
from scmle.data.detectors import certificates
from scmle.data.ledger import budget
from scmle.data.ledger import multiplicity
import scmle.data.ledger.budget

DETECTOR_ID = "f4-leakage-screen"
SKILL = "skill"
MISSING = "missingness"
PURITY = "record_purity"
MECHANISM_DEFECTS: Mapping[str, tuple[str, ...]] = {
    SKILL: ("leak_copy", "leak_noisy_copy", "leak_threshold"),
    MISSING: ("leak_missing_for_class",),
    PURITY: ("leak_rare_record",),
}
STORE_PATH = Path(__file__).with_name("screen_certificates.json")
KNOTS = 10
MAX_ROWS = 2000
MAX_PERMUTATIONS = 100_000
BINARY_CLASSES = 2
MIN_CLASSES = 2
# Cross-fitted fits needed by the permutation test: permutations x skill
# columns x classes (one-vs-rest when there are more than two). The screen
# runs inside the declared time budget by a deterministic cap, never by a
# clock: work_cap = time_budget_seconds * SCREEN_SHARE * FITS_PER_SECOND.
# Measured on the development machine (2026-11-06): a breast-cancer sized
# table, 72,000 fits, takes 1.4 s, i.e. about 51,000 fits per second.
# FITS_PER_SECOND is declared at about half of that so a slower machine still
# finishes inside the share. A cap from the clock would make the flags differ
# between machines; this one does not. The default budget of 1800 s allows
# 6.75 million fits, the 120 s budget of the CI benchmark 450,000.
SCREEN_SHARE = 0.15
FITS_PER_SECOND = 25_000
BATCH = 200
TIME_BLOCKS = 20
RIDGE = 1e-3
TOLERANCE = 1e-12

WORDS = {
    SKILL: "cross-fitted skill",
    MISSING: "association of being recorded with the target (Cramer's V)",
    PURITY: "share of one target class among the recorded values",
}
Statistic = Callable[[numpy.ndarray], numpy.ndarray]


def basis(series: pandas.Series, knots: int = KNOTS) -> numpy.ndarray:
    """Builds the design matrix of one column for the cross-fitted smoother.

    A numeric column with more than `knots` distinct values becomes a
    continuous piecewise-linear basis (hat functions) on its distinct
    quantiles, so a smooth or linear dependence is representable, plus an
    indicator for missing values. A column with few values, or any other
    column, gets one indicator per level (the `leakage.MAX_LEVELS` most
    frequent, the rest pooled) and one for missing values. Empty indicator
    columns are dropped, so the result may have a single column (a constant
    column), which carries no information.

    Args:
        series: One candidate column.
        knots: Number of quantile intervals of the piecewise-linear basis; a
            numeric column with at most this many distinct values gets
            indicators instead.

    Returns:
        A float64 array with one row per entry of `series` and one column
        per retained basis function or indicator.
    """
    numeric = pandas.api.types.is_numeric_dtype(
        series
    ) and not pandas.api.types.is_bool_dtype(series)
    if numeric:
        values = series.to_numpy(dtype="float64")
        absent = numpy.isnan(values)
        finite = values[~absent]
        if len(numpy.unique(finite)) > knots:
            cuts = numpy.unique(
                numpy.quantile(finite, numpy.linspace(0, 1, knots + 1))
            )
            where = numpy.clip(
                numpy.searchsorted(cuts, values, "right") - 1, 0, None
            )
            where = numpy.minimum(where, len(cuts) - 2)
            width = cuts[where + 1] - cuts[where]
            weight = numpy.where(absent, 0.0, (values - cuts[where]) / width)
            weight = numpy.clip(weight, 0.0, 1.0)
            phi = numpy.zeros((len(values), len(cuts) + 1))
            rows = numpy.arange(len(values))
            live = ~absent
            phi[rows[live], where[live]] = 1.0 - weight[live]
            phi[rows[live], where[live] + 1] = weight[live]
            phi[absent, -1] = 1.0
            return phi[:, phi.any(axis=0)]
        raw = numpy.where(absent, numpy.nan, values)
        labels = pandas.Series(raw).astype("object")
    else:
        labels = series.reset_index(drop=True).astype("object")
    labels = labels.where(labels.notna(), "<missing>").astype(str)
    top = labels.value_counts().index[: leakage.MAX_LEVELS]
    pooled = labels.where(labels.isin(top), "<other>")
    return pandas.get_dummies(pooled).to_numpy(dtype="float64")


def strata_of(
    n: int, groups: numpy.ndarray | None, block_size: int | None
) -> numpy.ndarray:
    """Assigns each row its permutation stratum: group, time block, or one.

    Args:
        n: Number of rows.
        groups: Group label per row, or None. Takes precedence.
        block_size: Consecutive rows per time block, or None.

    Returns:
        An int64 stratum label per row. All zeros when neither `groups` nor
        `block_size` is given.
    """
    if groups is not None:
        return pandas.factorize(pandas.Series(groups).astype(str))[0].astype(
            "int64"
        )
    if block_size is not None:
        return (numpy.arange(n) // block_size).astype("int64")
    return numpy.zeros(n, dtype="int64")


def batch_permutations(
    strata: numpy.ndarray, count: int, rng: numpy.random.Generator
) -> numpy.ndarray:
    """Draws `count` permutations of the rows, each within its stratum.

    Entry `p[b, i]` of the result is the position whose outcome row `i`
    receives in permutation `b`, so `y[p[b]]` is the outcome permuted within
    strata.

    Args:
        strata: Stratum label per row.
        count: Number of permutations to draw.
        rng: Random generator.

    Returns:
        An integer array of shape (count, n), one permutation per row.
    """
    n = len(strata)
    base = numpy.argsort(strata, kind="stable")
    keys = rng.random((count, n)) + strata[None, :]
    order = numpy.argsort(keys, axis=1)
    out = numpy.empty_like(order)
    out[:, base] = order
    return out


def fold_splits(
    n: int,
    folds: int,
    groups: numpy.ndarray | None,
    block_size: int | None,
    seed: int,
) -> leakage.Splits:
    """Cross-fitting folds that do not depend on the outcome.

    Folds fixed before permuting keep the observed statistic and the permuted
    ones the same function of the data, which the exact p-value needs.

    Args:
        n: Number of rows.
        folds: Requested number of folds.
        groups: Group label per row, or None.
        block_size: Consecutive rows per time block, or None.
        seed: Seed of the fold shuffling.

    Returns:
        A list of (train positions, test positions) pairs, one per fold.
    """
    return leakage.make_splits(
        numpy.zeros(n), "regression", folds, groups, block_size, seed
    )


def oof_fit(
    values: numpy.ndarray, phi: numpy.ndarray, splits: leakage.Splits
) -> numpy.ndarray:
    """Out-of-fold ridge fits of each row of `values` on the design `phi`.

    The fit is linear in `values` for fixed `phi` and folds, so a batch of
    permuted outcomes is fitted with matrix products. It is centred at the
    training mean and ridge-penalised by `RIDGE`, which keeps a level seen in
    few training rows near the mean.

    Args:
        values: Array of shape (B, n): B versions of a 0/1 indicator or of a
            continuous outcome.
        phi: Design of shape (n, p).
        splits: Cross-fitting (train, test) positions.

    Returns:
        Array of shape (B, n) of predictions for rows of each held-out fold;
        a row in no test fold is left uninitialised, so the folds must cover
        every row.
    """
    n, p = phi.shape
    total_cross = values @ phi
    total_sum = values.sum(axis=1)
    total_gram = phi.T @ phi
    total_cols = phi.sum(axis=0)
    out = numpy.empty_like(values)
    for split in splits:
        test = split[1]
        part = values[:, test]
        design = phi[test]
        size = n - len(test)
        mean = (total_sum - part.sum(axis=1)) / size
        gram = total_gram - design.T @ design + RIDGE * numpy.eye(p)
        cols = total_cols - design.sum(axis=0)
        cross = total_cross - part @ design - mean[:, None] * cols[None, :]
        beta = numpy.linalg.solve(gram, cross.T).T
        out[:, test] = mean[:, None] + beta @ design.T
    return out


def fold_concordance(
    score: numpy.ndarray,
    positive: numpy.ndarray,
    tests: Sequence[numpy.ndarray],
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Counts concordant pairs and pairs, per row, within folds only.

    Pairs are formed inside each held-out fold, never across folds. Across
    folds a row's prediction was made without its own label but with the
    labels of the rows of the other fold, which orders same-level rows of
    different folds against their labels and biases a pooled AUC downward.
    Ties count one half.

    Args:
        score: Array of shape (B, n): B versions of the out-of-fold scores.
        positive: Boolean array of shape (B, n) marking the positive class
            in each version.
        tests: Held-out row positions of each fold.

    Returns:
        A pair of float arrays of length B: for each version, the number of
        concordant (positive, negative) pairs, ties counting one half, and
        the number of such pairs, summed over the folds.
    """
    wins = numpy.zeros(score.shape[0])
    pairs = numpy.zeros(score.shape[0])
    for test in tests:
        flags = positive[:, test]
        n1 = flags.sum(axis=1)
        n0 = len(test) - n1
        ranks = stats.rankdata(score[:, test], axis=1)
        wins += (ranks * flags).sum(axis=1) - n1 * (n1 + 1) / 2.0
        pairs += n1 * n0
    return wins, pairs


def auc_rows(
    score: numpy.ndarray,
    positive: numpy.ndarray,
    tests: Sequence[numpy.ndarray],
) -> numpy.ndarray:
    """Computes the within-fold ROC AUC of each version of the scores.

    Args:
        score: Array of shape (B, n) of out-of-fold scores.
        positive: Boolean array of shape (B, n) marking the positive class.
        tests: Held-out row positions of each fold.

    Returns:
        A length-B array of AUC values; NaN where no fold has both classes.
    """
    wins, pairs = fold_concordance(score, positive, tests)
    with numpy.errstate(invalid="ignore", divide="ignore"):
        return wins / pairs


@dataclasses.dataclass(frozen=True, slots=True)
class Problem:
    """The outcome, task and folds shared by every column's test.

    Attributes:
        y: Class codes or floats.
        task: `classification` or `regression`.
        classes: Number of classes (0 for regression).
        splits: Cross-fitting folds.
        total_squares: Total sum of squares of a continuous outcome (0.0 for
            a classification).
    """

    y: numpy.ndarray
    task: str
    classes: int
    splits: leakage.Splits
    total_squares: float


def skill_statistic(problem: Problem, phi: numpy.ndarray) -> Statistic:
    """Builds the skill statistic of a column for batches of permuted outcomes.

    Args:
        problem: The outcome, task and folds.
        phi: Design of the column, of shape (n, p).

    Returns:
        A function mapping an array of shape (B, n) of outcome versions to
        a length-B array of skills: R^2 for regression, otherwise the
        within-fold AUC of class 1 when binary or its macro one-vs-rest
        average.
    """
    tests = [split[1] for split in problem.splits]

    def compute(labels: numpy.ndarray) -> numpy.ndarray:
        if problem.task == "regression":
            pred = oof_fit(labels, phi, problem.splits)
            squares = ((labels - pred) ** 2).sum(axis=1)
            return 1.0 - squares / problem.total_squares
        wanted = (
            [1] if problem.classes == BINARY_CLASSES else range(problem.classes)
        )
        scores = []
        for k in wanted:
            positive = labels == k
            pred = oof_fit(positive.astype("float64"), phi, problem.splits)
            scores.append(auc_rows(pred, positive, tests))
        return numpy.mean(scores, axis=0)

    return compute


def observed_predictions(
    problem: Problem, phi: numpy.ndarray
) -> list[numpy.ndarray]:
    """Out-of-fold predictions of the observed outcome.

    One array per scored target: the outcome for a regression, the indicator
    of each class for a classification (class 1 only when binary).

    Args:
        problem: The outcome, task and folds.
        phi: Design of the column, of shape (n, p).

    Returns:
        A list of arrays of length n, one per scored target.
    """
    y = problem.y
    if problem.task == "regression":
        return [oof_fit(y[None, :], phi, problem.splits)[0]]
    wanted = (
        [1] if problem.classes == BINARY_CLASSES else range(problem.classes)
    )
    return [
        oof_fit((y == k)[None, :].astype("float64"), phi, problem.splits)[0]
        for k in wanted
    ]


def sample_skill(
    problem: Problem,
    predictions: Sequence[numpy.ndarray],
    index: numpy.ndarray,
    fold_of: numpy.ndarray,
) -> float | None:
    """Computes the skill on a bootstrap sample of rows.

    Args:
        problem: The outcome, task and folds.
        predictions: Output of `observed_predictions`.
        index: Positions of the sampled rows, with repeats.
        fold_of: Fold number of each row.

    Returns:
        The skill on the sample, or None when it cannot be computed: a
        constant regression sample, or a classification sample in which, for
        some scored class, no fold holds both a positive and a negative row.
    """
    y = problem.y[index]
    if problem.task == "regression":
        total = float(((y - y.mean()) ** 2).sum())
        if total == 0.0:
            return None
        return 1.0 - float(((y - predictions[0][index]) ** 2).sum()) / total
    folds = fold_of[index]
    tests = [numpy.flatnonzero(folds == f) for f in numpy.unique(folds)]
    wanted = (
        [1] if problem.classes == BINARY_CLASSES else range(problem.classes)
    )
    scores = []
    for pred, k in zip(predictions, wanted, strict=True):
        wins, pairs = fold_concordance(
            pred[index][None, :], (y == k)[None, :], tests
        )
        if pairs[0] == 0:
            return None
        scores.append(wins[0] / pairs[0])
    return float(numpy.mean(scores))


def skill_bound(
    problem: Problem,
    phi: numpy.ndarray,
    cfg: leakage.Config,
    groups: numpy.ndarray | None,
    rng: numpy.random.Generator,
) -> float:
    """One-sided lower confidence bound of the skill by bootstrap.

    Rows are resampled, or whole groups when `groups` is given, and the
    out-of-fold predictions stay attached to their rows and folds. Draws whose
    skill cannot be computed (see `sample_skill`) are skipped; with none left
    the bound is -inf.

    Args:
        problem: The outcome, task and folds.
        phi: Design of the column, of shape (n, p).
        cfg: Settings (`bootstrap`, `confidence`).
        groups: Group label per row, or None.
        rng: Random generator.

    Returns:
        The `1 - confidence` quantile of the bootstrap skills, or -inf.
    """
    predictions = observed_predictions(problem, phi)
    fold_of = numpy.empty(len(problem.y), dtype="int64")
    for number, split in enumerate(problem.splits):
        fold_of[split[1]] = number
    members = (
        [numpy.flatnonzero(groups == g) for g in pandas.unique(groups)]
        if groups is not None
        else None
    )
    n = len(problem.y)
    values = []
    for draw in range(cfg.bootstrap):
        del draw
        if members is not None:
            picks = rng.integers(0, len(members), size=len(members))
            index = numpy.concatenate([members[i] for i in picks])
        else:
            index = rng.integers(0, n, size=n)
        value = sample_skill(problem, predictions, index, fold_of)
        if value is not None:
            values.append(value)
    if not values:
        return -math.inf
    return float(numpy.quantile(values, 1.0 - cfg.confidence))


def chi_square_rows(
    missing: numpy.ndarray, labels: numpy.ndarray, totals: numpy.ndarray
) -> numpy.ndarray:
    """Computes the Pearson chi-square of the 2 x K (missing by class) table.

    Both the missing and the present rows, and every class, must be
    non-empty; otherwise a zero expected count gives a non-finite value.

    Args:
        missing: Boolean array of length n, True where the value is missing.
        labels: Array of shape (B, n) of class codes, B versions of the
            outcome.
        totals: Number of rows of each class.

    Returns:
        A length-B array with the chi-square of each version.
    """
    n = labels.shape[1]
    n_missing = float(missing.sum())
    n_present = n - n_missing
    flags = missing.astype("float64")
    chi = numpy.zeros(labels.shape[0])
    for k, total in enumerate(totals):
        a = (labels == k) @ flags
        chi += (a - n_missing * total / n) ** 2 / (n_missing * total / n)
        chi += ((total - a) - n_present * total / n) ** 2 / (
            n_present * total / n
        )
    return chi


def missingness_statistic(
    missing: numpy.ndarray, totals: numpy.ndarray
) -> Statistic:
    """Builds the Cramer's V statistic of missingness against the outcome.

    Args:
        missing: Boolean array of length n, True where the value is missing.
        totals: Number of rows of each class.

    Returns:
        A function mapping an array of shape (B, n) of class-code versions
        to a length-B array of V values.
    """
    n = len(missing)

    def compute(labels: numpy.ndarray) -> numpy.ndarray:
        return numpy.sqrt(chi_square_rows(missing, labels, totals) / n)

    return compute


def purity_statistic(recorded: numpy.ndarray) -> Statistic:
    """Builds the record-purity statistic.

    Args:
        recorded: Boolean array of length n, True where the value is
            recorded.

    Returns:
        A function mapping an array of shape (B, n) of class-code versions
        to a length-B array holding, for each, the share of the commonest
        class among the recorded rows.
    """
    flags = recorded.astype("float64")
    count = float(flags.sum())

    def compute(labels: numpy.ndarray) -> numpy.ndarray:
        k = int(labels.max()) + 1
        shares = [(labels == c) @ flags / count for c in range(k)]
        return numpy.max(shares, axis=0)

    return compute


def purity_bound(
    recorded: numpy.ndarray,
    classes: numpy.ndarray,
    cfg: leakage.Config,
    groups: numpy.ndarray | None,
) -> float:
    """One-sided Clopper-Pearson lower bound of the commonest class's share.

    The share is taken among recorded rows. With groups the number of trials
    is the number of groups holding a recorded row (at most the rows), the
    conservative count when rows of one group move together.

    Args:
        recorded: Boolean array, True where the value is recorded.
        classes: Class code per row.
        cfg: Settings (`confidence`).
        groups: Group label per row, or None.

    Returns:
        The lower bound of the share, or 0.0 when the rounded count of
        successes is zero.
    """
    labels = classes[recorded]
    counts = numpy.bincount(labels)
    share = float(numpy.max(counts)) / len(labels)
    trials = len(labels)
    if groups is not None:
        trials = min(trials, len(pandas.unique(groups[recorded])))
    wins = round(share * trials)
    if wins == 0:
        return 0.0
    return float(stats.beta.ppf(1.0 - cfg.confidence, wins, trials - wins + 1))


def bootstrap_v(
    missing: numpy.ndarray,
    classes: numpy.ndarray,
    n_classes: int,
    cfg: leakage.Config,
    groups: numpy.ndarray | None,
    rng: numpy.random.Generator,
) -> float:
    """One-sided lower confidence bound of Cramer's V by bootstrap.

    Rows are resampled, or whole groups when `groups` is given. Draws that
    lose the missing rows, the present rows or a whole class are skipped;
    with none left the bound is -inf.

    Args:
        missing: Boolean array, True where the value is missing.
        classes: Class code per row.
        n_classes: Number of classes.
        cfg: Settings (`bootstrap`, `confidence`).
        groups: Group label per row, or None.
        rng: Random generator.

    Returns:
        The `1 - confidence` quantile of the bootstrap values of V, or
        -inf.
    """
    members = (
        [numpy.flatnonzero(groups == g) for g in pandas.unique(groups)]
        if groups is not None
        else None
    )
    n = len(missing)
    values = []
    for draw in range(cfg.bootstrap):
        del draw
        if members is not None:
            picks = rng.integers(0, len(members), size=len(members))
            index = numpy.concatenate([members[i] for i in picks])
        else:
            index = rng.integers(0, n, size=n)
        sample = missing[index]
        if sample.all() or not sample.any():
            continue
        labels = classes[index]
        totals = numpy.bincount(labels, minlength=n_classes).astype("float64")
        if (totals == 0).any():
            continue
        chi = chi_square_rows(sample, labels[None, :], totals)[0]
        values.append(math.sqrt(chi / len(sample)))
    if not values:
        return -math.inf
    return float(numpy.quantile(values, 1.0 - cfg.confidence))


@dataclasses.dataclass(slots=True)
class Pending:
    """One hypothesis while the shared permutations are drawn.

    Attributes:
        feature: Column name.
        mechanism: `skill`, `missingness` or `record_purity`.
        observed: Observed statistic.
        statistic: Function of a batch of permuted labels, or None when the
            p-value is analytic.
        uses_binned: Whether the statistic reads the binned outcome.
        p_value: Analytic p-value, when there is no statistic.
        exceed: Permutations with a statistic at least the observed one.
        drawn: Permutations drawn for it.
        active: Whether it still draws permutations.
    """

    feature: str
    mechanism: str
    observed: float
    statistic: Statistic | None
    uses_binned: bool
    p_value: float | None = None
    exceed: int = 0
    drawn: int = 0
    active: bool = True


def required_permutations(family: int, alpha: Fraction) -> int:
    """Finds the smallest `B` with `1 / (B + 1) <= alpha / (family * H_family)`.

    Args:
        family: Number of hypotheses adjusted together.
        alpha: Level of the Benjamini-Yekutieli step.

    Returns:
        The number of permutations `B`; 0 when `family` is not positive.
    """
    if family <= 0:
        return 0
    need = family * multiplicity.harmonic_exact(family) / alpha
    return max(0, math.ceil(need) - 1)


def draw_permutations(
    pending: Sequence[Pending],
    labels: numpy.ndarray,
    binned: numpy.ndarray,
    strata: numpy.ndarray,
    total: int,
    alpha: Fraction,
    seed: int,
) -> None:
    """Draws `total` shared permutations and updates every hypothesis in place.

    A hypothesis stops once its final p-value is certain to exceed the
    largest threshold of the step-up rule. The `exceed`, `drawn` and
    `active` fields of each hypothesis in `pending` with a statistic are
    updated; one with an analytic p-value is left unchanged.

    Args:
        pending: The hypotheses of the family.
        labels: Class codes (or values) of the outcome, per row.
        binned: Binned outcome codes, per row.
        strata: Permutation stratum per row.
        total: Number of permutations to draw.
        alpha: Level of the Benjamini-Yekutieli step.
        seed: Run seed of the permutation streams.
    """
    family = len(pending)
    ceiling = alpha / multiplicity.harmonic_exact(family)
    live = [h for h in pending if h.statistic is not None]
    done = 0
    batch = 0
    while done < total and any(h.active for h in live):
        count = min(BATCH, total - done)
        rng = randomness.component_rng(seed, "leakage.screen.batch", batch)
        index = batch_permutations(strata, count, rng)
        permuted = labels[index]
        permuted_bins = binned[index]
        for h in live:
            if not h.active or h.statistic is None:
                continue
            values = h.statistic(permuted_bins if h.uses_binned else permuted)
            h.exceed += int((values >= h.observed - TOLERANCE).sum())
            h.drawn += count
            if Fraction(h.exceed + 1, total + 1) > ceiling:
                h.active = False
        done += count
        batch += 1


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
            threshold.
        adjusted_p: Benjamini-Yekutieli adjusted p-value (1.0 when stopped
            early).
        significant: Rejected by the Benjamini-Yekutieli step.
        lower_bound: One-sided lower confidence bound of the effect size,
            computed for significant columns only.
        suspect: Significant and the bound exceeds the ceiling.
        complete: Whether all permutations were drawn.
        gate: Whether this mechanism has a current, passing certificate.
    """

    feature: str
    mechanism: str
    statistic: float
    p_value: float
    adjusted_p: float
    significant: bool
    lower_bound: float | None
    suspect: bool
    complete: bool
    gate: certificates.GateStatus

    @property
    def blocking(self) -> bool:
        """Whether this suspicion may stop a run (certified mechanism)."""
        return self.suspect and self.gate is certificates.GateStatus.GATING

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
        bound = "n/a" if self.lower_bound is None else f"{self.lower_bound:.3f}"
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
        unchecked_reason: Why they were declined, for the warning.
    """

    findings: tuple[Finding, ...]
    rows: int
    family: int
    permutations: int
    alpha: float
    gates: Mapping[str, certificates.GateStatus]
    ceilings: Mapping[str, float]
    notes: tuple[str, ...] = ()
    unchecked: tuple[str, ...] = ()
    unchecked_reason: str = ""

    @property
    def warning(self) -> str:
        """The loud statement of an incomplete screen, or an empty string."""
        if not self.unchecked:
            return ""
        return (
            f"leakage screen NOT RUN for {len(self.unchecked)} columns "
            f"({self.unchecked_reason}): these columns are unchecked, not "
            "clean; raise --time-budget or screen fewer columns"
        )

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
            warning, suspect and blocking column names, and the findings
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
            "ceilings": dict(self.ceilings),
            "notes": list(self.notes),
            "unchecked": list(self.unchecked),
            "warning": self.warning,
            "suspects": [f.feature for f in self.suspects],
            "blocking": [f.feature for f in self.blocking],
            "findings": [
                {
                    "feature": f.feature,
                    "mechanism": f.mechanism,
                    "statistic": f.statistic,
                    "p_value": f.p_value,
                    "adjusted_p": f.adjusted_p,
                    "significant": f.significant,
                    "lower_bound": (
                        f.lower_bound
                        if f.lower_bound is not None
                        and math.isfinite(f.lower_bound)
                        else None
                    ),
                    "suspect": f.suspect,
                    "complete": f.complete,
                }
                for f in self.findings
                if f.significant or f.suspect
            ],
        }


def subsample(
    n: int,
    groups: numpy.ndarray | None,
    block_size: int | None,
    seed: int,
) -> tuple[numpy.ndarray, int | None]:
    """Row positions to use, at most about `MAX_ROWS`, and the block size.

    The choice does not look at the outcome. Groups are kept whole, time
    blocks keep their order (every k-th row), otherwise rows are drawn at
    random.

    Args:
        n: Number of rows available.
        groups: Group label per row, or None.
        block_size: Consecutive rows per time block, or None.
        seed: Run seed of the draw.

    Returns:
        A pair (sorted row positions, block size to use). Everything is
        kept when `n` is at most `MAX_ROWS`. With groups, whole groups are
        drawn until `MAX_ROWS` rows are reached; with time blocks, every
        k-th row is kept and the block size shrinks by k (not below 2).
    """
    if n <= MAX_ROWS:
        return numpy.arange(n), block_size
    rng = randomness.component_rng(seed, "leakage.screen.rows", 0)
    if groups is not None:
        labels = pandas.unique(groups)
        chosen: list[numpy.ndarray] = []
        size = 0
        for label in labels[rng.permutation(len(labels))]:
            where = numpy.flatnonzero(groups == label)
            chosen.append(where)
            size += len(where)
            if size >= MAX_ROWS:
                break
        return numpy.sort(numpy.concatenate(chosen)), block_size
    if block_size is not None:
        stride = math.ceil(n / MAX_ROWS)
        return numpy.arange(0, n, stride), max(2, block_size // stride)
    return numpy.sort(rng.choice(n, MAX_ROWS, replace=False)), block_size


def gate_statuses(
    store: certificates.Store | None,
    cfg: leakage.Config,
    task: str,
    n_classes: int,
) -> dict[str, certificates.GateStatus]:
    """Determines the gate status per mechanism from the certificates on file.

    A mechanism gates only when the screen's current code has a passing
    certificate for every leak type of that mechanism, the settings equal the
    certified ones (any run seed), and the target is the one that was
    certified (binary classification). Anything else is descriptive.

    Args:
        store: Certificates on file, or None.
        cfg: The settings the screen will run with.
        task: `classification` or `regression`.
        n_classes: Number of classes of the target.

    Returns:
        `GATING` or `DESCRIPTIVE` for each mechanism.
    """
    descriptive = dict.fromkeys(
        MECHANISM_DEFECTS, certificates.GateStatus.DESCRIPTIVE
    )
    if store is None or task != "classification" or n_classes != BINARY_CLASSES:
        return descriptive
    if (
        dataclasses.replace(cfg, run_seed=leakage.CERTIFIED.run_seed)
        != leakage.CERTIFIED
    ):
        return descriptive
    digest = code_digest()
    return {
        mechanism: (
            certificates.GateStatus.GATING
            if all(
                store.status(DETECTOR_ID, digest, defect)
                is certificates.GateStatus.GATING
                for defect in defects
            )
            else certificates.GateStatus.DESCRIPTIVE
        )
        for mechanism, defects in MECHANISM_DEFECTS.items()
    }


def load_store(
    path: Path | None = None,
) -> certificates.Store | None:
    """Loads the certificates of the screen.

    Args:
        path: Certificate file; defaults to the one shipped with the
            package.

    Returns:
        The store, or None if the file is missing or unreadable.
    """
    target = path or STORE_PATH
    if not target.is_file():
        return None
    try:
        return certificates.Store.load(target)
    except (ValueError, KeyError, OSError):
        return None


@functools.cache
def shipped_store() -> certificates.Store | None:
    """Loads the certificates shipped with the package, once.

    Returns:
        The store, or None if it is missing or unreadable. The result is
        cached for the process.
    """
    return load_store()


def work_cap(time_budget_seconds: float | None) -> int | None:
    """Computes the most cross-fitted fits the screen may use in a budget.

    Deterministic: the budget alone decides, not the machine's speed.

    Args:
        time_budget_seconds: The declared budget, or None for no cap beyond
            `MAX_PERMUTATIONS` (the certification paths).

    Returns:
        The cap, or None for no cap.
    """
    if time_budget_seconds is None:
        return None
    return int(time_budget_seconds * SCREEN_SHARE * FITS_PER_SECOND)


def screen_table(
    frame: pandas.DataFrame,
    target: str,
    task: str,
    features: Sequence[Hashable],
    seed: int,
    group_column: str | None = None,
    time_column: str | None = None,
    store: certificates.Store | None = None,
    time_budget_seconds: float | None = None,
) -> Screen:
    """Screens training data with the certified settings.

    Groups keep entities whole. A time column orders the rows (stable sort)
    and defines consecutive blocks of `len(frame) // TIME_BLOCKS` rows, at
    least 2; groups win when both are given.

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
        store: Certificates for gating; defaults to the shipped ones.
        time_budget_seconds: The project's declared time budget; it bounds
            the screen's work by `work_cap`. None means no cap beyond
            `MAX_PERMUTATIONS`.

    Returns:
        The report, using `CERTIFIED` settings with `seed`. A screen that was
        declined says so in `unchecked` and `warning`. A `features` entry
        that is not a candidate column, or a missing `target`, raises as in
        `screen`.
    """
    cfg = dataclasses.replace(leakage.CERTIFIED, run_seed=seed)
    chosen = store if store is not None else shipped_store()
    groups = (
        frame[group_column].astype(str)
        if group_column and group_column in frame.columns
        else None
    )
    block_size = None
    order: numpy.ndarray = numpy.arange(len(frame))
    if groups is None and time_column and time_column in frame.columns:
        order = frame[time_column].argsort(kind="stable").to_numpy()
        block_size = max(2, len(frame) // TIME_BLOCKS)
    return screen(
        frame.iloc[order],
        target,
        task,
        cfg,
        features,
        None if groups is None else groups.iloc[order],
        block_size,
        chosen,
        time_budget_seconds,
    )


def code_digest() -> str:
    """Hash of the code the screen's decisions depend on.

    The key of its certificate: this module, the leakage helpers it reuses
    and the multiplicity step.

    Returns:
        The combined code hash of those modules.
    """
    return certificates.code_hash(sys.modules[__name__], leakage, multiplicity)


def ceilings_of(cfg: leakage.Config) -> dict[str, float]:
    """Looks up the pre-registered ceiling of each mechanism's effect size.

    Args:
        cfg: Settings holding the ceilings.

    Returns:
        A dict from mechanism to ceiling; missingness and record purity
        share the association ceiling.
    """
    return {
        SKILL: cfg.skill_ceiling,
        MISSING: cfg.association_ceiling,
        PURITY: cfg.association_ceiling,
    }


def empty_report(
    cfg: leakage.Config,
    gates: Mapping[str, certificates.GateStatus],
    rows: int,
    note: str,
) -> Screen:
    """Builds a report that tested nothing, and says why.

    Args:
        cfg: Settings (`alpha` and the ceilings are recorded).
        gates: Gate status per mechanism.
        rows: Rows the screen saw.
        note: Why nothing was tested.

    Returns:
        A report with no findings, a family of 0, and `note` as its only
        note.
    """
    return Screen(
        (),
        rows,
        0,
        0,
        cfg.alpha,
        gates,
        ceilings_of(cfg),
        (note,),
    )


def build_problem(
    target: pandas.Series,
    task: str,
    cfg: leakage.Config,
    groups: numpy.ndarray | None,
    block_size: int | None,
) -> Problem:
    """Encodes the outcome and fixes the folds.

    Args:
        target: The outcome.
        task: `classification` or `regression`.
        cfg: Settings (`folds`, `run_seed`).
        groups: Group label per row, or None.
        block_size: Consecutive rows per time block, or None.

    Returns:
        The encoded outcome with its folds.

    Raises:
        ValueError: If a regression target is constant, or the outcome is
            rejected by `leakage.encode_target` (missing values, fewer
            than two classes, or an unknown task).
    """
    y = leakage.encode_target(target, task)
    folds = fold_splits(len(y), cfg.folds, groups, block_size, cfg.run_seed)
    total = float(((y - y.mean()) ** 2).sum()) if task == "regression" else 0.0
    if task == "regression" and total == 0.0:
        raise ValueError("a constant target has no R^2")
    classes = int(y.max()) + 1 if task == "classification" else 0
    return Problem(y, task, classes, folds, total)


def skill_hypothesis(
    problem: Problem, name: str, phi: numpy.ndarray
) -> Pending:
    """Builds the skill hypothesis of one column.

    Args:
        problem: The encoded outcome with its folds.
        name: Column name.
        phi: The column's basis.

    Returns:
        The pending hypothesis with the observed skill.
    """
    statistic = skill_statistic(problem, phi)
    observed = float(statistic(problem.y[None, :])[0])
    return Pending(name, SKILL, observed, statistic, uses_binned=False)


def missingness_hypotheses(
    name: str,
    series: pandas.Series,
    outcome: pandas.Series,
    task: str,
    cfg: leakage.Config,
    class_codes: numpy.ndarray,
    totals: numpy.ndarray,
    *,
    structured: bool,
) -> list[Pending]:
    """Builds the record-purity and missingness hypotheses of one column.

    Args:
        name: Column name.
        series: The column.
        outcome: The target column.
        task: `classification` or `regression`.
        cfg: Settings.
        class_codes: Class code of the binned outcome per row.
        totals: Rows per class.
        structured: Whether rows are grouped or in time blocks, which
            replaces the analytic missingness test by a permutation one.

    Returns:
        No hypotheses when the column is all missing, never missing, or
        the outcome has a single class; otherwise the purity hypothesis
        followed by the missingness hypothesis, when one can be formed.
    """
    missing = series.isna().to_numpy()
    if missing.all() or not missing.any() or len(totals) < MIN_CLASSES:
        return []
    recorded = ~missing
    pure = purity_statistic(recorded)
    found = [
        Pending(
            name,
            PURITY,
            float(pure(class_codes[None, :])[0]),
            pure,
            uses_binned=True,
        )
    ]
    if structured:
        stat = missingness_statistic(missing, totals)
        observed = float(stat(class_codes[None, :])[0])
        found.append(Pending(name, MISSING, observed, stat, uses_binned=True))
        return found
    test = leakage.missingness_test(series, outcome, task, cfg)
    if test is not None:
        found.append(
            Pending(
                name,
                MISSING,
                test.cramers_v,
                None,
                uses_binned=True,
                p_value=test.p_value,
                active=False,
            )
        )
    return found


def limit_report(
    cfg: leakage.Config,
    gates: Mapping[str, certificates.GateStatus],
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
        ceilings_of(cfg),
        (note,),
        declined,
        limit,
    )


def finding_bound(
    h: Pending,
    problem: Problem,
    columns: Mapping[str, numpy.ndarray],
    frame: pandas.DataFrame,
    class_codes: numpy.ndarray,
    totals: numpy.ndarray,
    cfg: leakage.Config,
    groups: numpy.ndarray | None,
) -> float:
    """Bounds the effect of one significant hypothesis from below.

    Args:
        h: The significant hypothesis.
        problem: The encoded outcome with its folds.
        columns: Basis of each column with a skill hypothesis.
        frame: The chosen rows.
        class_codes: Class code of the binned outcome per row.
        totals: Rows per class.
        cfg: Settings.
        groups: Group label per row, or None.

    Returns:
        The lower confidence bound of the statistic the mechanism measures.
    """
    path = f"leakage.screen.{h.mechanism}.{h.feature}"
    rng = randomness.component_rng(cfg.run_seed, path, 0)
    if h.mechanism == SKILL:
        return skill_bound(problem, columns[h.feature], cfg, groups, rng)
    if h.mechanism == PURITY:
        recorded = frame[h.feature].notna().to_numpy()
        return purity_bound(recorded, class_codes, cfg, groups)
    missing = frame[h.feature].isna().to_numpy()
    return bootstrap_v(missing, class_codes, len(totals), cfg, groups, rng)


def scan_frame(
    frame: pandas.DataFrame,
    target: str,
    task: str,
    cfg: leakage.Config,
    names: Sequence[Hashable],
    groups: numpy.ndarray | None,
    block_size: int | None,
    gates: Mapping[str, certificates.GateStatus],
    time_budget_seconds: float | None = None,
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
        time_budget_seconds: Declared time budget, or None for no cap
            beyond `MAX_PERMUTATIONS`.

    Returns:
        The report. It has no findings, with the reason in its notes, when
        the outcome is invalid, there are too few rows, no column could be
        tested, or the permutations or work needed exceed the limits; in the
        last case the candidate columns are listed in `unchecked`.
    """
    n = len(frame)
    try:
        problem = build_problem(frame[target], task, cfg, groups, block_size)
    except ValueError as error:
        return empty_report(cfg, gates, n, f"screen not run: {error}")
    if not leakage.enough_rows(problem.y, task, cfg):
        return empty_report(
            cfg, gates, n, "screen not run: too few rows (low power)"
        )
    alpha = budget.exact(cfg.alpha, "alpha")
    strata = strata_of(n, groups, block_size)
    structured = groups is not None or block_size is not None
    classes = leakage.binned_target(frame[target], task)
    class_codes = pandas.factorize(classes, sort=True)[0].astype("int64")
    totals = numpy.bincount(class_codes).astype("float64")
    pending: list[Pending] = []
    columns: dict[str, numpy.ndarray] = {}
    for name in names:
        series = frame[name]
        phi = basis(series)
        if phi.shape[1] > 1:
            columns[str(name)] = phi
            pending.append(skill_hypothesis(problem, str(name), phi))
        pending.extend(
            missingness_hypotheses(
                str(name),
                series,
                frame[target],
                task,
                cfg,
                class_codes,
                totals,
                structured=structured,
            )
        )
    family = len(pending)
    ceilings = ceilings_of(cfg)
    if family == 0:
        return empty_report(cfg, gates, n, "no column could be tested")
    total = max(cfg.permutations, required_permutations(family, alpha))
    declined = tuple(dict.fromkeys(h.feature for h in pending))
    if total > MAX_PERMUTATIONS:
        return limit_report(
            cfg,
            gates,
            n,
            family,
            (
                f"{family} hypotheses need {total} permutations to resolve "
                f"the Benjamini-Yekutieli threshold; the limit is "
                f"{MAX_PERMUTATIONS}. Nothing was flagged (power, not size, "
                "is lost): screen fewer columns"
            ),
            declined,
            "permutation limit",
        )
    skill_columns = sum(
        1 for h in pending if h.mechanism == SKILL and h.statistic is not None
    )
    work = (
        total
        * skill_columns
        * (problem.classes if problem.classes > BINARY_CLASSES else 1)
    )
    cap = work_cap(time_budget_seconds)
    if cap is not None and work > cap:
        return limit_report(
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
    draw_permutations(
        pending, problem.y, class_codes, strata, total, alpha, cfg.run_seed
    )
    ps = [
        h.p_value if h.p_value is not None else (h.exceed + 1) / (h.drawn + 1)
        for h in pending
    ]
    complete = [h.statistic is None or h.drawn == total for h in pending]
    used = [p if done else 1.0 for p, done in zip(ps, complete, strict=True)]
    adjustment = multiplicity.adjust(
        used, scmle.data.ledger.budget.Correction.BY, alpha
    )
    findings = []
    for k, h in enumerate(pending):
        significant = adjustment.rejected[k]
        bound: float | None = None
        suspect = False
        if significant:
            bound = finding_bound(
                h, problem, columns, frame, class_codes, totals, cfg, groups
            )
            suspect = bound > ceilings[h.mechanism]
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
    return Screen(tuple(findings), n, family, total, cfg.alpha, gates, ceilings)


def screen(
    frame: pandas.DataFrame,
    target: str,
    task: str,
    cfg: leakage.Config,
    features: Sequence[Hashable] | None = None,
    groups: pandas.Series | None = None,
    block_size: int | None = None,
    store: certificates.Store | None = None,
    time_budget_seconds: float | None = None,
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
        store: Certificates for gating; None means every mechanism is
            descriptive.
        time_budget_seconds: Declared time budget; bounds the work through
            `work_cap`. None means no cap beyond `MAX_PERMUTATIONS`.

    Returns:
        The report. Nothing is raised for a degenerate target or too few
        rows: the report is empty and says why.

    Raises:
        ValueError: If both `groups` and `block_size` are given, `groups`
            differs in length from `frame`, or `features` names a column
            that is missing or is the target.
    """
    if groups is not None and block_size is not None:
        raise ValueError("use groups or block_size, not both")
    if groups is not None and len(groups) != len(frame):
        raise ValueError("groups and frame must have the same length")
    names = [c for c in frame.columns if c != target]
    chosen = names if features is None else list(features)
    unknown = [c for c in chosen if c not in names]
    if unknown:
        raise ValueError(f"not feature columns: {unknown}")
    n_classes = int(frame[target].nunique()) if len(frame) else 0
    gates = gate_statuses(store, cfg, task, n_classes)
    group_values = None if groups is None else groups.to_numpy()
    rows, blocks = subsample(len(frame), group_values, block_size, cfg.run_seed)
    sample = frame.iloc[rows].reset_index(drop=True)
    sample_groups = None if group_values is None else group_values[rows]
    return scan_frame(
        sample,
        target,
        task,
        cfg,
        chosen,
        sample_groups,
        blocks,
        gates,
        time_budget_seconds,
    )
