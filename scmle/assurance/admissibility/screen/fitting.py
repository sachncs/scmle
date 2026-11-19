"""Cross-fitted single-column skill: the basis, the folds and the bounds.

A ridge smoother of the outcome on one column's basis is fitted out of fold;
its held-out AUC or R^2 is the skill statistic the permutations re-evaluate.
"""

import collections
from collections.abc import Hashable
from collections.abc import Sequence
import dataclasses
import itertools
import math

import numpy
import pandas
from scipy import stats

from scmle.assurance.admissibility import leakage
from scmle.assurance.admissibility.screen.settings import BINARY_CLASSES
from scmle.assurance.admissibility.screen.settings import HELD
from scmle.assurance.admissibility.screen.settings import KNOTS
from scmle.assurance.admissibility.screen.settings import PAIR_LEVELS
from scmle.assurance.admissibility.screen.settings import RIDGE
from scmle.assurance.admissibility.screen.settings import Statistic
from scmle.assurance.admissibility.screen.settings import WORD
from scmle.contracts import errors


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
    if pandas.api.types.is_datetime64_any_dtype(series):
        series = (series - series.min()) / pandas.Timedelta(days=1)
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
    levels = pandas.get_dummies(pooled).to_numpy(dtype="float64")
    if labels.nunique() <= leakage.MAX_LEVELS:
        return levels
    return numpy.hstack([levels, words(labels)])


def line(series: pandas.Series, knots: int = KNOTS) -> numpy.ndarray | None:
    """Builds the linear design of an ordered column: its value, its gaps.

    The reading any model of one ordered column can make, ranking rows by
    the value, with none of the smoother's knots to fit (`hypotheses.reading`
    gives it to a skill suspect when it scores higher). A gap is filled with
    the mean of the recorded values and marked by an indicator column.

    Args:
        series: One candidate column.
        knots: As for `basis`: a column with at most this many distinct
            values gets indicators there, so it has no line here.

    Returns:
        A float64 array of shape (n, 1), or (n, 2) with gaps; None for a
        column that is not numeric or a date, or has few distinct values.
    """
    if pandas.api.types.is_datetime64_any_dtype(series):
        series = (series - series.min()) / pandas.Timedelta(days=1)
    numeric = pandas.api.types.is_numeric_dtype(
        series
    ) and not pandas.api.types.is_bool_dtype(series)
    if not numeric:
        return None
    values = series.to_numpy(dtype="float64")
    absent = numpy.isnan(values)
    if len(numpy.unique(values[~absent])) <= knots:
        return None
    filled = numpy.where(absent, values[~absent].mean(), values)
    if not absent.any():
        return filled[:, None]
    return numpy.column_stack([filled, absent.astype("float64")])


def words(labels: pandas.Series) -> numpy.ndarray:
    """Marks the commonest words of free text, one indicator per word.

    Text with more distinct values than `leakage.MAX_LEVELS` pools most
    rows into one level, so a target written into a sentence ("customer
    closed the account") is invisible to level indicators; the words carry
    it.

    Args:
        labels: The column as strings.

    Returns:
        A float64 array with one row per entry and one column for each of
        the `leakage.MAX_LEVELS` words found in the most rows (case folded),
        1.0 where the row holds the word.
    """
    held = [set(WORD.findall(text.lower())) for text in labels]
    counts = collections.Counter(w for row in held for w in row)
    top = [pair[0] for pair in counts.most_common(leakage.MAX_LEVELS)]
    return numpy.array([[w in row for w in top] for row in held], "float64")


def stratify(
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


def permute(
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


def crossfit(
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
    return leakage.carve(
        numpy.zeros(n), "regression", folds, groups, block_size, seed
    )


def fit(
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


def concordance(
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


def aucs(
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
    wins, pairs = concordance(score, positive, tests)
    with numpy.errstate(invalid="ignore", divide="ignore"):
        auc: numpy.ndarray = wins / pairs
    return auc


@dataclasses.dataclass(frozen=True, slots=True)
class Problem:
    """The outcome, task and folds shared by every column's test.

    Attributes:
        y: Class codes or floats.
        task: `classification` or `regression`.
        classes: Number of classes (0 for regression).
        splits: Cross-fitting folds.
        total: Total sum of squares of a continuous outcome (0.0 for
            a classification).
    """

    y: numpy.ndarray
    task: str
    classes: int
    splits: leakage.Splits
    total: float


def quantify(problem: Problem, phi: numpy.ndarray) -> Statistic:
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
            pred = fit(labels, phi, problem.splits)
            squares = ((labels - pred) ** 2).sum(axis=1)
            skill: numpy.ndarray = 1.0 - squares / problem.total
            return skill
        wanted = (
            [1] if problem.classes == BINARY_CLASSES else range(problem.classes)
        )
        scores = []
        for k in wanted:
            positive = labels == k
            pred = fit(positive.astype("float64"), phi, problem.splits)
            scores.append(aucs(pred, positive, tests))
        return numpy.mean(scores, axis=0)  # type: ignore[no-any-return, unused-ignore]  # Any on older NumPy stubs; editing code would stale the shipped certificate

    return compute


def predict(problem: Problem, phi: numpy.ndarray) -> list[numpy.ndarray]:
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
        return [fit(y[None, :], phi, problem.splits)[0]]
    wanted = (
        [1] if problem.classes == BINARY_CLASSES else range(problem.classes)
    )
    return [
        fit((y == k)[None, :].astype("float64"), phi, problem.splits)[0]
        for k in wanted
    ]


def replicate(
    problem: Problem,
    predictions: Sequence[numpy.ndarray],
    index: numpy.ndarray,
    fold_of: numpy.ndarray,
) -> float | None:
    """Computes the skill on a bootstrap sample of rows.

    Args:
        problem: The outcome, task and folds.
        predictions: Output of `predict`.
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
        wins, pairs = concordance(
            pred[index][None, :], (y == k)[None, :], tests
        )
        if pairs[0] == 0:
            return None
        scores.append(wins[0] / pairs[0])
    return float(numpy.mean(scores))


def temper(
    problem: Problem,
    phi: numpy.ndarray,
    cfg: leakage.Config,
    groups: numpy.ndarray | None,
    rng: numpy.random.Generator,
) -> float:
    """One-sided lower confidence bound of the skill by bootstrap.

    Rows are resampled, or whole groups when `groups` is given, and the
    out-of-fold predictions stay attached to their rows and folds. Draws whose
    skill cannot be computed (see `replicate`) are skipped; with none left
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
    predictions = predict(problem, phi)
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
        value = replicate(problem, predictions, index, fold_of)
        if value is not None:
            values.append(value)
    if not values:
        return -math.inf
    return float(numpy.quantile(values, 1.0 - cfg.confidence))


def margin(
    problem: Problem,
    mine: Sequence[numpy.ndarray],
    theirs: Sequence[numpy.ndarray],
    cfg: leakage.Config,
    groups: numpy.ndarray | None,
    rng: numpy.random.Generator,
) -> float:
    """One-sided lower confidence bound of one skill over another, paired.

    Both sets of out-of-fold predictions are scored on the same bootstrap
    rows (whole groups when given), so the noise the two share cancels: a
    column and its companion measured on a small table (iris petal width and
    length) differ by less than either's own bound.

    Args:
        problem: The outcome, task and folds.
        mine: Output of `predict` for one design.
        theirs: Output of `predict` (or a stack of the same form) for the
            other.
        cfg: Settings (`bootstrap`, `confidence`).
        groups: Group label per row, or None.
        rng: Random generator.

    Returns:
        The `1 - confidence` quantile of the skill differences, or -inf when
        no draw can be scored.
    """
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
        one = replicate(problem, mine, index, fold_of)
        other = replicate(problem, theirs, index, fold_of)
        if one is not None and other is not None:
            values.append(one - other)
    if not values:
        return -math.inf
    return float(numpy.quantile(values, 1.0 - cfg.confidence))


def pair(first: str, second: str) -> str:
    """Names the joint hypothesis of two columns.

    Args:
        first: One column.
        second: The other.

    Returns:
        The name findings of the pair carry.
    """
    return f"{first} & {second}"


def pairs(
    frame: pandas.DataFrame, names: Sequence[Hashable]
) -> list[tuple[Hashable, Hashable]]:
    """Lists the pairs of candidate columns tested jointly.

    The choice reads the columns alone, never the target.

    Args:
        frame: The chosen rows.
        names: Candidate columns.

    Returns:
        Every pair of columns with `PAIR_LEVELS` recorded values, in
        candidate order.
    """
    low = [n for n in names if frame[n].nunique() == PAIR_LEVELS]
    return list(itertools.combinations(low, 2))


def enough(y: numpy.ndarray, task: str) -> bool:
    """Decides whether the screen's tests can be formed at all.

    A cross-fitted skill needs, for every class, one row held out while
    another of its class is in the training folds: two rows a class. Fewer
    rows per class than folds (a fraud table with a handful of positives),
    or fewer rows than `min_rows` (a wide table of a hundred samples),
    lowers power, not size: each permutation p-value stays exact, a column
    below its ceiling cannot be a suspect however many rows there are, and
    the probe states which plants such rows hide. So a small table is
    screened, and `scan` notes the low power, rather than left unchecked.

    Args:
        y: The encoded target.
        task: `classification` or `regression`.

    Returns:
        False for classification with a class of fewer than two rows; True
        otherwise.
    """
    if task == "classification":
        return int(numpy.min(numpy.bincount(y))) >= HELD
    return True


def encode(
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
            rejected by `leakage.vectorize` (missing values, fewer
            than two classes, or an unknown task).
    """
    y = leakage.vectorize(target, task)
    folds = crossfit(len(y), cfg.folds, groups, block_size, cfg.seed)
    total = float(((y - y.mean()) ** 2).sum()) if task == "regression" else 0.0
    if task == "regression" and total == 0.0:
        raise errors.InputError("a constant target has no R^2")
    classes = int(y.max()) + 1 if task == "classification" else 0
    return Problem(y, task, classes, folds, total)
