"""Cross-fitted skill of one feature, its permutation null and its bounds.

The folds do not depend on the outcome, so the observed skill and every
permuted one come from the same procedure.
"""

from collections.abc import Callable
from collections.abc import Sequence
import itertools
import math
import operator

import numpy
import pandas
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor

from scmle.assurance.admissibility.leakage.settings import BINARY_CLASSES
from scmle.assurance.admissibility.leakage.settings import Config
from scmle.assurance.admissibility.leakage.settings import MAX_LEVELS
from scmle.assurance.admissibility.leakage.settings import MIN_CLASSES
from scmle.assurance.admissibility.leakage.settings import Splits
from scmle.assurance.admissibility.leakage.settings import TASKS
from scmle.contracts import errors
from scmle.runtime import randomness


def design(series: pandas.Series) -> numpy.ndarray:
    """Builds the numeric design of one feature, keeping missing values as NaN.

    A numeric feature is one column (trees send NaN down a learned branch).
    Anything else is one-hot encoded, with a level for missing values; levels
    beyond the `MAX_LEVELS` most frequent are pooled.

    Args:
        series: One feature.

    Returns:
        A float64 array with one row per entry of `series`: one column for
        a numeric (non-boolean) feature, otherwise one column per level.
    """
    if pandas.api.types.is_numeric_dtype(
        series
    ) and not pandas.api.types.is_bool_dtype(series):
        return series.to_numpy(dtype="float64").reshape(-1, 1)
    labels = series.astype("object").where(series.notna(), "<missing>")
    labels = labels.astype(str)
    top = labels.value_counts().index[:MAX_LEVELS]
    pooled = labels.where(labels.isin(top), "<other>")
    return pandas.get_dummies(pooled).to_numpy(dtype="float64")


def vectorize(y: pandas.Series, task: str) -> numpy.ndarray:
    """Encodes the target as class codes or floats.

    Args:
        y: The target.
        task: `classification` or `regression`.

    Returns:
        For classification, int64 codes 0..k-1 of the sorted classes; for
        regression, the target as float64.

    Raises:
        ValueError: On a bad task, missing values, or fewer than 2 classes.
    """
    if task not in TASKS:
        raise errors.InputError(f"task must be one of {TASKS}")
    if y.isna().any():
        raise errors.InputError("the target must not have missing values")
    if task == "regression":
        return y.to_numpy(dtype="float64")
    codes, uniques = pandas.factorize(y, sort=True)
    if len(uniques) < MIN_CLASSES:
        raise errors.InputError("a classification target needs two classes")
    return codes.astype("int64")


def permute_within(
    values: numpy.ndarray,
    rng: numpy.random.Generator,
    groups: numpy.ndarray | None,
    block_size: int | None,
) -> numpy.ndarray:
    """Draws a row permutation, global, within groups, or within time blocks.

    Args:
        values: The values to be permuted; only their count is used.
        rng: Random generator.
        groups: Group label per row, or None. Takes precedence over
            `block_size`.
        block_size: Consecutive rows per time block, or None.

    Returns:
        Integer positions; `values[positions]` is the permuted sequence.
        With `groups`, rows move only within their group; with
        `block_size`, only within their block of consecutive rows.
    """
    n = len(values)
    out = numpy.arange(n)
    if groups is not None:
        for label in pandas.unique(groups):
            where = numpy.flatnonzero(groups == label)
            out[where] = where[rng.permutation(len(where))]
    elif block_size is not None:
        for start in range(0, n, block_size):
            where = numpy.arange(start, min(start + block_size, n))
            out[where] = where[rng.permutation(len(where))]
    else:
        out = rng.permutation(n)
    return out


def carve(
    y: numpy.ndarray,
    task: str,
    folds: int,
    groups: numpy.ndarray | None,
    block_size: int | None,
    seed: int,
) -> Splits:
    """Makes cross-fitting folds: by group, contiguous in time, or stratified.

    Args:
        y: The encoded target.
        task: `classification` or `regression`.
        folds: Requested number of folds; capped at the number of groups
            when `groups` is given.
        groups: Group label per row, or None. Whole groups stay in one fold.
        block_size: Consecutive rows per time block, or None; when given
            (and `groups` is not), folds are contiguous and unshuffled.
        seed: Seed of the shuffling for the other cases.

    Returns:
        A list of (train positions, test positions) pairs, one per fold.
        Without groups or blocks the folds are shuffled, and stratified by
        class for classification.
    """
    rows = numpy.arange(len(y))
    if groups is not None:
        k = min(folds, len(pandas.unique(groups)))
        return list(GroupKFold(n_splits=k).split(rows, y, groups))
    if block_size is not None:
        return list(KFold(n_splits=folds).split(rows))
    if task == "classification":
        return list(
            StratifiedKFold(folds, shuffle=True, random_state=seed).split(
                rows, y
            )
        )
    return list(KFold(folds, shuffle=True, random_state=seed).split(rows))


def out_of_fold(
    x: numpy.ndarray,
    y: numpy.ndarray,
    task: str,
    splits: Splits,
    cfg: Config,
    seed: int,
) -> numpy.ndarray:
    """Predicts every row from a small tree fitted on the other folds.

    Args:
        x: Feature design, one row per entry of `y`.
        y: The encoded target.
        task: `classification` or `regression`.
        splits: Cross-fitting folds.
        cfg: Settings (`max_leaf_nodes`, `min_samples_leaf`).
        seed: Seed of the tree.

    Returns:
        For classification, an array of shape (rows, classes) of predicted
        class probabilities, with 0 for a class absent from a training
        fold; for regression, one predicted value per row. Each row is
        predicted only by a tree that did not see it.
    """
    n_classes = int(y.max()) + 1 if task == "classification" else 0
    shape = (len(y), n_classes) if task == "classification" else (len(y),)
    out = numpy.zeros(shape)
    for train, test in splits:
        if task == "classification":
            clf = DecisionTreeClassifier(
                max_leaf_nodes=cfg.max_leaf_nodes,
                min_samples_leaf=cfg.min_samples_leaf,
                random_state=seed,
            ).fit(x[train], y[train])
            out[numpy.ix_(test, clf.classes_)] = clf.predict_proba(x[test])
        else:
            reg = DecisionTreeRegressor(
                max_leaf_nodes=cfg.max_leaf_nodes,
                min_samples_leaf=cfg.min_samples_leaf,
                random_state=seed,
            ).fit(x[train], y[train])
            out[test] = reg.predict(x[test])
    return out


def auc(truth: numpy.ndarray, score: numpy.ndarray) -> float:
    """Computes the ROC AUC by the rank-sum formula, with ties averaged.

    Args:
        truth: Labels, 1 for the positive class and 0 otherwise.
        score: Score per row; higher means more likely positive.

    Returns:
        The area under the ROC curve, in [0, 1].

    Raises:
        ValueError: If `truth` does not hold both classes (0 and 1).
    """
    positive = truth == 1
    n1 = int(positive.sum())
    n0 = len(truth) - n1
    if n1 == 0 or n0 == 0:
        raise errors.InputError("both classes are needed for an AUC")
    ranks = stats.rankdata(score)
    return float((ranks[positive].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def predictiveness(pred: numpy.ndarray, y: numpy.ndarray, task: str) -> float:
    """Measures the skill of out-of-fold predictions: AUC or R^2.

    Args:
        pred: Output of `out_of_fold`.
        y: The encoded target.
        task: `classification` or `regression`.

    Returns:
        For regression, R^2. For classification, the ROC AUC of the second
        column when there are two classes, otherwise the macro one-vs-rest
        AUC.

    Raises:
        ValueError: If a classification sample lacks a class or a regression
            target is constant.
    """
    if task == "regression":
        total = float(((y - y.mean()) ** 2).sum())
        if total == 0.0:
            raise errors.InputError("a constant target has no R^2")
        return 1.0 - float(((y - pred) ** 2).sum()) / total
    if pred.shape[1] == BINARY_CLASSES:
        return auc(y, pred[:, 1])
    return float(
        roc_auc_score(
            y,
            pred,
            multi_class="ovr",
            average="macro",
            labels=list(range(pred.shape[1])),
        )
    )


def crossfit(
    x: numpy.ndarray,
    y: numpy.ndarray,
    task: str,
    cfg: Config,
    groups: numpy.ndarray | None,
    block_size: int | None,
    seed: int,
) -> tuple[float, numpy.ndarray]:
    """Computes the out-of-fold skill of a feature.

    Args:
        x: Feature design.
        y: The encoded target.
        task: `classification` or `regression`.
        cfg: Settings.
        groups: Group label per row, or None.
        block_size: Consecutive rows per time block, or None.
        seed: Seed of the folds and trees.

    Returns:
        A pair (skill, out-of-fold predictions), the predictions being the
        output of `out_of_fold`.
    """
    splits = carve(y, task, cfg.folds, groups, block_size, seed)
    pred = out_of_fold(x, y, task, splits, cfg, seed)
    return predictiveness(pred, y, task), pred


def rescore(
    pred: numpy.ndarray,
    y: numpy.ndarray,
    task: str,
    members: list[numpy.ndarray] | None,
    rng: numpy.random.Generator,
) -> float | None:
    """Computes the skill on one bootstrap resample.

    Args:
        pred: Output of `out_of_fold`.
        y: The encoded target.
        task: `classification` or `regression`.
        members: Row positions of each group, so whole groups are resampled,
            or None to resample rows.
        rng: Random generator.

    Returns:
        The skill on the resample, or None when it cannot be computed (a
        class is missing, or a regression target is constant).
    """
    n = len(y)
    if members is not None:
        picks = rng.integers(0, len(members), size=len(members))
        index = numpy.concatenate([members[i] for i in picks])
    else:
        index = rng.integers(0, n, size=n)
    try:
        return predictiveness(pred[index], y[index], task)
    except ValueError:
        # handled: uncomputable resample skill returned as None, documented
        return None


def lower_bound(
    pred: numpy.ndarray,
    y: numpy.ndarray,
    task: str,
    cfg: Config,
    groups: numpy.ndarray | None,
    rng: numpy.random.Generator,
) -> float:
    """Computes a one-sided lower confidence bound on the skill by bootstrap.

    Rows are resampled, or whole groups when `groups` is given. Draws whose
    skill cannot be computed (a class is missing, or a regression target is
    constant) are skipped; if every draw is skipped the bound is -inf.

    Args:
        pred: Output of `out_of_fold`.
        y: The encoded target.
        task: `classification` or `regression`.
        cfg: Settings (`bootstrap`, `confidence`).
        groups: Group label per row, or None.
        rng: Random generator.

    Returns:
        The `1 - confidence` quantile of the bootstrap skills, or -inf.
    """
    labels = pandas.unique(groups) if groups is not None else None
    members = (
        [numpy.flatnonzero(groups == g) for g in labels]
        if groups is not None and labels is not None
        else None
    )
    draws = list(
        map(
            operator.call,
            itertools.repeat(
                lambda: rescore(pred, y, task, members, rng),
                cfg.bootstrap,
            ),
        )
    )
    values = [v for v in draws if v is not None]
    if not values:
        return -math.inf
    return float(numpy.quantile(values, 1.0 - cfg.confidence))


def prepare(
    feature: pandas.Series,
    target: pandas.Series,
    task: str,
    groups: pandas.Series | None,
    block_size: int | None,
) -> tuple[str, numpy.ndarray, numpy.ndarray | None, numpy.ndarray]:
    """Encodes one feature and the target for a skill test.

    Args:
        feature: The feature under test.
        target: The outcome.
        task: `classification` or `regression`.
        groups: Group label per row, or None.
        block_size: Consecutive rows per time block, or None.

    Returns:
        The feature's name, the encoded target, the group labels (None
        without groups) and the feature design.

    Raises:
        ValueError: If both `groups` and `block_size` are given, or the
            target is invalid (see `vectorize`).
    """
    if groups is not None and block_size is not None:
        raise errors.InputError("use groups or block_size, not both")
    name = str(feature.name)
    y = vectorize(target, task)
    group_values = None if groups is None else groups.to_numpy()
    return name, y, group_values, design(feature)


def measure(
    x: numpy.ndarray,
    y: numpy.ndarray,
    task: str,
    cfg: Config,
    groups: numpy.ndarray | None,
    block_size: int | None,
    path: str,
    draw: Callable[
        [numpy.random.Generator], tuple[numpy.ndarray, numpy.ndarray]
    ],
) -> tuple[float, numpy.ndarray, list[float]]:
    """Cross-fits the observed skill and its permutation null.

    The folds and trees are seeded from `path` and are the same for the
    observed fit and every permuted one; `draw` permutes the design or the
    target with a generator seeded from `path` too.

    Args:
        x: Feature design.
        y: The encoded target.
        task: `classification` or `regression`.
        cfg: Settings (`permutations`, `run_seed`).
        groups: Group label per row, or None.
        block_size: Consecutive rows per time block, or None.
        path: Seed path of the test.
        draw: Returns one permuted (design, target) pair from the generator.

    Returns:
        The observed skill, its out-of-fold predictions, and the skill of
        each of `cfg.permutations` permuted fits.
    """
    seed = randomness.derive_int_seed(cfg.run_seed, path, 0)
    skill, pred = crossfit(x, y, task, cfg, groups, block_size, seed)
    rng = randomness.component_rng(cfg.run_seed, path, 1)
    null = list(
        map(
            operator.call,
            itertools.repeat(
                lambda: crossfit(
                    *draw(rng), task, cfg, groups, block_size, seed
                )[0],
                cfg.permutations,
            ),
        )
    )
    return skill, pred, null


def significance(observed: float, null: Sequence[float]) -> float:
    """Computes `(1 + #{null >= observed}) / (B + 1)` (Phipson and Smyth, 2010).

    Args:
        observed: The observed statistic.
        null: Statistics under permutations of the target.

    Returns:
        The permutation p-value, in (0, 1].
    """
    exceed = sum(1 for value in null if value >= observed)
    return (1.0 + exceed) / (len(null) + 1.0)


def sufficient(y: numpy.ndarray, task: str, cfg: Config) -> bool:
    """Decides whether there are enough rows for the tests to have any power.

    Args:
        y: The encoded target.
        task: `classification` or `regression`.
        cfg: Settings (`min_rows`, `folds`).

    Returns:
        False when there are fewer than `min_rows` rows or, for
        classification, the smallest class has fewer than `folds` rows;
        True otherwise.
    """
    if len(y) < cfg.min_rows:
        return False
    if task == "classification":
        return min(numpy.bincount(y).tolist()) >= cfg.folds  # type: ignore[no-any-return, unused-ignore]  # Any on older NumPy stubs; editing code would stale the shipped certificate
    return True
