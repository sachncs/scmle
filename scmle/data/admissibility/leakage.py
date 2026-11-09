"""Statistical leakage tests: suspicion, never proof.

Statistics cannot separate a leak from a strong legitimate predictor, so every
result here is a suspicion for a person (or a certified gate) to weigh against
provenance. All tests read the exploration set `X` only. Each test
returns a `Check` for the hypothesis ledger.

Tests
-----
1. Single-feature skill against a permutation null. A small tree
   (`max_leaf_nodes` leaves) is fitted on one feature, with cross-fitted
   (out-of-fold) predictions. Skill is the out-of-fold ROC AUC (macro
   one-vs-rest for more than two classes) or R^2 for a continuous target. The
   null is the same procedure on `B` permutations of the target, permuted
   within groups or consecutive time blocks when given, and the p-value is
   `(1 + #{null >= observed}) / (B + 1)`, the exact permutation p-value for
   randomly drawn permutations (Phipson and Smyth, 2010, Stat. Appl. Genet.
   Mol. Biol. 9(1): the formula was read in the arXiv version, 1603.05766).
   A one-sided lower confidence bound on the skill comes from a bootstrap of
   the out-of-fold predictions (clustered by group when groups are given).
   Folds share training rows, so the interval is approximate; its coverage is
   not claimed, only the measured size and recall of the decision rule.
   A feature is a *suspect* when the p-value is at most `alpha` AND the
   lower bound exceeds the plausibility ceiling `skill_ceiling`, and only
   when there are enough rows (otherwise `low_power`, never a suspect).
2. Conditional permutation test. The feature is permuted within strata of the
   other admitted features, then the same single-feature skill is compared
   with its permuted distribution. The strata are the leaves of a small tree
   that predicts the feature from the others, a
   crude estimate of its conditional distribution. This follows the idea of
   Berrett, Wang, Barber and Samworth (2020, JRSS-B 82(1): 175-197, the
   conditional permutation test: permute the feature in a way that respects
   its dependence on the other covariates). It is a simplification: strata
   instead of their MCMC sampler of permutations, and it has not been
   validated against the paper's sampler. That the paper's abstract
   describes the method was verified; the equivalence of this
   approximation was not.
3. Missingness as leakage. When values are recorded only for some outcomes,
   the missing-value indicator is tested against the target (binned to
   quartiles when continuous) with Fisher's exact test for a 2x2 table with
   an expected count below 5, otherwise Pearson's chi-square without
   continuity correction, plus Cramer's V.

Guarantees
----------
Under exchangeability within the permutation blocks, the permutation
p-value is valid, so the rate of calling a signal-free feature a suspect is
at most `alpha`, and in practice far lower because the lower bound must also
clear the ceiling. Nothing is claimed about power against leaks weaker than
the ceiling. The measured operating characteristics are in the certificate
(`benchmarks.qualification.f4_certify`); until a current certificate passes,
`gate_status` is descriptive and decisions are demoted to warnings.
"""

from collections.abc import Hashable
from collections.abc import Sequence
import dataclasses
import math
import sys
from typing import Any

import numpy
import pandas
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor

from scmle.contracts import qualification
from scmle.core import randomness
from scmle.data.detectors import certificates
from scmle.data.ledger import budget

DETECTOR_ID = "f4-leakage-skill"
LEAK_DEFECTS = (
    "leak_copy",
    "leak_noisy_copy",
    "leak_threshold",
    "leak_missing_for_class",
)
TASKS = ("classification", "regression")
MAX_LEVELS = 50
EXPECTED_FISHER = 5.0
TARGET_BINS = 4
MIN_STRATUM_ROWS = 2
SKILL = "leakage.skill"
CONDITIONAL = "leakage.conditional"
MISSING = "leakage.missingness"
MIN_CLASSES = 2
BINARY_CLASSES = 2

Splits = list[tuple[numpy.ndarray, numpy.ndarray]]


@dataclasses.dataclass(frozen=True, slots=True)
class Config:
    """Pre-registered settings of the leakage tests.

    Attributes:
        skill_ceiling: Highest skill (AUC or R^2) plausible for a legitimate
            single feature in this domain. Set by the charter; no default.
        association_ceiling: Cramer's V of missingness against the target
            above which missingness is suspect. Set by the charter.
        alpha: Level of the permutation tests.
        permutations: `B`; at least `budget.MIN_PERMUTATIONS` and at least
            `1 / alpha - 1`, so the level is reachable.
        folds: Cross-fitting folds.
        confidence: Level of the one-sided lower bound.
        bootstrap: Bootstrap draws for the lower bound.
        min_rows: Fewer rows than this is reported as low power.
        max_leaf_nodes: Size of the small tree.
        min_samples_leaf: Smallest leaf of the small tree.
        n_strata: Leaves of the tree that defines strata.
        run_seed: Run seed for every random stream.
    """

    skill_ceiling: float
    association_ceiling: float
    alpha: float = 0.05
    permutations: int = 199
    folds: int = 5
    confidence: float = 0.95
    bootstrap: int = 200
    min_rows: int = 100
    max_leaf_nodes: int = 8
    min_samples_leaf: int = 5
    n_strata: int = 8
    run_seed: int = 0

    def __post_init__(self) -> None:
        """Validates the settings.

        Raises:
            ValueError: If a ceiling is outside (0, 1], `alpha` or
                `confidence` is outside (0, 1), `permutations` is too few
                to reach `alpha` (`permutations + 1 < 1 / alpha`), or a
                count is not an integer at or above its minimum.
        """
        for name in ("skill_ceiling", "association_ceiling"):
            value = qualification.finite(getattr(self, name), name)
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be within (0, 1]")
        for name in ("alpha", "confidence"):
            value = qualification.finite(getattr(self, name), name)
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} must be within (0, 1)")
        qualification.count(
            self.permutations, "permutations", budget.MIN_PERMUTATIONS
        )
        if self.permutations + 1 < 1.0 / self.alpha:
            raise ValueError(
                "permutations too few to reach alpha: need B + 1 >= 1 / alpha"
            )
        qualification.count(self.folds, "folds", minimum=2)
        qualification.count(self.bootstrap, "bootstrap", minimum=2)
        qualification.count(self.min_rows, "min_rows", minimum=1)
        qualification.count(self.max_leaf_nodes, "max_leaf_nodes", minimum=2)
        qualification.count(self.min_samples_leaf, "min_samples_leaf", 1)
        qualification.count(self.n_strata, "n_strata", minimum=2)
        qualification.count(self.run_seed, "run_seed")


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


def encode_target(y: pandas.Series, task: str) -> numpy.ndarray:
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
        raise ValueError(f"task must be one of {TASKS}")
    if y.isna().any():
        raise ValueError("the target must not have missing values")
    if task == "regression":
        return y.to_numpy(dtype="float64")
    codes, uniques = pandas.factorize(y, sort=True)
    if len(uniques) < MIN_CLASSES:
        raise ValueError("a classification target needs two classes")
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


def make_splits(
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


def binary_auc(truth: numpy.ndarray, score: numpy.ndarray) -> float:
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
        raise ValueError("both classes are needed for an AUC")
    ranks = stats.rankdata(score)
    return float((ranks[positive].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def skill_of(pred: numpy.ndarray, y: numpy.ndarray, task: str) -> float:
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
            raise ValueError("a constant target has no R^2")
        return 1.0 - float(((y - pred) ** 2).sum()) / total
    if pred.shape[1] == BINARY_CLASSES:
        return binary_auc(y, pred[:, 1])
    return float(
        roc_auc_score(
            y,
            pred,
            multi_class="ovr",
            average="macro",
            labels=list(range(pred.shape[1])),
        )
    )


def cross_fit_skill(
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
    splits = make_splits(y, task, cfg.folds, groups, block_size, seed)
    pred = out_of_fold(x, y, task, splits, cfg, seed)
    return skill_of(pred, y, task), pred


def bootstrap_skill(
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
        return skill_of(pred[index], y[index], task)
    except ValueError:
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
    draws = [
        bootstrap_skill(pred, y, task, members, rng)
        for draw in range(cfg.bootstrap)
    ]
    values = [v for v in draws if v is not None]
    if not values:
        return -math.inf
    return float(numpy.quantile(values, 1.0 - cfg.confidence))


def permutation_p(observed: float, null: Sequence[float]) -> float:
    """Computes `(1 + #{null >= observed}) / (B + 1)` (Phipson and Smyth, 2010).

    Args:
        observed: The observed statistic.
        null: Statistics under permutations of the target.

    Returns:
        The permutation p-value, in (0, 1].
    """
    exceed = sum(1 for value in null if value >= observed)
    return (1.0 + exceed) / (len(null) + 1.0)


def enough_rows(y: numpy.ndarray, task: str, cfg: Config) -> bool:
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
        return min(numpy.bincount(y).tolist()) >= cfg.folds
    return True


@dataclasses.dataclass(frozen=True, slots=True)
class SkillTest:
    """Single-feature skill against a permutation null.

    Attributes:
        feature: Feature name.
        skill: Observed out-of-fold skill.
        null_mean: Mean skill over the permuted targets.
        lower_bound: One-sided lower bound on the skill.
        p_value: Permutation p-value.
        n: Rows.
        low_power: Too few rows; the feature is never a suspect.
        suspect: There are enough rows, p <= alpha and the lower bound is
            above the ceiling.
        check: Ledger record.
    """

    feature: str
    skill: float
    null_mean: float
    lower_bound: float
    p_value: float
    n: int
    low_power: bool
    suspect: bool
    check: qualification.Check


def decision_for(
    suspect: bool, low_power: bool, gate: certificates.GateStatus
) -> qualification.Decision:
    """Chooses the decision of a leakage test, demoted when not certified.

    Args:
        suspect: Whether the test called the feature a suspect.
        low_power: Whether there were too few rows.
        gate: Whether the detector may gate.

    Returns:
        `quarantine` for a suspect, `warn` for low power, `pass` otherwise;
        a quarantine is lowered to a warning unless `gate` is gating.
    """
    if suspect:
        raw = qualification.Decision.QUARANTINE
    elif low_power:
        raw = qualification.Decision.WARN
    else:
        raw = qualification.Decision.PASS
    return certificates.demote(raw, gate)


def skill_test(
    feature: pandas.Series,
    target: pandas.Series,
    task: str,
    cfg: Config,
    groups: pandas.Series | None = None,
    block_size: int | None = None,
    gate: certificates.GateStatus = certificates.GateStatus.DESCRIPTIVE,
    set_id: qualification.SetId = qualification.SetId.EXPLORATION,
) -> SkillTest:
    """Tests whether one feature predicts the target beyond chance.

    Args:
        feature: The feature, aligned to `target`.
        target: The outcome.
        task: `classification` or `regression`.
        cfg: Settings.
        groups: Group label per row; folds and permutations respect it.
        block_size: Consecutive rows per time block (permute within blocks
            and use contiguous folds); exclusive with `groups`.
        gate: Whether the detector has a current certificate.
        set_id: Set the rows belong to (the exploration set for anything
            touching the target).

    Returns:
        The test, with a ledger `Check`. The feature is a suspect only when
        there are enough rows (see `enough_rows`); with too few it is
        `low_power` and never a suspect. The check's interval is None when
        the lower bound is -inf.

    Raises:
        ValueError: If both `groups` and `block_size` are given, `feature`
            and `target` differ in length, or the target is invalid (see
            `encode_target`).
    """
    if groups is not None and block_size is not None:
        raise ValueError("use groups or block_size, not both")
    if len(feature) != len(target):
        raise ValueError("feature and target must have the same length")
    name = str(feature.name)
    y = encode_target(target, task)
    group_values = None if groups is None else groups.to_numpy()
    x = design(feature)
    path = f"{SKILL}.{name}"
    seed = randomness.derive_int_seed(cfg.run_seed, path, 0)
    skill, pred = cross_fit_skill(
        x, y, task, cfg, group_values, block_size, seed
    )
    rng = randomness.component_rng(cfg.run_seed, path, 1)
    null = [
        cross_fit_skill(
            x,
            y[permute_within(y, rng, group_values, block_size)],
            task,
            cfg,
            group_values,
            block_size,
            seed,
        )[0]
        for draw in range(cfg.permutations)
    ]
    p_value = permutation_p(skill, null)
    bound = lower_bound(
        pred,
        y,
        task,
        cfg,
        group_values,
        randomness.component_rng(cfg.run_seed, path, 2),
    )
    powered = enough_rows(y, task, cfg)
    suspect = powered and p_value <= cfg.alpha and bound > cfg.skill_ceiling
    check = qualification.Check(
        check_id=path,
        null_hypothesis=(
            "the feature has no predictive skill beyond a target permuted "
            "within its blocks"
        ),
        statistic=skill,
        p_value=p_value,
        interval=(bound, max(skill, bound)) if math.isfinite(bound) else None,
        effect=skill - float(numpy.mean(null)),
        n=len(y),
        decision=decision_for(suspect, not powered, gate),
        set_id=set_id,
        seed_path=randomness.seed_path(cfg.run_seed, path, 1),
        tier=qualification.Tier.UNVERIFIED,
    )
    return SkillTest(
        name,
        skill,
        float(numpy.mean(null)),
        bound,
        p_value,
        len(y),
        not powered,
        suspect,
        check,
    )


def strata_labels(
    feature: pandas.Series, others: pandas.DataFrame, cfg: Config
) -> numpy.ndarray:
    """Strata for the conditional permutation: leaves of a small tree.

    The tree predicts the feature from the other features (a crude estimate
    of its conditional mean or class): a regression tree for a numeric,
    non-boolean feature, whose missing values are filled with the median for
    the fit, and a classification tree otherwise. The strata depend only on
    the other features, so missingness of the feature is not itself a
    stratum: a feature recorded only for some outcomes must be able to move.
    With no other features there is one stratum.

    Args:
        feature: The feature to be permuted.
        others: The other admitted features; may have no columns.
        cfg: Settings (`n_strata`, `min_samples_leaf`).

    Returns:
        An integer label per row (the tree leaf of the row, at most
        `n_strata` distinct values).
    """
    if others.shape[1] == 0:
        return numpy.zeros(len(feature), dtype="int64")
    z = numpy.hstack([design(others[c]) for c in others.columns])
    if pandas.api.types.is_numeric_dtype(feature) and not (
        pandas.api.types.is_bool_dtype(feature)
    ):
        values = feature.to_numpy(dtype="float64")
        filled = numpy.where(
            numpy.isnan(values), numpy.nanmedian(values), values
        )
        tree: Any = DecisionTreeRegressor(
            max_leaf_nodes=cfg.n_strata,
            min_samples_leaf=cfg.min_samples_leaf,
            random_state=0,
        ).fit(z, filled)
    else:
        codes = pandas.factorize(feature.astype("object"))[0]
        tree = DecisionTreeClassifier(
            max_leaf_nodes=cfg.n_strata,
            min_samples_leaf=cfg.min_samples_leaf,
            random_state=0,
        ).fit(z, codes)
    return tree.apply(z).astype("int64")


def within_strata(
    n: int, members: list[numpy.ndarray], rng: numpy.random.Generator
) -> numpy.ndarray:
    """Draws a row order that permutes rows within each stratum.

    Args:
        n: Number of rows.
        members: Row positions of each stratum.
        rng: Random generator.

    Returns:
        Integer positions; rows move only within their stratum.
    """
    order = numpy.arange(n)
    for where in members:
        order[where] = where[rng.permutation(len(where))]
    return order


@dataclasses.dataclass(frozen=True, slots=True)
class ConditionalTest:
    """Conditional permutation test of one feature.

    Attributes:
        feature: Feature name.
        skill: Observed single-feature skill.
        null_mean: Mean skill over stratum-preserving permutations.
        p_value: Permutation p-value.
        strata: Number of strata used.
        keeps_signal: p <= alpha: the feature keeps predictive value given
            the others, to the extent the strata capture them.
        check: Ledger record.
    """

    feature: str
    skill: float
    null_mean: float
    p_value: float
    strata: int
    keeps_signal: bool
    check: qualification.Check


def conditional_test(
    feature: pandas.Series,
    others: pandas.DataFrame,
    target: pandas.Series,
    task: str,
    cfg: Config,
    groups: pandas.Series | None = None,
    block_size: int | None = None,
    set_id: qualification.SetId = qualification.SetId.EXPLORATION,
) -> ConditionalTest:
    """Tests whether the feature keeps predictive value given the others.

    Args:
        feature: The feature under test.
        others: The other admitted features (the conditioning set).
        target: The outcome.
        task: `classification` or `regression`.
        cfg: Settings.
        groups: Group label per row, for the cross-fitting folds.
        block_size: Consecutive rows per time block, for the folds.
        set_id: Set the rows belong to.

    Returns:
        The test. It never gates: it only supports an escalation.

    Raises:
        ValueError: If both `groups` and `block_size` are given, or the
            target is invalid (see `encode_target`).
    """
    if groups is not None and block_size is not None:
        raise ValueError("use groups or block_size, not both")
    name = str(feature.name)
    y = encode_target(target, task)
    group_values = None if groups is None else groups.to_numpy()
    x = design(feature)
    path = f"{CONDITIONAL}.{name}"
    seed = randomness.derive_int_seed(cfg.run_seed, path, 0)
    skill = cross_fit_skill(x, y, task, cfg, group_values, block_size, seed)[0]
    labels = strata_labels(feature, others, cfg)
    members = [numpy.flatnonzero(labels == v) for v in numpy.unique(labels)]
    rng = randomness.component_rng(cfg.run_seed, path, 1)
    null = [
        cross_fit_skill(
            x[within_strata(len(y), members, rng)],
            y,
            task,
            cfg,
            group_values,
            block_size,
            seed,
        )[0]
        for draw in range(cfg.permutations)
    ]
    p_value = permutation_p(skill, null)
    keeps = p_value <= cfg.alpha
    check = qualification.Check(
        check_id=path,
        null_hypothesis=(
            "given the other features, the feature has no predictive skill "
            "beyond permutation within strata"
        ),
        statistic=skill,
        p_value=p_value,
        interval=None,
        effect=skill - float(numpy.mean(null)),
        n=len(y),
        decision=(
            qualification.Decision.WARN
            if keeps
            else qualification.Decision.PASS
        ),
        set_id=set_id,
        seed_path=randomness.seed_path(cfg.run_seed, path, 1),
        tier=qualification.Tier.UNVERIFIED,
    )
    return ConditionalTest(
        name,
        skill,
        float(numpy.mean(null)),
        p_value,
        len(members),
        keeps,
        check,
    )


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
    statistic = float(stats.chi2_contingency(table, correction=False)[0])
    return math.sqrt(statistic / (total * smaller))


def binned_target(target: pandas.Series, task: str) -> pandas.Series:
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
class MissingnessTest:
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
    gate: certificates.GateStatus = certificates.GateStatus.DESCRIPTIVE,
    set_id: qualification.SetId = qualification.SetId.EXPLORATION,
) -> MissingnessTest | None:
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
        raise ValueError("feature and target must have the same length")
    missing = feature.isna().to_numpy()
    if missing.all() or not missing.any():
        return None
    classes = binned_target(target, task)
    table = pandas.crosstab(missing, classes).to_numpy()
    if table.shape[1] < MIN_CLASSES:
        return None
    if table.shape == (2, 2) and (
        stats.contingency.expected_freq(table).min() < EXPECTED_FISHER
    ):
        method = "fisher"
        p_value = float(stats.fisher_exact(table).pvalue)
    else:
        method = "chi2"
        p_value = float(stats.chi2_contingency(table, correction=False)[1])
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
        decision=decision_for(suspect, not powered, gate),
        set_id=set_id,
        seed_path=randomness.seed_path(cfg.run_seed, path, 0),
        tier=qualification.Tier.UNVERIFIED,
    )
    return MissingnessTest(
        str(feature.name),
        method,
        p_value,
        v,
        int(missing.sum()),
        len(feature),
        suspect,
        check,
    )


@dataclasses.dataclass(frozen=True, slots=True)
class FeatureAssessment:
    """All leakage tests for one feature.

    Attributes:
        feature: Feature name.
        skill: Skill test.
        conditional: Conditional test, run for suspects.
        missingness: Missingness test, when the feature has gaps.
        suspect: Either the skill or the missingness test called it one.
        escalated: A suspect that also keeps its signal given the others.
    """

    feature: str
    skill: SkillTest
    conditional: ConditionalTest | None
    missingness: MissingnessTest | None
    suspect: bool
    escalated: bool

    @property
    def checks(self) -> tuple[qualification.Check, ...]:
        """Every ledger record of this feature."""
        parts = (self.skill, self.conditional, self.missingness)
        return tuple(p.check for p in parts if p is not None)


def assess_feature(
    feature: pandas.Series,
    others: pandas.DataFrame,
    target: pandas.Series,
    task: str,
    cfg: Config,
    groups: pandas.Series | None = None,
    block_size: int | None = None,
    gate: certificates.GateStatus = certificates.GateStatus.DESCRIPTIVE,
    set_id: qualification.SetId = qualification.SetId.EXPLORATION,
) -> FeatureAssessment:
    """Runs the skill, missingness and (for suspects) conditional tests.

    Args:
        feature: The feature.
        others: The other admitted features.
        target: The outcome.
        task: `classification` or `regression`.
        cfg: Settings.
        groups: Group label per row.
        block_size: Consecutive rows per time block.
        gate: Whether the detector has a current certificate.
        set_id: Set the rows belong to.

    Returns:
        The assessment. A feature is a suspect when its skill test or its
        missingness test says so, and is escalated when the conditional
        test, run for suspects only, finds it keeps its signal.
    """
    skill = skill_test(
        feature, target, task, cfg, groups, block_size, gate, set_id
    )
    gaps = missingness_test(feature, target, task, cfg, gate, set_id)
    suspect = skill.suspect or (gaps is not None and gaps.suspect)
    conditional = (
        conditional_test(
            feature, others, target, task, cfg, groups, block_size, set_id
        )
        if suspect
        else None
    )
    escalated = conditional is not None and conditional.keeps_signal and suspect
    return FeatureAssessment(
        str(feature.name), skill, conditional, gaps, suspect, escalated
    )


@dataclasses.dataclass(frozen=True, slots=True)
class Assessments:
    """Leakage tests for many features, reported as suspicions only.

    Attributes:
        assessments: One per feature, in the order given.
        gate: Whether decisions were allowed to quarantine.
    """

    assessments: tuple[FeatureAssessment, ...]
    gate: certificates.GateStatus

    @property
    def suspects(self) -> dict[str, bool]:
        """Suspicion per feature."""
        return {a.feature: a.suspect for a in self.assessments}

    @property
    def checks(self) -> tuple[qualification.Check, ...]:
        """Every ledger record."""
        return tuple(c for a in self.assessments for c in a.checks)

    def to_dict(self) -> dict[str, Any]:
        """Converts the report to a JSON-ready document.

        A suspicion is not a finding, and the document says so.

        Returns:
            A dict with a note, the gate value and, per feature, the
            suspect, escalated and low-power flags, the skill, its lower
            bound (None when not finite) and p-value, the conditional and
            missingness p-values and the missingness Cramer's V (each None
            when that test did not run).
        """
        return {
            "note": "statistical suspicion only; not proof of leakage",
            "gate": self.gate.value,
            "features": [
                {
                    "feature": a.feature,
                    "suspect": a.suspect,
                    "escalated": a.escalated,
                    "skill": a.skill.skill,
                    "skill_lower_bound": (
                        a.skill.lower_bound
                        if math.isfinite(a.skill.lower_bound)
                        else None
                    ),
                    "skill_p_value": a.skill.p_value,
                    "low_power": a.skill.low_power,
                    "conditional_p_value": None
                    if a.conditional is None
                    else a.conditional.p_value,
                    "missingness_p_value": None
                    if a.missingness is None
                    else a.missingness.p_value,
                    "missingness_cramers_v": None
                    if a.missingness is None
                    else a.missingness.cramers_v,
                }
                for a in self.assessments
            ],
        }


def scan(
    frame: pandas.DataFrame,
    target: str,
    task: str,
    cfg: Config,
    features: Sequence[Hashable] | None = None,
    groups: pandas.Series | None = None,
    block_size: int | None = None,
    gate: certificates.GateStatus = certificates.GateStatus.DESCRIPTIVE,
    set_id: qualification.SetId = qualification.SetId.EXPLORATION,
) -> Assessments:
    """Tests every feature of the exploration set for leakage suspicion.

    Args:
        frame: Exploration-set data including the target.
        target: Name of the target column.
        task: `classification` or `regression`.
        cfg: Settings.
        features: Columns to test; defaults to all but the target.
        groups: Group label per row.
        block_size: Consecutive rows per time block.
        gate: Whether the detector has a current certificate.
        set_id: Set the rows belong to.

    Returns:
        The report, with one assessment per tested feature in the order
        given, recording `gate` as passed (see `gate_status` for deriving
        it from the certificates).

    Raises:
        ValueError: If `features` names a column that is missing or is the
            target, or a test rejects its inputs.
    """
    names = [c for c in frame.columns if c != target]
    chosen = names if features is None else list(features)
    unknown = [c for c in chosen if c not in names]
    if unknown:
        raise ValueError(f"not feature columns: {unknown}")
    outcomes = frame[target]
    assessments = tuple(
        assess_feature(
            frame[name],
            frame[[c for c in chosen if c != name]],
            outcomes,
            task,
            cfg,
            groups,
            block_size,
            gate,
            set_id,
        )
        for name in chosen
    )
    return Assessments(assessments, gate)


CERTIFIED = Config(skill_ceiling=0.95, association_ceiling=0.9)


def gate_status(
    store: certificates.Store,
    cfg: Config,
    task: str = "classification",
    n_classes: int = 2,
) -> certificates.GateStatus:
    """Decides whether the skill test may quarantine, from the certificates.

    Gating needs a current certificate whose every required leak type
    passed, and settings equal to the certified ones (`CERTIFIED`, any run
    seed). Only a binary classification target was certified; any other
    target is descriptive.

    Args:
        store: Certificates on file.
        cfg: The settings the tests will run with.
        task: `classification` or `regression`.
        n_classes: Classes of a classification target.

    Returns:
        `GATING` or `DESCRIPTIVE`.
    """
    if task != "classification" or n_classes != BINARY_CLASSES:
        return certificates.GateStatus.DESCRIPTIVE
    if dataclasses.replace(cfg, run_seed=CERTIFIED.run_seed) != CERTIFIED:
        return certificates.GateStatus.DESCRIPTIVE
    digest = code_digest()
    for defect in LEAK_DEFECTS:
        if store.status(DETECTOR_ID, digest, defect) is not (
            certificates.GateStatus.GATING
        ):
            return certificates.GateStatus.DESCRIPTIVE
    return certificates.GateStatus.GATING


def code_digest() -> str:
    """Hashes this module's source, the key of its certificate.

    Returns:
        The code hash of this module, as computed by
        `certificates.code_hash`.
    """
    return certificates.code_hash(sys.modules[__name__])
