"""Missingness and record-purity statistics and their lower bounds."""

import math

import numpy
import pandas
from scipy import stats

from scmle.assurance.admissibility import leakage
from scmle.assurance.admissibility.screen.settings import Statistic


def chisquare(
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


def cramer(missing: numpy.ndarray, totals: numpy.ndarray) -> Statistic:
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
        return numpy.sqrt(chisquare(missing, labels, totals) / n)

    return compute


def purge(
    share: numpy.ndarray, base: numpy.ndarray, classes: int
) -> numpy.ndarray:
    """Standardizes a class's share of the recorded rows to equal classes.

    A value recorded only for one outcome holds back every row of the
    others; a value recorded at random holds back none, and its commonest
    class keeps the table's base rate, which on an imbalanced table (98
    percent negatives) is already above any absolute ceiling. The share the
    recorded values hold back of the other classes' expected rows,
    `(1 - share) / (1 - base)` left in, is carried to a table of `classes`
    equal classes (direct standardization): `1 - (1 - share) (1 - 1 / K) /
    (1 - base)`. On a table of equal classes it is the share itself, so the
    association ceiling keeps its meaning there; it is 1 when only that
    class is recorded, whatever the base rate.

    Args:
        share: The class's share of the recorded rows.
        base: Its share of all rows.
        classes: Classes of the outcome, `K`.

    Returns:
        The standardized share; 0 where the class is every row (no other
        class can be held back).
    """
    rest = 1.0 - base
    whole = numpy.isclose(rest, 0.0)
    safe = numpy.where(whole, 1.0, rest)
    spread = 1.0 - 1.0 / max(classes, 1)
    return numpy.where(whole, 0.0, 1.0 - (1.0 - share) * spread / safe)


def assay(recorded: numpy.ndarray) -> Statistic:
    """Builds the record-purity statistic.

    Args:
        recorded: Boolean array of length n, True where the value is
            recorded.

    Returns:
        A function mapping an array of shape (B, n) of class-code versions
        to a length-B array holding, for each, the largest share of one
        class among the recorded rows, standardized to equal classes
        (`purge`).
    """
    flags = recorded.astype("float64")
    count = float(flags.sum())

    def compute(labels: numpy.ndarray) -> numpy.ndarray:
        k = int(labels.max()) + 1
        shares = [
            purge((labels == c) @ flags / count, (labels == c).mean(axis=-1), k)
            for c in range(k)
        ]
        return numpy.max(shares, axis=0)  # type: ignore[no-any-return, unused-ignore]  # Any on older NumPy stubs

    return compute


def dominance(
    recorded: numpy.ndarray,
    classes: numpy.ndarray,
    cfg: leakage.Config,
    groups: numpy.ndarray | None,
) -> float:
    """One-sided lower bound of the record purity (`assay`).

    The class is the one whose standardized share is largest; its share
    among recorded rows is bounded by Clopper-Pearson and standardized to
    equal classes (`purge`, increasing in the share, so the bound carries
    over). With groups the number of trials is the number of groups holding
    a recorded row (at most the rows), the conservative count when rows of
    one group move together.

    Args:
        recorded: Boolean array, True where the value is recorded.
        classes: Class code per row.
        cfg: Settings (`confidence`).
        groups: Group label per row, or None.

    Returns:
        The corrected lower bound, or 0.0 when the rounded count of
        successes is zero or the class is every row.
    """
    labels = classes[recorded]
    counts = numpy.bincount(labels, minlength=int(classes.max()) + 1)
    base = numpy.bincount(classes) / len(classes)
    shares = counts / len(labels)
    chosen = int(numpy.argmax(purge(shares, base, len(base))))
    trials = len(labels)
    if groups is not None:
        trials = min(trials, len(pandas.unique(groups[recorded])))
    wins = round(float(shares[chosen]) * trials)
    if wins == 0:
        return 0.0
    low = stats.beta.ppf(1.0 - cfg.confidence, wins, trials - wins + 1)
    return float(
        purge(numpy.asarray(low), numpy.asarray(base[chosen]), len(base))
    )


def resample(
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
        chi = chisquare(sample, labels[None, :], totals)[0]
        values.append(math.sqrt(chi / len(sample)))
    if not values:
        return -math.inf
    return float(numpy.quantile(values, 1.0 - cfg.confidence))
