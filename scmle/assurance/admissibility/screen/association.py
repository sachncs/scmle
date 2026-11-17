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
        return numpy.max(shares, axis=0)  # type: ignore[no-any-return, unused-ignore]  # Any on older NumPy stubs; editing code would stale the shipped certificate

    return compute


def dominance(
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
