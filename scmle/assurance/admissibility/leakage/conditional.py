"""Strata and permutations of the conditional skill test, and its result."""

import dataclasses
from typing import Any

import numpy
import pandas
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor

from scmle.assurance.admissibility.leakage.fitting import design
from scmle.assurance.admissibility.leakage.settings import Config
from scmle.contracts import qualification


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
    leaves: numpy.ndarray = tree.apply(z).astype("int64")
    return leaves


def scramble(
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
class Conditional:
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
