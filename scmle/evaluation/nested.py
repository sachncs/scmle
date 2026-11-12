"""Nested cross-validation: an unbiased score for the whole tuning procedure.

The cross-validation score of the winning configuration is optimistic: it was
chosen because it scored best on those folds. Nested validation repeats the
choosing inside every outer fold, using only the outer fold's training rows,
and scores the winner on rows the choosing never saw. The gap between the two
numbers is the optimism of the selection.
"""

from collections.abc import Callable
import dataclasses
import itertools
import operator
import random
from typing import Any
from typing import cast

import numpy
import pandas
from sklearn import base
from sklearn import model_selection

from scmle.contracts import models
from scmle.contracts import search

INNER_FOLDS = 3
CONFIGURATIONS = 4
MIN_INNER_ROWS = 30


@dataclasses.dataclass(frozen=True)
class Scores:
    """Outcome of a nested validation.

    Attributes:
        outer_scores: Score of each outer fold's winner on its own test rows
            (larger is better, in the scorer's orientation).
        inner_scores: The winner's inner score in each outer fold.
        mean: Mean outer score, the honest estimate.
        optimism: Mean inner score minus mean outer score.
        folds: Number of outer folds.
        configurations: Configurations compared in each fold.
    """

    outer_scores: tuple[float, ...]
    inner_scores: tuple[float, ...]
    mean: float
    optimism: float
    folds: int
    configurations: int

    def to_dict(self) -> dict[str, Any]:
        """Returns the JSON-ready record.

        Returns:
            A mapping of every field of the result.
        """
        return dataclasses.asdict(self)


def mean_cv_score(
    model: models.Estimator,
    features: pandas.DataFrame,
    target: pandas.Series,
    scorer: Callable[..., float],
    classification: bool,
    seed: int,
) -> float:
    """Mean scorer value over inner folds.

    Args:
        model: Unfitted candidate; cloned for each fold.
        features: Training features.
        target: Training labels.
        scorer: scikit-learn scorer (larger is better).
        classification: Whether the target holds class labels.
        seed: Seed of the inner folds.

    Returns:
        The mean score over the inner folds.
    """
    # Stratify only when every class has enough rows for each inner fold.
    if classification and target.value_counts().min() >= INNER_FOLDS:
        folds = model_selection.StratifiedKFold(
            INNER_FOLDS, shuffle=True, random_state=seed
        )
    else:
        folds = model_selection.KFold(
            INNER_FOLDS, shuffle=True, random_state=seed
        )
    scores = []
    for fit, held in folds.split(features, target):
        fitted = cast("models.Estimator", base.clone(model)).fit(
            features.iloc[fit], target.iloc[fit]
        )
        scores.append(scorer(fitted, features.iloc[held], target.iloc[held]))
    return float(numpy.mean(scores))


def nested_score(
    build: Callable[[dict[str, Any]], models.Estimator],
    space: search.SearchSpace | None,
    features: pandas.DataFrame,
    target: pandas.Series,
    outer_folds: list[tuple[Any, Any]],
    scorer: Callable[..., float],
    classification: bool,
    seed: int = 0,
    configurations: int = CONFIGURATIONS,
    should_stop: Callable[[], bool] = lambda: False,
) -> Scores | None:
    """Runs the tuning procedure inside every outer fold.

    Args:
        build: Makes an unfitted candidate from hyperparameter overrides.
        space: Hyperparameters to search (only the defaults when None).
        features: Development features.
        target: Development labels.
        outer_folds: `(train positions, test positions)` pairs.
        scorer: scikit-learn scorer (larger is better).
        classification: Whether the target holds class labels.
        seed: Random seed for configuration sampling and inner folds.
        configurations: Configurations compared per outer fold, including
            the defaults.
        should_stop: Called before each outer fold; True abandons the run.

    Returns:
        The result, or None when time ran out or no fold had enough rows.
    """
    rng = random.Random(seed)  # nosec B311  # noqa: S311 -- seeded, not security
    outer, inner = [], []
    candidates: list[dict[str, Any]] = []
    for train, test in outer_folds:
        if should_stop():
            return None
        x_train = features.iloc[numpy.asarray(train)]
        y_train = target.iloc[numpy.asarray(train)]
        if len(x_train) < MIN_INNER_ROWS:
            continue
        candidates = [{}]
        if space is not None:
            candidates += map(
                operator.call,
                itertools.repeat(
                    lambda: space.sample(rng), max(configurations - 1, 0)
                ),
            )
        scored = [
            (
                mean_cv_score(
                    build(params),
                    x_train,
                    y_train,
                    scorer,
                    classification,
                    seed,
                ),
                params,
            )
            for params in candidates
        ]
        best_score, best_params = max(scored, key=lambda item: item[0])
        winner = build(best_params).fit(x_train, y_train)
        held = numpy.asarray(test)
        outer.append(
            float(scorer(winner, features.iloc[held], target.iloc[held]))
        )
        inner.append(float(best_score))
    if not outer:
        return None
    return Scores(
        tuple(outer),
        tuple(inner),
        float(numpy.mean(outer)),
        float(numpy.mean(inner) - numpy.mean(outer)),
        len(outer),
        len(candidates),
    )
