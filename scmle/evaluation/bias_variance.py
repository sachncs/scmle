"""Bias and variance of the selected model, estimated by bootstrapping.

The same model is refit on many bootstrap resamples of the development rows.
Each row is predicted only by the refits that did not train on it (its
out-of-bag refits), and the spread of those predictions is the model's
variance there; the distance of their consensus from the truth is its bias.

* Regression, squared error: the consensus is the mean prediction. Variance
  is the mean prediction spread; `bias_plus_noise` is the squared distance of
  the consensus from the observed value. Noise cannot be separated from bias
  without extra assumptions, so the two are reported together.
* Classification, 0-1 error (Domingos): the consensus is the most common
  predicted label. Bias is the share of rows where it is wrong; variance is
  the share of individual predictions that disagree with it.

Both bias terms contain the data's irreducible noise, which is estimated from
nearest neighbours and removed before the verdict is given.

Limitation: resampling one fixed dataset reproduces the variance caused by
*which rows are drawn*, not the variance of the rows' own noise. Noise that is
baked into every resample is counted as bias, so a very flexible model's
variance is understated and its bias overstated. Read the numbers as a
direction (does a more or less flexible model help?), and compare candidates
rather than trusting one absolute split.
"""

from collections.abc import Callable
import dataclasses
from typing import Any

import numpy as np
import pandas as pd
from sklearn import base
from sklearn import neighbors

MIN_PREDICTIONS = 3
MIN_NOISE_ROWS = 50
VARIANCE_DOMINATED = 0.6
BIAS_DOMINATED = 0.4
ADVICE = {
    "variance-dominated": (
        "the model is unstable across resamples: more data, stronger "
        "regularization, bagging or a simpler model should help"
    ),
    "bias-dominated": (
        "the model is stable but systematically off: more capacity, better "
        "features or a more flexible family should help"
    ),
    "balanced": "bias and variance contribute about equally",
}


@dataclasses.dataclass(frozen=True)
class Decomposition:
    """Estimated components of the error.

    Attributes:
        kind: `squared_error` or `zero_one`.
        replicates: Bootstrap refits.
        rows: Rows predicted by enough out-of-bag refits.
        bias_plus_noise: Distance of the consensus from the observed values
            (squared for regression, 0-1 for classes); includes noise.
        noise: Estimated irreducible error (None when it cannot be
            estimated).
        bias: Reducible bias: `bias_plus_noise` minus `noise`, floored at 0
            (equal to `bias_plus_noise` when noise is unknown).
        variance: Variance term.
        total: Mean error of the individual out-of-bag predictions.
        variance_share: Variance as a share of bias plus variance.
        verdict: `variance-dominated`, `bias-dominated` or `balanced`.
        advice: What the verdict suggests.
    """

    kind: str
    replicates: int
    rows: int
    bias_plus_noise: float
    noise: float | None
    bias: float
    variance: float
    total: float
    variance_share: float
    verdict: str
    advice: str

    def to_dict(self) -> dict[str, Any]:
        """Returns the JSON-ready record.

        Returns:
            A mapping of every field of the decomposition.
        """
        return dataclasses.asdict(self)


def noise_floor(
    features: pd.DataFrame, target: pd.Series, classification: bool
) -> float | None:
    """Estimates the irreducible error from nearest neighbours.

    Two rows with almost the same features differ in the target only by
    noise (for a smooth relationship), so half the mean squared difference to
    the nearest neighbour estimates the noise variance. For classes, the
    1-nearest-neighbour error `e` bounds the Bayes error `b` through
    `e = 2b(1-b)` (Cover and Hart). Only numeric columns are used; with high
    dimension the estimate is biased upward, so treat it as a rough guide.

    Args:
        features: Candidate feature columns; only numeric ones are used.
        target: Target aligned with `features`.
        classification: Whether the target is a class label.

    Returns:
        The estimated noise (the Bayes error for classes, the noise variance
        otherwise), or None when there are no numeric features or fewer than
        `MIN_NOISE_ROWS` rows.
    """
    numeric = features.select_dtypes("number")
    if numeric.shape[1] == 0 or len(numeric) < MIN_NOISE_ROWS:
        return None
    matrix = numeric.fillna(numeric.median()).to_numpy(dtype=float)
    spread = np.where(matrix.std(axis=0) > 0, matrix.std(axis=0), 1.0)
    scaled = (matrix - matrix.mean(axis=0)) / spread
    neighbours = neighbors.NearestNeighbors(n_neighbors=2).fit(scaled)
    nearest = neighbours.kneighbors(scaled, return_distance=False)[:, 1]
    truth = target.to_numpy()
    if classification:
        error = float(np.mean(truth != truth[nearest]))
        return float((1.0 - np.sqrt(max(1.0 - 2.0 * error, 0.0))) / 2.0)
    gap = truth.astype(float) - truth[nearest].astype(float)
    return float(np.mean(gap**2) / 2.0)


def verdict_of(variance: float, bias: float) -> str:
    """Names the dominant source of error.

    Args:
        variance: The variance term.
        bias: The reducible bias term.

    Returns:
        `variance-dominated` when variance is at least `VARIANCE_DOMINATED`
        of their sum, `bias-dominated` when at most `BIAS_DOMINATED`,
        otherwise `balanced` (also when the sum is not positive).
    """
    share = variance / (variance + bias) if variance + bias > 0 else 0.5
    if share >= VARIANCE_DOMINATED:
        return "variance-dominated"
    if share <= BIAS_DOMINATED:
        return "bias-dominated"
    return "balanced"


def collect(
    build: Callable[[], base.BaseEstimator],
    features: pd.DataFrame,
    target: pd.Series,
    replicates: int,
    seed: int,
    should_stop: Callable[[], bool],
) -> list[list[Any]] | None:
    """Out-of-bag predictions per row from bootstrap refits.

    Args:
        build: Creates a fresh unfitted estimator.
        features: Development rows.
        target: Target aligned with `features`.
        replicates: Number of bootstrap refits.
        seed: Seed of the resampling.
        should_stop: Polled before each refit; True ends the work.

    Returns:
        For each row, the predictions made by the refits that did not train
        on it, or None when `should_stop` returned True.
    """
    rows = len(features)
    predictions: list[list[Any]] = [[] for position in range(rows)]
    for draw in range(replicates):
        if should_stop():
            return None
        drawn = np.random.default_rng([seed, draw]).integers(0, rows, rows)
        held = np.setdiff1d(np.arange(rows), drawn)
        if len(held) == 0:
            continue
        model = build().fit(features.iloc[drawn], target.iloc[drawn])
        guesses = np.asarray(model.predict(features.iloc[held]))
        for position, guess in zip(held, guesses, strict=True):
            predictions[position].append(guess)
    return predictions


def decompose(
    build: Callable[[], base.BaseEstimator],
    features: pd.DataFrame,
    target: pd.Series,
    classification: bool,
    replicates: int = 20,
    seed: int = 0,
    should_stop: Callable[[], bool] = lambda: False,
) -> Decomposition | None:
    """Estimates bias and variance of a model on the given rows.

    Args:
        build: Creates a fresh unfitted estimator.
        features: Development rows.
        target: Target aligned with `features`.
        classification: Whether to use 0-1 error rather than squared error.
        replicates: Number of bootstrap refits.
        seed: Seed of the resampling.
        should_stop: Polled before each refit; True ends the work.

    Returns:
        The decomposition, or None when time ran out or too few rows were
        predicted by enough refits.
    """
    predictions = collect(
        build, features, target, replicates, seed, should_stop
    )
    if predictions is None:
        return None
    truth = target.to_numpy()
    bias, variance, total, used = [], [], [], 0
    for position, guesses in enumerate(predictions):
        if len(guesses) < MIN_PREDICTIONS:
            continue
        used += 1
        if classification:
            labels, counts = np.unique(np.asarray(guesses), return_counts=True)
            consensus = labels[int(np.argmax(counts))]
            bias.append(float(consensus != truth[position]))
            variance.append(float(np.mean(np.asarray(guesses) != consensus)))
            total.append(float(np.mean(np.asarray(guesses) != truth[position])))
        else:
            values = np.asarray(guesses, dtype=float)
            # The mean of m noisy refits overstates the squared distance to
            # the truth by var / m; remove that finite-replicate excess.
            excess = float(values.var(ddof=1)) / len(values)
            gap = (values.mean() - float(truth[position])) ** 2
            bias.append(float(gap - excess))
            variance.append(float(values.var()))
            total.append(float(np.mean((values - float(truth[position])) ** 2)))
    if used == 0:
        return None
    raw_bias, mean_variance = float(np.mean(bias)), float(np.mean(variance))
    noise = noise_floor(features, target, classification)
    mean_bias = raw_bias if noise is None else max(raw_bias - noise, 0.0)
    verdict = verdict_of(mean_variance, mean_bias)
    return Decomposition(
        "zero_one" if classification else "squared_error",
        replicates,
        used,
        raw_bias,
        noise,
        mean_bias,
        mean_variance,
        float(np.mean(total)),
        mean_variance / (mean_variance + mean_bias)
        if mean_variance + mean_bias > 0
        else 0.5,
        verdict,
        ADVICE[verdict],
    )
