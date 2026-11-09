"""How much does the score move when something other than the model changes?

A single score hides three sources of movement, each answered separately:

* Training sensitivity: refit the same model with different random seeds and
  on different bootstrap resamples of the training rows. A large spread means
  the score reflects luck in training, so a "win" by that margin is not one.
* Evaluation sensitivity: score one fitted model on many different random
  held-out splits. The spread is how far the choice of split alone moves the
  metric; differences between models smaller than it should not be trusted.
* Performance sensitivity: score one fitted model while a growing amount of
  noise (in units of each numeric column's standard deviation) is added to
  the held-out features. The slope says how fast performance degrades.

Scorers follow scikit-learn: `scorer(estimator, features, target)` returns a
value where higher is better.
"""

from collections.abc import Callable
import dataclasses
from typing import Any

import numpy as np
import pandas as pd
from sklearn import base
from sklearn import model_selection

MIN_ROWS = 40
NOISE_LEVELS = (0.0, 0.1, 0.25, 0.5, 1.0)

Scorer = Callable[[Any, pd.DataFrame, pd.Series], float]


@dataclasses.dataclass(frozen=True)
class Spread:
    """Distribution of a score over repeated runs.

    Attributes:
        scores: The individual scores.
        mean: Their mean.
        std: Sample standard deviation (0.0 for a single score).
        range: Largest minus smallest score.
    """

    scores: tuple[float, ...]
    mean: float
    std: float
    range: float

    def to_dict(self) -> dict[str, Any]:
        """Returns the JSON-ready record.

        Returns:
            A mapping of the scores, mean, standard deviation and range.
        """
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class SensitivityReport:
    """The three sensitivities of one model on one dataset.

    Attributes:
        seed: Spread over training seeds (same rows, same test rows).
        resample: Spread over bootstrap resamples of the training rows.
        split: Spread over random train/test splits.
        noise: Mean score at each added-noise level.
        decay: Least-squares slope of the score on the noise level, so
            negative when noise hurts.
    """

    seed: Spread
    resample: Spread
    split: Spread
    noise: dict[float, float]
    decay: float

    def to_dict(self) -> dict[str, Any]:
        """Returns the JSON-ready record.

        Returns:
            A mapping of the three spreads as mappings, the noise curve
            keyed by the level as text, and the decay.
        """
        return {
            "seed": self.seed.to_dict(),
            "resample": self.resample.to_dict(),
            "split": self.split.to_dict(),
            "noise": {str(k): v for k, v in self.noise.items()},
            "decay": self.decay,
        }

    def dominant(self) -> str:
        """Returns which source moves the score most: seed, resample or split.

        Returns:
            The name of the spread with the largest standard deviation.
        """
        spreads = {
            "seed": self.seed.std,
            "resample": self.resample.std,
            "split": self.split.std,
        }
        return max(spreads, key=lambda name: spreads[name])


def spread_of(scores: list[float]) -> Spread:
    """Summarizes repeated scores.

    Args:
        scores: The scores of the repeated runs; at least one.

    Returns:
        Their mean, sample standard deviation (0.0 for one score) and range.
    """
    values = np.asarray(scores, dtype=float)
    return Spread(
        tuple(float(v) for v in values),
        float(values.mean()),
        float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        float(values.max() - values.min()),
    )


def reseeded(build: Callable[[], base.BaseEstimator], seed: int) -> Any:
    """Builds a fresh estimator with its random seed set, when it has one.

    Args:
        build: Creates an unfitted estimator.
        seed: Value for its `random_state`, if it has that parameter.

    Returns:
        The new estimator.
    """
    model = build()
    if "random_state" in model.get_params():
        model.set_params(random_state=seed)
    return model


def training_sensitivity(
    build: Callable[[], base.BaseEstimator],
    features: pd.DataFrame,
    target: pd.Series,
    scorer: Scorer,
    repeats: int = 10,
    seed: int = 0,
) -> tuple[Spread, Spread]:
    """Computes the spread over training seeds and bootstrap resamples.

    One fixed split (30% held out) is used so that only training varies. The
    seed spread refits on the same rows with `random_state` 0 to
    `repeats - 1`; the resample spread refits on bootstrap resamples of the
    training rows with `random_state` fixed at `seed`.

    Args:
        build: Creates an unfitted estimator.
        features: All rows' features.
        target: All rows' targets.
        scorer: Scores a fitted estimator; higher is better.
        repeats: Number of fits of each kind.
        seed: Seed of the split and of the resampling.

    Returns:
        (seed spread, resample spread).
    """
    train, test = model_selection.train_test_split(
        np.arange(len(target)), test_size=0.3, random_state=seed
    )
    held = features.iloc[test], target.iloc[test]
    by_seed = []
    by_resample = []
    generator = np.random.default_rng(seed)
    for index in range(repeats):
        model = reseeded(build, index).fit(
            features.iloc[train], target.iloc[train]
        )
        by_seed.append(scorer(model, *held))
        rows = generator.choice(train, size=len(train), replace=True)
        model = reseeded(build, seed).fit(
            features.iloc[rows], target.iloc[rows]
        )
        by_resample.append(scorer(model, *held))
    return spread_of(by_seed), spread_of(by_resample)


def evaluation_sensitivity(
    build: Callable[[], base.BaseEstimator],
    features: pd.DataFrame,
    target: pd.Series,
    scorer: Scorer,
    repeats: int = 10,
    seed: int = 0,
) -> Spread:
    """Computes the spread of the score over random train/test splits.

    Each split holds out 30% of the rows and the estimator's `random_state`
    is fixed at `seed`.

    Args:
        build: Creates an unfitted estimator.
        features: All rows' features.
        target: All rows' targets.
        scorer: Scores a fitted estimator; higher is better.
        repeats: Number of splits.
        seed: Base seed; split `i` uses `seed + i`.

    Returns:
        The spread of the held-out scores.
    """
    scores = []
    for index in range(repeats):
        train, test = model_selection.train_test_split(
            np.arange(len(target)), test_size=0.3, random_state=seed + index
        )
        model = reseeded(build, seed).fit(
            features.iloc[train], target.iloc[train]
        )
        scores.append(scorer(model, features.iloc[test], target.iloc[test]))
    return spread_of(scores)


def noisy(
    features: pd.DataFrame, level: float, generator: np.random.Generator
) -> pd.DataFrame:
    """Adds Gaussian noise of `level` standard deviations to numeric columns.

    Args:
        features: Rows to perturb.
        level: Noise standard deviation, in units of each column's own
            standard deviation.
        generator: Source of the noise.

    Returns:
        A perturbed copy; non-numeric columns are unchanged.
    """
    result = features.copy()
    for name in result.select_dtypes("number").columns:
        scale = float(result[name].std())
        if not np.isfinite(scale):
            scale = 0.0
        result[name] = result[name] + generator.normal(
            0.0, level * scale, len(result)
        )
    return result


def performance_sensitivity(
    model: Any,
    features: pd.DataFrame,
    target: pd.Series,
    scorer: Scorer,
    levels: tuple[float, ...] = NOISE_LEVELS,
    repeats: int = 5,
    seed: int = 0,
) -> tuple[dict[float, float], float]:
    """Scores a fitted model as noise is added to the features.

    Args:
        model: A fitted estimator.
        features: Held-out features.
        target: Held-out targets.
        scorer: Scores a fitted estimator; higher is better.
        levels: Noise levels, in standard deviations of each column.
        repeats: Noise draws averaged at each nonzero level.
        seed: Seed of the noise.

    Returns:
        (mean score per noise level, least-squares slope of score on level).
    """
    generator = np.random.default_rng(seed)
    curve = {
        level: float(
            np.mean(
                [
                    scorer(model, noisy(features, level, generator), target)
                    for unused in range(1 if level == 0 else repeats)
                ]
            )
        )
        for level in levels
    }
    slope = float(np.polyfit(list(curve), list(curve.values()), 1)[0])
    return curve, slope


def analyze(
    build: Callable[[], base.BaseEstimator],
    features: pd.DataFrame,
    target: pd.Series,
    scorer: Scorer,
    repeats: int = 10,
    seed: int = 0,
) -> SensitivityReport:
    """Computes training, evaluation and performance sensitivity of a model.

    The noise curve is computed for a model fitted on the 70% training part of
    the same split as `training_sensitivity`, scored on the held-out 30% with
    the default noise levels and draws.

    Args:
        build: Creates an unfitted estimator.
        features: All rows' features.
        target: All rows' targets.
        scorer: Scores a fitted estimator; higher is better.
        repeats: Number of fits or splits in each repeated measure.
        seed: Base seed.

    Returns:
        The three spreads, the noise curve and its slope.

    Raises:
        ValueError: With fewer than `MIN_ROWS` rows.
    """
    if len(target) < MIN_ROWS:
        raise ValueError(f"sensitivity analysis needs {MIN_ROWS}+ rows")
    by_seed, by_resample = training_sensitivity(
        build, features, target, scorer, repeats, seed
    )
    by_split = evaluation_sensitivity(
        build, features, target, scorer, repeats, seed
    )
    train, test = model_selection.train_test_split(
        np.arange(len(target)), test_size=0.3, random_state=seed
    )
    model = reseeded(build, seed).fit(features.iloc[train], target.iloc[train])
    curve, slope = performance_sensitivity(
        model, features.iloc[test], target.iloc[test], scorer, seed=seed
    )
    return SensitivityReport(by_seed, by_resample, by_split, curve, slope)
