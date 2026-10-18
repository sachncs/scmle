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
        std: Sample standard deviation.
        range: Largest minus smallest score.
    """

    scores: tuple[float, ...]
    mean: float
    std: float
    range: float

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready record."""
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class SensitivityReport:
    """The three sensitivities of one model on one dataset.

    Attributes:
        seed: Spread over training seeds (same rows, same test rows).
        resample: Spread over bootstrap resamples of the training rows.
        split: Spread over random train/test splits.
        noise: Mean score at each added-noise level.
        decay: Score lost per unit of added noise (least-squares slope,
            negative when noise hurts).
    """

    seed: Spread
    resample: Spread
    split: Spread
    noise: dict[float, float]
    decay: float

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready record."""
        return {
            "seed": self.seed.to_dict(),
            "resample": self.resample.to_dict(),
            "split": self.split.to_dict(),
            "noise": {str(k): v for k, v in self.noise.items()},
            "decay": self.decay,
        }

    def dominant(self) -> str:
        """Which source moves the score most: seed, resample or split."""
        spreads = {
            "seed": self.seed.std,
            "resample": self.resample.std,
            "split": self.split.std,
        }
        return max(spreads, key=lambda name: spreads[name])


def spread_of(scores: list[float]) -> Spread:
    """Summarizes repeated scores."""
    values = np.asarray(scores, dtype=float)
    return Spread(
        tuple(float(v) for v in values),
        float(values.mean()),
        float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        float(values.max() - values.min()),
    )


def reseeded(build: Callable[[], base.BaseEstimator], seed: int) -> Any:
    """A fresh estimator with its random seed set, when it has one."""
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
    """Spread over training seeds and over bootstrap resamples.

    One fixed split is used so that only training varies.

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
    """Spread of the score over random train/test splits."""
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
    """Adds Gaussian noise of `level` standard deviations to numeric columns."""
    result = features.copy()
    for name in result.select_dtypes("number").columns:
        scale = float(result[name].std() or 0.0)
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
    """Score of a fitted model as noise is added to the features.

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
    """Training, evaluation and performance sensitivity of one model.

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
