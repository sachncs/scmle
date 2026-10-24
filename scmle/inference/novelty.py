"""Out-of-distribution detection for incoming rows.

A model is only trustworthy on inputs that resemble its training data. The
detector learns what the training rows look like and flags a row in two
ways: an isolation forest on the encoded features scores how unusual the
combination is, and a range check names any numeric value beyond Tukey's
far-out fences (three interquartile ranges past the quartiles) of the
training data. The range check matters because tree-based detectors treat
anything beyond the training range like an edge point.
"""

import dataclasses
from typing import Any
from typing import Self

import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn import pipeline

from scmle.contracts import data as data_contracts
from scmle.data.curation import preprocessing

TRAIN_ROWS = 20000
TREES = 100
TAIL = 0.01
FENCE = 3.0


@dataclasses.dataclass
class NoveltyDetector:
    """Scores how unlike the training data a row is.

    Attributes:
        model: Encoder followed by an isolation forest.
        threshold: Scores below this (more anomalous) count as novel.
        fences: For each numeric feature, the (lower, upper) fence.
    """

    model: Any
    threshold: float
    fences: dict[str, tuple[float, float]] = dataclasses.field(
        default_factory=dict
    )

    @classmethod
    def fit(
        cls,
        features: pd.DataFrame,
        numeric: tuple[str, ...],
        categorical: tuple[str, ...],
        seed: int = 0,
    ) -> Self | None:
        """Learns the training distribution.

        Args:
            features: Training features.
            numeric: Numeric feature columns.
            categorical: Categorical feature columns.
            seed: Random seed.

        Returns:
            The detector, or None when there are no usable columns.
        """
        if not (numeric or categorical):
            return None
        plan = data_contracts.FeaturePlan(numeric, categorical)
        options = data_contracts.PreprocessingOptions(scale=True)
        encoder = preprocessing.build_preprocessor(plan, options)
        model = pipeline.Pipeline(
            [
                ("encode", encoder),
                (
                    "forest",
                    ensemble.IsolationForest(
                        n_estimators=TREES, random_state=seed, n_jobs=1
                    ),
                ),
            ]
        )
        sample = features[[*numeric, *categorical]]
        if len(sample) > TRAIN_ROWS:
            sample = sample.sample(TRAIN_ROWS, random_state=seed)
        model.fit(sample)
        scores = model.score_samples(sample)
        return cls(
            model,
            float(np.quantile(scores, TAIL)),
            cls.far_out_fences(features, numeric),
        )

    @staticmethod
    def far_out_fences(
        features: pd.DataFrame, numeric: tuple[str, ...]
    ) -> dict[str, tuple[float, float]]:
        """Tukey far-out fences of each numeric feature."""
        fences = {}
        for column in numeric:
            values = (
                pd.to_numeric(features[column], errors="coerce")
                .astype(float)
                .dropna()
            )
            if values.empty:
                continue
            low, high = values.quantile([0.25, 0.75])
            spread = high - low
            fences[column] = (
                float(low - FENCE * spread),
                float(high + FENCE * spread),
            )
        return fences

    def score(self, features: pd.DataFrame) -> pd.DataFrame:
        """Scores new rows.

        Args:
            features: Rows with the training feature columns.

        Returns:
            `novelty_score` (higher means more unusual), the numeric
            features beyond their far-out fences, and the boolean
            `out_of_distribution`.
        """
        raw = self.model.score_samples(features)
        outside = pd.DataFrame(index=features.index)
        for column, (low, high) in self.fences.items():
            values = pd.to_numeric(features[column], errors="coerce")
            outside[column] = (values < low) | (values > high)
        out_of_range = (
            outside.any(axis=1) if len(outside.columns) else raw < -np.inf
        )
        return pd.DataFrame(
            {
                "novelty_score": -raw,
                "out_of_range_features": [
                    [
                        c
                        for c, flag in zip(outside.columns, row, strict=True)
                        if flag
                    ]
                    for row in outside.to_numpy()
                ],
                "out_of_distribution": (raw < self.threshold) | out_of_range,
            },
            index=features.index,
        )
