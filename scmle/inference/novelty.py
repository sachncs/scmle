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

import numpy
import pandas
from sklearn import ensemble
from sklearn import pipeline

from scmle.contracts import data
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
        threshold: Raw isolation-forest score (lower is more anomalous)
            below which a row counts as novel; the 1% quantile of the
            training scores.
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
        features: pandas.DataFrame,
        numeric: tuple[str, ...],
        categorical: tuple[str, ...],
        seed: int = 0,
    ) -> Self | None:
        """Learns the training distribution.

        The forest is fitted on at most 20000 rows, a random sample when
        there are more, and the fences use all of the rows.

        Args:
            features: Training features.
            numeric: Numeric feature columns.
            categorical: Categorical feature columns.
            seed: Random seed.

        Returns:
            The detector, or None when both column tuples are empty.
        """
        if not (numeric or categorical):
            return None
        plan = data.FeaturePlan(numeric, categorical)
        options = data.PreprocessingOptions(scale=True)
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
            float(numpy.quantile(scores, TAIL)),
            cls.far_out_fences(features, numeric),
        )

    @staticmethod
    def far_out_fences(
        features: pandas.DataFrame, numeric: tuple[str, ...]
    ) -> dict[str, tuple[float, float]]:
        """Computes the Tukey far-out fences of each numeric feature.

        Args:
            features: Training features.
            numeric: Numeric feature columns.

        Returns:
            Per column, the (lower, upper) fence: the quartiles extended by
            three interquartile ranges. Columns without numeric values are
            omitted.
        """
        fences = {}
        for column in numeric:
            values = (
                pandas.to_numeric(features[column], errors="coerce")
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

    def score(self, features: pandas.DataFrame) -> pandas.DataFrame:
        """Scores new rows.

        Args:
            features: Rows with the training feature columns.

        Returns:
            A frame indexed like `features` with `novelty_score` (higher
            means more unusual), `out_of_range_features` (the numeric
            features beyond their far-out fences), and the boolean
            `out_of_distribution` (raw score below the threshold or any
            feature out of range; a missing or non-numeric value is never
            out of range). `novelty_score` is the negated raw score.
        """
        raw = self.model.score_samples(features)
        outside = pandas.DataFrame(index=features.index)
        for column, (low, high) in self.fences.items():
            values = pandas.to_numeric(features[column], errors="coerce")
            outside[column] = (values < low) | (values > high)
        out_of_range = (
            outside.any(axis=1) if len(outside.columns) else raw < -numpy.inf
        )
        return pandas.DataFrame(
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
