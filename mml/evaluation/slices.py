"""Subgroup evaluation: the score inside slices of the data, not just overall.

An average can hide a segment where the model fails. Slices come from the
levels of categorical features and, for metrics that are not rank-based,
from quantile bands of numeric features;
slices with too few rows to score reliably are skipped and reported.
"""

from typing import Any

import numpy as np
import pandas as pd

from mml.evaluation import diagnostics

MAX_FEATURES = 4
MAX_LEVELS = 12
BANDS = 4
MIN_SUPPORT = 30
WORST = 5
RANKING_METRICS = frozenset({"roc_auc", "roc_auc_ovr", "pr_auc"})


class SliceReport(diagnostics.Diagnostic):
    """Metric per slice, with the worst slices surfaced."""

    name = "slice_performance"

    def compute(self, inputs: diagnostics.DiagnosticInput) -> dict[str, Any]:
        """Scores the model inside each slice.

        Args:
            inputs: The fitted candidate and evaluation data.

        Returns:
            The overall score, every scored slice with its support and gap
            to the overall score (positive means worse than overall), the
            worst slices, and how many were skipped for lack of support.
        """
        metric = inputs.metric
        overall = metric.score(
            inputs.estimator,
            inputs.features,
            inputs.target,
            inputs.positive_label,
        )
        scored, skipped = [], 0
        for feature, labels in self.assignments(inputs):
            for level in labels.dropna().unique():
                rows = (labels == level).to_numpy()
                score = self.slice_score(inputs, rows)
                if score is None:
                    skipped += 1
                    continue
                gap = (
                    overall - score
                    if metric.greater_is_better
                    else score - overall
                )
                scored.append(
                    {
                        "feature": feature,
                        "slice": str(level),
                        "rows": int(rows.sum()),
                        "score": float(score),
                        "gap": float(gap),
                    }
                )
        scored.sort(key=lambda item: -item["gap"])
        return {
            "metric": metric.name,
            "overall": float(overall),
            "slices": scored,
            "worst": scored[:WORST],
            "skipped_for_support": skipped,
        }

    def assignments(
        self, inputs: diagnostics.DiagnosticInput
    ) -> list[tuple[str, pd.Series]]:
        """Slice labels for each feature used to slice the data."""
        plan = inputs.feature_plan
        found: list[tuple[str, pd.Series]] = []
        for column in plan.categorical:
            values = inputs.features[column].astype(object)
            if 1 < values.nunique() <= MAX_LEVELS:
                found.append((column, values))
        # A rank-based score inside a narrow numeric band is low because the
        # band restricts the range, not because the model is worse there.
        numeric = [] if inputs.metric.name in RANKING_METRICS else plan.numeric
        for column in numeric:
            values = pd.to_numeric(inputs.features[column])
            if values.nunique() > BANDS:
                bands = pd.qcut(values, BANDS, duplicates="drop").astype(str)
                found.append((column, bands.where(values.notna())))
        return found[:MAX_FEATURES]

    def slice_score(
        self, inputs: diagnostics.DiagnosticInput, rows: np.ndarray
    ) -> float | None:
        """Scores one slice, or None when it cannot be scored reliably."""
        if rows.sum() < MIN_SUPPORT:
            return None
        target = inputs.target[rows]
        if inputs.task.is_classification and target.nunique() < 2:
            return None
        try:
            return float(
                inputs.metric.score(
                    inputs.estimator,
                    inputs.features[rows],
                    target,
                    inputs.positive_label,
                )
            )
        except ValueError:
            return None
