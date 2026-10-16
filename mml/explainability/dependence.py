"""Partial dependence: how the average prediction moves with one feature."""

from typing import Any

import numpy as np
import pandas as pd

from mml.contracts import tasks
from mml.evaluation import diagnostics

MAX_FEATURES = 3
GRID_POINTS = 8
MAX_ROWS = 200
LOW_QUANTILE = 0.05
HIGH_QUANTILE = 0.95


class PartialDependence(diagnostics.Diagnostic):
    """Average prediction over a grid of one feature.

    Covers the numeric features most associated with the target (binary
    classification and regression).
    """

    name = "partial_dependence"
    applies_to = diagnostics.BINARY_ONLY | diagnostics.REGRESSION_ONLY

    def compute(self, inputs: diagnostics.DiagnosticInput) -> dict[str, Any]:
        """Sweeps each selected feature over its central range.

        Args:
            inputs: The fitted candidate and evaluation data.

        Returns:
            For each selected feature, the grid and the mean prediction (the
            positive-class probability for binary targets).
        """
        columns = list(inputs.feature_plan.input_columns)
        numeric = [c for c in inputs.feature_plan.numeric if c in columns]
        sample = inputs.features[columns].head(MAX_ROWS)
        outcome = self.numeric_target(inputs).head(MAX_ROWS)
        ranked = sorted(
            numeric,
            key=lambda c: -self.association(sample, c, outcome),
        )
        report = {}
        for column in ranked[:MAX_FEATURES]:
            values = sample[column].dropna()
            if values.nunique() < 2:
                continue
            grid = np.unique(
                np.quantile(
                    values,
                    np.linspace(LOW_QUANTILE, HIGH_QUANTILE, GRID_POINTS),
                )
            )
            report[column] = {
                "grid": [float(g) for g in grid],
                "mean_prediction": [
                    float(self.prediction(inputs, sample, column, g).mean())
                    for g in grid
                ],
            }
        return {"features": report}

    @staticmethod
    def association(
        sample: pd.DataFrame, name: str, outcome: pd.Series
    ) -> float:
        """Absolute Spearman correlation of a feature with the outcome."""
        values = pd.Series(sample[name].to_numpy(), dtype=float)
        return abs(values.corr(outcome, method="spearman") or 0.0)

    @staticmethod
    def numeric_target(inputs: diagnostics.DiagnosticInput) -> pd.Series:
        """The target as numbers (1 for the positive class when binary)."""
        if inputs.task is tasks.TaskType.REGRESSION:
            return pd.to_numeric(inputs.target).reset_index(drop=True)
        return (
            (inputs.target == inputs.positive_label)
            .astype(float)
            .reset_index(drop=True)
        )

    @staticmethod
    def prediction(
        inputs: diagnostics.DiagnosticInput,
        sample: pd.DataFrame,
        column: str,
        value: float,
    ) -> np.ndarray:
        """Predictions for the sample with one column held at `value`."""
        swept = sample.assign(**{column: value})
        if inputs.task is tasks.TaskType.REGRESSION:
            return np.asarray(inputs.estimator.predict(swept))
        return np.asarray(inputs.estimator.predict_proba(swept))[:, 1]
