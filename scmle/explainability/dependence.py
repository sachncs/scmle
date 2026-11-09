"""Partial dependence: how the average prediction moves with one feature."""

from typing import Any

import numpy
import pandas

from scmle.contracts import tasks
from scmle.evaluation import diagnostics

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
            grid = numpy.unique(
                numpy.quantile(
                    values,
                    numpy.linspace(LOW_QUANTILE, HIGH_QUANTILE, GRID_POINTS),
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
        sample: pandas.DataFrame, name: str, outcome: pandas.Series
    ) -> float:
        """Absolute Spearman correlation of a feature with the outcome.

        Args:
            sample: Rows holding the feature.
            name: Column of `sample` to correlate.
            outcome: Numeric outcome aligned by position with `sample`.

        Returns:
            The absolute rank correlation, or 0.0 when it is undefined.
        """
        values = pandas.Series(sample[name].to_numpy(), dtype=float)
        return abs(values.corr(outcome, method="spearman") or 0.0)

    @staticmethod
    def numeric_target(inputs: diagnostics.DiagnosticInput) -> pandas.Series:
        """Returns the target as numbers (1 for the positive class if binary).

        Args:
            inputs: The diagnostic inputs holding the task, target and
                positive label.

        Returns:
            The target with a fresh positional index: numeric values for
            regression, otherwise 1.0 where the target equals the positive
            label and 0.0 elsewhere.
        """
        if inputs.task is tasks.TaskType.REGRESSION:
            return pandas.to_numeric(inputs.target).reset_index(drop=True)
        return (
            (inputs.target == inputs.positive_label)
            .astype(float)
            .reset_index(drop=True)
        )

    @staticmethod
    def prediction(
        inputs: diagnostics.DiagnosticInput,
        sample: pandas.DataFrame,
        column: str,
        value: float,
    ) -> numpy.ndarray:
        """Predicts for the sample with one column held at `value`.

        Args:
            inputs: The diagnostic inputs holding the task and estimator.
            sample: Rows to predict for.
            column: Column of `sample` that is overwritten.
            value: Value written into every row of `column`.

        Returns:
            One prediction per row: the regression value, or the
            positive-class probability for classification.
        """
        swept = sample.assign(**{column: value})
        if inputs.task is tasks.TaskType.REGRESSION:
            return numpy.asarray(inputs.estimator.predict(swept))
        return numpy.asarray(inputs.estimator.predict_proba(swept))[:, 1]
