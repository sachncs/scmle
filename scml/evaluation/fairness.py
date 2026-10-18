"""Fairness report: do outcomes differ across groups of people?

For each sensitive column the evaluation rows are split into groups and the
model's decisions and errors are compared. Binary classification reports
selection rates and error rates (demographic parity, equal opportunity,
equalized odds, and the four-fifths ratio); regression reports bias and
error per group. The report describes differences; whether a difference is
unfair depends on the application, so it flags rather than decides.
"""

from typing import Any

import numpy as np
import pandas as pd

from scml.contracts import tasks
from scml.evaluation import diagnostics

MIN_GROUP_ROWS = 30
NUMERIC_BANDS = 4
MAX_LEVELS = 12
PARITY_TOLERANCE = 0.10
IMPACT_RATIO_FLOOR = 0.80


class FairnessReport(diagnostics.Diagnostic):
    """Group-wise outcome and error rates for the sensitive columns."""

    name = "fairness"
    applies_to = diagnostics.BINARY_ONLY | diagnostics.REGRESSION_ONLY

    def compute(self, inputs: diagnostics.DiagnosticInput) -> dict[str, Any]:
        """Compares decisions and errors across groups.

        Args:
            inputs: The fitted candidate and evaluation data, with
                `sensitive_columns` naming the columns to compare across.

        Returns:
            Per sensitive column: group statistics, disparity measures and
            flags; or an explanation when no column was named.
        """
        if not inputs.sensitive_columns:
            return {"available": False, "reason": "no sensitive columns"}
        predicted = np.asarray(inputs.estimator.predict(inputs.features))
        truth = np.asarray(inputs.target)
        report = {}
        for column in inputs.sensitive_columns:
            groups = self.groups(inputs.features[column])
            if inputs.task is tasks.TaskType.BINARY:
                report[column] = self.binary(
                    groups, truth, predicted, inputs.positive_label
                )
            else:
                report[column] = self.numeric(groups, truth, predicted)
        flagged = sorted(
            column for column, body in report.items() if body["flags"]
        )
        return {"available": True, "columns": report, "flagged": flagged}

    @staticmethod
    def groups(values: pd.Series) -> pd.Series:
        """Group labels: levels for categories, quartile bands for numbers."""
        if pd.api.types.is_numeric_dtype(values) and (
            values.nunique() > MAX_LEVELS
        ):
            bands = pd.qcut(values, NUMERIC_BANDS, duplicates="drop")
            return bands.astype(str).where(values.notna())
        return values.astype(object).where(values.notna()).astype(str)

    def binary(
        self,
        groups: pd.Series,
        truth: np.ndarray,
        predicted: np.ndarray,
        positive: Any,
    ) -> dict[str, Any]:
        """Selection, true-positive and false-positive rates per group."""
        actual = truth == positive
        chosen = predicted == positive
        stats = {}
        for level in groups.dropna().unique():
            rows = (groups == level).to_numpy()
            if rows.sum() < MIN_GROUP_ROWS:
                continue
            positives = actual[rows].sum()
            negatives = (~actual[rows]).sum()
            stats[str(level)] = {
                "rows": int(rows.sum()),
                "base_rate": float(actual[rows].mean()),
                "selection_rate": float(chosen[rows].mean()),
                "true_positive_rate": float(
                    (chosen[rows] & actual[rows]).sum() / positives
                )
                if positives
                else None,
                "false_positive_rate": float(
                    (chosen[rows] & ~actual[rows]).sum() / negatives
                )
                if negatives
                else None,
                "accuracy": float((chosen[rows] == actual[rows]).mean()),
            }
        return self.binary_summary(stats)

    def binary_summary(
        self, stats: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        """Disparity measures and flags over the group statistics."""
        if len(stats) < 2:
            return {
                "groups": stats,
                "flags": [],
                "note": "fewer than two groups",
            }
        selection = [g["selection_rate"] for g in stats.values()]
        true_positive = [
            g["true_positive_rate"]
            for g in stats.values()
            if g["true_positive_rate"] is not None
        ]
        false_positive = [
            g["false_positive_rate"]
            for g in stats.values()
            if g["false_positive_rate"] is not None
        ]
        parity = max(selection) - min(selection)
        opportunity = (
            max(true_positive) - min(true_positive) if true_positive else 0.0
        )
        odds = max(
            opportunity,
            max(false_positive) - min(false_positive)
            if false_positive
            else 0.0,
        )
        ratio = min(selection) / max(selection) if max(selection) > 0 else 1.0
        flags = []
        if parity > PARITY_TOLERANCE:
            flags.append(f"selection rates differ by {parity:.2f}")
        if ratio < IMPACT_RATIO_FLOOR:
            flags.append(f"disparate-impact ratio {ratio:.2f} < 0.80")
        if odds > PARITY_TOLERANCE:
            flags.append(f"error rates differ by {odds:.2f}")
        return {
            "groups": stats,
            "demographic_parity_difference": float(parity),
            "equal_opportunity_difference": float(opportunity),
            "equalized_odds_difference": float(odds),
            "disparate_impact_ratio": float(ratio),
            "flags": flags,
        }

    def numeric(
        self, groups: pd.Series, truth: np.ndarray, predicted: np.ndarray
    ) -> dict[str, Any]:
        """Mean prediction, bias and RMSE per group (regression)."""
        error = predicted.astype(float) - truth.astype(float)
        stats = {}
        for level in groups.dropna().unique():
            rows = (groups == level).to_numpy()
            if rows.sum() < MIN_GROUP_ROWS:
                continue
            stats[str(level)] = {
                "rows": int(rows.sum()),
                "mean_prediction": float(predicted[rows].mean()),
                "bias": float(error[rows].mean()),
                "rmse": float(np.sqrt(np.mean(error[rows] ** 2))),
            }
        if len(stats) < 2:
            return {
                "groups": stats,
                "flags": [],
                "note": "fewer than two groups",
            }
        spread = float(np.std(truth.astype(float))) or 1.0
        biases = [g["bias"] for g in stats.values()]
        gap = (max(biases) - min(biases)) / spread
        flags = (
            [f"bias differs by {gap:.2f} target standard deviations"]
            if gap > PARITY_TOLERANCE
            else []
        )
        return {"groups": stats, "bias_gap": float(gap), "flags": flags}
