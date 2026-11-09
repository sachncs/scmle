"""Fairness report: do outcomes differ across groups of people?

For each sensitive column the evaluation rows are split into groups and the
model's decisions and errors are compared. Binary classification reports
selection rates and error rates (demographic parity, equal opportunity,
equalized odds, and the four-fifths ratio); regression reports bias and
error per group. The report describes differences; whether a difference is
unfair depends on the application, so it flags rather than decides.
"""

from typing import Any

import numpy
import pandas

from scmle.contracts import tasks
from scmle.evaluation import diagnostics

MIN_GROUP_ROWS = 30
NUMERIC_BANDS = 4
MAX_LEVELS = 12
PARITY_TOLERANCE = 0.10
IMPACT_RATIO_FLOOR = 0.80


class Fairness(diagnostics.Diagnostic):
    """Group-wise outcome and error rates for the sensitive columns."""

    name = "fairness"
    applies_to = diagnostics.BINARY_ONLY | diagnostics.REGRESSION_ONLY

    def compute(self, inputs: diagnostics.DiagnosticInput) -> dict[str, Any]:
        """Compares decisions and errors across groups.

        Args:
            inputs: The fitted candidate and evaluation data, with
                `sensitive_columns` naming the columns to compare across.

        Returns:
            `{"available": False, "reason": ...}` when no sensitive column
            is named. Otherwise `available` True, `columns` (the result of
            `binary` or `numeric` for each sensitive column, plus
            `intersectional` when at least two columns are named and the
            joined groups exist) and `flagged`, the sorted names in
            `columns` whose `flags` list is not empty.
        """
        if not inputs.sensitive_columns:
            return {"available": False, "reason": "no sensitive columns"}
        predicted = numpy.asarray(inputs.estimator.predict(inputs.features))
        truth = numpy.asarray(inputs.target)
        report = {}
        for column in inputs.sensitive_columns:
            groups = self.groups(inputs.features[column])
            if inputs.task is tasks.Task.BINARY:
                report[column] = self.binary(
                    groups, truth, predicted, inputs.positive_label
                )
            else:
                report[column] = self.numeric(groups, truth, predicted)
        intersections = self.intersectional(
            inputs.features, inputs.sensitive_columns
        )
        if intersections is not None:
            if inputs.task is tasks.Task.BINARY:
                report["intersectional"] = self.binary(
                    intersections, truth, predicted, inputs.positive_label
                )
            else:
                report["intersectional"] = self.numeric(
                    intersections, truth, predicted
                )
        flagged = sorted(
            column for column, body in report.items() if body["flags"]
        )
        return {"available": True, "columns": report, "flagged": flagged}

    @staticmethod
    def groups(values: pandas.Series) -> pandas.Series:
        """Group labels: levels for categories, quartile bands for numbers.

        Args:
            values: One sensitive column.

        Returns:
            The group of each row as text, aligned with `values`: quartile
            bands for a numeric column with more than `MAX_LEVELS` distinct
            values, otherwise the value itself. Missing values stay missing.
        """
        if pandas.api.types.is_numeric_dtype(values) and (
            values.nunique() > MAX_LEVELS
        ):
            bands = pandas.qcut(values, NUMERIC_BANDS, duplicates="drop")
            return bands.astype(str).where(values.notna())
        return values.astype(object).where(values.notna()).astype(str)

    def intersectional(
        self, features: pandas.DataFrame, columns: tuple[str, ...]
    ) -> pandas.Series | None:
        """Builds bounded intersectional groups from every sensitive column.

        Numeric columns use the same deterministic bands as their standalone
        report. Missing values remain missing so they cannot be mistaken for
        a real group. A single sensitive column has no intersection to add.

        Args:
            features: Rows holding the sensitive columns.
            columns: The sensitive columns.

        Returns:
            For each row, the group labels of every column joined with
            ` | `, missing where any column is missing; None with fewer than
            two columns.
        """
        if len(columns) < 2:
            return None
        grouped = [self.groups(features[column]) for column in columns]
        frame = pandas.concat(grouped, axis=1)
        frame.columns = list(columns)
        valid = frame.notna().all(axis=1)
        labels = frame.astype(str).agg(" | ".join, axis=1)
        return labels.where(valid)

    def binary(
        self,
        groups: pandas.Series,
        truth: numpy.ndarray,
        predicted: numpy.ndarray,
        positive: Any,
    ) -> dict[str, Any]:
        """Summarizes outcome and error rates, and their disparities, by group.

        Each group with at least `MIN_GROUP_ROWS` rows gets its row count,
        base rate, selection rate, true-positive rate, false-positive rate
        and accuracy.

        Args:
            groups: Group label of each row.
            truth: Observed labels.
            predicted: Predicted labels.
            positive: The positive class.

        Returns:
            The `binary_summary` of those per-group statistics, with the
            groups below `MIN_GROUP_ROWS` rows reported as excluded. A
            true-positive or false-positive rate with no actual positives or
            negatives in the group is None.
        """
        actual = truth == positive
        chosen = predicted == positive
        stats = {}
        excluded = self.small_groups(groups)
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
        return self.binary_summary(stats, excluded)

    def small_groups(self, groups: pandas.Series) -> dict[str, int]:
        """Returns groups excluded because their cell size is too small.

        Args:
            groups: Group label of each row.

        Returns:
            The row count of each group with fewer than `MIN_GROUP_ROWS`
            rows, by group name.
        """
        return {
            str(level): int((groups == level).sum())
            for level in groups.dropna().unique()
            if int((groups == level).sum()) < MIN_GROUP_ROWS
        }

    def binary_summary(
        self,
        stats: dict[str, dict[str, Any]],
        excluded: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        """Disparity measures and flags over the group statistics.

        Args:
            stats: Per-group statistics from `binary`.
            excluded: Row counts of the groups left out as too small.

        Returns:
            The groups, the demographic parity, equal opportunity and
            equalized odds differences, the disparate-impact ratio, the
            flags and the exclusion metadata (`excluded_groups`,
            `power_warning`). Equal opportunity and equalized odds count a
            side with no defined rate as 0.0. With fewer than two groups,
            only the groups, empty flags, a note and the metadata.
        """
        excluded = excluded or {}
        metadata = {
            "excluded_groups": excluded,
            "power_warning": bool(excluded),
        }
        if len(stats) < 2:
            return {
                "groups": stats,
                "flags": [],
                "note": "fewer than two groups",
                **metadata,
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
            **metadata,
        }

    def numeric(
        self,
        groups: pandas.Series,
        truth: numpy.ndarray,
        predicted: numpy.ndarray,
    ) -> dict[str, Any]:
        """Mean prediction, bias and RMSE per group (regression).

        Args:
            groups: Group label of each row.
            truth: Observed values.
            predicted: Predicted values.

        Returns:
            The statistics of each group with at least `MIN_GROUP_ROWS`
            rows, the gap between the largest and smallest group bias in
            target standard deviations, flags, and the exclusion metadata.
            With fewer than two groups, only the groups, empty flags, a note
            and the metadata.
        """
        error = predicted.astype(float) - truth.astype(float)
        stats = {}
        excluded = self.small_groups(groups)
        for level in groups.dropna().unique():
            rows = (groups == level).to_numpy()
            if rows.sum() < MIN_GROUP_ROWS:
                continue
            stats[str(level)] = {
                "rows": int(rows.sum()),
                "mean_prediction": float(predicted[rows].mean()),
                "bias": float(error[rows].mean()),
                "rmse": float(numpy.sqrt(numpy.mean(error[rows] ** 2))),
            }
        if len(stats) < 2:
            return {
                "groups": stats,
                "flags": [],
                "note": "fewer than two groups",
                "excluded_groups": excluded,
                "power_warning": bool(excluded),
            }
        spread = float(numpy.std(truth.astype(float))) or 1.0
        biases = [g["bias"] for g in stats.values()]
        gap = (max(biases) - min(biases)) / spread
        flags = (
            [f"bias differs by {gap:.2f} target standard deviations"]
            if gap > PARITY_TOLERANCE
            else []
        )
        return {
            "groups": stats,
            "bias_gap": float(gap),
            "flags": flags,
            "excluded_groups": excluded,
            "power_warning": bool(excluded),
        }
