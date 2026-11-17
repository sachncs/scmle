"""Subgroup evaluation: the score inside slices of the data, not just overall.

An average can hide a segment where the model fails. Slices come from the
levels of categorical features and, for metrics that are not rank-based,
from quantile bands of numeric features;
slices with too few rows to score reliably are skipped and reported.
"""

from typing import Any

import numpy
import pandas

from scmle.evaluation import diagnostics

MAX_FEATURES = 4
MAX_LEVELS = 12
BANDS = 4
MIN_SUPPORT = 30
MIN_TARGET_LEVELS = 2
WORST = 5
RANKING_METRICS = frozenset({"roc_auc", "roc_auc_ovr", "pr_auc"})


class Slices(diagnostics.Diagnostic):
    """Metric per slice, with the worst slices surfaced.

    Cost model (31 runs, see `diagnostics.MARGIN`): 0.038 s plus 5.9
    predictions plus 21 microseconds per row for every slice; the smallest
    useful report is the overall score, one prediction.
    """

    name = "slice_performance"
    fixed = 0.038
    passes = 5.9
    rowcost = 2.1e-5

    def least(self, inputs: diagnostics.DiagnosticInput) -> float:
        """Predicts the seconds of the overall score alone.

        Args:
            inputs: The evaluation data, with `pace` when it was timed.

        Returns:
            One prediction over the evaluation rows, with the margin.
        """
        return diagnostics.MARGIN * (inputs.pace or 0.0)

    def compute(self, inputs: diagnostics.DiagnosticInput) -> dict[str, Any]:
        """Scores the model inside each slice.

        Args:
            inputs: The fitted candidate and evaluation data.

        Returns:
            A mapping with `metric` (name), the `overall` score, `slices`,
            `worst`, `skipped_for_support` and `skipped_for_time`. `slices`
            lists every scored slice (`feature`, `slice`, `rows`, `score`,
            `gap`), largest gap first, where the gap to the overall score is
            positive when the slice is worse. `worst` is the first `WORST`
            of them, `skipped_for_support` counts slices that could not be
            scored and `skipped_for_time` those not scored because the
            longest score so far would end past the deadline (then
            `truncated` and a `reason` are added). When even the overall
            score does not fit, the report is `skipped`.
        """
        metric = inputs.metric
        if not inputs.affords(self.least(inputs)):
            return diagnostics.skipped(
                "the deadline leaves no time for the overall score"
            )
        began = diagnostics.clock()
        overall = metric.score(
            inputs.estimator,
            inputs.features,
            inputs.target,
            inputs.positive_label,
        )
        longest = diagnostics.clock() - began
        scored: list[dict[str, Any]] = []
        skipped = 0
        late = 0
        for feature, labels in self.assignments(inputs):
            for level in labels.dropna().unique():
                if not inputs.affords(diagnostics.MARGIN * longest):
                    late += 1
                    continue
                rows = (labels == level).to_numpy()
                began = diagnostics.clock()
                score = self.grade(inputs, rows)
                longest = max(longest, diagnostics.clock() - began)
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
        report = {
            "metric": metric.name,
            "overall": float(overall),
            "slices": scored,
            "worst": scored[:WORST],
            "skipped_for_support": skipped,
            "skipped_for_time": late,
        }
        if late:
            report["truncated"] = True
            report["reason"] = (
                f"{late} slices not scored: each takes up to "
                f"{longest:.3f} s and the deadline came first"
            )
        return report

    def assignments(
        self, inputs: diagnostics.DiagnosticInput
    ) -> list[tuple[str, pandas.Series]]:
        """Returns slice labels for each feature used to slice the data.

        Args:
            inputs: The fitted candidate, evaluation data and feature plan.

        Returns:
            Up to `MAX_FEATURES` pairs of feature name and a slice label per
            row, categorical features first: the level for a categorical
            feature with between two and `MAX_LEVELS` levels, or a quantile
            band (missing where the value is missing) for a numeric feature
            with more than `BANDS` distinct values. Numeric features are not
            used for rank-based metrics.
        """
        plan = inputs.feature_plan
        found: list[tuple[str, pandas.Series]] = []
        for column in plan.categorical:
            values = inputs.features[column].astype(object)
            if 1 < values.nunique() <= MAX_LEVELS:
                found.append((column, values))
        # A rank-based score inside a narrow numeric band is low because the
        # band restricts the range, not because the model is worse there.
        numeric = [] if inputs.metric.name in RANKING_METRICS else plan.numeric
        for column in numeric:
            values = pandas.to_numeric(inputs.features[column])
            if values.nunique() > BANDS:
                bands = pandas.qcut(values, BANDS, duplicates="drop").astype(
                    str
                )
                found.append((column, bands.where(values.notna())))
        return found[:MAX_FEATURES]

    def grade(
        self, inputs: diagnostics.DiagnosticInput, rows: numpy.ndarray
    ) -> float | None:
        """Scores one slice, or None when it cannot be scored reliably.

        Args:
            inputs: The fitted candidate and evaluation data.
            rows: Boolean mask selecting the slice's rows.

        Returns:
            The metric on the slice, or None when it has fewer than
            `MIN_SUPPORT` rows, a classification slice holds one class, or
            the metric raises `ValueError`.
        """
        if rows.sum() < MIN_SUPPORT:
            return None
        target = inputs.target[rows]
        if (
            inputs.task.is_classification
            and target.nunique() < MIN_TARGET_LEVELS
        ):
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
            # handled: documented: None when metric raises ValueError
            return None
