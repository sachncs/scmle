"""Does the training sample look like the population it should represent?

A reference sample (a recent extract of the real population, or the rows
the model will see) is compared with the development rows. A classifier is
trained to tell the two apart: an out-of-fold ROC AUC near 0.5 means they are
indistinguishable, near 1.0 means the sample is not representative. Per-column
Kolmogorov-Smirnov tests and category-share gaps point at where.
"""

import dataclasses
from typing import Any
from typing import cast

import numpy
import pandas
from scipy import stats
from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection

from scmle.contracts import data
from scmle.contracts import errors

MIN_ROWS = 50
UNREPRESENTATIVE_AUC = 0.65
MAX_ROWS = 20000
FOLDS = 5


@dataclasses.dataclass(frozen=True)
class Gap:
    """How one column differs between the sample and the reference.

    Attributes:
        column: Column name.
        statistic: KS distance (numeric) or largest category-share gap.
        p_value: Probability of a gap this large by chance (numeric only).
    """

    column: str
    statistic: float
    p_value: float | None


@dataclasses.dataclass(frozen=True)
class Representativeness:
    """Comparison of a sample against a reference.

    Attributes:
        auc: Out-of-fold ROC AUC of telling sample from reference.
        representative: Whether `auc` is below the threshold.
        gaps: Per-column differences, largest first.
    """

    auc: float
    representative: bool
    gaps: tuple[Gap, ...]

    def to_dict(self) -> dict[str, Any]:
        """Builds the JSON-ready record.

        Returns:
            The report's fields as a dict, with `gaps` as a list of dicts.
        """
        return dataclasses.asdict(self)

    def quality_evidence(
        self,
        sample_rows: int,
        reference_rows: int,
        checked_sample_rows: int,
        checked_reference_rows: int,
        seed: int,
        shared_columns: tuple[str, ...],
    ) -> dict[str, object]:
        """Returns reproducible evidence for this comparison.

        Args:
            sample_rows: Rows available in the development sample.
            reference_rows: Rows available in the reference sample.
            checked_sample_rows: Development rows actually checked.
            checked_reference_rows: Reference rows actually checked.
            seed: Seed controlling folds and the classifier.
            shared_columns: Columns included in the comparison.

        Returns:
            JSON-native quality evidence.
        """
        return {
            "observation": "representativeness",
            "status": "flagged" if not self.representative else "clear",
            "method": "out_of_fold_reference_classifier_and_column_gaps",
            "thresholds": {
                "minimum_rows_per_side": MIN_ROWS,
                "maximum_rows_per_side": MAX_ROWS,
                "unrepresentative_auc": UNREPRESENTATIVE_AUC,
                "cross_validation_folds": FOLDS,
            },
            "sample_bounds": {
                "sample_rows": sample_rows,
                "reference_rows": reference_rows,
                "checked_sample_rows": checked_sample_rows,
                "checked_reference_rows": checked_reference_rows,
                "maximum_rows_per_side": MAX_ROWS,
                "truncated": checked_sample_rows < sample_rows
                or checked_reference_rows < reference_rows,
            },
            "seed": seed,
            "metrics": {
                "auc": self.auc,
                "representative": self.representative,
                "shared_columns": list(shared_columns),
                "column_gaps": [
                    {
                        "column": gap.column,
                        "statistic": gap.statistic,
                        "p_value": gap.p_value,
                    }
                    for gap in self.gaps
                ],
            },
            "limitations": [
                (
                    "Classifier distinguishability is a shift signal, not "
                    "proof of population bias."
                ),
                (
                    "Column gaps are univariate diagnostics and do not "
                    "establish causality."
                ),
                "Only the leading bounded rows from each side are checked.",
            ],
        }


def contrast(sample: pandas.Series, reference: pandas.Series) -> Gap:
    """Measures how one column differs between sample and reference.

    Args:
        sample: The column in the development rows.
        reference: The same column in the reference rows.

    Returns:
        For a numeric column, the two-sample Kolmogorov-Smirnov distance
        and p-value (missing values dropped). Otherwise the largest
        absolute difference in category share, with no p-value.
    """
    if pandas.api.types.is_numeric_dtype(sample):
        result = stats.ks_2samp(sample.dropna(), reference.dropna())
        return Gap(
            str(sample.name),
            float(result.statistic),  # pyright: ignore[reportAttributeAccessIssue]  # scipy.stats result types are untyped tuples
            float(result.pvalue),  # pyright: ignore[reportAttributeAccessIssue]  # scipy.stats result types are untyped tuples
        )
    left = sample.astype(str).value_counts(normalize=True)
    right = reference.astype(str).value_counts(normalize=True)
    gap = left.sub(right, fill_value=0.0).abs().max()
    return Gap(str(sample.name), float(gap), None)


def distinguish(
    sample: pandas.DataFrame, reference: pandas.DataFrame, seed: int = 0
) -> Representativeness:
    """Measures how distinguishable the sample is from the reference.

    Args:
        sample: Development rows.
        reference: Rows from the population of interest, same columns.
        seed: Seed for the classifier and folds.

    Returns:
        The report.

    Raises:
        ValueError: On differing columns or too few rows.
    """
    if set(sample.columns) != set(reference.columns):
        raise errors.InputError(
            "sample and reference must have the same columns"
        )
    if min(len(sample), len(reference)) < MIN_ROWS:
        raise errors.InputError(f"each side needs at least {MIN_ROWS} rows")
    columns = list(sample.columns)
    both = pandas.concat(
        [sample[columns], reference[columns]], ignore_index=True
    )
    for name in columns:
        if not pandas.api.types.is_numeric_dtype(both[name]):
            both[name] = both[name].astype("category").cat.codes
    side = numpy.r_[numpy.zeros(len(sample)), numpy.ones(len(reference))]
    scores = cast(
        "numpy.ndarray",
        model_selection.cross_val_predict(
            ensemble.HistGradientBoostingClassifier(random_state=seed),
            both,
            side,
            cv=model_selection.StratifiedKFold(
                FOLDS, shuffle=True, random_state=seed
            ),
            method="predict_proba",
        ),
    )[:, 1]
    auc = float(metrics.roc_auc_score(side, scores))
    gaps = sorted(
        (contrast(sample[name], reference[name]) for name in columns),
        key=lambda gap: -gap.statistic,
    )
    return Representativeness(
        auc=auc, representative=auc < UNREPRESENTATIVE_AUC, gaps=tuple(gaps)
    )


def withhold(
    note: str | None,
    seed: int,
    rows: tuple[int, int],
    checked: tuple[int, int],
    shared: tuple[str, ...],
    error: str | None,
    limitation: str,
) -> data.Observation:
    """Records a comparison that could not be evaluated.

    Args:
        note: Run note to show, or None.
        seed: Seed controlling folds and the classifier.
        rows: Sample and reference row counts.
        checked: Sample and reference rows that were to be compared; (0, 0)
            when nothing was compared.
        shared: Feature columns the two sides share.
        error: Why the comparison failed, or None.
        limitation: What the missing comparison means for the evidence.

    Returns:
        A `not_evaluated` observation with the thresholds and the bounds.
    """
    metrics: dict[str, list[str] | str] = {"shared_columns": list(shared)}
    if error is not None:
        metrics["error"] = error
    return data.Observation(
        note=note,
        evidence={
            "observation": "representativeness",
            "status": "not_evaluated",
            "method": "out_of_fold_reference_classifier_and_column_gaps",
            "thresholds": {
                "minimum_rows_per_side": MIN_ROWS,
                "maximum_rows_per_side": MAX_ROWS,
                "unrepresentative_auc": UNREPRESENTATIVE_AUC,
                "cross_validation_folds": FOLDS,
            },
            "sample_bounds": {
                "sample_rows": rows[0],
                "reference_rows": rows[1],
                "checked_sample_rows": checked[0],
                "checked_reference_rows": checked[1],
                "maximum_rows_per_side": MAX_ROWS,
                "truncated": any(checked)
                and (checked[0] < rows[0] or checked[1] < rows[1]),
            },
            "seed": seed,
            "metrics": metrics,
            "limitations": [limitation],
        },
    )


def assess_representativeness(
    sample: pandas.DataFrame, reference: pandas.DataFrame, seed: int = 0
) -> data.Observation:
    """Compares bounded samples and retains the complete quality evidence.

    Args:
        sample: Development rows.
        reference: Rows from the population of interest.
        seed: Seed controlling folds and the classifier.

    Returns:
        A quality observation even when the check cannot be evaluated.
    """
    sample_rows = len(sample)
    reference_rows = len(reference)
    shared = tuple(name for name in sample.columns if name in reference.columns)
    if not shared:
        return withhold(
            "representativeness: the reference shares no feature columns",
            seed,
            (sample_rows, reference_rows),
            (0, 0),
            shared,
            None,
            "No common feature columns were available for comparison.",
        )
    checked_sample = sample[list(shared)].head(MAX_ROWS)
    checked_reference = reference[list(shared)].head(MAX_ROWS)
    try:
        report = distinguish(checked_sample, checked_reference, seed)
    except ValueError as error:
        # handled: returned as failed finding/problem/refusal with error text
        return withhold(
            None,
            seed,
            (sample_rows, reference_rows),
            (len(checked_sample), len(checked_reference)),
            shared,
            str(error),
            (
                "The comparison was not evaluated because its data "
                "contract was not met."
            ),
        )
    note = None
    if not report.representative:
        worst = report.gaps[0]
        note = (
            f"representativeness: development rows are distinguishable from "
            f"the reference sample (AUC {report.auc:.2f}); the largest gap is "
            f"in '{worst.column}'"
        )
    return data.Observation(
        note=note,
        evidence=report.quality_evidence(
            sample_rows,
            reference_rows,
            len(checked_sample),
            len(checked_reference),
            seed,
            shared,
        ),
    )
