"""The coverage report of a run, measured on rows no interval was fit on.

A run that stores intervals or sets also says how often they held. After the
final test rows are released for the evaluation, every interval method the
bundle has (`residual`, `cv_plus` and `quantile` for regression, pooled and
class-conditional sets for classification) is applied to those rows at each
requested level, and `coverage.coverage_report` turns the hits into the exact
interval, the tier and, for sets, the coverage within each true class. The
rows are read once, by the evaluation that already opened them; this module
asks for nothing more of custody.

The claim is fixed first. `claim_document` states, before the rows are
released, the levels, the methods, the tolerance and the rule; its hash
(`claim_hash`, the digest the ledgers use) is registered by the run (the
custody log of a charter run, the event log always) and the report carries it,
so a report for a different claim is detectable (`claims verify`).

The tier is `confirmed` only when the Clopper-Pearson lower limit reaches the
nominal level less the tolerance. A shortfall is reported as `unverified`,
and nothing here tunes a level or an interval after seeing the rows.
"""

from collections.abc import Mapping
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from scmle.contracts import errors
from scmle.data.ledger import ledger as ledger_module
from scmle.evaluation import coverage
from scmle.evaluation import coverage_slices

REGRESSION_METHODS = ("residual", "cv_plus", "quantile")
SET_MODES = (None, "class")
SET_METHODS = ("pooled", "class_conditional")
INTERVAL_TASKS = ("regression",)
SET_TASKS = ("binary_classification", "multiclass_classification")
CLAIM_FILE = "coverage_claim.json"
REPORT_FILE = "coverage_report.json"
REGISTRATION_EVENT = "coverage_claim_registered"


def claim_document(
    *,
    task: str,
    levels: Sequence[float],
    tolerance: float,
    split_fingerprint: str,
    charter_hash: str | None,
    slice_columns: Sequence[str] = (),
) -> dict[str, Any] | None:
    """The coverage claim, fixed before the final rows are read.

    `slice_columns` are the charter's declared slice columns; they are part
    of the hashed document, so the slices read are the slices declared.

    Args:
        task: The task type value.
        levels: Coverage levels to be measured.
        tolerance: Slack below each level that still counts as met.
        split_fingerprint: Fingerprint of the data split.
        charter_hash: Hash of the charter, or None without one.
        slice_columns: Declared slice columns.

    Returns:
        The document, or None for a task that stores no interval or set.
    """
    if task in INTERVAL_TASKS:
        methods = list(REGRESSION_METHODS)
    elif task in SET_TASKS:
        methods = list(SET_METHODS)
    else:
        return None
    return {
        "schema_version": 1,
        "task": task,
        "levels": [float(v) for v in levels],
        "methods": methods,
        "tolerance": float(tolerance),
        "rule": "for each method the bundle holds and each level, the exact "
        "Clopper-Pearson lower limit of the coverage on the final test rows "
        "reaches the level less the tolerance; a shortfall is reported "
        "unverified and nothing is retuned",
        "rows_read": "once, by the final evaluation",
        "split_fingerprint": split_fingerprint,
        "charter_hash": charter_hash,
        **({"slice_columns": list(slice_columns)} if slice_columns else {}),
    }


def claim_hash(claim: dict[str, Any]) -> str:
    """The hash that registers the claim.

    Args:
        claim: The claim document.

    Returns:
        Its ledger digest.
    """
    return ledger_module.digest(claim)


def declared_slices(
    inside: np.ndarray,
    sizes: np.ndarray,
    labels: Mapping[str, Any],
    level: float,
) -> dict[str, Any]:
    """The slice descriptions of one method and level, per declared column.

    Args:
        inside: Whether the interval or set held, per row.
        sizes: Interval width or set size, per row.
        labels: Slice label of each row by column; None for a column that
            cannot be sliced.
        level: The coverage level.

    Returns:
        By column, a `coverage_slices.summary`, or a `not_assessable`
        record for a column without labels.
    """
    found: dict[str, Any] = {}
    for column, assigned in labels.items():
        if assigned is None:
            found[column] = {
                "status": "not_assessable",
                "reason": "the column is missing or has too many levels",
            }
        else:
            found[column] = coverage_slices.summary(
                inside, sizes, assigned, level
            )
    return found


def regression_reports(
    bundle: Any,
    features: pd.DataFrame,
    target: Any,
    levels: Sequence[float],
    slices: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Coverage reports of each interval method the bundle holds.

    Args:
        bundle: The frozen bundle that serves intervals.
        features: Features of the released rows.
        target: True values of those rows.
        levels: Coverage levels to measure.
        slices: Slice labels by declared column, or None.

    Returns:
        By method and level, a `coverage.coverage_report`. For each method
        the levels are tried in order and the first level the bundle cannot
        serve (`ArtifactError`) ends that method, so it and the later
        levels are omitted; a method with no served level is omitted.
    """
    values = np.asarray(target, dtype=float)
    reports: dict[str, Any] = {}
    for method in REGRESSION_METHODS:
        by_level: dict[str, Any] = {}
        for level in levels:
            try:
                band = bundle.predict_interval(
                    features, coverage=level, method=method
                )
            except errors.ArtifactError:
                break
            inside = (values >= band["lower"].to_numpy()) & (
                values <= band["upper"].to_numpy()
            )
            widths = (band["upper"] - band["lower"]).to_numpy()
            by_level[f"{level:g}"] = coverage.coverage_report(
                inside, widths, level
            )
            if slices:
                by_level[f"{level:g}"]["declared_slices"] = declared_slices(
                    inside, widths, slices, level
                )
        if by_level:
            reports[method] = by_level
    return reports


def classification_reports(
    bundle: Any,
    features: pd.DataFrame,
    target: Any,
    levels: Sequence[float],
    slices: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Coverage reports of pooled and class-conditional prediction sets.

    Args:
        bundle: The frozen bundle that serves prediction sets.
        features: Features of the released rows.
        target: True labels of those rows.
        levels: Coverage levels to measure.
        slices: Slice labels by declared column, or None.

    Returns:
        By `pooled` or `class_conditional` and level, a
        `coverage.coverage_report` with the coverage within each class.
    """
    truth = list(target)
    reports: dict[str, Any] = {}
    for mode in SET_MODES:
        by_level: dict[str, Any] = {}
        for level in levels:
            sets = bundle.predict_set(features, level, conditional=mode)
            inside = np.array(
                [
                    label in found
                    for label, found in zip(truth, sets, strict=True)
                ]
            )
            sizes = np.array([len(found) for found in sets], dtype=float)
            by_level[f"{level:g}"] = coverage.coverage_report(
                inside, sizes, level, true_classes=[str(t) for t in truth]
            )
            if slices:
                by_level[f"{level:g}"]["declared_slices"] = declared_slices(
                    inside, sizes, slices, level
                )
        reports["pooled" if mode is None else "class_conditional"] = by_level
    return reports


def run_report(
    bundle: Any,
    features: pd.DataFrame,
    target: Any,
    levels: Sequence[float],
    claim_sha256: str | None = None,
    slices: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """The `coverage_report.json` body, or None without intervals or sets.

    `claim_sha256` is the hash of the claim registered before the rows were
    read; the report repeats it. `slices` maps each declared slice column to
    the slice label of every row (or None when it cannot be sliced); each
    level then also gets `declared_slices`, descriptive only.

    Args:
        bundle: The frozen bundle, calibration included.
        features: Features of the released rows.
        target: True values or labels of those rows.
        levels: Coverage levels to measure.
        claim_sha256: Hash of the registered claim, repeated in the report.
        slices: Slice labels by declared column, or None.

    Returns:
        The kind, claim hash, row count, a note and the coverage reports by
        method; None when the bundle has neither conformal calibration nor
        quantile models.
    """
    if bundle.conformal is None and bundle.quantile is None:
        return None
    kind = (
        bundle.conformal.kind
        if bundle.conformal is not None
        else ("regression")
    )
    body = (
        regression_reports(bundle, features, target, levels, slices)
        if kind == "regression"
        else classification_reports(bundle, features, target, levels, slices)
    )
    return {
        "kind": kind,
        "claim_sha256": claim_sha256,
        "rows": len(features),
        "rows_read": "once, by the final evaluation",
        "note": (
            "coverage measured on the final test rows; the tier is "
            "'confirmed' only when the exact lower limit reaches the nominal "
            "level less the tolerance"
        ),
        "methods": body,
    }
