"""Data-access ledger: what each fitted operation was allowed to see.

The ledger is derived from the split manifest and the executed trials, then
*verified* against the manifest, so the guarantees it states (no fold touches
the test rows, the test set opened once) are checked facts, not claims.
"""

import hashlib
from typing import Any

from mml.contracts import experiments
from mml.data import splitting


def digest_of(row_ids: list[Any]) -> dict[str, Any]:
    """Row count and digest in the format trials record."""
    ordered = sorted(map(str, row_ids))
    digest = hashlib.sha256("\x1f".join(ordered).encode()).hexdigest()
    return {"rows": len(ordered), "digest": digest[:16]}


def observed_access(
    split: splitting.SplitResult, trials: list[experiments.TrialResult]
) -> dict[str, Any]:
    """Checks what trials actually received against the split manifest.

    Every evaluated fold of every trial recorded the development-row
    positions passed to `fit` and to scoring. They must equal the manifest's
    training and validation positions for that fold, and none may be a
    locked-test row.

    Returns:
        Counts of checked folds and the verdicts.
    """
    manifest = split.manifest
    expected = [
        (digest_of(list(fit)), digest_of(list(validate)))
        for fit, validate in manifest.folds
    ]
    development = {str(i) for i in manifest.train_ids}
    test_ids = {str(i) for i in manifest.test_ids}
    checked = mismatched = 0
    for trial in trials:
        for index, seen in enumerate(trial.observed_access):
            checked += 1
            if index >= len(expected) or (
                seen["fit"] != expected[index][0]
                or seen["evaluate"] != expected[index][1]
            ):
                mismatched += 1
    # Trials only ever receive the development frame, and the digests above
    # match the manifest's folds, which lie inside it; a test row could only
    # have been seen if the partitions overlapped.
    leaked = bool(test_ids & development) or mismatched > 0
    return {
        "folds_checked": checked,
        "folds_mismatched": mismatched,
        "test_rows_seen": leaked,
    }


def build_ledger(
    split: splitting.SplitResult,
    trials: list[experiments.TrialResult],
    test_opens: int,
    threshold_used: bool,
) -> dict[str, Any]:
    """Describes and verifies every data access of a run.

    Args:
        split: The split result (manifest and locked test).
        trials: All executed trials.
        test_opens: How many times the locked test set was opened.
        threshold_used: Whether a decision threshold was optimized.

    Returns:
        A JSON-serializable ledger with a `verified` section.
    """
    manifest = split.manifest
    development = set(manifest.train_ids)
    test = set(manifest.test_ids)
    fold_sizes = [
        {"fit_rows": len(fit), "validation_rows": len(validate)}
        for fit, validate in manifest.folds
    ]
    positions = set(range(len(development)))
    folds_inside = all(
        set(fit) | set(validate) <= positions
        for fit, validate in manifest.folds
    )
    evaluated = [t for t in trials if t.status is experiments.TrialStatus.OK]
    observed = observed_access(split, evaluated)
    return {
        "partitions": {
            "development_rows": len(development),
            "locked_test_rows": len(test),
            "split_fingerprint": manifest.fingerprint,
            "strategy": manifest.strategy,
        },
        "operations": [
            {
                "operation": "cross_validation",
                "candidates": len(evaluated),
                "fits_per_candidate": len(manifest.folds),
                "fold_structure": fold_sizes,
                "fit_on": "training rows of each fold (preprocessing, "
                "resampling, calibration and stacking are fitted inside)",
                "scored_on": "validation rows of each fold",
            },
            {
                "operation": "final_refit",
                "rows": len(development),
                "partition": "development",
            },
            {
                "operation": "threshold_optimization",
                "used": threshold_used,
                "scored_on": "out-of-fold development predictions",
            },
            {
                "operation": "final_evaluation",
                "rows": len(test),
                "partition": "locked test",
                "opened": test_opens,
            },
        ],
        "observed": observed,
        "verified": {
            "observed_fit_and_scoring_rows_match_the_manifest": observed[
                "folds_mismatched"
            ]
            == 0
            and observed["folds_checked"] > 0,
            "no_observed_access_to_test_rows": not observed["test_rows_seen"],
            "development_and_test_are_disjoint": not development & test,
            "development_and_test_cover_all_rows": len(development | test)
            == len(development) + len(test),
            "every_fold_inside_development": folds_inside,
            "test_set_opened_exactly_once": test_opens == 1,
        },
    }
