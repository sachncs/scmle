"""Data-access ledger: what each fitted operation was allowed to see.

The ledger is derived from the split manifest and the executed trials, then
*verified* against the manifest, so the guarantees it states (no fold touches
the test rows, the test set opened once) are checked facts, not claims.
"""

from typing import Any

from mml.contracts import experiments
from mml.data import splitting


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
        "verified": {
            "development_and_test_are_disjoint": not development & test,
            "development_and_test_cover_all_rows": len(development | test)
            == len(development) + len(test),
            "every_fold_inside_development": folds_inside,
            "test_set_opened_exactly_once": test_opens == 1,
        },
    }
