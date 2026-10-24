"""Replay verification: exact reproduction vs statistically comparable reruns.

Bit-for-bit reproduction cannot be guaranteed for every backend (threaded
libraries, GPUs), so a rerun is classified honestly:

* `exact`: identical configuration, data, split, trials, params and every
  fold score.
* `comparable`: same configuration, data, split and trial structure, and
  every trial's mean score within a tolerance of one fold standard deviation,
  with the same selected candidate.
* `different`: anything else, with the reasons.
"""

import dataclasses
import json
import pathlib
from typing import Any

EXACT = "exact"
COMPARABLE = "comparable"
DIFFERENT = "different"
DEFAULT_TOLERANCE_STD = 1.0
FLOOR = 1e-9


@dataclasses.dataclass(frozen=True)
class ReplayVerdict:
    """Outcome of comparing two runs.

    Attributes:
        level: `exact`, `comparable` or `different`.
        reasons: Why the level is not `exact`.
        max_abs_difference: Largest absolute fold-score difference.
        trials_compared: Number of trials compared.
        notes: Caveats, such as differing environments.
    """

    level: str
    reasons: list[str]
    max_abs_difference: float
    trials_compared: int
    notes: list[str]


def load_run(directory: str | pathlib.Path) -> dict[str, Any]:
    """Reads the comparable facts of a finished run from its directory."""
    root = pathlib.Path(directory)
    trials = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((root / "trials").glob("trial_*.json"))
    ]
    evaluation = json.loads(
        (root / "evaluation.json").read_text(encoding="utf-8")
    )
    parameter_path = root / "parameter_estimation.json"
    parameter_digest = None
    if parameter_path.exists():
        from scmle.evaluation import parameter_estimation

        parameter_digest = parameter_estimation.EstimationResult.read(
            parameter_path
        ).digest()
    return {
        "config": json.loads(
            (root / "config.json").read_text(encoding="utf-8")
        ),
        "dataset": json.loads(
            (root / "dataset_manifest.json").read_text(encoding="utf-8")
        ),
        "split": evaluation["split_fingerprint"],
        "trials": trials,
        "selected": (
            evaluation["selected_model"],
            evaluation["selected_strategy"],
            json.dumps(evaluation["selected_params"], sort_keys=True),
        ),
        "environment": (root / "environment.lock").read_text(encoding="utf-8")
        if (root / "environment.lock").exists()
        else "",
        "parameter_estimation": parameter_digest,
    }


def compare_runs(
    first: dict[str, Any],
    second: dict[str, Any],
    tolerance_std: float = DEFAULT_TOLERANCE_STD,
) -> ReplayVerdict:
    """Classifies a rerun against the original.

    Args:
        first: Facts of the original run (see `load_run`).
        second: Facts of the rerun.
        tolerance_std: Allowed mean-score difference, in units of the larger
            fold standard deviation, for a `comparable` verdict.

    Returns:
        The verdict.
    """
    reasons: list[str] = []
    notes: list[str] = []
    if first["environment"] != second["environment"]:
        notes.append("environments differ; exact replay is not guaranteed")
    if first["config"] != second["config"]:
        reasons.append("configurations differ")
    if first["dataset"]["data_hash"] != second["dataset"]["data_hash"]:
        reasons.append("datasets differ")
    if first["split"] != second["split"]:
        reasons.append("splits differ")
    if first.get("parameter_estimation") != second.get("parameter_estimation"):
        reasons.append("parameter-estimation evidence differs")
    a, b = first["trials"], second["trials"]
    identity = [
        (
            t["trial_id"],
            t["model_name"],
            t["strategy"],
            json.dumps(t["params"], sort_keys=True),
        )
        for t in a
    ]
    if identity != [
        (
            t["trial_id"],
            t["model_name"],
            t["strategy"],
            json.dumps(t["params"], sort_keys=True),
        )
        for t in b
    ]:
        reasons.append("the executed trials differ")
    if reasons:
        return ReplayVerdict(DIFFERENT, reasons, float("nan"), 0, notes)

    largest, exact, comparable = 0.0, True, True
    for left, right in zip(a, b, strict=True):
        if left["status"] != right["status"]:
            was, now = left["status"], right["status"]
            reasons.append(f"{left['trial_id']}: status {was} -> {now}")
            exact = False
            comparable = False
            continue
        gaps = [
            abs(x - y)
            for x, y in zip(
                left["fold_scores"], right["fold_scores"], strict=True
            )
        ]
        if gaps:
            largest = max(largest, *gaps)
        if left["fold_scores"] != right["fold_scores"]:
            exact = False
            allowed = tolerance_std * max(
                left["score_std"], right["score_std"], FLOOR
            )
            if abs(left["score_mean"] - right["score_mean"]) > allowed:
                comparable = False
                reasons.append(
                    f"{left['trial_id']}: mean score moved "
                    f"{abs(left['score_mean'] - right['score_mean']):.4g} "
                    f"(allowed {allowed:.4g})"
                )
    if first["selected"] != second["selected"]:
        exact = False
        comparable = False
        reasons.append("a different candidate was selected")
    if exact:
        return ReplayVerdict(EXACT, [], 0.0, len(a), notes)
    if comparable:
        reasons.append("fold scores differ within tolerance")
        return ReplayVerdict(COMPARABLE, reasons, largest, len(a), notes)
    return ReplayVerdict(DIFFERENT, reasons, largest, len(a), notes)
