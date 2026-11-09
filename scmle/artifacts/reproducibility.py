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

from scmle.evaluation import parameter_estimation

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
        reasons: Why the level is not `exact` (empty for `exact`).
        max_abs_difference: Largest absolute fold-score difference over the
            compared trials whose status matches; 0.0 for `exact`, and NaN
            when the runs differ before any trial is compared.
        trials_compared: Number of trials compared (0 when the runs differ
            before any trial is compared).
        notes: Caveats, such as differing environments (they do not change
            the level).
    """

    level: str
    reasons: list[str]
    max_abs_difference: float
    trials_compared: int
    notes: list[str]


def load_run(directory: str | pathlib.Path) -> dict[str, Any]:
    """Reads the comparable facts of a finished run from its directory.

    Args:
        directory: A completed run directory.

    Returns:
        A dict with the keys `config`, `dataset` (the manifest), `split`
        (the split fingerprint), `trials` (the `trials/trial_*.json` records
        in file-name order), `selected` (a tuple of the selected model, its
        strategy and its parameters as sorted-key JSON text), `environment`
        (the environment lock text, empty if absent) and
        `parameter_estimation` (the report digest, None if the run has no such
        report). A missing required file raises the underlying file or JSON
        error.
    """
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
        parameter_digest = parameter_estimation.Estimation.read(
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


def trial_identity(trial: dict[str, Any]) -> tuple[Any, ...]:
    """Names the part of a trial that must not change between runs.

    Args:
        trial: A trial record.

    Returns:
        Its id, model name, strategy and key-sorted JSON parameters.
    """
    return (
        trial["trial_id"],
        trial["model_name"],
        trial["strategy"],
        json.dumps(trial["params"], sort_keys=True),
    )


def structural_differences(
    first: dict[str, Any], second: dict[str, Any]
) -> tuple[list[str], list[str]]:
    """Compares what two runs must share before their scores are compared.

    Args:
        first: Facts of the original run (see `load_run`).
        second: Facts of the rerun.

    Returns:
        The reasons the runs differ (configuration, dataset, split,
        parameter-estimation evidence, executed trials) and the notes, which
        do not change the verdict (differing environments).
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
    if [trial_identity(t) for t in first["trials"]] != [
        trial_identity(t) for t in second["trials"]
    ]:
        reasons.append("the executed trials differ")
    return reasons, notes


def compare_trial_scores(
    first_trials: list[dict[str, Any]],
    second_trials: list[dict[str, Any]],
    tolerance_std: float,
) -> tuple[float, bool, bool, list[str]]:
    """Compares the scores of two runs' trials, which share their structure.

    Args:
        first_trials: Trial records of the original run.
        second_trials: Trial records of the rerun, in the same order.
        tolerance_std: Allowed difference between a trial's mean scores, in
            units of the larger of the two trials' score standard deviations
            (floored at `FLOOR`).

    Returns:
        The largest fold-score gap, whether every fold score is identical,
        whether every trial is within tolerance, and the reasons for any
        trial that is not.
    """
    reasons: list[str] = []
    largest, exact, comparable = 0.0, True, True
    for left, right in zip(first_trials, second_trials, strict=True):
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
    return largest, exact, comparable, reasons


def compare_runs(
    first: dict[str, Any],
    second: dict[str, Any],
    tolerance_std: float = DEFAULT_TOLERANCE_STD,
) -> ReplayVerdict:
    """Classifies a rerun against the original.

    Args:
        first: Facts of the original run (see `load_run`).
        second: Facts of the rerun.
        tolerance_std: Allowed difference between a trial's mean scores, in
            units of the larger of the two trials' score standard deviations
            (floored at 1e-9), for a `comparable` verdict.

    Returns:
        `different` if the configuration, dataset hash, split, parameter
        estimation digest or executed trials (id, model, strategy, params)
        differ, if a trial's status changed, a mean moved beyond the
        tolerance, or a different candidate was selected. Otherwise `exact` if
        every fold score is identical, else `comparable`.
    """
    reasons, notes = structural_differences(first, second)
    a, b = first["trials"], second["trials"]
    if reasons:
        return ReplayVerdict(DIFFERENT, reasons, float("nan"), 0, notes)

    largest, exact, comparable, score_reasons = compare_trial_scores(
        a, b, tolerance_std
    )
    reasons.extend(score_reasons)
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
