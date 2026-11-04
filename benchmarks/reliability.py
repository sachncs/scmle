"""Reliability checks: does the package behave as it claims?

`run_benchmarks` measures accuracy against a reference. This module measures
trustworthiness. Each check states a property the documentation promises, runs
it on synthetic data with known ground truth, and records the measured value
next to the threshold it must meet:

* `determinism`: the same seed gives the same selection and test score.
* `replay`: `MLProject.verify_replay` classifies a rerun as `exact`.
* `no_false_skill`: pure-noise labels never produce a convincing score.
* `leak_detection`: a planted leaked column is flagged and can block the run.
* `bundle_roundtrip`: a saved, signed bundle predicts identically when loaded.
* `tamper_detection`: a modified model file or a wrong key is refused.
* `interval_coverage`: conformal intervals cover at least the stated share of
  fresh rows.
* `drift_detection`: a shifted batch is flagged and a clean one is not.
* `failure_isolation`: a crashing candidate does not fail the run.

Usage:
    python -m benchmarks.reliability [--only NAME ...]
        [--output benchmarks/results_reliability.json]

The exit status is non-zero if any check fails, so it can gate a release.
"""

import argparse
from collections.abc import Callable
import dataclasses
import json
import logging
import pathlib
import sys
import tempfile
import time
import traceback
from typing import Any

import numpy as np
import pandas as pd
from sklearn import base

import scmle
from scmle.contracts import errors
from scmle.contracts import tasks
from scmle.evaluation import coverage
from scmle.monitoring import monitor

LOGGER = logging.getLogger(__name__)

SIGNING_KEY = "reliability-benchmark-key"
NOISE_SEEDS = (0, 1, 2, 3, 4)
CRASH_ROWS = 200


class CrashingClassifier(base.ClassifierMixin, base.BaseEstimator):
    """Fails on real-sized data; used to prove one bad candidate is isolated.

    Registration runs a conformance check on a tiny problem, so the estimator
    behaves until it sees `CRASH_ROWS` rows, then fails inside the trial.
    """

    def fit(self, features: Any, target: Any) -> "CrashingClassifier":
        """Fits a majority-class rule, or raises on real-sized data."""
        if len(features) >= CRASH_ROWS:
            raise RuntimeError("deliberate failure from the reliability check")
        self.classes_, counts = np.unique(
            np.asarray(target), return_counts=True
        )
        self.majority_ = self.classes_[int(np.argmax(counts))]
        return self

    def predict(self, features: Any) -> np.ndarray:
        """Predicts the majority class."""
        return np.full(len(features), self.majority_)

    def predict_proba(self, features: Any) -> np.ndarray:
        """One-hot probabilities of the majority class."""
        proba = np.zeros((len(features), len(self.classes_)))
        proba[:, int(np.where(self.classes_ == self.majority_)[0][0])] = 1.0
        return proba


@dataclasses.dataclass
class CheckRecord:
    """Outcome of one reliability check.

    Attributes:
        name: Check name.
        claim: The property being checked, in one sentence.
        passed: Whether the measured value met the threshold.
        measured: What was observed.
        threshold: What the claim requires.
        seconds: Wall-clock time.
        error: Traceback text if the check itself crashed.
    """

    name: str
    claim: str
    passed: bool = False
    measured: Any = None
    threshold: str = ""
    seconds: float = 0.0
    error: str = ""


def fast_project(
    data: pd.DataFrame, target: str, directory: pathlib.Path, **kwargs: Any
) -> scmle.MLProject:
    """Builds a small in-process project so checks stay quick."""
    kwargs.setdefault("search", tasks.SearchSpec(top_models=0))
    kwargs.setdefault("strategies", tasks.StrategySpec(top_models=0))
    kwargs.setdefault("ensemble", tasks.EnsembleSpec(size=0))
    kwargs.setdefault(
        "models",
        tasks.ModelSpec(
            exclude_families=["bagging", "deep_learning", "foundation"]
        ),
    )
    kwargs.setdefault("enforcement", "cooperative")
    kwargs.setdefault("time_budget", 120)
    return scmle.MLProject(data, target, output_dir=directory, **kwargs)


def binary_data(rows: int = 600, seed: int = 0) -> pd.DataFrame:
    """A binary problem with a real, learnable signal."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "a": rng.normal(size=rows),
            "b": rng.normal(size=rows),
            "c": rng.choice(["x", "y", "z"], rows),
        }
    )
    logit = 1.5 * frame.a - frame.b + (frame.c == "x") * 0.8
    frame["y"] = (logit + rng.normal(scale=0.7, size=rows) > 0).astype(int)
    return frame


def regression_data(rows: int, seed: int) -> pd.DataFrame:
    """A linear regression problem with unit-variance noise."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {"a": rng.normal(size=rows), "b": rng.normal(size=rows)}
    )
    frame["y"] = 3 * frame.a - 2 * frame.b + rng.normal(size=rows)
    return frame


def check_determinism(directory: pathlib.Path) -> CheckRecord:
    """Two runs with one seed select the same model with the same score."""
    record = CheckRecord(
        "determinism",
        "Same data and seed give the same selection and test score.",
        threshold="same model, |score difference| <= 1e-9",
    )
    data = binary_data()
    runs = [
        fast_project(data, "y", directory / f"run{i}", seed=7).run()
        for i in range(2)
    ]
    scores = [r.evaluation["test_metrics"]["roc_auc"] for r in runs]
    record.measured = {
        "models": [r.selected_model for r in runs],
        "score_difference": abs(scores[0] - scores[1]),
    }
    record.passed = (
        runs[0].selected_model == runs[1].selected_model
        and record.measured["score_difference"] <= 1e-9
    )
    return record


def check_replay(directory: pathlib.Path) -> CheckRecord:
    """The replay verifier agrees that an unchanged rerun is exact."""
    record = CheckRecord(
        "replay",
        "verify_replay classifies a rerun of the same configuration as exact.",
        threshold="level == exact",
    )
    data = binary_data()
    result = fast_project(data, "y", directory / "original", seed=3).run()
    verdict = scmle.MLProject.verify_replay(
        result.run_dir, data=data, output_dir=directory / "rerun"
    )
    record.measured = {
        "level": verdict.level,
        "reasons": list(verdict.reasons),
    }
    record.passed = verdict.level == "exact"
    return record


def check_no_false_skill(directory: pathlib.Path) -> CheckRecord:
    """Labels with no signal never yield a convincing score."""
    record = CheckRecord(
        "no_false_skill",
        "On pure-noise labels the locked-test ROC-AUC stays near 0.5.",
        threshold="mean |AUC - 0.5| <= 0.08 and worst <= 0.15 over 5 seeds",
    )
    gaps = []
    for seed in NOISE_SEEDS:
        rng = np.random.default_rng(seed)
        frame = pd.DataFrame({f"f{i}": rng.normal(size=500) for i in range(8)})
        frame["y"] = rng.integers(0, 2, 500)
        result = fast_project(
            frame, "y", directory / f"seed{seed}", seed=seed
        ).run()
        gaps.append(abs(result.evaluation["test_metrics"]["roc_auc"] - 0.5))
    record.measured = {"mean_gap": float(np.mean(gaps)), "worst": max(gaps)}
    record.passed = record.measured["mean_gap"] <= 0.08 and max(gaps) <= 0.15
    return record


def check_leak_detection(directory: pathlib.Path) -> CheckRecord:
    """A column that encodes the label is flagged, and can block the run.

    The flag is the permutation screen with Benjamini-Yekutieli control: the
    column must be a suspect (adjusted p-value below alpha and effect-size
    lower bound above the ceiling) whose mechanism is certified, and a clean
    column must not be.
    """
    record = CheckRecord(
        "leak_detection",
        "A planted leaked column is flagged by the certified screen and "
        "leakage_policy=block stops.",
        threshold="flagged, certified and blocked; clean columns not flagged",
    )
    leaky = binary_data().assign(leak=lambda f: f.y * 3.0 + 1)
    result = fast_project(leaky, "y", directory / "warn").run()
    screen = result.profile.leakage
    flagged = (
        screen["suspects"] == ["leak"]
        and result.profile.blocking_leaks == ("leak",)
        and any("'leak'" in item for item in result.profile.risk_findings)
    )
    try:
        fast_project(
            leaky, "y", directory / "block", leakage_policy="block"
        ).run()
        blocked = False
    except errors.BlockedError:
        blocked = True
    record.measured = {
        "flagged": flagged,
        "blocked": blocked,
        "suspects": screen["suspects"],
        "alpha": screen["alpha"],
    }
    record.passed = flagged and blocked
    return record


def check_bundle_roundtrip(directory: pathlib.Path) -> CheckRecord:
    """A saved, signed bundle predicts identically after loading."""
    record = CheckRecord(
        "bundle_roundtrip",
        "Predictions are identical before saving and after a verified load.",
        threshold="max |difference| == 0",
    )
    data = binary_data()
    result = fast_project(data, "y", directory / "run").run()
    features = data.drop(columns="y")
    bundle_directory = directory / "bundle"
    result.save(bundle_directory, signing_key=SIGNING_KEY)
    loaded = scmle.predict(bundle_directory, features, signing_key=SIGNING_KEY)
    difference = float(
        np.abs(
            np.asarray(result.predict(features), dtype=float)
            - np.asarray(loaded, dtype=float)
        ).max()
    )
    record.measured = {"max_difference": difference}
    record.passed = difference == 0.0
    return record


def check_tamper_detection(directory: pathlib.Path) -> CheckRecord:
    """A modified model file and a wrong key are both refused."""
    record = CheckRecord(
        "tamper_detection",
        "Loading refuses a wrong key and a modified model file.",
        threshold="both refused with ArtifactError",
    )
    data = binary_data()
    result = fast_project(data, "y", directory / "run").run()
    bundle_directory = directory / "bundle"
    result.save(bundle_directory, signing_key=SIGNING_KEY)

    def refused(key: str) -> bool:
        try:
            scmle.load_model(bundle_directory, signing_key=key)
        except errors.ArtifactError:
            return True
        return False

    wrong_key = refused("not-the-key")
    pipeline = bundle_directory / "pipeline.joblib"
    pipeline.write_bytes(pipeline.read_bytes() + b"\0")
    modified = refused(SIGNING_KEY)
    record.measured = {"wrong_key": wrong_key, "modified_file": modified}
    record.passed = wrong_key and modified
    return record


def check_interval_coverage(directory: pathlib.Path) -> CheckRecord:
    """Prediction intervals contain at least the stated share of new rows."""
    record = CheckRecord(
        "interval_coverage",
        "A 90% conformal interval covers at least 88% of fresh rows "
        "(nominal minus the 0.02 charter tolerance).",
        threshold="empirical coverage >= 0.88",
    )
    train = regression_data(800, seed=1)
    result = fast_project(train, "y", directory / "run").run()
    fresh = regression_data(2000, seed=2)
    intervals = result.bundle.predict_interval(
        fresh.drop(columns="y"), coverage=0.9
    )
    inside = (fresh.y >= intervals["lower"]) & (fresh.y <= intervals["upper"])
    hits = int(inside.sum())
    covered = hits / len(inside)
    low, high = coverage.clopper_pearson(hits, len(inside))
    record.measured = {
        "coverage": covered,
        "clopper_pearson_95": [low, high],
        "nominal": 0.9,
    }
    record.passed = covered >= 0.9 - coverage.DEFAULT_TOLERANCE
    return record


def check_drift_detection(directory: pathlib.Path) -> CheckRecord:
    """A shifted batch is flagged; a same-distribution batch is not."""
    record = CheckRecord(
        "drift_detection",
        "Monitoring flags a +3 sigma shift and stays quiet on clean data.",
        threshold="shifted != none and clean == none",
    )
    result = fast_project(binary_data(), "y", directory / "run").run()
    watcher = monitor.ModelMonitor(result.bundle)
    clean = watcher.assess(binary_data(500, seed=11).drop(columns="y"))
    moved = binary_data(500, seed=12).drop(columns="y")
    moved["a"] = moved["a"] + 3.0
    shifted = watcher.assess(moved)
    record.measured = {
        "clean": clean.recommendation.action,
        "shifted": shifted.recommendation.action,
    }
    quiet = clean.recommendation.action == "none"
    loud = shifted.recommendation.action != "none"
    record.passed = quiet and loud
    return record


def check_failure_isolation(directory: pathlib.Path) -> CheckRecord:
    """A candidate that always crashes does not fail the run."""
    record = CheckRecord(
        "failure_isolation",
        "A crashing candidate is recorded as failed; the run still completes.",
        threshold="run COMPLETED, crashing trial not ok, another trial ok",
    )
    scmle.register_estimator(
        "crashing_probe", classifier=CrashingClassifier, cost=1, replace=True
    )
    try:
        result = fast_project(binary_data(), "y", directory / "run").run()
    finally:
        scmle.unregister_estimator("crashing_probe")
    board = result.leaderboard()
    crashed = board[board.model_name == "crashing_probe"]
    others_ok = bool((board.status == "ok").any())
    record.measured = {
        "state": result.state,
        "crashing_status": [str(s) for s in crashed.status],
        "others_ok": others_ok,
    }
    # The probe must really have run and failed; absence would prove nothing.
    record.passed = (
        result.state == "COMPLETED"
        and others_ok
        and len(crashed) > 0
        and not (crashed.status == "ok").any()
    )
    return record


CHECKS: dict[str, Callable[[pathlib.Path], CheckRecord]] = {
    "determinism": check_determinism,
    "replay": check_replay,
    "no_false_skill": check_no_false_skill,
    "leak_detection": check_leak_detection,
    "bundle_roundtrip": check_bundle_roundtrip,
    "tamper_detection": check_tamper_detection,
    "interval_coverage": check_interval_coverage,
    "drift_detection": check_drift_detection,
    "failure_isolation": check_failure_isolation,
}


def run_check(name: str) -> CheckRecord:
    """Runs one check in a scratch directory; a crash is a failed record."""
    started = time.perf_counter()
    with tempfile.TemporaryDirectory() as scratch:
        try:
            record = CHECKS[name](pathlib.Path(scratch))
        except Exception:
            record = CheckRecord(name, CHECKS[name].__doc__ or "")
            record.error = traceback.format_exc()
    record.seconds = round(time.perf_counter() - started, 2)
    return record


def main(argv: list[str] | None = None) -> int:
    """Runs the selected checks, writes results, returns 1 on any failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", default=[], choices=list(CHECKS))
    parser.add_argument(
        "--output", default="benchmarks/results_reliability.json"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING, format="%(message)s", stream=sys.stdout
    )
    LOGGER.setLevel(logging.INFO)
    names = args.only or list(CHECKS)
    records = [run_check(name) for name in names]
    for record in records:
        LOGGER.info(
            "%-18s %s  %s (%.1fs)",
            record.name,
            "PASS" if record.passed else "FAIL",
            record.measured if record.measured is not None else record.error,
            record.seconds,
        )
    pathlib.Path(args.output).write_text(
        json.dumps(
            {
                "version": scmle.__version__,
                "results": [dataclasses.asdict(r) for r in records],
            },
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    failed = [r.name for r in records if not r.passed]
    LOGGER.info("%d/%d checks passed", len(records) - len(failed), len(records))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
