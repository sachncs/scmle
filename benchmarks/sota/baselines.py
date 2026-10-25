"""Reference systems for the state-of-the-art comparison, run out of process.

The baselines (AutoGluon, LightGBM, FLAML) cannot be installed next to this
repository because AutoGluon needs Python below 3.14. This module therefore
never imports them in the main environment: the driver launches a *worker* in a
separate interpreter, with a hard wall-clock limit, and turns whatever happens
into one `records.RunRecord` per (system, budget, task, fold, repeat). A
failure or a timeout is a record, never a silent drop.

Only development tasks may be run. A hold-out task id is refused unless
`--open-holdout` is given explicitly (the protocol allows that once).

Scores are recorded in the metric's natural units: ROC AUC (higher is better)
for binary tasks, log loss (lower) for multiclass, RMSE (lower) for
regression. `LOWER_IS_BETTER` says which way each metric points.

Usage:
    python -m benchmarks.sota.baselines run \
        --python /tmp/rel/ta/bin/python --tasks 363696 363685 \
        --systems autogluon-best lightgbm --budgets 60 600 --lite \
        --output results.jsonl

The worker (`python -m benchmarks.sota.baselines worker ...`) is started by
`run` and is not meant to be called by hand.
"""

import argparse
import dataclasses
import importlib
import json
import logging
import os
import pathlib
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Any

from benchmarks.sota import records
from benchmarks.sota import split

LOGGER = logging.getLogger(__name__)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SYSTEMS = ("autogluon-best", "autogluon-high", "lightgbm", "flaml")
PRESETS = {
    "autogluon-best": "best_quality",
    "autogluon-high": "high_quality",
}
METRICS = {
    "binary": "roc_auc",
    "multiclass": "log_loss",
    "regression": "rmse",
}
LOWER_IS_BETTER = {"roc_auc": False, "log_loss": True, "rmse": True}
FOLDS = 3
LITE_REPEATS = 1
FULL_REPEATS = 2
DEFAULT_CPUS = 12
MAX_CPUS = 12
GRACE_FRACTION = 0.25
MIN_GRACE_S = 120.0
SEED = 0
TAIL_CHARS = 1500


class HoldoutRefusedError(ValueError):
    """Raised when a hold-out task is requested without the explicit flag."""


@dataclasses.dataclass(frozen=True)
class Job:
    """One (system, budget, task, split) unit of work."""

    system: str
    budget_s: int
    task_id: int
    fold: int
    repeat: int


@dataclasses.dataclass
class Outcome:
    """What a worker reports back through its result file."""

    status: str
    score: float | None = None
    seconds: float = 0.0
    error: str = ""
    versions: dict[str, str] = dataclasses.field(default_factory=dict)


def sets() -> dict[str, list[int]]:
    """Reads the committed development / hold-out assignment."""
    document = json.loads(split.OUTPUT.read_text(encoding="utf-8"))
    return {
        "development": [int(i) for i in document["development"]],
        "holdout": [int(i) for i in document["holdout"]],
    }


def check_tasks(task_ids: list[int], open_holdout: bool = False) -> None:
    """Refuses hold-out tasks, and unknown ones, unless explicitly opened.

    The message names no hold-out id other than the offending ones the caller
    already supplied, and never loads or prints a score.

    Args:
        task_ids: OpenML task ids requested by the caller.
        open_holdout: True only for the one final evaluation.

    Raises:
        HoldoutRefusedError: If a requested id is in the hold-out set and
            `open_holdout` is false.
        ValueError: If a requested id is not a protocol task at all.
    """
    assignment = sets()
    known = set(assignment["development"]) | set(assignment["holdout"])
    unknown = sorted(set(task_ids) - known)
    if unknown:
        raise ValueError(f"not protocol tasks: {unknown}")
    sealed = sorted(set(task_ids) & set(assignment["holdout"]))
    if sealed and not open_holdout:
        raise HoldoutRefusedError(
            f"refusing hold-out tasks {sealed}: pass --open-holdout only "
            "for the single final evaluation"
        )


def problem_types() -> dict[int, str]:
    """Maps each protocol task id to its problem type."""
    return {
        int(row["task_id"]): row["problem_type"] for row in split.load_tasks()
    }


def splits(lite: bool) -> list[tuple[int, int]]:
    """The (fold, repeat) pairs to run: the first split only when lite."""
    repeats = LITE_REPEATS if lite else FULL_REPEATS
    pairs = [(f, r) for r in range(repeats) for f in range(FOLDS)]
    return pairs[:1] if lite else pairs


def plan(
    systems: list[str],
    budgets: list[int],
    task_ids: list[int],
    lite: bool,
) -> list[Job]:
    """Expands the request into jobs, ordered task by task."""
    return [
        Job(system, budget, task_id, fold, repeat)
        for task_id in task_ids
        for budget in budgets
        for system in systems
        for fold, repeat in splits(lite)
    ]


def done(path: pathlib.Path) -> set[tuple[str, int, int, int, int]]:
    """Keys of the records already in a results file, for resuming."""
    if not path.exists():
        return set()
    return {
        (r.system, r.budget_s, r.task_id, r.fold, r.repeat)
        for r in records.read(path)
    }


# --------------------------------------------------------------------------
# Driver side: runs in the main environment and imports no baseline.
# --------------------------------------------------------------------------


def worker_command(
    python: pathlib.Path,
    job: Job,
    cpus: int,
    result: pathlib.Path,
    open_holdout: bool = False,
) -> list[str]:
    """The command line that runs one job in the baseline environment."""
    command = [
        str(python),
        "-m",
        "benchmarks.sota.baselines",
        "worker",
        "--system",
        job.system,
        "--budget",
        str(job.budget_s),
        "--task",
        str(job.task_id),
        "--fold",
        str(job.fold),
        "--repeat",
        str(job.repeat),
        "--cpus",
        str(cpus),
        "--result",
        str(result),
    ]
    if open_holdout:
        command.append("--open-holdout")
    return command


def kill_group(process: "subprocess.Popen[str]") -> None:
    """Kills the worker and every process it spawned (Ray, joblib)."""
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        process.kill()


def execute(
    python: pathlib.Path, job: Job, cpus: int, open_holdout: bool = False
) -> tuple[Outcome, float]:
    """Runs one job in a subprocess under a hard wall-clock limit.

    Returns:
        The outcome and the elapsed wall-clock seconds of the subprocess.
    """
    grace = max(MIN_GRACE_S, GRACE_FRACTION * job.budget_s)
    limit = job.budget_s + grace
    scratch = pathlib.Path(tempfile.mkdtemp(prefix="sota_job_"))
    result = scratch / "result.json"
    environment = dict(os.environ, PYTHONPATH=str(REPO_ROOT))
    started = time.monotonic()
    process = subprocess.Popen(
        worker_command(python, job, cpus, result, open_holdout),
        cwd=REPO_ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        output, _ = process.communicate(timeout=limit)
    except subprocess.TimeoutExpired:
        kill_group(process)
        process.communicate()
        elapsed = time.monotonic() - started
        shutil.rmtree(scratch, ignore_errors=True)
        message = f"hard limit of {limit:.0f} s exceeded"
        return Outcome("timeout", error=message, seconds=elapsed), elapsed
    kill_group(process)
    elapsed = time.monotonic() - started
    outcome = read_outcome(result, process.returncode, output or "")
    shutil.rmtree(scratch, ignore_errors=True)
    return outcome, elapsed


def read_outcome(path: pathlib.Path, returncode: int, output: str) -> Outcome:
    """Turns a worker's result file and exit status into an outcome."""
    tail = output[-TAIL_CHARS:]
    if path.exists():
        try:
            fields = json.loads(path.read_text(encoding="utf-8"))
            return Outcome(**fields)
        except (ValueError, TypeError) as error:
            return Outcome("failed", error=f"unreadable result: {error}")
    return Outcome(
        "failed", error=f"worker exited {returncode} without a result: {tail}"
    )


def to_record(
    job: Job, metric: str, outcome: Outcome, elapsed: float
) -> records.RunRecord:
    """Builds the record for a job, whatever its outcome."""
    status = outcome.status
    score = outcome.score
    if status == "ok" and score is None:
        status = "failed"
        outcome.error = "worker reported ok without a score"
    if status != "ok":
        score = None
    return records.RunRecord(
        system=job.system,
        budget_s=job.budget_s,
        task_id=job.task_id,
        fold=job.fold,
        repeat=job.repeat,
        metric=metric,
        score=score,
        seconds=outcome.seconds or elapsed,
        status=status,
        error=outcome.error,
        versions=outcome.versions,
    )


def run_all(args: argparse.Namespace) -> int:
    """Runs the requested jobs sequentially and appends their records."""
    if not 1 <= args.cpus <= MAX_CPUS:
        raise ValueError(f"--cpus must be in 1..{MAX_CPUS}")
    task_ids = [int(i) for i in args.tasks]
    check_tasks(task_ids, open_holdout=args.open_holdout)
    types = problem_types()
    output = pathlib.Path(args.output)
    finished = done(output)
    python = pathlib.Path(args.python)
    failures = 0
    for job in plan(args.systems, args.budgets, task_ids, args.lite):
        key = (job.system, job.budget_s, job.task_id, job.fold, job.repeat)
        if key in finished:
            LOGGER.info("skipping finished %s", key)
            continue
        LOGGER.info("running %s", job)
        outcome, elapsed = execute(python, job, args.cpus, args.open_holdout)
        outcome.versions["cpus"] = str(args.cpus)
        record = to_record(job, METRICS[types[job.task_id]], outcome, elapsed)
        records.append(output, record)
        LOGGER.info(
            "%s score=%s seconds=%.1f",
            record.status,
            record.score,
            record.seconds,
        )
        failures += record.status != "ok"
    return 1 if failures else 0


# --------------------------------------------------------------------------
# Worker side: runs in the baseline environment (Python < 3.14).
# --------------------------------------------------------------------------


def load_split(task_id: int, fold: int, repeat: int) -> Any:
    """Loads a task's rows and its stored OpenML split through TabArena."""
    utils = importlib.import_module("tabarena.benchmark.task.openml.task_utils")
    task = utils.get_task_with_retry(task_id=task_id, download_splits=True)
    # Not `utils.get_task_data`: its cached-pickle shortcut fails to load
    # categorical columns written by another pandas major version.
    dataset = task.get_dataset(download_data=True)
    features, target, _, _ = dataset.get_data(task.target_name)
    train, test = task.get_train_test_split_indices(fold=fold, repeat=repeat)
    return task, features, target, train, test


def score_predictions(
    problem_type: str, truth: Any, prediction: Any, classes: Any
) -> float:
    """Scores predictions with the task metric, in natural units.

    Args:
        problem_type: `binary`, `multiclass` or `regression`.
        truth: True test targets (labels for classification).
        prediction: Class probabilities (n, k) or regression values (n,).
        classes: Class labels in the column order of `prediction`.

    Returns:
        ROC AUC, log loss or RMSE.
    """
    metrics = importlib.import_module("sklearn.metrics")
    if problem_type == "regression":
        return float(metrics.root_mean_squared_error(truth, prediction))
    if problem_type == "binary":
        return float(
            metrics.roc_auc_score(truth == classes[1], prediction[:, 1])
        )
    return float(metrics.log_loss(truth, prediction, labels=list(classes)))


def autogluon_fit(
    system: str, frame: Any, target: str, problem_type: str, run: Job, cpus: int
) -> Any:
    """Fits AutoGluon with the preset of `system` inside the budget."""
    tabular = importlib.import_module("autogluon.tabular")
    path = tempfile.mkdtemp(prefix="sota_ag_")
    predictor = tabular.TabularPredictor(
        label=target,
        problem_type=problem_type,
        eval_metric=METRICS[problem_type],
        path=path,
        verbosity=0,
    ).fit(
        frame,
        presets=PRESETS[system],
        time_limit=run.budget_s,
        num_cpus=cpus,
        num_gpus=0,
    )
    return predictor, path


def lightgbm_fit(
    train_x: Any,
    train_y: Any,
    problem_type: str,
    budget_s: float,
    cpus: int,
) -> Any:
    """Random-searches LightGBM on a holdout, then refits the best config.

    The search uses at most 70% of the budget; the refit on all training rows
    uses the best iteration count found on the holdout.
    """
    lgb = importlib.import_module("lightgbm")
    numpy = importlib.import_module("numpy")
    selection = importlib.import_module("sklearn.model_selection")
    started = time.monotonic()
    search_until = started + 0.7 * budget_s
    stratify = train_y if problem_type != "regression" else None
    fit_x, val_x, fit_y, val_y = selection.train_test_split(
        train_x,
        train_y,
        test_size=0.2,
        random_state=SEED,
        stratify=stratify,
    )
    generator = numpy.random.default_rng(SEED)
    estimator = (
        lgb.LGBMRegressor
        if problem_type == "regression"
        else lgb.LGBMClassifier
    )
    eval_metric = {
        "binary": "auc",
        "multiclass": "multi_logloss",
        "regression": "rmse",
    }[problem_type]

    def stop_at_deadline(env: Any) -> None:
        if time.monotonic() > search_until:
            raise lgb.callback.EarlyStopException(
                env.iteration, env.evaluation_result_list
            )

    best_params: dict[str, Any] = {}
    best_value = None
    best_iteration = 100
    trial = 0
    while trial == 0 or time.monotonic() < search_until:
        params = lightgbm_params(generator, trial)
        model = estimator(
            n_estimators=2000,
            n_jobs=cpus,
            random_state=SEED,
            verbose=-1,
            **params,
        )
        model.fit(
            fit_x,
            fit_y,
            eval_set=[(val_x, val_y)],
            eval_metric=eval_metric,
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                stop_at_deadline,
            ],
        )
        name = next(iter(model.best_score_["valid_0"]))
        value = model.best_score_["valid_0"][name]
        better = best_value is None or (
            value > best_value if eval_metric == "auc" else value < best_value
        )
        if better:
            best_value = value
            best_params = params
            best_iteration = max(int(model.best_iteration_ or 100), 10)
        trial += 1
    final = estimator(
        n_estimators=max(10, int(best_iteration * 1.1)),
        n_jobs=cpus,
        random_state=SEED,
        verbose=-1,
        **best_params,
    )
    final.fit(train_x, train_y)
    return final


def lightgbm_params(generator: Any, trial: int) -> dict[str, Any]:
    """Draws a LightGBM configuration; trial 0 is the usual default-like one."""
    if trial == 0:
        return {"learning_rate": 0.05, "num_leaves": 31}
    return {
        "learning_rate": float(10 ** generator.uniform(-2.0, -1.0)),
        "num_leaves": int(2 ** generator.uniform(3.0, 8.0)),
        "min_child_samples": int(2 ** generator.uniform(1.0, 6.5)),
        "colsample_bytree": float(generator.uniform(0.4, 1.0)),
        "subsample": float(generator.uniform(0.5, 1.0)),
        "subsample_freq": 1,
        "reg_lambda": float(10 ** generator.uniform(-3.0, 1.0)),
        "reg_alpha": float(10 ** generator.uniform(-3.0, 0.0)),
    }


def prepare_frame(features: Any) -> Any:
    """Gives object columns the category dtype that LightGBM and FLAML use."""
    frame = features.copy()
    for column in frame.columns:
        if frame[column].dtype == object:
            frame[column] = frame[column].astype("category")
    return frame


def flaml_fit(
    train_x: Any, train_y: Any, problem_type: str, budget_s: float, cpus: int
) -> Any:
    """Fits FLAML's AutoML inside the budget."""
    flaml = importlib.import_module("flaml")
    automl = flaml.AutoML()
    automl.fit(
        train_x,
        train_y,
        task="regression" if problem_type == "regression" else "classification",
        metric=METRICS[problem_type],
        time_budget=budget_s,
        n_jobs=cpus,
        seed=SEED,
        verbose=0,
    )
    return automl


def versions_of(system: str) -> dict[str, str]:
    """Versions of the system and its main dependencies."""
    names = {
        "autogluon-best": ("autogluon.tabular", "lightgbm"),
        "autogluon-high": ("autogluon.tabular", "lightgbm"),
        "lightgbm": ("lightgbm",),
        "flaml": ("flaml", "lightgbm"),
    }[system]
    found = {"python": platform.python_version()}
    for name in names:
        module = importlib.import_module(name)
        found[name] = str(getattr(module, "__version__", "unknown"))
    return found


def work(job: Job, cpus: int) -> Outcome:
    """Fits one system on one stored split and scores the test rows."""
    task, features, target, train, test = load_split(
        job.task_id, job.fold, job.repeat
    )
    problem_type = problem_types()[job.task_id]
    frame = prepare_frame(features)
    train_x, test_x = frame.iloc[train], frame.iloc[test]
    train_y, test_y = target.iloc[train], target.iloc[test]
    started = time.monotonic()
    path = None
    if job.system in PRESETS:
        combined = train_x.copy()
        combined[task.target_name] = train_y.to_numpy()
        model, path = autogluon_fit(
            job.system, combined, task.target_name, problem_type, job, cpus
        )
        if problem_type == "regression":
            prediction = model.predict(test_x).to_numpy()
            classes = None
        else:
            proba = model.predict_proba(test_x, as_multiclass=True)
            classes = list(proba.columns)
            prediction = proba.to_numpy()
    else:
        if problem_type == "regression":
            classes = None
            fit_y = train_y.astype(float)
        else:
            classes = sorted(train_y.unique().tolist())
            fit_y = train_y
        if job.system == "lightgbm":
            model = lightgbm_fit(
                train_x, fit_y, problem_type, job.budget_s, cpus
            )
        else:
            model = flaml_fit(train_x, fit_y, problem_type, job.budget_s, cpus)
        if problem_type == "regression":
            prediction = model.predict(test_x)
        else:
            classes = list(model.classes_)
            prediction = model.predict_proba(test_x)
    seconds = time.monotonic() - started
    if path:
        shutil.rmtree(path, ignore_errors=True)
    score = score_predictions(problem_type, test_y, prediction, classes)
    return Outcome("ok", score, seconds, "", versions_of(job.system))


def run_worker(args: argparse.Namespace) -> int:
    """Worker entry point: always writes a result file, even on failure."""
    check_tasks([int(args.task)], open_holdout=args.open_holdout)
    job = Job(args.system, args.budget, args.task, args.fold, args.repeat)
    started = time.monotonic()
    try:
        outcome = work(job, args.cpus)
    except Exception:  # Every failure must become a record.
        outcome = Outcome(
            "failed",
            seconds=time.monotonic() - started,
            error=traceback.format_exc()[-TAIL_CHARS:],
        )
    pathlib.Path(args.result).write_text(
        json.dumps(dataclasses.asdict(outcome)), encoding="utf-8"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Builds the command-line parser for `run` and `worker`."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="run jobs and append records")
    run.add_argument("--python", required=True, help="baseline interpreter")
    run.add_argument("--tasks", nargs="+", type=int, required=True)
    run.add_argument("--systems", nargs="+", choices=SYSTEMS, default=SYSTEMS)
    run.add_argument("--budgets", nargs="+", type=int, default=[60, 600])
    run.add_argument("--lite", action="store_true", help="first split only")
    run.add_argument("--output", required=True, help="JSON-lines results")
    run.add_argument("--cpus", type=int, default=DEFAULT_CPUS)
    run.add_argument("--open-holdout", action="store_true")

    work_parser = commands.add_parser("worker", help="run one job")
    work_parser.add_argument("--system", required=True, choices=SYSTEMS)
    work_parser.add_argument("--budget", required=True, type=int)
    work_parser.add_argument("--task", required=True, type=int)
    work_parser.add_argument("--fold", required=True, type=int)
    work_parser.add_argument("--repeat", required=True, type=int)
    work_parser.add_argument("--cpus", required=True, type=int)
    work_parser.add_argument("--result", required=True)
    work_parser.add_argument("--open-holdout", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = build_parser().parse_args(argv)
    try:
        if args.command == "worker":
            return run_worker(args)
        return run_all(args)
    except HoldoutRefusedError as error:
        LOGGER.error("%s", error)
        return 2


if __name__ == "__main__":
    sys.exit(main())
