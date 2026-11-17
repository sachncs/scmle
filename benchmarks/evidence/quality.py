"""Claim C4: at least a competent practitioner's quality, on real tables.

Pre-registered before any score was computed (this docstring is the
registration; changing it changes `digest` and invalidates the stored JSON).

Tasks. TabArena-v0.1 OpenML tasks of the development set of
`benchmarks/sota/split.json` (the hold-out stays sealed): the five with the
fewest rows of each problem type, listed in `TASKS`. Each task is scored on
its own OpenML split, repeat 0 fold 0: the system sees the training rows
only and is scored on the test rows with TabArena's metric (binary ROC AUC,
multiclass log loss, regression RMSE). The files are cached under
`data/quality/` (see its README for source and licence).

Competent-practitioner baseline (`practitioner`), fixed in advance:
scikit-learn `HistGradientBoosting{Classifier,Regressor}` with native
categorical and missing-value handling, inner early stopping (10 percent
validation, 20 rounds patience, at most 500 iterations), and a grid of six
configurations (`GRID`: learning rate 0.1 or 0.05 by 15, 31 or 63 leaves),
the scikit-learn default first. Each configuration is scored by 5-fold
cross-validation (stratified for classes) on the training rows with the
task's metric; the best mean is refit on all training rows. It obeys the
same wall-clock budget as the product: no new configuration starts once the
last one would end past `CUTOFF` of the budget, and the first always runs.

Product (`product`): `scmle.api.project.Project` with its defaults except
the task, the task's metric, `time_budget=BUDGET` (the default is 1800 s)
and a seed derived from the task (`scmle_runner.seed_for`).

Normalised gap per task (`gap`): with `o` the score oriented so higher is
better, the anchor `t` a trivial predictor (AUC 0.5, the training class
frequencies, the training mean) and `b` the better of the two systems,
`norm(s) = (max(o(s), t) - t) / (b - t)` and the gap is
`norm(practitioner) - norm(product)`. Min-max over two systems would only
ever give 0 or 1, so the worst end is the trivial predictor; a failure
scores the trivial value (the worst), as the plan requires for every
system. A negative gap means the product was better.

Decision per problem type: the median gap at most `benchmark_median_gap_max`
and the one-sided `alpha` percentile-bootstrap upper bound of the median
(`bootstrap_resamples` task resamples) at most `benchmark_gap_upper_max`.
At least `minimum()` = ceil(ln alpha / ln 0.5) tasks are needed: below it
not even the largest gap is a distribution-free 1 - alpha upper bound of
the median. The row passes only when every problem type passes AND every
other data kind is measured; those are not, so the best verdict here is
"not measured", and any failing problem type makes it "fail".

NOT measured: data kinds other than tabular, the hold-out tasks, more than
one split per task, the default 1800 s budget, AutoGluon-class baselines.

    python -m benchmarks.evidence.quality --write
    python -m benchmarks.evidence.quality --check
"""

import argparse
from concurrent import futures
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import time
import traceback
from typing import Any
from urllib import request

import numpy
import pandas
from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection

from benchmarks import acceptance
from benchmarks.sota import scmle_runner
import scmle
import scmle.api.project
from scmle.contracts import errors
from scmle.data.detectors import certificates

PATH = Path(__file__).with_suffix(".json")
DATA = Path(__file__).parent / "data" / "quality"
MEDIAN = "benchmark_median_gap_max"
UPPER = "benchmark_gap_upper_max"
TASKS = {
    "binary": (363621, 363696, 363682, 363684, 363674),
    "multiclass": (363685, 363711, 363702, 363704, 363699),
    "regression": (363698, 363625, 363612, 363708, 363686),
}
BUDGET = 60
CUTOFF = 0.8
FOLDS = 5
SEED = 0
GRID = (
    {"learning_rate": 0.1, "max_leaf_nodes": 31},
    {"learning_rate": 0.1, "max_leaf_nodes": 15},
    {"learning_rate": 0.1, "max_leaf_nodes": 63},
    {"learning_rate": 0.05, "max_leaf_nodes": 15},
    {"learning_rate": 0.05, "max_leaf_nodes": 31},
    {"learning_rate": 0.05, "max_leaf_nodes": 63},
)
FIXED = {
    "max_iter": 500,
    "early_stopping": True,
    "validation_fraction": 0.1,
    "n_iter_no_change": 20,
    "categorical_features": "from_dtype",
    "random_state": SEED,
}
SCORERS = {
    "binary": "roc_auc",
    "multiclass": "neg_log_loss",
    "regression": "neg_root_mean_squared_error",
}
# HistGradientBoosting takes categories only up to its 255 bins.
LEVELS = 255
WORKERS = 2
OPENML = "https://www.openml.org/api/v1/xml"
# A split line: type, row id, repeat, fold.
FIELDS = 4
NOT_MEASURED = (
    (
        "data kinds other than tabular (text, document, image, audio, "
        "video, graph, geospatial, raster, related tables)"
    ),
    "the sealed hold-out tasks",
    "more than one split per task",
    "the default 1800 s budget",
    "AutoGluon-class reference systems",
)


def fetch(task: int) -> None:
    """Downloads one OpenML task into `DATA` unless it is there.

    Writes `<task>.pq` (the dataset's published Parquet file) and
    `<task>.json` (source, licence, target, the test row ids of repeat 0
    fold 0 and the Parquet file's SHA-256).

    Args:
        task: OpenML task id.
    """
    if (DATA / f"{task}.json").exists():
        return
    xml = text(f"{OPENML}/task/{task}")
    dataset = int(tag(xml, "data_set_id"))
    described = text(f"{OPENML}/data/{dataset}")
    parquet = request.urlopen(tag(described, "parquet_url")).read()
    splits = text(tag(xml, "data_splits_url")).splitlines()
    test = sorted(
        int(parts[1])
        for parts in (line.split(",") for line in splits)
        if len(parts) == FIELDS
        and parts[0] == "TEST"
        and parts[2:] == ["0", "0"]
    )
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / f"{task}.pq").write_bytes(parquet)
    meta = {
        "task": task,
        "dataset": dataset,
        "name": tag(described, "name"),
        "version": tag(described, "version"),
        "licence": tag(described, "licence"),
        "target": tag(xml, "target_feature"),
        "ignore": re.findall(r"<oml:ignore_attribute>([^<]*)", described),
        "source": f"https://www.openml.org/t/{task}",
        "sha256": hashlib.sha256(parquet).hexdigest(),
        "test": test,
    }
    (DATA / f"{task}.json").write_text(json.dumps(meta) + "\n", "utf-8")


def text(url: str) -> str:
    """Reads a URL as text.

    Args:
        url: An https URL.

    Returns:
        The decoded body.
    """
    with request.urlopen(url, timeout=120) as response:
        body: bytes = response.read()
    return body.decode("utf-8")


def tag(xml: str, name: str) -> str:
    """The first `<oml:name>` value of an OpenML XML reply.

    Args:
        xml: The reply.
        name: The tag without its prefix.

    Returns:
        The stripped value.

    Raises:
        InputError: If the tag is absent.
    """
    found = re.search(rf"<oml:{name}>([^<]*)<", xml)
    if found is None:
        raise errors.InputError(f"OpenML reply has no {name}")
    return found.group(1).strip()


def kind(task: int) -> str:
    """The problem type a task was registered under.

    Args:
        task: OpenML task id.

    Returns:
        `binary`, `multiclass` or `regression`.

    Raises:
        InputError: For a task not in `TASKS`.
    """
    for name, ids in TASKS.items():
        if task in ids:
            return name
    raise errors.InputError(f"task {task} is not registered")


def load(task: int) -> tuple[pandas.DataFrame, pandas.DataFrame, str]:
    """The training and test rows of a cached task.

    Args:
        task: OpenML task id.

    Returns:
        Training frame, test frame (both with the target) and the target;
        a class target is text.
    """
    meta = json.loads((DATA / f"{task}.json").read_text("utf-8"))
    frame = pandas.read_parquet(DATA / f"{task}.pq")
    frame = frame.drop(columns=[c for c in meta["ignore"] if c in frame])
    target = meta["target"]
    if kind(task) == "regression":
        frame[target] = frame[target].astype(float)
    else:
        frame[target] = frame[target].astype(str)
    held = numpy.zeros(len(frame), dtype=bool)
    held[meta["test"]] = True
    train = frame[~held].reset_index(drop=True)
    return train, frame[held].reset_index(drop=True), target


def encode(
    frame: pandas.DataFrame, reference: pandas.DataFrame
) -> pandas.DataFrame:
    """Gives text columns the training rows' categories, as a practitioner.

    Args:
        frame: Features to encode.
        reference: Training features, which fix the categories.

    Returns:
        A copy: text, category and boolean columns become categories (codes
        as numbers above `LEVELS` levels); levels unseen in training become
        missing.
    """
    out = frame.copy()
    for column in out.columns:
        if pandas.api.types.is_numeric_dtype(reference[column]) and not (
            pandas.api.types.is_bool_dtype(reference[column])
        ):
            continue
        levels = sorted(reference[column].dropna().astype(str).unique())
        values = out[column].astype(str).where(out[column].notna())
        coded = pandas.Categorical(values, categories=levels)
        if len(levels) <= LEVELS:
            out[column] = coded
        else:
            numbers = coded.codes.astype(float)
            numbers[coded.codes < 0] = numpy.nan
            out[column] = numbers
    return out


def evaluate(
    task: int, model: Any, rows: pandas.DataFrame, truth: pandas.Series
) -> float:
    """Scores a fitted scikit-learn model with the task's metric.

    Args:
        task: OpenML task id.
        model: A fitted HistGradientBoosting model.
        rows: Encoded test features.
        truth: Test target.

    Returns:
        ROC AUC, log loss or RMSE.
    """
    name = kind(task)
    if name == "regression":
        return float(
            metrics.root_mean_squared_error(truth, model.predict(rows))
        )
    proba = model.predict_proba(rows)
    if name == "binary":
        positive = truth == model.classes_[1]
        return float(metrics.roc_auc_score(positive, proba[:, 1]))
    return float(metrics.log_loss(truth, proba, labels=model.classes_))


def practitioner(
    task: int, train: pandas.DataFrame, test: pandas.DataFrame, target: str
) -> dict[str, Any]:
    """The pre-registered competent-practitioner baseline on one task.

    Args:
        task: OpenML task id.
        train: Training rows with the target.
        test: Test rows with the target.
        target: Target column.

    Returns:
        Score, seconds, configurations tried and the one chosen.
    """
    started = time.monotonic()
    name = kind(task)
    features = train.drop(columns=[target])
    rows = encode(features, features)
    model = (
        ensemble.HistGradientBoostingRegressor
        if name == "regression"
        else ensemble.HistGradientBoostingClassifier
    )
    splitter = (
        model_selection.KFold(FOLDS, shuffle=True, random_state=SEED)
        if name == "regression"
        else model_selection.StratifiedKFold(
            FOLDS, shuffle=True, random_state=SEED
        )
    )
    best, chosen, tried, last = -math.inf, GRID[0], 0, 0.0
    for params in GRID:
        elapsed = time.monotonic() - started
        if tried and elapsed + last > CUTOFF * BUDGET:
            break
        begun = time.monotonic()
        try:
            value = float(
                numpy.mean(
                    model_selection.cross_val_score(
                        model(**FIXED, **params),
                        rows,
                        train[target],
                        cv=splitter,
                        scoring=SCORERS[name],
                        error_score="raise",
                    )
                )
            )
        except ValueError:
            value = -math.inf
        last, tried = time.monotonic() - begun, tried + 1
        if value > best:
            best, chosen = value, params
    fitted = model(**FIXED, **chosen).fit(rows, train[target])
    score = evaluate(
        task,
        fitted,
        encode(test.drop(columns=[target]), features),
        test[target],
    )
    return {
        "score": score,
        "seconds": round(time.monotonic() - started, 1),
        "tried": tried,
        "chosen": chosen,
        "cv": best,
    }


def product(
    task: int, train: pandas.DataFrame, test: pandas.DataFrame, target: str
) -> dict[str, Any]:
    """The product's default run on one task; a crash is recorded.

    Args:
        task: OpenML task id.
        train: Training rows with the target.
        test: Test rows with the target.
        target: Target column.

    Returns:
        Score (None on failure), seconds and the error text.
    """
    name = kind(task)
    started = time.monotonic()
    found: dict[str, Any] = {"score": None, "error": ""}
    with tempfile.TemporaryDirectory() as folder:
        try:
            result = scmle.api.project.Project(
                train,
                target,
                task="regression" if name == "regression" else "classification",
                metric=scmle_runner.METRICS[name],
                time_budget=BUDGET,
                seed=scmle_runner.seed_for(task, 0, 0),
                output_dir=Path(folder) / "run",
            ).run()
            found["score"] = scmle_runner.score(result, test, target, name)
        except Exception:  # noqa: BLE001 -- a failure scores the worst
            found["error"] = traceback.format_exc(limit=4)
    found["seconds"] = round(time.monotonic() - started, 1)
    return found


def trivial(
    task: int, train: pandas.DataFrame, test: pandas.DataFrame, target: str
) -> float:
    """The score of a predictor that ignores the features.

    Args:
        task: OpenML task id.
        train: Training rows with the target.
        test: Test rows with the target.
        target: Target column.

    Returns:
        AUC 0.5; log loss of the training class frequencies; RMSE of the
        training mean.
    """
    name = kind(task)
    if name == "binary":
        return 0.5
    if name == "regression":
        mean = numpy.full(len(test), train[target].mean())
        return float(metrics.root_mean_squared_error(test[target], mean))
    shares = train[target].value_counts(normalize=True).sort_index()
    proba = numpy.tile(shares.to_numpy(), (len(test), 1))
    return float(metrics.log_loss(test[target], proba, labels=shares.index))


def gap(
    name: str, baseline: float | None, ours: float | None, anchor: float
) -> float:
    """The normalised gap, practitioner minus product (module docstring).

    Args:
        name: Problem type; binary scores are higher-is-better.
        baseline: Practitioner score, None on failure.
        ours: Product score, None on failure.
        anchor: Trivial predictor's score.

    Returns:
        The gap; 0 when neither system beats the trivial predictor.

    Raises:
        InputError: For an unknown problem type.
    """
    if name not in TASKS:
        raise errors.InputError(f"unknown problem type {name!r}")
    sign = 1.0 if name == "binary" else -1.0
    floor = sign * anchor
    values = [
        floor if v is None or not math.isfinite(v) else max(sign * v, floor)
        for v in (baseline, ours)
    ]
    best = max(values)
    if best <= floor:
        return 0.0
    return (values[0] - values[1]) / (best - floor)


def run(task: int) -> dict[str, Any]:
    """Both systems and the trivial predictor on one task.

    Args:
        task: OpenML task id.

    Returns:
        The task's raw scores, times and gap.
    """
    train, test, target = load(task)
    baseline = practitioner(task, train, test, target)
    ours = product(task, train, test, target)
    anchor = trivial(task, train, test, target)
    meta = json.loads((DATA / f"{task}.json").read_text("utf-8"))
    return {
        "task": task,
        "name": meta["name"],
        "kind": kind(task),
        "metric": scmle_runner.METRICS[kind(task)],
        "train_rows": len(train),
        "test_rows": len(test),
        "trivial": anchor,
        "practitioner": baseline,
        "product": ours,
        "gap": gap(kind(task), baseline["score"], ours["score"], anchor),
    }


def minimum() -> int:
    """Tasks a problem type needs before it can be decided.

    Returns:
        ceil(ln alpha / ln 0.5): with fewer tasks even the largest gap is
        not a distribution-free one-sided 1 - alpha bound of the median.
    """
    alpha = acceptance.load()["alpha"].value
    return math.ceil(math.log(alpha) / math.log(0.5))


def upper(gaps: list[float]) -> float:
    """One-sided percentile-bootstrap upper bound of the median gap.

    Args:
        gaps: One gap per task (paired: both systems on the same split).

    Returns:
        The 1 - alpha quantile of `bootstrap_resamples` resampled medians.
    """
    variables = acceptance.load()
    alpha = variables["alpha"].value
    resamples = int(variables["bootstrap_resamples"].value)
    values = numpy.asarray(gaps, dtype=float)
    picks = numpy.random.default_rng(SEED).integers(
        0, len(values), (resamples, len(values))
    )
    return float(numpy.quantile(numpy.median(values[picks], axis=1), 1 - alpha))


def tally(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Decides one problem type.

    Args:
        name: Problem type.
        rows: What `run` returned for its tasks.

    Returns:
        The gaps, their median and upper bound, failures and whether the
        type passes.
    """
    variables = acceptance.load()
    gaps = [r["gap"] for r in rows]
    median = float(numpy.median(gaps)) if gaps else None
    bound = upper(gaps) if gaps else None
    passed = (
        len(gaps) >= minimum()
        and median is not None
        and bound is not None
        and median <= variables[MEDIAN].value
        and bound <= variables[UPPER].value
    )
    return {
        "kind": name,
        "tasks": len(gaps),
        "gaps": gaps,
        "median": median,
        "upper": bound,
        "failures": sum(r["product"]["score"] is None for r in rows),
        "passed": passed,
    }


def digest() -> str:
    """Hash of the measured entry point and of this module.

    Returns:
        The `code_hash` of `Project`, the runner helpers and this module.
    """
    return certificates.code_hash(
        scmle.api.project.Project,
        scmle_runner.score,
        scmle_runner.seed_for,
        sys.modules[__name__],
    )


def files() -> dict[str, str]:
    """SHA-256 of every cached data file.

    Returns:
        Task id (text) to the hash of its Parquet file, for cached tasks.
    """
    found = {}
    for ids in TASKS.values():
        for task in ids:
            path = DATA / f"{task}.pq"
            if path.exists():
                found[str(task)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return found


def measure(workers: int = WORKERS) -> dict[str, Any]:
    """Fetches the tasks, runs both systems and builds the document.

    Args:
        workers: Concurrent tasks.

    Returns:
        The JSON-ready document.
    """
    started = time.monotonic()
    os.environ["OMP_NUM_THREADS"] = "1"
    order = [t for ids in TASKS.values() for t in ids]
    for task in order:
        fetch(task)
    with futures.ProcessPoolExecutor(workers) as pool:
        done = list(pool.map(run, order))
    variables = acceptance.load()
    return {
        "claim": "C4",
        "median_max": variables[MEDIAN].value,
        "upper_max": variables[UPPER].value,
        "alpha": variables["alpha"].value,
        "minimum_tasks": minimum(),
        "design": {
            "budget_seconds": BUDGET,
            "split": "OpenML repeat 0 fold 0",
            "grid": list(GRID),
            "fixed": FIXED,
            "folds": FOLDS,
            "scmle": scmle.__version__,
        },
        "kinds": [tally(n, [r for r in done if r["kind"] == n]) for n in TASKS],
        "tasks": done,
        "data": files(),
        "not_measured": list(NOT_MEASURED),
        "code": digest(),
        "seconds": round(time.monotonic() - started, 1),
    }


def read(document: dict[str, Any] | None) -> dict[str, Any] | None:
    """The given document, or the stored one.

    Args:
        document: A document, or None to read `PATH`.

    Returns:
        The document, or None when nothing is stored.
    """
    if document is not None:
        return document
    if not PATH.exists():
        return None
    loaded: dict[str, Any] = json.loads(PATH.read_text(encoding="utf-8"))
    return loaded


def problems(document: dict[str, Any] | None = None) -> list[str]:
    """Why the document cannot decide the row.

    Args:
        document: The document; read from `PATH` when None.

    Returns:
        Messages for a missing file, changed code, thresholds other than
        acceptance.toml's, changed or missing data, or a problem type with
        fewer tasks than `minimum()`; empty when the document decides.
    """
    found = read(document)
    if found is None:
        return [f"{PATH.name} is missing"]
    issues = []
    if found.get("code") != digest():
        issues.append("quality: measured with other code")
    variables = acceptance.load()
    for key, name in (("median_max", MEDIAN), ("upper_max", UPPER)):
        if found.get(key) != variables[name].value:
            issues.append(f"quality: {key} is not acceptance.toml's {name}")
    if found.get("data") != files():
        issues.append("quality: cached data missing or changed")
    kinds = {k.get("kind"): k for k in found.get("kinds", [])}
    for name in TASKS:
        count = kinds.get(name, {}).get("tasks", 0)
        if count < minimum():
            issues.append(
                f"quality: {name} has {count} tasks; none can pass below "
                f"{minimum()}"
            )
    return issues


def summary(document: dict[str, Any] | None = None) -> tuple[str, str]:
    """The scorecard's value and verdict for the row.

    Args:
        document: The document; read from `PATH` when None.

    Returns:
        The per-type median and upper bound, and `fail` when any measured
        type fails; otherwise `not measured`, since data kinds other than
        tabular are not measured (and always on any `problems`).
    """
    found = read(document)
    if found is None or not found.get("kinds"):
        return "no measurement", "not measured"
    parts = [
        f"{k['kind']} median {k['median']:.3f} upper {k['upper']:.3f} "
        f"({k['tasks']} tasks)"
        for k in found["kinds"]
        if k.get("median") is not None
    ]
    text = (
        "tabular, real OpenML development tasks, one split, "
        f"{found.get('design', {}).get('budget_seconds')} s: "
        + "; ".join(parts)
        + "; other data kinds not measured"
    )
    if problems(found):
        return text, "not measured"
    if not all(k["passed"] for k in found["kinds"]):
        return text, "fail"
    return text, "not measured"


def main(arguments: list[str] | None = None) -> int:
    """Writes (`--write`) or checks (`--check`) the stored measurement.

    Args:
        arguments: Command-line arguments; None reads `sys.argv`.

    Returns:
        0 after `--write`; for `--check`, 0 when `problems` is empty and 1
        otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    options = parser.parse_args(arguments)
    if options.write:
        document = measure()
        PATH.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary(document)))
        return 0
    issues = problems()
    print(json.dumps({"ok": not issues, "problems": issues}, indent=2))
    return 0 if not issues else 1


if __name__ == "__main__":
    sys.exit(main())
