"""Claim C4: at least a competent practitioner's quality, on real tables.

Pre-registered before any score was computed (this docstring is the
registration; changing it changes `digest` and invalidates the stored JSON).

Tasks. Per problem type, the tasks of `TASKS`, chosen by a rule fixed
before any of the added tasks was run (scmle's results on the first five
of each type were known; they are kept, never dropped):

1. Candidates, in order: the TabArena-v0.1 tasks of the development set
   of `benchmarks/sota/split.json` (the hold-out stays sealed), then the
   tasks of published OpenML suites, OpenML-CC18 (study 99) and the
   AutoML benchmark's classification suite (study 271) for binary and
   multiclass, OpenML-CTR23 (study 353) and the AutoML benchmark's
   regression suite (study 269) for regression; within each source by
   fewest rows, ties by task id. A suite task is binary with two classes
   and multiclass with more, by OpenML's `NumberOfClasses`.
2. A candidate is skipped when its table is already taken (by OpenML data
   id), is a copy of a TabArena-v0.1 table under another id (development
   or sealed hold-out: same data under another name, such as `dna` for
   `splice` or `red_wine` for `wine_quality`), is a copy listed in
   `benchmarks.evidence.refusal.COPIES`, or states no licence on OpenML.
3. Candidates are taken in order until custody
   (`benchmarks.evidence.custody`) gives each of validation and acceptance
   `minimum()` tasks of the type. Twice `minimum()` is the least that can
   do it; a table that an earlier evidence set already placed keeps that
   split, so a type can need more. This gives 10 binary, 14 multiclass and
   11 regression tasks.

Every task's Parquet file is pinned by SHA-256 with the licence OpenML
states (`PINS`); `fetch` refuses a file that differs. A TabArena table that
another evidence set uses under its original OpenML id is keyed by that id
in custody (`SAME`), so both stay in one split.

Each task is scored on its own OpenML split, repeat 0 fold 0: the system
sees the training rows only and is scored on the test rows with
TabArena's metric (binary ROC AUC, multiclass log loss, regression RMSE).
The rule's unit is the task: the folds and repeats of one task share its
rows, so they are not independent trials and add none; the rule needs one
split per task. The files are cached under `data/quality/` (see its README
for source and licence).

Run-to-run noise. The product is run `runs()` times on each task, run 0
with the registered seed and the rest with seeds of their own (`seed`);
only run 0 decides. `runs()` = `ceil(1 + log2(1 / alpha))` = 6 is the
fewest runs whose range covers the median run's gap with chance at least
`1 - alpha` (all runs fall on one side of it with chance `2 * 0.5 ** runs`),
so a task whose decision gap lies outside that range is visibly noisy.
Each task reports the gaps of every run and their spread; each type, the
median spread and its median gap under each run.

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
and a seed derived from the task and run (`seed`).

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
The run-to-run noise is reported, not decided.

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
from scmle.assurance.detectors import certificates
from scmle.contracts import errors

PATH = Path(__file__).with_suffix(".json")
DATA = Path(__file__).parent / "data" / "quality"
MEDIAN = "benchmark_median_gap_max"
UPPER = "benchmark_gap_upper_max"
SPLIT = Path(__file__).resolve().parents[1] / "sota" / "split.json"
# In the order of the selection rule (module docstring).
TASKS = {
    "binary": (
        363621,
        363696,
        363682,
        363684,
        363674,
        363700,
        363620,
        363623,
        363694,
        363706,
    ),
    "multiclass": (
        363685,
        363711,
        363702,
        363704,
        363699,
        11,
        2079,
        3560,
        3549,
        53,
        3022,
        9981,
        146800,
        23,
    ),
    "regression": (
        363698,
        363625,
        363612,
        363708,
        363686,
        363678,
        363705,
        363672,
        361618,
        361619,
        361622,
    ),
}
# The OpenML data id of each task's table.
TABLES = {
    363621: 46913,
    363696: 46952,
    363682: 46938,
    363684: 46940,
    363674: 46930,
    363700: 46956,
    363620: 46912,
    363623: 46915,
    363694: 46950,
    363706: 46962,
    363685: 46941,
    363711: 46980,
    363702: 46958,
    363704: 46960,
    363699: 46955,
    11: 11,
    2079: 188,
    3560: 469,
    3549: 458,
    53: 54,
    3022: 307,
    9981: 1468,
    146800: 40966,
    23: 23,
    363698: 46954,
    363625: 46917,
    363612: 46904,
    363708: 46964,
    363686: 46942,
    363678: 46934,
    363705: 46961,
    363672: 46928,
    361618: 44962,
    361619: 44967,
    361622: 44994,
}
# TabArena tables that another evidence set uses under their original
# OpenML id, by name and size: custody keys both by the original id.
SAME = {
    46913: 1464,  # blood-transfusion-service-center
    46952: 1494,  # qsar-biodeg
    46912: 4134,  # Bioresponse
    46915: 40701,  # churn
    46958: 46,  # splice
    46964: 287,  # wine_quality
    46934: 537,  # houses
    46961: 44964,  # superconductivity
}
# SHA-256 of each task's Parquet file as OpenML serves it, and the licence
# OpenML states for it.
PINS: dict[int, tuple[str, str]] = {
    363621: (
        "8e9fecb0a04af9e4c8d3c625c8cd1c173a15de655254ddaed480d77a9008d00a",
        "CC BY 4.0",
    ),
    363696: (
        "864c269749defdf2e50e52fa631fa89d4ad8360f3bf82bcd47e2ceaf0f48773d",
        "CC BY 4.0",
    ),
    363682: (
        "3aaa17c39e531131573284fcdcf511c1a763d4593ff9d8b87f61536af912ddcf",
        "CC0: Public Domain",
    ),
    363684: (
        "e468d6379be297953dac8780594ccdd98af606a6c5bd7ee43408365a27a360a6",
        "Public",
    ),
    363674: (
        "e77e8942836e009170196386614de7e378cde967787bef2445e241aaaa98791e",
        "CC BY-SA",
    ),
    363700: (
        "907dfa3a1301742a86bd6d3e2901814d8f50489567530ce99429a4002c426a4b",
        "CC BY 4.0",
    ),
    363620: (
        "44ec2028029fc3a234e63483dbcd69455c70c353a7a6c1df8952fff823dc096e",
        "Public Domain",
    ),
    363623: (
        "22960d2796ac7ec4fa7f2ba775095746ba82d42b3e4e798d3a4adb5b0a1907d7",
        "MIT License",
    ),
    363694: (
        "afe5fcf497e005cd4da0e6c36ce87e899f8b3080abe53b0e52f6e6d92f3d24e1",
        "CC BY 4.0",
    ),
    363706: (
        "0c39553394605c0807d94797ec59b9ac7cb1ab88be24b31bfe2f284a752ba55d",
        "CC BY 4.0",
    ),
    363685: (
        "d9697e031fb0e3db02f3f62445ebbfdb15baff56c8cc2bc75520a4deaba83bc4",
        "CC BY 4.0",
    ),
    363711: (
        "c11fbe1287697050d5bfb4b99bec749cb6ec59518feb86af573dedd55515cb30",
        "CC BY 4.0",
    ),
    363702: (
        "7a6c5fae358d9e0eb0acc1defab3d9eb395e87da294396e40802caf4515c3dd4",
        "CC BY 4.0",
    ),
    363704: (
        "37fb1767ab33ed77c316eae266e9f868f90d923af0ce68a4137e5e80b71c14ec",
        "CC BY 4.0",
    ),
    363699: (
        "668a363a27266198aa6c0da81e338e52aa91d1ca0f06a3bfc487f041f392f042",
        "Public Domain",
    ),
    11: (
        "ff6000f4086ffc8540a891c2eecd5692080c85226e30f5aa6e54ed6d7905ea2c",
        "Public",
    ),
    2079: (
        "40d70a65201f73b1000552323048c7a80816693a4f8cdde66c2ce3971d80a8b8",
        "Public",
    ),
    3560: (
        "e0202b33dad1fb0463b77327f08332afe35596a108918488a945c91ae54be70e",
        "Public",
    ),
    3549: (
        "253fe3ef7808becbafeecd5d8279f44d73bd6d1d037ff76e65c0ed30e0a52006",
        "Public",
    ),
    53: (
        "240c7d30b152b614a514e07303e86d502486230e69aef60efcf52d4fa9e4ed65",
        "Public",
    ),
    3022: (
        "4bc383f7d2cdcb9f53ee53e88783e63d9e7b83b65a5c34d32276354065f85ebe",
        "Public",
    ),
    9981: (
        "59db7443e1e54cab22384fd3ba99eb8d70996737c77d4eb2008c84811dd50854",
        "Public",
    ),
    146800: (
        "7ced21880af8755dcbb7a11c8da181361b2948426151241a435a05b6f0cb07a9",
        "Public",
    ),
    23: (
        "e27aa2f656cddac85adabdc4d2d4398656b9d499b4efce447640b15549cd5640",
        "Public",
    ),
    363698: (
        "ccf8655ae2a13062a436cd7e95c7594710576d7655ef148c320da6266ef5626c",
        "CC BY 4.0",
    ),
    363625: (
        "c5853f3c8e7e10524d81878cd2b0d61d8129242467a814187f4d015b4895df21",
        "CC BY 4.0",
    ),
    363612: (
        "d98c8cf8b64d81cbd43aabd76222f4b54ae02f4cb2b67d4fb1c4ee9f309509d8",
        "CC BY 4.0",
    ),
    363708: (
        "d34509cc5ecaca4dfe35ea0eb72211f20aed8b88b6f727ab4a5d4bf4943277ee",
        "CC BY 4.0",
    ),
    363686: (
        "eb38dc11f87c375ca4a66222ce03cb7cd1d0e38084d9a91d4635f09900dfb898",
        "CC BY-NC-SA",
    ),
    363678: (
        "450e15e660005e5f286d69a1651e99aab276420bc8557c82bad86a3d1d084090",
        "Public",
    ),
    363705: (
        "5dd5598eb8039c9ad2c485202769a7840b7f06babcbfc700aa2e0d4712d12426",
        "CC BY 4.0",
    ),
    363672: (
        "6e0c67b2857816548bee0cb159893443ad1a575a5af133cd1505696e75b12e54",
        "Database Contents License (DbCL) v1.0",
    ),
    361618: (
        "ab189460ac8f4f45e6d77cb1a45e8955efa46b0168a50507d5f7eea3708d39aa",
        "CC BY 4.0",
    ),
    361619: (
        "392bad0e6113e69c7eb29e8e46b318684be42203ba4068e6be3e534a598754f5",
        "CC BY 4.0",
    ),
    361622: (
        "eb6f68449c8722a4d4848cbbb9fdddd34fad4211483c182906d8dcdff32b8c28",
        "Public",
    ),
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
    verify(task, parquet, tag(described, "licence"))
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
        "ignore": re.findall(
            r"<oml:(?:ignore|row_id)_attribute>([^<]*)", described
        ),
        "source": f"https://www.openml.org/t/{task}",
        "sha256": hashlib.sha256(parquet).hexdigest(),
        "test": test,
    }
    (DATA / f"{task}.json").write_text(json.dumps(meta) + "\n", "utf-8")


def verify(task: int, parquet: bytes, licence: str) -> None:
    """Refuses a task file other than the pinned one.

    Args:
        task: OpenML task id.
        parquet: The Parquet file as served.
        licence: The licence OpenML states.

    Raises:
        InputError: If the task has a pin and the file's SHA-256 or the
            licence differs from it.
    """
    if task not in PINS:
        return
    digest, stated = PINS[task]
    found = hashlib.sha256(parquet).hexdigest()
    if found != digest or licence != stated:
        raise errors.InputError(
            f"task {task}: file {found[:12]} under {licence!r} is not the "
            f"pinned {digest[:12]} under {stated!r}"
        )


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


def runs() -> int:
    """Product runs per task, for the run-to-run noise.

    Returns:
        `ceil(1 + log2(1 / alpha))`: the fewest runs whose range covers the
        median run with chance at least `1 - alpha`.
    """
    alpha = acceptance.load()["alpha"].value
    return math.ceil(1 + math.log2(1 / alpha))


def seed(task: int, run: int) -> int:
    """The product's seed for one run on one task.

    Args:
        task: OpenML task id.
        run: The run, from 0.

    Returns:
        The registered `scmle_runner.seed_for(task, 0, 0)` for run 0; a
        31-bit seed from SHA-256 of the task and run otherwise.
    """
    if run == 0:
        return scmle_runner.seed_for(task, 0, 0)
    key = f"scmle-quality-noise:{task}:{run}".encode()
    return int(hashlib.sha256(key).hexdigest(), 16) % 2**31


def product(
    task: int,
    train: pandas.DataFrame,
    test: pandas.DataFrame,
    target: str,
    run: int = 0,
) -> dict[str, Any]:
    """The product's default run on one task; a crash is recorded.

    Args:
        task: OpenML task id.
        train: Training rows with the target.
        test: Test rows with the target.
        target: Target column.
        run: The run, which picks the seed (`seed`).

    Returns:
        Score (None on failure), seconds, the seed and the error text.
    """
    name = kind(task)
    started = time.monotonic()
    found: dict[str, Any] = {
        "score": None,
        "error": "",
        "seed": seed(task, run),
    }
    with tempfile.TemporaryDirectory() as folder:
        try:
            result = scmle.api.project.Project(
                train,
                target,
                task="regression" if name == "regression" else "classification",
                metric=scmle_runner.METRICS[name],
                time_budget=BUDGET,
                seed=found["seed"],
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
        The task's raw scores, times and gap (run 0's), the further runs
        and the spread of every run's gap.
    """
    train, test, target = load(task)
    baseline = practitioner(task, train, test, target)
    name = kind(task)
    anchor = trivial(task, train, test, target)
    done = [product(task, train, test, target, r) for r in range(runs())]
    for each in done:
        each["gap"] = gap(name, baseline["score"], each["score"], anchor)
    ours, repeats = done[0], done[1:]
    gaps = [each["gap"] for each in done]
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
        "gap": ours["gap"],
        "repeats": repeats,
        "spread": max(gaps) - min(gaps),
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
        The gaps, their median and upper bound, failures, whether the type
        passes, and the noise: the median spread of a task's runs and the
        median gap under each run (reported, not decided).
    """
    variables = acceptance.load()
    gaps = [r["gap"] for r in rows]
    repeated = [r for r in rows if r.get("repeats")]
    width = min((len(r["repeats"]) for r in repeated), default=0)
    medians = [float(numpy.median(gaps))] if gaps else []
    medians += [
        float(numpy.median([r["repeats"][i]["gap"] for r in repeated]))
        for i in range(width)
    ]
    spreads = [r["spread"] for r in repeated]
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
        "noise": {
            "spread": float(numpy.median(spreads)) if spreads else None,
            "medians": medians,
        },
    }


def digest() -> str:
    """Hash of the measured entry point and of this module.

    Returns:
        The `code_hash` of `Project`, the runner helpers and this module.
    """
    return certificates.fingerprint(
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
            "runs": runs(),
            "noise": "range of runs() product runs per task; run 0 decides",
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
