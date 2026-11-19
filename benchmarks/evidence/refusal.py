"""Claim C14: the product refuses only when it should.

Criterion (plan section 5, `acceptance.toml`): on in-envelope tasks of every
kind the refusal-rate upper bound is at most `refusal_rate_max`, and of the
refusals reviewed the justified share has a lower bound of at least
`refusal_precision_min`. Both are decided by the rules of
`benchmarks.acceptance`, which refuse a pass on fewer trials than the rule's
minimum (59 tasks for the rate, 29 reviewed refusals for the precision).

What is run. Each task is one end-to-end `Project(...).run()` through the
public API with the product's defaults, a short `BUDGET` and only the
declarations its kind needs (the task, the event, treatment or query column,
the ordinal order, the forecast horizon). Each run is a separate process with
the network cut (a dead proxy), so only data already on this machine is
read. A run ends in one of three ways:

* `model`: it returned a run with a selected model;
* `refused`: it raised `BlockedError` or `LeakageError`; the refusing gate
  is the innermost frame of the traceback that `gates.toml` registers;
* `failed`: any other exception (a crash, a bad option, all trials failed).

The tasks are of three groups:

* `real`: public tables already cached on this laptop (scikit-learn's
  bundled sets and the OpenML cache, extended by the `EXTRA` tables listed in
  `data/refusal/README.md`), each with the task a practitioner would
  declare. No real survival, ranking, forecast or uplift table is cached, so
  those kinds are simulated only. Every OpenML table is pinned: `fetch`
  refuses one whose ARFF file, as OpenML serves it, does not have the
  SHA-256 of `PINS`.

Selection rule for the tables added in the second round (fixed before any
of them was run, so the set is not chosen by the product's results): every
dataset of four published OpenML suites, in this order, OpenML-CC18 (study
99) and the AutoML benchmark's classification suite (study 271) declared
`classification`, OpenML-CTR23 (study 353) and the AutoML benchmark's
regression suite (study 269) declared `regression`, each with OpenML's
default target, except a dataset that

1. is already in the catalog (by data id);
2. is a copy of a table already used by an evidence set (refusal, leakage
   or quality) under another id, or another feature view or labelling of
   the same objects: `COPIES` names each with the table it repeats;
3. has more than `CELLS` cells (rows times features), the memory bound
   below;
4. states no licence on OpenML (`colleges`, 42727, says "NA");
5. is a copy of a table of the sealed TabArena-v0.1 hold-out
   (`benchmarks/sota/split.json`), which no development or evidence run may
   read: `SEALED` names each with the hold-out table it repeats (credit-g,
   diabetes, diamonds, Amazon_employee_access, Diabetes130US,
   KDDCup09_appetency, physiochemical_protein and QSAR-TID-11 were in the
   catalog until this rule removed them);
6. is deactivated on OpenML.

Third round (`FURTHER`). With rule 5 applied, custody
(`benchmarks.evidence.custody`) gave validation and acceptance fewer than
the rate's minimum of real tasks, and the four suites hold no further
admissible table. The datasets of OpenML100 (study 14) are then taken in
the study's order, declared `classification` with OpenML's default target,
under rules 1 to 6, until custody gives each of validation and acceptance
the minimum. These tables are not screened by the leakage set (C2), so
custody places them by this set's needs.

The suites hold binary, multiclass and regression tables and the awkward
cases with them: wide (Bioresponse, Internet-Advertisements, arcene),
imbalanced (APSFailure, shuttle), many categories (black_friday,
Click_prediction_small), missing values (sick, okcupid-stem,
house_prices_nominal, soybean) and dates turned into numbers
(nyc-taxi-green-dec-2016, kick). None of them stores a date as a
calendar value, so a date column is not covered by real tables.
* `simulated`: clean generated tasks of every kind, several sizes and shapes
  (plain, mixed types with missing values, wide with noise columns):
  `worlds()` of them, world `i` the layout `LAYOUTS[i % len(LAYOUTS)]`
  with seed `i`, twice the rate's minimum so custody can give each of
  validation and acceptance the minimum.
* `planted`: tasks built to be refused (a constant or identifier target,
  missing labels, a class of one row, too few rows, a planted leak under
  `leakage_policy="block"`, malformed event, treatment or label columns, too
  few queries). Each names the gates that may refuse it.

Rate: the plan's criterion is about real datasets, so only `real` runs of
tasks that `benchmarks.evidence.custody` places in the `validation` split
decide the row (acceptance items are sealed for stage D, development items
decide nothing); every other run, and the simulated group, is reported
beside it. Each such run is a trial, and every trial that did not return a
model is a hit: a refusal, a crash and a run past `TIMEOUT` all count
against the claim, never as a refusal-free trial.
Precision is computed with an automated rubric (a refusal is justified when
its task was planted and its gate is one the task names); the plan asks for
a validator's review against a written rubric on the acceptance split, so it
is reported as automated.

Time. The tasks run in order, `PARALLEL` at a time, until `SECONDS` of wall
clock have passed; tasks not started are listed as not run, and a run past
`TIMEOUT` is killed and listed as unfinished. When fewer trials than the
rule's minimum finish, the verdict is `not measured` with the trials still
needed.

    python -m benchmarks.evidence.refusal --write
    python -m benchmarks.evidence.refusal --check
"""

import argparse
from collections.abc import Callable
import concurrent.futures
import contextlib
import dataclasses
import functools
import gzip
import hashlib
import json
import logging
import os
import pathlib
import subprocess
import sys
import tempfile
import time
import tomllib
from typing import Any
import warnings

import numpy
import pandas
from sklearn import datasets

from benchmarks import acceptance
from scmle.assurance.detectors import certificates
from scmle.contracts import errors

PATH = pathlib.Path(__file__).with_suffix(".json")
ROOT = pathlib.Path(__file__).resolve().parents[2]
GATES = ROOT / "gates.toml"
# Policy: seconds of training budget per run; short, as a quick first look.
BUDGET = 30
# Policy: wall clock for the whole measurement. Every validation task must
# run for the rate to be decided; about 285 tasks (the simulated ones take
# some 10 s, the rest at most the 30 s budget), `PARALLEL` at a time, take
# about 25 minutes on an idle laptop, and twice that allows a loaded one.
SECONDS = 3000
# Policy: a run is killed past this; 12 times the budget allows a loaded
# laptop (one run at a 60 s budget took 514 s with load average 60).
TIMEOUT = 360
PARALLEL = 4
GROUPS = ("real", "simulated", "planted")
# The custody record of `benchmarks.evidence.custody`, read as data because
# that module imports this one.
CUSTODY = PATH.with_name("custody.json")
# The split whose items decide a row before the sealed acceptance run.
DECIDING = "validation"
DEAD = "http://127.0.0.1:9"
# Rows kept of a large real table, so a run fits the short budget.
ROWS = 4000
# Largest table, in rows times features, the selection rule admits. Each of
# the `PARALLEL` runs holds the whole table as float64 before keeping `ROWS`
# rows: 20 million cells are 160 MB, about 640 MB for four runs and some
# 2 GB with pandas' copies, an eighth of a 16 GB laptop.
CELLS = 20_000_000
# Share of missing values in a mixed simulated table.
MISSING = 0.05
# Documents per query of a simulated ranking task.
QUERY = 15


@dataclasses.dataclass(frozen=True)
class Task:
    """One end-to-end run of the product.

    Attributes:
        name: Unique name.
        kind: The task kind, such as `binary` or `survival`.
        group: `real`, `simulated` or `planted`.
        build: Returns the table, the target and the further `Project`
            arguments.
        expected: For a planted task, the gates allowed to refuse it; empty
            for an in-envelope task.
        cause: For a planted task, the defect planted.
    """

    name: str
    kind: str
    group: str
    build: Callable[[], tuple[pandas.DataFrame, Any, dict[str, Any]]]
    expected: tuple[str, ...] = ()
    cause: str = ""


Labels = tuple[Any, dict[str, Any]]


def flip(
    frame: pandas.DataFrame, signal: Any, rng: numpy.random.Generator
) -> Labels:
    """Adds a binary label drawn from a logistic model.

    Args:
        frame: The features; the label column is added in place.
        signal: The linear predictor.
        rng: Random generator.

    Returns:
        The target and the `Project` arguments.
    """
    chance = 1 / (1 + numpy.exp(-signal))
    frame["y"] = (rng.random(len(frame)) < chance).astype(int)
    return "y", {"task": "classification"}


def choose(
    frame: pandas.DataFrame, signal: Any, rng: numpy.random.Generator
) -> Labels:
    """Adds a three-class label from noisy class scores.

    Args:
        frame: The features; the label column is added in place.
        signal: The linear predictor.
        rng: Random generator.

    Returns:
        The target and the `Project` arguments.
    """
    scores = numpy.column_stack([signal, -signal, 0.8 * frame.x4])
    scores = scores + rng.gumbel(size=(len(frame), 3))
    frame["y"] = numpy.array(["red", "green", "blue"])[scores.argmax(1)]
    return "y", {"task": "classification"}


def regress(
    frame: pandas.DataFrame, signal: Any, rng: numpy.random.Generator
) -> Labels:
    """Adds a numeric outcome with normal noise.

    Args:
        frame: The features; the outcome column is added in place.
        signal: The linear predictor.
        rng: Random generator.

    Returns:
        The target and the `Project` arguments.
    """
    frame["y"] = 2 * signal + rng.normal(size=len(frame))
    return "y", {"task": "regression"}


def censor(
    frame: pandas.DataFrame, signal: Any, rng: numpy.random.Generator
) -> Labels:
    """Adds a right-censored exponential time to event.

    Args:
        frame: The features; time and event columns are added in place.
        signal: The log hazard ratio.
        rng: Random generator.

    Returns:
        The target and the `Project` arguments.
    """
    happens = rng.exponential(1 / numpy.exp(signal))
    stops = rng.exponential(1.5, len(frame))
    frame["time"] = numpy.minimum(happens, stops)
    frame["event"] = (happens <= stops).astype(int)
    return "time", {"task": "survival", "event_column": "event"}


def grade(
    frame: pandas.DataFrame, signal: Any, rng: numpy.random.Generator
) -> Labels:
    """Adds four ordered levels cut from a logistic latent score.

    Args:
        frame: The features; the level column is added in place.
        signal: The linear predictor.
        rng: Random generator.

    Returns:
        The target and the `Project` arguments.
    """
    order = ["low", "medium", "high", "extreme"]
    latent = signal + rng.logistic(size=len(frame))
    cuts = numpy.quantile(latent, [0.25, 0.5, 0.75])
    frame["level"] = numpy.array(order)[numpy.searchsorted(cuts, latent)]
    return "level", {"task": "ordinal", "ordinal_order": order}


def rank(
    frame: pandas.DataFrame, signal: Any, rng: numpy.random.Generator
) -> Labels:
    """Adds queries of 15 documents and a relevance grade from 0 to 3.

    Args:
        frame: The features; query and grade columns are added in place.
        signal: Unused; the grade follows `x1`.
        rng: Random generator.

    Returns:
        The target and the `Project` arguments.
    """
    del signal
    rows = len(frame)
    frame["query"] = [f"q{i // QUERY}" for i in range(rows)]
    value = numpy.round(1.2 * frame.x1 + rng.normal(0, 0.6, rows))
    frame["grade"] = numpy.clip(value, 0, 3).astype(int)
    return "grade", {"task": "ranking", "query_column": "query"}


def tag(
    frame: pandas.DataFrame, signal: Any, rng: numpy.random.Generator
) -> Labels:
    """Adds three 0/1 labels with different links to the signal.

    Args:
        frame: The features; the label columns are added in place.
        signal: The linear predictor.
        rng: Random generator.

    Returns:
        The targets and the `Project` arguments.
    """
    names = []
    for i, weight in enumerate((1.0, -1.0, 0.5)):
        chance = 1 / (1 + numpy.exp(-weight * signal))
        frame[f"label{i}"] = (rng.random(len(frame)) < chance).astype(int)
        names.append(f"label{i}")
    return names, {"task": "multilabel"}


def pair(
    frame: pandas.DataFrame, signal: Any, rng: numpy.random.Generator
) -> Labels:
    """Adds two numeric outcomes.

    Args:
        frame: The features; the outcome columns are added in place.
        signal: The linear predictor.
        rng: Random generator.

    Returns:
        The targets and the `Project` arguments.
    """
    rows = len(frame)
    frame["first"] = signal + rng.normal(0, 0.3, rows)
    frame["second"] = frame.x4 - signal + rng.normal(0, 0.3, rows)
    return ["first", "second"], {"task": "multioutput"}


def treat(
    frame: pandas.DataFrame, signal: Any, rng: numpy.random.Generator
) -> Labels:
    """Adds a randomized arm and an outcome it lifts where `x1 > 0`.

    Args:
        frame: The features; arm and outcome columns are added in place.
        signal: Unused; the effect follows `x1`.
        rng: Random generator.

    Returns:
        The target and the `Project` arguments.
    """
    del signal
    rows = len(frame)
    arm = rng.integers(0, 2, rows)
    effect = 0.35 * (frame.x1 > 0)
    frame["treated"] = arm
    frame["bought"] = (rng.random(rows) < 0.25 + arm * effect).astype(int)
    return "bought", {"task": "uplift", "treatment_column": "treated"}


LABELERS: dict[
    str,
    Callable[[pandas.DataFrame, Any, numpy.random.Generator], Labels],
] = {
    "binary": flip,
    "multiclass": choose,
    "regression": regress,
    "survival": censor,
    "ordinal": grade,
    "ranking": rank,
    "multilabel": tag,
    "multioutput": pair,
    "uplift": treat,
}


def forecast(
    rows: int, rng: numpy.random.Generator
) -> tuple[pandas.DataFrame, Any, dict[str, Any]]:
    """A daily series with trend and weekly season.

    Args:
        rows: Days in the series.
        rng: Random generator.

    Returns:
        The series, its value column and the `Project` arguments.
    """
    steps = numpy.arange(rows)
    value = (
        20
        + 0.03 * steps
        + 4 * numpy.sin(2 * numpy.pi * steps / 7)
        + rng.normal(0, 0.4, rows)
    )
    frame = pandas.DataFrame(
        {"date": pandas.date_range("2022-01-01", periods=rows), "sales": value}
    )
    options = {
        "task": "forecast",
        "horizon": 4,
        "time_column": "date",
        "seasonal_period": 7,
    }
    return frame, "sales", options


def simulate(
    kind: str, rows: int, seed: int, shape: str = "plain"
) -> tuple[pandas.DataFrame, Any, dict[str, Any]]:
    """A clean generated task of one kind.

    Args:
        kind: `forecast` or a key of `LABELERS`.
        rows: Rows of the table.
        seed: Random seed.
        shape: `plain` (numeric features), `mixed` (also a categorical
            column and `MISSING` of `x2` and `x3` missing) or `wide` (also
            40 noise columns). A forecast is always plain.

    Returns:
        The table, the target and the `Project` arguments the kind needs.

    Raises:
        InputError: For an unknown kind or shape.
    """
    if shape not in ("plain", "mixed", "wide"):
        raise errors.InputError(f"unknown shape {shape!r}")
    if kind != "forecast" and kind not in LABELERS:
        raise errors.InputError(f"unknown kind {kind!r}")
    rng = numpy.random.default_rng(seed)
    if kind == "forecast":
        return forecast(rows, rng)
    frame = pandas.DataFrame(
        {f"x{i}": rng.normal(size=rows) for i in range(1, 5)}
    )
    signal = 1.2 * frame.x1 - 0.8 * frame.x2 + 0.5 * frame.x3
    if shape == "mixed":
        region = rng.choice(["north", "south", "east", "west"], rows)
        frame["region"] = region
        signal = signal + 0.6 * (region == "north")
    if shape == "wide":
        for i in range(40):
            frame[f"noise{i}"] = rng.normal(size=rows)
    target, options = LABELERS[kind](frame, signal, rng)
    if shape == "mixed":
        for column in ("x2", "x3"):
            frame.loc[rng.random(rows) < MISSING, column] = numpy.nan
    return frame, target, options


def cached(identifier: int) -> bool:
    """Whether an OpenML table is in scikit-learn's local cache.

    Args:
        identifier: The OpenML data id.

    Returns:
        True when its description file is cached.
    """
    home = pathlib.Path(datasets.get_data_home())
    folder = home / "openml" / "openml.org" / "api" / "v1" / "json" / "data"
    return (folder / f"{identifier}.gz").exists()


def pinned(identifier: int) -> str:
    """SHA-256 of a cached OpenML table's ARFF file as OpenML serves it.

    scikit-learn stores the file gzipped, compressing it itself when the
    server did not, so the digest is taken on the decompressed bytes.

    Args:
        identifier: The OpenML data id; its description and file must be
            cached.

    Returns:
        The hexadecimal digest.

    Raises:
        InputError: If the cache does not hold exactly one data file.
    """
    home = pathlib.Path(datasets.get_data_home()) / "openml" / "openml.org"
    description = home / "api" / "v1" / "json" / "data" / f"{identifier}.gz"
    found = json.loads(gzip.decompress(description.read_bytes()))
    number = found["data_set_description"]["file_id"]
    folder = home / "data" / "v1" / "download" / str(number)
    files = sorted(folder.iterdir()) if folder.is_dir() else []
    if len(files) != 1:
        raise errors.InputError(f"OpenML {identifier}: no single cached file")
    return hashlib.sha256(gzip.decompress(files[0].read_bytes())).hexdigest()


def fetch(identifier: int | str, rows: int = ROWS) -> pandas.DataFrame:
    """A real table from the local caches, without the network.

    Args:
        identifier: An OpenML data id, or the name of a scikit-learn
            `load_*` function such as `load_wine`.
        rows: Rows kept, drawn with a fixed seed, when the table is longer.

    Returns:
        The table, with categorical columns as text.

    Raises:
        InputError: If an OpenML table is not cached, or its file is not the
            one `PINS` names.
    """
    if isinstance(identifier, str):
        bunch = getattr(datasets, identifier)(as_frame=True)
        frame: pandas.DataFrame = bunch.frame.copy()
    else:
        if not cached(identifier):
            raise errors.InputError(f"OpenML {identifier} is not cached")
        if pinned(identifier) != PINS.get(identifier):
            raise errors.InputError(
                f"OpenML {identifier}: cached file differs from its pin"
            )
        bunch = datasets.fetch_openml(
            data_id=identifier, as_frame="auto", parser="auto"
        )
        loaded: pandas.DataFrame | None = bunch.frame
        frame = loaded if loaded is not None else pandas.DataFrame()
        if loaded is None:
            # A sparse ARFF file arrives as a matrix; it is made dense.
            frame = pandas.DataFrame(
                bunch.data.toarray(), columns=bunch.feature_names
            )
            frame[bunch.target_names[0]] = numpy.asarray(bunch.target)
        frame = frame.copy()
    for column in frame.select_dtypes("category").columns:
        frame[column] = frame[column].astype(object)
    if len(frame) > rows:
        frame = frame.sample(rows, random_state=0)
    return frame.reset_index(drop=True)


def table(
    identifier: int | str,
    target: Any,
    task: str,
    drop: tuple[str, ...] = (),
    **options: Any,
) -> Callable[[], tuple[pandas.DataFrame, Any, dict[str, Any]]]:
    """A builder of one real task.

    Args:
        identifier: As for `fetch`.
        target: The target column or columns.
        task: The task a practitioner would declare.
        drop: Columns a practitioner would remove first (known leaks).
        **options: Further `Project` arguments.

    Returns:
        A function returning the table, target and arguments; a regression
        or multi-output target stored as text is converted to numbers, and
        multi-label text labels to 0/1.
    """

    def build() -> tuple[pandas.DataFrame, Any, dict[str, Any]]:
        """Loads the table.

        Returns:
            The table, the target and the `Project` arguments.
        """
        frame = fetch(identifier).drop(columns=list(drop))
        names = [target] if isinstance(target, str) else list(target)
        for name in names:
            if task in ("regression", "multioutput"):
                frame[name] = pandas.to_numeric(frame[name])
            if task == "multilabel":
                text = frame[name].astype(str).str.upper()
                frame[name] = text.isin(["TRUE", "1"]).astype(int)
        settings = {"task": task, **options}
        if task == "ordinal":
            settings["ordinal_order"] = sorted(
                frame[target].unique().tolist(), key=float
            )
        return frame, target, settings

    return build


# Arguments a planted defect changes; the others edit a column.
OVERRIDES: dict[str, dict[str, Any]] = {
    "identifier": {"task": "auto"},
    "leak": {"leakage_policy": "block"},
    "binary": {"task": "binary_classification"},
}


def spoil(frame: pandas.DataFrame, target: Any, defect: str) -> None:
    """Plants one defect in a column of a clean table, in place.

    Args:
        frame: The clean table.
        target: Its target; for several targets the first is edited.
        defect: One of `constant`, `identifier`, `unlabeled`, `singleton`,
            `leak`, `labels`, `events`, `arm`, `queries`, `text`, `binary`
            (the last edits nothing: its defect is in `OVERRIDES`).

    Raises:
        InputError: For an unknown defect.
    """
    first = target if isinstance(target, str) else target[0]
    rows = len(frame)
    noise = numpy.random.default_rng(0).normal(0, 0.01, rows)
    edits: dict[str, Callable[[], tuple[str, Any]]] = {
        "constant": lambda: (first, 1),
        "identifier": lambda: (first, [f"id{i:05d}" for i in range(rows)]),
        "unlabeled": lambda: (
            first,
            frame[first].astype(float).mask(frame.index % 10 == 0),
        ),
        "singleton": lambda: (
            first,
            frame[first].mask(frame.index == 0, "violet"),
        ),
        "leak": lambda: ("leak", frame[first] + noise),
        "labels": lambda: (first, frame[first] + 0.5),
        "events": lambda: ("event", frame["event"] * 2),
        "arm": lambda: ("treated", 1),
        "queries": lambda: ("query", [f"q{i % 3}" for i in range(rows)]),
        "text": lambda: (first, numpy.where(frame[first] > 0, "high", "low")),
        "binary": lambda: (first, frame[first]),
    }
    if defect not in edits:
        raise errors.InputError(f"unknown defect {defect!r}")
    column, values = edits[defect]()
    frame[column] = values


def plant(
    kind: str, rows: int, defect: str
) -> Callable[[], tuple[pandas.DataFrame, Any, dict[str, Any]]]:
    """A builder of one task that should be refused.

    Args:
        kind: The clean task to start from (see `simulate`).
        rows: Its rows.
        defect: As for `spoil`.

    Returns:
        A function returning the defective table, target and arguments.
    """

    def build() -> tuple[pandas.DataFrame, Any, dict[str, Any]]:
        """Builds the defective task.

        Returns:
            The table, the target and the `Project` arguments.
        """
        frame, target, options = simulate(kind, rows, 7)
        spoil(frame, target, defect)
        return frame, target, {**options, **OVERRIDES.get(defect, {})}

    return build


# Further public OpenML tables fetched into scikit-learn's cache for C14:
# id -> (task name, kind, target, declared task). Fetched once with network
# by `fetch_openml(data_id=...)`; the runs then read the cache offline.
EXTRA: dict[int, tuple[str, str, str, str]] = {
    3: ("kr_vs_kp", "binary", "class", "classification"),
    11: ("balance_scale", "multiclass", "class", "classification"),
    15: ("breast_w", "binary", "Class", "classification"),
    23: ("cmc", "multiclass", "Contraceptive_method_used", "classification"),
    50: ("tic_tac_toe", "binary", "Class", "classification"),
    53: ("heart_statlog", "binary", "class", "classification"),
    59: ("ionosphere", "binary", "class", "classification"),
    61: ("iris", "multiclass", "class", "classification"),
    188: ("eucalyptus", "multiclass", "Utility", "classification"),
    307: ("vowel", "multiclass", "Class", "classification"),
    333: ("monks_1", "binary", "class", "classification"),
    1462: ("banknote", "binary", "Class", "classification"),
    1480: ("ilpd", "binary", "Class", "classification"),
    1494: ("qsar_biodeg", "binary", "Class", "classification"),
    1049: ("pc4", "binary", "c", "classification"),
    1063: ("kc2", "binary", "problems", "classification"),
    1068: ("pc1", "binary", "defects", "classification"),
    40982: ("steel_plates", "multiclass", "target", "classification"),
    40983: ("wilt", "binary", "class", "classification"),
    40994: ("climate_crashes", "binary", "outcome", "classification"),
    1487: ("ozone", "binary", "Class", "classification"),
    197: ("cpu_act", "regression", "usr", "regression"),
    216: ("elevators", "regression", "Goal", "regression"),
    218: ("house_8l", "regression", "price", "regression"),
    507: ("space_ga", "regression", "ln(VOTES/POP)", "regression"),
    503: ("wind", "regression", "MAL", "regression"),
    196: ("auto_mpg", "regression", "class", "regression"),
    230: ("machine_cpu", "regression", "class", "regression"),
    315: ("us_crime", "regression", "ViolentCrimesPerPop", "regression"),
    41021: ("moneyball", "regression", "RS", "regression"),
    560: ("bodyfat", "regression", "class", "regression"),
}
# The tables the selection rule of the module docstring chose, in suite
# order, in the same layout as `EXTRA`.
SUITE: dict[int, tuple[str, str, str, str]] = {
    12: ("mfeat_factors", "multiclass", "class", "classification"),
    32: ("pendigits", "multiclass", "class", "classification"),
    182: ("satimage", "multiclass", "class", "classification"),
    38: ("sick", "binary", "Class", "classification"),
    300: ("isolet", "multiclass", "class", "classification"),
    458: ("analcatdata_authorship", "multiclass", "Author", "classification"),
    469: ("analcatdata_dmft", "multiclass", "Prevention", "classification"),
    1050: ("pc3", "binary", "c", "classification"),
    1053: ("jm1", "binary", "defects", "classification"),
    4134: ("bioresponse", "binary", "target", "classification"),
    1497: ("wall_robot_navigation", "multiclass", "Class", "classification"),
    1501: ("semeion", "multiclass", "Class", "classification"),
    1485: ("madelon", "binary", "Class", "classification"),
    1486: ("nomao", "binary", "Class", "classification"),
    1468: ("cnae_9", "multiclass", "Class", "classification"),
    1475: (
        "first_order_theorem_proving",
        "multiclass",
        "Class",
        "classification",
    ),
    4534: ("phishingwebsites", "binary", "Result", "classification"),
    6332: ("cylinder_bands", "binary", "band_type", "classification"),
    4538: (
        "gesturephasesegmentationprocessed",
        "multiclass",
        "Phase",
        "classification",
    ),
    1478: ("har", "multiclass", "Class", "classification"),
    23381: ("dresses_sales", "binary", "Class", "classification"),
    40499: ("texture", "multiclass", "Class", "classification"),
    40668: ("connect_4", "multiclass", "class", "classification"),
    40966: ("miceprotein", "multiclass", "class", "classification"),
    41027: (
        "jungle_chess_2pcs_raw_endgame_complete",
        "multiclass",
        "class",
        "classification",
    ),
    23517: ("numerai28_6", "binary", "attribute_21", "classification"),
    40978: ("internet_advertisements", "binary", "class", "classification"),
    44958: (
        "auction_verification",
        "regression",
        "verification.time",
        "regression",
    ),
    44964: ("superconductivity", "regression", "critical_temp", "regression"),
    44965: (
        "geographical_origin_of_music",
        "regression",
        "latitude",
        "regression",
    ),
    44966: ("solar_flare", "regression", "c_class_flares", "regression"),
    44969: (
        "naval_propulsion_plant",
        "regression",
        "gt_compressor_decay_state_coefficient",
        "regression",
    ),
    44973: ("grid_stability", "regression", "stab", "regression"),
    44974: ("video_transcoding", "regression", "utime", "regression"),
    44975: ("wave_energy", "regression", "energy_total", "regression"),
    44976: ("sarcos", "regression", "V22", "regression"),
    44981: ("pumadyn32nh", "regression", "thetadd6", "regression"),
    44984: ("cps88wages", "regression", "wage", "regression"),
    44987: (
        "socmob",
        "regression",
        "counts_for_sons_current_occupation",
        "regression",
    ),
    44990: ("brazilian_houses", "regression", "total", "regression"),
    44992: ("fps_benchmark", "regression", "FPS", "regression"),
    44993: ("health_insurance", "regression", "whrswk", "regression"),
    45012: ("fifa", "regression", "wage_eur", "regression"),
    44962: ("forest_fires", "regression", "area", "regression"),
    44967: ("student_performance_por", "regression", "G3", "regression"),
    44994: ("cars", "regression", "Price", "regression"),
    181: (
        "yeast",
        "multiclass",
        "class_protein_localization",
        "classification",
    ),
    1457: ("amazon_commerce_reviews", "multiclass", "Class", "classification"),
    41138: ("apsfailure", "binary", "class", "classification"),
    41164: ("fabert", "multiclass", "class", "classification"),
    41143: ("jasmine", "binary", "class", "classification"),
    1169: ("airlines", "binary", "Delay", "classification"),
    41158: ("gina", "binary", "class", "classification"),
    41144: ("madeline", "binary", "class", "classification"),
    41145: ("philippine", "binary", "class", "classification"),
    41156: ("ada", "binary", "class", "classification"),
    41157: ("arcene", "binary", "class", "classification"),
    41168: ("jannis", "multiclass", "class", "classification"),
    1515: ("micro_mass", "multiclass", "Class", "classification"),
    41146: ("sylvine", "binary", "class", "classification"),
    41142: ("christine", "binary", "class", "classification"),
    41169: ("helena", "multiclass", "class", "classification"),
    41166: ("volkert", "multiclass", "class", "classification"),
    40685: ("shuttle", "multiclass", "class", "classification"),
    41150: ("miniboone", "binary", "signal", "classification"),
    41162: ("kick", "binary", "IsBadBuy", "classification"),
    42733: ("click_prediction_small", "binary", "click", "classification"),
    42734: ("okcupid_stem", "multiclass", "job", "classification"),
    42732: ("sf_police_incidents", "binary", "ViolentCrime", "classification"),
    42570: (
        "mercedes_benz_greener_manufacturing",
        "regression",
        "y",
        "regression",
    ),
    550: ("quake", "regression", "col_4", "regression"),
    546: ("sensory", "regression", "Score", "regression"),
    505: ("tecator", "regression", "fat", "regression"),
    41540: ("black_friday", "regression", "Purchase", "regression"),
    422: ("topo_2_1", "regression", "oz267", "regression"),
    416: ("yprop_4_1", "regression", "oz252", "regression"),
    42724: ("onlinenewspopularity", "regression", "shares", "regression"),
    42729: (
        "nyc_taxi_green_dec_2016",
        "regression",
        "tip_amount",
        "regression",
    ),
    201: ("pol", "regression", "foo", "regression"),
    41980: (
        "sat11_hand_runtime_regression",
        "regression",
        "runtime",
        "regression",
    ),
    531: ("boston", "regression", "MEDV", "regression"),
    42563: ("house_prices_nominal", "regression", "SalePrice", "regression"),
    574: ("house_16h", "regression", "price", "regression"),
    3277: ("qsar_tid_10980", "regression", "MEDIAN_PXC50", "regression"),
    43071: ("mip_2016_regression", "regression", "PAR10", "regression"),
}
# The third round: OpenML100 tables taken by the rule of the module
# docstring, in the study's order, in the same layout as `EXTRA`.
FURTHER: dict[int, tuple[str, str, str, str]] = {
    42: ("soybean", "multiclass", "class", "classification"),
    60: ("waveform_5000", "multiclass", "class", "classification"),
    312: ("scene", "binary", "Urban", "classification"),
    375: ("japanesevowels", "multiclass", "speaker", "classification"),
    377: ("synthetic_control", "multiclass", "class", "classification"),
    451: ("irish", "binary", "Leaving_Certificate", "classification"),
    470: ("profb", "binary", "Home/Away", "classification"),
}
# Suite datasets the selection rule drops as copies (rule 2): id -> the
# table already used that it repeats.
COPIES: dict[int, str] = {
    14: "mfeat-factors (12): another feature view of the same numerals",
    16: "mfeat-factors (12): another feature view of the same numerals",
    18: "mfeat-factors (12): another feature view of the same numerals",
    22: "mfeat-factors (12): another feature view of the same numerals",
    40979: "mfeat-factors (12): another feature view of the same numerals",
    28: "load_digits, the UCI optdigits test set",
    1590: "adult (179)",
    1510: "load_breast_cancer, the UCI wdbc table",
    40975: "car (991)",
    40984: "segment (36)",
    46: "quality task 363702 (splice)",
    40670: "quality task 363702 (splice), recoded",
    44956: "abalone (183)",
    42726: "abalone (183)",
    44957: "quality task 363612 (airfoil_self_noise)",
    44959: "quality task 363625 (concrete_compressive_strength)",
    44970: "quality task 363698 (QSAR_fish_toxicity)",
    44983: "quality task 363686 (miami_housing)",
    44960: "energy loads (1472)",
    44971: "wine_quality (287), white wines",
    44972: "wine_quality (287) and wine_red_quality (40691)",
    40498: "wine_quality (287), white wines",
    44977: "california_housing (43939)",
    44978: "cpu_act (197)",
    44979: "diamonds (42225)",
    44980: "kin8nm (189)",
    44989: "house_sales (42731)",
    45402: "space_ga (507)",
    42730: "us_crime (315)",
    40981: "credit_approval (29), recoded as Statlog Australian",
    40900: "satimage (182), relabelled for anomaly detection",
    541: "socmob (44987), taken earlier from OpenML-CTR23",
    42688: "brazilian_houses (44990), taken earlier from OpenML-CTR23",
    20: "mfeat-factors (12): another feature view of the same numerals",
    21: "car (991), before it was made binary",
    334: "monks-problems-1 (333): the same robots under another concept",
    335: "monks-problems-1 (333): the same robots under another concept",
}
# Datasets the selection rule drops as copies of the sealed TabArena-v0.1
# hold-out (rule 5): id -> the hold-out table it repeats.
SEALED: dict[int, str] = {
    31: "credit-g (TabArena task 363626)",
    37: "diabetes (TabArena task 363629)",
    42225: "diamonds (TabArena task 363631)",
    4135: "Amazon_employee_access (TabArena task 363613)",
    4541: "Diabetes130US (TabArena task 363630)",
    1111: "kddcup09_appetency (TabArena task 363683)",
    44963: "physiochemical_protein (TabArena task 363693)",
    3050: "QSAR-TID-11 (TabArena task 363697)",
}
# OpenML name, version and licence of each `EXTRA`, `SUITE` and `FURTHER`
# table, as fetched.
SOURCES: dict[int, tuple[str, int, str]] = {
    3: ("kr-vs-kp", 1, "Public"),
    11: ("balance-scale", 1, "Public"),
    15: ("breast-w", 1, "Public"),
    23: ("cmc", 1, "Public"),
    50: ("tic-tac-toe", 1, "Public"),
    53: ("heart-statlog", 1, "Public"),
    59: ("ionosphere", 1, "Public"),
    61: ("iris", 1, "Public"),
    188: ("eucalyptus", 1, "Public"),
    307: ("vowel", 2, "Public"),
    333: ("monks-problems-1", 1, "Public"),
    1462: ("banknote-authentication", 1, "Public"),
    1480: ("ilpd", 1, "Public"),
    1494: ("qsar-biodeg", 1, "Public"),
    1049: ("pc4", 1, "Public"),
    1063: ("kc2", 1, "Public"),
    1068: ("pc1", 1, "Public"),
    40982: ("steel-plates-fault", 3, "Public"),
    40983: ("wilt", 2, "Public"),
    40994: ("climate-model-simulation-crashes", 4, "Public"),
    1487: ("ozone-level-8hr", 1, "Public"),
    197: ("cpu_act", 1, "Public"),
    216: ("elevators", 1, "Public"),
    218: ("house_8L", 1, "Public"),
    507: ("space_ga", 1, "Public"),
    503: ("wind", 1, "Public"),
    196: ("autoMpg", 1, "Public"),
    230: ("machine_cpu", 1, "Public"),
    315: ("us_crime", 1, "Public"),
    41021: ("Moneyball", 2, "Public"),
    560: ("bodyfat", 1, "Public"),
    12: ("mfeat-factors", 1, "Public"),
    32: ("pendigits", 1, "Public"),
    182: ("satimage", 1, "Public"),
    38: ("sick", 1, "Public"),
    300: ("isolet", 1, "Public"),
    458: ("analcatdata_authorship", 1, "Public"),
    469: ("analcatdata_dmft", 1, "Public"),
    1050: ("pc3", 1, "Public"),
    1053: ("jm1", 1, "Public"),
    4134: ("Bioresponse", 1, "Public"),
    1497: ("wall-robot-navigation", 1, "Public"),
    1501: ("semeion", 1, "Public"),
    1485: ("madelon", 1, "Public"),
    1486: ("nomao", 1, "Public"),
    1468: ("cnae-9", 1, "Public"),
    1475: ("first-order-theorem-proving", 1, "Public"),
    4534: ("PhishingWebsites", 1, "Public"),
    6332: ("cylinder-bands", 2, "Public"),
    4538: ("GesturePhaseSegmentationProcessed", 1, "Public"),
    1478: ("har", 1, "Public"),
    23381: ("dresses-sales", 2, "Public"),
    40499: ("texture", 1, "Public"),
    40668: ("connect-4", 2, "public"),
    40966: ("MiceProtein", 4, "Public"),
    41027: ("jungle_chess_2pcs_raw_endgame_complete", 1, "Public"),
    23517: ("numerai28.6", 2, "Public"),
    40978: ("Internet-Advertisements", 2, "Public"),
    44958: ("auction_verification", 7, "CC BY 4.0"),
    44964: ("superconductivity", 7, "CC BY 4.0"),
    44965: ("geographical_origin_of_music", 7, "CC BY 4.0"),
    44966: ("solar_flare", 7, "CC BY 4.0"),
    44969: ("naval_propulsion_plant", 7, "CC BY 4.0"),
    44973: ("grid_stability", 7, "CC BY 4.0"),
    44974: ("video_transcoding", 6, "CC BY 4.0"),
    44975: ("wave_energy", 6, "CC BY 4.0"),
    44976: ("sarcos", 9, "Public"),
    44981: ("pumadyn32nh", 2, "Public"),
    44984: ("cps88wages", 2, "Public"),
    44987: ("socmob", 4, "Non-commercial scholarly and teaching purposes."),
    44990: ("brazilian_houses", 16, "CC 0: Public Domain"),
    44992: ("fps_benchmark", 2, "CC BY"),
    44993: ("health_insurance", 3, "GPL (>= 2)"),
    45012: ("fifa", 19, "CC0: Public Domain"),
    44962: ("forest_fires", 9, "CC BY 4.0"),
    44967: ("student_performance_por", 8, "CC BY 4.0"),
    44994: ("cars", 4, "Public"),
    181: ("yeast", 1, "Public"),
    1457: ("amazon-commerce-reviews", 1, "Public"),
    41138: ("APSFailure", 1, "CC0"),
    41164: ("fabert", 1, "Public"),
    41143: ("jasmine", 1, "Public"),
    1169: ("airlines", 1, "Public"),
    41158: ("gina", 1, "Public"),
    41144: ("madeline", 1, "Public"),
    41145: ("philippine", 1, "Public"),
    41156: ("ada", 1, "Public"),
    41157: ("arcene", 2, "Public"),
    41168: ("jannis", 1, "Public"),
    1515: ("micro-mass", 2, "Public"),
    41146: ("sylvine", 1, "Public"),
    41142: ("christine", 1, "Public"),
    41169: ("helena", 1, "Public"),
    41166: ("volkert", 1, "Public"),
    40685: ("shuttle", 1, "public"),
    41150: ("MiniBooNE", 1, "CC0"),
    41162: ("kick", 1, "CC0"),
    42733: ("Click_prediction_small", 10, "Public"),
    42734: ("okcupid-stem", 3, "Public"),
    42732: ("sf-police-incidents", 6, "CC0"),
    42570: ("Mercedes_Benz_Greener_Manufacturing", 2, "Public"),
    550: ("quake", 2, "Public"),
    546: ("sensory", 1, "Public"),
    505: ("tecator", 1, "Public"),
    41540: ("black_friday", 1, "GPL-1"),
    422: ("topo_2_1", 1, "Public"),
    416: ("yprop_4_1", 1, "Public"),
    42724: ("OnlineNewsPopularity", 2, "Public"),
    42729: ("nyc-taxi-green-dec-2016", 3, "Public"),
    201: ("pol", 1, "Public"),
    41980: ("SAT11-HAND-runtime-regression", 1, "Public"),
    531: ("boston", 1, "Public"),
    42563: ("house_prices_nominal", 1, "Public"),
    574: ("house_16H", 1, "Public"),
    3277: ("QSAR-TID-10980", 1, "Public"),
    43071: ("MIP-2016-regression", 3, "Public"),
    42: ("soybean", 1, "Public"),
    60: ("waveform-5000", 1, "Public"),
    312: ("scene", 1, "Public"),
    375: ("JapaneseVowels", 1, "Public"),
    377: ("synthetic_control", 1, "Public"),
    451: ("irish", 1, "Public"),
    470: ("profb", 1, "Public"),
}
# SHA-256 of each OpenML table's ARFF file as OpenML serves it (`pinned`):
# every table read through `fetch`, the leakage set's (C2) among them.
PINS: dict[int, str] = {
    14: "bc5dd56b0cf2ee0ced33c0cc17d82b855c3eef43e1644b3969e9b44d2b2e3933",
    16: "1f9c621a9ee7435e84dfcaf02740e498815b3b626dda54a1d8faa41897c9ea80",
    18: "0ad5a3afe6ad0a4851d2888980ce099b69c76030801bafab1e0e7e2e284837ea",
    22: "24e10bd4fcec1559b0e9db31a96a3b2cdf0832b2345d00d0c383e459fc6b7db9",
    28: "e86f102c957f41c7022b187766b0f9949a54ae8cedc492de1cc29765f060358a",
    46: "4b22fd2fc0564f0b5af59a5a40ee0dd77476835ba48b717960d85d095be53036",
    150: "c9db5c4c8abe3231778f431a932700f11fbc3bbb53c044e93a2789910a6a5bbb",
    195: "9120e9b75b5a8a38ada475833170048cd5826f8afa2b7e9a426f0a6f9bda1ac0",
    537: "d9747476bff8fcce7d3531190bb3b638413bf5cbf05d4175a229760946dd3a27",
    1510: "e850cb4cc32dacf29d3fa639d0029cb45582d4838a62322e7b3fbac1b8452f1d",
    1597: "fdaf12730dc1fc426f318b71349f24f5c5fd00aa1152940be7e7509ae3d89d2a",
    4532: "89c9a6b5de12e0fc1050eebf545adfd3fefde94b0b480f000e8eb03a459d0fc1",
    40498: "c7fe675c2f0f2c1a092f2e919ec2296970676ca9ade7e10b7a5d2863e83a4d47",
    40670: "bcd958ef5021e769e73dce7e629878a56338213a18bc108fef638c78b5b6941c",
    41700: "58896d520b079bfb3879655154f096ddbbf07f48af86ae3c8adb8e21aa170be6",
    3: "b22a8a12bd40648400b000bad0545683b8ecd33ac84f2c265dd5c309c832fd69",
    6: "a31f2d9ad119a53e182e7371d48981982dbb2b0a5e9f0aa7b73f373857b64e2e",
    11: "87d3943cea423bb376be52b5dfec17f79c63108dddcf40e0a46e5ba7196a5c24",
    12: "609d8f01dbf3f34a953617cc3586ddbb897d98013eba4ef4a4bf73791205ae5c",
    15: "cb44e7f62f4a962881147264e5f985eacc13256712e7a4b9d1731ced05f3ee9b",
    23: "9d8694d39e05d9d2963fccbe4f71e2af04c72c13de45f5dcbe775a586f214567",
    24: "7b653776e56943dfff0296b5e103159893646cc46ced112f72ff23e9b7ba595d",
    27: "4d609f1611113bec1a2b67f30115f0054945b0e123260f0cfcaa7b6325eb1841",
    29: "ae8e5eadb70e187885ddc6defb7f3e4afa726aa472ac63ebf5e387f0a8467891",
    31: "768be84ef749110d5fcac109a6541f2df190e8c6987e1dc432c0a4f29d0642f5",
    32: "5be0ce5f618a248a15e953bb63bd2c6bd99c1b9cdcd59c0a6abf06ae58758bca",
    36: "f54038600f6b8ab6cb6edf58887ba98c891781250b4140f5edeb74296e466854",
    37: "4eddd5b2b64679e8888348e306520a393d6a28e1ddc9643cfb76fc5d912d6d40",
    38: "85c39615380ca4c05c17a53c3faad9523924958d1e9bc5af0dcbefc6339e7800",
    44: "f0f6ccd8e8c6b1feeb09c0565ac691d7906cca5373959981512b9934ab4112a6",
    50: "09cb4ceb7d410dc7663a5ae8a4d9da6d82270468bdfa5d73a4161204cbb90133",
    53: "e16082cc6158f29be2e7b081953cd6edf342760d135bccdb64e191d0834d6300",
    54: "727ab4eb1bf76c0f01adccea696a07c690b5f1e25e15b3defc68c015360edf36",
    59: "bcbf8c656fe9a44a87df76003b9b2b6b36bfc52c01f0e316014cebbce9ccb701",
    61: "38110f3fd3cef003ca4909038a4d6b235f36a85d036dc7eedeff67c5925e826c",
    151: "2d86fbc74c69a5c04bcde0119bce280dd6d8efa0115987edc75b77ccf024d7f2",
    179: "d32d87c2dd406870351f1d49acb4ddca358dad6708c453b7adf0d81841fce4fc",
    181: "ae97eccd3ad5f9a998efec8acb2026f1748df6c8a392a5b2b0f321c2ef053933",
    182: "85a02ca850ecd8a005f746fa4d4dfe543250d8da51ef90590ba9db0f31546850",
    183: "a75351597c04719e793dd5fc9bee926bae0f06e413fe77591f01e4688154988e",
    188: "59a2fe2cb13540ea64a01faba34048209e096bb75a105b9c053950df4f6d678c",
    189: "698fb5297abde73c001b07f8fb2fe31626366cf179678aed0990737cc0689c79",
    196: "fe7e942c14f790bc7a350b030244519d0d96453ebf1b1c743b368fa3500bc05f",
    197: "16ada2bd7088e92ac1cdc2c5285275a0bfd72dcafa7095ac3870f9943423c507",
    201: "0f5f397fb144dfb7269a03e4eccc84a413a48982ceef0a91b142c9917b39675e",
    216: "dbf7a8f4531b11a3b295a5f0e84503032c66a64986d74dd673112a0da5cd41f2",
    218: "99e8d7e69102d6b57423eff5bf63445876d17e09ed89c763b7dbc99bd77c2d07",
    227: "bd183e8f4310187838d4fad6a5ed64e07371c21cff45fb8db75400ffbd29a951",
    230: "a07e9467e9daec5cb93a73eb791b9703795c2c4b920a4f16677d79976e323ede",
    287: "a06ffbd99a40aaf09315f914a3a681f7d6d757a614c5d414410a4877f5709093",
    300: "b18b12257017ab9fac7015f93cfefaee0291b28c6bcbdc11520c0ac2aa1a9ee2",
    307: "a0526fad8f69a04a9b8844b756046ab0dca5a97954e14b5af08d0c85e74e7e72",
    315: "a641e86073daa9d9f79eae19cefbc293540851d4d95eb40e59a2ff0022c883db",
    333: "87e0569e25818d0ed9ce935e9c12217b56471138370d878008ea85ef0a76f7b1",
    416: "13bf426f16a95f257f345800152806178c1cfc25e5e1a399c8a4973519cf497b",
    422: "79021c5bf78310c1c941db24944559eb11526607354f605b98f31bdea2711617",
    458: "8674a81705440cfcb458e481300185293c1fe75576c2d86d426b0be8148d4506",
    469: "d1fce1e52ef3e29c3cb8af6fc8215dd2982f829c0bfcd18a8be39269db4df501",
    503: "b821c83919b428787553318679c9c21be4bdaa186ae3b15451e5d18bf70c5b69",
    505: "47044ff39e813ee22441f718f33cca130686ac130bbf79756f8f7c5e9c2c05f5",
    507: "2ec8a4ea375118b249f592e440d99cfde435a67fae1574db6328ef3f7eccbea3",
    531: "2826ddb8cbce925e98114491f4774544b17d70ff0b7b1eede2af0a2ffd85a4f0",
    546: "158968b205cce12fa1b58da519aece8a1e396a633ce5fbf6a960be4413b93d56",
    550: "73b599c906cc2b6ff39c99513679db5f6a9d3802b4a5da1dfabac4386b77ee92",
    560: "6cb62e841e7036c63bd73bf5be53404f0e06c0a778d85040934b111111d9187f",
    574: "c9d4c46cb5f598fd9e3e1993a1ff06d85c662e50a9fe1fef9c1dc7d067d3b311",
    991: "74e9c6962376fbfbdea0c4daced642fa4e9e4be68725473ec19baec1ef89de86",
    1049: "a257bbf9d05df9b6a12675e5a74da2b138c59591f24bc3866c68f8abc67624c1",
    1050: "2ef1d5e4712544e44213a23532cf5fd8e8515ad22fa428b9f3443e4190a8e57e",
    1053: "8f3795e84e8755f1b21b4820c9f3286e60843d3f0a9751c595e0c47e786a7b52",
    1063: "832b27ed0b34faddd8d4b3c23cdc65e5118bd7694f016b449f3d0b70c8575fd5",
    1067: "fe0ea9ab3977bf3d2411dc72eefed2462ef4bf750a77668bdcf2acbb351c521d",
    1068: "a22470def91454aa442e54104e2856c53725e8c9108bdfaac867504982483f97",
    1120: "ff35972a47140925b0b698b3f0de8471d3d2f0ede19a553b2d3c58ba7caa19ee",
    1169: "c7683a90b40c2cabb1ddc950413da8efa231c058327b8c95511c40af13516c14",
    1457: "b5388cae01347259e7f8801c4220022f0d1349d88dd3e8ff9d0c21d2e3c23438",
    1461: "7381f5eec0cfcd20b70e70988ead1cff758fa1e9402e1f800ee8a4b59529e4b2",
    1462: "ba23e986caffafbdce29c651278bb0455cdae5c2274e86afe4716c47a4fb278d",
    1464: "ee1304cac4a650ac31afe7395a100536a165f935bbe718a4643757c1a842316a",
    1468: "5bff41d9dc6ab5a11059d4c9fbb09564739d2f98f836b6b90005ab8c7e45b420",
    1472: "8e2ca7aabe95e1f29deb432ff8c52f27c23a090172859fafad535e18ecd82fa4",
    1475: "8ada14230e4df0a3eb12aafd9af2bff99ee83bbb539c97d4fef169d981ca8c51",
    1478: "16e55052cbf8a3f9d7bc769501e4cb1c7909f50fd3947b8457f5a8c1f75fef4b",
    1480: "62b65e434ad612e1fa4cbd34888811f5f89d39e3f2744de5bd2a6a77b9926ddd",
    1485: "ac33b79e8849b0e2e0417c62c50a421237b3dd522c2b344f8da3c0f72a80ad7c",
    1486: "94b4e6229e434d04fb61f081e547c95ec0bfa99e9ea9e880acd5276d13b972e6",
    1487: "2d5cd15a541f72a46ececf2332e611944752cce78a79d1a3311bec8fbb51d10e",
    1489: "8dcd1c5236cea66cdc4e75cb84fbb4da7a4b1e8b1fe5ad5cd3fd825a1ae4c48b",
    1494: "959bdf9d436405525d63b01c85d8d5f6f9d5f8ff90f9b1d163ef8b5d7255722c",
    1497: "a7facd7720353055e1ef0af024448e002c551d0fd5defe0fc651661b7c187b3b",
    1501: "3a0bfdb44b8a11647d9ada08ceadac8523b220d880aff266d7321e605576246e",
    1515: "b4109c3ac28c6d248993fb1936df6137414c9d90efb46179880948e4cb46aa9f",
    3277: "54d22f442cb7cc1cdd54ea60d549d68faab9da1e78bea3784b68ec47056252a7",
    4134: "b8fbe587f40c8f48d1f30003b0b40f085cd32997f7167898228178c5a0919d92",
    4534: "3bcf93b1612306537ee2737751f1bac185b2a77886ab376644c3bf0c7f7cb3e6",
    4538: "939a5a0dda808213908be1f2385fc1bbb13e9df68ca7d64d1466f2da3f9824e2",
    6332: "6e792f09b2d59d0500a2ba9a5975b31b5eef9534b98f5bf87ab8047452e4c45e",
    23381: "44057bb677eb4a144d01816262fc164f923a46fdda019a026e83e0c44b58ee95",
    23517: "df837470119d913b9291342efe4b19abcb302c80bb86b5e372ededc4591b9d14",
    40499: "9143690415c61a8d9c33ededf947126fa71cced8f91d50642f06621f84e3e9bd",
    40589: "fb692f6e76b2c8941e90f01999eb8788bb1ab8307967583741f851d21ff286b8",
    40668: "430cc660591a281550b4270886776a7c0a7f8047bbdbbcc6ee428f223369e219",
    40685: "d0e47eb009fe0b77e67ec16a21d64efb69073dfb5f405936b7a51fc1dc22c9fd",
    40691: "f1179c9abed3a09cf4202caa08306f8b78c7e401bf37d31440fdeac003796314",
    40701: "6245b866e90138c5cdda040d2295c12c03fce552e881043ad0f6eed5feb63335",
    40945: "c56e3785c96cd8aceac85f11bff21c2bf72841426f50903a6f27be8f334a7d03",
    40966: "f48ced98ecfe8ff056d8ea3507e067258c82f29199a1af2261b516010827cc4d",
    40978: "853b895fb12183ee665de29559d8aab1dc77abcf24602e4c3132b23b035b9a09",
    40982: "635644c094456648c14501b2b565df6eb4bf436f63145ce7bf0421ee2b7f9531",
    40983: "e29d2b9c08fbd21fe7104df37f16eed8fd0d3a9c782cbb8efada0cf7ebb8d6bb",
    40994: "5be0901079f4d3317631ad9e955cd5e45d5eb8c753565b197e49e461524085b8",
    41021: "1ef7013b57ff2ce89a0b8b25dc643ce7eca4b24d6b93785c2945983b2e8364e2",
    41027: "647f9cded1b34558bd4587106bfabcb908e706194e78d3014fea6d594e909d75",
    41138: "e95128fc82ef5d5bc5a732b0bc63dea0f6e285cc60f31290c81f82ea13429c4f",
    41142: "307099164edf07b914eeb2b968b0ef8e0393dedb7bfe0c04937aea49d9f949cd",
    41143: "08142863cfdf0754752dd67bfb3ee7d9e29030c9b7fca6c85d69927000858ae2",
    41144: "8562d10a1b945172055dcd11852525962f952587eeee173790779c90af4f5ae4",
    41145: "819470adb79ea428c9f7c1cd9e58ceb8ab77c4d2a3249f22393318c4d0d0484d",
    41146: "cbd39adc5849eb951434e7194090e2611ff8a34033dbb0a78dd7c4249791cf48",
    41150: "907fae4b0de2f61e88f6cda799f53a9f7e39ee6876aa971a9211f5802d89a79d",
    41156: "9f487bfee74bcf72538fcdc1a1336931e4437824843eef090fdf71506602a290",
    41157: "c58c70465382108be4261cd6828b01c54e49fa53282e88d1417928c5f29ad66f",
    41158: "2511d829e62d473e1a9ee833770bfac65bdf9df321268a77e15db680afcd6a49",
    41162: "918b922249d410f9274557850e0f494d51a450a97c4755ce0c6889b82811d0ff",
    41164: "3afd98b3ff70400e7096ba55257947127abca4d3fb9f11bb9e465aafbc5329ed",
    41166: "0b782b59f04b71397da65a05d1117095944d35450ec4d54c423eda6809122d29",
    41168: "ffc0530b7f98207096b603b33c5cb33d1f92bc8b4f2a62acbeb5fe70e4107d4c",
    41169: "351c371c164b5ceacd9a87697d9efde170daf51ce4b3ce3819f25b7912e1042a",
    41540: "784e88ab33be4b7aae60a85ec8e1cad53ab3d27ef3350ada00fc93038d06b751",
    41980: "1ab43811428dfbd685e41edd34fd63222d4da768ab5060c8ee820268740a848b",
    42225: "0bbef7a24061e5cc4e8591eea59417948a2d2286401675b57163a555e1849b40",
    42563: "1e83c38ec2f33615fc3308a641e6f3a1fe4a7b4fbb8dd54913c6082799602bdc",
    42570: "44cb0065e346847de80f8a2a0fdfa9f38cfff90275129f3419cffb420d690910",
    42712: "24c7f663c80a0f0820f92b00fbc76d63c66aac4a133164ce51513a6a8bbb8cd6",
    42724: "bb6f42580b1a79c755c8d7652e13b8033b881d09c1c82ac8d721e5007b9cd84a",
    42729: "d5a584ef08e6dc02bf46cf20d2021f93fd687716b984e5a1b86a66d339814ba6",
    42731: "86508fd41de8e54786c8bb61527ddd8163a740390dbc9028fc104ca2e52f9928",
    42732: "f6415889d9952c71226edceaba74f4f105416eb77977b632307d4a787e50d9bb",
    42733: "add9a17b955be5ee4a89c3dc2c52544eb1f05098b404c99183d209caca4dbc39",
    42734: "1af41ae7bd750443ea535b324b91e830799c13b9bc2c2f64ea4fff29634c024d",
    43071: "8b80c61c8c093f4c336b92552afb0d51c6ccdd7b49bb75b763e992391da297a1",
    43939: "c4d895f661384b4030fb560ad031bb82f3e826af3c64907217bb633c0db65386",
    44958: "f0dcaf7165b3235a6031395f2068bb7833bfbeb32c263713cf8834bd679a4d97",
    44962: "25672c7dd97b7af4c00b5d5812c17f6fde8a9e422835e3610c730117c8d6e2b5",
    44964: "fdd7d82630afd1b35718bcd1772403db6068d85ec4a1cd8b8dc3b9ca0e0440c8",
    44965: "56fda911ab67dbccc093345d5e8e112ee40186899c20a8dc8f29bd15b063adb8",
    44966: "34f377d0c949b0a35e7b6ce5fd9854f4154842dd03e46d24cfcebe2d5c267794",
    44967: "ab4db935dca19424413e68129cab61609d6340d83194f185bb4e74e2268e4bd6",
    44969: "566a503cb6bcd1166f79346cadef9101a4be46573471b80158a6419fd0deea58",
    44973: "3e3e746b09ed94fac41732fe72912302c0b0a7b9ac683abf0db721299007d6a4",
    44974: "4842dde3ca6ad962c613e02b7b2627ba70e6cdd078a8be885b52d6ce8e3d49de",
    44975: "e16a2d7bb0b8d1594816aa44474cd1a6c4750498afe3ebe29d51fa5744144410",
    44976: "737ec889f0b6622917746d05f9737fad1c3cfb239114994ec65f718d79774d52",
    44981: "0756f4da4500b61e2025862e2b550ee83fc8684616b1a77cebf6dccd5e13d705",
    44984: "2bbaec07b4fc663edfbe3c3a0ef0b300516043195f36f5d1f69c38a8e9733306",
    44987: "299f17112726f2f67e3802ec99a3847930ff4e1c1cc6150736afbcbda406f8ac",
    44990: "094759fa443ddbcf885910e4a0fe291fdda5f226d30bc67df29840bef8781619",
    44992: "0a4488f0e01971e5a3bf8eaf07848da90b9acc555ddd7278c111ce4150017e38",
    44993: "f97193f812b47d30dc3ae87cd30fab5c468ef04893e7792933df873340514605",
    44994: "98a42023fa9701a1266dcea95b0808807f33b1f346d24b5503d89048656ae48c",
    45012: "00b905455c0717e88a08e3e2021437889fd6a91d869b936cb1f46fc66f6e8297",
    42: "673e0965c2cc18e70b177866afc8fcd15ad553a0b2098229abb2e0b3d4312f33",
    60: "ea857b7e905331181e43f56af95b3d512d08d0b03a65d2af48cb4dbdda2334a3",
    312: "bf94fc39f84c51b04cc0001f44e332e171c03d371faaa4611f579539d5b7de93",
    375: "2290186e0b8723a88c8e52e48c396f06f6cf98733f13d0e62d08b2b5c6184190",
    377: "2692c0c1bbf7b14485c94b151da9577f4b33acdd3352b448804f74f23e1dab39",
    451: "4d79ad272bdbdc5e4cdbcaa30e153aa50439d0e145a86c5ca9900d1224c502d2",
    470: "8a699382dcbf580a3b875644008d82171a2e54ef005bf20599b854cfb78cd23d",
}


# Kind, rows and shape of each simulated world, cycled: world `i` is
# `LAYOUTS[i % len(LAYOUTS)]` simulated with seed `i`.
LAYOUTS = (
    ("binary", 500, "plain"),
    ("regression", 300, "mixed"),
    ("multiclass", 1500, "plain"),
    ("survival", 800, "plain"),
    ("ordinal", 1200, "mixed"),
    ("ranking", 1500, "plain"),
    ("multilabel", 1000, "plain"),
    ("multioutput", 600, "plain"),
    ("forecast", 400, "plain"),
    ("uplift", 3000, "plain"),
    ("binary", 2000, "wide"),
    ("regression", 2000, "wide"),
    ("multiclass", 600, "mixed"),
    ("survival", 2000, "mixed"),
    ("forecast", 1000, "plain"),
    ("uplift", 1500, "mixed"),
    ("binary", 1000, "mixed"),
    ("ordinal", 400, "plain"),
    ("multilabel", 2000, "wide"),
    ("multioutput", 2000, "mixed"),
    ("ranking", 600, "mixed"),
    ("regression", 800, "plain"),
)


def worlds() -> int:
    """Simulated worlds: enough for both deciding splits of custody.

    `benchmarks.evidence.custody` needs `refusal_rate_max`'s ceiling
    minimum, `ceil(ln alpha / ln(1 - 0.05))` = 59, in each of validation
    and acceptance, so the group holds twice that. The rule was fixed
    before any added world was run: world `i` takes `LAYOUTS[i % 22]` with
    seed `i`, so the first 22 are the worlds run before.

    Returns:
        `2 * Ceiling(refusal_rate_max).minimum(alpha)`.
    """
    alpha = acceptance.load()["alpha"].value
    return 2 * acceptance.ceiling("refusal_rate_max").minimum(alpha)


def catalog() -> dict[str, Task]:
    """Every task, in the order they are run.

    The three groups are interleaved so a measurement cut short by time
    still has some of each.

    Returns:
        Task name to task.
    """
    real = [
        Task(
            "breast_cancer",
            "binary",
            "real",
            table("load_breast_cancer", "target", "classification"),
        ),
        Task(
            "wine",
            "multiclass",
            "real",
            table("load_wine", "target", "classification"),
        ),
        Task(
            "diabetes_progression",
            "regression",
            "real",
            table("load_diabetes", "target", "regression"),
        ),
        Task(
            "abalone",
            "regression",
            "real",
            table(183, "Class_number_of_rings", "regression"),
        ),
        Task(
            "vehicle",
            "multiclass",
            "real",
            table(54, "Class", "classification"),
        ),
        Task(
            "energy_loads",
            "multioutput",
            "real",
            table(1472, ["y1", "y2"], "multioutput"),
        ),
        Task(
            "wine_red_quality",
            "ordinal",
            "real",
            table(40691, "class", "ordinal"),
        ),
        Task(
            "emotions",
            "multilabel",
            "real",
            table(
                40589,
                [
                    "amazed.suprised",
                    "happy.pleased",
                    "relaxing.calm",
                    "quiet.still",
                    "sad.lonely",
                    "angry.aggresive",
                ],
                "multilabel",
            ),
        ),
        Task(
            "blood_transfusion",
            "binary",
            "real",
            table(1464, "Class", "classification"),
        ),
        Task("kin8nm", "regression", "real", table(189, "y", "regression")),
        Task(
            "titanic",
            "binary",
            "real",
            table(40945, "survived", "classification", drop=("boat", "body")),
        ),
        Task(
            "credit_approval",
            "binary",
            "real",
            table(29, "class", "classification"),
        ),
        Task(
            "segment",
            "multiclass",
            "real",
            table(36, "class", "classification"),
        ),
        Task(
            "cpu_small", "regression", "real", table(227, "usr", "regression")
        ),
        Task(
            "kc1_defects",
            "binary",
            "real",
            table(1067, "defects", "classification"),
        ),
        Task(
            "car", "binary", "real", table(991, "binaryClass", "classification")
        ),
        Task(
            "digits",
            "multiclass",
            "real",
            table("load_digits", "target", "classification"),
        ),
        Task(
            "wine_quality",
            "regression",
            "real",
            table(287, "quality", "regression"),
        ),
        Task(
            "colic",
            "binary",
            "real",
            table(27, "surgical_lesion", "classification"),
        ),
        Task(
            "phoneme", "binary", "real", table(1489, "Class", "classification")
        ),
        Task(
            "california_housing",
            "regression",
            "real",
            table(43939, "median_house_value", "regression"),
        ),
        Task(
            "spambase", "binary", "real", table(44, "class", "classification")
        ),
        Task(
            "churn", "binary", "real", table(40701, "class", "classification")
        ),
        Task("adult", "binary", "real", table(179, "class", "classification")),
        Task(
            "letter", "multiclass", "real", table(6, "class", "classification")
        ),
        Task(
            "bike_sharing",
            "regression",
            "real",
            table(42712, "count", "regression"),
        ),
        Task(
            "bank_marketing",
            "binary",
            "real",
            table(1461, "Class", "classification"),
        ),
        Task(
            "magic_telescope",
            "binary",
            "real",
            table(1120, "class:", "classification"),
        ),
        Task(
            "house_sales",
            "regression",
            "real",
            table(42731, "price", "regression"),
        ),
        Task(
            "mushroom", "binary", "real", table(24, "class", "classification")
        ),
        Task(
            "electricity",
            "binary",
            "real",
            table(151, "class", "classification"),
        ),
    ]
    real.extend(
        Task(name, kind, "real", table(identifier, target, task))
        for identifier, (name, kind, target, task) in (
            EXTRA | SUITE | FURTHER
        ).items()
    )
    simulated = [
        Task(
            f"sim_{kind}_{rows}_{shape}"
            + (f"_{index // len(LAYOUTS)}" if index >= len(LAYOUTS) else ""),
            kind,
            "simulated",
            functools.partial(simulate, kind, rows, index, shape),
        )
        for index, (kind, rows, shape) in (
            (i, LAYOUTS[i % len(LAYOUTS)]) for i in range(worlds())
        )
    ]
    defects = (
        (
            "constant",
            "binary",
            400,
            ("task.classification", "task.auto"),
            "constant target",
        ),
        (
            "unlabeled",
            "regression",
            400,
            ("data.unlabeled", "profile.target_and_features", "task.resolve"),
            "a tenth of the target missing",
        ),
        (
            "identifier",
            "multiclass",
            400,
            ("task.auto", "task.resolve"),
            "target is a row identifier",
        ),
        (
            "singleton",
            "multiclass",
            400,
            ("split.stratified.stratify", "split.stratified.folds"),
            "one class has a single row",
        ),
        (
            "leak",
            "binary",
            600,
            ("leakage.policy",),
            "copy of the target under leakage_policy block",
        ),
        (
            "labels",
            "multilabel",
            400,
            ("target.columns",),
            "a multi-label column that is not 0/1",
        ),
        ("events", "survival", 400, ("target.events",), "event column not 0/1"),
        ("arm", "uplift", 400, ("target.treatment",), "every row treated"),
        (
            "queries",
            "ranking",
            300,
            ("split.group.partition",),
            "only three queries",
        ),
        (
            "text",
            "regression",
            400,
            ("task.regression",),
            "regression declared on a text target",
        ),
        (
            "binary",
            "multiclass",
            400,
            ("task.classification",),
            "binary declared on three classes",
        ),
        (
            "singleton",
            "multiclass",
            12,
            ("split.stratified.stratify", "split.stratified.folds"),
            "twelve rows, one class of one row",
        ),
    )
    planted = [
        Task(
            f"planted_{defect}_{kind}_{rows}",
            kind,
            "planted",
            plant(kind, rows, defect),
            gates,
            cause,
        )
        for defect, kind, rows, gates, cause in defects
    ]
    order: list[Task] = []
    for index in range(max(len(real), len(simulated), len(planted))):
        order.extend(
            group[index]
            for group in (real, simulated, planted)
            if index < len(group)
        )
    return {task.name: task for task in order}


def registry(path: pathlib.Path = GATES) -> dict[str, str]:
    """Raising site to gate name, from the gate registry.

    Args:
        path: The registry file.

    Returns:
        Each `module:qualified.name` site mapped to its gate's dotted name.
    """

    def walk(tree: dict[str, Any], prefix: tuple[str, ...]) -> dict[str, str]:
        """Flattens the nested tables of dotted gate names.

        Args:
            tree: A table of the registry.
            prefix: The names above it.

        Returns:
            Site to gate for every gate below it.
        """
        found: dict[str, str] = {}
        for key, value in tree.items():
            if isinstance(value, dict) and "site" in value:
                found[str(value["site"])] = ".".join((*prefix, key))
            elif isinstance(value, dict):
                found.update(walk(value, (*prefix, key)))
        return found

    return walk(tomllib.loads(path.read_text(encoding="utf-8")), ())


def locate(error: BaseException, sites: dict[str, str]) -> tuple[str, str]:
    """The registered gate that raised a refusal.

    Args:
        error: The refusal raised by the run.
        sites: As returned by `registry`.

    Returns:
        The gate and its site, from the innermost traceback frame that the
        registry lists; `unregistered` and the innermost frame when none is.
    """
    frames = []
    trace = error.__traceback__
    while trace is not None:
        frame = trace.tb_frame
        module = frame.f_globals.get("__name__", "")
        frames.append(f"{module}:{frame.f_code.co_qualname}")
        trace = trace.tb_next
    for site in reversed(frames):
        if site in sites:
            return sites[site], site
    return "unregistered", frames[-1] if frames else ""


def attempt(task: Task, budget: int = BUDGET) -> dict[str, Any]:
    """Runs one task end to end and records how it ended.

    Args:
        task: The task.
        budget: Training budget in seconds.

    Returns:
        The task's name, kind and group; the outcome (`model`, `refused`
        or `failed`); for a refusal, the gate, site and message; for a
        failure, the error type and message; the selected model; seconds.
    """
    import scmle.api.project

    started = time.monotonic()
    record: dict[str, Any] = {
        "name": task.name,
        "kind": task.kind,
        "group": task.group,
    }
    try:
        frame, target, options = task.build()
        record["rows"] = len(frame)
        with tempfile.TemporaryDirectory() as folder:
            result = scmle.api.project.Project(
                frame,
                target,
                time_budget=budget,
                output_dir=folder,
                **options,
            ).run()
            record["outcome"] = "model"
            record["model"] = result.selected_model
    except (errors.BlockedError, errors.LeakageError) as error:
        gate, site = locate(error, registry())
        record.update(
            outcome="refused",
            gate=gate,
            site=site,
            error=type(error).__name__,
            message=str(error)[:500],
        )
    except Exception as error:  # noqa: BLE001 -- every ending is recorded
        record.update(
            outcome="failed",
            error=type(error).__name__,
            category=getattr(error, "category", ""),
            message=str(error)[:500],
        )
    record["seconds"] = round(time.monotonic() - started, 1)
    return record


def launch(name: str, folder: pathlib.Path, budget: int) -> dict[str, Any]:
    """Runs one task in a fresh process with the network cut.

    Args:
        name: The task's name.
        folder: Where the process writes its record.
        budget: Training budget in seconds.

    Returns:
        The record; `unfinished` when the process outlives `TIMEOUT` or
        writes nothing.
    """
    output = folder / f"{name}.json"
    environment = dict(os.environ)
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        environment[key] = DEAD
    environment["HF_HUB_OFFLINE"] = "1"
    command = [
        sys.executable,
        "-m",
        "benchmarks.evidence.refusal",
        "--task",
        name,
        "--out",
        str(output),
        "--budget",
        str(budget),
    ]
    started = time.monotonic()
    with contextlib.suppress(subprocess.TimeoutExpired):
        subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=TIMEOUT,
            check=False,
        )
    if output.exists():
        return dict(json.loads(output.read_text(encoding="utf-8")))
    return {
        "name": name,
        "outcome": "unfinished",
        "seconds": round(time.monotonic() - started, 1),
    }


def tally(
    records: list[dict[str, Any]], tasks: dict[str, Task]
) -> dict[str, Any]:
    """Counts the outcomes the criterion is decided on.

    Args:
        records: One record per task run.
        tasks: As returned by `catalog`.

    Returns:
        Per group: runs, models, refusals, failures, unfinished and refusals
        by gate, and `deciding`: the runs of the group's `DECIDING`-split
        items and their hits, every run that did not return a model (a
        refusal, a crash or a run past the timeout all count against the
        claim); and the precision counts: refusals, and those justified (a
        planted task refused by one of the gates it names).
    """
    counts: dict[str, Any] = {}
    placed = splits()
    for group in GROUPS:
        mine = [r for r in records if tasks[r["name"]].group == group]
        gates: dict[str, int] = {}
        for record in mine:
            if record["outcome"] == "refused":
                gates[record["gate"]] = gates.get(record["gate"], 0) + 1
        finished = [r for r in mine if r["outcome"] != "unfinished"]
        held = [r for r in mine if placed.get(r["name"]) == DECIDING]
        counts[group] = {
            "trials": len(finished),
            "models": sum(r["outcome"] == "model" for r in mine),
            "refusals": sum(r["outcome"] == "refused" for r in mine),
            "failures": sum(r["outcome"] == "failed" for r in mine),
            "unfinished": len(mine) - len(finished),
            "gates": dict(sorted(gates.items())),
            "deciding": {
                "trials": len(held),
                "hits": sum(r["outcome"] != "model" for r in held),
            },
        }
    refused = [r for r in records if r["outcome"] == "refused"]
    counts["precision"] = {
        "refusals": len(refused),
        "justified": sum(
            r.get("gate") in tasks[r["name"]].expected for r in refused
        ),
        "rubric": "automated: justified when the task was planted and the "
        "refusing gate is one the task names; not validator-reviewed",
    }
    return counts


def splits() -> dict[str, str]:
    """The custody split of each refusal task.

    Returns:
        Task name to split, from `CUSTODY`; empty when it is missing, so no
        task decides the rate.
    """
    if not CUSTODY.exists():
        return {}
    document = json.loads(CUSTODY.read_text(encoding="utf-8"))
    for each in document["sets"]:
        if each["name"] == "refusal":
            return dict(each["items"])
    return {}


def fingerprint() -> str:
    """Hash of the code that decides what is measured.

    Returns:
        The combined source hash.
    """
    return certificates.fingerprint(
        Task,
        simulate,
        worlds,
        *LABELERS.values(),
        forecast,
        fetch,
        pinned,
        table,
        spoil,
        plant,
        catalog,
        locate,
        attempt,
        tally,
        splits,
    )


def measure(
    seconds: float = SECONDS, parallel: int = PARALLEL, budget: int = BUDGET
) -> dict[str, Any]:
    """Runs the tasks in order until the wall clock runs out.

    Args:
        seconds: Wall clock after which no new task starts.
        parallel: Tasks run at once.
        budget: Training budget of each run.

    Returns:
        The document written to `refusal.json`: the settings, every task
        with its group and, for planted ones, the gates it names; the run
        records; the counts; the tasks not run; the code hash and duration.

    Raises:
        InputError: If `seconds`, `parallel` or `budget` is not positive.
    """
    if seconds <= 0 or parallel < 1 or budget < 1:
        raise errors.InputError("seconds, parallel and budget must be > 0")
    started = time.monotonic()
    tasks = catalog()
    names = list(tasks)
    records: list[dict[str, Any]] = []
    with (
        tempfile.TemporaryDirectory() as folder,
        concurrent.futures.ThreadPoolExecutor(parallel) as pool,
    ):
        running: set[concurrent.futures.Future[dict[str, Any]]] = set()
        queue = list(names)
        while queue or running:
            while (
                queue
                and len(running) < parallel
                and time.monotonic() - started < seconds
            ):
                name = queue.pop(0)
                running.add(
                    pool.submit(launch, name, pathlib.Path(folder), budget)
                )
                print(f"refusal: started {name}", file=sys.stderr)
            if not running:
                break
            done, running = concurrent.futures.wait(
                running, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in done:
                record = future.result()
                records.append(record)
                print(
                    f"refusal: {record['name']} {record['outcome']} "
                    f"{record.get('gate', '')}",
                    file=sys.stderr,
                )
    ran = {r["name"] for r in records}
    records.sort(key=lambda r: names.index(r["name"]))
    return {
        "version": 1,
        "claim": "C14",
        "settings": {
            "budget": budget,
            "seconds": seconds,
            "parallel": parallel,
            "timeout": TIMEOUT,
            "api": "scmle.api.project.Project(...).run(), defaults",
            "network": "cut in each run (dead proxy); local caches only",
        },
        "tasks": {
            n: {
                "kind": t.kind,
                "group": t.group,
                "expected": list(t.expected),
                "cause": t.cause,
            }
            for n, t in tasks.items()
        },
        "sources": {
            str(i): {"name": n, "version": v, "licence": lic}
            for i, (n, v, lic) in SOURCES.items()
        },
        "records": records,
        "counts": tally(records, tasks),
        "not_run": [n for n in names if n not in ran],
        "not_measured": [
            (
                "real survival, ranking, forecast and uplift tables (none "
                "is cached offline; those kinds are simulated only)"
            ),
            (
                "the sealed acceptance split and the validator's review of "
                "each refusal against a written rubric (plan section 6)"
            ),
            "budgets other than the short one used here",
        ],
        "code_hash": fingerprint(),
        "seconds": round(time.monotonic() - started, 1),
    }


def needed(
    rule: acceptance.Ceiling, hits: int, trials: int, alpha: float
) -> int | None:
    """Further trials, all without a refusal, before the rate can pass.

    Args:
        rule: The ceiling.
        hits: Refusals so far.
        trials: Trials so far.
        alpha: One-sided error rate.

    Returns:
        The number of further trials; None if none up to the acceptance
        search limit would do.
    """
    total = max(trials, rule.minimum(alpha))
    while not rule.passes(hits, total, alpha):
        total += 1
        if total > acceptance.LIMIT:
            return None
    return total - trials


def decide(
    rate: acceptance.Ceiling, deciding: dict[str, int], alpha: float
) -> tuple[str, str]:
    """The verdict and text of the refusal rate.

    Args:
        rate: The ceiling of `refusal_rate_max`.
        deciding: The `DECIDING`-split trials and their hits, every run
            without a model (a refusal, a crash or a timeout) being a hit.
        alpha: One-sided error rate.

    Returns:
        `pass` or `fail` from at least the rule's minimum trials, otherwise
        `not measured`; and the text, with the further trials needed.
    """
    hits, trials = deciding["hits"], deciding["trials"]
    held = f"{DECIDING} split: {hits}/{trials} without a model"
    if trials >= rate.minimum(alpha):
        passed = rate.passes(hits, trials, alpha)
        return (
            "pass" if passed else "fail",
            f"rate {'passes' if passed else 'fails'} ({held})",
        )
    more = needed(rate, hits, trials, alpha)
    return "not measured", (
        f"rate not measured ({held}): {more} more real {DECIDING} trials "
        f"that return a model needed (minimum {rate.minimum(alpha)})"
    )


def summary(document: dict[str, Any] | None = None) -> tuple[str, str]:
    """The scorecard value and verdict of claim C14.

    The rate is decided on the real validation-split trials only (`decide`);
    the others are reported beside it. The verdict is `pass` only when the
    rate and the precision both pass on at least their rules' minimum
    samples, `fail` when either is decided and fails, and `not measured`
    otherwise.

    Args:
        document: The stored document; read from `PATH` when None.

    Returns:
        The value text and the verdict.

    Raises:
        InputError: If the document lacks the counts or the deciding counts.
    """
    if document is None:
        if not PATH.exists():
            return "no refusal measurements", "not measured"
        document = json.loads(PATH.read_text(encoding="utf-8"))
    counts = document.get("counts")
    if not isinstance(counts, dict) or "real" not in counts:
        raise errors.InputError("refusal document has no counts")
    variables = acceptance.load()
    alpha = variables["alpha"].value
    rate = variables["refusal_rate_max"].rule
    floor = variables["refusal_precision_min"].rule
    if not isinstance(rate, acceptance.Ceiling) or not isinstance(
        floor, acceptance.Floor
    ):
        raise errors.InputError(
            "refusal_rate_max must be a ceiling and refusal_precision_min "
            "a floor"
        )
    real, simulated = counts["real"], counts["simulated"]
    precision = counts["precision"]
    verdicts = []
    if "deciding" not in real:
        raise errors.InputError("refusal document has no deciding counts")
    verdict, rate_text = decide(rate, real["deciding"], alpha)
    verdicts.append(verdict)
    if precision["refusals"] >= floor.minimum(alpha):
        passed = floor.passes(
            precision["justified"], precision["refusals"], alpha
        )
        verdicts.append("pass" if passed else "fail")
        precision_text = f"precision {'passes' if passed else 'fails'}"
    else:
        verdicts.append("not measured")
        precision_text = (
            f"precision not measured ({precision['refusals']} of "
            f"{floor.minimum(alpha)} refusals)"
        )
    gates = dict(real["gates"])
    for gate, count in simulated["gates"].items():
        gates[gate] = gates.get(gate, 0) + count
    planted = counts["planted"]
    text = (
        f"real: {real['refusals']}/{real['trials']} refused, "
        f"{real['failures']} failed; simulated (not the real criterion): "
        f"{simulated['refusals']}/{simulated['trials']} refused, "
        f"{simulated['failures']} failed; in-envelope refusals by gate "
        f"{gates or 'none'}; planted: {planted['refusals']}/"
        f"{planted['trials']} refused, {precision['justified']}/"
        f"{precision['refusals']} refusals justified (automated rubric, "
        f"not validator-reviewed); {rate_text}; {precision_text}; budget "
        f"{document['settings']['budget']} s"
    )
    if "fail" in verdicts:
        return text, "fail"
    if all(v == "pass" for v in verdicts):
        return text, "pass"
    return text, "not measured"


def problems(document: dict[str, Any] | None = None) -> list[str]:
    """Why the stored measurements are not for the current code.

    Args:
        document: The measurements; read from `PATH` when None.

    Returns:
        A message for a missing file, a code hash that differs from the
        current code, or tasks the current catalog does not have; empty
        when current.
    """
    if document is None:
        if not PATH.exists():
            return ["refusal.json is missing"]
        document = json.loads(PATH.read_text(encoding="utf-8"))
    found = []
    if document.get("code_hash") != fingerprint():
        found.append("refusal: measurements are for other code")
    unknown = set(document.get("tasks", {})) ^ set(catalog())
    if unknown:
        found.append(f"refusal: task catalog differs: {sorted(unknown)}")
    return found


def main(arguments: list[str] | None = None) -> int:
    """Writes (`--write`) or checks (`--check`) the measurements.

    `--task NAME --out FILE` runs one task and writes its record; it is how
    `measure` runs each task in its own process.

    Args:
        arguments: Command-line arguments; None reads `sys.argv`.

    Returns:
        0 for `--write` and `--task`; for `--check`, 0 when the stored
        measurements are for the current code and 1 otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--task")
    parser.add_argument("--out", type=pathlib.Path)
    parser.add_argument("--seconds", type=float, default=SECONDS)
    parser.add_argument("--parallel", type=int, default=PARALLEL)
    parser.add_argument("--budget", type=int, default=BUDGET)
    options = parser.parse_args(arguments)
    if options.task:
        if options.out is None:
            raise errors.InputError("--task needs --out")
        logging.disable(logging.WARNING)
        warnings.filterwarnings("ignore")
        record = attempt(catalog()[options.task], options.budget)
        options.out.write_text(json.dumps(record), encoding="utf-8")
        return 0
    if options.write:
        document = measure(options.seconds, options.parallel, options.budget)
        PATH.write_text(
            json.dumps(document, indent=2, sort_keys=True, default=float)
            + "\n",
            encoding="utf-8",
        )
        value, verdict = summary(document)
        print(json.dumps({"value": value, "verdict": verdict}, indent=2))
        return 0
    issues = problems()
    print(json.dumps({"ok": not issues, "problems": issues}, indent=2))
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
