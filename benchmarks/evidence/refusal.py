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
  bundled sets and the OpenML cache), each with the task a practitioner would
  declare. No real survival, ranking, forecast or uplift table is cached, so
  those kinds are simulated only.
* `simulated`: clean generated tasks of every kind, several sizes and shapes
  (plain, mixed types with missing values, wide with noise columns).
* `planted`: tasks built to be refused (a constant or identifier target,
  missing labels, a class of one row, too few rows, a planted leak under
  `leakage_policy="block"`, malformed event, treatment or label columns, too
  few queries). Each names the gates that may refuse it.

Rate: in-envelope trials are the `real` and `simulated` runs that finished;
a refusal is a hit. The plan's criterion is about real datasets, so only the
`real` count decides the row; the simulated count is reported beside it.
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
from scmle.contracts import errors
from scmle.data.detectors import certificates

PATH = pathlib.Path(__file__).with_suffix(".json")
ROOT = pathlib.Path(__file__).resolve().parents[2]
GATES = ROOT / "gates.toml"
# Policy: seconds of training budget per run; short, as a quick first look.
BUDGET = 30
# Policy: wall clock for the whole measurement, about ten minutes.
SECONDS = 600
# Policy: a run is killed past this; 12 times the budget allows a loaded
# laptop (one run at a 60 s budget took 514 s with load average 60).
TIMEOUT = 360
PARALLEL = 4
GROUPS = ("real", "simulated", "planted")
DEAD = "http://127.0.0.1:9"
# Rows kept of a large real table, so a run fits the short budget.
ROWS = 4000
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


def fetch(identifier: int | str, rows: int = ROWS) -> pandas.DataFrame:
    """A real table from the local caches, without the network.

    Args:
        identifier: An OpenML data id, or the name of a scikit-learn
            `load_*` function such as `load_wine`.
        rows: Rows kept, drawn with a fixed seed, when the table is longer.

    Returns:
        The table, with categorical columns as text.

    Raises:
        InputError: If an OpenML table is not cached.
    """
    if isinstance(identifier, str):
        bunch = getattr(datasets, identifier)(as_frame=True)
        frame: pandas.DataFrame = bunch.frame.copy()
    else:
        if not cached(identifier):
            raise errors.InputError(f"OpenML {identifier} is not cached")
        bunch = datasets.fetch_openml(
            data_id=identifier, as_frame=True, parser="auto"
        )
        frame = bunch.frame.copy()
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
            "credit_g", "binary", "real", table(31, "class", "classification")
        ),
        Task(
            "pima_diabetes",
            "binary",
            "real",
            table(37, "class", "classification"),
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
        Task(
            "diamonds",
            "regression",
            "real",
            table(42225, "price", "regression"),
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
    layouts = (
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
    simulated = [
        Task(
            f"sim_{kind}_{rows}_{shape}",
            kind,
            "simulated",
            functools.partial(simulate, kind, rows, index, shape),
        )
        for index, (kind, rows, shape) in enumerate(layouts)
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
        by gate; and the precision counts: refusals, and those justified (a
        planted task refused by one of the gates it names).
    """
    counts: dict[str, Any] = {}
    for group in GROUPS:
        mine = [r for r in records if tasks[r["name"]].group == group]
        gates: dict[str, int] = {}
        for record in mine:
            if record["outcome"] == "refused":
                gates[record["gate"]] = gates.get(record["gate"], 0) + 1
        finished = [r for r in mine if r["outcome"] != "unfinished"]
        counts[group] = {
            "trials": len(finished),
            "models": sum(r["outcome"] == "model" for r in mine),
            "refusals": sum(r["outcome"] == "refused" for r in mine),
            "failures": sum(r["outcome"] == "failed" for r in mine),
            "unfinished": len(mine) - len(finished),
            "gates": dict(sorted(gates.items())),
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


def fingerprint() -> str:
    """Hash of the code that decides what is measured.

    Returns:
        The combined source hash.
    """
    return certificates.code_hash(
        Task,
        simulate,
        *LABELERS.values(),
        forecast,
        fetch,
        table,
        spoil,
        plant,
        catalog,
        locate,
        attempt,
        tally,
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


def summary(document: dict[str, Any] | None = None) -> tuple[str, str]:
    """The scorecard value and verdict of claim C14.

    The rate is decided on the real in-envelope trials only; the simulated
    ones are reported beside it. The verdict is `pass` only when the rate
    and the precision both pass on at least their rules' minimum samples,
    `fail` when either is decided and fails, and `not measured` otherwise.

    Args:
        document: The stored document; read from `PATH` when None.

    Returns:
        The value text and the verdict.

    Raises:
        InputError: If the document lacks the counts.
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
    if real["trials"] >= rate.minimum(alpha):
        passed = rate.passes(real["refusals"], real["trials"], alpha)
        verdicts.append("pass" if passed else "fail")
        rate_text = f"rate {'passes' if passed else 'fails'}"
    else:
        more = needed(rate, real["refusals"], real["trials"], alpha)
        verdicts.append("not measured")
        rate_text = (
            f"rate not measured: {more} more refusal-free real trials "
            f"needed (minimum {rate.minimum(alpha)})"
        )
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
