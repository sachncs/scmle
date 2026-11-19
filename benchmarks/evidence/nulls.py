"""Claim C3, null clause: the product rarely claims skill where none exists.

A null task is a table whose target is independent of its features, so no
model can beat the trivial one on fresh rows. Each task is run end to end
through the public `scmle.api.project.Project`, with its defaults except the
declared task (`classification` or `regression`), `time_budget=BUDGET` and
the task's seed, and the run's own outputs are read for a claim of skill.

Two populations (docs/preregistration/c3.md):

* `real`: every pinned table of `benchmarks.evidence.quality` (OpenML,
  SHA-256 pinned; the sealed hold-out is not among them), rows without a
  label removed, the target permuted by a seeded generator. The features
  keep every real-world quirk (types, missing values, duplicates, width);
  only the link to the target is cut. A permutation spends nothing of the
  real target's evidence, so custody of those tables is untouched.
* `simulated`: `WORLDS` x `KINDS` x `ROWS`, the target drawn from a
  generator of its own, so it is independent of the features by
  construction (the same seed gives the same target in every world).

Seeds come from the registered validation stream `STREAM`, never drawn
before this module.

The product's skill claim. A run says its model beats its baseline (the
`dummy` model, the prior or the mean, which every run trains) in either of
its two interval reports or in its own fit verdict, so a hit is any of:

* `cv`: the selection-corrected 95 percent interval of the selected
  model's cross-validation score (`evaluation.json`, `confidence`) lies
  wholly on the better side of the baseline's cross-validation score
  (`baseline`, same file);
* `test`: the 95 percent interval of the test metric
  (`diagnostics/metric_uncertainty.json`, `ci_95`) lies wholly on the
  better side of the baseline's score on the same test rows: the product's
  `dummy` model fitted on the run's development rows and scored with the
  run's metric on its test rows (`validation_manifest.json`);
* `health`: the product's fit verdict (`evaluation.json`, `health`,
  from `scmle.evaluation.fit_health`) is `healthy`, which it gives only
  when the validation gain over the baseline is clear and carries over from
  training. `overfit` and `unstable` are not claims: the product says the
  gain does not survive validation, or is too noisy to rank, so it does
  not assert skill on fresh rows; `underfit` says nothing beats the
  baseline.

The direction (`evaluation.json`, `direction`) must be `higher` or
`lower`; any other value is refused rather than read as one of them.

A run that crashes counts as a hit: a crash cannot show that no claim was
made, and the product must not pass by failing. A refusal (`BlockedError`,
`LeakageError`) packages no model, so it claims nothing; refusals are
counted and reported.

Decision: the one-sided Clopper-Pearson upper bound at `alpha` of hits over
tasks, pooled over both populations, at most `null_skill_max`
(`acceptance.toml`, the `Ceiling` rule of `benchmarks.acceptance`). Each
population is reported beside it and decides nothing.

Sample size (`required`). The ceiling can pass at all only from
`ceil(ln alpha / ln(1 - 0.05))` = 59 tasks. A size gate of nominal level L
uses the ceiling L (1 + `size_margin`) so that a test of true size L passes
with chance `power` (`acceptance.margin`). Here the ceiling is the bound
itself, so the same margin read the other way gives the rate a product must
have to pass with chance `power`: `null_skill_max / (1 + size_margin)` =
0.0296. The planned count is the smallest whose exact chance of passing at
that rate reaches `power`, `Ceiling(0.05).plan(0.0296, 0.05, 0.8)` = 577,
and never fewer than `null_tasks_min` (200). Half the planned tasks
(`SHARE`) are real, as whole rounds of permutations of every table, and
the simulated tasks fill the rest.

NOT measured: data kinds other than tabular; task kinds other than binary,
multiclass and regression; the default 1800 s budget; declared success
criteria or a charter; null tasks with entities or time order.

    python -m benchmarks.evidence.nulls --write
    python -m benchmarks.evidence.nulls --check
"""

import abc
import argparse
from concurrent import futures
import dataclasses
import functools
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback
from typing import Any
from typing import ClassVar

import numpy
import pandas

from benchmarks import acceptance
from benchmarks import evidence
from benchmarks.evidence import quality
import scmle.api.project
from scmle.assurance.detectors import certificates
from scmle.contracts import errors
from scmle.contracts import tasks
from scmle.evaluation import metrics
from scmle.models import registry

PATH = Path(__file__).with_suffix(".json")
PACKAGE = Path(__file__).resolve().parents[2] / "scmle"
VARIABLE = "null_skill_max"
COUNT = "null_tasks_min"
MARGIN = "size_margin"
STREAM = "coverage.null.validation"
POPULATIONS = ("real", "simulated")
KINDS = ("binary", "multiclass", "regression")
WORLDS = ("gaussian", "mixed", "wide")
ROWS = (300, 1000, 3000)
STATUSES = frozenset({"claimed", "silent", "refused", "failed"})
# Outcomes that count against the product: a claim, and a crash that
# cannot show no claim was made.
HITS = frozenset({"claimed", "failed"})
TARGET = "target"
# Seconds per run, the C4 benchmark's budget (`quality.BUDGET`, kept equal
# by a test): a 300-row table finishes in about 20 s, so the budget bounds
# only the largest tables.
BUDGET = 60
# Share of the planned tasks drawn from real tables: half, so each
# population on its own is over four times the 59 tasks the ceiling needs.
SHARE = 0.5
# Simulated worlds. Class shares are unequal, as in practice; a balanced
# target would hide a model that wins by predicting the majority.
PREVALENCE = 0.3
SHARES = (0.4, 0.3, 0.2, 0.1)
# Student t degrees of freedom of a regression target: heavy tails, finite
# variance, so RMSE is defined.
TAILS = 5
NUMERIC = 6
WIDE = 60
LEVELS = (3, 12)
MISSING = 0.1
REFUSALS = (errors.BlockedError, errors.LeakageError)
DIRECTIONS = frozenset({"higher", "lower"})
# The fit verdict that asserts the model beats the baseline on validation.
HEALTHY = "healthy"
READINGS = ("cv", "test", "health")
NOT_MEASURED = (
    "data kinds other than tabular",
    "task kinds other than binary, multiclass and regression",
    "the default 1800 s budget",
    "declared success criteria or an analysis charter",
    "null tasks with entities or time order",
)


def seed(population: str, index: int) -> int:
    """A task's seed from the registered validation stream.

    Args:
        population: `real` or `simulated`.
        index: The task's position within its population.

    Returns:
        A 31-bit seed from SHA-256 of `STREAM`, the population and index.
    """
    key = f"{STREAM}:{population}:{index}".encode()
    return int(hashlib.sha256(key).hexdigest(), 16) % 2**31


@dataclasses.dataclass(frozen=True)
class Null(abc.ABC):
    """One null task: a table whose target the features cannot predict.

    Attributes:
        name: Unique name.
        population: `real` or `simulated`.
        source: The OpenML task id of a real table, or the world.
        kind: `binary`, `multiclass` or `regression`.
        seed: Seeds the draw and the product's run.
    """

    name: str
    population: str
    source: int | str
    kind: str
    seed: int

    @abc.abstractmethod
    def draw(self) -> pandas.DataFrame:
        """The table, with the target in `self.target()`.

        Returns:
            Features and target.
        """

    @abc.abstractmethod
    def target(self) -> str:
        """The target column.

        Returns:
            Its name.
        """


@dataclasses.dataclass(frozen=True)
class Permuted(Null):
    """A real table with its target permuted."""

    @staticmethod
    @functools.cache
    def original(table: int) -> pandas.DataFrame:
        """A pinned table of the C4 corpus, every labelled row.

        Args:
            table: OpenML task id in `quality.TASKS`.

        Returns:
            Training and test rows together, ignored columns removed, rows
            without a label removed, a class target as text; the target's
            name in `attrs["target"]`.
        """
        meta = json.loads((quality.DATA / f"{table}.json").read_text("utf-8"))
        frame = pandas.read_parquet(quality.DATA / f"{table}.pq")
        frame = frame.drop(columns=[c for c in meta["ignore"] if c in frame])
        name = meta["target"]
        frame = frame[frame[name].notna()].reset_index(drop=True)
        if quality.kind(table) == "regression":
            frame[name] = frame[name].astype(float)
        else:
            frame[name] = frame[name].astype(str)
        frame.attrs["target"] = name
        found: pandas.DataFrame = frame
        return found

    def target(self) -> str:
        """The table's target column.

        Returns:
            Its name.
        """
        return str(self.original(int(self.source)).attrs["target"])

    def draw(self) -> pandas.DataFrame:
        """The table with its target permuted.

        Returns:
            A copy whose target is a seeded permutation of the original.
        """
        frame = self.original(int(self.source)).copy()
        name = self.target()
        order = numpy.random.default_rng(self.seed).permutation(len(frame))
        frame[name] = frame[name].to_numpy()[order]
        return frame


@dataclasses.dataclass(frozen=True)
class Simulated(Null):
    """A simulated world whose target ignores the features.

    Attributes:
        rows: Rows drawn.
    """

    rows: int = ROWS[0]

    labels: ClassVar[tuple[str, ...]] = ("a", "b", "c", "d")

    def target(self) -> str:
        """The target column.

        Returns:
            `TARGET`.
        """
        return TARGET

    def draw(self) -> pandas.DataFrame:
        """Features from the world and a target from its own generator.

        Returns:
            The table.

        Raises:
            InputError: For an unknown world or kind.
        """
        if self.source not in WORLDS:
            raise errors.InputError(f"unknown world {self.source!r}")
        if self.kind not in KINDS:
            raise errors.InputError(f"unknown kind {self.kind!r}")
        frame = self.features(numpy.random.default_rng([self.seed, 0]))
        frame[TARGET] = self.outcome(numpy.random.default_rng([self.seed, 1]))
        return frame

    def features(self, rng: numpy.random.Generator) -> pandas.DataFrame:
        """The world's features.

        Args:
            rng: The features' generator.

        Returns:
            `gaussian`: `NUMERIC` correlated normal columns; `mixed`: those
            plus categorical columns of `LEVELS` levels, with `MISSING` of
            every column missing; `wide`: `WIDE` normal columns.
        """
        width = WIDE if self.source == "wide" else NUMERIC
        values = rng.standard_normal((self.rows, width))
        values[:, 1:] += 0.5 * values[:, :1]
        frame = pandas.DataFrame(
            values, columns=[f"x{i}" for i in range(width)]
        )
        if self.source != "mixed":
            return frame
        for level in LEVELS:
            codes = rng.integers(0, level, self.rows)
            frame[f"c{level}"] = numpy.array([f"v{c}" for c in codes], object)
        for column in frame.columns:
            frame.loc[rng.uniform(size=self.rows) < MISSING, column] = None
        return frame

    def outcome(self, rng: numpy.random.Generator) -> pandas.Series:
        """The target, drawn without reference to the features.

        Args:
            rng: The target's generator.

        Returns:
            `binary`: `a` or `b` with `PREVALENCE` of `b`; `multiclass`:
            four classes with `SHARES`; `regression`: Student t with
            `TAILS` degrees of freedom.
        """
        if self.kind == "regression":
            return pandas.Series(rng.standard_t(TAILS, self.rows), dtype=float)
        shares = (
            (1 - PREVALENCE, PREVALENCE) if self.kind == "binary" else SHARES
        )
        picked = rng.choice(len(shares), self.rows, p=shares)
        return pandas.Series([self.labels[i] for i in picked], dtype=object)


@functools.cache
def required() -> tuple[int, int]:
    """Null tasks needed, from acceptance.toml.

    Returns:
        The smallest count that can pass at all, and the planned count:
        the smallest whose exact chance of passing reaches `power` at the
        rate `assumed()`, and never fewer than `null_tasks_min`.
    """
    variables = acceptance.load()
    alpha, power = variables["alpha"].value, variables["power"].value
    rule = acceptance.ceiling(VARIABLE, variables)
    planned = rule.plan(assumed(), alpha, power)
    return rule.minimum(alpha), max(planned, int(variables[COUNT].value))


def assumed() -> float:
    """The skill-claim rate at which the row should pass with `power`.

    Returns:
        `null_skill_max / (1 + size_margin)`.
    """
    variables = acceptance.load()
    return variables[VARIABLE].value / (1 + variables[MARGIN].value)


def catalog() -> list[Null]:
    """Every null task, in a fixed order.

    Returns:
        Whole rounds of permutations of every real table, enough for
        `SHARE` of the planned count, then simulated tasks cycling through
        every world, kind and size until the planned count.
    """
    planned = required()[1]
    tables = [t for ids in quality.TASKS.values() for t in ids]
    rounds = math.ceil(planned * SHARE / len(tables))
    found: list[Null] = []
    for index in range(rounds * len(tables)):
        table = tables[index % len(tables)]
        found.append(
            Permuted(
                f"real/{table}/{index // len(tables)}",
                "real",
                table,
                quality.kind(table),
                seed("real", index),
            )
        )
    cells = [(w, k, r) for r in ROWS for k in KINDS for w in WORLDS]
    for index in range(planned - len(found)):
        world, kind, rows = cells[index % len(cells)]
        found.append(
            Simulated(
                f"simulated/{world}/{kind}/{rows}/{index}",
                "simulated",
                world,
                kind,
                seed("simulated", index),
                rows,
            )
        )
    return found


def better(
    interval: list[float] | None, anchor: float | None, higher: bool
) -> bool | None:
    """Whether an interval lies wholly on the better side of the baseline.

    Args:
        interval: Lower and upper end, or None when none was reported.
        anchor: The baseline's score, or None when it has none.
        higher: Whether larger scores are better.

    Returns:
        True when the whole interval is strictly better than `anchor`,
        False when it is not, None when either is missing or not finite.
    """
    if interval is None or anchor is None:
        return None
    values = [float(v) for v in interval]
    if not all(math.isfinite(v) for v in [*values, float(anchor)]):
        return None
    return values[0] > anchor if higher else values[1] < anchor


def judge(
    evaluation: dict[str, Any],
    uncertainty: dict[str, Any],
    anchor: float | None,
) -> dict[str, bool | None]:
    """Reads each of a run's skill claims.

    Args:
        evaluation: The run's `evaluation.json` record.
        uncertainty: The run's `metric_uncertainty` report.
        anchor: The baseline's test score (`anchor`).

    Returns:
        `cv` and `test`, each `better` of the interval against the baseline,
        and `health`, whether the fit verdict is `HEALTHY` (None when the
        run reports none).

    Raises:
        InputError: When the direction is neither `higher` nor `lower`.
    """
    direction = evaluation.get("direction")
    if direction not in DIRECTIONS:
        raise errors.InputError(f"unknown direction {direction!r}")
    higher = direction == "higher"
    health = evaluation.get("health") or {}
    verdict = health.get("verdict")
    confidence = evaluation.get("confidence") or {}
    interval = (
        confidence.get("interval") if confidence.get("available") else None
    )
    return {
        "cv": better(interval, evaluation.get("baseline"), higher),
        "test": better(uncertainty.get("ci_95"), anchor, higher),
        "health": None if verdict is None else verdict == HEALTHY,
    }


def anchor(
    frame: pandas.DataFrame,
    target: str,
    split: dict[str, Any],
    metric: str,
    task: tasks.Task,
) -> float:
    """The product's baseline scored on the product's test rows.

    Args:
        frame: The table the run was given.
        target: Its target column.
        split: The run's `validation_manifest.json`, whose `training` and
            `test` hold row positions.
        metric: The run's metric.
        task: The run's task.

    Returns:
        The `dummy` model fitted on the development rows, scored with the
        metric on the test rows; NaN when the metric is undefined there.

    Raises:
        InputError: If the split's rows are not exactly the table's rows.
    """
    training = numpy.asarray(split["training"], dtype=numpy.intp)
    test = numpy.asarray(split["test"], dtype=numpy.intp)
    rows = numpy.sort(numpy.concatenate([training, test]))
    if not numpy.array_equal(rows, numpy.arange(len(frame))):
        raise errors.InputError(
            f"the split covers {len(training) + len(test)} rows, not the "
            f"table's {len(frame)}"
        )
    model = registry.Catalog.stock().get("dummy").build(task, 0)
    features = frame.drop(columns=[target])
    model.fit(features.iloc[training], frame[target].iloc[training])
    classes = getattr(model, "classes_", None)
    positive = (
        classes[1]
        if task is tasks.Task.BINARY and classes is not None
        else None
    )
    try:
        return float(
            metrics.CATALOG.resolve(metric, task).score(
                model, features.iloc[test], frame[target].iloc[test], positive
            )
        )
    except ValueError:
        return math.nan


def run(task: Null, budget: int = BUDGET) -> dict[str, Any]:
    """One null task, end to end; a crash is recorded.

    Args:
        task: The task.
        budget: The run's time budget in seconds.

    Returns:
        The task, its status (`claimed`, `silent`, `refused` or `failed`),
        both readings, the metric, the baseline's scores, both intervals,
        the error text and the seconds taken.
    """
    started = time.monotonic()
    found: dict[str, Any] = {
        "name": task.name,
        "population": task.population,
        "source": task.source,
        "kind": task.kind,
        "seed": task.seed,
        "readings": dict.fromkeys(READINGS),
        "metric": None,
        "anchor": None,
        "baseline": None,
        "cv_interval": None,
        "test_interval": None,
        "error": "",
    }
    with tempfile.TemporaryDirectory() as folder:
        try:
            frame = task.draw()
            result = scmle.api.project.Project(
                frame,
                task.target(),
                task="regression"
                if task.kind == "regression"
                else "classification",
                time_budget=budget,
                seed=task.seed,
                output_dir=Path(folder) / "run",
            ).run()
            evaluation = result.evaluation
            split = json.loads(
                (Path(result.run_dir) / "validation_manifest.json").read_text(
                    "utf-8"
                )
            )
            uncertainty = result.diagnostics.get("metric_uncertainty", {})
            found["metric"] = evaluation["metric"]
            found["anchor"] = anchor(
                frame,
                task.target(),
                split,
                evaluation["metric"],
                tasks.Task(evaluation["task"]),
            )
            found["baseline"] = evaluation.get("baseline")
            found["cv_interval"] = (evaluation.get("confidence") or {}).get(
                "interval"
            )
            found["test_interval"] = uncertainty.get("ci_95")
            found["readings"] = judge(evaluation, uncertainty, found["anchor"])
            found["status"] = (
                "claimed" if any(found["readings"].values()) else "silent"
            )
        except REFUSALS:
            found["status"] = "refused"
            found["error"] = traceback.format_exc(limit=4)
        except Exception:  # noqa: BLE001 -- a crash counts as a claim
            found["status"] = "failed"
            found["error"] = traceback.format_exc(limit=4)
    found["seconds"] = round(time.monotonic() - started, 1)
    return found


def tally(name: str, outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    """Counts outcomes and decides them.

    Args:
        name: The population, or `pooled`.
        outcomes: What `run` returned for each task.

    Returns:
        Hits (claims and crashes), tasks, the count of every status, the
        rate, the one-sided upper bound at `alpha` and whether the
        `null_skill_max` ceiling is met.

    Raises:
        InputError: For an unknown status.
    """
    unknown = {o["status"] for o in outcomes} - STATUSES
    if unknown:
        raise errors.InputError(f"unknown status {sorted(unknown)}")
    alpha = acceptance.load()["alpha"].value
    rule = acceptance.ceiling(VARIABLE)
    hits = sum(o["status"] in HITS for o in outcomes)
    trials = len(outcomes)
    counts = {
        s: sum(o["status"] == s for o in outcomes) for s in sorted(STATUSES)
    }
    return {
        "population": name,
        "hits": hits,
        "trials": trials,
        **counts,
        "rate": hits / trials if trials else None,
        "upper": rule.limit(hits, trials, alpha),
        "passed": rule.passes(hits, trials, alpha),
    }


def digest() -> str:
    """Hash of the code measured and the code measuring it.

    Returns:
        The combined hash of this module, of `quality` (which tables make up
        the catalog and how they load), of `acceptance` (how the thresholds
        are read and decided) and of every Python file of every product
        subpackage: a null task runs the whole product.
    """
    own = certificates.fingerprint(sys.modules[__name__], quality, acceptance)
    names = tuple(
        sorted(
            p.name
            for p in PACKAGE.iterdir()
            if p.is_dir() and (p / "__init__.py").exists()
        )
    )
    product = evidence.source(names, PACKAGE)
    return hashlib.sha256(f"{own}{product}".encode()).hexdigest()


def measure(workers: int | None = None) -> dict[str, Any]:
    """Runs every null task and builds the stored document.

    Args:
        workers: Concurrent tasks; None uses half the cores, since a run
            uses more than one.

    Returns:
        The JSON-ready document: both populations, the pooled count that
        decides, the thresholds and sample sizes it was decided with, the
        design, every task's outcome, what was not measured, the code hash
        and the duration.
    """
    started = time.monotonic()
    os.environ["OMP_NUM_THREADS"] = "1"
    for table in {t for ids in quality.TASKS.values() for t in ids}:
        quality.fetch(table)
    order = catalog()
    count = workers or max(1, (os.cpu_count() or 2) // 2)
    with futures.ProcessPoolExecutor(count) as pool:
        done = list(pool.map(run, order))
    variables = acceptance.load()
    least, planned = required()
    return {
        "claim": "C3",
        "variable": VARIABLE,
        "bound": variables[VARIABLE].value,
        "alpha": variables["alpha"].value,
        "power": variables["power"].value,
        "margin": variables[MARGIN].value,
        "assumed": assumed(),
        "minimum_tasks": least,
        "planned_tasks": planned,
        "design": {
            "budget_seconds": BUDGET,
            "stream": STREAM,
            "share_real": SHARE,
            "worlds": list(WORLDS),
            "kinds": list(KINDS),
            "rows": list(ROWS),
            "claim": (
                "cv or test interval wholly better than the dummy, "
                "or a healthy fit verdict"
            ),
            "failed": "counted as a claim",
            "refused": "no model packaged, no claim",
        },
        "populations": [
            tally(p, [o for o in done if o["population"] == p])
            for p in POPULATIONS
        ],
        "pooled": tally("pooled", done),
        "tasks": done,
        "data": quality.files(),
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
        Messages for a missing file, code or data that changed, a bound,
        alpha, power or margin other than acceptance.toml's, or fewer tasks
        than planned; empty when the document decides.
    """
    found = read(document)
    if found is None:
        return [f"{PATH.name} is missing"]
    issues = []
    if found.get("code") != digest():
        issues.append("nulls: measured with other code")
    if found.get("data") != quality.files():
        issues.append("nulls: measured on other data")
    variables = acceptance.load()
    for key, name in (
        ("bound", VARIABLE),
        ("alpha", "alpha"),
        ("power", "power"),
        ("margin", MARGIN),
    ):
        value = variables[name].value
        if found.get(key) != value:
            issues.append(f"nulls: {key} {found.get(key)} is not {value}")
    least, planned = required()
    trials = (found.get("pooled") or {}).get("trials", 0)
    if trials < least:
        issues.append(f"nulls: {trials} tasks; none can pass below {least}")
    elif trials < planned:
        issues.append(
            f"nulls: {trials} tasks, fewer than the {planned} planned"
        )
    return issues


def summary(document: dict[str, Any] | None = None) -> tuple[str, str]:
    """The scorecard's value and verdict for the clause.

    Args:
        document: The document; read from `PATH` when None.

    Returns:
        The value text (pooled hits over tasks, the upper bound, failures
        and refusals, and each population) and `pass` or `fail` by the
        ceiling on the pooled count; `not measured` when nothing is stored
        or `problems` finds any.
    """
    found = read(document)
    if found is None or not found.get("pooled"):
        return "no measurement", "not measured"
    alpha = acceptance.load()["alpha"].value
    rule = acceptance.ceiling(VARIABLE)
    pooled = found["pooled"]
    upper = rule.limit(pooled["hits"], pooled["trials"], alpha)
    parts = ", ".join(
        f"{p['population']} {p['hits']}/{p['trials']}"
        for p in found.get("populations", [])
    )
    text = (
        f"{pooled['hits']}/{pooled['trials']} null tasks claimed skill "
        f"(failures {pooled.get('failed', 0)} counted as claims, "
        f"{pooled.get('refused', 0)} refused), upper bound {upper:.4f}; "
        f"{parts}"
    )
    if problems(found):
        return text, "not measured"
    passed = rule.passes(pooled["hits"], pooled["trials"], alpha)
    return text, "pass" if passed else "fail"


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
    parser.add_argument("--workers", type=int, default=None)
    options = parser.parse_args(arguments)
    if options.write:
        document = measure(options.workers)
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
