"""Claim C3: the 95 percent metric intervals cover fresh-data metrics.

The product reports, for the primary test metric, a 95 percent bootstrap
percentile interval (`scmle.evaluation.bootstrap.Uncertainty`, the
`metric_uncertainty` report). This module checks that interval against
known truth on simulated tasks. One trial:

1. draws training rows from a known world and fits the product's pipeline
   (`preprocessing.instantiate` with a stock model family);
2. draws `TEST_ROWS` test rows and asks `Uncertainty.compute` for the
   interval, exactly as the evaluation stage does;
3. scores the same fitted model on `FRESH_ROWS` fresh rows from the same
   world with the same metric; that is the fresh-data metric, the truth;
4. counts a hit when the interval contains it.

Families, each with its default metric and resampling unit:

* `binary_rows`: logistic world, `logistic_regression`, ROC AUC, rows.
* `regression_rows`: linear world with Gaussian noise, `ridge`, RMSE, rows.
* `binary_groups`: logistic world with a random entity intercept, test rows
  in `GROUPS` entities of `GROUP_ROWS` rows, ROC AUC, whole groups.

The row of the scorecard passes when, for every family, the one-sided
Clopper-Pearson lower bound at `alpha` of hits over trials is at least
`interval_coverage_min` (`acceptance.toml`, the `Floor` rule of
`benchmarks.acceptance`). The worst family decides; the pooled count is
reported beside it and decides nothing.

NOT measured: multiclass log loss, temporal (block) resampling, other
metrics than each task's default, other model families, real data.

    python -m benchmarks.evidence.coverage --write
    python -m benchmarks.evidence.coverage --check
"""

import argparse
from concurrent import futures
import functools
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy
import pandas
import sklearn

from benchmarks import acceptance
from scmle.contracts import data
from scmle.contracts import errors
from scmle.contracts import tasks
from scmle.data.curation import preprocessing
from scmle.data.detectors import certificates
from scmle.evaluation import bootstrap
from scmle.evaluation import diagnostics
from scmle.evaluation import metrics
import scmle.evaluation.coverage
from scmle.models import registry

PATH = Path(__file__).with_suffix(".json")
VARIABLE = "interval_coverage_min"
# Trials per family. The floor 0.93 at alpha 0.05 can pass at all only from
# ceil(ln(0.05) / ln(0.93)) = 42 trials, every one a hit. When the interval
# is exactly nominal (true coverage 0.95), the smallest n whose exact chance
# of passing reaches power 0.80 is `Floor(0.93).plan(0.95, 0.05, 0.80)` =
# 882. The worst family decides, so every family runs that many;
# `problems` recomputes both numbers from acceptance.toml and refuses fewer.
TRIALS = 882
# Assumed true coverage for the planned sample: the nominal level.
NOMINAL = 0.95
TRAIN_ROWS = 400
TEST_ROWS = 300
# Rows behind the truth. Its standard error (about 0.0015 for an AUC near
# 0.8, 0.002 relative for RMSE) is a thirtieth of the interval's half-width
# at 300 test rows, so a miss is the interval's, not the truth's.
FRESH_ROWS = 100_000
GROUPS = 60
GROUP_ROWS = 5
# standard deviation of the entity intercept on the logit scale
GROUP_SPREAD = 1.0
FEATURES = 5
NOISE = 1.0
FAMILIES = ("binary_rows", "regression_rows", "binary_groups")
NOT_MEASURED = (
    "multiclass log loss",
    "temporal (block) resampling",
    "metrics other than each task's default",
    "model families other than logistic regression and ridge",
    "real datasets",
)
PLAN = data.Blueprint(tuple(f"x{i}" for i in range(FEATURES)), ())


def simulate(
    family: str, rng: numpy.random.Generator, rows: int
) -> tuple[pandas.DataFrame, pandas.Series, pandas.Series | None]:
    """Draws rows from a family's world.

    Args:
        family: One of `FAMILIES`.
        rng: Random generator; the world's coefficients are fixed, only the
            rows are drawn from it.
        rows: Rows to draw; for `binary_groups`, a multiple of
            `GROUP_ROWS`.

    Returns:
        Features, target and, for `binary_groups`, the entity of each row
        (None otherwise).

    Raises:
        InputError: For an unknown family.
    """
    if family not in FAMILIES:
        raise errors.InputError(f"unknown family {family!r}")
    weights = numpy.linspace(1.0, 0.2, FEATURES)
    features = rng.standard_normal((rows, FEATURES))
    frame = pandas.DataFrame(features, columns=list(PLAN.numeric))
    signal = features @ weights
    if family == "regression_rows":
        values = signal + NOISE * rng.standard_normal(rows)
        return frame, pandas.Series(values), None
    entities = None
    if family == "binary_groups":
        count = rows // GROUP_ROWS
        entities = pandas.Series(numpy.repeat(numpy.arange(count), GROUP_ROWS))
        signal = signal + numpy.repeat(
            GROUP_SPREAD * rng.standard_normal(count), GROUP_ROWS
        )
    labels = rng.uniform(size=rows) < 1.0 / (1.0 + numpy.exp(-signal))
    return frame, pandas.Series(labels.astype(int)), entities


def run(family: str, trial: int) -> dict[str, Any]:
    """One trial: fit, report the interval, score fresh rows.

    Args:
        family: One of `FAMILIES`.
        trial: Trial number; seeds `default_rng([trial, family index])`.

    Returns:
        The interval (None if the product gave none), the truth and whether
        the interval contains it.
    """
    rng = numpy.random.default_rng([trial, FAMILIES.index(family)])
    task = (
        tasks.Task.REGRESSION
        if family == "regression_rows"
        else tasks.Task.BINARY
    )
    adapter = registry.Catalog.stock().get(
        "ridge" if task is tasks.Task.REGRESSION else "logistic_regression"
    )
    features, target, _ = simulate(family, rng, TRAIN_ROWS)
    model = preprocessing.instantiate(PLAN, adapter, task, trial)
    model.fit(features, target)
    rows = GROUPS * GROUP_ROWS if family == "binary_groups" else TEST_ROWS
    features, target, entities = simulate(family, rng, rows)
    metric = metrics.CATALOG.resolve(None, task)
    inputs = diagnostics.DiagnosticInput(
        model, features, target, task, metric, PLAN, trial, groups=entities
    )
    report = bootstrap.Uncertainty().compute(inputs)
    interval = report[f"ci_{bootstrap.Uncertainty.confidence}"]
    fresh, truth, _ = simulate(family, rng, FRESH_ROWS)
    value = float(metric.score(model, fresh, truth, inputs.positive_label))
    hit = interval is not None and interval[0] <= value <= interval[1]
    return {"interval": interval, "truth": value, "hit": bool(hit)}


def tally(family: str, outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    """Counts one family's trials and decides it.

    Args:
        family: One of `FAMILIES`.
        outcomes: What `run` returned for each trial.

    Returns:
        Hits, trials, misses below and above, trials without an interval,
        the Clopper-Pearson lower bound and whether the floor is met.
    """
    variables = acceptance.load()
    alpha = variables["alpha"].value
    hits = sum(o["hit"] for o in outcomes)
    missing = [o for o in outcomes if o["interval"] is None]
    present = [o for o in outcomes if o["interval"] is not None]
    return {
        "family": family,
        "hits": hits,
        "trials": len(outcomes),
        "rate": hits / len(outcomes) if outcomes else None,
        "below": sum(o["truth"] < o["interval"][0] for o in present),
        "above": sum(o["truth"] > o["interval"][1] for o in present),
        "no_interval": len(missing),
        "mean_width": float(
            numpy.mean([o["interval"][1] - o["interval"][0] for o in present])
        )
        if present
        else None,
        "lower": bound(hits, len(outcomes), alpha),
        "passed": rule().passes(hits, len(outcomes), alpha),
    }


def bound(hits: int, trials: int, alpha: float) -> float | None:
    """One-sided Clopper-Pearson lower bound at `alpha`.

    Args:
        hits: Intervals that covered.
        trials: Trials run.
        alpha: One-sided error rate.

    Returns:
        The lower bound, or None with no trials.
    """
    if trials == 0:
        return None
    return float(
        scmle.evaluation.coverage.clopper_pearson(hits, trials, 1 - 2 * alpha)[
            0
        ]
    )


def rule() -> acceptance.Floor:
    """The decision rule of the coverage row.

    Returns:
        The `Floor` of `interval_coverage_min`.

    Raises:
        InputError: If acceptance.toml gives the row another rule.
    """
    found = acceptance.load()[VARIABLE].rule
    if not isinstance(found, acceptance.Floor):
        raise errors.InputError(f"{VARIABLE} must have the floor rule")
    return found


@functools.cache
def required() -> tuple[int, int]:
    """Trials needed per family, from acceptance.toml.

    The exact power search takes seconds, so it runs once per process.

    Returns:
        The smallest count that can pass at all, and the planned count for
        `power` when the true coverage is `NOMINAL`.
    """
    variables = acceptance.load()
    alpha, power = variables["alpha"].value, variables["power"].value
    floor = rule()
    return floor.minimum(alpha), floor.plan(NOMINAL, alpha, power)


def digest() -> str:
    """Hash of the code measured and the code measuring it.

    Returns:
        The `code_hash` of the product's interval and this module's trial.
    """
    return certificates.code_hash(
        bootstrap.Uncertainty,
        bootstrap.redraw,
        bootstrap.Cache,
        scmle.evaluation.coverage.clopper_pearson,
        simulate,
        run,
        tally,
    )


def prepare() -> None:
    """Turns off scikit-learn's argument checks in a worker process.

    The checks dominate the cost of scoring 2,000 resamples and do not
    change a score (a trial gives the same interval and truth either way);
    the product's code still runs unchanged.
    """
    sklearn.set_config(skip_parameter_validation=True, assume_finite=True)


def measure(trials: int = TRIALS, workers: int | None = None) -> dict[str, Any]:
    """Runs every family and builds the stored document.

    Args:
        trials: Trials per family.
        workers: Worker processes; None uses every core.

    Returns:
        The JSON-ready document: the families, the pooled count, the
        threshold and sample sizes it was decided with, what was not
        measured, the code hash and the duration.
    """
    started = time.monotonic()
    variables = acceptance.load()
    least, planned = required()
    jobs = [(f, t) for f in FAMILIES for t in range(trials)]
    with futures.ProcessPoolExecutor(
        workers or os.cpu_count(), initializer=prepare
    ) as pool:
        done = list(pool.map(run, *zip(*jobs, strict=True), chunksize=8))
    families = [
        tally(f, [o for (g, _), o in zip(jobs, done, strict=True) if g == f])
        for f in FAMILIES
    ]
    hits = sum(f["hits"] for f in families)
    count = sum(f["trials"] for f in families)
    alpha = variables["alpha"].value
    return {
        "claim": "C3",
        "variable": VARIABLE,
        "floor": variables[VARIABLE].value,
        "alpha": alpha,
        "power": variables["power"].value,
        "level": bootstrap.Uncertainty.confidence / 100,
        "minimum_trials": least,
        "planned_trials": planned,
        "design": {
            "train_rows": TRAIN_ROWS,
            "test_rows": TEST_ROWS,
            "groups": GROUPS,
            "group_rows": GROUP_ROWS,
            "fresh_rows": FRESH_ROWS,
            "seeds": "default_rng([trial, family index])",
        },
        "families": families,
        "pooled": {
            "hits": hits,
            "trials": count,
            "rate": hits / count if count else None,
            "lower": bound(hits, count, alpha),
        },
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
        Messages for a missing file, code that changed, a threshold other
        than acceptance.toml's, a missing family, or a family with fewer
        trials than the planned sample; empty when the document decides.
    """
    found = read(document)
    if found is None:
        return [f"{PATH.name} is missing"]
    issues = []
    if found.get("code") != digest():
        issues.append("coverage: measured with other code")
    floor = acceptance.load()[VARIABLE].value
    if found.get("floor") != floor:
        issues.append(f"coverage: floor {found.get('floor')} is not {floor}")
    least, planned = required()
    families = {f.get("family"): f for f in found.get("families", [])}
    for name in FAMILIES:
        if name not in families:
            issues.append(f"coverage: family {name} not measured")
            continue
        trials = families[name].get("trials", 0)
        if trials < least:
            issues.append(
                f"coverage: {name} has {trials} trials; none can pass "
                f"below {least}"
            )
        elif trials < planned:
            issues.append(
                f"coverage: {name} has {trials} trials, fewer than the "
                f"{planned} planned"
            )
    return issues


def summary(document: dict[str, Any] | None = None) -> tuple[str, str]:
    """The scorecard's value and verdict for the row.

    Args:
        document: The document; read from `PATH` when None.

    Returns:
        The value text (the worst family's count and lower bound, and the
        pooled count) and `pass` or `fail` by the floor rule on the worst
        family; `not measured` when nothing is stored or `problems` finds
        any (stale code, a missing family, too few trials).
    """
    found = read(document)
    if found is None or not found.get("families"):
        return "no measurement", "not measured"
    alpha = acceptance.load()["alpha"].value
    worst = min(
        found["families"],
        key=lambda f: bound(f["hits"], f["trials"], alpha) or 0.0,
    )
    lower = bound(worst["hits"], worst["trials"], alpha) or 0.0
    pooled = found["pooled"]
    text = (
        f"worst family {worst['family']}: {worst['hits']}/{worst['trials']} "
        f"covered, lower bound {lower:.4f}; pooled "
        f"{pooled['hits']}/{pooled['trials']}"
    )
    if problems(found):
        return text, "not measured"
    passed = rule().passes(worst["hits"], worst["trials"], alpha)
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
