"""Claim C3, reference agreement: product statistics against references.

The criterion (docs/preregistration/c3.md): every reported statistic within
`reference_rel_tol` of a reference implementation, and every statistic that
decides a gate also matches a second implementation.

Each `Statistic` pairs a product computation with references written here:
`scratch` follows the textbook formula in plain Python or NumPy, `scipy`
and `sklearn` call a library function the product does not call for that
value. A statistic that decides a gate of `gates.toml` (`DECIDING`) needs
`SECOND` references, the product being the first implementation; every
other gate is a predicate on counts, sums, types or times (`PREDICATES`).
The reported inventory is the metric catalog's report names of every task
(`reported`); a reported name no comparison reaches is listed, not skipped.
Cases are real tables bundled with scikit-learn (`source: real`) and
simulated worlds; forecast, uplift and survival have no real table offline.

The error of a value is `|product - reference| / max(|reference|, 1)`,
relative above one and absolute below; two NaNs or two equal infinities
agree, and a key a reference does not produce has an infinite error.

Verdicts (`summary`, per scope `reported` or `deciding`): "fail" when any
comparison of the scope is beyond the tolerance; "not measured" when the
document is missing or stale or the scope's `residue` (what is not
compared, stated in `RESIDUE`) is not empty; "pass" otherwise.

    python -m benchmarks.evidence.statistics --show [--statistic NAME ...]
    python -m benchmarks.evidence.statistics --write
    python -m benchmarks.evidence.statistics --check
"""

import argparse
from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
from fractions import Fraction
import functools
import hashlib
import json
import math
import pathlib
import sys
import time
from typing import Any
from unittest import mock

import numpy
import pandas
from scipy import optimize
from scipy import special
from scipy import stats
from sklearn import base
from sklearn import datasets
from sklearn import linear_model
from sklearn import tree
import sklearn.metrics

from benchmarks import acceptance
from benchmarks import evidence
from scmle.assurance.admissibility import leakage
from scmle.assurance.admissibility import screen
from scmle.assurance.admissibility.leakage.settings import EXPECTED_FISHER
from scmle.assurance.admissibility.screen import fitting
from scmle.assurance.admissibility.screen import hypotheses
from scmle.assurance.admissibility.screen import planning
from scmle.assurance.admissibility.screen import report
from scmle.assurance.admissibility.screen import settings
from scmle.assurance.ledger import budget
from scmle.assurance.ledger import multiplicity
from scmle.assurance.ledger.confirm import thresholdout
from scmle.contracts import errors
from scmle.contracts import tasks
from scmle.data.splitting.splitters import partitions
from scmle.evaluation import coverage
from scmle.evaluation import forecast
from scmle.evaluation import metrics
from scmle.evaluation import resources
from scmle.evaluation import survival
from scmle.evaluation import uplift
from scmle.inference import onnx_export
from scmle.runtime import randomness

ROOT = pathlib.Path(__file__).resolve().parents[2]
PATH = pathlib.Path(__file__).with_suffix(".json")
GATES = ROOT / "gates.toml"
VARIABLE = "reference_rel_tol"
# The product is the first implementation of a gate statistic; the clause
# asks for a reference and a second one beside it.
SECOND = 2
ALPHA = Fraction(1, 20)
# Code whose change can change a compared value.
MEASURED = (
    "evaluation",
    "assurance",
    "inference",
    "data",
    "models",
    "contracts",
)
Values = dict[str, float]


@dataclasses.dataclass(frozen=True)
class Case:
    """One input a statistic is computed on.

    Attributes:
        name: What the input is.
        source: `real` for a real table, `simulated` for a drawn world.
        data: What the product and the references read.
    """

    name: str
    source: str
    data: Mapping[str, Any]


@dataclasses.dataclass(frozen=True)
class Statistic:
    """A product statistic and its independent references.

    Attributes:
        name: Key of the statistic in the document.
        cases: Builds the inputs.
        product: The product's values on one input, by key.
        references: Reference name to a function giving the same keys.
    """

    name: str
    cases: Callable[[], list[Case]]
    product: Callable[[Case], Values]
    references: Mapping[str, Callable[[Case], Values]]


@dataclasses.dataclass(frozen=True)
class Comparison:
    """One product value against every reference of it.

    Attributes:
        statistic: The statistic's name.
        case: The input's name.
        source: `real` or `simulated`.
        key: Which value of the statistic.
        deciding: Whether the statistic decides a gate.
        product: The product's value.
        references: Each reference's value.
        error: The largest error against a reference (`error`).
        within: Every error within the tolerance, with enough references.
    """

    statistic: str
    case: str
    source: str
    key: str
    deciding: bool
    product: float
    references: dict[str, float]
    error: float
    within: bool


def error(found: float, expected: float) -> float:
    """Relative error above one, absolute below.

    Args:
        found: The product's value.
        expected: The reference's value.

    Returns:
        `|found - expected| / max(|expected|, 1)`; 0 for two NaNs or equal
        values, infinity when only one is finite or NaN.
    """
    if (math.isnan(found) and math.isnan(expected)) or found == expected:
        return 0.0
    if not (math.isfinite(found) and math.isfinite(expected)):
        return math.inf
    return abs(found - expected) / max(abs(expected), 1.0)


def compare(
    statistic: Statistic, case: Case, tolerance: float, *, deciding: bool
) -> list[Comparison]:
    """Compares every value of the product with every reference.

    Args:
        statistic: What to compare.
        case: The input.
        tolerance: The largest error allowed.
        deciding: Whether `SECOND` references are needed.

    Returns:
        One comparison per product key, in the product's order.
    """
    found = statistic.product(case)
    expected = {n: f(case) for n, f in statistic.references.items()}
    out = []
    for key, value in found.items():
        refs = {n: float(v[key]) for n, v in expected.items() if key in v}
        worst = max((error(value, r) for r in refs.values()), default=math.inf)
        enough = len(refs) >= (SECOND if deciding else 1)
        out.append(
            Comparison(
                statistic.name,
                case.name,
                case.source,
                key,
                deciding,
                float(value),
                refs,
                worst,
                enough and worst <= tolerance,
            )
        )
    return out


class Held(base.RegressorMixin, base.BaseEstimator):  # type: ignore[misc]  # scikit-learn ships no types
    """The fixed outputs of a fitted model, read by the product's scorers.

    Attributes:
        guess: What `predict` returns.
        rates: Exponential hazard per row, for `predict_survival`.
        target_columns: Target columns, the first the scored one.
        value_column: Current value of a forecast.
        query_column: Query of a ranking row.
        treatment_column: Treatment arm of an uplift row.
        event_column: Event indicator of a survival row.
    """

    def __init__(
        self,
        guess: Any = None,
        rates: Any = None,
        target_columns: Any = (),
        value_column: str = "",
        query_column: str = "",
        treatment_column: str = "",
        event_column: str = "",
    ) -> None:
        """Holds the outputs; see the class attributes."""
        self.guess = guess
        self.rates = rates
        self.target_columns = target_columns
        self.value_column = value_column
        self.query_column = query_column
        self.treatment_column = treatment_column
        self.event_column = event_column

    def predict(self, features: pandas.DataFrame) -> numpy.ndarray:
        """The held predictions, one per row of `features`."""
        return numpy.asarray(self.guess)[: len(features)]

    def survival(self, times: numpy.ndarray) -> numpy.ndarray:
        """`exp(-rate t)` per row and time."""
        return numpy.exp(-numpy.outer(self.rates, times))

    def predict_survival(self, features: Any, times: Any) -> numpy.ndarray:
        """The scorer's name for `survival`."""
        del features
        return self.survival(numpy.asarray(times, dtype=float))


def wins(score: numpy.ndarray, positive: numpy.ndarray) -> tuple[float, float]:
    """Concordant positive-negative pairs, ties one half, and the pairs."""
    high, low = score[positive], score[~positive]
    above = (high[:, None] > low[None, :]).sum()
    tied = (high[:, None] == low[None, :]).sum()
    return float(above + 0.5 * tied), float(len(high) * len(low))


def precision(truth: numpy.ndarray, score: numpy.ndarray) -> float:
    """Average precision: precision weighted by each recall step."""
    order = sorted(range(len(score)), key=lambda i: -score[i])
    hits, positives = 0, float(truth.sum())
    total, last = 0.0, 0.0
    for place, i in enumerate(order):
        hits += int(truth[i])
        tail = place + 1 == len(order) or score[order[place + 1]] != score[i]
        if tail:
            recall = hits / positives
            total += (recall - last) * hits / (place + 1)
            last = recall
    return total


def harmonic(truth: numpy.ndarray, guess: numpy.ndarray, label: Any) -> float:
    """F-measure of one label; 0 when it is never true nor guessed."""
    tp = float(((truth == label) & (guess == label)).sum())
    wrong = float(((truth == label) != (guess == label)).sum())
    return 2 * tp / (2 * tp + wrong) if tp + wrong else 0.0


def kappa(truth: numpy.ndarray, guess: numpy.ndarray) -> float:
    """Quadratic-weighted kappa on level values, absent levels kept apart."""
    low = int(min(truth.min(), guess.min()))
    size = int(max(truth.max(), guess.max())) - low + 1
    seen = numpy.zeros((size, size))
    for a, b in zip(truth, guess, strict=True):
        seen[int(a) - low, int(b) - low] += 1
    weights = numpy.subtract.outer(range(size), range(size)) ** 2.0
    chance = numpy.outer(seen.sum(1), seen.sum(0)) / len(truth)
    return float(1 - (weights * seen).sum() / (weights * chance).sum())


def labelled(case: Case) -> Values:
    """Every classification metric from the model's outputs."""
    model, x, y = (case.data[k] for k in ("estimator", "features", "target"))
    truth, guess = numpy.asarray(y), model.predict(x)
    proba = numpy.clip(model.predict_proba(x), 2.0**-52, 1 - 2.0**-52)
    column = numpy.searchsorted(model.classes_, truth)
    labels = numpy.union1d(truth, guess)
    present = numpy.unique(truth)
    out = {
        "log_loss": float(
            -numpy.log(proba[numpy.arange(len(y)), column]).mean()
        ),
        "f1_macro": float(
            numpy.mean([harmonic(truth, guess, k) for k in labels])
        ),
        "balanced_accuracy": float(
            numpy.mean([(guess[truth == k] == k).mean() for k in present])
        ),
        "accuracy": float((truth == guess).mean()),
    }
    if case.data["task"] is tasks.Task.BINARY:
        positive = truth == model.classes_[1]
        pairs = wins(proba[:, 1], positive)
        out["roc_auc"] = pairs[0] / pairs[1]
        out["pr_auc"] = precision(positive, proba[:, 1])
        out["f1"] = harmonic(truth, guess, model.classes_[1])
    return out


def numeric(case: Case) -> Values:
    """Regression, ordinal, multilabel and multi-output metrics."""
    model, x, y = (case.data[k] for k in ("estimator", "features", "target"))
    guess = numpy.asarray(model.predict(x), dtype=float)
    truth = numpy.asarray(y, dtype=float)
    task = case.data["task"]
    if task is tasks.Task.ORDINAL:
        return {
            "qwk": kappa(truth, guess),
            "mace": float(numpy.abs(truth - guess).mean()),
            "accuracy_levels": float((truth == guess).mean()),
        }
    if task in (tasks.Task.MULTILABEL, tasks.Task.MULTIOUTPUT):
        others = x[list(model.target_columns[1:])].to_numpy(float)
        truth = numpy.column_stack([truth, others])
    gap = truth - guess
    if task is tasks.Task.MULTILABEL:
        hit = float((truth * guess).sum())
        return {
            "f1_macro_labels": float(
                numpy.mean(
                    [
                        harmonic(truth[:, j], guess[:, j], 1.0)
                        for j in range(truth.shape[1])
                    ]
                )
            ),
            "f1_micro_labels": 2
            * hit
            / (2 * hit + float(numpy.abs(gap).sum())),
            "hamming": float((gap != 0).mean()),
            "subset_accuracy": float((gap == 0).all(axis=1).mean()),
        }
    spread = ((truth - truth.mean(0)) ** 2).sum(0)
    if task is tasks.Task.MULTIOUTPUT:
        scale = numpy.where(truth.std(0) > 0, truth.std(0), 1.0)
        return {
            "r2_mean": float(numpy.mean(1 - (gap**2).sum(0) / spread)),
            "mae_scaled": float((numpy.abs(gap) / scale).mean()),
        }
    return {
        "mae": float(numpy.abs(gap).mean()),
        "rmse": math.sqrt(float((gap**2).mean())),
        "r2": float(1 - (gap**2).sum() / spread),
    }


def walk(case: Case) -> Values:
    """Forecast and uplift metrics, by explicit loops."""
    model, x, y = (case.data[k] for k in ("estimator", "features", "target"))
    guess = numpy.asarray(model.predict(x), dtype=float)
    if case.data["task"] is tasks.Task.FORECAST:
        truth = numpy.column_stack(
            [
                numpy.asarray(y, float),
                x[list(model.target_columns[1:])].to_numpy(float),
            ]
        )
        naive = x[model.value_column].to_numpy(float)[:, None]
        mean = float(numpy.abs(truth - guess).mean())
        size = (numpy.abs(truth) + numpy.abs(guess)) / 2
        return {
            "mase": mean
            / max(float(numpy.abs(truth - naive).mean()), forecast.SCALE_FLOOR),
            "forecast_mae": mean,
            "forecast_smape": float(
                (
                    numpy.abs(truth - guess)
                    / numpy.maximum(size, forecast.SCALE_FLOOR)
                ).mean()
            ),
        }
    arm = x[model.treatment_column].to_numpy(int)
    outcome = numpy.asarray(y, dtype=float)
    order = sorted(range(len(guess)), key=lambda i: -guess[i])
    curve, treated, control, gain, loss = [], 0, 0, 0.0, 0.0
    for i in order:
        treated, control = treated + arm[i], control + 1 - arm[i]
        gain, loss = (
            gain + outcome[i] * arm[i],
            loss + outcome[i] * (1 - arm[i]),
        )
        curve.append(gain - (loss * treated / control if control else 0.0))
    n = len(curve)
    rest = [c - (k + 1) / n * curve[-1] for k, c in enumerate(curve)]
    area = sum((rest[k] + rest[k + 1]) / 2 for k in range(n - 1))
    top = order[: max(int(n * uplift.TOP_SHARE), 1)]
    lifts = [[outcome[i] for i in top if arm[i] == a] for a in (1, 0)]
    return {
        "qini": area / n**2,
        "uplift_at_30": float(numpy.mean(lifts[0]) - numpy.mean(lifts[1])),
    }


def ordered(relevance: list[float], score: list[float], cutoff: int) -> Values:
    """One query's ranking metrics; ties of score keep row order."""
    order = sorted(range(len(score)), key=lambda i: -score[i])
    hits = [relevance[i] > 0 for i in order]
    found = [k for k, h in enumerate(hits) if h]
    gains = [0.0] * len(order)
    start = 0
    while start < len(order):
        stop = start
        while (
            stop + 1 < len(order)
            and score[order[stop + 1]] == score[order[start]]
        ):
            stop += 1
        mean = sum(relevance[i] for i in order[start : stop + 1]) / (
            stop - start + 1
        )
        for k in range(start, stop + 1):
            gains[k] = mean
        start = stop + 1
    ideal = sorted(relevance, reverse=True)
    discount = [1 / math.log2(k + 2) for k in range(min(cutoff, len(order)))]
    top = hits[:cutoff]
    return {
        "ndcg": sum(g * d for g, d in zip(gains, discount, strict=False))
        / sum(g * d for g, d in zip(ideal, discount, strict=False)),
        "map": sum(
            sum(top[: k + 1]) / (k + 1) for k in range(len(top)) if top[k]
        )
        / min(len(found), cutoff),
        "mrr": 1 / (found[0] + 1),
        "precision_at_10": sum(top) / min(cutoff, len(hits)),
        "recall_at_10": sum(top) / len(found),
    }


def ranked(case: Case) -> Values:
    """Ranking metrics, averaged over the queries that can be ranked."""
    model, x, y = (case.data[k] for k in ("estimator", "features", "target"))
    guess = [float(v) for v in model.predict(x)]
    truth = [float(v) for v in y]
    queries: dict[str, list[int]] = {}
    for row, label in enumerate(x[model.query_column].astype(str)):
        queries.setdefault(label, []).append(row)
    scores = [
        ordered([truth[r] for r in rows], [guess[r] for r in rows], 10)
        for rows in queries.values()
        if len(rows) > 1 and max(truth[r] for r in rows) > 0
    ]
    return {k: float(numpy.mean([s[k] for s in scores])) for k in scores[0]}


def quantile(values: Sequence[float], share: float) -> float:
    """The type-7 sample quantile (linear between order statistics)."""
    ordered = sorted(values)
    place = (len(ordered) - 1) * share
    low = math.floor(place)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (place - low) * (ordered[high] - ordered[low])


def censored(case: Case) -> Values:
    """Harrell's index, the integrated Brier score and the dynamic AUC."""
    model, x, y = (case.data[k] for k in ("estimator", "features", "target"))
    time = [float(v) for v in y]
    event = [bool(v) for v in x[model.event_column]]
    risk = [float(v) for v in model.predict(x)]
    n = range(len(time))
    pairs = [(i, j) for i in n if event[i] for j in n if time[j] > time[i]]
    concordant = sum(
        (risk[i] > risk[j]) + 0.5 * (risk[i] == risk[j]) for i, j in pairs
    )
    level, steps = 1.0, []
    for moment in sorted(
        {t for t, e in zip(time, event, strict=True) if not e}
    ):
        exposed = sum(t >= moment for t in time)
        gone = sum(
            t == moment and not e for t, e in zip(time, event, strict=True)
        )
        level *= 1 - gone / exposed
        steps.append((moment, level))

    def kept(moment: float) -> float:
        found = [v for m, v in steps if m <= moment]
        return max(
            found[-1] if found else 1.0, survival.MIN_CENSORING_PROBABILITY
        )

    source = [t for t, e in zip(time, event, strict=True) if e] or time
    grid = sorted({quantile(source, 0.1 + 0.8 * k / 9) for k in range(10)})
    curves = model.survival(numpy.asarray(grid))
    briers, aucs = [], []
    for k, moment in enumerate(grid):
        cases = [i for i in n if time[i] <= moment and event[i]]
        controls = [j for j in n if time[j] > moment]
        briers.append(
            numpy.mean(
                [curves[i, k] ** 2 / kept(time[i]) for i in cases]
                + [(1 - curves[j, k]) ** 2 / kept(moment) for j in controls]
                + [0.0] * (len(time) - len(cases) - len(controls))
            )
        )
        if cases and controls:
            weight = sum(1 / kept(time[i]) for i in cases) * len(controls)
            won = sum(
                ((risk[i] > risk[j]) + 0.5 * (risk[i] == risk[j]))
                / kept(time[i])
                for i in cases
                for j in controls
            )
            aucs.append(won / weight)
    span = grid[-1] - grid[0]
    area = sum(
        (briers[k] + briers[k + 1]) / 2 * (grid[k + 1] - grid[k])
        for k in range(len(grid) - 1)
    )
    return {
        "c_index": concordant / len(pairs),
        "ibs": float(area / span if len(grid) > 1 else briers[0]),
        "td_auc": float(numpy.mean(aucs)),
    }


def stream(name: str, index: int) -> numpy.random.Generator:
    """The seeded generator of one simulated input."""
    return randomness.component_rng(0, f"evidence.statistics.{name}", index)


def bundled(loader: str) -> tuple[pandas.DataFrame, pandas.Series]:
    """A real table that ships with scikit-learn, features and target."""
    found = getattr(datasets, loader)(return_X_y=True, as_frame=True)
    return found[0], found[1].rename("target")


def fitted(task: tasks.Task, name: str, x: Any, y: Any) -> Case:
    """Fits a linear model on the even rows and holds out the odd rows."""
    learn = (
        linear_model.LogisticRegression(max_iter=5000)
        if task in metrics.CLASSIFICATION
        else linear_model.Ridge()
    )
    scaled = (x - x.mean()) / x.std().replace(0, 1)
    model = learn.fit(scaled.iloc[::2], y.iloc[::2])
    test = slice(1, None, 2)
    return Case(
        name,
        "real",
        {
            "task": task,
            "estimator": model,
            "features": scaled.iloc[test].reset_index(drop=True),
            "target": y.iloc[test].reset_index(drop=True),
        },
    )


def held(task: tasks.Task, name: str, source: str, **parts: Any) -> Case:
    """An input whose model outputs are fixed (`Held`)."""
    features = parts.pop("features")
    target = parts.pop("target")
    return Case(
        name,
        source,
        {
            "task": task,
            "estimator": Held(**parts),
            "features": features,
            "target": target,
        },
    )


def progression() -> tuple[Case, numpy.ndarray]:
    """The diabetes regression case and its held-out predictions."""
    case = fitted(tasks.Task.REGRESSION, "diabetes", *bundled("load_diabetes"))
    return case, case.data["estimator"].predict(case.data["features"])


def graded() -> list[Case]:
    """Diabetes progression in quintile levels, read off the regression."""
    case, guess = progression()
    edges = numpy.quantile(bundled("load_diabetes")[1].iloc[::2], QUINTILES)
    levels = pandas.Series(numpy.digitize(case.data["target"], edges))
    return [
        held(
            tasks.Task.ORDINAL,
            "diabetes levels",
            "real",
            guess=numpy.digitize(guess, edges),
            features=case.data["features"],
            target=levels,
        )
    ]


def aged() -> list[Case]:
    """Diabetes patients ranked within age deciles by predicted progress."""
    case, guess = progression()
    ages = bundled("load_diabetes")[0]["age"].iloc[1::2].to_numpy()
    frame = case.data["features"].assign(
        query=pandas.qcut(ages, DECILES, labels=False)
    )
    grades = pandas.qcut(case.data["target"], QUARTILES, labels=False)
    return [
        held(
            tasks.Task.RANKING,
            "diabetes by age",
            "real",
            guess=guess,
            query_column="query",
            features=frame,
            target=pandas.Series(grades),
        )
    ]


def linnerud() -> list[Case]:
    """Three physiological measures predicted from three exercises."""
    x, y = datasets.load_linnerud(return_X_y=True, as_frame=True)
    model = linear_model.Ridge().fit(x.iloc[::2], y.iloc[::2])
    rows = y.iloc[1::2].reset_index(drop=True)
    names = tuple(rows.columns)
    return [
        held(
            tasks.Task.MULTIOUTPUT,
            "linnerud",
            "real",
            target_columns=names,
            guess=model.predict(x.iloc[1::2]),
            features=rows,
            target=rows[names[0]],
        )
    ]


def tags(digits: numpy.ndarray) -> numpy.ndarray:
    """Three labels of a digit: even, five or more, drawn with a loop."""
    return numpy.column_stack(
        [digits % 2 == 0, digits >= HIGH, numpy.isin(digits, LOOPED)]
    ).astype(int)


def written() -> list[Case]:
    """The labels of handwritten digits, read off a digit classifier."""
    case = fitted(tasks.Task.MULTICLASS, "digits", *bundled("load_digits"))
    named = case.data["estimator"].predict(case.data["features"])
    rows = pandas.DataFrame(
        tags(case.data["target"].to_numpy()), columns=["even", "high", "loop"]
    )
    return [
        held(
            tasks.Task.MULTILABEL,
            "digits labels",
            "real",
            guess=tags(named),
            target_columns=tuple(rows.columns),
            features=rows,
            target=rows["even"],
        )
    ]


QUINTILES = (0.2, 0.4, 0.6, 0.8)
DECILES = 10
QUARTILES = 4
HIGH = 5
LOOPED = (0, 6, 8, 9)
ACTUAL: dict[tasks.Task, Callable[[], list[Case]]] = {
    tasks.Task.BINARY: lambda: [
        fitted(
            tasks.Task.BINARY, "breast_cancer", *bundled("load_breast_cancer")
        )
    ],
    tasks.Task.MULTICLASS: lambda: [
        fitted(tasks.Task.MULTICLASS, name, *bundled(f"load_{name}"))
        for name in ("wine", "digits")
    ],
    tasks.Task.REGRESSION: lambda: [progression()[0]],
    tasks.Task.ORDINAL: graded,
    tasks.Task.RANKING: aged,
    tasks.Task.MULTIOUTPUT: linnerud,
    tasks.Task.MULTILABEL: written,
    tasks.Task.FORECAST: list,
    tasks.Task.UPLIFT: list,
    tasks.Task.SURVIVAL: list,
}


@dataclasses.dataclass(frozen=True)
class World:
    """The draws a simulated input starts from.

    Attributes:
        task: The task simulated.
        index: Which input; `TIED` holds ties or an absent level.
        rng: The input's own generator.
        features: `SIZE` rows of three standard normal columns.
        signal: `a - b / 2`, what the outcome follows.
    """

    task: tasks.Task
    index: int
    rng: numpy.random.Generator
    features: pandas.DataFrame
    signal: pandas.Series

    @property
    def name(self) -> str:
        """The input's name."""
        return f"{self.task.value} {self.index}"

    def hold(self, guess: Any, target: Any, **parts: Any) -> Case:
        """A simulated input whose model outputs are `guess`."""
        features = parts.pop("features", self.features)
        return held(
            self.task,
            self.name,
            "simulated",
            guess=guess,
            features=features,
            target=pandas.Series(target),
            **parts,
        )


SIZE = 240
TIED = 1


def classes(world: World) -> Case:
    """Labels from a noisy signal, scored by a fitted model.

    Input `TIED` uses a depth-two tree, whose probabilities tie in blocks
    and reach 0 and 1.
    """
    cuts = [0.0] if world.task is tasks.Task.BINARY else [-0.5, 0.5]
    noisy = world.signal + world.rng.normal(size=SIZE)
    y = pandas.Series(numpy.digitize(noisy, cuts)).map(
        {0: "no", 1: "yes", 2: "maybe"}
    )
    model = (
        tree.DecisionTreeClassifier(max_depth=2, random_state=0)
        if world.index == TIED
        else linear_model.LogisticRegression()
    ).fit(world.features, y)
    return Case(
        world.name,
        "simulated",
        {
            "task": world.task,
            "estimator": model,
            "features": world.features,
            "target": y,
        },
    )


def numbers(world: World) -> Case:
    """A noisy signal, predicted by a ridge fitted on half the rows."""
    y = pandas.Series(world.signal + world.rng.normal(size=SIZE))
    half = SIZE // 2
    model = linear_model.Ridge().fit(world.features.iloc[:half], y.iloc[:half])
    return Case(
        world.name,
        "simulated",
        {
            "task": world.task,
            "estimator": model,
            "features": world.features,
            "target": y,
        },
    )


def levels(world: World) -> Case:
    """Five levels guessed within one; input `TIED` never holds level 2."""
    truth = numpy.clip(numpy.round(world.signal + 2), 0, 4).astype(int)
    guess = numpy.clip(truth + world.rng.integers(-1, 2, size=SIZE), 0, 4)
    if world.index == TIED:
        truth, guess = (numpy.where(v == 2, 3, v) for v in (truth, guess))  # noqa: PLR2004 -- the level left out
    return world.hold(guess, truth)


def joint(world: World) -> Case:
    """Three targets, numbers or their signs, guessed with noise."""
    truth = numpy.column_stack(
        [world.signal, world.features["b"], world.features["c"]]
    )
    guess = truth + world.rng.normal(scale=0.7, size=truth.shape)
    if world.task is tasks.Task.MULTILABEL:
        truth, guess = (truth > 0).astype(int), (guess > 0).astype(int)
    frame = world.features.assign(t1=truth[:, 1], t2=truth[:, 2])
    return world.hold(
        guess, truth[:, 0], features=frame, target_columns=("t0", "t1", "t2")
    )


def horizon(world: World) -> Case:
    """A random walk forecast three steps ahead."""
    path = numpy.cumsum(world.rng.normal(size=(SIZE, 4)), axis=1) + 10
    frame = world.features.assign(now=path[:, 0], s2=path[:, 2], s3=path[:, 3])
    guess = path[:, 1:] + world.rng.normal(scale=0.5, size=(SIZE, 3))
    return world.hold(
        guess,
        path[:, 1],
        features=frame,
        value_column="now",
        target_columns=("s1", "s2", "s3"),
    )


def arms(world: World) -> Case:
    """A randomized treatment helping where `c > 0`; `TIED` rounds guesses."""
    arm = world.rng.integers(0, 2, size=SIZE)
    lift = world.features["c"] > 0
    outcome = world.signal + arm * lift + world.rng.normal(size=SIZE)
    noise = world.rng.normal(scale=0.3, size=SIZE)
    guess = numpy.round(numpy.asarray(lift + noise), 1 - world.index)
    return world.hold(
        guess,
        outcome,
        treatment_column="arm",
        features=world.features.assign(arm=arm),
    )


def queries(world: World) -> Case:
    """Queries of eight documents graded 0 to 3; `TIED` rounds scores."""
    grades = numpy.clip(numpy.round(world.signal + 1), 0, 3)
    guess = numpy.round(
        numpy.asarray(world.signal + world.rng.normal(size=SIZE)),
        2 - world.index * 2,
    )
    return world.hold(
        guess,
        grades,
        query_column="query",
        features=world.features.assign(query=numpy.arange(SIZE) // 8),
    )


def lifetimes(world: World) -> Case:
    """Exponential lifetimes under exponential censoring; `TIED` rounds."""
    rates = 0.1 * numpy.exp(world.signal.to_numpy())
    event = world.rng.exponential(1 / rates)
    stop = world.rng.exponential(15.0, size=SIZE)
    duration = numpy.round(numpy.minimum(event, stop), 3 - 2 * world.index)
    return world.hold(
        rates,
        duration + 0.01,
        rates=rates,
        event_column="event",
        features=world.features.assign(event=(event <= stop).astype(int)),
    )


WORLDS: dict[tasks.Task, Callable[[World], Case]] = {
    tasks.Task.BINARY: classes,
    tasks.Task.MULTICLASS: classes,
    tasks.Task.REGRESSION: numbers,
    tasks.Task.ORDINAL: levels,
    tasks.Task.MULTILABEL: joint,
    tasks.Task.MULTIOUTPUT: joint,
    tasks.Task.FORECAST: horizon,
    tasks.Task.UPLIFT: arms,
    tasks.Task.RANKING: queries,
    tasks.Task.SURVIVAL: lifetimes,
}


def simulate(task: tasks.Task, index: int) -> Case:
    """A simulated input of a task; index `TIED` holds ties."""
    rng = stream(task.value, index)
    x = pandas.DataFrame(rng.normal(size=(SIZE, 3)), columns=["a", "b", "c"])
    world = World(task, index, rng, x, x["a"] - 0.5 * x["b"])
    return WORLDS[task](world)


def stepup(
    values: Sequence[float], rule: str
) -> tuple[list[float], list[bool]]:
    """Benjamini-Hochberg (`bh`) or -Yekutieli (`by`) from their definition.

    Args:
        values: The p-values.
        rule: `bh` or `by`.

    Returns:
        Adjusted p-values (running minimum of `p m c / k` from the largest
        rank down, at most 1) and the exact step-up decisions at `ALPHA`.
    """
    m = len(values)
    factor = sum((Fraction(1, i) for i in range(1, m + 1)), Fraction(0))
    factor = factor if rule == "by" else Fraction(1)
    order = sorted(range(m), key=lambda i: (values[i], i))
    count = max(
        (
            k
            for k in range(1, m + 1)
            if Fraction(values[order[k - 1]]) <= k * ALPHA / (m * factor)
        ),
        default=0,
    )
    adjusted, running = [1.0] * m, 1.0
    for k in reversed(range(m)):
        running = min(running, values[order[k]] * float(m * factor) / (k + 1))
        adjusted[order[k]] = running
    rejected = [False] * m
    for rank in range(count):
        rejected[order[rank]] = True
    return adjusted, rejected


def stepdown(
    values: Sequence[float], rule: str
) -> tuple[list[float], list[bool]]:
    """Holm (`holm`) or Bonferroni (`bonferroni`) from their definition.

    Args:
        values: The p-values.
        rule: `holm` or `bonferroni`.

    Returns:
        Adjusted p-values (Holm: running maximum of `p (m - k)`; Bonferroni:
        `p m`; at most 1) and the exact decisions at `ALPHA`.
    """
    m = len(values)
    order = sorted(range(m), key=lambda i: (values[i], i))
    adjusted, rejected = [1.0] * m, [False] * m
    running, going = 0.0, True
    for k, i in enumerate(order):
        scale = m - k if rule == "holm" else m
        scaled = min(1.0, values[i] * scale)
        running = max(running, scaled) if rule == "holm" else scaled
        adjusted[i] = running
        passed = Fraction(values[i]) <= ALPHA / scale
        going = going and passed
        rejected[i] = going if rule == "holm" else passed
    return adjusted, rejected


def flatten(rules: Sequence[str], found: Mapping[str, Any]) -> Values:
    """Keys `rule|adjusted|i` and `rule|rejected|i` of adjusted families."""
    out: Values = {}
    for rule in rules:
        adjusted, rejected = found[rule]
        for i, (a, r) in enumerate(zip(adjusted, rejected, strict=True)):
            out[f"{rule}|adjusted|{i}"] = float(a)
            out[f"{rule}|rejected|{i}"] = float(r)
    return out


def adjusting(rules: Sequence[str]) -> Callable[[Case], Values]:
    """The product's `multiplicity.adjust` under each rule."""

    def run(case: Case) -> Values:
        found = {}
        for rule in rules:
            correction = budget.Correction(rule)
            done = multiplicity.adjust(case.data["p"], correction, ALPHA)
            found[rule] = (done.adjusted, done.rejected)
        return flatten(rules, found)

    return run


def defined(rules: Sequence[str]) -> Callable[[Case], Values]:
    """The scratch adjustment under each rule (`stepup`, `stepdown`)."""

    def run(case: Case) -> Values:
        values = case.data["p"]
        return flatten(
            rules,
            {
                r: (stepup if r in ("bh", "by") else stepdown)(values, r)
                for r in rules
            },
        )

    return run


def library(rules: Sequence[str]) -> Callable[[Case], Values]:
    """`scipy.stats.false_discovery_control`, for `bh` and `by` only."""

    def run(case: Case) -> Values:
        found = {}
        for rule in (r for r in rules if r in ("bh", "by")):
            adjusted = stats.false_discovery_control(
                case.data["p"], method=rule
            )
            found[rule] = (adjusted, adjusted <= float(ALPHA))
        return flatten(list(found), found)

    return run


def families() -> list[Case]:
    """P-value families: real tests on bundled tables, and simulated ones."""
    x, y = bundled("load_breast_cancer")
    welch = [
        stats.ttest_ind(x[c][y == 0], x[c][y == 1], equal_var=False).pvalue
        for c in x
    ]
    x, y = bundled("load_wine")
    anova = [
        stats.f_oneway(*(x[c][y == k] for k in range(3))).pvalue for c in x
    ]
    found = [
        Case("breast_cancer welch", "real", {"p": [float(p) for p in welch]}),
        Case("wine anova", "real", {"p": [float(p) for p in anova]}),
        Case(
            "ties and extremes",
            "simulated",
            {"p": [1e-300, 0.01, 0.01, 0.04, 0.5, 1.0, 0.0]},
        ),
    ]
    for index, size in enumerate((1, 7, 40, 200)):
        rng = stream("multiplicity", index)
        values = numpy.concatenate(
            [rng.uniform(size=size), rng.beta(0.1, 1.0, size=size // 4)]
        )
        found.append(
            Case(f"mixture {size}", "simulated", {"p": values.tolist()})
        )
    return found


def exact(covered: int, rows: int, level: float) -> tuple[float, float]:
    """Clopper-Pearson limits by bisection on the exact binomial tails.

    Args:
        covered: Successes.
        rows: Trials.
        level: Two-sided confidence level.

    Returns:
        The rate whose upper tail at `covered` is `(1 - level) / 2`, and the
        one whose lower tail is; 0 and 1 at the ends.
    """
    tail = (1 - level) / 2

    def below(rate: float, most: int) -> float:
        if most < 0:
            return 0.0
        if rate in (0.0, 1.0):
            return 1.0 if rate == 0.0 or most >= rows else 0.0
        return sum(
            math.exp(
                math.lgamma(rows + 1)
                - math.lgamma(k + 1)
                - math.lgamma(rows - k + 1)
                + k * math.log(rate)
                + (rows - k) * math.log1p(-rate)
            )
            for k in range(most + 1)
        )

    def solve(target: Callable[[float], float]) -> float:
        low, high = 0.0, 1.0
        for _ in range(200):
            middle = (low + high) / 2
            if target(middle) < 0:
                low = middle
            else:
                high = middle
        return (low + high) / 2

    lower = (
        0.0
        if covered == 0
        else solve(lambda r: (1 - below(r, covered - 1)) - tail)
    )
    upper = (
        1.0 if covered == rows else solve(lambda r: tail - below(r, covered))
    )
    return lower, upper


def counts() -> list[Case]:
    """Success counts: class sizes of real tables and a simulated grid."""
    found = []
    for loader in ("load_breast_cancer", "load_digits"):
        y = bundled(loader)[1]
        k = int((y == y.iloc[0]).sum())
        found.append(Case(f"{loader[5:]} class", "real", {"k": k, "n": len(y)}))
    grid = [
        (0, 1),
        (0, 5),
        (1, 5),
        (5, 5),
        (3, 10),
        (0, 59),
        (1, 124),
        (816, 882),
    ]
    found += [
        Case(f"{k} of {n}", "simulated", {"k": k, "n": n}) for k, n in grid
    ]
    return found


LEVELS = (0.9, 0.95, 0.99)


def bounds(
    function: Callable[[int, int, float], Any],
) -> Callable[[Case], Values]:
    """Both limits of an exact interval at every level of `LEVELS`."""

    def run(case: Case) -> Values:
        out = {}
        for level in LEVELS:
            low, high = function(case.data["k"], case.data["n"], level)
            out[f"{level}|lower"], out[f"{level}|upper"] = (
                float(low),
                float(high),
            )
        return out

    return run


def binomial(covered: int, rows: int, level: float) -> tuple[float, float]:
    """`scipy.stats.binomtest(...).proportion_ci(method="exact")`."""
    found = stats.binomtest(covered, rows).proportion_ci(level, "exact")
    return found.low, found.high


# Relative size of the last term at which the series and the continued
# fraction stop: below the float64 resolution, so the sum is final.
PRECISION = 1e-17
# Relative slack under which two table probabilities count as equal in
# Fisher's two-sided test (R's and scipy's convention).
SLACK = 1e-7


def gamma(shape: float, point: float) -> float:
    """Regularized upper incomplete gamma `Q(shape, point)`.

    A series for `P` below `shape + 1`, Lentz's continued fraction for `Q`
    above (Numerical Recipes, section 6.2).
    """
    if point <= 0:
        return 1.0
    front = math.exp(-point + shape * math.log(point) - math.lgamma(shape))
    if point < shape + 1:
        term = total = 1 / shape
        step = shape
        while abs(term) > abs(total) * PRECISION:
            step += 1
            term *= point / step
            total += term
        return 1 - total * front
    tiny = sys.float_info.min
    b = point + 1 - shape
    c, d = 1 / tiny, 1 / b
    h = d
    for i in range(1, 10_000):
        an = -i * (i - shape)
        b += 2
        d = an * d + b
        d = 1 / (d if abs(d) > tiny else tiny)
        c = b + an / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        if abs(d * c - 1) < PRECISION:
            break
    return front * h


def table(case: Case) -> numpy.ndarray:
    """The missing-by-class table the missingness test reads."""
    missing = case.data["feature"].isna().to_numpy()
    classes = leakage.bucket(case.data["target"], case.data["task"])
    return pandas.crosstab(missing, classes).to_numpy().astype(float)


def fisher(counts: numpy.ndarray, mass: Callable[[int], float]) -> float:
    """Two-sided exact p: tables no likelier than the observed one."""
    rows, column = int(counts[0].sum()), int(counts[:, 0].sum())
    total = int(counts.sum())
    support = range(max(0, rows + column - total), min(rows, column) + 1)
    seen = mass(int(counts[0, 0]))
    return min(
        1.0, sum(mass(a) for a in support if mass(a) <= seen * (1 + SLACK))
    )


def association(case: Case, flavour: str) -> Values:
    """The missingness test's p-value and Cramer's V.

    Args:
        case: Holds `feature`, `target` and `task`.
        flavour: `scratch` (formulas, `gamma`, exact counts) or `scipy`
            (`chisquare`, `association`, `hypergeom`).

    Returns:
        `p_value` and `cramers_v`; Fisher's exact test on a 2 x 2 table
        with an expected count below `EXPECTED_FISHER`, else chi-square.
    """
    counts = table(case)
    total = counts.sum()
    expected = numpy.outer(counts.sum(1), counts.sum(0)) / total
    chi = float(((counts - expected) ** 2 / expected).sum())
    width = min(counts.shape) - 1
    small = counts.shape == (2, 2) and numpy.min(expected) < EXPECTED_FISHER
    rows, column, size = (
        int(counts[0].sum()),
        int(counts[:, 0].sum()),
        int(total),
    )
    if flavour == "scratch":

        def mass(a: int) -> float:
            ways = math.comb(column, a) * math.comb(size - column, rows - a)
            return ways / math.comb(size, rows)

        p = (
            fisher(counts, mass)
            if small
            else gamma((counts.shape[1] - 1) / 2, chi / 2)
        )
        return {"p_value": p, "cramers_v": math.sqrt(chi / (total * width))}
    law = stats.hypergeom(size, column, rows)
    p = (
        fisher(counts, lambda a: float(law.pmf(a)))
        if small
        else float(
            stats.chisquare(
                counts.ravel(), expected.ravel(), ddof=counts.shape[1]
            ).pvalue
        )
    )
    v = stats.contingency.association(
        counts.astype(int), method="cramer", correction=False
    )
    return {"p_value": p, "cramers_v": float(v)}


def tested(case: Case) -> Values:
    """The product's `missingness_test` on the case."""
    found = leakage.test(
        case.data["feature"],
        case.data["target"],
        case.data["task"],
        leakage.CERTIFIED,
    )
    if found is None:
        raise errors.InputError(f"{case.name}: no missingness test is formed")
    return {"p_value": found.p_value, "cramers_v": found.association}


def holes(
    name: str,
    source: str,
    x: pandas.DataFrame,
    y: pandas.Series,
    rows: numpy.ndarray,
    task: str,
) -> Case:
    """A column of `x` with values removed at `rows`, against `y`."""
    feature = x.iloc[:, 0].copy()
    feature.iloc[rows] = numpy.nan
    return Case(name, source, {"feature": feature, "target": y, "task": task})


def permutation(index: numpy.ndarray) -> numpy.ndarray:
    """Checks that every row of a draw is a permutation of the rows.

    Raises:
        InputError: If a row repeats or skips a position.
    """
    if not (numpy.sort(index, axis=1) == numpy.arange(index.shape[1])).all():
        raise errors.InputError("a drawn index is not a permutation")
    return index


def ridge(
    values: numpy.ndarray, phi: numpy.ndarray, splits: Any, flavour: str
) -> numpy.ndarray:
    """Out-of-fold ridge fits at the training mean, `RIDGE` penalty.

    `sklearn` fits `Ridge(fit_intercept=False)` on the centred outcome;
    `scratch` solves the augmented least-squares system
    `[phi; sqrt(RIDGE) I] beta = [y - mean; 0]`.
    """
    out = numpy.empty_like(values, dtype=float)
    width = phi.shape[1]
    for train, test in splits:
        mean = values[:, train].mean(axis=1)
        centred = values[:, train] - mean[:, None]
        if flavour == "sklearn":
            beta = (
                linear_model.Ridge(
                    alpha=settings.RIDGE, fit_intercept=False, solver="svd"
                )
                .fit(phi[train], centred.T)
                .coef_
            )
        else:
            design = numpy.vstack(
                [phi[train], math.sqrt(settings.RIDGE) * numpy.eye(width)]
            )
            right = numpy.vstack([centred.T, numpy.zeros((width, len(values)))])
            beta = numpy.linalg.lstsq(design, right, rcond=None)[0].T
        out[:, test] = (
            mean[:, None] + beta.reshape(len(values), width) @ phi[test].T
        )
    return out


def pooled(
    score: numpy.ndarray, positive: numpy.ndarray, tests: Any, flavour: str
) -> float:
    """Within-fold AUC pooled over folds: concordant pairs over pairs."""
    above = pairs = 0.0
    for test in tests:
        hit = positive[test]
        if hit.all() or not hit.any():
            continue
        count = float(hit.sum() * (~hit).sum())
        if flavour == "sklearn":
            above += sklearn.metrics.roc_auc_score(hit, score[test]) * count
        else:
            above += wins(score[test], hit)[0]
        pairs += count
    return above / pairs if pairs else math.nan


def determination(labels: numpy.ndarray, fit: numpy.ndarray) -> numpy.ndarray:
    """`1 - SSE / SST` of each row of `labels` against `fit`."""
    spread = ((labels - labels.mean(1)[:, None]) ** 2).sum(1)
    found: numpy.ndarray = 1 - ((labels - fit) ** 2).sum(1) / spread
    return found


def skill(
    problem: Any, phi: numpy.ndarray, flavour: str
) -> Callable[[numpy.ndarray], numpy.ndarray]:
    """The screen's skill statistic for a batch of outcomes."""
    tests = [split[1] for split in problem.splits]

    def compute(labels: numpy.ndarray) -> numpy.ndarray:
        if problem.task == "regression":
            fit = ridge(labels, phi, problem.splits, flavour)
            if flavour == "sklearn":
                return numpy.array(
                    [
                        sklearn.metrics.r2_score(t, f)
                        for t, f in zip(labels, fit, strict=True)
                    ]
                )
            return determination(labels, fit)
        wanted = (
            [1]
            if problem.classes == settings.BINARY_CLASSES
            else range(problem.classes)
        )
        scores = []
        for k in wanted:
            fit = ridge(
                (labels == k).astype(float), phi, problem.splits, flavour
            )
            scores.append(
                [
                    pooled(f, t == k, tests, flavour)
                    for f, t in zip(fit, labels, strict=True)
                ]
            )
        found: numpy.ndarray = numpy.mean(scores, axis=0)
        return found

    return compute


def echo(
    problem: Any, data: numpy.ndarray, flavour: str
) -> Callable[[numpy.ndarray], numpy.ndarray]:
    """Agreement of a column read as the target, or its unfitted R^2."""

    def compute(labels: numpy.ndarray) -> numpy.ndarray:
        if problem.task == "classification":
            if flavour == "sklearn":
                return numpy.array(
                    [sklearn.metrics.accuracy_score(t, data) for t in labels]
                )
            share: numpy.ndarray = (labels == data[None, :]).mean(axis=1)
            return share
        if flavour == "sklearn":
            return numpy.array(
                [sklearn.metrics.r2_score(t, data) for t in labels]
            )
        return determination(labels, data[None, :])

    return compute


def standard(share: float, base: float, classes: int) -> float:
    """A class's share of recorded rows carried to equal classes."""
    if math.isclose(1 - base, 0.0, abs_tol=1e-8):
        return 0.0
    return 1 - (1 - share) * (1 - 1 / max(classes, 1)) / (1 - base)


def purity(recorded: numpy.ndarray) -> Callable[[numpy.ndarray], numpy.ndarray]:
    """The largest standardized class share among the recorded rows."""

    def compute(labels: numpy.ndarray) -> numpy.ndarray:
        size = int(labels.max()) + 1
        out = []
        for row in labels:
            kept = row[recorded]
            out.append(
                max(
                    standard(
                        float((kept == c).mean()),
                        float((row == c).mean()),
                        size,
                    )
                    for c in range(size)
                )
            )
        return numpy.array(out)

    return compute


@dataclasses.dataclass
class Hypothesis:
    """One hypothesis of the screen's family, as a reference recounts it.

    Attributes:
        finding: The product's finding, read for its feature and mechanism
            (and its p-value when no reference recounts it).
        measure: The reference statistic of a batch, or None when analytic.
        binned: Whether `measure` reads the binned outcome.
        observed: The reference's observed statistic, or NaN.
        p: The analytic p-value, or None.
        exceed: Draws at least the observed statistic.
        drawn: Draws made.
        active: Whether it still draws.
    """

    finding: Any
    measure: Callable[[numpy.ndarray], numpy.ndarray] | None
    binned: bool = False
    observed: float = math.nan
    p: float | None = None
    exceed: int = 0
    drawn: int = 0
    active: bool = False


def reading(
    sample: pandas.DataFrame, feature: str, problem: Any, ceiling: float
) -> numpy.ndarray:
    """A column read as the target, for an echo hypothesis.

    Raises:
        InputError: If the column cannot echo the target above `ceiling`,
            so the product's echo hypothesis has no reading here.
    """
    column = hypotheses.forecast(
        sample[feature], sample["target"], problem, ceiling
    )
    if column is None:
        raise errors.InputError(f"{feature} cannot echo the target here")
    return column


def design(sample: pandas.DataFrame, feature: str) -> numpy.ndarray:
    """The skill design of a column, or of a pair `a & b` jointly."""
    if feature in sample.columns:
        return fitting.basis(sample[feature])
    first, second = feature.split(" & ")
    joint = sample[first].astype(str) + " & " + sample[second].astype(str)
    return fitting.basis(joint)


def recount(case: Case, flavour: str) -> Values:
    """Recomputes the screen's statistics, p-values, BY step and bounds.

    The inputs are the product's: its chosen rows, folds, column designs,
    hypothesis family and permutation streams (each draw checked to be a
    permutation). Every statistic, count, adjustment and bound is computed
    here, by `flavour` (`sklearn` or `scratch`).
    """
    data = case.data
    cfg, found, task = data["cfg"], data["report"], data["task"]
    rows = planning.subsample(len(data["frame"]), None, None, cfg.seed)[0]
    sample = data["frame"].iloc[rows].reset_index(drop=True)
    problem = fitting.encode(sample["target"], task, cfg, None, None)
    codes = report.tally(sample["target"], task)[0]
    family = []
    for f in found.findings:
        h = Hypothesis(f, None)
        if f.mechanism == settings.SKILL:
            h.measure = skill(problem, design(sample, f.feature), flavour)
        elif f.mechanism == settings.ECHO:
            ceiling = found.ceilings[settings.ECHO]
            h.measure = echo(
                problem, reading(sample, f.feature, problem, ceiling), flavour
            )
        elif f.mechanism == settings.PURITY:
            h.measure, h.binned = (
                purity(sample[f.feature].notna().to_numpy()),
                True,
            )
        elif f.mechanism == settings.MISSING:
            analytic = association(
                Case(
                    "",
                    "",
                    {
                        "feature": sample[f.feature],
                        "target": sample["target"],
                        "task": task,
                    },
                ),
                "scipy" if flavour == "sklearn" else "scratch",
            )
            h.p, h.observed = analytic["p_value"], analytic["cramers_v"]
        else:
            h.p = f.p_value
        if h.measure is not None:
            labels = codes if h.binned else problem.y
            h.observed = float(h.measure(labels[None, :])[0])
            h.active = h.observed > found.ceilings[f.mechanism]
        family.append(h)
    draw(
        family,
        problem.y,
        codes,
        found.permutations,
        Fraction(str(cfg.alpha)),
        cfg.seed,
    )
    return conclude(
        family, found.permutations, sample, problem, codes, cfg, flavour
    )


def draw(
    family: list[Hypothesis],
    y: numpy.ndarray,
    codes: numpy.ndarray,
    total: int,
    alpha: Fraction,
    seed: int,
) -> None:
    """Counts each active hypothesis over the screen's permutation stream.

    A hypothesis stops once `(exceed + 1) / (total + 1)` passes the largest
    Benjamini-Yekutieli threshold `alpha / H_m`, as the screen's does.
    """
    ceiling = alpha / sum(
        (Fraction(1, i) for i in range(1, len(family) + 1)), Fraction(0)
    )
    live = [h for h in family if h.measure is not None]
    strata = numpy.zeros(len(y), dtype="int64")
    done = batch = 0
    while done < total and any(h.active for h in live):
        count = min(settings.BATCH, total - done)
        rng = randomness.component_rng(seed, "leakage.screen.batch", batch)
        index = permutation(fitting.permute(strata, count, rng))
        for h in live:
            if not h.active or h.measure is None:
                continue
            values = h.measure((codes if h.binned else y)[index])
            h.exceed += int((values >= h.observed - settings.TOLERANCE).sum())
            h.drawn += count
            h.active = Fraction(h.exceed + 1, total + 1) <= ceiling
        done += count
        batch += 1


def conclude(
    family: list[Hypothesis],
    total: int,
    sample: pandas.DataFrame,
    problem: Any,
    codes: numpy.ndarray,
    cfg: Any,
    flavour: str,
) -> Values:
    """P-values, the BY step at `ALPHA` and bounds of significant ones."""
    if Fraction(str(cfg.alpha)) != ALPHA:
        raise errors.InputError(
            f"the screen's level {cfg.alpha} is not {ALPHA}"
        )
    ps = [
        h.p if h.p is not None else (h.exceed + 1) / (h.drawn + 1)
        for h in family
    ]
    complete = [h.p is not None or h.drawn == total for h in family]
    used = [p if c else 1.0 for p, c in zip(ps, complete, strict=True)]
    if flavour == "sklearn":
        adjusted = stats.false_discovery_control(used, method="by").tolist()
        rejected = [a <= float(ALPHA) for a in adjusted]
    else:
        adjusted, rejected = stepup(used, "by")
    out: Values = {}
    for k, h in enumerate(family):
        f = h.finding
        key = f"{f.feature}|{f.mechanism}"
        if f.mechanism in RECOUNTED:
            out[f"{key}|statistic"], out[f"{key}|p"] = h.observed, ps[k]
        out[f"{key}|adjusted"] = adjusted[k] if complete[k] else 1.0
        out[f"{key}|significant"] = float(rejected[k])
        if rejected[k] and f.mechanism in RECOUNTED:
            out[f"{key}|bound"] = bound(h, sample, problem, codes, cfg, flavour)
    return out


RECOUNTED = (settings.SKILL, settings.ECHO, settings.PURITY, settings.MISSING)


def bound(
    h: Hypothesis,
    sample: pandas.DataFrame,
    problem: Any,
    codes: numpy.ndarray,
    cfg: Any,
    flavour: str,
) -> float:
    """The lower confidence bound of a significant hypothesis.

    Record purity: the exact binomial bound of the chosen class's share,
    standardized. Otherwise the `1 - confidence` quantile (`quantile`, or
    NumPy's for `sklearn`) of the statistic over `cfg.bootstrap` row
    resamples drawn from the screen's stream for that hypothesis.
    """
    f = h.finding
    n, size = len(problem.y), int(codes.max()) + 1
    if f.mechanism == settings.PURITY:
        kept = codes[sample[f.feature].notna().to_numpy()]
        shares = [float((kept == c).mean()) for c in range(size)]
        base = [float((codes == c).mean()) for c in range(size)]
        chosen = int(
            numpy.argmax(
                [
                    standard(s, b, size)
                    for s, b in zip(shares, base, strict=True)
                ]
            )
        )
        hits = round(shares[chosen] * len(kept))
        if hits == 0:
            return 0.0
        limit = (binomial if flavour == "sklearn" else exact)(
            hits, len(kept), 2 * cfg.confidence - 1
        )[0]
        return standard(float(limit), base[chosen], size)
    path = f"leakage.screen.{f.mechanism}.{f.feature}"
    rng = randomness.component_rng(cfg.seed, path, 0)
    replicate = resampler(h, sample, problem, codes, flavour)
    values = []
    for _ in range(cfg.bootstrap):
        value = replicate(rng.integers(0, n, size=n))
        if value is not None:
            values.append(value)
    if not values:
        return -math.inf
    share = 1 - cfg.confidence
    return (
        quantile(values, share)
        if flavour == "scratch"
        else float(numpy.quantile(values, share))
    )


Resample = Callable[[numpy.ndarray], float | None]


def cramer(
    missing: numpy.ndarray, codes: numpy.ndarray, flavour: str
) -> Resample:
    """Cramer's V of missingness on a resample; None without a full table."""
    size = int(codes.max()) + 1

    def compute(index: numpy.ndarray) -> float | None:
        gone, labels = missing[index], codes[index]
        if gone.all() or not gone.any() or len(numpy.unique(labels)) < size:
            return None
        counts = numpy.array(
            [
                [((gone == m) & (labels == k)).sum() for k in range(size)]
                for m in (True, False)
            ]
        )
        if flavour == "sklearn":
            return float(
                stats.contingency.association(
                    counts, method="cramer", correction=False
                )
            )
        expected = numpy.outer(counts.sum(1), counts.sum(0)) / len(index)
        chi = float(((counts - expected) ** 2 / expected).sum())
        return math.sqrt(chi / len(index))

    return compute


def agreement(problem: Any, column: numpy.ndarray) -> Resample:
    """An echo's agreement (or unfitted R^2) on a resample."""

    def compute(index: numpy.ndarray) -> float | None:
        y, x = problem.y[index], column[index]
        if problem.task == "classification":
            return float((y == x).mean())
        total = float(((y - y.mean()) ** 2).sum())
        return 1 - float(((y - x) ** 2).sum()) / total if total > 0 else None

    return compute


def skillful(problem: Any, phi: numpy.ndarray, flavour: str) -> Resample:
    """A design's skill on a resample, its out-of-fold fits held fixed."""
    regression = problem.task == "regression"
    binary = problem.classes == settings.BINARY_CLASSES
    wanted = [1] if binary else list(range(problem.classes))
    targets = (
        [problem.y.astype(float)]
        if regression
        else [(problem.y == k).astype(float) for k in wanted]
    )
    fits = [ridge(t[None, :], phi, problem.splits, flavour)[0] for t in targets]
    fold = numpy.empty(len(problem.y), dtype=int)
    for number, split in enumerate(problem.splits):
        fold[split[1]] = number

    def compute(index: numpy.ndarray) -> float | None:
        y = problem.y[index]
        if regression:
            total = float(((y - y.mean()) ** 2).sum())
            miss = float(((y - fits[0][index]) ** 2).sum())
            return 1 - miss / total if total else None
        tests = [
            numpy.flatnonzero(fold[index] == g)
            for g in numpy.unique(fold[index])
        ]
        scores = [
            pooled(fit[index], y == k, tests, flavour)
            for fit, k in zip(fits, wanted, strict=True)
        ]
        if any(math.isnan(s) for s in scores):
            return None
        return float(numpy.mean(scores))

    return compute


def resampler(
    h: Hypothesis,
    sample: pandas.DataFrame,
    problem: Any,
    codes: numpy.ndarray,
    flavour: str,
) -> Resample:
    """The statistic of one row resample, None when it cannot be scored."""
    f = h.finding
    if f.mechanism == settings.MISSING:
        return cramer(sample[f.feature].isna().to_numpy(), codes, flavour)
    if f.mechanism == settings.ECHO:
        return agreement(problem, reading(sample, f.feature, problem, 0.0))
    return skillful(problem, design(sample, f.feature), flavour)


def screened(
    name: str, source: str, frame: pandas.DataFrame, task: str
) -> Case:
    """Runs the product's screen once; its report is the case's product."""
    cfg = leakage.CERTIFIED
    found = screen.screen(frame, "target", task, cfg)
    return Case(
        name,
        source,
        {"frame": frame, "task": task, "cfg": cfg, "report": found},
    )


def findings(case: Case) -> Values:
    """The screen report's values that references recompute."""
    out: Values = {}
    for f in case.data["report"].findings:
        key = f"{f.feature}|{f.mechanism}"
        if f.mechanism in RECOUNTED:
            out[f"{key}|statistic"], out[f"{key}|p"] = f.statistic, f.p_value
        out[f"{key}|adjusted"] = f.adjusted_p
        out[f"{key}|significant"] = float(f.significant)
        if f.significant and f.mechanism in RECOUNTED and f.lower is not None:
            out[f"{key}|bound"] = f.lower
    return out


# Design of the plants (not thresholds): the share of the upper outcome's
# rows whose value is removed, and the share of an echo's labels flipped.
HOLES = 0.3
FLIPS = 0.05


def planted(
    x: pandas.DataFrame, y: pandas.Series, name: str, column: str, index: int
) -> pandas.DataFrame:
    """A table with a value missing for more rows of the upper outcome."""
    rng = stream(name, index)
    high = (y > y.median()) if y.dtype.kind == "f" else (y == y.max())
    frame = x.assign(target=y)
    frame.loc[high & (rng.uniform(size=len(y)) < HOLES), column] = numpy.nan
    return frame


def terms(case: Case) -> Values:
    """The thresholdout's derived terms of the case's parameters."""
    p = case.data
    found = thresholdout.Terms(**p)
    rest = {
        k: p[k]
        for k in ("tolerance", "confidence", "queries", "budget", "step")
    }
    return {
        "transcripts": thresholdout.transcripts(
            p["queries"], p["budget"], p["step"]
        ),
        "floor": thresholdout.floor(p["confidence"], p["step"]),
        "sampling": found.sampling,
        "threshold": found.threshold,
        "noise": found.noise,
        "minimum": float(thresholdout.minimum(**rest)),
    }


def derive(case: Case, flavour: str) -> Values:
    """The thresholdout terms from their derivation.

    `scratch` counts transcripts exactly in integers and solves each bound
    in closed form; `scipy` uses `logsumexp` over `gammaln` and root
    finding (`brentq`) for the threshold, floor and the fewest rows.
    """
    p = case.data
    q, b, step, c = p["queries"], p["budget"], p["step"], p["confidence"]
    levels = math.floor(1 / step) + 1
    g, ratio = math.log(2 / c), thresholdout.RATIO
    if flavour == "scratch":
        count = math.log(sum(math.comb(q, k) * levels**k for k in range(b + 1)))
    else:
        k = numpy.arange(b + 1)
        count = float(
            special.logsumexp(
                special.gammaln(q + 1)
                - special.gammaln(k + 1)
                - special.gammaln(q - k + 1)
                + k * math.log(levels)
            )
        )
    spread = count + math.log(4 * q / c)
    sampling = math.sqrt(spread / (2 * p["size"]))
    if flavour == "scratch":
        least = (1 + 2 * g / ratio) * step / 2 / (1 + g / ratio)
        threshold = (p["tolerance"] - sampling) / (1 + 2 * g / ratio)
        slack = p["tolerance"] - least
        rows = 1
        while math.sqrt(spread / (2 * rows)) > slack:
            rows *= 2
        low = rows // 2
        while rows - low > 1:
            middle = (low + rows) // 2
            low, rows = (
                (middle, rows)
                if math.sqrt(spread / (2 * middle)) > slack
                else (low, middle)
            )
    else:
        edge = optimize.brentq(
            lambda t: t * (1 + g / ratio) - step / 2,
            0,
            1,
            xtol=1e-16,
            rtol=1e-15,
        )
        least = edge * (1 + 2 * g / ratio)
        threshold = optimize.brentq(
            lambda t: t * (1 + 2 * g / ratio) + sampling - p["tolerance"],
            -1,
            1,
            xtol=1e-16,
            rtol=1e-15,
        )
        root = optimize.brentq(
            lambda n: math.sqrt(spread / (2 * n)) - (p["tolerance"] - least),
            1e-9,
            1e12,
            rtol=1e-15,
        )
        rows = math.ceil(root - 1e-9 * root)
    return {
        "transcripts": count,
        "floor": least,
        "sampling": sampling,
        "threshold": threshold,
        "noise": threshold / ratio,
        "minimum": float(rows),
    }


def sessions() -> list[Case]:
    """Thresholdout parameters: the defaults and two other sessions."""
    defaults = {
        "tolerance": thresholdout.TOLERANCE,
        "confidence": thresholdout.CONFIDENCE,
        "queries": thresholdout.QUERIES,
        "budget": thresholdout.BUDGET,
        "step": thresholdout.STEP,
    }
    variants: tuple[tuple[str, dict[str, Any]], ...] = (
        ("defaults", {}),
        ("short", {"queries": 20, "budget": 5, "step": 0.05}),
        ("long", {"queries": 1000, "budget": 100, "step": 0.001}),
    )
    return [
        Case(
            f"{name} {size} rows",
            "derivation",
            {**defaults, **extra, "size": size},
        )
        for name, extra in variants
        for size in (
            thresholdout.minimum(**extra),
            3 * thresholdout.minimum(**extra),
        )
    ]


# The error rate of guessing between two outcomes.
COIN = 0.5


def support(flavour: str) -> Callable[[Case], Values]:
    """Fewest test rows of a class: an exact bound with no error below 1/2.

    `scratch` searches `1 - alpha^(1/m) < 1/2`; `scipy` searches the
    Clopper-Pearson upper limit `beta.ppf(1 - alpha, 1, m) < 1/2`.
    """

    def run(case: Case) -> Values:
        alpha, m = case.data["alpha"], 1
        while (
            1 - alpha ** (1 / m)
            if flavour == "scratch"
            else stats.beta.ppf(1 - alpha, 1, m)
        ) >= COIN:
            m += 1
        return {"rows": float(m)}

    return run


def divergence(flavour: str) -> Callable[[Case], Values]:
    """The ONNX gate's float32 tolerance and its fewest held-out rows.

    `scratch`: `gamma_n` in exact fractions with `u = 2^-24`, and the
    smallest `n` with `(1 - RATE)^n <= 1 - CONFIDENCE` exactly; `scipy`:
    `u` from `numpy.finfo(float32).epsneg` and `binom.pmf(0, n, RATE)`.
    """

    def run(case: Case) -> Values:
        n, rate, confidence = (
            case.data["operations"],
            case.data["rate"],
            case.data["confidence"],
        )
        if flavour == "scratch":
            unit = Fraction(1, 2**24)
            tolerance = float(n * unit / (1 - n * unit))
            rows = 1
            while (1 - Fraction(rate)) ** rows > 1 - Fraction(confidence):
                rows += 1
        else:
            rounding = float(numpy.finfo(numpy.float32).epsneg)
            tolerance = n * rounding / (1 - n * rounding)
            rows = 1
            while stats.binom.pmf(0, rows, rate) > 1 - confidence:
                rows += 1
        return {"tolerance": tolerance, "rows": float(rows)}

    return run


def missings() -> list[Case]:
    """Missingness tables: planted holes in real tables, simulated ones."""
    found = []
    for loader, task in (
        ("load_breast_cancer", "classification"),
        ("load_wine", "classification"),
        ("load_diabetes", "regression"),
    ):
        x, y = bundled(loader)
        frame = planted(x, y, loader, x.columns[0], 0)
        found.append(
            Case(
                f"{loader[5:]} planted",
                "real",
                {"feature": frame[x.columns[0]], "target": y, "task": task},
            )
        )
    return found


def twins(task: tasks.Task) -> list[Case]:
    """Both simulated inputs of a task: plain and `TIED`."""
    return [simulate(task, 0), simulate(task, TIED)]


SIMULATED: dict[str, Callable[[], list[Case]]] = {
    **{f"metrics.{t.value}": functools.partial(twins, t) for t in tasks.Task},
    "missingness": lambda: [
        holes(
            "fisher 2x2",
            "simulated",
            pandas.DataFrame({"a": numpy.arange(40.0)}),
            pandas.Series([0, 1] * 20),
            numpy.array([1, 3, 5]),
            "classification",
        ),
        holes(
            "chi-square 2x3",
            "simulated",
            pandas.DataFrame({"a": numpy.arange(90.0)}),
            pandas.Series([0, 1, 2] * 30),
            numpy.arange(0, 90, 4),
            "classification",
        ),
    ],
    "screen": lambda: [
        screened("signal and echo", "simulated", toy(), "classification")
    ],
    "latency": lambda: timings(),  # noqa: PLW0108  # defined further down
}


def toy() -> pandas.DataFrame:
    """A small simulated table: noise, one strong column, an echo."""
    rng = stream("screen", 0)
    x = pandas.DataFrame(
        rng.normal(size=(SIZE, 4)), columns=["n1", "n2", "n3", "n4"]
    )
    y = (x["n1"] + rng.normal(scale=0.3, size=SIZE) > 0).astype(int)
    copy = numpy.where(rng.uniform(size=SIZE) < FLIPS, 1 - y, y)
    return x.drop(columns="n1").assign(strong=x["n1"], copy=copy, target=y)


def actuals() -> list[Case]:
    """Real tables of the screen: plants of an echo and of missingness."""
    x, y = bundled("load_breast_cancer")
    rng = stream("screen.real", 0)
    cancer = planted(x, y, "screen.cancer", "mean radius", 0).assign(
        copy=numpy.where(rng.uniform(size=len(y)) < FLIPS, 1 - y, y)
    )
    wine = bundled("load_wine")
    x, y = bundled("load_diabetes")
    diabetes = planted(x, y, "screen.diabetes", "bmi", 0)
    return [
        screened("breast_cancer with plants", "real", cancer, "classification"),
        screened(
            "wine", "real", wine[0].assign(target=wine[1]), "classification"
        ),
        screened("diabetes with a plant", "real", diabetes, "regression"),
    ]


def simulated(name: str) -> list[Case]:
    """The simulated cases of a statistic (none for a derivation)."""
    return SIMULATED.get(name, list)()


def statistic(
    name: str,
    real: Callable[[], list[Case]],
    product: Callable[[Case], Values],
    references: Mapping[str, Callable[[Case], Values]],
) -> Statistic:
    """A statistic measured on its real and its simulated cases."""
    return Statistic(
        name, lambda: real() + simulated(name), product, references
    )


def threshold(case: Case) -> Values:
    """The product's `MIN_TEST_CLASS_ROWS`, at its `EVIDENCE_ALPHA`.

    Raises:
        InputError: If the case is at another level than the product's.
    """
    if case.data["alpha"] != partitions.EVIDENCE_ALPHA:
        raise errors.InputError("the support case is not at EVIDENCE_ALPHA")
    return {"rows": float(partitions.MIN_TEST_CLASS_ROWS)}


def gate(case: Case) -> Values:
    """The ONNX gate's `DEFAULT_TOLERANCE` and `ROWS`, at its parameters.

    Raises:
        InputError: If the case's parameters are not the product's.
    """
    own = {
        "operations": onnx_export.OPERATIONS,
        "rate": onnx_export.RATE,
        "confidence": onnx_export.CONFIDENCE,
    }
    if dict(case.data) != own:
        raise errors.InputError("the onnx case is not at the gate's parameters")
    return {
        "tolerance": onnx_export.DEFAULT_TOLERANCE,
        "rows": float(onnx_export.ROWS),
    }


def catalog(case: Case) -> Values:
    """Every report metric of the case's task, as a run reports it."""
    d = case.data
    return metrics.CATALOG.evaluate(
        d["estimator"], d["features"], d["target"], d["task"]
    )


class Clock:
    """Stands in for `time` inside `resources`: records or replays ticks.

    Attributes:
        ticks: The `perf_counter` readings, recorded or to replay.
        replay: Whether the ticks are replayed rather than recorded.
        position: How many ticks have been replayed.
    """

    def __init__(self, ticks: Sequence[float] | None = None) -> None:
        """Records real readings when `ticks` is None, else replays them."""
        self.ticks = list(ticks or ())
        self.replay = ticks is not None
        self.position = 0

    def perf_counter(self) -> float:
        """The next tick: a real reading when recording, else the stored one.

        Raises:
            InputError: If a replay asks for more ticks than were stored.
        """
        if not self.replay:
            tick = time.perf_counter()
            self.ticks.append(tick)
            return tick
        if self.position >= len(self.ticks):
            raise errors.InputError("the latency case ran out of ticks")
        self.position += 1
        return self.ticks[self.position - 1]


class Idle:
    """An estimator whose prediction takes no time of its own."""

    def predict(self, features: pandas.DataFrame) -> numpy.ndarray:
        """Zeros, one per row."""
        return numpy.zeros(len(features))


def timed(
    clock: Clock,
    rows: int,
    estimator: Any = None,
    features: pandas.DataFrame | None = None,
) -> resources.Usage:
    """Runs the product's `resources.measure` with `clock` as its timer.

    Args:
        clock: The timer `measure` reads; only `resources` sees it.
        rows: How many single-row predictions `measure` times.
        estimator: What is timed; `Idle` when None.
        features: The rows predicted; `rows` empty rows when None.
    """
    model = Idle() if estimator is None else estimator
    frame = (
        pandas.DataFrame(index=range(rows)) if features is None else features
    )
    with mock.patch.object(resources, "time", clock):
        return resources.measure(model, frame, rows)


def ticks(spans: Sequence[float]) -> list[float]:
    """Timer readings whose start-to-end gaps are `spans` milliseconds."""
    out: list[float] = []
    now = 0.0
    for span in spans:
        out += [now, now + span / 1000.0]
        now += span / 1000.0
    return out


def spans(readings: Sequence[float]) -> list[float]:
    """The milliseconds between each start and end reading, as timed."""
    return [
        (readings[i + 1] - readings[i]) * 1000.0
        for i in range(0, len(readings) - 1, 2)
    ]


def recorded() -> Case:
    """A real timing vector: `measure` on a ridge fitted to diabetes."""
    x, y = bundled("load_diabetes")
    clock = Clock()
    timed(clock, resources.LATENCY_ROWS, linear_model.Ridge().fit(x, y), x)
    return Case("diabetes ridge", "real", {"ticks": clock.ticks})


def latency(case: Case) -> Values:
    """The product's p95 latency, replaying the case's timer readings."""
    readings = case.data["ticks"]
    usage = timed(Clock(readings), len(readings) // 2)
    return {"p95": usage.p95_latency_ms}


def percentile(flavour: str) -> Callable[[Case], Values]:
    """The 95th percentile of the case's spans, by hand or by SciPy."""

    def compute(case: Case) -> Values:
        found = spans(case.data["ticks"])
        if flavour == "scratch":
            return {"p95": quantile(found, 0.95)}
        return {
            "p95": float(
                stats.mstats.mquantiles(found, 0.95, alphap=1, betap=1)[0]
            )
        }

    return compute


# Timing vectors a serving gate can meet: one row, two rows, all tied, and
# the 95th percentile landing exactly on (21 rows) or between (30 rows) tied
# order statistics; `timings` adds drawn vectors with ties at 0.1 and 1 ms.
FIXED = {
    "one row": [3.7],
    "two rows": [1.0, 4.0],
    "all tied": [1.2] * resources.LATENCY_ROWS,
    "order statistic at 21": [1.0] * 18 + [2.0, 2.0, 5.0],
    "between ties at 30": [1.0] * 27 + [2.0, 2.0, 8.0],
}


def timings() -> list[Case]:
    """The timing vectors of `FIXED` and two drawn ones, all with ties."""
    generator = stream("latency", 0)
    drawn = {
        "lognormal at 0.1 ms": numpy.round(
            generator.lognormal(0.0, 0.5, resources.LATENCY_ROWS), 1
        ),
        "heavy tail at 1 ms": numpy.round(generator.pareto(2.0, 100) + 1.0),
    }
    vectors = FIXED | {k: [float(v) for v in d] for k, d in drawn.items()}
    return [
        Case(name, "simulated", {"ticks": ticks(vector)})
        for name, vector in vectors.items()
    ]


SCRATCH = {
    tasks.Task.BINARY: labelled,
    tasks.Task.MULTICLASS: labelled,
    tasks.Task.REGRESSION: numeric,
    tasks.Task.ORDINAL: numeric,
    tasks.Task.MULTILABEL: numeric,
    tasks.Task.MULTIOUTPUT: numeric,
    tasks.Task.FORECAST: walk,
    tasks.Task.UPLIFT: walk,
    tasks.Task.RANKING: ranked,
    tasks.Task.SURVIVAL: censored,
}
STATISTICS = (
    *(
        statistic(
            f"metrics.{t.value}",
            ACTUAL[t],
            catalog,
            {"scratch": SCRATCH[t]},
        )
        for t in tasks.Task
    ),
    statistic(
        "multiplicity",
        families,
        adjusting(("bh", "holm", "bonferroni")),
        {
            "scratch": defined(("bh", "holm", "bonferroni")),
            "scipy": library(("bh",)),
        },
    ),
    statistic(
        "multiplicity.by",
        families,
        adjusting(("by",)),
        {"scratch": defined(("by",)), "scipy": library(("by",))},
    ),
    statistic(
        "clopper",
        counts,
        bounds(coverage.clopper_pearson),
        {"scratch": bounds(exact), "scipy": bounds(binomial)},
    ),
    statistic(
        "missingness",
        missings,
        tested,
        {
            "scratch": lambda c: association(c, "scratch"),
            "scipy": lambda c: association(c, "scipy"),
        },
    ),
    statistic(
        "screen",
        actuals,
        findings,
        {
            "scratch": lambda c: recount(c, "scratch"),
            "sklearn": lambda c: recount(c, "sklearn"),
        },
    ),
    statistic(
        "thresholdout",
        sessions,
        terms,
        {
            "scratch": lambda c: derive(c, "scratch"),
            "scipy": lambda c: derive(c, "scipy"),
        },
    ),
    statistic(
        "support",
        lambda: [
            Case("policy", "derivation", {"alpha": partitions.EVIDENCE_ALPHA})
        ],
        threshold,
        {"scratch": support("scratch"), "scipy": support("scipy")},
    ),
    statistic(
        "latency",
        lambda: [recorded()],
        latency,
        {"scratch": percentile("scratch"), "scipy": percentile("scipy")},
    ),
    statistic(
        "onnx",
        lambda: [
            Case(
                "float32",
                "derivation",
                {
                    "operations": onnx_export.OPERATIONS,
                    "rate": onnx_export.RATE,
                    "confidence": onnx_export.CONFIDENCE,
                },
            )
        ],
        gate,
        {"scratch": divergence("scratch"), "scipy": divergence("scipy")},
    ),
)
SCREEN = ("screen", "multiplicity.by", "missingness", "clopper")
DECIDING: dict[str, tuple[str, ...]] = {
    "leakage.policy": SCREEN,
    "admissibility.quarantine_all": SCREEN,
    "confirm.reuse_exhausted": ("thresholdout",),
    "split.test_support": ("support", "clopper"),
    "export.onnx.divergence": ("onnx",),
    "select.serving_limits": ("latency",),
    "select.pareto_front": ("latency",),
}
COUNT = (
    "compares counts of rows, classes, groups or windows with fixed minimums"
)
PRESENCE = "checks the presence, type or values of a declared column or setting"
CUSTODY = (
    "checks the custody record of a sealed set (token, seal, digest, journal)"
)
SPEND = "compares measured or estimated seconds or bytes with a budget (C1, C7)"
SUMS = "reproduces declared totals and key uniqueness exactly (C5, C7)"
PREDICATES: dict[str, str] = {
    "ingest.cli.unit_verdict": SUMS,
    "ingest.stage.run": SUMS,
    "ingest.reconcile": SUMS,
    "policy.checkpoints": "asks the registered checkpoints for approval",
    "data.unlabeled": PRESENCE,
    "profile.monotonic_columns": PRESENCE,
    "profile.target_and_features": PRESENCE,
    "split.task_supported": PRESENCE,
    "budget.minimum_run": SPEND,
    "train.first_trial": SPEND,
    "confirm.single_use": CUSTODY,
    "confirm.reuse_journal": CUSTODY,
    "split.resident.bind": CUSTODY,
    "split.resident.open": CUSTODY,
    "split.resident.reopen": CUSTODY,
    "split.resident.journal": CUSTODY,
    "split.resident.records": CUSTODY,
    "vault.bind": CUSTODY,
    "vault.open": CUSTODY,
    "split.event.folds": COUNT,
    "split.event.stratify": COUNT,
    "split.group.class_complete_folds": COUNT,
    "split.group.partition": COUNT,
    "split.rolling.partition": COUNT,
    "split.stratified.folds": COUNT,
    "split.stratified.stratify": COUNT,
    "split.temporal.holdout": COUNT,
    "split.temporal.times": PRESENCE,
    "split.temporal.entities": COUNT,
    "split.temporal.partition": COUNT,
    "split.treatment.folds": COUNT,
    "split.treatment.stratify": COUNT,
    "task.resolve": PRESENCE,
    "task.auto": PRESENCE,
    "task.classification": PRESENCE,
    "task.ordinal": PRESENCE,
    "task.ranking": PRESENCE,
    "task.regression": PRESENCE,
    "task.survival": PRESENCE,
    "task.uplift": PRESENCE,
    "target.events": COUNT,
    "target.columns": PRESENCE,
    "target.treatment": COUNT,
    "plan.gpu_required": PRESENCE,
    "plan.monotonic": PRESENCE,
    "estimate.refusal": "the estimator's own refusal to fit the declared model",
}
RESIDUE = {
    "deciding": [
        "leakage screen: p-values of the encoding and duplication hypotheses",
        (
            "leakage screen: the paired margin a skill suspect keeps over the "
            "rest of its table"
        ),
        (
            "export.onnx.divergence: the probability gap itself (needs "
            "onnxruntime and an exported graph)"
        ),
    ],
    "reported": [
        (
            "metric intervals (Uncertainty.compute) and the diagnostic reports "
            "(calibration, fairness, drift, robustness, slices, learning "
            "curves, parameter estimates)"
        ),
        (
            "forecast, uplift and survival metrics on real data (no such table "
            "ships offline)"
        ),
    ],
}


def reported() -> set[str]:
    """Every metric name a run reports, over all tasks."""
    return {n for names in metrics.CATALOG.report_names.values() for n in names}


def tolerance() -> float:
    """`reference_rel_tol` from `acceptance.toml`."""
    return acceptance.load()[VARIABLE].value


def fingerprint() -> str:
    """Hash of this module and the product code it compares."""
    digest = hashlib.sha256(pathlib.Path(__file__).read_bytes())
    digest.update(evidence.source(MEASURED, ROOT / "scmle").encode())
    return digest.hexdigest()


def measure(names: Sequence[str] | None = None) -> dict[str, Any]:
    """Compares the chosen statistics (all when None) on all their cases.

    Raises:
        InputError: If a name is not a statistic.
    """
    known = {s.name: s for s in STATISTICS}
    unknown = sorted(set(names or ()) - set(known))
    if unknown:
        raise errors.InputError(
            f"unknown statistic {unknown}; known: {sorted(known)}"
        )
    chosen = [known[n] for n in (names or known)]
    deciding = {n for group in DECIDING.values() for n in group}
    limit = tolerance()
    rows = [
        dataclasses.asdict(c)
        for s in chosen
        for case in s.cases()
        for c in compare(s, case, limit, deciding=s.name in deciding)
    ]
    seen = {r["key"] for r in rows if r["statistic"].startswith("metrics.")}
    unread = [
        f"reported metric {n}: never compared"
        for n in sorted(reported() - seen)
    ]
    return {
        "version": 1,
        "claim": "C3",
        "code": fingerprint(),
        "tolerance": {"name": VARIABLE, "value": limit},
        "comparisons": rows,
        "residue": {
            "reported": unread + RESIDUE["reported"] + RESIDUE["deciding"],
            "deciding": list(RESIDUE["deciding"]),
        },
    }


def read(document: dict[str, Any] | None) -> dict[str, Any] | None:
    """The given document, or the stored one; None when nothing is stored."""
    if document is not None:
        return document
    if not PATH.exists():
        return None
    loaded: dict[str, Any] = json.loads(PATH.read_text(encoding="utf-8"))
    return loaded


def problems(document: dict[str, Any] | None = None) -> list[str]:
    """Why the document cannot decide the clauses; empty when it can."""
    found = read(document)
    if found is None:
        return [f"statistics: {PATH.name} is missing"]
    issues = []
    if found.get("code") != fingerprint():
        issues.append("statistics: measured on other code; run --write")
    if found.get("tolerance", {}).get("value") != tolerance():
        issues.append(
            f"statistics: measured at another {VARIABLE}; run --write"
        )
    return issues


def summary(
    document: dict[str, Any] | None = None, scope: str = "reported"
) -> tuple[str, str]:
    """The scorecard's value and verdict of one clause.

    Args:
        document: The document; read from `PATH` when None.
        scope: `reported` (every reported statistic within the tolerance
            of a reference) or `deciding` (gate statistics also matching a
            second implementation).

    Returns:
        The value text and "pass", "fail" or "not measured" (see the
        module docstring).

    Raises:
        InputError: If the scope is neither `reported` nor `deciding`.
    """
    if scope not in ("reported", "deciding"):
        raise errors.InputError(
            f"unknown scope {scope!r}; known: reported, deciding"
        )
    found = read(document)
    if found is None:
        return "no measurement", "not measured"
    issues = problems(found)
    if issues:
        return "; ".join(issues), "not measured"
    rows = [
        c for c in found["comparisons"] if scope == "reported" or c["deciding"]
    ]
    if not rows:
        return "no comparison", "not measured"
    inputs = {
        s: {(c["statistic"], c["case"]) for c in rows if c["source"] == s}
        for s in ("real", "simulated", "derivation")
    }
    beyond = [c for c in rows if not c["within"]]
    names = {c["statistic"] for c in rows}
    largest = max(c["error"] for c in rows)
    limit = found["tolerance"]["value"]
    value = (
        f"{len(rows)} values of {len(names)} statistics on "
        f"{len(inputs['real'])} real, {len(inputs['simulated'])} simulated "
        f"and {len(inputs['derivation'])} derivation inputs; largest error "
        f"{largest:.2g} (tolerance {limit:g})"
    )
    if beyond:
        named = ", ".join(
            f"{c['statistic']} {c['case']} {c['key']} ({c['error']:.2g})"
            for c in beyond[:6]
        )
        return f"{value}; {len(beyond)} beyond: {named}", "fail"
    left = found["residue"][scope]
    if left:
        return f"{value}; not compared: {'; '.join(left)}", "not measured"
    return value, "pass"


def main(arguments: list[str] | None = None) -> int:
    """Shows (`--show`), writes (`--write`) or checks (`--check`).

    Args:
        arguments: Command-line arguments; None reads `sys.argv`.

    Returns:
        0 after `--show` or `--write`; for `--check`, 0 when `problems` is
        empty and 1 otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--show", action="store_true")
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    parser.add_argument("--statistic", action="append", metavar="NAME")
    options = parser.parse_args(arguments)
    if options.check:
        issues = problems()
        print(json.dumps({"ok": not issues, "problems": issues}, indent=2))
        return 0 if not issues else 1
    if options.write and options.statistic:
        parser.error("--write measures every statistic")
    document = measure(options.statistic)
    verdicts = {s: summary(document, s) for s in ("reported", "deciding")}
    if options.write:
        PATH.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(verdicts, indent=2))
        return 0
    print(
        json.dumps({**document, "verdicts": verdicts}, indent=2, sort_keys=True)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
