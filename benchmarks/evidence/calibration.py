"""Claim C3: every interval and test nominal by simulation-based calibration.

Each `Scenario` simulates data from a known truth, runs one interval or test
of the product exactly as a user would call it, and records what the product
reported against that truth. Its `Check`s judge the records:

* `Rate` of kind `coverage`: the share of intervals holding the truth;
  nominal is the level the product promises (for conformal margins the
  finite-sample rank `attained`, for class-conditional sets its mixture
  over class sizes, `mondrian`).
* `Rate` of kind `size`: the share of rejections when the null holds;
  nominal is the level of the test, or for a Holm family the exact chance
  that any exact test of the family rejects (`holm`).
* `Uniform` (rank): the null p-value's complement, or the rank of a new
  conformal score among the calibration scores, binned into `BINS`
  equal-probability bins; a calibrated procedure fills every bin evenly.

Exact bounds. Every rate gets a two-sided Clopper-Pearson interval at
confidence `1 - level`, and every bin of a rank one at `1 - level / BINS`,
with `level = alpha / family` (Bonferroni over every check of every
scenario, `alpha` from acceptance.toml), so a registry of perfectly nominal
procedures reports a miscalibration with chance at most `alpha`. A check is
`nominal` when its interval holds the nominal value; otherwise it is
`liberal` (intervals too narrow, tests too ready to reject), `conservative`
(the reverse) or `nonuniform`. A trial the product refuses counts against
it: a miss for coverage, a rejection for size, and `errored` for a rank.

Sample. `DELTA` is the miscalibration every check must detect with chance
`power` (acceptance.toml): a coverage that far below nominal, a size that
far above it, or a rank bin that much heavier than its share. `planned`
finds that count by an exact search over the binomial (`chance`, `search`).

The row passes only when every check is nominal and `NOT_MEASURED` names no
product interval or test; with every check nominal but some surface still
unmeasured it is `partial`. Any other state goes to the document's
`miscalibrated` list with the product location it measured. A field a trial
ran but did not report counts against the product like a refusal and is
named in the scenario's messages.

NOT measured: the product surfaces in `NOT_MEASURED`, by path; `LIMITS`.

    python -m benchmarks.evidence.calibration --write
    python -m benchmarks.evidence.calibration --check
"""

import argparse
from collections.abc import Callable
from collections.abc import Sequence
from concurrent import futures
import dataclasses
import functools
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any
import zlib

import numpy
import pandas
from scipy import stats
import sklearn
from sklearn import linear_model

from benchmarks import acceptance
from benchmarks import evidence
from benchmarks.evidence import coverage
from scmle.assurance.admissibility import leakage
from scmle.assurance.admissibility import screen
from scmle.assurance.charter import causal
from scmle.assurance.detectors import hashing
from scmle.causal.identification import aipw
from scmle.causal.identification import did
from scmle.causal.identification import iv
from scmle.causal.identification import overid
from scmle.causal.identification import rd
from scmle.causal.identification import synth
from scmle.causal.identification.run import adjustment
from scmle.causal.identification.run import strata
from scmle.contracts import errors
from scmle.evaluation import confidence
from scmle.evaluation import conformal
from scmle.evaluation import cv_plus
from scmle.evaluation import parameter_estimation
import scmle.evaluation.coverage

PATH = Path(__file__).with_suffix(".json")
PACKAGE = Path(__file__).resolve().parents[2] / "scmle"
# Every product package a scenario runs; a change to any file in them can
# move a record, so the stored result goes stale.
PRODUCT = (
    "assurance",
    "causal",
    "contracts",
    "evaluation",
    "models",
    "preparation",
    "runtime",
)
# Half the gap between the conventional levels 0.90 and 0.95: an interval
# whose coverage is off by more reports closer to another conventional
# level than its own, and a size 0.075 is half again the 0.05 promised.
DELTA = 0.025
# Rank bins: ten keeps each bin's share (0.1) four times DELTA, so a bin
# DELTA heavy is a 25 percent excess, and divides the 200-value rank grids
# of the conformal and permutation scenarios into whole bins.
BINS = 10
# A rank on a grid k / m lands on a bin edge when m is a multiple of BINS;
# 1 - p can round to just below that edge (1 - 0.8 = 0.19999999999999996).
# EDGE lifts such ranks back: it is ~1e6 times the double rounding error of
# a rank times BINS (~1e-15) and ~1e-6 of the finest grid step used (1/200),
# so no rank moves past an edge it truly lies below.
EDGE = 1e-9
STATES = ("nominal", "liberal", "conservative", "nonuniform", "errored")
STREAM = 3
# Bootstrap trials reuse `coverage.run`, seeded by its trial number; seeds
# from OFFSET up never meet that module's 0..1677 development trials.
OFFSET = 2**20
LEVEL = 0.95
# n + 1 = 200 calibration scores: (n + 1) * 0.9 and (n + 1) * 0.85 are whole,
# so the split-conformal coverage is exactly nominal, not just above it.
CALIBRATION = 199
TRAIN = 100
FEATURES = 5
WEIGHTS = numpy.linspace(1.0, 0.2, FEATURES)
SHARES = (0.5, 0.3, 0.2)
CENTRES = numpy.array([[0.0, 0.0], [1.5, 0.0], [0.0, 1.5]])
SHIFT = 0.5
FOLDS = 5
VALIDATION_ROWS = 200
RD_ROWS = 500
JUMP = 1.0
EFFECT = 1.0
COHORT = 20
PERIODS = 6
RANDOM_ROWS = 200
SUBGROUP = 120
GROUPS = ("a", "b", "c")
CAUSAL_ROWS = 500
MODEL_ROWS = 200
BAYES_ROWS = 50
PROBE_ROWS = 200
CLUSTER = 5
SCREEN_ROWS = 300
GAPS = 0.2
PREVALENCE = 0.4
SLICES = 4
SLICE_ROWS = 100
AIPW_ROWS = 500
IV_ROWS = 500
# A first-stage slope of 0.5 on 500 rows gives an expected robust F near
# 1 + 500 * 0.25 = 126, far above iv.F_THRESHOLD; 0.05 gives one near 2,
# far below it, where the Anderson-Rubin set is the headline.
STRONG = 0.5
WEAK = 0.05
INSTRUMENTS = 3
# Endogeneity: the correlation of the structural and first-stage errors.
ENDOGENEITY = 0.5
# The fewest donors that can reach p <= 1 - LEVEL (synth.quorum).
DONORS = synth.quorum(1 - LEVEL)
# More pre-periods than donors: the treated unit then lies outside the
# donors' hull with chance one, so no fit is exact and no ratio infinite.
PRE = 2 * DONORS
POST = 5
# Product intervals and tests no scenario calibrates, `path:name` and what;
# while any is listed the row is `partial`, never `pass`.
NOT_MEASURED: tuple[tuple[str, str], ...] = (
    (
        "scmle/evaluation/bootstrap.py:Uncertainty.compute",
        (
            "metrics other than each task's default, multiclass log loss and "
            "temporal block resampling"
        ),
    ),
    (
        "scmle/causal/identification/did.py:study",
        "the event-study coefficients and their covariance",
    ),
    (
        "scmle/causal/identification/did.py:Estimate.sensitivity",
        "the relative-magnitudes set",
    ),
    (
        "scmle/causal/identification/rd.py:falsify",
        "the effect at bandwidth multiples",
    ),
    (
        (
            "scmle/evaluation/parameter_estimation/hierarchical.py:"
            "fit_hierarchical_estimator"
        ),
        "hierarchical coefficient intervals",
    ),
    (
        (
            "scmle/evaluation/parameter_estimation/continuous.py:"
            "fit_continuous_estimator"
        ),
        "continuous-treatment effect intervals",
    ),
    (
        "scmle/evaluation/sequential/estimator.py:Estimator",
        "the sustained-treatment contrast interval",
    ),
    ("scmle/assurance/ledger/online_fdr/rules.py:Lond", "online FDR"),
    ("scmle/assurance/ledger/online_fdr/rules.py:Lord", "online FDR"),
    ("scmle/assurance/ledger/online_fdr/rules.py:Saffron", "online FDR"),
    (
        "scmle/monitoring/sequential_tests.py:Betting",
        "the anytime-valid betting test",
    ),
    (
        "scmle/monitoring/sequential_tests.py:Mixture",
        "the restarted betting mixture",
    ),
    (
        "scmle/monitoring/sequential_tests.py:Confidence",
        "the time-uniform confidence sequence",
    ),
    (
        "scmle/monitoring/sequential_tests.py:Comparison",
        "the promotion and rollback e-process",
    ),
    (
        "scmle/decision/anytime.py:confidence_bound",
        "the anytime-valid confidence bound",
    ),
    (
        "scmle/assurance/admissibility/leakage/__init__.py:condition",
        "the conditional permutation test",
    ),
    (
        "scmle/assurance/admissibility/screen/fitting.py:skill_bound",
        "the screen's skill lower bound",
    ),
    (
        "scmle/assurance/admissibility/screen/association.py:dominance",
        "the record-purity lower bound",
    ),
)
# Not a product surface, so it never makes the row partial.
LIMITS = ("real data: every scenario is simulated",)


class Check:
    """A judgement of one recorded field over the trials of a scenario."""

    field: str
    location: str

    def judge(
        self, values: Sequence[float | None], level: float
    ) -> dict[str, Any]:
        """Decides whether the records are nominal.

        Args:
            values: One record per trial; None where the trial errored.
            level: Error rate of the exact bound.

        Returns:
            The counts, the exact bound and the state.

        Raises:
            NotImplementedError: In the base class.
        """
        raise NotImplementedError

    def plan(self, level: float) -> int:
        """Trials that detect a `DELTA` miscalibration with chance `power`.

        Args:
            level: Error rate of the exact bound.

        Returns:
            The trial count.

        Raises:
            NotImplementedError: In the base class.
        """
        raise NotImplementedError

    def chance(self, trials: int, level: float) -> float:
        """Exact chance of detecting a `DELTA` miscalibration.

        Args:
            trials: Trials run.
            level: Error rate of the exact bound.

        Returns:
            The probability that the bound excludes the nominal value.

        Raises:
            NotImplementedError: In the base class.
        """
        raise NotImplementedError


@dataclasses.dataclass(frozen=True)
class Rate(Check):
    """A coverage or a size against its nominal rate.

    Attributes:
        field: The 0/1 record judged.
        nominal: The rate the product promises.
        liberal: -1 when a low rate is the unsafe side (coverage), +1 when
            a high rate is (size).
        location: The product code that produced the record, when narrower
            than the scenario's.
    """

    field: str
    nominal: float
    liberal: int
    location: str = ""

    @property
    def kind(self) -> str:
        """`coverage` or `size`."""
        return "coverage" if self.liberal < 0 else "size"

    def judge(
        self, values: Sequence[float | None], level: float
    ) -> dict[str, Any]:
        """Counts the hits and bounds their rate exactly.

        Args:
            values: One 0/1 record per trial; None where the trial errored,
                counted against the product.
            level: Two-sided error rate of the Clopper-Pearson interval.

        Returns:
            The kind, nominal rate, count, trials, rate, bounds, errors and
            state.
        """
        against = 1.0 if self.liberal > 0 else 0.0
        flags = [against if v is None else v for v in values]
        count, trials = int(sum(flags)), len(flags)
        low, high = scmle.evaluation.coverage.clopper_pearson(
            count, trials, 1 - level
        )
        state = "nominal"
        if high < self.nominal:
            state = "liberal" if self.liberal < 0 else "conservative"
        elif low > self.nominal:
            state = "liberal" if self.liberal > 0 else "conservative"
        return {
            "field": self.field,
            "kind": self.kind,
            "nominal": self.nominal,
            "count": count,
            "trials": trials,
            "rate": count / trials if trials else None,
            "low": low,
            "high": high,
            "errors": sum(v is None for v in values),
            "state": state,
        }

    def plan(self, level: float) -> int:
        """Trials that see the rate `DELTA` off on its unsafe side.

        Args:
            level: Two-sided error rate of the bound.

        Returns:
            The planned count (`search`).
        """
        truth = self.nominal + self.liberal * DELTA
        return search(self.nominal, truth, level)

    def chance(self, trials: int, level: float) -> float:
        """Exact chance of seeing the rate `DELTA` off on its unsafe side.

        Args:
            trials: Trials run.
            level: Two-sided error rate of the bound.

        Returns:
            The probability (`chance`).
        """
        truth = self.nominal + self.liberal * DELTA
        return chance(self.nominal, truth, trials, level)


@dataclasses.dataclass(frozen=True)
class Uniform(Check):
    """A rank that is uniform on [0, 1) when the procedure is calibrated.

    Attributes:
        field: The rank judged, in [0, 1).
        location: The product code that produced the record, when narrower
            than the scenario's.
        bins: Equal-probability bins.
    """

    field: str
    location: str = ""
    bins: int = BINS

    @property
    def kind(self) -> str:
        """`rank`."""
        return "rank"

    def judge(
        self, values: Sequence[float | None], level: float
    ) -> dict[str, Any]:
        """Fills the bins and bounds each bin's share exactly.

        Args:
            values: One rank per trial; None where the trial errored.
            level: Error rate shared by the bins (Bonferroni).

        Returns:
            The kind, bin counts, trials, the share and bounds of the bin
            furthest from `1 / bins`, errors and state.
        """
        ranks = [v for v in values if v is not None]
        counts = [0] * self.bins
        for rank in ranks:
            place = math.floor(rank * self.bins + EDGE)
            counts[min(place, self.bins - 1)] += 1
        share = 1 / self.bins
        worst, state = 0, "nominal"
        bounds = [(0.0, 1.0)] * self.bins
        if ranks:
            bounds = [
                scmle.evaluation.coverage.clopper_pearson(
                    c, len(ranks), 1 - level / self.bins
                )
                for c in counts
            ]
            worst = max(
                range(self.bins),
                key=lambda b: abs(counts[b] / len(ranks) - share),
            )
            if any(high < share or low > share for low, high in bounds):
                state = "nonuniform"
        errored = len(values) - len(ranks)
        return {
            "field": self.field,
            "kind": self.kind,
            "nominal": share,
            "bins": counts,
            "trials": len(values),
            "rate": counts[worst] / len(ranks) if ranks else None,
            "low": bounds[worst][0],
            "high": bounds[worst][1],
            "errors": errored,
            "state": "errored" if errored else state,
        }

    def plan(self, level: float) -> int:
        """Trials that see one bin `DELTA` heavier than its share.

        Args:
            level: Error rate shared by the bins.

        Returns:
            The planned count (`search`).
        """
        share = 1 / self.bins
        return search(share, share + DELTA, level / self.bins)

    def chance(self, trials: int, level: float) -> float:
        """Exact chance of seeing one bin `DELTA` heavier than its share.

        Args:
            trials: Trials run.
            level: Error rate shared by the bins.

        Returns:
            The probability (`chance`).
        """
        share = 1 / self.bins
        return chance(share, share + DELTA, trials, level / self.bins)


@dataclasses.dataclass(frozen=True)
class Scenario:
    """A simulated world and the product call it calibrates.

    Attributes:
        name: Unique name; it also seeds the trials.
        location: The product function called, `path:qualified name`.
        draw: One trial: simulates, calls the product, returns the records.
        checks: What is judged of the records.
    """

    name: str
    location: str
    draw: Callable[[numpy.random.Generator], dict[str, float]]
    checks: tuple[Check, ...]


def chance(null: float, truth: float, trials: int, level: float) -> float:
    """Exact chance that the two-sided bound excludes `null`.

    Args:
        null: The nominal rate.
        truth: The true rate.
        trials: Trials run.
        level: Two-sided error rate of the Clopper-Pearson interval.

    Returns:
        The probability, under `truth`, that the interval misses `null`.
    """
    k = numpy.arange(trials + 1)
    low = numpy.where(
        k > 0, stats.beta.ppf(level / 2, numpy.maximum(k, 1), trials - k + 1), 0
    )
    high = numpy.where(
        k < trials,
        stats.beta.ppf(1 - level / 2, k + 1, numpy.maximum(trials - k, 1)),
        1,
    )
    excluded = (high < null) | (low > null)
    return float(stats.binom.pmf(k, trials, truth)[excluded].sum())


@functools.cache
def search(null: float, truth: float, level: float) -> int:
    """Trials whose exact chance of seeing `truth` reaches `power`.

    The chance is a saw-tooth in the count, so the bisection returns a count
    that reaches `power`, not always the smallest that does.

    Args:
        null: The nominal rate.
        truth: The miscalibrated rate to detect.
        level: Two-sided error rate of the bound.

    Returns:
        The count.

    Raises:
        InputError: If no count up to `acceptance.LIMIT` reaches `power`.
    """
    power = acceptance.load()["power"].value
    low, high = 0, 1
    while chance(null, truth, high, level) < power:
        low, high = high, high * 2
        if high > acceptance.LIMIT:
            raise errors.InputError(
                f"no count up to {acceptance.LIMIT} detects {truth}"
            )
    while high - low > 1:
        middle = (low + high) // 2
        if chance(null, truth, middle, level) >= power:
            high = middle
        else:
            low = middle
    return high


def attained(rows: int, coverage: float) -> float:
    """The exact coverage of a split-conformal margin from `rows` scores.

    Args:
        rows: Calibration scores (continuous, so without ties).
        coverage: The coverage asked for.

    Returns:
        `ceil((n + 1) c) / (n + 1)`, or 1 when that rank exceeds `n` (the
        margin is then infinite); the product's own rank expression.
    """
    rank = math.ceil((rows + 1) * coverage)
    return 1.0 if rank > rows else rank / (rows + 1)


def mondrian() -> float:
    """Exact coverage of class-conditional sets in the `classify` world.

    Returns:
        The mixture over the test row's class `k` (chance `SHARES[k]`) and
        its calibration count `m ~ Binomial(CALIBRATION, SHARES[k])` of
        `attained(m, conformal.DEFAULT_COVERAGE)`.
    """
    rows = numpy.arange(CALIBRATION + 1)
    exact = numpy.array(
        [attained(int(m), conformal.DEFAULT_COVERAGE) for m in rows]
    )
    return float(
        sum(
            share * (stats.binom.pmf(rows, CALIBRATION, share) * exact).sum()
            for share in SHARES
        )
    )


def holm(rows: int, null: float, tests: int) -> float:
    """Exact familywise rate of the slice flags when every slice is at null.

    Holm rejects any slice exactly when the smallest one-sided binomial
    p-value is at most `familywise / tests`; the slices are independent.

    Args:
        rows: Rows per slice.
        null: Each row's chance of being covered.
        tests: Slices.

    Returns:
        `1 - (1 - r)^tests`, `r` the chance one slice's p-value reaches the
        cut.
    """
    cut = scmle.evaluation.coverage.DEFAULT_FAMILYWISE / tests
    tail = stats.binom.cdf(numpy.arange(rows + 1), rows, null)
    reach = tail[tail <= cut]
    one = float(reach.max()) if reach.size else 0.0
    return 1 - (1 - one) ** tests


def risk() -> float:
    """Expected squared error of least squares on a validation fold's rows.

    Returns:
        `1 + p / (n - p - 1)` for `p = FEATURES` Gaussian features, unit
        noise and `n` training rows of one fold, the Nadeau-Bengio target.
    """
    rows = VALIDATION_ROWS * (FOLDS - 1) // FOLDS
    return 1 + FEATURES / (rows - FEATURES - 1)


def rank(p: float) -> float:
    """The complement of a p-value, uniform on [0, 1) under the null."""
    return min(1.0 - p, math.nextafter(1.0, 0.0))


def regress(
    rng: numpy.random.Generator, rows: int
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Rows of a linear world with heteroskedastic heavy-tailed noise."""
    x = rng.standard_normal((rows, FEATURES))
    noise = (0.5 + numpy.abs(x[:, 0])) * rng.standard_t(5, rows)
    return x, x @ WEIGHTS + noise


def calibrate(
    rng: numpy.random.Generator, tests: int
) -> tuple[conformal.ConformalCalibrator, numpy.ndarray, numpy.ndarray]:
    """Fits least squares, calibrates on fresh rows, predicts test rows."""
    x, y = regress(rng, TRAIN + CALIBRATION + tests)
    coef = numpy.linalg.lstsq(x[:TRAIN], y[:TRAIN], rcond=None)[0]
    pred = x @ coef
    split = TRAIN + CALIBRATION
    calibrator = conformal.ConformalCalibrator.for_regression(
        y[TRAIN:split].tolist(), pred[TRAIN:split].tolist()
    )
    return calibrator, y[split:], pred[split:]


def conform(rng: numpy.random.Generator) -> dict[str, float]:
    """Split-conformal regression interval and the new score's rank."""
    calibrator, y, pred = calibrate(rng, 1)
    low, high = calibrator.interval(pred, conformal.DEFAULT_COVERAGE)
    score = abs(y[0] - pred[0])
    below = int((numpy.asarray(calibrator.scores) < score).sum())
    return {
        "covered": float(low[0] <= y[0] <= high[0]),
        "rank": below / (CALIBRATION + 1),
    }


def classify(rng: numpy.random.Generator) -> dict[str, float]:
    """Pooled and class-conditional sets from true class probabilities."""
    labels = rng.choice(len(SHARES), CALIBRATION + 1, p=SHARES)
    x = CENTRES[labels] + rng.standard_normal((CALIBRATION + 1, 2))
    gaps = ((x[:, None, :] - CENTRES[None]) ** 2).sum(axis=2)
    density = numpy.exp(-gaps / 2) * numpy.asarray(SHARES)
    chances = density / density.sum(axis=1, keepdims=True)
    classes = tuple(range(len(SHARES)))
    calibrator = conformal.ConformalCalibrator.classify(
        labels[:-1].tolist(), chances[:-1], classes
    )
    truth = int(labels[-1])
    pooled = calibrator.sets(chances[-1:], conformal.DEFAULT_COVERAGE)[0]
    within = calibrator.sets(chances[-1:], conformal.DEFAULT_COVERAGE, "class")[
        0
    ]
    score = 1 - chances[-1, truth]
    below = int((numpy.asarray(calibrator.scores) < score).sum())
    return {
        "covered": float(truth in pooled),
        "conditional": float(truth in within),
        "rank": below / (CALIBRATION + 1),
    }


def reweight(rng: numpy.random.Generator) -> dict[str, float]:
    """Weighted conformal under a covariate shift with the exact ratio."""
    x = rng.standard_normal(CALIBRATION)
    y = x + (0.5 + numpy.abs(x)) * rng.standard_normal(CALIBRATION)
    new = SHIFT + rng.standard_normal()
    outcome = new + (0.5 + abs(new)) * rng.standard_normal()
    calibrator = conformal.ConformalCalibrator.for_regression(
        y.tolist(), x.tolist()
    )
    low, high = calibrator.weighted_interval(
        [new],
        numpy.exp(SHIFT * x - SHIFT**2 / 2),
        [math.exp(SHIFT * new - SHIFT**2 / 2)],
        conformal.DEFAULT_COVERAGE,
    )
    return {"covered": float(low[0] <= outcome <= high[0])}


def folds(rng: numpy.random.Generator, rows: int) -> numpy.ndarray:
    """A shuffled fold index per row, folds as equal as they come."""
    found = numpy.arange(rows) % FOLDS
    rng.shuffle(found)
    return found


def crossfit(rng: numpy.random.Generator) -> dict[str, float]:
    """CV+ interval from stored fold models at one new row."""
    x, y = regress(rng, TRAIN + 1)
    frame = pandas.DataFrame(x, columns=[f"x{i}" for i in range(FEATURES)])
    fold = folds(rng, TRAIN)
    residuals = numpy.empty(TRAIN)
    models = []
    for k in range(FOLDS):
        held = numpy.flatnonzero(fold == k)
        kept = numpy.flatnonzero(fold != k)
        model = linear_model.LinearRegression()
        model.fit(frame.iloc[kept], y[kept])
        residuals[held] = numpy.abs(y[held] - model.predict(frame.iloc[held]))
        models.append(model)
    ensemble = cv_plus.CvPlusEnsemble(tuple(models), fold, residuals)
    low, high = ensemble.interval(
        frame.iloc[[TRAIN]], conformal.DEFAULT_COVERAGE
    )
    return {"covered": float(low[0] <= y[TRAIN] <= high[0])}


def validate(rng: numpy.random.Generator) -> dict[str, float]:
    """The Nadeau-Bengio interval of a cross-validated squared error."""
    x = rng.standard_normal((VALIDATION_ROWS, FEATURES))
    y = x @ WEIGHTS + rng.standard_normal(VALIDATION_ROWS)
    fold = folds(rng, VALIDATION_ROWS)
    scores = []
    for k in range(FOLDS):
        held, kept = fold == k, fold != k
        coef = numpy.linalg.lstsq(x[kept], y[kept], rcond=None)[0]
        scores.append(float(((y[held] - x[held] @ coef) ** 2).mean()))
    share = VALIDATION_ROWS / FOLDS
    found = confidence.cv_interval(
        scores,
        VALIDATION_ROWS - share,
        share,
        "random",
        candidate_scores=None,
        greater_is_better=False,
    )
    low, high = found["interval"]
    return {"covered": float(low <= risk() <= high)}


def jump(rng: numpy.random.Generator) -> dict[str, float]:
    """Sharp RD with a smooth covariate, placebo cutoffs and a flat density."""
    x = rng.uniform(-1, 1, RD_ROWS)
    mean = 0.5 * x - 0.3 * x**2 + JUMP * (x >= 0)
    frame = pandas.DataFrame(
        {
            "x": x,
            "y": mean + 0.5 * rng.standard_normal(RD_ROWS),
            "z": 0.5 * x + rng.standard_normal(RD_ROWS),
        }
    )
    found = rd.estimate(frame, "x", "y", 0.0, LEVEL, covariates=("z",))
    records = {"covered": float(found.low <= JUMP <= found.high)}
    tests = {
        "balance": found.balance.get("z", {}).get("p"),
        "left": found.placebos.get("left_median", {}).get("p"),
        "right": found.placebos.get("right_median", {}).get("p"),
        "density": found.density_p,
    }
    for name, p in tests.items():
        if p is not None:
            records[f"{name}.rejected"] = float(p <= 1 - LEVEL)
            records[f"{name}.rank"] = rank(p)
    return records


def adopt(rng: numpy.random.Generator) -> dict[str, float]:
    """Staggered adoption under parallel trends with a constant effect."""
    starts = [3.0] * COHORT + [4.0] * COHORT + [math.nan] * COHORT
    level = rng.standard_normal(len(starts))
    rows = [
        {
            "unit": unit,
            "time": period,
            "adoption": start,
            "y": level[unit]
            + 0.3 * period
            + EFFECT * (period >= start)
            + rng.standard_normal(),
        }
        for unit, start in enumerate(starts)
        for period in range(PERIODS)
    ]
    found = did.estimate(
        pandas.DataFrame(rows), "unit", "time", "adoption", "y", LEVEL
    )
    records = {"covered": float(found.low <= EFFECT <= found.high)}
    if found.pretrend_p is not None:
        records["pretrend.rejected"] = float(found.pretrend_p <= 1 - LEVEL)
        records["pretrend.rank"] = rank(found.pretrend_p)
    return records


def question(**changes: Any) -> causal.Question:
    """A randomized charter question on columns `t` and `y`."""
    return causal.Question(
        treatment="t",
        outcome="y",
        estimand="ATE",
        assumptions=("randomized_assignment", "no_interference"),
        strategies=("randomized_diff",),
        **changes,
    )


def randomize(rng: numpy.random.Generator) -> dict[str, float]:
    """Difference in means of a skewed outcome under Bernoulli assignment."""
    t = rng.integers(0, 2, RANDOM_ROWS)
    frame = pandas.DataFrame(
        {"t": t, "y": rng.exponential(1.0, RANDOM_ROWS) + EFFECT * t}
    )
    adapter = adjustment.RandomizedEstimator(
        question(), frame, 0, "linear", 1 - LEVEL
    )
    found = adapter.estimate(frame, LEVEL)
    return {"covered": float(found.low <= EFFECT <= found.high)}


def divide(rng: numpy.random.Generator) -> dict[str, float]:
    """Declared subgroups with no effect in any, adjusted by Holm."""
    rows = SUBGROUP * len(GROUPS)
    # Subgroups differ in their outcome level, never in the effect (zero).
    shift = numpy.repeat([0.0, 1.0, -1.0], SUBGROUP)
    frame = pandas.DataFrame(
        {
            "g": numpy.repeat(GROUPS, SUBGROUP),
            "t": rng.integers(0, 2, rows),
            "y": shift + rng.standard_normal(rows),
        }
    )
    spec = question(subgroups=(causal.Subgroup("g", GROUPS),))
    adapter = adjustment.RandomizedEstimator(
        spec, frame, 0, "linear", 1 - LEVEL
    )
    found = strata.stratify(adapter, frame, LEVEL, spec.subgroups)
    first = found["hypotheses"][0]
    records = {"rejected": float(bool(found["rejected"]))}
    if first["skipped"] is None:
        records["covered"] = float(first["low"] <= 0 <= first["high"])
        records["rank"] = rank(first["p_value"])
    return records


def weigh(
    rng: numpy.random.Generator,
    estimator: parameter_estimation.EstimatorKind,
    effect: float,
) -> dict[str, float]:
    """A propensity-weighted ATE under measured confounding."""
    x = rng.standard_normal(CAUSAL_ROWS)
    t = (rng.uniform(size=CAUSAL_ROWS) < 1 / (1 + numpy.exp(-0.5 * x))) * 1.0
    y = x + effect * t + rng.standard_normal(CAUSAL_ROWS)
    estimand = parameter_estimation.Estimand(
        name="ate",
        outcome="y",
        population="simulated",
        likelihood="gaussian",
        feature_names=("x", "t"),
        assumptions=("iid",),
    )
    policy = parameter_estimation.Config(
        estimator=estimator,
        covariance="iid",
        treatment_position=1,
        causal_assumptions=(
            "consistency",
            "conditional_exchangeability",
            "positivity",
        ),
    )
    found = parameter_estimation.estimate_parameters(
        numpy.column_stack([x, t]), y, estimand, policy
    ).parameters[0]
    return {
        "covered": float(found.interval[0] <= effect <= found.interval[1]),
        "rejected": float(found.p_value <= 1 - LEVEL),
        "rank": rank(found.p_value),
    }


def fit(
    rng: numpy.random.Generator, estimator: parameter_estimation.EstimatorKind
) -> dict[str, float]:
    """A regression with one real slope `a` and one null slope `b`."""
    x = rng.standard_normal((MODEL_ROWS, 2))
    intercept = -0.5 if estimator == "logistic" else 1.0
    signal = intercept + x[:, 0]
    if estimator == "logistic":
        y = (rng.uniform(size=MODEL_ROWS) < 1 / (1 + numpy.exp(-signal))) * 1.0
        likelihood = "bernoulli"
    else:
        noise = (0.5 + numpy.abs(x[:, 0])) * rng.standard_normal(MODEL_ROWS)
        y = signal + noise
        likelihood = "gaussian"
    estimand = parameter_estimation.Estimand(
        name="slopes",
        outcome="y",
        population="simulated",
        likelihood=likelihood,
        feature_names=("a", "b"),
        assumptions=("iid",),
    )
    policy = parameter_estimation.Config(estimator=estimator)
    found = {
        p.name: p
        for p in parameter_estimation.estimate_parameters(
            x, y, estimand, policy
        ).parameters
    }
    low, high = found["a"].interval
    return {
        "covered": float(low <= 1.0 <= high),
        "rejected": float(found["b"].p_value <= 1 - LEVEL),
        "rank": rank(found["b"].p_value),
    }


def sample(rng: numpy.random.Generator) -> dict[str, float]:
    """Coefficients drawn from the conjugate prior; the credible interval."""
    prior = parameter_estimation.BayesianPrior(
        mean=(0.0, 0.0, 0.0),
        precision=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        shape=3.0,
        scale=2.0,
    )
    variance = prior.scale / rng.gamma(prior.shape)
    beta = math.sqrt(variance) * rng.standard_normal(3)
    x = rng.standard_normal((BAYES_ROWS, 2))
    noise = math.sqrt(variance) * rng.standard_normal(BAYES_ROWS)
    y = beta[0] + x @ beta[1:] + noise
    estimand = parameter_estimation.Estimand(
        name="slopes",
        outcome="y",
        population="simulated",
        likelihood="gaussian",
        feature_names=("a", "b"),
        assumptions=("iid",),
    )
    policy = parameter_estimation.Config(
        estimator="bayesian_linear", prior=prior, covariance="iid"
    )
    found = parameter_estimation.estimate_parameters(x, y, estimand, policy)
    low, high = found.parameters[1].interval
    return {"covered": float(low <= beta[1] <= high)}


def settings(rng: numpy.random.Generator) -> leakage.Config:
    """The certified leakage settings with a fresh run seed."""
    return dataclasses.replace(
        leakage.CERTIFIED, run_seed=int(rng.integers(2**31))
    )


def probe(rng: numpy.random.Generator, grouped: bool) -> dict[str, float]:
    """The single-feature skill permutation test on a null feature."""
    groups = None
    if grouped:
        count = PROBE_ROWS // CLUSTER
        units = numpy.repeat(numpy.arange(count), CLUSTER)
        feature = rng.standard_normal(size=count)[units]
        feature = feature + 0.3 * rng.standard_normal(PROBE_ROWS)
        logit = rng.standard_normal(size=count)[units]
        groups = pandas.Series(units)
    else:
        feature = rng.standard_normal(PROBE_ROWS)
        logit = numpy.zeros(PROBE_ROWS)
    target = (rng.uniform(size=PROBE_ROWS) < 1 / (1 + numpy.exp(-logit))) * 1
    cfg = settings(rng)
    found = leakage.probe(
        pandas.Series(feature, name="f"),
        pandas.Series(target),
        "classification",
        cfg,
        groups,
    )
    return {
        "rejected": float(found.p_value <= cfg.alpha),
        "rank": rank(found.p_value),
    }


def blank(rng: numpy.random.Generator, rows: int) -> numpy.ndarray:
    """A null feature with values missing completely at random."""
    values = rng.standard_normal(rows)
    values[rng.uniform(size=rows) < GAPS] = math.nan
    return values


def gap(rng: numpy.random.Generator) -> dict[str, float]:
    """The missingness test on gaps independent of the outcome."""
    target = (rng.uniform(size=SCREEN_ROWS) < PREVALENCE) * 1
    found = leakage.missingness_test(
        pandas.Series(blank(rng, SCREEN_ROWS), name="f"),
        pandas.Series(target),
        "classification",
        leakage.CERTIFIED,
    )
    if found is None:
        return {}
    return {
        "rejected": float(found.p_value <= leakage.CERTIFIED.alpha),
        "rank": rank(found.p_value),
    }


def scan(rng: numpy.random.Generator) -> dict[str, float]:
    """The leakage screen on a table where no column carries the outcome."""
    frame = pandas.DataFrame(
        {f"c{i}": rng.standard_normal(SCREEN_ROWS) for i in range(4)}
    )
    frame["gaps"] = blank(rng, SCREEN_ROWS)
    frame["y"] = (rng.uniform(size=SCREEN_ROWS) < PREVALENCE) * 1
    found = screen.screen(frame, "y", "classification", settings(rng))
    return {"rejected": float(any(f.significant for f in found.findings))}


def partition(rng: numpy.random.Generator, conformed: bool) -> dict[str, float]:
    """Slice flags on exchangeable slices at their null coverage."""
    labels = numpy.repeat(numpy.arange(SLICES), SLICE_ROWS)
    floor = conformal.DEFAULT_COVERAGE - scmle.evaluation.coverage.SLICE_MARGIN
    if conformed:
        calibrator, y, pred = calibrate(rng, SLICES * SLICE_ROWS)
        low, high = calibrator.interval(pred, conformal.DEFAULT_COVERAGE)
        first = (low <= y) & (y <= high)
        low, high = calibrator.interval(pred, floor)
        second = (low <= y) & (y <= high)
    else:
        first = rng.uniform(size=labels.size) < conformal.DEFAULT_COVERAGE
        second = rng.uniform(size=labels.size) < floor
    names = labels.tolist()
    nominal = scmle.evaluation.coverage.slice_coverage(
        first, names, conformal.DEFAULT_COVERAGE
    )
    margin = scmle.evaluation.coverage.undercovered(
        second, names, conformal.DEFAULT_COVERAGE
    )
    return {
        "coverage.flagged": float(bool(nominal["flagged"])),
        "floor.flagged": float(bool(margin["flagged"])),
    }


def resample(rng: numpy.random.Generator, family: str) -> dict[str, float]:
    """One trial of the bootstrap metric interval (`coverage.run`).

    It runs under `coverage.prepare`'s scikit-learn settings, as that
    module's own trials do, inside a context that restores the caller's
    settings, so no other scenario runs under them.
    """
    with sklearn.config_context():
        coverage.prepare()
        found = coverage.run(family, int(rng.integers(OFFSET, 2**31)))
    return {"covered": float(found["hit"])}


def confound(rng: numpy.random.Generator, rows: int) -> pandas.DataFrame:
    """Rows with a measured confounder `x`, treatment `t` and outcome `y`."""
    x = rng.standard_normal(rows)
    t = (rng.uniform(size=rows) < 1 / (1 + numpy.exp(-0.5 * x))) * 1.0
    y = x + EFFECT * t + rng.standard_normal(rows)
    return pandas.DataFrame({"x": x, "t": t, "y": y})


def augment(rng: numpy.random.Generator) -> dict[str, float]:
    """AIPW with nuisances frozen on exploration rows, scored on fresh ones.

    The effect is constant, so it is both the ATE and the ATT.
    """
    nuisances = aipw.prefit(confound(rng, AIPW_ROWS), ("x",), "t", "y")
    frame = confound(rng, AIPW_ROWS)
    records = {}
    for name, estimand in (("covered", "ATE"), ("treated.covered", "ATT")):
        found = aipw.estimate(nuisances, frame, "t", "y", estimand, LEVEL)
        records[name] = float(found.low <= EFFECT <= found.high)
    return records


def instrument(
    rng: numpy.random.Generator, strength: float, count: int
) -> pandas.DataFrame:
    """Valid instruments, a control `w` and an endogenous regressor `d`.

    The structural error is heteroskedastic in the control and correlated
    `ENDOGENEITY` with the first-stage error; it is independent of every
    instrument, which is what validity means.
    """
    z = rng.standard_normal((IV_ROWS, count))
    w = rng.standard_normal(IV_ROWS)
    v = rng.standard_normal(IV_ROWS)
    own = rng.standard_normal(IV_ROWS)
    u = ENDOGENEITY * v + math.sqrt(1 - ENDOGENEITY**2) * own
    d = z.sum(axis=1) * strength + 0.3 * w + v
    frame = pandas.DataFrame({f"z{k}": z[:, k] for k in range(count)})
    frame["w"] = w
    frame["d"] = d
    frame["y"] = EFFECT * d + 0.5 * w + (0.5 + 0.5 * numpy.abs(w)) * u
    return frame


def identify(rng: numpy.random.Generator, strength: float) -> dict[str, float]:
    """The just-identified IV fit: headline, 2SLS and Anderson-Rubin sets."""
    frame = instrument(rng, strength, 1)
    found = iv.estimate(frame, "d", "y", "z0", ("w",), LEVEL)
    anderson = dataclasses.replace(found, headline="anderson_rubin")
    # The weak scenario judges no 2SLS record: the product does not offer
    # that interval as the headline below its F threshold.
    return {
        "covered": float(not found.excludes(EFFECT)),
        "anderson.covered": float(not anderson.excludes(EFFECT)),
        "wald.covered": float(found.wald_low <= EFFECT <= found.wald_high),
    }


def overidentify(rng: numpy.random.Generator) -> dict[str, float]:
    """Sargan and Hansen J with every instrument valid.

    The structural error is homoskedastic here: Sargan is valid only then.
    """
    z = rng.standard_normal((IV_ROWS, INSTRUMENTS))
    w = rng.standard_normal(IV_ROWS)
    v = rng.standard_normal(IV_ROWS)
    own = rng.standard_normal(IV_ROWS)
    u = ENDOGENEITY * v + math.sqrt(1 - ENDOGENEITY**2) * own
    d = z.sum(axis=1) * STRONG + 0.3 * w + v
    y = EFFECT * d + 0.5 * w + u
    records = {}
    for name, test in (("sargan", overid.sargan), ("hansen", overid.hansen_j)):
        p = test(y, d, z, w)["p_value"]
        if p is not None:
            records[f"{name}.rejected"] = float(p <= 1 - LEVEL)
            records[f"{name}.rank"] = rank(p)
    return records


def donate(rng: numpy.random.Generator) -> dict[str, float]:
    """Synthetic control on exchangeable units with no effect.

    Every unit follows its own level and loading on one shared random-walk
    factor; unit 0 is called treated. Its placebo p-value is then uniform
    on the grid `k / (DONORS + 1)`.
    """
    units = DONORS + 1
    periods = PRE + POST
    factor = numpy.cumsum(rng.standard_normal(periods))
    level = rng.standard_normal(units)
    loading = 1 + 0.3 * rng.standard_normal(units)
    y = (
        level[None, :]
        + factor[:, None] * loading[None, :]
        + rng.standard_normal((periods, units))
    )
    frame = pandas.DataFrame(
        {
            "unit": numpy.tile(numpy.arange(units), periods),
            "time": numpy.repeat(numpy.arange(periods), units),
            "y": y.ravel(),
        }
    )
    found = synth.estimate(frame, "unit", "time", "y", 0, PRE, 1 - LEVEL)
    return {
        "rejected": float(found.p_value <= 1 - LEVEL),
        "rank": rank(found.p_value),
    }


EVALUATION = "scmle/evaluation/"
CONFORMAL = EVALUATION + "conformal.py:ConformalCalibrator."
ESTIMATION = EVALUATION + "parameter_estimation/estimate.py:estimate_parameters"
RD = "scmle/causal/identification/rd.py:estimate"
SLICED = EVALUATION + "coverage.py:"
IDENTIFICATION = "scmle/causal/identification/"
OVERID = IDENTIFICATION + "overid.py:"
LEAKAGE = "scmle/assurance/admissibility/leakage/"
FLOOR = conformal.DEFAULT_COVERAGE - scmle.evaluation.coverage.SLICE_MARGIN
HOLM = 1 - (1 - (1 - LEVEL) / len(GROUPS)) ** len(GROUPS)
PROPENSITY: tuple[parameter_estimation.EstimatorKind, ...] = (
    "ipw_ate",
    "aipw_ate",
)
REGRESSION: tuple[parameter_estimation.EstimatorKind, ...] = (
    "linear",
    "logistic",
)


def judged(name: str, nominal: float, location: str = "") -> tuple[Check, ...]:
    """The size and rank checks of one reported p-value."""
    return (
        Rate(f"{name}.rejected", nominal, 1, location),
        Uniform(f"{name}.rank", location),
    )


def null(nominal: float) -> tuple[Check, ...]:
    """The size and rank checks of a scenario's single test."""
    return (Rate("rejected", nominal, 1), Uniform("rank"))


SCENARIOS: tuple[Scenario, ...] = (
    *(
        Scenario(
            f"bootstrap.{family}",
            EVALUATION + "bootstrap.py:Uncertainty.compute",
            functools.partial(resample, family=family),
            (Rate("covered", LEVEL, -1),),
        )
        for family in coverage.FAMILIES
    ),
    Scenario(
        "conformal.regression",
        CONFORMAL + "interval",
        conform,
        (
            Rate(
                "covered", attained(CALIBRATION, conformal.DEFAULT_COVERAGE), -1
            ),
            Uniform("rank"),
        ),
    ),
    Scenario(
        "conformal.sets",
        CONFORMAL + "sets",
        classify,
        (
            Rate(
                "covered", attained(CALIBRATION, conformal.DEFAULT_COVERAGE), -1
            ),
            Rate("conditional", mondrian(), -1, CONFORMAL + "class_margins"),
            Uniform("rank"),
        ),
    ),
    Scenario(
        "conformal.weighted",
        CONFORMAL + "weighted_interval",
        reweight,
        (Rate("covered", conformal.DEFAULT_COVERAGE, -1),),
    ),
    Scenario(
        "conformal.cvplus",
        EVALUATION + "cv_plus.py:CvPlusEnsemble.interval",
        crossfit,
        (Rate("covered", conformal.DEFAULT_COVERAGE, -1),),
    ),
    Scenario(
        "validation.interval",
        EVALUATION + "confidence.py:cv_interval",
        validate,
        (Rate("covered", confidence.LEVEL, -1),),
    ),
    Scenario(
        "rd.effect",
        RD,
        jump,
        (
            Rate("covered", LEVEL, -1),
            *judged("balance", 1 - LEVEL, RD + " (balance, discontinuity)"),
            *judged("left", 1 - LEVEL, RD + " (placebos, falsify)"),
            *judged("right", 1 - LEVEL, RD + " (placebos, falsify)"),
            *judged("density", 1 - LEVEL, RD + " (density_p, binomtest)"),
        ),
    ),
    Scenario(
        "did.effect",
        "scmle/causal/identification/did.py:estimate",
        adopt,
        (
            Rate("covered", LEVEL, -1),
            *judged(
                "pretrend",
                1 - LEVEL,
                "scmle/causal/identification/did.py:pretest",
            ),
        ),
    ),
    Scenario(
        "aipw.frozen",
        IDENTIFICATION + "aipw.py:estimate",
        augment,
        (Rate("covered", LEVEL, -1), Rate("treated.covered", LEVEL, -1)),
    ),
    Scenario(
        "iv.strong",
        IDENTIFICATION + "iv.py:estimate",
        functools.partial(identify, strength=STRONG),
        (
            Rate("covered", LEVEL, -1),
            Rate("wald.covered", LEVEL, -1),
            Rate(
                "anderson.covered",
                LEVEL,
                -1,
                IDENTIFICATION + "iv.py:anderson_rubin",
            ),
        ),
    ),
    Scenario(
        "iv.weak",
        IDENTIFICATION + "iv.py:estimate",
        functools.partial(identify, strength=WEAK),
        (
            Rate("covered", LEVEL, -1),
            Rate(
                "anderson.covered",
                LEVEL,
                -1,
                IDENTIFICATION + "iv.py:anderson_rubin",
            ),
        ),
    ),
    Scenario(
        "overid.valid",
        OVERID + "sargan",
        overidentify,
        (
            *judged("sargan", 1 - LEVEL, OVERID + "sargan"),
            *judged("hansen", 1 - LEVEL, OVERID + "hansen_j"),
        ),
    ),
    Scenario(
        "synth.placebo",
        IDENTIFICATION + "synth.py:estimate",
        donate,
        null(1 - LEVEL),
    ),
    Scenario(
        "randomized.effect",
        "scmle/causal/identification/run/adjustment.py:"
        "RandomizedEstimator.estimate",
        randomize,
        (Rate("covered", LEVEL, -1),),
    ),
    Scenario(
        "subgroups.null",
        "scmle/causal/identification/run/strata.py:stratify",
        divide,
        (
            Rate("rejected", HOLM, 1),
            Rate("covered", LEVEL, -1),
            Uniform("rank"),
        ),
    ),
    *(
        Scenario(
            f"{kind.removesuffix('_ate')}.effect",
            f"{ESTIMATION} ({kind})",
            functools.partial(weigh, estimator=kind, effect=EFFECT),
            (Rate("covered", LEVEL, -1),),
        )
        for kind in PROPENSITY
    ),
    *(
        Scenario(
            f"{kind.removesuffix('_ate')}.null",
            f"{ESTIMATION} ({kind})",
            functools.partial(weigh, estimator=kind, effect=0.0),
            null(1 - LEVEL),
        )
        for kind in PROPENSITY
    ),
    *(
        Scenario(
            f"{kind}.slopes",
            f"{ESTIMATION} ({kind})",
            functools.partial(fit, estimator=kind),
            (Rate("covered", LEVEL, -1), *null(1 - LEVEL)),
        )
        for kind in REGRESSION
    ),
    Scenario(
        "bayesian.prior",
        f"{ESTIMATION} (bayesian_linear)",
        sample,
        (Rate("covered", LEVEL, -1),),
    ),
    Scenario(
        "leakage.skill",
        LEAKAGE + "__init__.py:probe",
        functools.partial(probe, grouped=False),
        null(leakage.CERTIFIED.alpha),
    ),
    Scenario(
        "leakage.grouped",
        LEAKAGE + "__init__.py:probe",
        functools.partial(probe, grouped=True),
        null(leakage.CERTIFIED.alpha),
    ),
    Scenario(
        "leakage.missingness",
        LEAKAGE + "missingness.py:missingness_test",
        gap,
        null(leakage.CERTIFIED.alpha),
    ),
    Scenario(
        "screen.null",
        "scmle/assurance/admissibility/screen/__init__.py:screen",
        scan,
        (Rate("rejected", leakage.CERTIFIED.alpha, 1),),
    ),
    *(
        Scenario(
            f"slices.{name}",
            SLICED + "slice_coverage",
            functools.partial(partition, conformed=name == "conformal"),
            (
                Rate(
                    "coverage.flagged",
                    holm(SLICE_ROWS, conformal.DEFAULT_COVERAGE, SLICES),
                    1,
                ),
                Rate(
                    "floor.flagged",
                    holm(SLICE_ROWS, FLOOR, SLICES),
                    1,
                    SLICED + "undercovered",
                ),
            ),
        )
        for name in ("boundary", "conformal")
    ),
)


def lookup(name: str) -> Scenario:
    """The scenario of a name.

    Args:
        name: A `Scenario.name`.

    Returns:
        The scenario.

    Raises:
        InputError: For an unknown name.
    """
    for scenario in SCENARIOS:
        if scenario.name == name:
            return scenario
    raise errors.InputError(f"unknown scenario {name!r}")


def family() -> int:
    """Checks over every scenario, the Bonferroni family."""
    return sum(len(s.checks) for s in SCENARIOS)


def level() -> float:
    """Error rate of each check's exact bound: `alpha / family`."""
    return acceptance.load()["alpha"].value / family()


def planned(scenario: Scenario) -> int:
    """Trials of a scenario at which every check reaches `power`.

    It starts at the most any check needs alone; the chance is a saw-tooth
    in the count, so a check planned lower can dip below `power` there, and
    the count then grows until none does.
    """
    chosen = level()
    power = acceptance.load()["power"].value
    count = max(check.plan(chosen) for check in scenario.checks)
    while min(check.chance(count, chosen) for check in scenario.checks) < (
        power
    ):
        count += 1
    return count


def run(name: str, trial: int) -> dict[str, Any]:
    """One trial of a scenario.

    Args:
        name: The scenario.
        trial: Trial number; seeds `default_rng([trial, crc32(name),
            STREAM])`.

    Returns:
        The records, or `error` with the type and message of a typed
        refusal by the product (ValueError, arithmetic and linear-algebra
        failures).
    """
    scenario = lookup(name)
    rng = numpy.random.default_rng([trial, zlib.crc32(name.encode()), STREAM])
    try:
        return scenario.draw(rng)
    except (ValueError, ArithmeticError, numpy.linalg.LinAlgError) as error:
        # handled: a refusal is recorded and counted against the product
        return {"error": f"{type(error).__name__}: {error}"}


def tally(scenario: Scenario, outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    """Judges every check of a scenario.

    Args:
        scenario: The scenario.
        outcomes: What `run` returned for each trial.

    Returns:
        The name, location, trials, planned trials, failed trials (refused,
        or missing a judged field) with up to five distinct refusals and
        one note per missing field, and
        one judgement per check, each with its `absent` count: trials that
        ran but did not report the field, counted against the product.
    """
    chosen = level()
    checks = []
    failed = {i for i, o in enumerate(outcomes) if "error" in o}
    notes = []
    for check in scenario.checks:
        absent = [
            i
            for i, o in enumerate(outcomes)
            if "error" not in o and check.field not in o
        ]
        if absent:
            failed.update(absent)
            notes.append(f"{check.field} not reported in {len(absent)} trials")
        values = [
            None if "error" in o else o.get(check.field) for o in outcomes
        ]
        found = check.judge(values, chosen)
        found["absent"] = len(absent)
        found["location"] = check.location or scenario.location
        checks.append(found)
    refusals = sorted({o["error"] for o in outcomes if "error" in o})
    return {
        "name": scenario.name,
        "location": scenario.location,
        "trials": len(outcomes),
        "planned": planned(scenario),
        "errors": len(failed),
        "messages": refusals[:5] + notes,
        "checks": checks,
    }


def digest() -> str:
    """Hash of this module, the bootstrap trial and the product packages."""
    own = hashing.code_hash(sys.modules[__name__], coverage.run)
    product = evidence.source(PRODUCT, PACKAGE)
    return hashlib.sha256(f"{own}{product}".encode()).hexdigest()


def measure(
    trials: int | None = None,
    workers: int | None = None,
    names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Runs the scenarios and builds the stored document.

    Args:
        trials: Trials per scenario; None runs each scenario's `planned`.
        workers: Worker processes; None uses every core, 1 runs in this
            process.
        names: Scenarios to run; None runs all.

    Returns:
        The JSON-ready document.
    """
    started = time.monotonic()
    chosen = [lookup(n) for n in names] if names else list(SCENARIOS)
    jobs = [
        (s.name, t)
        for s in chosen
        for t in range(trials if trials is not None else planned(s))
    ]
    if workers == 1:
        done = [run(n, t) for n, t in jobs]
    else:
        with futures.ProcessPoolExecutor(workers or os.cpu_count()) as pool:
            done = list(pool.map(run, *zip(*jobs, strict=True), chunksize=4))
    scenarios = [
        tally(
            s, [o for (n, _), o in zip(jobs, done, strict=True) if n == s.name]
        )
        for s in chosen
    ]
    return {
        "claim": "C3",
        "clause": "every interval and test nominal by simulation-based "
        "calibration",
        "alpha": acceptance.load()["alpha"].value,
        "power": acceptance.load()["power"].value,
        "family": family(),
        "level": level(),
        "delta": DELTA,
        "bins": BINS,
        "seeds": "default_rng([trial, crc32(name), 3])",
        "scenarios": scenarios,
        "miscalibrated": [
            {
                "scenario": s["name"],
                **{k: c[k] for k in ("field", "kind", "state", "location")},
                **{k: c[k] for k in ("nominal", "rate", "low", "high")},
            }
            for s in scenarios
            for c in s["checks"]
            if c["state"] != "nominal"
        ],
        "not_measured": [
            {"location": location, "what": what}
            for location, what in NOT_MEASURED
        ],
        "limits": list(LIMITS),
        "code": digest(),
        "seconds": round(time.monotonic() - started, 1),
    }


def read(document: dict[str, Any] | None) -> dict[str, Any] | None:
    """The given document, or the stored one (None when nothing is)."""
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
        Messages for a missing file, other code, another family, a missing
        scenario, other checks, or fewer trials than planned; empty when the
        document decides.
    """
    found = read(document)
    if found is None:
        return [f"{PATH.name} is missing"]
    issues = []
    if found.get("code") != digest():
        issues.append("calibration: measured with other code")
    if found.get("family") != family():
        issues.append(
            f"calibration: family of {found.get('family')} checks is not "
            f"{family()}"
        )
    stored = {s.get("name"): s for s in found.get("scenarios", [])}
    for scenario in SCENARIOS:
        record = stored.get(scenario.name)
        if record is None:
            issues.append(f"calibration: scenario {scenario.name} not measured")
            continue
        fields = [c.get("field") for c in record.get("checks", [])]
        if fields != [c.field for c in scenario.checks]:
            issues.append(f"calibration: {scenario.name} has other checks")
        count, least = record.get("trials", 0), planned(scenario)
        if count < least:
            issues.append(
                f"calibration: {scenario.name} has {count} trials, fewer "
                f"than the {least} planned"
            )
    return issues


def summary(document: dict[str, Any] | None = None) -> tuple[str, str]:
    """The scorecard's value and verdict for the clause.

    Args:
        document: The document; read from `PATH` when None.

    Returns:
        The count of nominal checks, the miscalibrated ones by name and
        state, and the count of product intervals and tests not measured;
        `fail` when any check is not nominal, `partial` when every check is
        but `NOT_MEASURED` names any product interval or test, `pass` only
        when neither; `not measured` when nothing is stored or `problems`
        finds any.
    """
    found = read(document)
    if found is None or not found.get("scenarios"):
        return "no measurement", "not measured"
    states = [
        (f"{s['name']}.{c['field']}", c["state"])
        for s in found["scenarios"]
        for c in s["checks"]
    ]
    bad = [f"{name} ({state})" for name, state in states if state != "nominal"]
    text = f"{len(states) - len(bad)}/{len(states)} checks nominal"
    if bad:
        text += "; miscalibrated: " + ", ".join(bad)
    if NOT_MEASURED:
        text += (
            f"; {len(NOT_MEASURED)} product intervals and tests not measured"
        )
    if problems(found):
        return text, "not measured"
    if bad:
        return text, "fail"
    return text, "partial" if NOT_MEASURED else "pass"


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
        document = measure(workers=options.workers)
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
