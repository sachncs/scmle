"""Measured gates for the causal identification methods.

Every floor below is fixed in this file before the first run and is not
retuned: a gate that misses is filed as measured and `passed` is false, so
that method is descriptive and cannot count toward a `confirmed` claim.
Certification seeds are `0 .. replicates - 1`; development looks use seeds
from `DEV_SEED_BASE`, which never overlap. Intervals are Wilson 95 percent.

All worlds are simulated (`f10_generators`), including their covariates;
real-covariate worlds, the NSW and STAR gold standards, the frozen
`rdrobust`/`sensemakr`/`EValue`/`did`/`HonestDiD` reference outputs and the
outside-party replication need data or tools this repository does not have
(`NOT_MEASURED`). Real data may violate every assumption made here.

Gates (`R` is the replicate count):

* selector: 200 charters with no eligible strategy are all refused and 200
  eligible charters are none refused; Wilson lower bound of each `>= 0.98`.
* coverage_aipw: 95 percent interval for ATE and ATT (linear learners, `R =
  2000`) and ATE with boosted learners on a nonlinear surface (`R = 500`);
  Wilson lower bound of coverage `>= 0.93` and mean (estimate - truth) / se
  within `0.1` of zero, in every world.
* coverage_randomized, coverage_did (one group and staggered), coverage_rd:
  Wilson lower bound `>= 0.93` (`R = 2000`, staggered DiD `R = 1000`); the
  mean standardised error within `0.1` for DiD and RD.
* coverage_iv: headline set coverage `>= 0.93` (Wilson lower) with a strong
  instrument (`R = 2000`); the standardised-error criterion does not apply
  to IV.
* iv_strength: in a weak-instrument world the headline set (Anderson-Rubin
  when `F < 10`) has Wilson lower coverage `>= 0.93`, the weak flag has
  recall (Wilson lower) `>= 0.8`, and in a strong world the flag's false
  rate is `<= 0.05`.
* size_synth: with no effect, placebo-in-space rejects at `p <= 0.05` no more
  often than `0.05 + 2 sqrt(0.05 * 0.95 / R)` (`R = 500`).
* overlap: the overlap gate (`aipw.overlap_report`) refuses valid overlap
  (minimum propensity 0.5 and 0.1) in `<= 0.05` of replicates and flags the
  two worst settings (0.02 and 0.005) with recall (Wilson lower) `>= 0.8`;
  coverage and effective sample size are reported as curves, not gated.
* battery_nco, battery_pretrend, density_rd, battery_rd: false-alarm rate
  with no violation `<= alpha_falsify + 2 sqrt(alpha_falsify (1 -
  alpha_falsify) / R)` with `alpha_falsify = 0.01`; and a smallest planted
  violation `delta_min` with power (Wilson lower) `>= 0.8` exists in the
  tested grid. The full power curve is stored.
* sensitivity_bound: a hidden confounder planted at half the strength of the
  strongest observed covariate (inside the benchmark) is covered by the
  benchmarked interval with Wilson lower `>= 0.93` (`R = 1000`); the same at
  the boundary and at twice the benchmark are reported, not gated.
* sensitivity_did: with a linear differential trend, the relative-magnitudes
  interval at `M = 1` covers the truth with Wilson lower `>= 0.93` (`R =
  500`); `M = 0` is reported.
* false_confirm: through `run.identify` with every gate marked certified and
  a declared DAG, the share of runs that reach `confirmed` when a hidden
  confounder is shared with the negative control (`R = 300`) is `<= 0.05` with
  Wilson upper `<= 0.08`. The unshared-confounder blind spot and the clean
  world are reported, not gated.

    python -m benchmarks.qualification.f10_certify --write
    python -m benchmarks.qualification.f10_certify --check
"""

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy
import pandas

from benchmarks.qualification import f10_generators
import scmle.data.admissibility.dag
from scmle.data.charter import causal
from scmle.data.detectors import certificates
from scmle.data.identification import aipw
from scmle.data.identification import battery
from scmle.data.identification import certified
from scmle.data.identification import did
from scmle.data.identification import iv
from scmle.data.identification import rd
from scmle.data.identification import run
from scmle.data.identification import select
from scmle.data.identification import sensitivity
from scmle.data.identification import synth
from scmle.data.ledger import budget

STORE_PATH = Path(__file__).with_name("f10_certificates.json")
DEV_SEED_BASE = 10_000
ALPHA_FALSIFY = 0.01
LEVEL = 0.95
COVERAGE_FLOOR = 0.93
# largest mean standardised error that still counts as centred
MEAN_Z_TOLERANCE = 0.1
# floor on the Wilson lower bound of the power at the smallest planted violation
POWER_FLOOR = 0.8
# floor on the Wilson lower bound of a flag recall
RECALL_FLOOR = 0.8
# floor on the Wilson lower bound of the strategy selection rates
SELECTION_FLOOR = 0.98
# largest false-alarm or false-confirmation rate allowed
FALSE_ALARM_CEILING = 0.05
# largest Wilson upper bound of the false-confirmation rate
FALSE_CONFIRM_UPPER_CEILING = 0.08
# level of the placebo-in-space test of the synthetic control
SYNTH_LEVEL = 0.05
# effect planted in the backdoor and randomized worlds
AIPW_TRUE_EFFECT = 2.0
# effect planted in the panel worlds
DID_TRUE_EFFECT = 1.5
GATES = (
    "selector",
    "coverage_aipw",
    "coverage_randomized",
    "coverage_did",
    "coverage_iv",
    "coverage_rd",
    "size_synth",
    "iv_strength",
    "overlap",
    "battery_nco",
    "battery_pretrend",
    "density_rd",
    "battery_rd",
    "sensitivity_bound",
    "sensitivity_did",
    "false_confirm",
)
NOT_MEASURED = (
    "real-covariate worlds (plan A3: planted effect on real X)",
    "NSW with CPS/PSID comparison groups and Tennessee STAR",
    "ACIC and IHDP generators",
    "frozen reference outputs: rdrobust, sensemakr, EValue, did, HonestDiD",
    "outside-party replication of the gold-standard table",
    "battery_iv: no falsification test for a just-identified instrument",
    "battery_synth: placebo-in-space is the inference itself",
    "bit-identical reports on Python 3.11 and 3.14",
)


def stream(base: int, index: int, which: int = 0) -> numpy.random.Generator:
    """Independent random stream for one replicate.

    Args:
        base: Seed offset of the run (0 for certification).
        index: Replicate number.
        which: Distinguishes the streams drawn within one replicate.

    Returns:
        A generator seeded with `[base + index, which]`.
    """
    return numpy.random.default_rng([base + index, which])


def rate(hits: int, n: int) -> dict[str, Any]:
    """A proportion with its Wilson 95 percent interval.

    Args:
        hits: Number of successes.
        n: Number of trials.

    Returns:
        The counts, their ratio under `rate` and the interval as
        `[low, high]` under `wilson`.
    """
    low, high = certificates.wilson_interval(hits, n, 0.95)
    return {
        "hits": int(hits),
        "n": int(n),
        "rate": hits / n,
        "wilson": [low, high],
    }


def coverage_row(
    covered: list[bool], z: list[float] | None = None
) -> dict[str, Any]:
    """Coverage with its interval and the mean standardised error.

    Args:
        covered: Whether each replicate's interval covered the truth.
        z: Standardised error `(estimate - truth) / se` of each replicate,
            or None when it does not apply; non-finite values are skipped.

    Returns:
        The `rate` row plus `meets_floor` (Wilson lower bound at least
        `COVERAGE_FLOOR`) and, when `z` is given, `mean_z` and whether it is
        within 0.1 of zero.
    """
    row = rate(sum(covered), len(covered))
    row["meets_floor"] = row["wilson"][0] >= COVERAGE_FLOOR
    if z is not None:
        arr = numpy.array([v for v in z if math.isfinite(v)])
        row["mean_z"] = float(arr.mean())
        row["z_within_0.1"] = bool(abs(arr.mean()) <= MEAN_Z_TOLERANCE)
    return row


def passes(row: dict[str, Any]) -> bool:
    """Whether a coverage row meets its floors.

    Args:
        row: A row from `coverage_row`.

    Returns:
        True if the coverage floor is met and, when present, the mean
        standardised error is within 0.1.
    """
    return bool(row["meets_floor"] and row.get("z_within_0.1", True))


def alarm_bound(alpha: float, replicates: int) -> float:
    """Largest false-alarm rate accepted for `alpha` at `replicates`.

    Args:
        alpha: Nominal false-alarm level.
        replicates: Number of replicates the rate is observed over.

    Returns:
        ``alpha + 2 * sqrt(alpha * (1 - alpha) / replicates)``.
    """
    return alpha + 2 * math.sqrt(alpha * (1 - alpha) / replicates)


def power_curve(
    grid: list[float], hits: dict[float, tuple[int, int]]
) -> dict[str, Any]:
    """Power by planted size and the smallest size with Wilson lower >= 0.8.

    Args:
        grid: Planted sizes; 0 is the no-violation world.
        hits: Mapping of each size to (flagged, replicates).

    Returns:
        The curve (a `rate` row per size) and `delta_min`, the smallest
        positive size whose Wilson lower bound is at least 0.8 (None if
        none).
    """
    curve = {}
    delta_min = None
    for delta in grid:
        h, n = hits[delta]
        row = rate(h, n)
        curve[str(delta)] = row
        if delta > 0 and delta_min is None and row["wilson"][0] >= POWER_FLOOR:
            delta_min = delta
    return {"curve": curve, "delta_min": delta_min}


def alarm_gate(
    detect: Any, grid: list[float], replicates: int, base: int
) -> dict[str, Any]:
    """False-alarm rate at 0 and power over the grid for one detector.

    `detect(delta, rng_x, rng_c)` returns True when the test flags.

    Args:
        detect: The detector, called with the planted size and two
            generators (exploration and confirmation).
        grid: Planted sizes, including 0.
        replicates: Replicates at size 0; other sizes use a quarter (at
            least 100).
        base: Seed offset.

    Returns:
        The false-alarm level, rate and bound, the power curve, `delta_min`
        and `passed` (false alarms within the bound and some size reaching
        power 0.8).
    """
    hits: dict[float, tuple[int, int]] = {}
    for delta in grid:
        runs = replicates if delta == 0 else max(replicates // 4, 100)
        flagged = sum(
            bool(detect(delta, stream(base, i, 0), stream(base, i, 1)))
            for i in range(runs)
        )
        hits[delta] = (flagged, runs)
    curve = power_curve(grid, hits)
    false_alarm = hits[0.0][0] / hits[0.0][1]
    bound = alarm_bound(ALPHA_FALSIFY, hits[0.0][1])
    return {
        "alpha_falsify": ALPHA_FALSIFY,
        "false_alarm": rate(*hits[0.0]),
        "false_alarm_bound": bound,
        **curve,
        "passed": bool(false_alarm <= bound and curve["delta_min"] is not None),
    }


# ---------------------------------------------------------------- selector
def gate_selector(replicates: int, base: int = 0) -> dict[str, Any]:
    """Refusal and non-refusal over generated charters.

    Args:
        replicates: Number of charters of each kind.
        base: Seed offset (0 for certification).

    Returns:
        The refusal rate when no strategy is eligible, the acceptance rate
        when one is, and `passed`.
    """
    covariates = "unconfoundedness_given:x0,x1"
    fields: dict[str, Any] = {
        "unit": "id",
        "time": "week",
        "adoption": "first",
        "instrument": "z",
        "running_variable": "r",
        "cutoff": 0.0,
    }
    names = list(select.STRATEGIES)

    def build(rng: numpy.random.Generator, eligible: bool) -> causal.Question:
        """A charter for a random strategy, made ineligible when asked.

        Args:
            rng: Random generator.
            eligible: Whether to keep every assumption and design field.

        Returns:
            The causal specification; when not eligible, one assumption word or
            one design field is dropped at random.
        """
        name = names[int(rng.integers(len(names)))]
        strategy = select.STRATEGIES[name]
        words = [
            covariates if w == "unconfoundedness_given" else w
            for w in strategy.assumptions
        ]
        kept = {k: v for k, v in fields.items() if k in strategy.design}
        if not eligible:
            pool = [(i, "") for i in range(len(words))] + [
                (-1, d) for d in strategy.design
            ]
            index, field = pool[int(rng.integers(len(pool)))]
            if index >= 0:
                words.pop(index)
            else:
                kept.pop(field)
        return causal.Question(
            treatment="t",
            outcome="y",
            estimand=strategy.estimands[0],
            assumptions=tuple(words),
            strategies=(name,),
            **kept,
        )

    refused = sum(
        select.select(build(stream(base, i), eligible=False)).status
        == select.NOT_IDENTIFIED
        for i in range(replicates)
    )
    kept_ok = sum(
        select.select(build(stream(base, i, 1), eligible=True)).status
        == select.IDENTIFIED
        for i in range(replicates)
    )
    refusal, acceptance = rate(refused, replicates), rate(kept_ok, replicates)
    return {
        "replicates": replicates,
        "refuses_when_none_eligible": refusal,
        "accepts_when_eligible": acceptance,
        "passed": bool(
            refusal["wilson"][0] >= SELECTION_FLOOR
            and acceptance["wilson"][0] >= SELECTION_FLOOR
        ),
    }


# ---------------------------------------------------------------- coverage
def aipw_world(
    name: str,
    estimand: str,
    learner: str,
    nonlinear: bool,
    runs: int,
    base: int,
) -> tuple[str, dict[str, Any]]:
    """Coverage of one AIPW configuration.

    Args:
        name: Label of the configuration.
        estimand: `ATE` or `ATT`.
        learner: Nuisance learner, `linear` or `boosted`.
        nonlinear: Whether the outcome surface is nonlinear.
        runs: Number of replicates.
        base: Seed offset.

    Returns:
        The label and the coverage row of the 95 percent interval for the
        true effect 2.0.
    """
    covered, z = [], []
    cols = tuple(f"x{j}" for j in range(f10_generators.COVARIATES))
    for i in range(runs):
        x = f10_generators.backdoor(
            stream(base, i, 0), 1000, nonlinear=nonlinear
        )
        c = f10_generators.backdoor(
            stream(base, i, 1), 1500, nonlinear=nonlinear
        )
        nuisances = aipw.fit_nuisances(x, cols, "t", "y", i, learner)
        out = aipw.estimate(nuisances, c, "t", "y", estimand, LEVEL)
        covered.append(out.low <= AIPW_TRUE_EFFECT <= out.high)
        z.append((out.estimate - AIPW_TRUE_EFFECT) / out.se)
    return name, coverage_row(covered, z)


def gate_coverage_aipw(replicates: int, base: int = 0) -> dict[str, Any]:
    """Coverage of the AIPW interval in three worlds.

    Args:
        replicates: Replicates for the linear worlds; the boosted world
            uses a quarter (at least 50).
        base: Seed offset (0 for certification).

    Returns:
        The coverage row of each world and `passed`.
    """
    boosted = max(replicates // 4, 50)
    rows = dict(
        [
            aipw_world(
                "linear_ate",
                "ATE",
                "linear",
                nonlinear=False,
                runs=replicates,
                base=base,
            ),
            aipw_world(
                "linear_att",
                "ATT",
                "linear",
                nonlinear=False,
                runs=replicates,
                base=base,
            ),
            aipw_world(
                "boosted_ate",
                "ATE",
                "boosted",
                nonlinear=True,
                runs=boosted,
                base=base,
            ),
        ]
    )
    return {
        "worlds": rows,
        "passed": bool(all(passes(r) for r in rows.values())),
    }


def gate_coverage_randomized(replicates: int, base: int = 0) -> dict[str, Any]:
    """Coverage of the difference in means.

    Args:
        replicates: Number of replicates.
        base: Seed offset (0 for certification).

    Returns:
        The coverage row and `passed`.
    """
    covered = []
    for i in range(replicates):
        frame = f10_generators.randomized(stream(base, i), 800)
        out = aipw.difference_in_means(frame, "t", "y", LEVEL)
        covered.append(out.low <= AIPW_TRUE_EFFECT <= out.high)
    row = coverage_row(covered)
    return {"coverage": row, "passed": passes(row)}


def gate_coverage_did(replicates: int, base: int = 0) -> dict[str, Any]:
    """Coverage of the group-time ATT, one group and staggered.

    Args:
        replicates: Replicates for the single-group world; the staggered
            world uses half (at least 50).
        base: Seed offset (0 for certification).

    Returns:
        The coverage row of each world and `passed`.
    """
    rows = {}
    for name, second, runs in (
        ("single_group", False, replicates),
        ("staggered", True, max(replicates // 2, 50)),
    ):
        covered, z = [], []
        for i in range(runs):
            frame = f10_generators.panel(
                stream(base, i), 100, 100, second_group=second
            )
            out = did.estimate(frame, "id", "t", "g", "y", LEVEL)
            covered.append(out.low <= DID_TRUE_EFFECT <= out.high)
            z.append((out.estimate - DID_TRUE_EFFECT) / out.se)
        rows[name] = coverage_row(covered, z)
    return {
        "worlds": rows,
        "passed": bool(all(passes(r) for r in rows.values())),
    }


def gate_coverage_iv(replicates: int, base: int = 0) -> dict[str, Any]:
    """Coverage of the headline set with a strong instrument.

    Args:
        replicates: Number of replicates.
        base: Seed offset (0 for certification).

    Returns:
        The coverage row, the number of weak-instrument flags and `passed`.
    """
    covered = []
    weak = 0
    for i in range(replicates):
        out = iv.estimate(
            f10_generators.iv(stream(base, i), 500, 1.0), "d", "y", "z"
        )
        covered.append(not out.excludes(2.0))
        weak += out.weak
    row = coverage_row(covered)
    return {"coverage": row, "weak_flags": weak, "passed": passes(row)}


def gate_coverage_rd(replicates: int, base: int = 0) -> dict[str, Any]:
    """Coverage of the bias-corrected interval.

    Args:
        replicates: Number of replicates.
        base: Seed offset (0 for certification).

    Returns:
        The coverage row and `passed`.
    """
    covered, z = [], []
    for i in range(replicates):
        out = rd.estimate(
            f10_generators.discontinuity(stream(base, i), 1500), "x", "y", 0.0
        )
        covered.append(out.low <= 1.0 <= out.high)
        z.append((out.bias_corrected - 1.0) / out.se)
    row = coverage_row(covered, z)
    return {"coverage": row, "passed": passes(row)}


def gate_iv_strength(replicates: int, base: int = 0) -> dict[str, Any]:
    """Weak-instrument handling: AR coverage, flag recall, false flags.

    Args:
        replicates: Number of replicates in each world.
        base: Seed offset (0 for certification).

    Returns:
        The headline coverage and weak-flag recall in the weak world, the
        false weak flags in the strong world and `passed`.
    """
    covered, flagged = [], 0
    for i in range(replicates):
        out = iv.estimate(
            f10_generators.iv(stream(base, i), 300, 0.08), "d", "y", "z"
        )
        covered.append(not out.excludes(2.0))
        flagged += out.weak
    strong = sum(
        iv.estimate(
            f10_generators.iv(stream(base, i, 1), 500, 1.0), "d", "y", "z"
        ).weak
        for i in range(replicates)
    )
    cover = coverage_row(covered)
    recall = rate(flagged, replicates)
    false_weak = rate(strong, replicates)
    return {
        "weak_world_headline_coverage": cover,
        "weak_flag_recall": recall,
        "strong_world_false_weak": false_weak,
        "passed": bool(
            cover["meets_floor"]
            and recall["wilson"][0] >= RECALL_FLOOR
            and false_weak["rate"] <= FALSE_ALARM_CEILING
        ),
    }


def gate_size_synth(replicates: int, base: int = 0) -> dict[str, Any]:
    """Rejection rate of placebo-in-space with no effect.

    Args:
        replicates: Scales the number of replicates (a quarter, at least
            100).
        base: Seed offset (0 for certification).

    Returns:
        The size with its bound, the power at an effect of 3 over 60
        replicates and `passed`.
    """
    runs = max(replicates // 4, 100)
    rejected = 0
    for i in range(runs):
        frame = f10_generators.synthetic(stream(base, i), tau=0.0)
        out = synth.estimate(frame, "u", "t", "y", 0, 12)
        rejected += out.p_value <= SYNTH_LEVEL
    bound = 0.05 + 2 * math.sqrt(0.05 * 0.95 / runs)
    power_runs = 60
    detected = sum(
        synth.estimate(
            f10_generators.synthetic(stream(base, i, 1), tau=3.0),
            "u",
            "t",
            "y",
            0,
            12,
        ).p_value
        <= SYNTH_LEVEL
        for i in range(power_runs)
    )
    return {
        "replicates": runs,
        "size": rate(int(rejected), runs),
        "size_bound": bound,
        "power_at_effect_3": rate(int(detected), power_runs),
        "passed": bool(rejected / runs <= bound),
    }


def gate_overlap(replicates: int, base: int = 0) -> dict[str, Any]:
    """False refusals, recall and coverage curves over planted overlap.

    Args:
        replicates: Scales the number of replicates per setting (a quarter,
            at least 100).
        base: Seed offset (0 for certification).

    Returns:
        Per minimum propensity the gate-fire rate, coverage, mean estimated
        minimum propensity and effective sample size share; the worst false
        refusal rate and worst recall bound; and `passed`.
    """
    runs = max(replicates // 4, 100)
    cols = tuple(f"x{j}" for j in range(f10_generators.COVARIATES))
    curves: dict[str, dict[str, Any]] = {}
    for p_min in (0.5, 0.1, 0.02, 0.005):
        flagged, covered, ess, lowest = 0, [], [], []
        for i in range(runs):
            x = f10_generators.backdoor(stream(base, i, 0), 1000, p_min=p_min)
            c = f10_generators.backdoor(stream(base, i, 1), 1500, p_min=p_min)
            nuisances = aipw.fit_nuisances(x, cols, "t", "y", i)
            e = nuisances.predict(x)[0]
            t = x["t"].to_numpy()
            report = aipw.overlap_report(e, t)
            flagged += not report["passed"]
            lowest.append(report["min_propensity"])
            out = aipw.estimate(nuisances, c, "t", "y", "ATE", LEVEL)
            covered.append(out.low <= AIPW_TRUE_EFFECT <= out.high)
            ess.append(out.overlap["ess_treated"] / out.n)
        curves[str(p_min)] = {
            "gate_fires": rate(flagged, runs),
            "coverage_when_untrimmed": rate(sum(covered), runs),
            "mean_min_estimated_propensity": float(numpy.mean(lowest)),
            "mean_ess_treated_over_n": float(numpy.mean(ess)),
        }
    valid = [curves["0.5"]["gate_fires"], curves["0.1"]["gate_fires"]]
    bad = [curves["0.02"]["gate_fires"], curves["0.005"]["gate_fires"]]
    false_refuse = max(v["rate"] for v in valid)
    return {
        "replicates": runs,
        "eps": aipw.OVERLAP_EPS,
        "share_gate": aipw.OVERLAP_SHARE,
        "curves": curves,
        "false_refuse_worst": false_refuse,
        "recall_wilson_lower_worst": min(v["wilson"][0] for v in bad),
        "passed": bool(
            false_refuse <= FALSE_ALARM_CEILING
            and all(v["wilson"][0] >= RECALL_FLOOR for v in bad)
        ),
    }


# ----------------------------------------------------------------- battery
def gate_battery_nco(replicates: int, base: int = 0) -> dict[str, Any]:
    """Negative-control outcome against a shared hidden confounder.

    Args:
        replicates: Replicates at the no-violation size.
        base: Seed offset (0 for certification).

    Returns:
        The `alarm_gate` result over confounder strengths 0 to 0.8.
    """
    cols = tuple(f"x{j}" for j in range(f10_generators.COVARIATES))

    def detect(delta: float, rx: Any, rc: Any) -> bool:
        """Whether the negative-control test flags a world.

        Args:
            delta: Strength of the shared hidden confounder.
            rx: Generator for the exploration sample.
            rc: Generator for the confirmation sample.

        Returns:
            True if the falsification battery on the negative-control outcome
            fails.
        """
        x = f10_generators.backdoor(
            rx, 1000, hidden_d=delta, hidden_y=delta, hidden_nco=delta
        )
        c = f10_generators.backdoor(
            rc, 3000, hidden_d=delta, hidden_y=delta, hidden_nco=delta
        )
        nuisances = aipw.fit_nuisances(x, cols, "t", "nco")
        out = aipw.estimate(nuisances, c, "t", "nco", "ATE", LEVEL)
        test = battery.wald_test(
            "nco", "negative_outcome", out.estimate, out.se
        )
        return not battery.run_battery([test], ALPHA_FALSIFY).passed

    return alarm_gate(detect, [0.0, 0.1, 0.2, 0.3, 0.5, 0.8], replicates, base)


def gate_battery_pretrend(replicates: int, base: int = 0) -> dict[str, Any]:
    """Pre-period Wald test against a differential linear trend.

    Args:
        replicates: Replicates at the no-violation size.
        base: Seed offset (0 for certification).

    Returns:
        The `alarm_gate` result over trends 0 to 0.3 per period.
    """

    def detect(delta: float, rx: Any, rc: Any) -> bool:  # noqa: ARG001 -- alarm_gate calls detect(delta, rng_x, rng_c)
        """Whether the pre-trend test flags a world.

        Args:
            delta: Differential linear trend per period.
            rx: Generator for the panel.
            rc: Unused.

        Returns:
            True if the falsification battery on the pre-trend p-value fails.
        """
        frame = f10_generators.panel(rx, 150, 150, drift=delta)
        out = did.estimate(frame, "id", "t", "g", "y")
        test = battery.Test("pretrend", "placebo", float(out.pretrend_p or 1.0))
        return not battery.run_battery([test], ALPHA_FALSIFY).passed

    return alarm_gate(
        detect, [0.0, 0.05, 0.1, 0.15, 0.2, 0.3], replicates, base
    )


def gate_density_rd(replicates: int, base: int = 0) -> dict[str, Any]:
    """Binomial density check against sorting across the cutoff.

    Args:
        replicates: Replicates at the no-violation size.
        base: Seed offset (0 for certification).

    Returns:
        The `alarm_gate` result over sorting shares 0 to 0.7.
    """

    def detect(delta: float, rx: Any, rc: Any) -> bool:  # noqa: ARG001 -- alarm_gate calls detect(delta, rng_x, rng_c)
        """Whether the density check flags a world.

        Args:
            delta: Share of units just below the cutoff that move above it.
            rx: Generator for the world.
            rc: Unused.

        Returns:
            True if the density p-value is below `ALPHA_FALSIFY`.
        """
        out = rd.estimate(
            f10_generators.discontinuity(rx, 1500, sorting=delta), "x", "y", 0.0
        )
        return out.density_p < ALPHA_FALSIFY

    return alarm_gate(detect, [0.0, 0.1, 0.2, 0.3, 0.5, 0.7], replicates, base)


def gate_battery_rd(replicates: int, base: int = 0) -> dict[str, Any]:
    """Placebo cutoffs and covariate balance against a covariate jump.

    Args:
        replicates: Scales the replicates at the no-violation size (half,
            at least 200).
        base: Seed offset (0 for certification).

    Returns:
        The `alarm_gate` result over covariate jumps 0 to 0.5.
    """

    def detect(delta: float, rx: Any, rc: Any) -> bool:  # noqa: ARG001 -- alarm_gate calls detect(delta, rng_x, rng_c)
        """Whether the placebo and balance tests flag a world.

        Args:
            delta: Jump of the predetermined covariate at the cutoff.
            rx: Generator for the world.
            rc: Unused.

        Returns:
            True if the falsification battery over the placebo cutoffs and the
            covariate balance fails.
        """
        out = rd.estimate(
            f10_generators.discontinuity(rx, 1500, jump=delta),
            "x",
            "y",
            0.0,
            covariates=("z",),
        )
        tests = [
            battery.Test(f"placebo:{k}", "placebo", float(v["p"]))
            for k, v in out.placebos.items()
        ] + [
            battery.Test(f"balance:{k}", "negative_outcome", float(v["p"]))
            for k, v in out.balance.items()
        ]
        return not battery.run_battery(tests, ALPHA_FALSIFY).passed

    return alarm_gate(
        detect, [0.0, 0.05, 0.1, 0.2, 0.3, 0.5], max(replicates // 2, 200), base
    )


# ------------------------------------------------------------- sensitivity
def sensitivity_world(
    rng: numpy.random.Generator, n: int, multiple: float
) -> pandas.DataFrame:
    """Hidden `u` at `multiple` times the strongest covariate's strength.

    Args:
        rng: Random generator.
        n: Number of rows.
        multiple: Strength of the hidden confounder relative to the
            strongest observed covariate.

    Returns:
        A frame with covariates `x0`..`x2`, treatment `d` and outcome `y`
        (true effect 1.0).
    """
    x = rng.standard_normal((n, 3))
    u = rng.standard_normal(n)
    d = (
        0.5 * x[:, 0]
        + 0.2 * x[:, 1]
        + 0.5 * multiple * u
        + rng.standard_normal(n)
    )
    y = (
        1.0 * d
        + 1.0 * x[:, 0]
        + 0.4 * x[:, 1]
        + 1.0 * multiple * u
        + rng.standard_normal(n)
    )
    return pandas.DataFrame(
        {"x0": x[:, 0], "x1": x[:, 1], "x2": x[:, 2], "d": d, "y": y}
    )


def gate_sensitivity_bound(replicates: int, base: int = 0) -> dict[str, Any]:
    """Coverage of the benchmarked interval inside, at and beyond the bound.

    Args:
        replicates: Scales the replicates (half, at least 200).
        base: Seed offset (0 for certification).

    Returns:
        For hidden confounders at half, once and twice the benchmark, the
        coverage and the share of runs where the claim still holds, and
        `passed` (the inside-the-benchmark coverage floor).
    """
    runs = max(replicates // 2, 200)
    rows = {}
    for name, multiple in (
        ("inside_half", 0.5),
        ("boundary", 1.0),
        ("beyond_2x", 2.0),
    ):
        covered, holds = [], 0
        for i in range(runs):
            frame = sensitivity_world(stream(base, i), 1500, multiple)
            report = sensitivity.assess(
                frame, "d", "y", ("x0", "x1", "x2"), "strongest_observed", 1.0
            )
            covered.append(report.low <= 1.0 <= report.high)
            holds += report.holds
        rows[name] = {
            "coverage": rate(sum(covered), runs),
            "claim_still_holds": rate(holds, runs),
        }
    inside = rows["inside_half"]["coverage"]
    return {
        "replicates": runs,
        "worlds": rows,
        "passed": bool(inside["wilson"][0] >= COVERAGE_FLOOR),
    }


def gate_sensitivity_did(replicates: int, base: int = 0) -> dict[str, Any]:
    """Coverage of the relative-magnitudes interval under a linear trend.

    Args:
        replicates: Scales the replicates (a quarter, at least 100).
        base: Seed offset (0 for certification).

    Returns:
        The coverage rows of the sensitivity interval at M = 1 and of the
        plain interval, the drift and `passed`.
    """
    runs = max(replicates // 4, 100)
    covered_m1, covered_m0 = [], []
    for i in range(runs):
        frame = f10_generators.panel(stream(base, i), 150, 150, drift=0.1)
        out = did.estimate(frame, "id", "t", "g", "y")
        low, high = out.sensitivity(1.0)
        covered_m1.append(low <= DID_TRUE_EFFECT <= high)
        covered_m0.append(out.low <= DID_TRUE_EFFECT <= out.high)
    m1 = coverage_row(covered_m1)
    return {
        "replicates": runs,
        "drift_per_period": 0.1,
        "m1_coverage": m1,
        "m0_coverage": rate(sum(covered_m0), runs),
        "passed": bool(m1["meets_floor"]),
    }


# ------------------------------------------------------------ false confirm
def confirm_world(kind: str, runs: int, base: int) -> dict[str, Any]:
    """Share of end-to-end runs that reach `confirmed` in one world.

    Args:
        kind: `shared_confounder`, `unshared_confounder` or `clean`.
        runs: Number of end-to-end runs.
        base: Seed offset.

    Returns:
        The rate of runs whose claim tier is `confirmed` and of those whose
        tier is `attested`.
    """
    knobs: dict[str, Any] = {
        "shared_confounder": {"hidden_d": 0.8, "hidden_y": 0.8},
        "unshared_confounder": {
            "hidden_d": 0.8,
            "hidden_y": 0.8,
            "hidden_nco": 0.0,
        },
        "clean": {},
    }[kind]
    graph = scmle.data.admissibility.dag.Dag.build(
        source="certificate",
        edges=(
            *(
                e
                for j in range(f10_generators.COVARIATES)
                for e in ((f"x{j}", "t"), (f"x{j}", "y"), (f"x{j}", "nco"))
            ),
            ("t", "y"),
        ),
    )
    columns = ",".join(f"x{j}" for j in range(f10_generators.COVARIATES))
    spec = causal.Question(
        treatment="t",
        outcome="y",
        estimand="ATE",
        assumptions=(
            f"unconfoundedness_given:{columns}",
            "positivity",
            "no_interference",
        ),
        strategies=("backdoor_aipw",),
        negative_controls=(causal.NegativeControl("outcome", "nco"),),
        strata=("region",),
        sensitivity=causal.Confounding("strongest_observed", 0.2),
    )
    named = dict.fromkeys(spec.assumptions, "certificate harness, 2026-11-04")
    error_budget = budget.ErrorBudget.create(0.05, 0.1)
    confirmed = tiers_attested = 0
    for i in range(runs):
        x = f10_generators.backdoor(stream(base, i, 0), 1200, **knobs)
        c = f10_generators.backdoor(stream(base, i, 1), 3000, **knobs)
        out = run.identify(
            spec,
            x,
            c,
            error_budget,
            attestors=named,
            dag=graph,
            seed=i,
            certified=GATES_FOR_RUN,
        )
        tier = out.documents["causal_claims.json"]["tier"]
        confirmed += tier == "confirmed"
        tiers_attested += tier == "attested"
    return {
        "confirmed": rate(confirmed, runs),
        "attested": rate(tiers_attested, runs),
    }


GATES_FOR_RUN = tuple(
    g for g in GATES if g not in ("selector", "false_confirm")
)


def gate_false_confirm(replicates: int, base: int = 0) -> dict[str, Any]:
    """The tier rule end to end against shared and unshared confounding.

    Args:
        replicates: Scales the runs per world (a sixth, at least 100).
        base: Seed offset (0 for certification).

    Returns:
        The confirmation rates in the three worlds, a note and `passed`
        (shared-confounder confirmation within 0.05 with Wilson upper bound
        at most 0.08).
    """
    runs = max(replicates // 6, 100)
    rows = {
        kind: confirm_world(kind, runs, base)
        for kind in ("shared_confounder", "unshared_confounder", "clean")
    }
    shared = rows["shared_confounder"]["confirmed"]
    return {
        "replicates": runs,
        "worlds": rows,
        "note": (
            "all other gates were marked certified for this run; the "
            "unshared-confounder row is the known blind spot"
        ),
        "passed": bool(
            shared["rate"] <= FALSE_ALARM_CEILING
            and shared["wilson"][1] <= FALSE_CONFIRM_UPPER_CEILING
        ),
    }


RUNNERS = {
    "selector": (gate_selector, 200),
    "coverage_aipw": (gate_coverage_aipw, 2000),
    "coverage_randomized": (gate_coverage_randomized, 2000),
    "coverage_did": (gate_coverage_did, 2000),
    "coverage_iv": (gate_coverage_iv, 2000),
    "coverage_rd": (gate_coverage_rd, 2000),
    "size_synth": (gate_size_synth, 500),
    "iv_strength": (gate_iv_strength, 2000),
    "overlap": (gate_overlap, 2000),
    "battery_nco": (gate_battery_nco, 2000),
    "battery_pretrend": (gate_battery_pretrend, 2000),
    "density_rd": (gate_density_rd, 2000),
    "battery_rd": (gate_battery_rd, 2000),
    "sensitivity_bound": (gate_sensitivity_bound, 1000),
    "sensitivity_did": (gate_sensitivity_did, 500),
    "false_confirm": (gate_false_confirm, 1800),
}


def runner_objects() -> dict[str, list[Any]]:
    """Harness code each gate rests on, besides the package methods.

    Returns:
        Mapping of gate name to the runner, the generators and the shared
        helpers.
    """
    return {
        name: [entry[0], f10_generators, alarm_gate, coverage_row, rate]
        for name, entry in RUNNERS.items()
    }


def current_digests() -> dict[str, dict[str, str]]:
    """Method and harness hashes of every gate.

    Returns:
        Mapping of gate name to its `method_hash` and `code_hash`.
    """
    methods = certified.method_hashes()
    harness = runner_objects()
    return {
        name: {
            "method_hash": methods[name],
            "code_hash": certificates.code_hash(
                *certified.METHOD_SOURCES[name], *harness[name]
            ),
        }
        for name in GATES
    }


def certify_all(only: tuple[str, ...] = ()) -> dict[str, Any]:
    """Runs the gates on certification seeds.

    Args:
        only: Names of the gates to run; all when empty.

    Returns:
        The certificate document: version, development seed base, what was
        not measured and, per gate, the measurement with its hashes and seed
        range.
    """
    document: dict[str, Any] = {
        "version": 1,
        "dev_seed_base": DEV_SEED_BASE,
        "not_measured": list(NOT_MEASURED),
        "gates": {},
    }
    digests = current_digests()
    for name, (runner, runs) in RUNNERS.items():
        if only and name not in only:
            continue
        measured = runner(runs, 0)
        measured.update(digests[name])
        measured["seeds"] = [0, runs]
        document["gates"][name] = measured
        print(name, measured["passed"], file=sys.stderr)
    return document


def load() -> dict[str, Any] | None:
    """The stored certificates, or None.

    Returns:
        The parsed certificate file, or None when it does not exist.
    """
    if not STORE_PATH.exists():
        return None
    return json.loads(STORE_PATH.read_text(encoding="utf-8"))


def problems(document: dict[str, Any] | None = None) -> list[str]:
    """Why the stored certificates cannot be trusted (empty when current).

    Args:
        document: The certificate document; loaded from `STORE_PATH` when
            None.

    Returns:
        Messages for a missing file, a gate without a certificate and a
        gate whose code hash differs from the current code.
    """
    if document is None:
        document = load()
        if document is None:
            return ["f10_certificates.json is missing"]
    found = []
    digests = current_digests()
    for name in GATES:
        entry = document.get("gates", {}).get(name)
        if entry is None:
            found.append(f"{name}: no certificate")
        elif entry.get("code_hash") != digests[name]["code_hash"]:
            found.append(f"{name}: stale (code changed since measurement)")
    return found


def descriptive_only(document: dict[str, Any] | None = None) -> list[str]:
    """Gates that missed their floors; their methods stay descriptive.

    Args:
        document: The certificate document; loaded from `STORE_PATH` when
            None.

    Returns:
        The names of gates that did not pass; every gate when the file is
        missing.
    """
    document = document or load()
    if document is None:
        return list(GATES)
    return [k for k, v in document["gates"].items() if not v["passed"]]


def main(arguments: list[str] | None = None) -> int:
    """Writes or checks the certificate file.

    Args:
        arguments: Command-line arguments; `None` reads `sys.argv`.

    Returns:
        0 after a write, or after a check that finds no problem; 1 when the
        check finds problems.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    parser.add_argument("--only", nargs="*", default=[])
    options = parser.parse_args(arguments)
    if options.write:
        document = certify_all(tuple(options.only))
        if options.only and STORE_PATH.exists():
            merged = json.loads(STORE_PATH.read_text(encoding="utf-8"))
            merged["gates"].update(document["gates"])
            document = merged
        STORE_PATH.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"descriptive_only": descriptive_only(document)}))
        return 0
    issues = problems()
    print(
        json.dumps(
            {
                "ok": not issues,
                "problems": issues,
                "descriptive_only": descriptive_only(),
            },
            indent=2,
        )
    )
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
