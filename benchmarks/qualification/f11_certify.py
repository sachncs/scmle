"""Measured gates for the decision layer.

Every floor below is fixed in this docstring before the first certification
run and is not retuned: a gate that misses its floor is filed with its
numbers and `passed` false, and that method stays descriptive. Certification
seeds are `0 .. runs - 1`; development looks use seeds from `DEV_SEED_BASE`,
which never overlap. Intervals are Wilson 95 percent. `tol(p, R)` is
`2 sqrt(p (1 - p) / R)`. Worlds are simulated, except in the three gates
that use the randomized Hillstrom email data (biased subsamples, A/A
promotion and the positive control); real data may violate them. Each gate is
the function named in brackets and is filed under the key named with it,
as listed in `GATES`.

Offline

* Threshold (gate A1 `gate_a1`). `n_dev = 5000`, `R = 200` worlds each for
  score skews 1.0 (calibrated), 0.6 and 1.6 (monotone distortions). Costs
  `c_FP = 5`, `c_FN = 100`. Excess cost is the exact expected cost of the
  frozen rule minus that of the oracle rule on the true risk, relative to the
  oracle. Pass: median excess `<= 1%` in every world. The full distribution is
  filed.
* Calibration check (gate A1c `gate_a1c`). Level 0.01, `n = 5000`, `R = 2000`.
  Pass: false-alarm rate on calibrated scores `<= 0.01 + tol(0.01, R)`; power
  `>= 0.80` at skews 0.6 and 1.6.
* Net benefit identity (gate A2 `gate_a2`):
  `NB(t) = (cost_none - cost_rule) / c_FN` to 1e-12 on 1000 random tables; the
  sample net benefit is within `3 se` of the binormal closed form over
  `R = 200` worlds of `n = 20000`.
* Band coverage (gate A3 `gate_a3`). Binormal world, prevalence 0.2,
  `delta = 1.5`, rule `score >= 0.8`, range `[0.1, 0.4]` on 21 points, two
  baselines, `B = 400`, level 0.05 one-sided, `R = 2000` per setting:
  independent rows at `n = 2000` and `20000`; clusters of 10 with shared sd 0.5
  (`n = 2000`, `20000`). Pass: Wilson lower bound of "the true differences are
  above the band" `>= 0.93` in all four settings.
* False usefulness (gate A4 `gate_a4`). Score independent of the label,
  prevalence 0.2, rule acts on 20 percent, `R = 2000` at `n = 2000` and
  `20000`. Pass: share called `useful` `<= 0.05 + tol(0.05, R)`. Power by
  effect size is filed.
* Off-policy evaluation (gate A5 `gate_a5`). `n = 2000`, `R = 2000`, propensity
  floor 0.05, sharpness 1 (`ESS/n` is filed). Estimators: IPW, SNIPS, DR with
  an outcome model fitted on a separate log of 3000 (with the step feature, and
  without it), IPW and DR clipped at 10. Pass: DR coverage (both outcome
  models) Wilson lower `>= 0.93` among replicates with `ESS/n >= 0.2`. Refusal:
  worlds with a planted support violation (the logging policy never treats
  `x0 > 1`; or treats with probability 0.005 for `x0 > 1.5`), `R = 300` each,
  refused in `>= 95%` with Wilson lower `>= 0.90`; worlds without, refused in
  `<= 5%`.
* Hillstrom biased subsamples (gate A6 `gate_a6`; needs the data file). Men's
  e-mail against no e-mail, conversion. Development half: outcome model and a
  threshold policy; evaluation half: full-data value (randomized, `mu = 0.5`)
  as reference; 200 subsamples with covariate-dependent logging propensity
  `clip(sigmoid(z(recency)), 0.1, 0.9)`. Pass: the DR interval covers the
  reference in `>= 90%` of 200 subsamples (Wilson interval filed).
* Fairness verdicts (gate A7 `gate_a7`). Two groups with true-positive rates
  0.7 and `0.7 - gap` at a fixed rule, `n = 20000`, `R = 2000`,
  `max_gap = 0.05`, `gap` in 0, 0.03, 0.07, 0.10, 0.15. Pass: "satisfied" when
  `gap > 0.05` has Wilson upper bound `<= 0.08` (and point rate `<= 0.05`) at
  0.07, 0.10, 0.15; the upper bound covers the true gap with Wilson lower
  `>= 0.93` at every gap. The share satisfied at 0 and 0.03 is filed.
* Group thresholds (gate A7b `gate_a7b`). Binormal groups (shift 1.0),
  `t = 0.3`, 20 worlds of `n = 100000`. Pass: the true net benefit of the
  fitted cuts is within 2 percent of the equal-true-positive-rate oracle
  (fitted may exceed it: the development bound allows a small gap).
* Infeasible sets (gate A8 `gate_a8`). Incompatible pairs with base rates 0.2
  and 0.5 (`n = 2000`), `R = 100` per pair for three pairs: refused in 100
  percent. Base rates both 0.3: refused in `<= 0.05 + tol(0.05, 300)` of 300.

Online (level `alpha_promote = 1/50` for one experiment, `alpha_rollback =
1/100`)

* Optional stopping (gate B1 `gate_b1`). Control 0.10, challenger 0.12
  (`Delta = delta_min = 0.02`), `q = 0.5`, a look after every unit up to
  `n_max = 3000`, `R = 10000`. Pass: share of runs that ever promote
  `<= 0.02 + tol(0.02, R)`. The same world with `abtest.compare_arms` after
  each batch of 500 (6 looks, `R = 100`, null `Delta = 0`) is run and its
  false-adopt rate filed, not gated. 100 runs are also passed through the
  `Rollout` class and must agree with the vectorised computation exactly.
* Intersection-union (gate B2 `gate_b2`). Primary effect 0.06 (3 delta_min),
  guardrail at the boundary (control 0.20, challenger 0.25, margin 0.05),
  `n_max = 6000`, `R = 5000`. Pass: promotion share `<= 0.02 + tol(0.02, R)`.
* Cost of anytime-validity (gate B3 `gate_b3`). Expected sample number to
  promote at `Delta = 2 delta_min` for a binary metric (control 0.10) and a
  bounded metric (Beta(2, 8) control, mean shift 0.04), `n_max = 20000`,
  `R = 500`, against the fixed-horizon sizes
  (`feasibility.n_required_two_proportions` and the null-shifted normal
  approximation). Reported, no floor.
* Rollback (gate B4 `gate_b4`). Boundary `Delta = -tau` (0.20 against 0.18,
  `tau = 0.02`), `n_max = 4000`, `R = 10000`. Pass: false rollback
  `<= 0.01 + tol(0.01, R)`. Detection delay (median and 90th percentile of
  units, `R = 1000`, `n_max = 20000`) for true losses `2 tau` and `4 tau` is
  filed.
* Budget algebra (gate B5 `gate_b5`). 2000 random experiment sequences:
  assigned levels sum to at most `alpha_promote` as `Fraction`s, an overspend
  raises, a serialised budget rebuilds bit for bit. Pass: no violation.
* Sample-ratio check (gate B6 `gate_b6`) at level 0.001. Correct assignment (q
  = 0.5), `n = 100000`, a look after every unit for the anytime form,
  `R = 1000`. Pass: size `<= 0.01 + tol(0.01, R)` for the anytime form and for
  the exact test at the final count; power `>= 0.95` at a 2 percent distortion
  (0.52) for both.
* A/A on Hillstrom (gate B7 `gate_b7`; needs the data file). All 64000 rows,
  arms relabelled at random, random stream order, first 20000 rows, conversion,
  `delta_min = 0.001`, `R = 1000`. Pass: false promotion
  `<= 0.02 + tol(0.02, R)`.
* Positive control on Hillstrom (gate B8 `gate_b8`; needs the data file). The
  real arms, conversion, `delta_min = 0.002`, 200 stream orders. Pass: promoted
  in `>= 90%` of orders and the direction agrees with the fixed-sample result
  of `scipy.stats.norm` on the full data; the numbers are filed, not asserted.
* Lifecycle with drift (gate B9 `gate_b9`). The control rate random-walks. (a)
  no effect: false promotion `<= 0.02 + tol`, `R = 2000`; (b) effect
  `delta_min / 2`: false promotion `<= 0.02 + tol` and `hold_final` in
  `>= 1 - 0.02 - tol` of runs; (c) a champion drifting from 0 to -0.015 (never
  beyond `tau`): false rollback `<= 0.01 + tol(0.01, R)`. (d) drifting from 0
  to -0.06: rollback share filed. Amendment of 2026-11-06, written before the
  first run of the hook path and leaving the four floors above as they were:
  the same worlds are also run through `RolloutHook` on a real
  `LifecycleManager` and a registry of two small bundles (`through_hook`). Per
  scenario `HOOK_WORLDS = 100` worlds (seeds `0 .. 99` of the scenario's
  stream, the first worlds of the vectorised run), fed in batches of 500 to
  `hook.observe` (promotion scenarios a, b) or `hook.watch` after a real
  `hook.promote` with a signed approval (rollback scenarios c, d). The hook's
  decision in each world is compared with the vectorised e-process on the same
  world at the level the hook assigned (its experiment's or its monitor's own
  target). Added pass condition: the two agree in every world
  (`agree == worlds`) in each of the four scenarios; the hook-path false
  promotion and false rollback counts are filed with Wilson intervals and are
  not gated (100 worlds cannot bound a rate of 0.01). A world in which the hook
  stops on the sample-ratio check (`hold_invalid`) is counted separately and
  not as a disagreement.
* Determinism (gate B10 `gate_b10`): two runs of a fixed battery give the same
  SHA-256 of the report, and no module of `scmle.decision` imports custody,
  vault or ledger code. The check ran on one interpreter (the one that
  measured).

What is not measured here: reproduction by an outside party, other
interpreters (Python 3.11 and 3.14 bit identity of the ledger replay), the
Criteo uplift data (its licence is non-commercial and 300 MB; not
downloaded), the Open Bandit Dataset, R `dcurves` / `rmda` and Open Bandit
Pipeline reference outputs (not installed here), and `confseq`.

    python -m benchmarks.qualification.f11_certify --write
    python -m benchmarks.qualification.f11_certify --check
    python -m benchmarks.qualification.f11_certify --dev
"""

import argparse
import ast
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

import numpy
import pandas
from scipy import stats

from benchmarks.qualification import f11_generators
from scmle.artifacts import bundle
from scmle.contracts import tasks
import scmle.contracts.data
from scmle.core import randomness
from scmle.data.charter import feasibility
import scmle.data.charter.decision
from scmle.data.curation import preprocessing
from scmle.data.detectors import certificates
from scmle.data.framing import costs
import scmle.data.ledger.budget
from scmle.decision import anytime
from scmle.decision import constraints
from scmle.decision import design
from scmle.decision import netbenefit
from scmle.decision import ope
from scmle.decision import rollout
from scmle.decision import threshold
from scmle.governance import deployment_ledger
from scmle.governance import lifecycle
from scmle.governance import rollout_hook
import scmle.governance.registry
import scmle.models.registry
from scmle.monitoring import abtest

STORE_PATH = Path(__file__).with_name("f11_certificates.json")
DEV_SEED_BASE = 10_000
HILLSTROM_ENV = "SCMLE_HILLSTROM"
HILLSTROM_SHA256 = (
    "0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece"
)
ALPHA_PROMOTE = 0.02
ALPHA_ROLLBACK = 0.01
C_FP, C_FN = 5.0, 100.0
GATES = (
    "A1",
    "A1c",
    "A2",
    "A3",
    "A4",
    "A5",
    "A6",
    "A7",
    "A7b",
    "A8",
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B9",
    "B10",
)
NOT_MEASURED = (
    "outside-party reproduction",
    "Python 3.11 versus 3.14 bit identity (one interpreter only)",
    "Criteo uplift referent (not downloaded; licence is non-commercial)",
    "Open Bandit Dataset referent",
    "R dcurves / rmda decision-curve reference output",
    "Open Bandit Pipeline reference output",
    "confseq reference confidence sequences",
)


def tol(p: float, runs: int) -> float:
    """Two binomial standard errors at rate `p`.

    Args:
        p: The rate.
        runs: Number of replicates the rate is observed over.

    Returns:
        ``2 * sqrt(p * (1 - p) / runs)``.
    """
    return 2.0 * math.sqrt(p * (1.0 - p) / runs)


def wilson(hits: int, runs: int) -> list[float]:
    """Wilson 95 percent interval as a list.

    Args:
        hits: Number of successes.
        runs: Number of trials.

    Returns:
        The lower and upper bounds.
    """
    low, high = certificates.wilson_interval(hits, runs, 0.95)
    return [low, high]


def rng_for(base: int, name: str, index: int = 0) -> numpy.random.Generator:
    """A named stream for one replicate or chunk.

    Args:
        base: Seed offset.
        name: Name of the stream within the gate.
        index: Replicate or chunk number.

    Returns:
        A generator seeded from `base + index` and `name`.
    """
    return randomness.component_rng(base + index, f"f11.{name}", 0)


def cost_model() -> costs.DecisionCostModel:
    """The cost model of the threshold worlds.

    Returns:
        A model with false-positive cost `C_FP` and false-negative cost
        `C_FN`, both from a human source.
    """
    human = (costs.CostSource.HUMAN,) * 4
    return costs.DecisionCostModel(
        0.0, C_FN, C_FP, 0.0, 0.02, 0.15, human, costs.CostSource.HUMAN
    )


def summary(values: numpy.ndarray) -> dict[str, float]:
    """Quantiles of a distribution.

    Args:
        values: The sample.

    Returns:
        The 5th, 25th, 50th, 75th and 95th percentiles and the maximum.
    """
    q = numpy.quantile(values, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "q05": float(q[0]),
        "q25": float(q[1]),
        "median": float(q[2]),
        "q75": float(q[3]),
        "q95": float(q[4]),
        "max": float(values.max()),
    }


# --- threshold, calibration check, net benefit identity ----------------------


def gate_a1(runs: int, base: int = 0) -> dict[str, Any]:
    """Excess cost of the chosen threshold over the oracle.

    Args:
        runs: Number of worlds per score skew.
        base: Seed offset (0 for certification).

    Returns:
        Per skew the oracle cost, the distribution of relative excess cost,
        the counts of decision paths and whether the median floor is met;
        and `passed`.
    """
    model = cost_model()
    target = threshold.bayes_threshold(C_FP, C_FN)
    result: dict[str, Any] = {"worlds": {}}
    passed = True
    for skew in (1.0, 0.6, 1.6):
        fresh_rng = rng_for(base, "a1.fresh", int(skew * 10))
        _, p_fresh, _ = f11_generators.calibration_world(
            fresh_rng, 400_000, 1.0
        )
        oracle = f11_generators.beta_cost(
            p_fresh, (p_fresh >= target).astype(float), C_FP, C_FN
        )
        excess = []
        paths: dict[str, int] = {}
        for index in range(runs):
            rng = rng_for(base, f"a1.{skew}", index)
            score, _, y = f11_generators.calibration_world(rng, 5000, skew)
            decision = threshold.choose_threshold(
                score, y, model, seed=base + index, replicates=50
            )
            paths[decision.rule.path] = paths.get(decision.rule.path, 0) + 1
            fresh_score = p_fresh**skew
            cost = f11_generators.beta_cost(
                p_fresh, decision.rule.act(fresh_score), C_FP, C_FN
            )
            excess.append((cost - oracle) / oracle)
        values = numpy.array(excess)
        meets = float(numpy.median(values)) <= 0.01
        passed = passed and meets
        result["worlds"][str(skew)] = {
            "oracle_cost": oracle,
            "excess": summary(values),
            "paths": paths,
            "meets_floor": meets,
        }
    result["passed"] = passed
    result["replicates"] = runs
    return result


def gate_a1c(runs: int, base: int = 0) -> dict[str, Any]:
    """Size and power of the calibration check.

    Args:
        runs: Number of worlds per skew.
        base: Seed offset (0 for certification).

    Returns:
        The false-alarm rate on calibrated scores with its interval and
        bound, the power at the two distorted skews and `passed`.
    """
    level = 0.01
    rejects = {1.0: 0, 0.6: 0, 1.6: 0}
    for index in range(runs):
        for skew in rejects:
            rng = rng_for(base, f"a1c.{skew}", index)
            score, _, y = f11_generators.calibration_world(rng, 5000, skew)
            check = threshold.calibration_check(score, y, level)
            rejects[skew] += int(not check.calibrated)
    size = rejects[1.0] / runs
    power = {str(k): v / runs for k, v in rejects.items() if k != 1.0}
    meets_size = size <= level + tol(level, runs)
    meets_power = all(v >= 0.8 for v in power.values())
    return {
        "level": level,
        "false_alarm_rate": size,
        "false_alarm_interval": wilson(rejects[1.0], runs),
        "size_bound": level + tol(level, runs),
        "power": power,
        "passed": bool(meets_size and meets_power),
        "replicates": runs,
    }


def gate_a2(runs: int, base: int = 0) -> dict[str, Any]:
    """The identity to 1e-12 and the binormal closed form.

    Args:
        runs: Number of binormal worlds.
        base: Seed offset (0 for certification).

    Returns:
        The largest identity error over 1000 random tables, the Monte Carlo
        mean net benefit, the closed-form truth, the standard error and
        `passed`.
    """
    worst = 0.0
    rng = rng_for(base, "a2.tables", 0)
    for _ in range(1000):
        n = 300
        y = (rng.random(n) < rng.uniform(0.1, 0.6)).astype(float)
        a = (rng.random(n) < 0.4).astype(float)
        fp, fn = rng.uniform(1, 9), rng.uniform(1, 90)
        human = (costs.CostSource.HUMAN,) * 4
        model = costs.DecisionCostModel(
            0.0, fn, fp, 0.0, 0.001, 0.999, human, costs.CostSource.HUMAN
        )
        t = fp / (fp + fn)
        lhs = costs.net_benefit(y, a, t)
        rhs = (fn * y.mean() - costs.expected_cost(y, a, model)) / fn
        worst = max(worst, abs(lhs - rhs))
    cut, prevalence, delta, t = 0.8, 0.2, 1.5, 0.2
    truth = float(
        f11_generators.binormal_nb(prevalence, delta, cut, numpy.array([t]))[0]
    )
    values = []
    for index in range(runs):
        score, y, _ = f11_generators.binormal(
            rng_for(base, "a2.world", index), 20000, prevalence, delta
        )
        values.append(costs.net_benefit(y, (score >= cut).astype(float), t))
    arr = numpy.array(values)
    se = float(arr.std(ddof=1) / math.sqrt(runs))
    inside = abs(float(arr.mean()) - truth) <= 3 * se
    return {
        "max_identity_error": worst,
        "monte_carlo_mean": float(arr.mean()),
        "truth": truth,
        "se": se,
        "passed": bool(worst <= 1e-12 and inside),
        "replicates": runs,
    }


# --- band coverage, false usefulness -----------------------------------------

RANGE = (0.1, 0.4)
CUT = 0.8
PREVALENCE = 0.2
DELTA = 1.5
BOOTSTRAPS = 400


def band_setting(runs: int, n: int, cluster: bool, base: int) -> dict[str, Any]:
    """Coverage of the true differences by the lower band.

    Args:
        runs: Number of worlds.
        n: Rows per world.
        cluster: Whether rows come in clusters of 10 with a shared shift.
        base: Seed offset.

    Returns:
        The number and share of worlds whose lower band lies below the true
        differences at every grid point, the Wilson interval and whether its
        lower bound is at least 0.93.
    """
    size = 10 if cluster else 1
    shared = 0.5 if cluster else 0.0
    sd = math.sqrt(1.0 + shared**2)
    grid = numpy.linspace(RANGE[0], RANGE[1], netbenefit.GRID_POINTS)
    truth = f11_generators.binormal_differences(
        PREVALENCE, DELTA, CUT, grid, sd
    )
    hits = 0
    for index in range(runs):
        score, y, groups = f11_generators.binormal(
            rng_for(base, f"a3.{n}.{cluster}", index),
            n,
            PREVALENCE,
            DELTA,
            size,
            shared,
        )
        report = netbenefit.net_benefit_band(
            y,
            (score >= CUT).astype(float),
            RANGE,
            clusters=groups if cluster else None,
            replicates=BOOTSTRAPS,
            seed=base + index,
        )
        lower = numpy.array(report.lower)
        hits += int((lower <= truth + 1e-15).all())
    return {
        "rows": n,
        "clustered": cluster,
        "covered": hits,
        "runs": runs,
        "coverage": hits / runs,
        "interval": wilson(hits, runs),
        "meets_floor": wilson(hits, runs)[0] >= 0.93,
    }


def gate_a3(runs: int, base: int = 0) -> dict[str, Any]:
    """Simultaneous band coverage in four settings.

    Args:
        runs: Number of worlds per setting.
        base: Seed offset (0 for certification).

    Returns:
        The `band_setting` result for each setting, the bootstrap count and
        `passed`.
    """
    settings = {
        f"{n}/{'clustered' if c else 'independent'}": band_setting(
            runs, n, c, base
        )
        for n in (2000, 20000)
        for c in (False, True)
    }
    return {
        "settings": settings,
        "bootstraps": BOOTSTRAPS,
        "passed": all(v["meets_floor"] for v in settings.values()),
        "replicates": runs,
    }


def gate_a4(runs: int, base: int = 0) -> dict[str, Any]:
    """False usefulness with a score independent of the label.

    Args:
        runs: Number of worlds per size.
        base: Seed offset (0 for certification).

    Returns:
        The share called `useful` at each size with its interval, the bound,
        the power at four effect sizes (300 worlds each) and `passed`.
    """
    result: dict[str, Any] = {"null": {}}
    bound = 0.05 + tol(0.05, runs)
    passed = True
    for n in (2000, 20000):
        hits = 0
        for index in range(runs):
            rng = rng_for(base, f"a4.{n}", index)
            y = (rng.random(n) < PREVALENCE).astype(float)
            acts = (rng.random(n) < PREVALENCE).astype(float)
            report = netbenefit.net_benefit_band(
                y,
                acts,
                RANGE,
                replicates=BOOTSTRAPS,
                seed=base + index,
            )
            hits += int(report.claim == netbenefit.USEFUL)
        meets = hits / runs <= bound
        passed = passed and meets
        result["null"][str(n)] = {
            "useful_rate": hits / runs,
            "interval": wilson(hits, runs),
            "meets_floor": meets,
        }
    power = {}
    for delta in (0.25, 0.5, 1.0, 1.5):
        hits = 0
        for index in range(300):
            score, y, _ = f11_generators.binormal(
                rng_for(base, f"a4.power.{delta}", index),
                2000,
                PREVALENCE,
                delta,
            )
            report = netbenefit.net_benefit_band(
                y,
                (score >= CUT).astype(float),
                RANGE,
                replicates=BOOTSTRAPS,
                seed=base + index,
            )
            hits += int(report.claim == netbenefit.USEFUL)
        power[str(delta)] = hits / 300
    result.update(
        {
            "bound": bound,
            "power_n2000": power,
            "passed": passed,
            "replicates": runs,
        }
    )
    return result


# --- off-policy evaluation ---------------------------------------------------

BANDIT_ROWS = 2000
DEV_ROWS = 3000
CLIP = 10.0


def outcome_models(base: int) -> dict[str, Any]:
    """Outcome models fitted on a separate log (with and without the step).

    Args:
        base: Seed offset.

    Returns:
        Two outcome-model functions: `dr`, which sees the step feature, and
        `dr_misspec`, which does not.
    """
    dev = f11_generators.bandit(rng_for(base, "a5.dev", 0), DEV_ROWS, 1.0)
    full = ope.fit_outcome_model(
        f11_generators.outcome_features(dev["x"]),
        dev["action"],
        dev["reward"],
        2,
    )
    plain = ope.fit_outcome_model(dev["x"], dev["action"], dev["reward"], 2)
    return {
        "dr": lambda x: full(f11_generators.outcome_features(x)),
        "dr_misspec": plain,
    }


def gate_a5(runs: int, base: int = 0) -> dict[str, Any]:
    """Coverage of the estimators and the support and ESS gates.

    Args:
        runs: Number of logs.
        base: Seed offset (0 for certification).

    Returns:
        The true value, per estimator the bias, mean effective-sample-size
        ratio and coverage among replicates with ratio at least 0.2, whether
        the doubly robust floor is met, the refusal rows and `passed`.
    """
    models = outcome_models(base)
    truth = f11_generators.target_value()
    cells = {
        "ipw": ("ipw", None, None),
        "snips": ("snips", None, None),
        "ipw_clipped": ("ipw", CLIP, None),
        "dr": ("dr", None, "dr"),
        "dr_clipped": ("dr", CLIP, "dr"),
        "dr_misspec": ("dr", None, "dr_misspec"),
    }
    tally: dict[str, dict[str, Any]] = {
        k: {"hits": 0, "values": [], "ess": [], "used": 0} for k in cells
    }
    for index in range(runs):
        data = f11_generators.bandit(
            rng_for(base, "a5.log", index), BANDIT_ROWS, 1.0
        )
        target = f11_generators.target_policy(data["x"])
        for name, (estimator, clip, model_name) in cells.items():
            q = None if model_name is None else models[model_name](data["x"])
            log = ope.LoggedBandit(
                data["action"],
                data["reward"],
                data["behaviour"],
                target,
                q,
            )
            estimate = ope.estimate_value(log, estimator, clip=clip)
            tally[name]["ess"].append(estimate.ess_ratio)
            if estimate.status == ope.REFUSED:
                continue
            low, high = estimate.interval or (math.nan, math.nan)
            tally[name]["values"].append(estimate.value)
            if estimate.ess_ratio >= 0.2:
                tally[name]["used"] += 1
                tally[name]["hits"] += int(low <= truth <= high)
    table = {}
    for name, cell in tally.items():
        values = numpy.array(cell["values"])
        used = cell["used"]
        table[name] = {
            "bias": float(values.mean() - truth),
            "mean_ess_ratio": float(numpy.mean(cell["ess"])),
            "coverage": cell["hits"] / used if used else None,
            "interval": wilson(cell["hits"], used) if used else None,
            "replicates_used": used,
        }
    dr_ok = all(
        table[name]["interval"] is not None
        and table[name]["interval"][0] >= 0.93
        for name in ("dr", "dr_misspec")
    )
    refusals = refusal_rows(base)
    return {
        "truth": truth,
        "estimators": table,
        "dr_floor_met": dr_ok,
        "refusal": refusals,
        "passed": bool(dr_ok and refusals["meets_floor"]),
        "replicates": runs,
    }


def refusal_rows(base: int) -> dict[str, Any]:
    """The support gate on planted violations and clean worlds.

    Args:
        base: Seed offset.

    Returns:
        The number of refusals (with intervals) in worlds with a hard
        violation, a weak violation and none, the runs per kind and whether
        the refusal floors are met.
    """
    runs = 300
    fired = {"hard": 0, "weak": 0, "clean": 0}
    for index in range(runs):
        for kind in fired:
            rng = rng_for(base, f"a5.refuse.{kind}", index)
            if kind == "clean":
                data = f11_generators.bandit(rng, BANDIT_ROWS, 1.0)
            elif kind == "hard":
                data = f11_generators.bandit(
                    rng, BANDIT_ROWS, 1.0, blind_above=1.0
                )
            else:
                data = f11_generators.bandit(rng, BANDIT_ROWS, 1.0)
                tail = data["x"][:, 0] > 1.5
                data["behaviour"][tail] = numpy.array([0.995, 0.005])
            target = f11_generators.target_policy(data["x"])
            log = ope.LoggedBandit(
                data["action"], data["reward"], data["behaviour"], target
            )
            estimate = ope.estimate_value(log, "ipw")
            fired[kind] += int(estimate.status == ope.REFUSED)
    hard = wilson(fired["hard"], runs)
    weak = wilson(fired["weak"], runs)
    meets = (
        fired["hard"] / runs >= 0.95
        and hard[0] >= 0.90
        and fired["weak"] / runs >= 0.95
        and weak[0] >= 0.90
        and fired["clean"] / runs <= 0.05
    )
    return {
        "hard_violation": {"refused": fired["hard"], "interval": hard},
        "weak_violation": {"refused": fired["weak"], "interval": weak},
        "clean": {
            "refused": fired["clean"],
            "interval": wilson(fired["clean"], runs),
        },
        "runs_each": runs,
        "meets_floor": bool(meets),
    }


# --- Hillstrom: biased subsamples, A/A, positive control ---------------------


def hillstrom_path() -> Path | None:
    """The downloaded data file, or None.

    Returns:
        The path named by the `SCMLE_HILLSTROM` environment variable if it
        exists, otherwise None.
    """
    import os

    given = os.environ.get(HILLSTROM_ENV)
    if given and Path(given).exists():
        return Path(given)
    return None


def load_hillstrom() -> pandas.DataFrame | None:
    """The data frame when the file is present and its hash matches.

    Returns:
        The table, or None when the file is missing or its SHA-256 differs
        from `HILLSTROM_SHA256`.
    """
    path = hillstrom_path()
    if path is None:
        return None
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != HILLSTROM_SHA256:
        return None
    return pandas.read_csv(path)


def not_run(reason: str) -> dict[str, Any]:
    """A gate that could not run; it is not passed.

    Args:
        reason: Why the gate could not run.

    Returns:
        A result with `passed` false and the reason under `not_run`.
    """
    return {"passed": False, "not_run": reason}


def covariates(frame: pandas.DataFrame) -> numpy.ndarray:
    """Numeric covariates of the Hillstrom rows.

    Args:
        frame: The Hillstrom table.

    Returns:
        The standardised columns recency, history, newbie, mens and womens.
    """
    columns = ["recency", "history", "newbie", "mens", "womens"]
    values = frame[columns].to_numpy(dtype=float)
    return (values - values.mean(axis=0)) / values.std(axis=0)


def gate_a6(runs: int, base: int = 0) -> dict[str, Any]:
    """Biased subsamples of the randomized Hillstrom data.

    Args:
        runs: Number of subsamples.
        base: Seed offset (0 for certification).

    Returns:
        The full-data reference value and interval, the policy cut, the
        treated share, the coverage of the doubly robust interval with its
        Wilson interval, the mean subsample size, the data hash, the number
        of subsamples and `passed` (coverage at least 0.90); or a not-run
        result when the data file is absent.
    """
    frame = load_hillstrom()
    if frame is None:
        return not_run(f"set {HILLSTROM_ENV} to the verified data file")
    two = frame[frame["segment"].isin(["Mens E-Mail", "No E-Mail"])]
    two = two.reset_index(drop=True)
    arm = numpy.asarray(two["segment"] == "Mens E-Mail", dtype=numpy.intp)
    y = two["conversion"].to_numpy(dtype=float)
    x = covariates(two)
    order = rng_for(base, "a6.split", 0).permutation(len(two))
    dev, evaluation = order[: len(two) // 2], order[len(two) // 2 :]
    predict = ope.fit_outcome_model(x[dev], arm[dev], y[dev], 2, ridge=10.0)
    tau_dev = predict(x[dev])[:, 1] - predict(x[dev])[:, 0]
    dev_log = ope.LoggedBandit(
        arm[dev],
        y[dev],
        numpy.full((len(dev), 2), 0.5),
        numpy.full((len(dev), 2), 0.5),
        predict(x[dev]),
    )
    policy = ope.fit_threshold_policy(dev_log, tau_dev)
    x_e, a_e, y_e = x[evaluation], arm[evaluation], y[evaluation]
    tau_e = predict(x_e)[:, 1] - predict(x_e)[:, 0]
    pi_e = policy.probabilities(tau_e)
    full_log = ope.LoggedBandit(
        a_e, y_e, numpy.full((len(a_e), 2), 0.5), pi_e, predict(x_e)
    )
    reference = ope.estimate_value(full_log, "dr")
    assert reference.value is not None
    treat_p = numpy.clip(1 / (1 + numpy.exp(-x_e[:, 0])), 0.1, 0.9)
    behaviour = numpy.column_stack([1 - treat_p, treat_p])
    logged = behaviour[numpy.arange(len(a_e)), a_e]
    hits = 0
    kept_rows = []
    for index in range(runs):
        keep = rng_for(base, "a6.keep", index).random(len(a_e)) < logged / 0.9
        log = ope.LoggedBandit(
            a_e[keep],
            y_e[keep],
            behaviour[keep],
            pi_e[keep],
            predict(x_e[keep]),
        )
        estimate = ope.estimate_value(log, "dr")
        low, high = estimate.interval or (math.nan, math.nan)
        hits += int(low <= reference.value <= high)
        kept_rows.append(int(keep.sum()))
    coverage = hits / runs
    return {
        "reference_value": reference.value,
        "reference_interval": list(reference.interval or ()),
        "policy_cut": policy.cut,
        "treated_share": float(pi_e[:, 1].mean()),
        "coverage": coverage,
        "interval": wilson(hits, runs),
        "mean_kept_rows": float(numpy.mean(kept_rows)),
        "data_sha256": HILLSTROM_SHA256,
        "passed": bool(coverage >= 0.90),
        "replicates": runs,
    }


def peeking_paths(
    arm: numpy.ndarray,
    outcome: numpy.ndarray,
    q: float,
    delta_min: float,
) -> numpy.ndarray:
    """Log e-process paths of the primary metric for a batch of runs.

    Args:
        arm: Treatment indicators, shape (runs, units).
        outcome: Outcomes, same shape.
        q: Assignment probability of the treated arm.
        delta_min: Smallest effect of interest.

    Returns:
        The log wealth after each unit, same shape.
    """
    return anytime.centered_paths(arm, outcome, q, delta_min)


def gate_b7(runs: int, base: int = 0) -> dict[str, Any]:
    """A/A false promotion on the Hillstrom outcomes.

    Args:
        runs: Number of A/A streams.
        base: Seed offset (0 for certification).

    Returns:
        The false promotions, their rate, interval and bound, the design,
        the data hash and `passed`; or a not-run result when the data file is
        absent.
    """
    frame = load_hillstrom()
    if frame is None:
        return not_run(f"set {HILLSTROM_ENV} to the verified data file")
    y_all = frame["conversion"].to_numpy(dtype=float)
    n_max, delta_min = 20000, 0.001
    hits = 0
    chunk = 100
    for start in range(0, runs, chunk):
        size = min(chunk, runs - start)
        rng = rng_for(base, "b7", start)
        y = numpy.stack([rng.permutation(y_all)[:n_max] for _ in range(size)])
        arm = (rng.random((size, n_max)) < 0.5).astype(float)
        paths = peeking_paths(arm, y, 0.5, delta_min)
        hits += int((paths.max(axis=1) >= -math.log(ALPHA_PROMOTE)).sum())
    rate = hits / runs
    return {
        "false_promotions": hits,
        "rate": rate,
        "interval": wilson(hits, runs),
        "bound": ALPHA_PROMOTE + tol(ALPHA_PROMOTE, runs),
        "delta_min": delta_min,
        "n_max": n_max,
        "data_sha256": HILLSTROM_SHA256,
        "passed": bool(rate <= ALPHA_PROMOTE + tol(ALPHA_PROMOTE, runs)),
        "replicates": runs,
    }


def gate_b8(runs: int, base: int = 0) -> dict[str, Any]:
    """Real arms: promotion and stopping time against a fixed-sample test.

    Args:
        runs: Number of stream orders.
        base: Seed offset (0 for certification).

    Returns:
        The fixed-sample reference, the assignment share, the share of
        orders promoted, the median stop, the full-sample lower bound, the
        row count, the data hash and `passed`; or a not-run result when the
        data file is absent.
    """
    frame = load_hillstrom()
    if frame is None:
        return not_run(f"set {HILLSTROM_ENV} to the verified data file")
    two = frame[frame["segment"].isin(["Mens E-Mail", "No E-Mail"])]
    arm_all = (two["segment"] == "Mens E-Mail").to_numpy(dtype=float)
    y_all = two["conversion"].to_numpy(dtype=float)
    share = float(arm_all.mean())
    p1, p0 = y_all[arm_all == 1].mean(), y_all[arm_all == 0].mean()
    n1, n0 = int((arm_all == 1).sum()), int((arm_all == 0).sum())
    se = math.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0)
    delta_min = 0.002
    reference = {
        "difference": float(p1 - p0),
        "z_against_delta_min": float((p1 - p0 - delta_min) / se),
        "p_value_one_sided": float(stats.norm.sf((p1 - p0 - delta_min) / se)),
        "interval_95": [
            float(p1 - p0 - 1.96 * se),
            float(p1 - p0 + 1.96 * se),
        ],
    }
    stops = []
    for index in range(runs):
        order = rng_for(base, "b8", index).permutation(len(y_all))
        arm = arm_all[order][None, :]
        y = y_all[order][None, :]
        paths = peeking_paths(arm, y, share, delta_min)
        hit = numpy.nonzero(paths[0] >= -math.log(ALPHA_PROMOTE))[0]
        stops.append(int(hit[0]) + 1 if hit.size else None)
    promoted = [s for s in stops if s is not None]
    direction = (p1 - p0) > 0 and bool(promoted)
    z_stream = anytime.pseudo_outcomes(
        arm_all, y_all, share, anytime.running_center(y_all)
    )
    stream = anytime.centered_stream(share, delta_min)
    bound = anytime.confidence_bound(z_stream, stream, ALPHA_PROMOTE)
    share_promoted = len(promoted) / runs
    return {
        "reference_fixed_sample": reference,
        "assignment_share": share,
        "delta_min": delta_min,
        "promoted_share": share_promoted,
        "median_stop": float(numpy.median(promoted)) if promoted else None,
        "full_sample_lower_bound": bound + 0.0,
        "rows": len(y_all),
        "data_sha256": HILLSTROM_SHA256,
        "passed": bool(share_promoted >= 0.9 and direction),
        "replicates": runs,
    }


# --- fairness: verdicts, group thresholds, infeasible sets -------------------


def fair_constraint(
    gap: float = 0.05,
) -> scmle.data.charter.decision.FairnessConstraint:
    """The constraint of the fairness worlds.

    Args:
        gap: Largest allowed true-positive-rate gap.

    Returns:
        An equal-opportunity constraint on `group` with that gap.
    """
    return scmle.data.charter.decision.FairnessConstraint(
        "group", "equal_opportunity", gap, "group_thresholds", "certify"
    )


def gate_a7(runs: int, base: int = 0) -> dict[str, Any]:
    """Verdicts and gap bounds with planted true-positive-rate gaps.

    Args:
        runs: Number of worlds per gap.
        base: Seed offset (0 for certification).

    Returns:
        Per gap the share declared satisfied and the coverage of the gap's
        upper bound with their intervals, whether the floors are met, and
        `passed`.
    """
    rule = fair_constraint()
    table = {}
    passed = True
    for gap in (0.0, 0.03, 0.07, 0.10, 0.15):
        satisfied = 0
        covered = 0
        for index in range(runs):
            score, y, groups = f11_generators.fairness_world(
                rng_for(base, f"a7.{gap}", index), 20000, gap
            )
            test = constraints.satisfaction_test(
                y, (score >= 0.0).astype(float), groups, rule
            )
            satisfied += int(test.status == constraints.SATISFIED)
            covered += int(
                test.upper_bound is not None and test.upper_bound >= gap
            )
        cover = wilson(covered, runs)
        row: dict[str, Any] = {
            "satisfied_rate": satisfied / runs,
            "satisfied_interval": wilson(satisfied, runs),
            "bound_covers": covered / runs,
            "bound_cover_interval": cover,
        }
        meets = cover[0] >= 0.93
        if gap > 0.05:
            wrong = wilson(satisfied, runs)
            meets = meets and wrong[1] <= 0.08 and satisfied / runs <= 0.05
        row["meets_floor"] = bool(meets)
        passed = passed and meets
        table[str(gap)] = row
    return {"gaps": table, "passed": bool(passed), "replicates": runs}


def gate_a7b(runs: int, base: int = 0) -> dict[str, Any]:
    """Fitted group cuts against the equal-rate oracle.

    Args:
        runs: Number of worlds.
        base: Seed offset (0 for certification).

    Returns:
        The oracle net benefit, the distribution of the relative loss of the
        fitted cuts and `passed` (worst loss at most 2 percent).
    """
    rule = fair_constraint()
    t, shift = 0.3, 1.0
    oracle = f11_generators.threshold_oracle(shift, t)
    w = t / (1.0 - t)
    losses = []
    for index in range(runs):
        score, y, groups = f11_generators.threshold_world(
            rng_for(base, "a7b", index), 100_000, shift
        )
        # The fit reads probabilities; a monotone map of the score keeps the
        # equal-true-positive-rate oracle unchanged.
        risk = 1.0 / (1.0 + numpy.exp(-score))
        fitted = constraints.fit_group_thresholds(
            risk, y, groups, rule, t, margin=0.5
        )
        value = 0.0
        for g in (0, 1):
            cut = float(
                numpy.log(fitted.cuts[str(g)] / (1 - fitted.cuts[str(g)]))
            )
            tpr = 1.0 - f11_generators.Z.cdf(cut - (1.5 - shift * g))
            fpr = 1.0 - f11_generators.Z.cdf(cut + 0.5)
            value += 0.5 * (0.3 * tpr - 0.7 * w * fpr)
        losses.append((oracle - value) / abs(oracle))
    worst = float(max(losses))
    return {
        "oracle_net_benefit": oracle,
        "relative_loss": summary(numpy.array(losses)),
        "passed": bool(worst <= 0.02),
        "replicates": runs,
    }


def gate_a8(runs: int, base: int = 0) -> dict[str, Any]:
    """Infeasible criteria refused when base rates differ, not when equal.

    Args:
        runs: Number of worlds per criterion pair.
        base: Seed offset (0 for certification).

    Returns:
        The refusals for each unequal-base-rate pair, the false refusals and
        rate with their bound for equal base rates, and `passed`.
    """
    pairs = (
        ("demographic_parity", "equalized_odds"),
        ("demographic_parity", "predictive_parity"),
        ("equalized_odds", "predictive_parity"),
    )
    refused = {}
    for pair in pairs:
        declared = tuple(
            scmle.data.charter.decision.FairnessConstraint(
                "group", c, 0.05, "reweighing"
            )
            for c in pair
        )
        count = 0
        for index in range(runs):
            y, groups = f11_generators.base_rate_world(
                rng_for(base, f"a8.{pair[0]}.{pair[1]}", index),
                4000,
                (0.2, 0.5),
            )
            try:
                constraints.check_feasibility(declared, y, groups)
            except constraints.InfeasibleConstraintsError:
                count += 1
        refused["/".join(pair)] = count
    declared = tuple(
        scmle.data.charter.decision.FairnessConstraint(
            "group", c, 0.05, "reweighing"
        )
        for c in pairs[0]
    )
    false_refusals = 0
    equal_runs = 3 * runs
    for index in range(equal_runs):
        y, groups = f11_generators.base_rate_world(
            rng_for(base, "a8.equal", index), 4000, (0.3, 0.3)
        )
        try:
            constraints.check_feasibility(declared, y, groups)
        except constraints.InfeasibleConstraintsError:
            false_refusals += 1
    bound = 0.05 + tol(0.05, equal_runs)
    return {
        "refused_unequal": refused,
        "runs_each": runs,
        "false_refusals_equal": false_refusals,
        "false_refusal_rate": false_refusals / equal_runs,
        "false_refusal_bound": bound,
        "passed": bool(
            all(v == runs for v in refused.values())
            and false_refusals / equal_runs <= bound
        ),
    }


# --- online: stopping, guardrail, cost, rollback, budget, sample ratio -------


def plan_for(
    delta_min: float = 0.02,
    n_max: int = 3000,
    guardrails: tuple[scmle.data.charter.decision.Guardrail, ...] = (),
    tolerance: float = 0.02,
) -> scmle.data.charter.decision.RolloutPlan:
    """The rollout plan the online worlds use.

    Args:
        delta_min: Smallest effect worth promoting.
        n_max: Maximum number of units.
        guardrails: Guardrail metrics.
        tolerance: Loss that triggers a rollback.

    Returns:
        A plan for a conversion experiment with assignment probability 0.5
        and the module's promotion and rollback levels.
    """
    return scmle.data.charter.decision.RolloutPlan(
        unit="user",
        primary="conversion",
        assignment_prob=0.5,
        delta_min=delta_min,
        tolerance=tolerance,
        n_max=n_max,
        alpha_promote=scmle.data.ledger.budget.exact(ALPHA_PROMOTE, "a"),
        alpha_rollback=scmle.data.ledger.budget.exact(ALPHA_ROLLBACK, "a"),
        guardrails=guardrails,
    )


def compare_arms_rate(base: int) -> dict[str, Any]:
    """Realized false-adopt rate of the bootstrap rule under 6 looks.

    Args:
        base: Seed offset.

    Returns:
        The runs, looks, false adoptions, rate and Wilson interval, and a
        note on the null being tested.
    """
    runs, looks, n = 100, 6, 3000
    classes = numpy.array([0, 1])
    adopted = 0
    for index in range(runs):
        rng = rng_for(base, "b1.compare", index)
        arm = rng.random(n) < 0.5
        y = (rng.random(n) < 0.10).astype(int)
        for look in range(1, looks + 1):
            m = look * (n // looks)
            seen_arm, seen_y = arm[:m], y[:m]
            control = abtest.Arm(
                numpy.ones((~seen_arm).sum(), dtype=int),
                seen_y[~seen_arm],
                classes=classes,
            )
            treat = abtest.Arm(
                numpy.ones(seen_arm.sum(), dtype=int),
                seen_y[seen_arm],
                classes=classes,
            )
            result = abtest.compare_arms(
                tasks.TaskType.BINARY,
                "accuracy",
                control,
                treat,
                seed=base + index,
            )
            if result.decision == "adopt":
                adopted += 1
                break
    return {
        "runs": runs,
        "looks": looks,
        "false_adopt": adopted,
        "rate": adopted / runs,
        "interval": wilson(adopted, runs),
        "null": "Delta = 0 for a rule that claims to test Delta <= 0",
    }


def class_agreement(base: int) -> dict[str, Any]:
    """The Rollout class and the vectorised paths decide identically.

    Args:
        base: Seed offset.

    Returns:
        The number of runs and the number in which the class and the
        vectorised computation promote at the same unit (or both do not).
    """
    plan = plan_for()
    same = 0
    runs = 100
    for index in range(runs):
        rng = rng_for(base, "b1.class", index)
        arm, y = f11_generators.experiment(rng, 1, plan.n_max, 0.5, 0.10, 0.04)
        paths = peeking_paths(arm, y, 0.5, plan.delta_min)
        vector_stop = numpy.nonzero(paths[0] >= -math.log(ALPHA_PROMOTE))[0]
        run = rollout.Rollout(plan, f"e{index}", ALPHA_PROMOTE)
        decision = run.observe(arm[0], y[0])
        vector = int(vector_stop[0]) + 1 if vector_stop.size else None
        mine = decision.units if decision.status == rollout.PROMOTE else None
        same += int(vector == mine)
    return {"runs": runs, "agree": same}


def gate_b1(runs: int, base: int = 0) -> dict[str, Any]:
    """Ever-promote rate at the boundary null under peeking at every unit.

    Args:
        runs: Number of streams.
        base: Seed offset (0 for certification).

    Returns:
        The ever-promote count, rate, interval and bound, the class
        agreement, the bootstrap-rule comparison (only for 1000 or more
        runs) and `passed`.
    """
    plan = plan_for()
    hits = 0
    chunk = 500
    for start in range(0, runs, chunk):
        size = min(chunk, runs - start)
        arm, y = f11_generators.experiment(
            rng_for(base, "b1", start), size, plan.n_max, 0.5, 0.10, 0.02
        )
        paths = peeking_paths(arm, y, 0.5, plan.delta_min)
        hits += int((paths.max(axis=1) >= -math.log(ALPHA_PROMOTE)).sum())
    bound = ALPHA_PROMOTE + tol(ALPHA_PROMOTE, runs)
    agreement = class_agreement(base)
    compare = compare_arms_rate(base) if runs >= 1000 else None
    return {
        "ever_promote": hits,
        "rate": hits / runs,
        "interval": wilson(hits, runs),
        "bound": bound,
        "class_agreement": agreement,
        "compare_arms": compare,
        "passed": bool(
            hits / runs <= bound and agreement["agree"] == agreement["runs"]
        ),
        "replicates": runs,
    }


def gate_b2(runs: int, base: int = 0) -> dict[str, Any]:
    """Promotion when only a guardrail null is true.

    Args:
        runs: Number of streams.
        base: Seed offset (0 for certification).

    Returns:
        The promotions (primary and guardrail both crossing), rate,
        interval, bound and `passed`.
    """
    guard = scmle.data.charter.decision.Guardrail("bounce", margin=0.05)
    plan = plan_for(n_max=6000, guardrails=(guard,))
    hits = 0
    chunk = 250
    for start in range(0, runs, chunk):
        size = min(chunk, runs - start)
        rng = rng_for(base, "b2", start)
        arm = (rng.random((size, plan.n_max)) < 0.5).astype(float)
        primary = (rng.random(arm.shape) < 0.10 + 0.06 * arm).astype(float)
        bounce = (rng.random(arm.shape) < 0.20 + 0.05 * arm).astype(float)
        target = -math.log(ALPHA_PROMOTE)
        both = [
            anytime.centered_paths(arm, primary, 0.5, plan.delta_min) >= target,
            anytime.centered_paths(arm, bounce, 0.5, guard.margin, "less")
            >= target,
        ]
        hits += int((both[0] & both[1]).any(axis=1).sum())
    bound = ALPHA_PROMOTE + tol(ALPHA_PROMOTE, runs)
    return {
        "promotions": hits,
        "rate": hits / runs,
        "interval": wilson(hits, runs),
        "bound": bound,
        "passed": bool(hits / runs <= bound),
        "replicates": runs,
    }


def fixed_horizon_normal(
    p_control: float, delta: float, delta_min: float, alpha: float
) -> int:
    """Total units of a fixed-horizon test of `Delta <= delta_min` at power 0.8.

    Normal approximation with unpooled variances; `alpha` one-sided.

    Args:
        p_control: Control rate.
        delta: True effect.
        delta_min: Effect under the null being tested.
        alpha: One-sided level.

    Returns:
        The total number of units over both arms.
    """
    p1 = p_control + delta
    v0 = p_control * (1 - p_control) + p1 * (1 - p1)
    v1 = v0
    z_a, z_b = stats.norm.ppf(1 - alpha), stats.norm.ppf(0.8)
    per_arm = (z_a * math.sqrt(v0) + z_b * math.sqrt(v1)) ** 2 / (
        delta - delta_min
    ) ** 2
    return math.ceil(2 * per_arm)


def esn_metric(kind: str, runs: int, base: int) -> dict[str, Any]:
    """Expected sample number for one metric type.

    Args:
        kind: `binary` or any other value for a bounded metric.
        runs: Number of streams.
        base: Seed offset.

    Returns:
        The share of streams that cross, the mean and median units to
        promote, the fixed-horizon sizes and the ratios to them.
    """
    plan = plan_for(delta_min=0.02, n_max=20000)
    stops = []
    chunk = 100
    for start in range(0, runs, chunk):
        size = min(chunk, runs - start)
        rng = rng_for(base, f"b3.{kind}", start)
        arm = (rng.random((size, plan.n_max)) < 0.5).astype(float)
        if kind == "binary":
            y = (rng.random(arm.shape) < 0.10 + 0.04 * arm).astype(float)
        else:
            y = numpy.where(
                arm == 1,
                rng.beta(2.4, 7.6, arm.shape),
                rng.beta(2.0, 8.0, arm.shape),
            )
        paths = peeking_paths(arm, y, 0.5, plan.delta_min)
        hit = paths >= -math.log(ALPHA_PROMOTE)
        crossed = hit.any(axis=1)
        when = numpy.where(crossed, hit.argmax(axis=1) + 1, plan.n_max)
        stops.extend(when[crossed].tolist() + [None] * int((~crossed).sum()))
    done = [s for s in stops if s is not None]
    if kind == "binary":
        fixed_zero = 2 * feasibility.n_required_two_proportions(
            0.10, 0.14, alpha=2 * ALPHA_PROMOTE, k_max=1, power=0.8
        )
        fixed_null = fixed_horizon_normal(0.10, 0.04, 0.02, ALPHA_PROMOTE)
    else:
        sd = math.sqrt(2 * (0.0145 + 0.0140) / 2 * 1.0)
        per_arm = (
            (stats.norm.ppf(1 - ALPHA_PROMOTE) + stats.norm.ppf(0.8))
            * sd
            / 0.02
        ) ** 2
        fixed_zero = None
        fixed_null = math.ceil(2 * per_arm)
    mean_stop = float(numpy.mean(done)) if done else None
    return {
        "crossed_share": len(done) / runs,
        "mean_units": mean_stop,
        "median_units": float(numpy.median(done)) if done else None,
        "fixed_horizon_zero_null": fixed_zero,
        "fixed_horizon_null_shifted": fixed_null,
        "ratio_to_null_shifted": (
            mean_stop / fixed_null if mean_stop is not None else None
        ),
        "ratio_to_zero_null": (
            mean_stop / fixed_zero
            if mean_stop is not None and fixed_zero
            else None
        ),
    }


def gate_b3(runs: int, base: int = 0) -> dict[str, Any]:
    """Expected sample number of the anytime design (reported, no floor).

    Args:
        runs: Number of streams per metric.
        base: Seed offset (0 for certification).

    Returns:
        The results for the binary and the bounded metric, a note that no
        floor applies and `passed` (always true).
    """
    return {
        "binary": esn_metric("binary", runs, base),
        "bounded_mean": esn_metric("bounded", runs, base),
        "floor": "none: reported",
        "passed": True,
        "replicates": runs,
    }


def rollback_paths(
    arm: numpy.ndarray, y: numpy.ndarray, tolerance: float
) -> numpy.ndarray:
    """Log e-process paths of the rollback test for a batch of runs.

    Args:
        arm: Treatment indicators, shape (runs, units).
        y: Outcomes, same shape.
        tolerance: Loss at which the champion should be rolled back to.

    Returns:
        The log wealth after each unit, same shape.
    """
    return anytime.centered_paths(arm, y, 0.5, -tolerance, "less")


def gate_b4(runs: int, base: int = 0) -> dict[str, Any]:
    """False rollback at the boundary and detection delay.

    Args:
        runs: Number of streams at the boundary.
        base: Seed offset (0 for certification).

    Returns:
        The false rollbacks, rate, interval and bound, the detected share
        and the median and 90th percentile delay for true losses of 2 and 4
        tolerances, and `passed`.
    """
    tau, target = 0.02, -math.log(ALPHA_ROLLBACK)
    hits = 0
    chunk = 500
    for start in range(0, runs, chunk):
        size = min(chunk, runs - start)
        arm, y = f11_generators.experiment(
            rng_for(base, "b4", start), size, 4000, 0.5, 0.20, -tau
        )
        hits += int((rollback_paths(arm, y, tau).max(axis=1) >= target).sum())
    delays = {}
    for factor in (2, 4):
        stops = []
        for start in range(0, 1000, 250):
            arm, y = f11_generators.experiment(
                rng_for(base, f"b4.delay.{factor}", start),
                250,
                20000,
                0.5,
                0.20,
                -factor * tau,
            )
            hit = rollback_paths(arm, y, tau) >= target
            crossed = hit.any(axis=1)
            stops.extend((hit.argmax(axis=1) + 1)[crossed].tolist())
        delays[f"{factor}tau"] = {
            "detected_share": len(stops) / 1000,
            "median": float(numpy.median(stops)) if stops else None,
            "p90": float(numpy.quantile(stops, 0.9)) if stops else None,
        }
    bound = ALPHA_ROLLBACK + tol(ALPHA_ROLLBACK, runs)
    return {
        "false_rollbacks": hits,
        "rate": hits / runs,
        "interval": wilson(hits, runs),
        "bound": bound,
        "detection_delay": delays,
        "passed": bool(hits / runs <= bound),
        "replicates": runs,
    }


def gate_b5(runs: int, base: int = 0) -> dict[str, Any]:
    """Exact budget algebra on random experiment sequences.

    Args:
        runs: Number of sequences.
        base: Seed offset (0 for certification).

    Returns:
        The number of sequences, the violations found and `passed` (no
        violation).
    """
    rng = rng_for(base, "b5", 0)
    violations = 0
    for _ in range(runs):
        alpha = scmle.data.ledger.budget.exact(
            float(rng.integers(1, 100)) / 1000, "a"
        )
        horizon = None if rng.random() < 0.5 else int(rng.integers(1, 30))
        budget = scmle.data.ledger.budget.DecisionBudget(
            alpha, scmle.data.ledger.budget.exact(0.01, "b"), horizon
        )
        total = Fraction(0)
        for index in range(int(rng.integers(0, 40))):
            try:
                budget, level = scmle.data.ledger.budget.assign_experiment(
                    budget, f"e{index}"
                )
            except scmle.data.ledger.budget.BudgetExhaustedError:
                break
            total += level
        if budget.spent != total or total > alpha:
            violations += 1
        try:
            scmle.data.ledger.budget.assign_experiment(
                budget, "overspend", alpha - budget.spent + 1
            )
            violations += 1
        except scmle.data.ledger.budget.BudgetExhaustedError:
            pass
        rebuilt = scmle.data.ledger.budget.DecisionBudget.from_dict(
            json.loads(json.dumps(budget.to_dict()))
        )
        if rebuilt != budget:
            violations += 1
    return {
        "sequences": runs,
        "violations": violations,
        "passed": violations == 0,
        "replicates": runs,
    }


def gate_b6(runs: int, base: int = 0) -> dict[str, Any]:
    """Size and power of the sample-ratio checks.

    Args:
        runs: Number of streams per case.
        base: Seed offset (0 for certification).

    Returns:
        For the correct and the distorted assignment the rejection rate of
        the anytime and the exact form with intervals, the level, the size
        bound and `passed`.
    """
    n, level = 100_000, 0.001
    target = -math.log(level)
    totals = numpy.arange(1, n + 1, dtype=float)
    out: dict[str, Any] = {}
    for label, q_true in (("size", 0.5), ("power", 0.52)):
        anytime_hits = exact_hits = 0
        chunk = 50
        for start in range(0, runs, chunk):
            size = min(chunk, runs - start)
            rng = rng_for(base, f"b6.{label}", start)
            counts = numpy.cumsum(rng.random((size, n)) < q_true, axis=1)
            bf = design.srm_log_bayes_factor(counts, totals, 0.5)
            anytime_hits += int((bf.max(axis=1) >= target).sum())
            exact_hits += sum(
                design.srm_p_value(int(k), n, 0.5) < level
                for k in counts[:, -1]
            )
        out[label] = {
            "anytime": anytime_hits / runs,
            "anytime_interval": wilson(anytime_hits, runs),
            "exact": exact_hits / runs,
            "exact_interval": wilson(exact_hits, runs),
        }
    bound = 0.01 + tol(0.01, runs)
    meets = (
        out["size"]["anytime"] <= bound
        and out["size"]["exact"] <= bound
        and out["power"]["anytime"] >= 0.95
        and out["power"]["exact"] >= 0.95
    )
    out.update({"level": level, "size_bound": bound, "passed": bool(meets)})
    out["replicates"] = runs
    return out


# --- lifecycle with drift, determinism ---------------------------------------


HOOK_WORLDS = 100
HOOK_BATCH = 500
HOOK_KEY = "f11-b9-approval-key"


def hook_bundle(seed: int) -> bundle.ModelBundle:
    """A small fitted logistic model for the registry the hook runs on.

    Args:
        seed: Seed of the training data.

    Returns:
        A bundle of a logistic regression on two features.
    """
    rng = numpy.random.default_rng(seed)
    frame = pandas.DataFrame(
        {"a": rng.normal(size=300), "b": rng.normal(size=300)}
    )
    frame["y"] = (frame.a + 0.3 * frame.b + rng.normal(0, 0.5, 300) > 0).astype(
        int
    )
    plan = scmle.contracts.data.FeaturePlan(("a", "b"), ())
    models = scmle.models.registry.ModelRegistry.with_builtins()
    features = frame.drop(columns="y")
    fitted = preprocessing.build_pipeline(
        plan, models.get("logistic_regression"), tasks.TaskType.BINARY, 0
    ).fit(features, frame.y)
    return bundle.ModelBundle.create(
        fitted,
        plan,
        features,
        "y",
        "binary_classification",
        "roc_auc",
        "logistic_regression",
    )


def hook_approval(record: Any) -> scmle.governance.registry.PromotionApproval:
    """A signed independent approval for one registered version.

    Args:
        record: The registered model version.

    Returns:
        An approved, signed promotion approval for that version.
    """
    return scmle.governance.registry.PromotionApproval(
        model=record.model,
        version=record.version,
        content_id=record.content_id,
        reviewer="f11-b9-reviewer",
        decision=scmle.governance.registry.PromotionDecision.APPROVED,
        reason="benchmark fixture",
        evidence=("f11_certify.py",),
        approved_at="2026-11-06T00:00:00+00:00",
    ).sign(HOOK_KEY)


def hook_registry(root: Path) -> tuple[Any, Any, Any]:
    """A registry with champion v1 and a staged challenger v2.

    Args:
        root: Folder for the registry and the work files.

    Returns:
        The registry store, the lifecycle manager and the challenger's
        record.
    """
    store = scmle.governance.registry.ModelRegistryStore(root / "registry")
    manager = lifecycle.LifecycleManager(store, "m", str(root / "work"))
    first = store.register(
        "m", hook_bundle(0).save(root / "export1"), trusted=True
    )
    store.promote(
        "m", first.version, approval=hook_approval(first), signing_key=HOOK_KEY
    )
    second = store.register(
        "m", hook_bundle(1).save(root / "export2"), trusted=True
    )
    store.promote("m", second.version, scmle.governance.registry.CHALLENGER)
    return store, manager, second


def hook_in(manager: Any, plan: Any) -> rollout_hook.RolloutHook:
    """A fresh hook, with its own in-memory ledger, on the shared manager.

    Args:
        manager: The lifecycle manager.
        plan: The rollout plan.

    Returns:
        The rollout hook.
    """
    ledger = deployment_ledger.DeploymentLedger.create(
        None, charter_hash="b" * 64, policy={"b9": 1}, alpha_promo="1/20"
    )
    return rollout_hook.RolloutHook(manager, plan, ledger)


def hook_promotes(
    hook: rollout_hook.RolloutHook, record: Any, seed: int
) -> None:
    """Starts an experiment, wins it on strong traffic and promotes.

    Args:
        hook: The rollout hook.
        record: The challenger's record, approved for promotion.
        seed: Seed of the traffic.
    """
    hook.start(f"promote-{seed}")
    rng = numpy.random.default_rng(seed)
    arm = (rng.random(6000) < 0.5).astype(float)
    outcome = (rng.random(6000) < 0.15 + 0.25 * arm).astype(float)
    hook.observe(arm, outcome)
    hook.promote(hook_approval(record), signing_key=HOOK_KEY)


def hook_path(base: int = 0) -> dict[str, Any]:
    """The four lifecycle scenarios run through `RolloutHook` on a registry.

    Each of the first `HOOK_WORLDS` worlds of a scenario is fed to the hook in
    batches; the hook's decision is compared with the vectorised e-process on
    the same world at the level the hook itself assigned.

    Args:
        base: Seed offset.

    Returns:
        Per scenario the worlds, the worlds decided, those held for an
        invalid sample ratio, the hook's decisions, the agreements with the
        vectorised path and the rate with its interval.
    """
    n = 6000
    plan = plan_for(delta_min=0.02, n_max=n)
    scenarios = {
        "no_effect": ("promotion", 0.0, 0.0),
        "below_delta_min": ("promotion", 0.01, 0.01),
        "false_rollback": ("rollback", 0.0, -0.015),
        "true_loss_rollback": ("rollback", 0.0, -0.06),
    }
    stream_names = {
        "no_effect": "b9.no_effect",
        "below_delta_min": "b9.below_delta_min",
        "false_rollback": "b9.rollback",
        "true_loss_rollback": "b9.harm",
    }
    out: dict[str, Any] = {}
    with tempfile.TemporaryDirectory() as scratch:
        store, manager, challenger = hook_registry(Path(scratch))
        for label, (kind, start, end) in scenarios.items():
            arm, y = f11_generators.drifting_experiment(
                rng_for(base, stream_names[label], 0),
                HOOK_WORLDS,
                n,
                0.5,
                start,
                end,
            )
            hits = agree = invalid = 0
            for world in range(HOOK_WORLDS):
                hook = hook_in(manager, plan)
                if kind == "promotion":
                    hook.start(f"w{world}")
                    status = ""
                    for first in range(0, n, HOOK_BATCH):
                        window = slice(first, first + HOOK_BATCH)
                        status = hook.observe(
                            arm[world, window], y[world, window]
                        ).status
                    target = hook.started().target
                    path = peeking_paths(
                        arm[world], y[world], 0.5, plan.delta_min
                    )
                    expected = bool(path.max() >= target)
                    won = status == rollout.PROMOTE
                else:
                    hook_promotes(hook, challenger, base + world)
                    monitor = hook.monitor
                    assert monitor is not None
                    status = ""
                    for first in range(0, n, HOOK_BATCH):
                        window = slice(first, first + HOOK_BATCH)
                        status = hook.watch(
                            arm[world, window], y[world, window]
                        )
                    target = monitor.target
                    path = rollback_paths(arm[world], y[world], 0.02)
                    expected = bool(path.max() >= target)
                    won = status == rollout.ROLLBACK
                    store.promote(
                        "m",
                        1,
                        approval=hook_approval(store.resolve("m", 1)),
                        signing_key=HOOK_KEY,
                    )
                    store.promote(
                        "m",
                        challenger.version,
                        scmle.governance.registry.CHALLENGER,
                    )
                if status == rollout.HOLD_INVALID:
                    invalid += 1
                    continue
                hits += int(won)
                agree += int(won == expected)
            decided = HOOK_WORLDS - invalid
            out[label] = {
                "worlds": HOOK_WORLDS,
                "decided": decided,
                "hold_invalid": invalid,
                "hook_decisions": hits,
                "agree": agree,
                "rate": hits / max(1, decided),
                "interval": wilson(hits, max(1, decided)),
            }
    return out


def gate_b9(runs: int, base: int = 0) -> dict[str, Any]:
    """Promotion and rollback under a wandering control rate.

    Args:
        runs: Number of streams per scenario (half for the true-loss one).
        base: Seed offset (0 for certification).

    Returns:
        The false promotions in the no-effect and below-minimum scenarios,
        the false rollbacks, the share of true losses rolled back, the
        through-hook results, the bounds, whether the hook agrees and
        `passed`.
    """
    n = 6000
    plan = plan_for(delta_min=0.02, n_max=n)
    target_p, target_r = -math.log(ALPHA_PROMOTE), -math.log(ALPHA_ROLLBACK)
    result: dict[str, Any] = {}

    def promoted(arm: numpy.ndarray, y: numpy.ndarray) -> numpy.ndarray:
        """Which streams ever cross the promotion level.

        Args:
            arm: Treatment indicators, shape (runs, units).
            y: Outcomes, same shape.

        Returns:
            A boolean array, True where the e-process reaches the level.
        """
        paths = peeking_paths(arm, y, 0.5, plan.delta_min)
        return numpy.asarray(paths.max(axis=1) >= target_p)

    for label, (start, end) in {
        "no_effect": (0.0, 0.0),
        "below_delta_min": (0.01, 0.01),
    }.items():
        hits = 0
        for chunk in range(0, runs, 500):
            arm, y = f11_generators.drifting_experiment(
                rng_for(base, f"b9.{label}", chunk),
                min(500, runs - chunk),
                n,
                0.5,
                start,
                end,
            )
            hits += int(promoted(arm, y).sum())
        result[label] = {
            "false_promotions": hits,
            "rate": hits / runs,
            "hold_final_share": 1 - hits / runs,
            "interval": wilson(hits, runs),
        }
    hits = 0
    for chunk in range(0, runs, 500):
        arm, y = f11_generators.drifting_experiment(
            rng_for(base, "b9.rollback", chunk),
            min(500, runs - chunk),
            n,
            0.5,
            0.0,
            -0.015,
        )
        hits += int(
            (rollback_paths(arm, y, 0.02).max(axis=1) >= target_r).sum()
        )
    result["false_rollback"] = {
        "rollbacks": hits,
        "rate": hits / runs,
        "interval": wilson(hits, runs),
    }
    caught = 0
    harm_runs = runs // 2
    for chunk in range(0, harm_runs, 500):
        arm, y = f11_generators.drifting_experiment(
            rng_for(base, "b9.harm", chunk),
            min(500, harm_runs - chunk),
            n,
            0.5,
            0.0,
            -0.06,
        )
        caught += int(
            (rollback_paths(arm, y, 0.02).max(axis=1) >= target_r).sum()
        )
    result["true_loss_rollback_share"] = caught / harm_runs
    result["through_hook"] = hook_path(base)
    hook_agrees = all(
        v["agree"] == v["decided"] for v in result["through_hook"].values()
    )
    promote_bound = ALPHA_PROMOTE + tol(ALPHA_PROMOTE, runs)
    rollback_bound = ALPHA_ROLLBACK + tol(ALPHA_ROLLBACK, runs)
    meets = (
        hook_agrees
        and result["no_effect"]["rate"] <= promote_bound
        and result["below_delta_min"]["rate"] <= promote_bound
        and result["below_delta_min"]["hold_final_share"] >= 1 - promote_bound
        and result["false_rollback"]["rate"] <= rollback_bound
    )
    result.update(
        {
            "promote_bound": promote_bound,
            "rollback_bound": rollback_bound,
            "hook_agrees": hook_agrees,
            "passed": bool(meets),
            "replicates": runs,
        }
    )
    return result


def forbidden_imports() -> list[str]:
    """Custody, vault or sealed-ledger imports found in `scmle.decision`.

    Returns:
        One `file: module` entry per forbidden import.
    """
    root = Path(__file__).parents[2] / "scmle" / "decision"
    found = []
    for path in sorted(root.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.ImportFrom) and node.module:
                names = [
                    node.module,
                    *(f"{node.module}.{a.name}" for a in node.names),
                ]
            elif isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            for name in names:
                if any(
                    part in name
                    for part in (
                        "data.custody",
                        "governance.vault",
                        "ledger.ledger",
                        "ledger.confirm",
                    )
                ):
                    found.append(f"{path.name}: {name}")
    return found


def battery_digest(base: int) -> str:
    """Hash of a small fixed battery of decision-layer outputs.

    Args:
        base: Seed offset.

    Returns:
        The SHA-256 of the threshold decision, the net-benefit band and a
        rollout run, serialised.
    """
    rng = rng_for(base, "b10", 0)
    score, _, y = f11_generators.calibration_world(rng, 3000, 0.7)
    decision = threshold.choose_threshold(
        score, y, cost_model(), seed=1, replicates=50
    )
    sample, outcome, groups = f11_generators.binormal(
        rng, 2000, 0.2, 1.5, 10, 0.5
    )
    band = netbenefit.net_benefit_band(
        outcome, (sample >= 0.8).astype(float), RANGE, clusters=groups, seed=2
    )
    arm, y2 = f11_generators.experiment(rng, 1, 3000, 0.5, 0.1, 0.04)
    run = rollout.Rollout(plan_for(), "e", ALPHA_PROMOTE).observe(arm[0], y2[0])
    payload = json.dumps(
        [decision.to_dict(), band.to_dict(), run.to_dict()],
        sort_keys=True,
        default=float,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def gate_b10(runs: int, base: int = 0) -> dict[str, Any]:
    """Determinism and the import graph.

    Args:
        runs: Unused; the gate always runs the battery twice.
        base: Seed offset (0 for certification).

    Returns:
        The digest, whether the repeat is equal, the forbidden imports, the
        interpreter measured, the replicate count (2) and `passed`.
    """
    first, second = battery_digest(base), battery_digest(base)
    forbidden = forbidden_imports()
    return {
        "digest": first,
        "repeat_equal": first == second,
        "forbidden_imports": forbidden,
        "interpreters_measured": [sys.version.split()[0]],
        "passed": bool(first == second and not forbidden),
        "replicates": 2,
    }


RUNNERS = {
    "A1": (gate_a1, 200),
    "A1c": (gate_a1c, 2000),
    "A2": (gate_a2, 200),
    "A3": (gate_a3, 2000),
    "A4": (gate_a4, 2000),
    "A5": (gate_a5, 2000),
    "A6": (gate_a6, 200),
    "A7": (gate_a7, 2000),
    "A7b": (gate_a7b, 20),
    "A8": (gate_a8, 100),
    "B1": (gate_b1, 10000),
    "B2": (gate_b2, 5000),
    "B3": (gate_b3, 500),
    "B4": (gate_b4, 10000),
    "B5": (gate_b5, 2000),
    "B6": (gate_b6, 1000),
    "B7": (gate_b7, 1000),
    "B8": (gate_b8, 200),
    "B9": (gate_b9, 2000),
    "B10": (gate_b10, 2),
}


def sources() -> dict[str, list[Any]]:
    """Code each gate rests on (hashed so a change makes it stale).

    Returns:
        Mapping of gate key to the functions its certificate covers.
    """
    return {
        "A1": [
            threshold.choose_threshold,
            threshold.calibration_check,
            threshold.fit_recalibrator,
            threshold.cross_fitted,
            threshold.ThresholdRule,
            threshold.bayes_threshold,
            gate_a1,
            f11_generators.calibration_world,
        ],
        "A1c": [threshold.calibration_check, gate_a1c],
        "A2": [costs.net_benefit, costs.expected_cost, gate_a2],
        "A3": [
            netbenefit.net_benefit_band,
            netbenefit.resample_means,
            netbenefit.diffs,
            band_setting,
            gate_a3,
            f11_generators.binormal,
            f11_generators.binormal_differences,
        ],
        "A4": [netbenefit.net_benefit_band, netbenefit.resample_means, gate_a4],
        "A5": [
            ope.estimate_value,
            ope.terms,
            ope.weights,
            ope.mean_interval,
            ope.support_violations,
            ope.fit_outcome_model,
            gate_a5,
            refusal_rows,
            f11_generators.bandit,
        ],
        "A6": [
            ope.estimate_value,
            ope.terms,
            ope.fit_threshold_policy,
            gate_a6,
        ],
        "A7": [
            constraints.satisfaction_test,
            constraints.newcombe,
            constraints.rate_tables,
            gate_a7,
            f11_generators.fairness_world,
        ],
        "A7b": [constraints.fit_group_thresholds, gate_a7b],
        "A8": [constraints.check_feasibility, gate_a8],
        "B1": [
            anytime.log_wealth,
            anytime.bet_grid,
            anytime.pseudo_outcomes,
            anytime.running_center,
            anytime.centered_paths,
            anytime.centered_stream,
            rollout.Rollout,
            gate_b1,
            peeking_paths,
            class_agreement,
        ],
        "B2": [
            anytime.log_wealth,
            anytime.pseudo_outcomes,
            anytime.running_center,
            anytime.centered_paths,
            anytime.centered_stream,
            rollout.metric_stream,
            rollout.metric_values,
            gate_b2,
        ],
        "B3": [
            anytime.log_wealth,
            anytime.pseudo_outcomes,
            anytime.running_center,
            anytime.centered_paths,
            anytime.centered_stream,
            gate_b3,
            esn_metric,
        ],
        "B4": [
            anytime.log_wealth,
            anytime.pseudo_outcomes,
            anytime.running_center,
            anytime.centered_paths,
            anytime.centered_stream,
            rollback_paths,
            gate_b4,
        ],
        "B5": [
            scmle.data.ledger.budget.assign_experiment,
            scmle.data.ledger.budget.DecisionBudget,
            gate_b5,
        ],
        "B6": [design.srm_log_bayes_factor, design.srm_p_value, gate_b6],
        "B7": [
            anytime.log_wealth,
            anytime.pseudo_outcomes,
            anytime.running_center,
            anytime.centered_paths,
            anytime.centered_stream,
            peeking_paths,
            gate_b7,
        ],
        "B8": [
            anytime.log_wealth,
            anytime.pseudo_outcomes,
            anytime.running_center,
            anytime.centered_paths,
            anytime.centered_stream,
            anytime.confidence_bound,
            gate_b8,
        ],
        "B9": [
            anytime.log_wealth,
            anytime.pseudo_outcomes,
            anytime.running_center,
            anytime.centered_paths,
            anytime.centered_stream,
            rollback_paths,
            rollout.Rollout,
            rollout.RollbackMonitor,
            rollout_hook.RolloutHook,
            lifecycle.LifecycleManager.defer_to_rollout,
            hook_bundle,
            hook_approval,
            hook_registry,
            hook_in,
            hook_promotes,
            hook_path,
            gate_b9,
        ],
        "B10": [battery_digest, forbidden_imports, gate_b10],
    }


def current_digests() -> dict[str, str]:
    """Code hash of every gate.

    Returns:
        Mapping of gate key to its current code hash.
    """
    return {k: certificates.code_hash(*v) for k, v in sources().items()}


def certify_all(base: int = 0, scale: float = 1.0) -> dict[str, Any]:
    """Runs every gate; `scale` shrinks replicate counts for dev looks.

    Args:
        base: Seed offset (0 for certification, `DEV_SEED_BASE` for looks).
        scale: Fraction of each gate's replicates to run (at least 2).

    Returns:
        The certificate document: version, development seed base, what was
        not measured and, per gate, the measurement with its code hash, seed
        range and duration.
    """
    document: dict[str, Any] = {
        "version": 1,
        "dev_seed_base": DEV_SEED_BASE,
        "not_measured": list(NOT_MEASURED),
        "gates": {},
    }
    digests = current_digests()
    for name, (runner, runs) in RUNNERS.items():
        started = time.monotonic()
        count = max(2, int(runs * scale)) if runs > 2 else runs
        measured = runner(count, base)
        measured["code_hash"] = digests[name]
        measured["seeds"] = [base, base + count]
        measured["seconds"] = round(time.monotonic() - started, 1)
        document["gates"][name] = measured
        print(name, measured["passed"], measured["seconds"], file=sys.stderr)
    return document


def problems(document: dict[str, Any] | None = None) -> list[str]:
    """Why the stored certificates cannot be trusted (empty when current).

    Args:
        document: The certificate document; read from `STORE_PATH` when
            None.

    Returns:
        Messages for a missing file, a gate without a certificate and a
        gate whose code hash differs from the current code.
    """
    if document is None:
        if not STORE_PATH.exists():
            return ["f11_certificates.json is missing"]
        document = json.loads(STORE_PATH.read_text(encoding="utf-8"))
    found = []
    digests = current_digests()
    for name in GATES:
        entry = document.get("gates", {}).get(name)
        if entry is None:
            found.append(f"{name}: no certificate")
        elif entry.get("code_hash") != digests[name]:
            found.append(f"{name}: stale (code changed since measurement)")
    return found


def passed_gates(document: dict[str, Any]) -> list[str]:
    """Gates that met their pre-registered floors.

    Args:
        document: The certificate document.

    Returns:
        The keys of gates whose `passed` is true.
    """
    return [k for k, v in document["gates"].items() if v["passed"]]


def descriptive_only(document: dict[str, Any] | None = None) -> list[str]:
    """Gates that missed their floors or could not run; filed, not gating.

    Args:
        document: The certificate document; read from `STORE_PATH` when
            None.

    Returns:
        The keys of gates that did not pass; every gate when the file is
        missing.
    """
    if document is None:
        if not STORE_PATH.exists():
            return list(GATES)
        document = json.loads(STORE_PATH.read_text(encoding="utf-8"))
    return [k for k, v in document["gates"].items() if not v["passed"]]


def check() -> dict[str, Any]:
    """The `--check f11` report.

    Returns:
        `ok`, the problems, the descriptive-only gates (empty when there are
        problems) and what was not measured.
    """
    issues = problems()
    return {
        "ok": not issues,
        "problems": issues,
        "descriptive_only": [] if issues else descriptive_only(),
        "not_measured": list(NOT_MEASURED),
    }


def main(arguments: list[str] | None = None) -> int:
    """Writes (`--write`), checks (`--check`) or looks (`--dev`).

    Args:
        arguments: Command-line arguments; `None` reads `sys.argv`.

    Returns:
        0 after `--write` and `--dev`; for `--check`, 0 when the certificates
        are current and 1 otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--dev", action="store_true")
    parser.add_argument("--scale", type=float, default=0.05)
    options = parser.parse_args(arguments)
    if options.dev:
        document = certify_all(DEV_SEED_BASE, options.scale)
        print(json.dumps(document, indent=2, sort_keys=True, default=float))
        return 0
    if options.write:
        document = certify_all()
        STORE_PATH.write_text(
            json.dumps(document, indent=2, sort_keys=True, default=float)
            + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"passed": passed_gates(document)}))
        return 0
    report = check()
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
