"""Measured gates for the closed loop.

Every threshold below is fixed in this file before the measurement and is not
retuned: a gate that misses its floor is filed as measured with `passed`
false, so the rule it covers stays descriptive. Certification seeds are
`0 .. replicates - 1`. All worlds are simulated (`streams.py`); real streams
may violate them.

Charter inputs used by the gates (inputs, not claims): reference risk
`R_REF = 0.10`, tolerance `DELTA = 0.05` (loss units), `ALPHA_ALARM = 0.05`,
`D_MAX = 50` batches, `ALPHA_PROMO = 0.05`, `DELTA_PROMO = 0`, `ALPHA_RB =
0.05`, `DELTA_RB = 0.05`, `R_STAR = 0.05`, `DELTA_CONF = 0.1`, batches of 50
cases. The monitors take the batch mean as the observation.

Each gate is the function named in brackets and is filed under the key in
`GATES`:

* False alarm (`gate_h1`): `R` stable streams of `T` batches. Pass: P(any
  alarm) `<= alpha + 2 sqrt(alpha (1 - alpha) / R)` both at the stable risk
  `R_REF` and on the boundary of the null (risk `R_REF + DELTA`). The
  existing fixed rule (`Thresholds`: 10 percent relative
  degradation on batches of 30 or more labelled rows) is measured on the same
  streams.
* Detection delay (`gate_h2`): abrupt rises to `R_REF + k DELTA`, `k` in
  1.5, 2, 3 (a rise of exactly `DELTA` is the boundary of the null and is
  undetectable by construction). Pass: P(detect `k = 2` within `D_MAX`) has a
  Clopper-Pearson lower bound `>= 0.90`.
* Benign shift (`gate_h3`): covariate-only shift. Pass: retrain-trigger rate
  `<= alpha + tolerance`; the PSI rule of the existing monitor is measured on
  the same streams.
* Promotion (`gate_h4`): sequences of 200 challengers, 80 percent true nulls,
  dependent through a shared latent; e-LOND over the sequence, decided in
  registration order (the loop's rule). Pass: realised FDR `<= alpha_promo +
  3 se`. The per-review bootstrap rule of `champion.py` is measured as a
  baseline.
* Rollback (`gate_h5`): false-rollback rate at no difference
  `<= alpha_rb + 3 se`.
* Selection (`gate_h6`): MAR with logged propensity. Pass: bias of the
  Horvitz-Thompson risk within 3 se, the confidence sequence covers the true
  risk at all times in at least `1 - alpha - 3 se` of the streams, and the
  naive labelled-only mean is biased (published). MNAR without an audit slice:
  the loop takes no retrain, promotion or rollback action and tiers the stream
  `unverified`.
* Maturity (`gate_h7`): outcome-dependent delay. Pass: the
  maturity-by-prediction-time estimate is within 3 se of the truth and the
  arrival-based one is below it by more than 3 se.
* Feedback (`gate_h8`): the bypass-slice test. Pass: false detection in
  loop-free worlds `<= alpha + tolerance`; power reported by bypass share;
  without a bypass slice the verdict is `unverified`.
* Threshold (`gate_h9`): `P(R(tau_hat) > r*) <= delta + 2 sqrt(delta (1 -
  delta) / R)`; the best-observed-risk cut is measured as the naive baseline.
* Under shift (`gate_h10`): the selective-risk alarm detects a rise of the
  answered cases' risk to `2 r*` within `D_MAX` with probability `>= 0.90`
  (Clopper-Pearson lower bound); the window before the halt is reported.

NOT measured here: agreement with a domain owner's signed escalation audit
and replay of recorded incident streams. No owner audit exists, no replay
corpus is verified, and Python 3.11 replay of the ledger was not run (only
the interpreter of the environment). The loop stays `unverified` for
production until those exist.

    python -m benchmarks.qualification.loop_certify --write
    python -m benchmarks.qualification.loop_certify --check
"""

import argparse
import json
import math
from pathlib import Path
import sys
import tempfile
import types
from typing import Any

import numpy
import pandas

from benchmarks.qualification import streams
from scmle.data.detectors import certificates
from scmle.evaluation import coverage
from scmle.governance import escalation
import scmle.governance.loop
from scmle.monitoring import labels
from scmle.monitoring import reference
from scmle.monitoring import sequential_tests
import scmle.monitoring.monitor

STORE_PATH = Path(__file__).with_name("loop_certificates.json")
DEV_SEED_BASE = 10_000
R_REF = 0.10
DELTA = 0.05
ALPHA_ALARM = 0.05
D_MAX = 50
ALPHA_PROMO = 0.05
DELTA_PROMO = 0.0
ALPHA_RB = 0.05
DELTA_RB = 0.05
R_STAR = 0.05
DELTA_CONF = 0.1
BATCH = 50
REPLICATES = 2000
T_FALSE_ALARM = 5000
# floor on the Wilson lower bound of the share of detections within D_MAX
WITHIN_D_MAX_FLOOR = 0.90
# share of cases on which a challenger sees the champion's latent
SHARED_LATENT_SHARE = 0.7
# share of challengers with a true advantage
ADVANTAGE_SHARE = 0.2
GATES = ("H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8", "H9", "H10")
NOT_MEASURED = (
    "H11 agreement with a domain owner's signed escalation audit",
    "replay of recorded incident-like streams (no verified corpus)",
    "ledger replay on Python 3.11 (only the environment's interpreter ran)",
    "independent confidence-sequence implementation (none pinned)",
)


def rate_floor(rate: float, alpha: float, runs: int) -> dict[str, Any]:
    """A rate with the `alpha + 2 sqrt(alpha (1 - alpha) / R)` bound.

    Args:
        rate: The observed rate.
        alpha: The nominal level.
        runs: Number of replicates the rate is observed over.

    Returns:
        The rate, the bound and whether the rate is within it.
    """
    bound = alpha + 2.0 * math.sqrt(alpha * (1.0 - alpha) / runs)
    return {"rate": rate, "bound": bound, "meets": rate <= bound}


def lower_bound(hits: int, runs: int) -> float:
    """Clopper-Pearson 95 percent lower limit.

    Args:
        hits: Number of successes.
        runs: Number of trials.

    Returns:
        The lower limit of the Clopper-Pearson interval computed at
        confidence 0.90 (a one-sided 95 percent limit).
    """
    return coverage.clopper_pearson(hits, runs, 0.90)[0]


def batch_means(
    rng: numpy.random.Generator, risk: float, runs: int
) -> numpy.ndarray:
    """One batch mean loss per stream, for Bernoulli(risk) cases.

    Args:
        rng: Random generator.
        risk: Loss probability of each case.
        runs: Number of streams.

    Returns:
        An array of `runs` batch means over `BATCH` cases.
    """
    return rng.binomial(BATCH, risk, runs) / BATCH


def fixed_rule(batch_mean: numpy.ndarray) -> numpy.ndarray:
    """The existing rule: loss above `R_REF` by more than 10 percent.

    Args:
        batch_mean: Batch mean losses.

    Returns:
        A boolean array, True where the mean exceeds
        `R_REF * (1 + performance_tolerance)`.
    """
    tolerance = scmle.monitoring.monitor.Thresholds().performance_tolerance
    return batch_mean > R_REF * (1.0 + tolerance)


def gate_h1(replicates: int, base: int = 0) -> dict[str, Any]:
    """False alarm at the stable risk and on the null's boundary.

    Args:
        replicates: Number of streams.
        base: Seed of the random generator.

    Returns:
        For the stable and the boundary risk, the alarm rate of the
        sequential test with its bound and the alarm rate of the fixed
        rule, and `passed`.
    """
    out: dict[str, Any] = {"replicates": replicates, "batches": T_FALSE_ALARM}
    for name, risk in (("stable", R_REF), ("boundary", R_REF + DELTA)):
        rng = numpy.random.default_rng(base)
        monitor = sequential_tests.MixtureTest(
            numpy.full(replicates, (R_REF + DELTA)),
            sequential_tests.GREATER,
            ALPHA_ALARM,
        )
        fixed = numpy.zeros(replicates, dtype=bool)
        for _ in range(T_FALSE_ALARM):
            values = batch_means(rng, risk, replicates)
            monitor.update(values)
            fixed |= fixed_rule(values)
        out[name] = {
            "true_risk": risk,
            "betting": rate_floor(
                float(monitor.ever_crossed.mean()), ALPHA_ALARM, replicates
            ),
            "fixed_rule_rate": float(fixed.mean()),
        }
    out["passed"] = bool(
        out["stable"]["betting"]["meets"]
        and out["boundary"]["betting"]["meets"]
    )
    return out


def detection(
    replicates: int, risk: float, pre: int, horizon: int, base: int
) -> dict[str, Any]:
    """Batches from an abrupt change to the first alarm.

    Args:
        replicates: Number of streams.
        risk: Loss rate after the change.
        pre: Batches at the reference risk before the change.
        horizon: Batches after the change.
        base: Seed of the random generator.

    Returns:
        The risk, the share detected, the share detected within `D_MAX` with
        its lower bound, and the median delay of the sequential test and of
        the fixed rule (None when none was detected).
    """
    rng = numpy.random.default_rng(base)
    monitor = sequential_tests.MixtureTest(
        numpy.full(replicates, R_REF + DELTA),
        sequential_tests.GREATER,
        ALPHA_ALARM,
    )
    first = numpy.full(replicates, -1)
    fixed_first = numpy.full(replicates, -1)
    for step in range(pre + horizon):
        level = R_REF if step < pre else risk
        values = batch_means(rng, level, replicates)
        monitor.update(values)
        late = step >= pre
        hit = monitor.ever_crossed & (first < 0) & late
        first[hit] = step - pre
        fixed_hit = fixed_rule(values) & (fixed_first < 0) & late
        fixed_first[fixed_hit] = step - pre
    found = first >= 0
    within = int(((first >= 0) & (first < D_MAX)).sum())
    return {
        "risk": risk,
        "detected": float(found.mean()),
        "within_d_max": within / replicates,
        "within_d_max_lower": lower_bound(within, replicates),
        "median_delay": (
            float(numpy.median(first[found])) if found.any() else None
        ),
        "fixed_rule_median_delay": (
            float(numpy.median(fixed_first[fixed_first >= 0]))
            if (fixed_first >= 0).any()
            else None
        ),
    }


def gate_h2(replicates: int, base: int = 0) -> dict[str, Any]:
    """Detection delay.

    Args:
        replicates: Number of streams.
        base: Seed of the random generator.

    Returns:
        The `detection` result for each rise factor, `d_max` and `passed`.
    """
    rows: dict[str, Any] = {}
    for factor in (1.5, 2.0, 3.0):
        rows[str(factor)] = detection(
            replicates, R_REF + factor * DELTA, 200, 400, base
        )
    return {
        "replicates": replicates,
        "d_max": D_MAX,
        "by_rise": rows,
        "passed": bool(rows["2.0"]["within_d_max_lower"] >= WITHIN_D_MAX_FLOOR),
    }


def gate_h3(replicates: int, base: int = 0) -> dict[str, Any]:
    """Covariate-only shift: the loss monitor stays quiet, PSI does not.

    Args:
        replicates: Number of streams.
        base: Seed offset.

    Returns:
        The retrain-trigger rate with its bound, the rate at which the PSI
        rule flags the shift and `passed`.
    """
    trigger = 0
    psi_flag = 0
    for index in range(replicates):
        stream = streams.simulate(
            streams.StreamSpec(
                kind="covariate",
                batches=200,
                size=BATCH,
                base=R_REF,
                change=100,
                shift=2.0,
                seed=base + index,
            )
        )
        monitor = sequential_tests.MixtureTest(
            R_REF + DELTA, sequential_tests.GREATER, ALPHA_ALARM
        )
        for step in range(stream.spec.batches):
            monitor.update(float(stream.loss[step].mean()))
        trigger += int(monitor.ever_crossed)
        edges = numpy.quantile(stream.z[:100].ravel(), numpy.linspace(0, 1, 11))
        edges[0], edges[-1] = -numpy.inf, numpy.inf
        expected = numpy.histogram(stream.z[:100].ravel(), edges)[0]
        actual = numpy.histogram(stream.z[150:].ravel(), edges)[0]
        psi = reference.population_stability_index(
            expected / expected.sum(), actual / actual.sum()
        )
        psi_flag += int(psi >= scmle.monitoring.monitor.Thresholds().psi_high)
    rate = trigger / replicates
    bound = rate_floor(rate, ALPHA_ALARM, replicates)
    return {
        "replicates": replicates,
        "retrain_trigger_rate": bound,
        "psi_rule_flag_rate": psi_flag / replicates,
        "passed": bool(bound["meets"]),
    }


def sequence_gaps(
    rng: numpy.random.Generator,
    advantages: numpy.ndarray,
    batches: int,
    base: float,
) -> numpy.ndarray:
    """Batch mean gaps (champion minus challenger), shape `(count, batches)`.

    Every case has a shared latent `u`; the champion errs when `u < base`. A
    challenger sees `u` on a random 70 percent of the cases and its own
    independent latent on the rest, and errs when its latent is below
    `base - advantage`. So every challenger has the champion's error rate
    minus its advantage (a true null has exactly the champion's rate and a
    noisy, mean-zero gap), and the challengers are dependent through `u`.

    Args:
        rng: Random generator.
        advantages: True advantage of each challenger.
        batches: Number of batches.
        base: Champion error rate.

    Returns:
        The mean loss difference per batch for every challenger.
    """
    latent = rng.random((batches, BATCH))
    count = len(advantages)
    own = rng.random((count, batches, BATCH))
    shared = rng.random((count, batches, BATCH)) < SHARED_LATENT_SHARE
    seen = numpy.where(shared, latent[None], own)
    low = numpy.clip(base - advantages, 0.0, 1.0)[:, None, None]
    champion = (latent < base).astype(float)[None]
    challenger = (seen < low).astype(float)
    difference: numpy.ndarray = (champion - challenger).mean(axis=2)
    return difference


def eprocess_paths(gaps: numpy.ndarray) -> numpy.ndarray:
    """Log-capital paths of a mixture betting test run on each row of gaps.

    Each gap is mapped from [-1, 1] to [0, 1] and tested against the null
    mean for the promotion margin, one test per row.

    Args:
        gaps: Batch mean gaps in [-1, 1], shape (challengers, batches).

    Returns:
        The log capital of each row's test after each batch, same shape.
    """
    count, batches = gaps.shape
    test = sequential_tests.MixtureTest(
        numpy.full(count, (DELTA_PROMO + 1.0) / 2.0),
        sequential_tests.GREATER,
        ALPHA_PROMO,
    )
    paths = numpy.empty((count, batches))
    for step in range(batches):
        test.update((gaps[:, step] + 1.0) / 2.0)
        paths[:, step] = test.log_capital
    return paths


def elond_decisions(paths: numpy.ndarray, horizon: int) -> numpy.ndarray:
    """Replays the loop's rule: heads are decided in order by e-LOND.

    Args:
        paths: Log-capital paths, shape (challengers, batches).
        horizon: Number of batches.

    Returns:
        A boolean array, True for each promoted challenger.
    """
    sequence = sequential_tests.PromotionSequence("1/20")
    rejected = numpy.zeros(paths.shape[0], dtype=bool)
    pointer = 0
    for index in range(paths.shape[0]):
        name = f"c{index}"
        sequence.register(name)
        level = float(sequence.rule.level(sequence.state))
        log_need = math.log(1.0 / level)
        reached = numpy.flatnonzero(paths[index, pointer:] >= log_need)
        moment = pointer + int(reached[0]) if reached.size else horizon - 1
        pointer = moment
        step = sequence.decide(
            name, float(math.exp(min(paths[index, moment], 700.0)))
        )
        rejected[index] = step.rejected
    return rejected


def bootstrap_rule(
    gaps: numpy.ndarray, rng: numpy.random.Generator
) -> numpy.ndarray:
    """The existing rule per review: promote if the bootstrap lower end > 0.

    Reviews every 10 batches; a challenger is promoted at its first review at
    which the 2.5 percent bootstrap quantile of the mean gap is above zero.
    The batch gaps are multiples of 1/50 and are resampled as cases through
    their counts (a multinomial over the observed gap values).

    Args:
        gaps: Batch mean gaps, shape (challengers, batches).
        rng: Random generator.

    Returns:
        A boolean array, True for each promoted challenger.
    """
    count, batches = gaps.shape
    promoted = numpy.zeros(count, dtype=bool)
    for index in range(count):
        for review in range(10, batches + 1, 10):
            cases = numpy.rint(gaps[index, :review] * BATCH).astype(int)
            total = review * BATCH
            plus = int(numpy.clip(cases, 0, None).sum())
            minus = int(numpy.clip(-cases, 0, None).sum())
            probabilities = (
                numpy.array([plus, minus, total - plus - minus], dtype=float)
                / total
            )
            draws = rng.multinomial(total, probabilities, size=200)
            means = (draws[:, 0] - draws[:, 1]) / total
            if numpy.quantile(means, 0.025) > DELTA_PROMO:
                promoted[index] = True
                break
    return promoted


def gate_h4(replicates: int, base: int = 0) -> dict[str, Any]:
    """False promotion over dependent challenger sequences.

    Args:
        replicates: Number of sequences.
        base: Seed offset.

    Returns:
        The design, the false-discovery rate and power of e-LOND (with the
        standard error) and of the bootstrap rule, whether the bootstrap
        rule exceeds the level, and `passed`.
    """
    count, batches, base_risk, better = 200, 150, 0.30, 0.20
    fdp: list[float] = []
    power: list[float] = []
    fdp_boot: list[float] = []
    power_boot: list[float] = []
    for index in range(replicates):
        rng = numpy.random.default_rng(base + index)
        truth = rng.random(count) < ADVANTAGE_SHARE
        advantages = numpy.where(truth, better, 0.0)
        gaps = sequence_gaps(rng, advantages, batches, base_risk)
        decided = elond_decisions(eprocess_paths(gaps), batches)
        boot = bootstrap_rule(gaps, numpy.random.default_rng(base + index + 7))
        for rejected, fp, pw in (
            (decided, fdp, power),
            (boot, fdp_boot, power_boot),
        ):
            discoveries = int(rejected.sum())
            fp.append(
                float((rejected & ~truth).sum()) / discoveries
                if discoveries
                else 0.0
            )
            pw.append(
                float((rejected & truth).sum()) / max(int(truth.sum()), 1)
            )
    standard = float(numpy.std(fdp, ddof=1) / math.sqrt(replicates))
    mean = float(numpy.mean(fdp))
    return {
        "replicates": replicates,
        "challengers": count,
        "true_better_share": 0.2,
        "true_advantage": better,
        "horizon_batches": batches,
        "elond_fdr": mean,
        "elond_se": standard,
        "elond_power": float(numpy.mean(power)),
        "bootstrap_fdr": float(numpy.mean(fdp_boot)),
        "bootstrap_power": float(numpy.mean(power_boot)),
        "bootstrap_exceeds_level": bool(numpy.mean(fdp_boot) > ALPHA_PROMO),
        "passed": bool(mean <= ALPHA_PROMO + 3.0 * standard),
    }


def gate_h5(replicates: int, base: int = 0) -> dict[str, Any]:
    """Rollback: false rollback at no difference, power at worse champions.

    Args:
        replicates: Number of streams.
        base: Seed of the random generator.

    Returns:
        For each degradation of the champion (0, 1 and 2 times `DELTA_RB`)
        the rollback rate and median delay, and `passed`.
    """
    rows: dict[str, Any] = {}
    for name, factor in (("0", 0.0), ("1", 1.0), ("2", 2.0)):
        rng = numpy.random.default_rng(base)
        process = sequential_tests.MixtureTest(
            numpy.full(replicates, (DELTA_RB + 1.0) / 2.0),
            sequential_tests.GREATER,
            ALPHA_RB,
        )
        first = numpy.full(replicates, -1)
        for step in range(400):
            worse = factor * DELTA_RB
            gap = rng.binomial(BATCH, worse, replicates) / BATCH
            process.update((gap + 1.0) / 2.0)
            hit = process.ever_crossed & (first < 0)
            first[hit] = step
        found = first >= 0
        rows[name] = {
            "worse_by": factor * DELTA_RB,
            "rollback_rate": float(found.mean()),
            "median_delay": float(numpy.median(first[found]))
            if found.any()
            else None,
        }
    standard = math.sqrt(ALPHA_RB * (1 - ALPHA_RB) / replicates)
    return {
        "replicates": replicates,
        "by_difference": rows,
        "passed": bool(rows["0"]["rollback_rate"] <= ALPHA_RB + 3 * standard),
    }


def mar_stream(seed: int, batches: int = 120) -> streams.Stream:
    """A MAR stream whose loss depends on the selecting covariate.

    Args:
        seed: Stream seed.
        batches: Number of batches.

    Returns:
        The simulated stream.
    """
    return streams.simulate(
        streams.StreamSpec(
            labels="mar",
            floor=0.1,
            dependence=0.4,
            batches=batches,
            size=100,
            window=0,
            seed=seed,
        )
    )


def run_mnar_loop(seed: int) -> dict[str, Any]:
    """A rise on an MNAR stream without an audit slice, through the loop.

    Args:
        seed: Stream seed.

    Returns:
        The number of retrain, promotion-proposal and rollback-proposal
        entries in the ledger, whether the loop tiers the stream
        `unverified`, and whether the ledger halted.
    """
    stream = streams.simulate(
        streams.StreamSpec(
            kind="abrupt",
            labels="mnar",
            batches=60,
            size=BATCH,
            change=20,
            delta=0.3,
            window=0,
            seed=seed,
        )
    )
    charter = types.SimpleNamespace(charter_hash="c" * 64)
    policy = scmle.governance.loop.Preregistration(
        reference_risk=R_REF,
        tolerance=DELTA,
        r_star=0.6,
        initial_tau=0.5,
        min_matured=5,
        outcome_window=0,
    )
    with tempfile.TemporaryDirectory() as directory:
        loop = scmle.governance.loop.Loop.open(
            directory,
            charter,
            policy,
            hooks=scmle.governance.loop.Hooks(retrain=lambda: "x"),
        )
        for step in range(stream.spec.batches):
            frame = pandas.DataFrame(
                {"case_id": [f"b{step}-{i}" for i in range(BATCH)]}
            )
            loop.observe(frame, [0.0] * BATCH, ["a"] * BATCH)
            arrived = {
                f"b{b}-{p}": float(stream.loss[b, p])
                for b in range(step + 1)
                for p in numpy.flatnonzero(
                    (stream.arrival[b] == step) & stream.selected[b]
                )
            }
            loop.mature(arrived)
            loop.step()
        kinds = [row["kind"] for row in loop.ledger.rows]
        return {
            "actions": sum(
                kinds.count(k)
                for k in ("retrain", "promotion_proposal", "rollback_proposal")
            ),
            "unverified": loop.mechanism is labels.Mechanism.UNVERIFIED,
            "halted": bool(loop.ledger.halted),
        }


def gate_h6(replicates: int, base: int = 0) -> dict[str, Any]:
    """Selection correction with logged propensity; MNAR without a slice.

    Args:
        replicates: Number of streams (at most 300 are run).
        base: Seed offset.

    Returns:
        The bias of the weighted estimator and its standard error, the bias
        of the naive mean, the time-uniform coverage of the confidence
        sequence and its floor, the MNAR results and `passed`.
    """
    runs = min(replicates, 300)
    weighted, naive, truth, covered = [], [], [], 0
    for index in range(runs):
        stream = mar_stream(base + index)
        estimates, seen_loss, seen_n = [], 0.0, 0
        truth.append(streams.marginal_risk(R_REF, stream.spec.dependence))
        sequence = sequential_tests.ConfidenceSequence(ALPHA_ALARM, 99)
        inside = True
        scale = 1.0 / 0.1
        for step in range(stream.spec.batches):
            observed = stream.selected[step]
            estimate = labels.estimate_batch(
                numpy.where(observed, stream.loss[step], numpy.nan),
                observed,
                stream.pi[step],
                None,
                0.1,
            )
            assert estimate.mean is not None
            estimates.append(estimate.mean)
            seen_loss += float(stream.loss[step][observed].sum())
            seen_n += int(observed.sum())
            low, high = sequence.update(estimate.mean / scale)
            inside &= low <= truth[-1] / scale <= high
        weighted.append(float(numpy.mean(estimates)))
        naive.append(seen_loss / seen_n)
        covered += int(inside)
    diff = numpy.array(weighted) - numpy.array(truth)
    standard = float(diff.std(ddof=1) / math.sqrt(runs))
    bias = float(diff.mean())
    naive_bias = float(numpy.mean(numpy.array(naive) - numpy.array(truth)))
    cover = covered / runs
    floor = 1.0 - ALPHA_ALARM - 3.0 * math.sqrt(0.05 * 0.95 / runs)
    mnar = [run_mnar_loop(base + i) for i in range(10)]
    quiet = all(m["actions"] == 0 and m["unverified"] for m in mnar)
    return {
        "replicates": runs,
        "weighted_bias": bias,
        "weighted_bias_se": standard,
        "naive_bias": naive_bias,
        "cs_time_uniform_coverage": cover,
        "cs_coverage_floor": floor,
        "mnar_without_audit_slice": {
            "runs": len(mnar),
            "actions_taken": sum(m["actions"] for m in mnar),
            "all_unverified": all(m["unverified"] for m in mnar),
        },
        "passed": bool(
            abs(bias) <= 3.0 * standard
            and cover >= floor
            and naive_bias > 3.0 * standard
            and quiet
        ),
    }


def gate_h7(replicates: int, base: int = 0) -> dict[str, Any]:
    """Maturity by prediction time against arrival time.

    Args:
        replicates: Number of streams.
        base: Seed offset.

    Returns:
        The bias and standard error of the maturity-based and of the
        arrival-based estimates, and `passed`.
    """
    now, window, slow, base_risk = 150, 2, 6, 0.2
    by_prediction, by_arrival = [], []
    for index in range(replicates):
        stream = streams.simulate(
            streams.StreamSpec(
                batches=now + 1,
                size=100,
                window=window,
                slow=slow,
                base=base_risk,
                seed=base + index,
            )
        )
        position = numpy.arange(now + 1)[:, None] + numpy.zeros_like(
            stream.loss
        )
        horizon = window + slow
        recent = (position > now - 2 * horizon) & (position <= now)
        arrived = stream.arrival <= now
        by_arrival.append(float(stream.loss[recent & arrived].mean()))
        by_prediction.append(
            float(stream.loss[position <= now - 2 * horizon].mean())
        )
    prediction = numpy.array(by_prediction) - base_risk
    arrival = numpy.array(by_arrival) - base_risk
    p_se = float(prediction.std(ddof=1) / math.sqrt(replicates))
    a_se = float(arrival.std(ddof=1) / math.sqrt(replicates))
    return {
        "replicates": replicates,
        "maturity_bias": float(prediction.mean()),
        "maturity_se": p_se,
        "arrival_bias": float(arrival.mean()),
        "arrival_se": a_se,
        "passed": bool(
            abs(prediction.mean()) <= 3 * p_se and arrival.mean() < -3 * a_se
        ),
    }


def gate_h8(replicates: int, base: int = 0) -> dict[str, Any]:
    """Feedback detection by the bypass slice.

    Args:
        replicates: Number of worlds per case (at most 100 are run).
        base: Seed offset.

    Returns:
        The detection rate for the loop-free world and for bypass shares of
        0.05, 0.10 and 0.20, the false-detection bound, the verdict without
        a bypass slice and `passed`.
    """
    runs = min(replicates, 100)
    results: dict[str, Any] = {}
    for name, gamma, bypass in (
        ("loop_free", 0.0, 0.2),
        ("loop_bypass_0.05", 0.8, 0.05),
        ("loop_bypass_0.10", 0.8, 0.10),
        ("loop_bypass_0.20", 0.8, 0.20),
    ):
        fired = 0
        for index in range(runs):
            stream = streams.feedback_stream(
                40, 100, gamma, bypass, base + index
            )
            frame = pandas.DataFrame({"z": stream.next_z.reshape(-1)})
            flags = stream.bypass.reshape(-1)
            check = labels.bypass_test(
                frame[~flags].iloc[:300],
                frame[flags].iloc[:300],
                base + index,
                permutations=99,
            )
            fired += int(
                labels.feedback_verdict(bypass, check) == "loop_detected"
            )
        results[name] = fired / runs
    bound = ALPHA_ALARM + 2 * math.sqrt(ALPHA_ALARM * (1 - ALPHA_ALARM) / runs)
    return {
        "replicates": runs,
        "detection_rate": results,
        "false_detection_bound": bound,
        "without_bypass_slice": labels.feedback_verdict(0.0, None),
        "passed": bool(
            results["loop_free"] <= bound
            and labels.feedback_verdict(0.0, None) == labels.UNVERIFIED
        ),
    }


def gate_h9(replicates: int, base: int = 0) -> dict[str, Any]:
    """The selected threshold violates the target risk with prob <= delta.

    Args:
        replicates: Number of calibration sets (at most 1000 are run).
        base: Seed offset.

    Returns:
        The violation rate and bound, the naive best-cut violation rate, the
        rate at which everything escalates, the mean coverage at the chosen
        threshold, the oracle threshold and coverage, and `passed`.
    """
    runs = min(replicates, 1000)
    violations, naive_violations, coverage_values, infinite = 0, 0, [], 0
    grid = numpy.linspace(1.0, 0.0, escalation.DEFAULT_GRID)
    for index in range(runs):
        data = streams.draw_calibration(2000, R_STAR, base + index)
        result = escalation.select_threshold(
            data["score"], data["loss"], R_STAR, DELTA_CONF
        )
        if result.everything_escalates:
            infinite += 1
        else:
            violations += int(streams.oracle_risk(result.tau) > R_STAR)
            coverage_values.append(1.0 - result.tau)
        best = 1.0
        for tau in grid:
            kept = data["score"] >= tau
            if kept.any() and data["loss"][kept].mean() <= R_STAR:
                best = float(tau)
        naive_violations += int(streams.oracle_risk(best) > R_STAR)
    rate = violations / runs
    bound = DELTA_CONF + 2.0 * math.sqrt(DELTA_CONF * (1 - DELTA_CONF) / runs)
    oracle_tau = 1.0 - math.sqrt(R_STAR / 0.1)
    return {
        "replicates": runs,
        "violation_rate": rate,
        "violation_bound": bound,
        "naive_best_cut_violation_rate": naive_violations / runs,
        "everything_escalates_rate": infinite / runs,
        "mean_coverage_at_tau": float(numpy.mean(coverage_values)),
        "oracle_tau": oracle_tau,
        "oracle_coverage": 1.0 - oracle_tau,
        "passed": bool(rate <= bound),
    }


def gate_h10(replicates: int, base: int = 0) -> dict[str, Any]:
    """Selective-risk alarm after a rise of the answered cases' risk.

    Args:
        replicates: Number of streams.
        base: Seed of the random generator.

    Returns:
        For rises to 2 and 3 times `R_STAR` the detection within `D_MAX`
        with its lower bound and the delays, the coverage after a halt and
        `passed`.
    """
    rows: dict[str, Any] = {}
    for name, level in (("r_star_x_2", 2 * R_STAR), ("r_star_x_3", 3 * R_STAR)):
        rng = numpy.random.default_rng(base)
        monitor = sequential_tests.MixtureTest(
            numpy.full(replicates, R_STAR), sequential_tests.GREATER, DELTA_CONF
        )
        first = numpy.full(replicates, -1)
        pre = 100
        for step in range(pre + 300):
            risk = 0.5 * R_STAR if step < pre else level
            monitor.update(batch_means(rng, risk, replicates))
            hit = monitor.ever_crossed & (first < 0) & (step >= pre)
            first[hit] = step - pre
        within = int(((first >= 0) & (first < D_MAX)).sum())
        found = first >= 0
        rows[name] = {
            "risk_after": level,
            "within_d_max": within / replicates,
            "within_d_max_lower": lower_bound(within, replicates),
            "median_delay_batches": (
                float(numpy.median(first[found])) if found.any() else None
            ),
            "excess_cases_before_halt": (
                float(numpy.median(first[found]) * BATCH)
                if found.any()
                else None
            ),
        }
    escalate = escalation.route(numpy.ones(10), 0.5, 1.0, halted=True)
    return {
        "replicates": replicates,
        "by_rise": rows,
        "coverage_after_halt": escalate.coverage,
        "passed": bool(
            rows["r_star_x_2"]["within_d_max_lower"] >= WITHIN_D_MAX_FLOOR
            and escalate.coverage == 0.0
        ),
    }


RUNNERS = {
    "H1": (gate_h1, REPLICATES),
    "H2": (gate_h2, REPLICATES),
    "H3": (gate_h3, 300),
    "H4": (gate_h4, 100),
    "H5": (gate_h5, REPLICATES),
    "H6": (gate_h6, 300),
    "H7": (gate_h7, 300),
    "H8": (gate_h8, 100),
    "H9": (gate_h9, 1000),
    "H10": (gate_h10, REPLICATES),
}


def sources() -> dict[str, list[Any]]:
    """Code each gate rests on (hashed so a change makes it stale).

    Returns:
        Mapping of gate key to the functions its certificate covers.
    """
    return {
        "H1": [sequential_tests.MixtureTest, fixed_rule, batch_means, gate_h1],
        "H2": [sequential_tests.MixtureTest, fixed_rule, detection, gate_h2],
        "H3": [sequential_tests.MixtureTest, streams.simulate, gate_h3],
        "H4": [
            sequential_tests.MixtureTest,
            sequential_tests.PromotionSequence,
            sequence_gaps,
            eprocess_paths,
            elond_decisions,
            bootstrap_rule,
            gate_h4,
        ],
        "H5": [sequential_tests.MixtureTest, gate_h5],
        "H6": [
            labels.estimate_batch,
            labels.classify,
            sequential_tests.ConfidenceSequence,
            streams.simulate,
            run_mnar_loop,
            scmle.governance.loop.Loop.step,
            scmle.governance.loop.Loop.decide,
            gate_h6,
        ],
        "H7": [labels.is_matured, streams.simulate, gate_h7],
        "H8": [
            labels.bypass_test,
            labels.feedback_verdict,
            streams.feedback_stream,
            gate_h8,
        ],
        "H9": [
            escalation.select_threshold,
            escalation.binomial_p_value,
            streams.draw_calibration,
            gate_h9,
        ],
        "H10": [sequential_tests.MixtureTest, escalation.route, gate_h10],
    }


def current_digests() -> dict[str, str]:
    """Code hash of every gate.

    Returns:
        Mapping of gate key to its current code hash.
    """
    return {k: certificates.code_hash(*v) for k, v in sources().items()}


def certify_all() -> dict[str, Any]:
    """Runs every gate on certification seeds.

    Returns:
        The certificate document: version, development seed base, simulator
        hash, charter inputs, what was not measured and, per gate, the
        measurement with its code hash and seed range.
    """
    document: dict[str, Any] = {
        "version": 1,
        "dev_seed_base": DEV_SEED_BASE,
        "simulator_hash": certificates.code_hash(streams),
        "charter_inputs": {
            "reference_risk": R_REF,
            "tolerance": DELTA,
            "alpha_alarm": ALPHA_ALARM,
            "d_max": D_MAX,
            "alpha_promo": ALPHA_PROMO,
            "delta_promo": DELTA_PROMO,
            "alpha_rb": ALPHA_RB,
            "delta_rb": DELTA_RB,
            "r_star": R_STAR,
            "delta": DELTA_CONF,
            "batch_size": BATCH,
        },
        "not_measured": list(NOT_MEASURED),
        "gates": {},
    }
    digests = current_digests()
    for name, (runner, runs) in RUNNERS.items():
        measured = runner(runs, 0)
        measured["code_hash"] = digests[name]
        measured["seeds"] = [0, runs]
        document["gates"][name] = measured
        print(name, measured["passed"], file=sys.stderr)
    return document


def problems(document: dict[str, Any] | None = None) -> list[str]:
    """Why the stored certificates cannot be trusted (empty when current).

    Args:
        document: The certificate document; read from `STORE_PATH` when
            None.

    Returns:
        Messages for a missing file, a changed simulator, a gate without a
        certificate and a gate whose code hash differs from the current
        code.
    """
    if document is None:
        if not STORE_PATH.exists():
            return ["loop_certificates.json is missing"]
        document = json.loads(STORE_PATH.read_text(encoding="utf-8"))
    found = []
    if document.get("simulator_hash") != certificates.code_hash(streams):
        found.append("simulator: stale (streams.py changed since measurement)")
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
    """Gates that missed their floors; their rules stay descriptive.

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
    """The `--check loop` result.

    Returns:
        `ok`, the problems, the descriptive-only gates (empty when there are
        problems) and what was not measured.
    """
    issues = problems()
    return {
        "ok": not issues,
        "problems": issues,
        "descriptive_only": descriptive_only() if not issues else [],
        "not_measured": list(NOT_MEASURED),
    }


def main(arguments: list[str] | None = None) -> int:
    """Writes or checks the certificate file.

    Args:
        arguments: Command-line arguments; `None` reads `sys.argv`.

    Returns:
        0 after `--write`, `--gate`, or a check without problems; 1 when the
        check finds problems.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--gate", choices=GATES)
    parser.add_argument("--runs", type=int, default=0)
    options = parser.parse_args(arguments)
    if options.gate:
        runner, runs = RUNNERS[options.gate]
        print(json.dumps(runner(options.runs or runs, DEV_SEED_BASE), indent=2))
        return 0
    if options.write:
        document = certify_all()
        STORE_PATH.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"passed": passed_gates(document)}))
        return 0
    found = check()
    print(json.dumps(found, indent=2))
    return 0 if found["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
