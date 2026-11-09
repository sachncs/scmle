"""Measured gate for the Holm family of subgroup hypotheses.

Registered on 2026-11-06, before the first run of this module. Every floor,
seed and design choice below is fixed here and is not retuned: a gate that
misses is filed with its numbers and `passed` false, and subgroup results are
then reported `descriptive`. The gate decides only the word the report
carries (`gating` or `descriptive`) for `identification_report.json` and
`causal_claims.json`. The subgroup family never changes the tier: the tier
rule (`identification.tier.decide`) has no subgroup input and none is added.

The method is `run.subgroup_report`: for each declared (column, level) the
strategy's estimator on the rows of `C` in the level, a Wald test of zero,
Holm's step-down at the primary level `alpha = 0.05` over the whole declared
family (a level that cannot be estimated stays in at p = 1). Its code hash
(with `battery.wald_test` and `multiplicity.adjust`) is the certificate's
`method_hash`.

World. A randomized experiment (strategy `randomized_diff`, assignment
probability 1/2) on `C` with a declared column `g` of `LEVELS = 8` levels
`g0 .. g7`, each with exactly `rows_per_level` rows; outcome
`y = tau_g * t + N(0, 1)`. Replicate `i` draws from
`numpy.random.default_rng([base + i, which])`; certification uses
`base = 0` and seeds `0 .. replicates - 1`, development looks use
`DEV_SEED_BASE`, which never overlaps.

Gates (`R` replicates):

* Null world: `tau_g = 0` for all eight levels, `rows_per_level = 200`,
  `R = 2000`. Family-wise error rate (any level rejected) `<= 0.05 + 2 sqrt(
  0.05 * 0.95 / R)`.
* Partial null: `tau_g0 = tau_g1 = 0.5`, the other six zero,
  `rows_per_level = 200`, `R = 2000`. Probability that any of the six true
  nulls is rejected `<= 0.05 + 2 sqrt(0.05 * 0.95 / R)`.
* Power: one planted effect `tau_g0 = 0.5` (half the outcome standard
  deviation), `rows_per_level` in 100, 200, 400, `R = 500` each. Power is the
  share of replicates in which `g0` is rejected. Power is certified from the
  smallest tested size whose Wilson 95 percent lower bound is `>= 0.80`;
  `min_certified_rows` is that size. Other effect sizes (0.3, 0.8) are filed
  and not gated.
* Pass: both error rates within their bounds and at least one size
  certified. A report calls the family `gating` only when `passed`; a
  subgroup with fewer than `min_certified_rows` rows is described as below
  the certified size.

Worlds are simulated with independent homoscedastic normal noise and equal
level sizes; real subgroups are unequal, correlated with covariates and not
randomized by level, so the numbers are optimistic. NOT measured: unequal
levels, outcome models other than the difference in means (the Wald test of
the other strategies rests on their own gates), overlapping subgroup columns,
and data of a real trial.

    python -m benchmarks.qualification.subgroup_certify --write
    python -m benchmarks.qualification.subgroup_certify --check
"""

import argparse
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy
import pandas

from scmle.data.charter import causal
from scmle.data.detectors import certificates
from scmle.data.identification import certified
from scmle.data.identification import run

STORE_PATH = Path(__file__).with_name("subgroup_certificates.json")
DEV_SEED_BASE = 10_000
LEVELS = 8
LEVEL = 0.95
ALPHA = 1.0 - LEVEL
NULL_REPLICATES = 2000
POWER_REPLICATES = 500
EFFECT = 0.5
POWER_FLOOR = 0.80
POWER_SIZES = (100, 200, 400)
FILED_EFFECTS = (0.3, 0.8)
NOT_MEASURED = (
    "unequal subgroup sizes",
    "strategies other than the randomized difference in means",
    "overlapping subgroup columns",
    "a real trial's subgroups",
)


def stream(base: int, index: int, which: int = 0) -> numpy.random.Generator:
    """Independent random stream for one replicate.

    Args:
        base: Seed offset of the run (0 for certification).
        index: Replicate number.
        which: Distinguishes the designs drawn within one replicate.

    Returns:
        A generator seeded with `[base + index, which]`.
    """
    return numpy.random.default_rng([base + index, which])


def names() -> tuple[str, ...]:
    """The declared levels.

    Returns:
        The level names `g0` to `g7`.
    """
    return tuple(f"g{k}" for k in range(LEVELS))


def world(
    rng: numpy.random.Generator, rows_per_level: int, effects: dict[str, float]
) -> pandas.DataFrame:
    """A randomized experiment whose effect differs by level of `g`.

    Args:
        rng: Random generator.
        rows_per_level: Rows in each of the levels.
        effects: Treatment effect by level name; levels not listed have none.

    Returns:
        A frame with the level `g`, a treatment `t` assigned with probability
        1/2 and the outcome `y` (effect times treatment plus standard normal
        noise).
    """
    group = numpy.repeat(names(), rows_per_level)
    tau = numpy.array([effects.get(g, 0.0) for g in group])
    t = (rng.uniform(size=len(group)) < 0.5).astype(float)
    y = tau * t + rng.standard_normal(len(group))
    return pandas.DataFrame({"g": group, "t": t, "y": y})


def adapter_for() -> run.Adapter:
    """The randomized-difference adapter the report is run with.

    Returns:
        An adapter for the `randomized_diff` strategy on a randomized
        treatment `t` and outcome `y`.
    """
    spec = causal.CausalSpec(
        treatment="t",
        outcome="y",
        estimand="ATE",
        assumptions=("randomized_assignment", "no_interference"),
        strategies=("randomized_diff",),
    )
    return run.ADAPTERS["randomized_diff"](
        spec, pandas.DataFrame({"t": [0.0], "y": [0.0]}), 0, "linear", 0.01
    )


def rejected_levels(frame: pandas.DataFrame, adapter: run.Adapter) -> set[str]:
    """Levels the Holm family rejects on one replicate.

    Args:
        frame: One replicate's data.
        adapter: The estimation adapter.

    Returns:
        The level names whose hypothesis the subgroup report rejects.
    """
    found = run.subgroup_report(
        adapter, frame, LEVEL, (causal.Subgroup("g", names()),)
    )
    return {r["value"] for r in found["hypotheses"] if r["rejected"]}


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


def bound_for(replicates: int) -> float:
    """The pre-registered family-wise bound at this replicate count.

    Args:
        replicates: Number of replicates in the gate.

    Returns:
        `0.05 + 2 sqrt(0.05 * 0.95 / replicates)`.
    """
    return ALPHA + 2 * math.sqrt(ALPHA * (1 - ALPHA) / replicates)


def gate_subgroup_holm(replicates: int, base: int = 0) -> dict[str, Any]:
    """Family-wise error under two nulls and power by subgroup size.

    Args:
        replicates: Replicates for the two error-rate designs; the power
            designs use a quarter of them, at least 100.
        base: Seed offset (0 for certification).

    Returns:
        The measured rates with Wilson intervals, the bound, the power by
        subgroup size, the effects filed without gating, the smallest size
        whose power lower bound reaches the floor (None if none does), and
        whether the gate passed.
    """
    adapter = adapter_for()
    bound = bound_for(replicates)
    null_hits = sum(
        bool(rejected_levels(world(stream(base, i), 200, {}), adapter))
        for i in range(replicates)
    )
    partial = {"g0": EFFECT, "g1": EFFECT}
    true_nulls = set(names()) - set(partial)
    partial_hits = sum(
        bool(
            rejected_levels(world(stream(base, i, 1), 200, partial), adapter)
            & true_nulls
        )
        for i in range(replicates)
    )
    runs = max(replicates // 4, 100)
    power: dict[str, Any] = {}
    for size in POWER_SIZES:
        hits = sum(
            "g0"
            in rejected_levels(
                world(stream(base, i, 2), size, {"g0": EFFECT}), adapter
            )
            for i in range(runs)
        )
        power[str(size)] = rate(hits, runs)
    filed: dict[str, Any] = {}
    for effect in FILED_EFFECTS:
        filed[str(effect)] = {
            str(size): rate(
                sum(
                    "g0"
                    in rejected_levels(
                        world(stream(base, i, 3), size, {"g0": effect}),
                        adapter,
                    )
                    for i in range(runs)
                ),
                runs,
            )
            for size in POWER_SIZES
        }
    sizes = [int(k) for k, v in power.items() if v["wilson"][0] >= POWER_FLOOR]
    null_ok = null_hits / replicates <= bound
    partial_ok = partial_hits / replicates <= bound
    return {
        "replicates": replicates,
        "power_replicates": runs,
        "alpha": ALPHA,
        "familywise_error_null": rate(null_hits, replicates),
        "familywise_error_partial_null": rate(partial_hits, replicates),
        "familywise_bound": bound,
        "power_at_effect_half_sd": power,
        "power_other_effects_filed": filed,
        "min_certified_rows": min(sizes) if sizes else None,
        "passed": bool(null_ok and partial_ok and sizes),
    }


def current_method_hash() -> str:
    """Hash of the package code the gate measures.

    Returns:
        The method hash of `run.subgroup_report` and the code it depends on.
    """
    return certified.subgroup_method_hash(run.subgroup_report)


def certify_all(base: int = 0, scale: float = 1.0) -> dict[str, Any]:
    """Runs the gate; `scale` shrinks replicate counts for dev looks.

    Args:
        base: Seed offset (0 for certification, `DEV_SEED_BASE` for looks).
        scale: Fraction of `NULL_REPLICATES` to run, with a floor of 40.

    Returns:
        The document written to the certificate file: version, development
        seed base, what was not measured and the measured gate with its
        hashes, seed range and duration.
    """
    started = time.monotonic()
    count = max(40, int(NULL_REPLICATES * scale))
    measured = gate_subgroup_holm(count, base)
    measured["method_hash"] = current_method_hash()
    measured["code_hash"] = certificates.code_hash(
        gate_subgroup_holm, world, rejected_levels, adapter_for
    )
    measured["seeds"] = [base, base + count]
    measured["seconds"] = round(time.monotonic() - started, 1)
    print(certified.SUBGROUP_GATE, measured["passed"], file=sys.stderr)
    return {
        "version": 1,
        "dev_seed_base": DEV_SEED_BASE,
        "not_measured": list(NOT_MEASURED),
        "gates": {certified.SUBGROUP_GATE: measured},
    }


def problems(document: dict[str, Any] | None = None) -> list[str]:
    """Why the stored certificate cannot be trusted (empty when current).

    Args:
        document: The certificate document; read from `STORE_PATH` when
            None.

    Returns:
        Messages for a missing file or entry, a changed method hash and a
        changed gate hash; empty when the certificate is current.
    """
    if document is None:
        if not STORE_PATH.exists():
            return ["subgroup_certificates.json is missing"]
        document = json.loads(STORE_PATH.read_text(encoding="utf-8"))
    entry = document.get("gates", {}).get(certified.SUBGROUP_GATE)
    if entry is None:
        return [f"{certified.SUBGROUP_GATE}: no certificate"]
    found = []
    if entry.get("method_hash") != current_method_hash():
        found.append(
            f"{certified.SUBGROUP_GATE}: "
            "stale (method code changed since measurement)"
        )
    gate_hash = certificates.code_hash(
        gate_subgroup_holm, world, rejected_levels, adapter_for
    )
    if entry.get("code_hash") != gate_hash:
        found.append(
            f"{certified.SUBGROUP_GATE}: "
            "stale (gate code changed since measurement)"
        )
    return found


def descriptive_only(document: dict[str, Any] | None = None) -> list[str]:
    """Gates that missed their floors; subgroup results stay descriptive.

    Args:
        document: The certificate document; read from `STORE_PATH` when
            None.

    Returns:
        The names of gates that did not pass; the gate itself when the file
        is missing.
    """
    if document is None:
        if not STORE_PATH.exists():
            return [certified.SUBGROUP_GATE]
        document = json.loads(STORE_PATH.read_text(encoding="utf-8"))
    return [k for k, v in document["gates"].items() if not v["passed"]]


def main(arguments: list[str] | None = None) -> int:
    """Writes (`--write`), checks (`--check`) or looks (`--dev`).

    Args:
        arguments: Command-line arguments; `None` reads `sys.argv`.

    Returns:
        0 for `--dev` and `--write`; for `--check`, 0 when the certificate is
        current and 1 otherwise.
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
        print(json.dumps({"passed": not descriptive_only(document)}))
        return 0
    issues = problems()
    print(json.dumps({"ok": not issues, "problems": issues}, indent=2))
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
