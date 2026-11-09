"""Subgroup family under unequal levels and overlapping columns.

Registered on 2026-11-07, before the first run of this module. Every design,
seed and floor below is fixed here and is not retuned: a design that misses
its floor is filed with its numbers and `passed` false.

`subgroup_certify.py` measures the Holm family of `run.subgroup_report` on
eight equal levels of one column and lists unequal levels and overlapping
columns as not measured. This module measures them. It changes nothing in
that certificate: the word `gating` or `descriptive` that a report carries
still follows `subgroup_certificates.json`, and these results are filed in
`subgroup_extended_measurements.json` as a description of what was measured
beyond it. The method, the world generator and the replicate streams are the
same ones (`numpy.random.default_rng([base + i, which])`, certification
`base = 0`, development `DEV_SEED_BASE`).

Designs (`R = 2000` replicates unless stated; `bound = 0.05 + 2 sqrt(0.05 *
0.95 / R)`, the family-wise bound of the equal-level gate):

* Unequal null: eight levels of one column with `UNEQUAL_SIZES = (20, 30, 50,
  80, 120, 200, 300, 500)` rows, no effect anywhere. Probability that any
  level is rejected `<= bound`.
* Unequal partial null: `tau = 0.5` in the levels with 200 and 500 rows, zero
  elsewhere. Probability that any of the six true nulls is rejected `<=
  bound`.
* Unequal power (`R = 500` each): one planted effect of 0.5 in a level of
  `n` rows, `n` in 50, 100, 200, 400, the other seven levels keeping the sizes
  `UNEQUAL_SIZES[:7]`. Filed with Wilson intervals; the smallest `n` whose
  Wilson lower bound is `>= 0.80` is reported as `min_rows_power_80`, and it
  does not gate.
* Overlapping null: two declared columns `g` and `h` of four levels each,
  every row carrying one level of each drawn independently and uniformly,
  800 rows per replicate, one Holm family of the eight hypotheses, no effect.
  Probability that any hypothesis is rejected `<= bound`.
* Overlapping partial null: the effect is 0.5 for rows with `g == g0` and zero
  otherwise. The hypotheses `g1`, `g2`, `g3` are true nulls (their subgroups
  have mean effect zero); the `h` levels each hold a quarter of the `g0` rows
  and so have a nonzero mean effect and are not counted. Probability that any
  of `g1`, `g2`, `g3` is rejected `<= bound`.

Pass: all four error rates within `bound`. Holm controls the family-wise
error rate under any dependence, so the overlapping designs are expected to
pass; the measurement says whether the implementation does.

NOT measured: strategies other than the randomized difference in means,
heteroscedastic or heavy-tailed outcomes, non-randomized assignment, and the
subgroups of a real trial.

    python -m benchmarks.qualification.subgroup_extended --write
    python -m benchmarks.qualification.subgroup_extended --check
"""

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy
import pandas

from benchmarks.qualification import subgroup_certify
from scmle.data.charter import causal
from scmle.data.detectors import certificates
from scmle.data.identification import run

MEASUREMENTS_PATH = Path(__file__).with_name(
    "subgroup_extended_measurements.json"
)
UNEQUAL_SIZES = (20, 30, 50, 80, 120, 200, 300, 500)
POWER_SIZES = (50, 100, 200, 400)
# share of rows treated
TREATED_SHARE = 0.5
OVERLAP_LEVELS = 4
OVERLAP_ROWS = 800
EFFECT = 0.5
POWER_FLOOR = 0.80
NULL_REPLICATES = 2000
POWER_REPLICATES = 500


def unequal_world(
    rng: numpy.random.Generator,
    sizes: tuple[int, ...],
    effects: dict[str, float],
) -> pandas.DataFrame:
    """A randomized experiment with levels of different sizes.

    Args:
        rng: Random generator.
        sizes: Rows in each level, in level order.
        effects: Treatment effect by level name; levels not listed have none.

    Returns:
        A frame with the level `g`, a treatment `t` assigned with probability
        1/2 and the outcome `y`.
    """
    levels = subgroup_certify.names()
    group = numpy.concatenate(
        [
            numpy.repeat(level, size)
            for level, size in zip(levels, sizes, strict=True)
        ]
    )
    tau = numpy.array([effects.get(g, 0.0) for g in group])
    t = (rng.uniform(size=len(group)) < TREATED_SHARE).astype(float)
    y = tau * t + rng.standard_normal(len(group))
    return pandas.DataFrame({"g": group, "t": t, "y": y})


def overlap_names(prefix: str) -> tuple[str, ...]:
    """The declared levels of one overlapping column.

    Args:
        prefix: Letter that starts each level name.

    Returns:
        The level names, such as `g0` to `g3`.
    """
    return tuple(f"{prefix}{k}" for k in range(OVERLAP_LEVELS))


def overlap_world(
    rng: numpy.random.Generator, effect_g0: float
) -> pandas.DataFrame:
    """Two crossed columns; the effect is carried by `g == g0` rows.

    Args:
        rng: Random generator.
        effect_g0: Treatment effect for rows with `g == g0`.

    Returns:
        A frame of `OVERLAP_ROWS` rows with the columns `g` and `h` (levels
        drawn independently and uniformly), treatment `t` and outcome `y`.
    """
    g = rng.integers(0, OVERLAP_LEVELS, OVERLAP_ROWS)
    h = rng.integers(0, OVERLAP_LEVELS, OVERLAP_ROWS)
    tau = numpy.where(g == 0, effect_g0, 0.0)
    t = (rng.uniform(size=OVERLAP_ROWS) < TREATED_SHARE).astype(float)
    y = tau * t + rng.standard_normal(OVERLAP_ROWS)
    return pandas.DataFrame(
        {
            "g": [f"g{k}" for k in g],
            "h": [f"h{k}" for k in h],
            "t": t,
            "y": y,
        }
    )


def rejected(
    frame: pandas.DataFrame,
    adapter: run.Estimator,
    groups: tuple[causal.Subgroup, ...],
) -> set[tuple[str, str]]:
    """(column, level) pairs the Holm family rejects on one replicate.

    Args:
        frame: One replicate's data.
        adapter: The estimation adapter.
        groups: The declared subgroup columns.

    Returns:
        The pairs whose hypothesis the subgroup report rejects.
    """
    found = run.subgroup_report(adapter, frame, subgroup_certify.LEVEL, groups)
    return {
        (r["column"], r["value"]) for r in found["hypotheses"] if r["rejected"]
    }


def any_rejected(
    pairs: set[tuple[str, str]], among: set[tuple[str, str]] | None
) -> bool:
    """Whether a rejection falls among `among` (all when None).

    Args:
        pairs: The rejected (column, level) pairs.
        among: The pairs of interest, or None for all.

    Returns:
        True if any rejected pair is among those of interest.
    """
    return bool(pairs if among is None else pairs & among)


def gate_unequal(replicates: int, base: int) -> dict[str, Any]:
    """Family-wise error and power with unequal level sizes.

    Args:
        replicates: Replicates for the error-rate designs; the power design
            uses a quarter of them, at least 100.
        base: Seed offset.

    Returns:
        The sizes used, the two family-wise rates with Wilson intervals, the
        bound, the power by planted level size, the smallest size whose
        lower bound reaches the floor (None if none) and whether both error
        rates are within the bound.
    """
    adapter = subgroup_certify.adapter_for()
    groups = (causal.Subgroup("g", subgroup_certify.names()),)
    bound = subgroup_certify.bound_for(replicates)
    sizes = UNEQUAL_SIZES
    null_hits = sum(
        bool(
            rejected(
                unequal_world(subgroup_certify.stream(base, i), sizes, {}),
                adapter,
                groups,
            )
        )
        for i in range(replicates)
    )
    planted = {"g5": EFFECT, "g7": EFFECT}
    true_nulls = {("g", g) for g in subgroup_certify.names()} - {
        ("g", "g5"),
        ("g", "g7"),
    }
    partial_hits = sum(
        any_rejected(
            rejected(
                unequal_world(
                    subgroup_certify.stream(base, i, 1), sizes, planted
                ),
                adapter,
                groups,
            ),
            true_nulls,
        )
        for i in range(replicates)
    )
    runs = max(replicates // 4, 100)
    power: dict[str, Any] = {}
    for rows in POWER_SIZES:
        layout = (rows, *sizes[:7])
        hits = sum(
            ("g", "g0")
            in rejected(
                unequal_world(
                    subgroup_certify.stream(base, i, 2), layout, {"g0": EFFECT}
                ),
                adapter,
                groups,
            )
            for i in range(runs)
        )
        power[str(rows)] = subgroup_certify.rate(hits, runs)
    enough = [int(k) for k, v in power.items() if v["wilson"][0] >= POWER_FLOOR]
    return {
        "sizes": list(sizes),
        "replicates": replicates,
        "power_replicates": runs,
        "familywise_error_null": subgroup_certify.rate(null_hits, replicates),
        "familywise_error_partial_null": subgroup_certify.rate(
            partial_hits, replicates
        ),
        "familywise_bound": bound,
        "power_at_effect_half_sd": power,
        "min_rows_power_80": min(enough) if enough else None,
        "passed": bool(
            null_hits / replicates <= bound
            and partial_hits / replicates <= bound
        ),
    }


def gate_overlap(replicates: int, base: int) -> dict[str, Any]:
    """Family-wise error with two overlapping declared columns.

    Args:
        replicates: Number of replicates.
        base: Seed offset.

    Returns:
        The two family-wise rates with Wilson intervals, the bound, the
        design sizes and whether both rates are within the bound.
    """
    adapter = subgroup_certify.adapter_for()
    groups = (
        causal.Subgroup("g", overlap_names("g")),
        causal.Subgroup("h", overlap_names("h")),
    )
    bound = subgroup_certify.bound_for(replicates)
    null_hits = sum(
        bool(
            rejected(
                overlap_world(subgroup_certify.stream(base, i, 3), 0.0),
                adapter,
                groups,
            )
        )
        for i in range(replicates)
    )
    true_nulls = {("g", "g1"), ("g", "g2"), ("g", "g3")}
    partial_hits = sum(
        any_rejected(
            rejected(
                overlap_world(subgroup_certify.stream(base, i, 4), EFFECT),
                adapter,
                groups,
            ),
            true_nulls,
        )
        for i in range(replicates)
    )
    return {
        "rows": OVERLAP_ROWS,
        "levels_per_column": OVERLAP_LEVELS,
        "replicates": replicates,
        "familywise_error_null": subgroup_certify.rate(null_hits, replicates),
        "familywise_error_true_nulls_with_effect_in_g0": subgroup_certify.rate(
            partial_hits, replicates
        ),
        "familywise_bound": bound,
        "passed": bool(
            null_hits / replicates <= bound
            and partial_hits / replicates <= bound
        ),
    }


def measure_all(base: int = 0, scale: float = 1.0) -> dict[str, Any]:
    """Runs every design; `scale` shrinks replicate counts for dev looks.

    Args:
        base: Seed offset (0 for the filed measurement).
        scale: Fraction of `NULL_REPLICATES` to run, with a floor of 40.

    Returns:
        The document written to the measurements file: both designs, what was
        not measured, the code hash and the duration.
    """
    started = time.monotonic()
    count = max(40, int(NULL_REPLICATES * scale))
    document = {
        "version": 1,
        "registered": "2026-11-07",
        "note": "description beside subgroup_certificates.json; it does not "
        "change the gate that labels a report gating or descriptive",
        "seeds": [base, base + count],
        "unequal_levels": gate_unequal(count, base),
        "overlapping_columns": gate_overlap(count, base),
        "not_measured": [
            "strategies other than the randomized difference in means",
            "heteroscedastic or heavy-tailed outcomes",
            "non-randomized assignment",
            "a real trial's subgroups",
        ],
    }
    document["code_hash"] = certificates.code_hash(
        gate_unequal,
        gate_overlap,
        unequal_world,
        overlap_world,
        rejected,
    )
    document["seconds"] = round(time.monotonic() - started, 1)
    print("subgroup_extended", file=sys.stderr)
    return document


def problems(document: dict[str, Any] | None = None) -> list[str]:
    """Why the stored measurements are not for the current code.

    Args:
        document: The measurements document; read from `MEASUREMENTS_PATH`
            when None.

    Returns:
        A message for a missing file or a code hash that differs from the
        current code; empty when current.
    """
    if document is None:
        if not MEASUREMENTS_PATH.exists():
            return ["subgroup_extended_measurements.json is missing"]
        document = json.loads(MEASUREMENTS_PATH.read_text(encoding="utf-8"))
    current = certificates.code_hash(
        gate_unequal,
        gate_overlap,
        unequal_world,
        overlap_world,
        rejected,
    )
    if document.get("code_hash") != current:
        return ["subgroup_extended: measurements are for other code"]
    return []


def main(arguments: list[str] | None = None) -> int:
    """Writes (`--write`), checks (`--check`) or looks (`--dev`).

    Args:
        arguments: Command-line arguments; `None` reads `sys.argv`.

    Returns:
        0 for `--dev` and `--write`; for `--check`, 0 when the measurements
        are for the current code and 1 otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--dev", action="store_true")
    parser.add_argument("--scale", type=float, default=0.05)
    options = parser.parse_args(arguments)
    if options.dev:
        document = measure_all(subgroup_certify.DEV_SEED_BASE, options.scale)
        print(json.dumps(document, indent=2, sort_keys=True, default=float))
        return 0
    if options.write:
        document = measure_all()
        MEASUREMENTS_PATH.write_text(
            json.dumps(document, indent=2, sort_keys=True, default=float)
            + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "unequal": document["unequal_levels"]["passed"],
                    "overlapping": document["overlapping_columns"]["passed"],
                }
            )
        )
        return 0
    issues = problems()
    print(json.dumps({"ok": not issues, "problems": issues}, indent=2))
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
