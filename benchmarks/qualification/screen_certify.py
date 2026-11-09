"""Certificate for the leakage screen with multiplicity control.

The detector is `scmle.data.admissibility.screen` (id `f4-leakage-screen`),
the test that replaced the profiler's fixed cutoffs. It is measured on the
breast-cancer table that ships with scikit-learn, with the settings
`leakage.CERTIFIED`, fixed before measuring and not tuned.

* Clean side: `NULL_REPLICATES` null worlds. Each keeps `TABLE_COLUMNS` real
  feature columns (a random subset), two of them with 20 percent of values
  deleted completely at random, and permutes the outcome, so the whole family
  of tests is run and a false alarm is any suspect in the world. Strong
  legitimate predictors are not on the clean side: the test is expected to
  call those suspects and the certificate says nothing about them. The size
  of the Benjamini-Yekutieli step alone (any column significant) is recorded
  against a 99 percent binomial band in `screen_measurements.json`.
* Plants: `PLANT_REPLICATES` tables per leak type, each the real outcome with
  `TABLE_COLUMNS - 1` real columns and one planted column. A plant counts as
  found when the column is a suspect by the mechanism that type belongs to
  (`screen.MECHANISM_DEFECTS`). Types: a copy of the target, a copy with 2
  percent of labels flipped, a threshold rule with noise, a value recorded
  only when the target is not one class, and a value recorded for a random
  quarter of one class only (so a column recorded for about 9 percent of the
  rows, all of one class).
* The old profiler rule (association at least 0.98, or a missingness score
  of at least 0.9) is scored on the same worlds and plants as a baseline.

Plants are synthetic, so recall is optimistic.

    python -m benchmarks.qualification.screen_certify --write
    python -m benchmarks.qualification.screen_certify --check
"""

import argparse
from collections.abc import Sequence
import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy
import pandas

from benchmarks.qualification import certify
from benchmarks.qualification import f4_certify
from benchmarks.qualification import simulate
from scmle.contracts import qualification
from scmle.contracts import tasks
from scmle.core import randomness
from scmle.data.admissibility import leakage
from scmle.data.admissibility import screen
from scmle.data.detectors import certificates
from scmle.data.validation import columns

STORE_PATH = screen.STORE_PATH
MEASUREMENTS_PATH = Path(__file__).with_name("screen_measurements.json")
TARGET = f4_certify.TARGET
NULL_REPLICATES = 1000
PLANT_REPLICATES = 200
GROUPED_REPLICATES = 200
GROUP_SIZE = 10
TABLE_COLUMNS = 10
GAP_COLUMNS = 2
GAP_RATE = 0.2
RARE_SHARE = 0.25
OLD_ASSOCIATION = 0.98
OLD_MISSINGNESS = 0.9
DEFECTS = tuple(d for ds in screen.MECHANISM_DEFECTS.values() for d in ds)
POLICY = f4_certify.LEAK_POLICY
Z_99 = f4_certify.Z_99


def plant_column(
    frame: pandas.DataFrame, defect: str, seed: int
) -> pandas.Series:
    """The planted leak column for one defect type.

    Args:
        frame: The clean table, including the target.
        defect: A leak type in `DEFECTS`.
        seed: Run seed.

    Returns:
        The planted column, aligned to `frame`; for `leak_rare_record` it is
        standard normal noise recorded only for a random quarter of the rows
        that share the first row's class (missing elsewhere).
    """
    if defect != "leak_rare_record":
        return f4_certify.plant_leak(frame, defect, seed)
    rng = randomness.component_rng(seed, "screen.rare", 0)
    values = pandas.Series(rng.normal(size=len(frame)), index=frame.index)
    first = frame[TARGET] == frame[TARGET].iloc[0]
    kept = first & pandas.Series(
        rng.random(len(frame)) < RARE_SHARE, frame.index
    )
    return values.where(kept)


def real_columns(frame: pandas.DataFrame, count: int, seed: int) -> list[str]:
    """A random subset of the real feature columns.

    Args:
        frame: The table, including the target.
        count: Number of columns to pick.
        seed: Run seed.

    Returns:
        The chosen column names, in table order.
    """
    rng = randomness.component_rng(seed, "screen.columns", 0)
    names = [c for c in frame.columns if c != TARGET]
    return [names[i] for i in sorted(rng.choice(len(names), count, False))]


def with_gaps(features: pandas.DataFrame, seed: int) -> pandas.DataFrame:
    """Deletes values completely at random from the first columns.

    Args:
        features: Feature table.
        seed: Run seed.

    Returns:
        A copy in which `GAP_RATE` of the values of the first `GAP_COLUMNS`
        columns are missing.
    """
    rng = randomness.component_rng(seed, "screen.gaps", 0)
    out = features.copy()
    for name in list(out.columns)[:GAP_COLUMNS]:
        out.loc[rng.random(len(out)) < GAP_RATE, name] = numpy.nan
    return out


def old_rule(frame: pandas.DataFrame, names: Sequence[str]) -> dict[str, bool]:
    """The retired profiler cutoffs, per column.

    Args:
        frame: Table including the target column.
        names: Columns to judge.

    Returns:
        Mapping of column name to True when its association with the target
        is at least `OLD_ASSOCIATION` or its missingness association at
        least `OLD_MISSINGNESS`.
    """
    target = frame[TARGET]
    flagged = {}
    for name in names:
        kind = columns.kind_of(frame[name])
        task = tasks.TaskType.BINARY
        flagged[name] = (
            kind.association(frame[name], target, task) >= OLD_ASSOCIATION
            or kind.missingness_association(frame[name], target, task)
            >= OLD_MISSINGNESS
        )
    return flagged


def run_screen(frame: pandas.DataFrame, seed: int, groups: Any = None) -> Any:
    """Screens every column but the target with the certified settings.

    Args:
        frame: Table including the target column.
        seed: Run seed of the screen.
        groups: Optional row groups for grouped permutation.

    Returns:
        The screen report.
    """
    cfg = dataclasses.replace(leakage.CERTIFIED, run_seed=seed)
    return screen.screen(frame, TARGET, "classification", cfg, groups=groups)


def null_replicate(seed: int, grouped: bool = False) -> dict[str, Any]:
    """One null world: permuted outcome, gaps in two columns.

    Args:
        seed: Replicate seed.
        grouped: Whether rows come in groups of `GROUP_SIZE` that are
            permuted together.

    Returns:
        Whether the screen found any suspect or any significant column, the
        size of the tested family, and whether the old rule flagged any
        column.
    """
    frame = f4_certify.breast_cancer()
    names = real_columns(frame, TABLE_COLUMNS, seed)
    groups = (
        pandas.Series(numpy.arange(len(frame)) // GROUP_SIZE, index=frame.index)
        if grouped
        else None
    )
    world = simulate.null_world(
        frame[names], frame[TARGET], seed, groups=groups
    )
    features = with_gaps(world.features, seed)
    table = features.assign(**{TARGET: world.outcome})
    report = run_screen(table, seed, groups)
    old = old_rule(table, list(features.columns))
    return {
        "suspect": bool(report.suspects),
        "significant": any(f.significant for f in report.findings),
        "family": report.family,
        "old": any(old.values()),
    }


def plant_replicate(defect: str, seed: int) -> dict[str, Any]:
    """One planted leak among real columns.

    Args:
        defect: Leak type to plant.
        seed: Replicate seed.

    Returns:
        Whether the planted column was found by the mechanism of its type,
        whether it was a suspect by any mechanism, and whether the old rule
        flagged it.
    """
    frame = f4_certify.breast_cancer()
    names = real_columns(frame, TABLE_COLUMNS - 1, seed)
    table = frame[[*names, TARGET]].assign(
        **{defect: plant_column(frame, defect, seed)}
    )
    report = run_screen(table, seed)
    mechanisms = {
        m for m, kinds in screen.MECHANISM_DEFECTS.items() if defect in kinds
    }
    found = any(
        f.suspect and f.feature == defect and f.mechanism in mechanisms
        for f in report.findings
    )
    any_mechanism = any(
        f.suspect and f.feature == defect for f in report.findings
    )
    old = old_rule(table, [defect])
    return {
        "found": bool(found),
        "found_any_mechanism": bool(any_mechanism),
        "old": bool(old[defect]),
    }


def replicate(kind: str, seed: int) -> dict[str, Any]:
    """A null (`""`), grouped null (`"grouped"`) or planted replicate.

    Args:
        kind: `""` for a null world, `"grouped"` for a grouped null world,
            otherwise a leak type to plant.
        seed: Replicate seed.

    Returns:
        The result of `null_replicate` or `plant_replicate`.
    """
    if kind == "":
        return null_replicate(seed)
    if kind == "grouped":
        return null_replicate(seed, grouped=True)
    return plant_replicate(kind, seed)


def run_parallel(
    jobs: Sequence[tuple[str, int]], workers: int
) -> list[dict[str, Any]]:
    """Runs the replicates in order.

    Args:
        jobs: (kind, seed) pairs as accepted by `replicate`.
        workers: Number of parallel workers.

    Returns:
        The replicate results in the order of `jobs`.
    """
    from joblib import delayed
    from joblib import Parallel

    return list(
        Parallel(n_jobs=workers)(delayed(replicate)(k, s) for k, s in jobs)
    )


def wilson(successes: int, trials: int) -> tuple[float, float]:
    """95 percent Wilson interval.

    Args:
        successes: Number of successes.
        trials: Number of trials.

    Returns:
        The lower and upper bounds.
    """
    return certificates.wilson_interval(successes, trials, 0.95)


def band_99(alpha: float, runs: int) -> tuple[float, float]:
    """99 percent normal-approximation band for a rate of `alpha`.

    Args:
        alpha: The nominal rate.
        runs: Number of replicates the rate is observed over.

    Returns:
        The lower and upper edges of the band.
    """
    return f4_certify.band_99(alpha, runs)


def certify_screen(workers: int = 1) -> tuple[Any, dict[str, Any]]:
    """Measures the screen on null worlds and planted leaks.

    Args:
        workers: Number of parallel workers.

    Returns:
        (certificate, measurements).
    """
    frame = f4_certify.breast_cancer()
    jobs: list[tuple[str, int]] = [("", s) for s in range(NULL_REPLICATES)]
    jobs += [("grouped", s) for s in range(GROUPED_REPLICATES)]
    for defect in DEFECTS:
        jobs += [(defect, s) for s in range(PLANT_REPLICATES)]
    outcomes = run_parallel(jobs, workers)
    nulls = outcomes[:NULL_REPLICATES]
    grouped = outcomes[NULL_REPLICATES : NULL_REPLICATES + GROUPED_REPLICATES]
    start = NULL_REPLICATES + GROUPED_REPLICATES
    false_alarms = sum(item["suspect"] for item in nulls)
    results = []
    recall: dict[str, Any] = {}
    for k, defect in enumerate(DEFECTS):
        part = outcomes[start + k * PLANT_REPLICATES :][:PLANT_REPLICATES]
        detected = sum(item["found"] for item in part)
        old_detected = sum(item["old"] for item in part)
        low, high = wilson(detected, PLANT_REPLICATES)
        old_low, old_high = wilson(old_detected, PLANT_REPLICATES)
        recall[defect] = {
            "detected": detected,
            "wilson_95": [low, high],
            "detected_any_mechanism": sum(
                item["found_any_mechanism"] for item in part
            ),
            "old_rule_detected": old_detected,
            "old_rule_wilson_95": [old_low, old_high],
        }
        results.append(
            qualification.Certificate(
                detector_id=screen.DETECTOR_ID,
                defect_type=defect,
                planted=PLANT_REPLICATES,
                detected=detected,
                clean_rows=NULL_REPLICATES,
                false_alarms=false_alarms,
                seed_path=randomness.seed_path(0, "screen.certify", 0),
            )
        )
    cert = certificates.certify(
        screen.DETECTOR_ID,
        screen.code_digest(),
        certificates.corpus_hash([certify.frame_digest(frame)]),
        [0],
        results,
        POLICY,
    )
    alpha = leakage.CERTIFIED.alpha
    significant = sum(item["significant"] for item in nulls)
    old_false = sum(item["old"] for item in nulls)
    low, high = band_99(alpha, NULL_REPLICATES)
    measurements = {
        "alpha": alpha,
        "rule": "benjamini-yekutieli",
        "null_replicates": NULL_REPLICATES,
        "plant_replicates_per_type": PLANT_REPLICATES,
        "table_columns": TABLE_COLUMNS,
        "family_sizes": sorted({item["family"] for item in nulls}),
        "suspect_false_alarms": false_alarms,
        "suspect_false_alarm_wilson_95": list(
            wilson(false_alarms, NULL_REPLICATES)
        ),
        "worlds_with_a_significant_column": significant,
        "significant_rate": significant / NULL_REPLICATES,
        "band_99": [low, high],
        "significant_rate_within_alpha": (
            significant / NULL_REPLICATES <= high
        ),
        "grouped_null_replicates": GROUPED_REPLICATES,
        "grouped_suspect_false_alarms": sum(i["suspect"] for i in grouped),
        "grouped_significant_worlds": sum(i["significant"] for i in grouped),
        "old_rule_false_alarms": old_false,
        "old_rule_false_alarm_wilson_95": list(
            wilson(old_false, NULL_REPLICATES)
        ),
        "recall": recall,
        "settings": dataclasses.asdict(leakage.CERTIFIED),
    }
    return cert, measurements


def problems(store: certificates.CertificateStore) -> list[str]:
    """Why the certificate file is unusable: missing or stale entries.

    Args:
        store: The loaded certificate store.

    Returns:
        A message per leak type whose certificate is missing or stale; empty
        when all are current.
    """
    found = []
    digest = screen.code_digest()
    for defect in DEFECTS:
        report = certificates.audit(
            store, {screen.DETECTOR_ID: (digest, defect)}
        )
        if report.missing:
            found.append(f"{screen.DETECTOR_ID}/{defect}: missing")
        if report.stale:
            found.append(f"{screen.DETECTOR_ID}/{defect}: stale")
    return found


def uncertified(store: certificates.CertificateStore) -> list[str]:
    """Defect types that were measured and did not meet the policy.

    Args:
        store: The loaded certificate store.

    Returns:
        The leak types whose status is not gating.
    """
    digest = screen.code_digest()
    return [
        defect
        for defect in DEFECTS
        if store.status(screen.DETECTOR_ID, digest, defect)
        is not certificates.GateStatus.GATING
    ]


def main(arguments: list[str] | None = None) -> int:
    """Measures (`--write`) or verifies (`--check`) the screen certificate.

    Args:
        arguments: Command-line arguments; `None` reads `sys.argv`.

    Returns:
        With `--check`, 1 if any certificate is missing or stale, otherwise
        0; with `--write`, always 0.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true")
    action.add_argument("--check", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    options = parser.parse_args(arguments)
    if options.check:
        store = certificates.CertificateStore.load(STORE_PATH)
        found = problems(store)
        print(
            json.dumps(
                {
                    "ok": not found,
                    "problems": found,
                    "descriptive_only": uncertified(store),
                },
                indent=2,
            )
        )
        return 1 if found else 0
    cert, measurements = certify_screen(options.workers)
    certificates.CertificateStore([cert]).save(STORE_PATH)
    MEASUREMENTS_PATH.write_text(
        json.dumps(measurements, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(measurements, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
