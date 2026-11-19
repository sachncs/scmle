"""Certificate for the leakage screen with multiplicity control.

The detector is `scmle.assurance.admissibility.screen`, the test that replaced
the profiler's fixed cutoffs. It is measured once per kind of target, each on a
real table that ships with scikit-learn: two classes on breast cancer (id
`f4-leakage-screen`), ten classes on digits (`-multiclass`) and a continuous
target on diabetes (`-regression`), with the settings `leakage.CERTIFIED`, fixed
before measuring and not tuned.

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
  (`screen.MECHANISM_DEFECTS`). The real columns keep the real outcome, so
  strong legitimate predictors are here: any of them flagged is a false
  alarm (`alarmed`).
  The plants are the product's own
  (`scmle.assurance.admissibility.plants`), the same every run probes for: a
  copy of the target, a noisy copy, a field derived from the target with noise,
  a value recorded only for part of the target's range, a value recorded for
  about 9 percent of the rows, all of one class or band, a target mean over
  groups of five rows (a target encoding fitted on all rows), and three in ten
  rows copied (`plants.copy`; found when the duplication test names the rows).
* False alarms are counted over every world, null, grouped null and
  planted: a world is a false alarm when any column other than its planted
  one is a suspect. The certificate's `clean` is the number of worlds and
  `alarms` the number with a false alarm, so a screen that flags legitimate
  columns next to a plant cannot pass on permuted nulls alone.
* The old profiler rule (association at least 0.98, or a missingness score
  of at least 0.9) is scored on the same worlds and plants as a baseline.

Plants are synthetic, so recall is optimistic.

The dated check (`screen.dated`, id `f4-leakage-dated`) reads a declared
prediction time rather than the outcome, so it has its own certificate,
keyed by its own code (`screen.dated_digest`). On each of the three tables,
`DATED_REPLICATES` worlds plant a closing date as the C2 evidence does
(`benchmarks.evidence.leakage`: the cutoff `START` plus `SHIFT` days times
the target's rank plus exponential noise of mean `JITTER`), which must be
found, and as many clean worlds plant an opening date with the same trend
before the cutoff, a legitimate trend no column may be flagged for. The
check is a reading of the declaration, not a test, so the certificate
confirms the code on real tables; it is no estimate of power.

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
from sklearn.datasets import load_diabetes
from sklearn.datasets import load_digits

import benchmarks.evidence.leakage
from benchmarks.qualification import certify
from benchmarks.qualification import f4_certify
from benchmarks.qualification import simulate
from scmle.assurance.admissibility import leakage
from scmle.assurance.admissibility import plants
from scmle.assurance.admissibility import screen
from scmle.assurance.detectors import certificates
from scmle.contracts import qualification
from scmle.contracts import tasks
from scmle.data.validation import columns
from scmle.runtime import randomness

MEASUREMENTS_PATH = Path(__file__).with_name("screen_measurements.json")
NULL_REPLICATES = 1000
PLANT_REPLICATES = 200
GROUPED_REPLICATES = 200
GROUP_SIZE = 10
TABLE_COLUMNS = 10
GAP_COLUMNS = 2
GAP_RATE = 0.2
OLD_ASSOCIATION = 0.98
OLD_MISSINGNESS = 0.9
DEFECTS = tuple(d for ds in screen.MECHANISM_DEFECTS.values() for d in ds)
DATED_REPLICATES = 200


def digits() -> pandas.DataFrame:
    """The scikit-learn digits table: ten classes, 1,797 rows.

    Returns:
        The pixel columns that vary, and `target`.
    """
    frame: pandas.DataFrame = load_digits(as_frame=True).frame
    spread = frame.drop(columns=f4_certify.TARGET).std() > 0
    return frame[[*spread[spread].index, f4_certify.TARGET]]


def diabetes() -> pandas.DataFrame:
    """The scikit-learn diabetes table: a continuous target, 442 rows.

    Returns:
        Its ten features and `target`.
    """
    frame: pandas.DataFrame = load_diabetes(as_frame=True).frame
    return frame


TABLES = {
    "binary": (f4_certify.breast_cancer, "classification", tasks.Task.BINARY),
    "multiclass": (digits, "classification", tasks.Task.MULTICLASS),
    "regression": (diabetes, "regression", tasks.Task.REGRESSION),
}
KINDS = tuple(TABLES)


def plant_column(
    frame: pandas.DataFrame, defect: str, seed: int, kind: str = "binary"
) -> pandas.Series:
    """The planted leak column for one defect type.

    The product's own plants (`scmle.assurance.admissibility.plants`), so the
    certificate measures the leaks every run probes for.

    Args:
        frame: The clean table, including the target.
        defect: A leak type in `DEFECTS`.
        seed: Run seed.
        kind: `binary`, `multiclass` or `regression`.

    Returns:
        The planted column, aligned to `frame`.
    """
    return plants.plant(
        frame[f4_certify.TARGET], defect, seed, kind == "regression"
    )


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
    names = [c for c in frame.columns if c != f4_certify.TARGET]
    return [
        names[i] for i in sorted(rng.choice(len(names), count, replace=False))
    ]


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


def old_rule(
    frame: pandas.DataFrame, names: Sequence[str], kind: str = "binary"
) -> dict[str, bool]:
    """The retired profiler cutoffs, per column.

    Args:
        frame: Table including the target column.
        names: Columns to judge.
        kind: `binary`, `multiclass` or `regression`.

    Returns:
        Mapping of column name to True when its association with the target
        is at least `OLD_ASSOCIATION` or its missingness association at
        least `OLD_MISSINGNESS`.
    """
    target = frame[f4_certify.TARGET]
    task = TABLES[kind][2]
    flagged = {}
    for name in names:
        column = columns.categorize(frame[name])
        flagged[name] = (
            column.association(frame[name], target, task) >= OLD_ASSOCIATION
            or column.discriminate(frame[name], target, task) >= OLD_MISSINGNESS
        )
    return flagged


def run_screen(
    frame: pandas.DataFrame, seed: int, groups: Any = None, kind: str = "binary"
) -> Any:
    """Screens every column but the target with the certified settings.

    Args:
        frame: Table including the target column.
        seed: Run seed of the screen.
        groups: Optional row groups for grouped permutation.
        kind: `binary`, `multiclass` or `regression`.

    Returns:
        The screen report.
    """
    cfg = dataclasses.replace(leakage.CERTIFIED, seed=seed)
    return screen.screen(
        frame, f4_certify.TARGET, TABLES[kind][1], cfg, groups=groups
    )


def alarmed(findings: Sequence[Any], planted: str) -> bool:
    """Whether a world flagged a legitimate column.

    Args:
        findings: The screen's findings in one world.
        planted: The planted column, or `""` in a null world.

    Returns:
        True when any column other than `planted` is a suspect.
    """
    return any(f.suspect and f.feature != planted for f in findings)


def null_replicate(
    seed: int, grouped: bool = False, kind: str = "binary"
) -> dict[str, Any]:
    """One null world: permuted outcome, gaps in two columns.

    Args:
        seed: Replicate seed.
        grouped: Whether rows come in groups of `GROUP_SIZE` that are
            permuted together.
        kind: `binary`, `multiclass` or `regression`.

    Returns:
        Whether the screen found any suspect (`alarm`, every one a false
        alarm) or any significant column, the size of the tested family,
        and whether the old rule flagged any column.
    """
    frame = TABLES[kind][0]()
    names = real_columns(frame, TABLE_COLUMNS, seed)
    groups = (
        pandas.Series(numpy.arange(len(frame)) // GROUP_SIZE, index=frame.index)
        if grouped
        else None
    )
    world = simulate.null_world(
        frame[names].astype(float),
        frame[f4_certify.TARGET],
        seed,
        groups=groups,
    )
    features = with_gaps(world.features, seed)
    table = features.assign(**{f4_certify.TARGET: world.outcome})
    report = run_screen(table, seed, groups, kind)
    old = old_rule(table, list(features.columns), kind)
    return {
        "suspect": bool(report.suspects),
        "alarm": alarmed(report.findings, ""),
        "significant": any(f.significant for f in report.findings),
        "family": report.family,
        "old": any(old.values()),
    }


def plant_replicate(
    defect: str, seed: int, kind: str = "binary"
) -> dict[str, Any]:
    """One planted leak among real columns.

    Args:
        defect: Leak type to plant.
        seed: Replicate seed.
        kind: `binary`, `multiclass` or `regression`.

    Returns:
        Whether the planted column was found by the mechanism of its type,
        whether it was a suspect by any mechanism, whether a legitimate
        column was a suspect (`alarm`), and whether the old rule flagged
        the plant.
    """
    frame = TABLES[kind][0]()
    rows = defect == plants.DUPLICATES
    names = real_columns(frame, TABLE_COLUMNS - (0 if rows else 1), seed)
    table = frame[[*names, f4_certify.TARGET]]
    if rows:
        table = plants.copy(table, seed)
    else:
        table = table.assign(
            **{defect: plant_column(frame, defect, seed, kind)}
        )
    report = run_screen(table, seed, kind=kind)
    mechanisms = {
        m for m, kinds in screen.MECHANISM_DEFECTS.items() if defect in kinds
    }
    feature = screen.ROWS if rows else defect
    found = any(
        f.suspect and f.feature == feature and f.mechanism in mechanisms
        for f in report.findings
    )
    any_mechanism = any(
        f.suspect and f.feature == feature for f in report.findings
    )
    old = False if rows else old_rule(table, [defect], kind)[defect]
    return {
        "found": bool(found),
        "found_any_mechanism": bool(any_mechanism),
        "alarm": alarmed(report.findings, feature),
        "old": bool(old),
    }


def replicate(defect: str, seed: int, kind: str = "binary") -> dict[str, Any]:
    """A null (`""`), grouped null (`"grouped"`) or planted replicate.

    Args:
        defect: `""` for a null world, `"grouped"` for a grouped null world,
            otherwise a leak type to plant.
        seed: Replicate seed.
        kind: `binary`, `multiclass` or `regression`.

    Returns:
        The result of `null_replicate` or `plant_replicate`.
    """
    if defect == "":
        return null_replicate(seed, kind=kind)
    if defect == "grouped":
        return null_replicate(seed, grouped=True, kind=kind)
    return plant_replicate(defect, seed, kind)


def run_parallel(
    jobs: Sequence[tuple[str, int, str]], workers: int
) -> list[dict[str, Any]]:
    """Runs the replicates in order.

    Args:
        jobs: (defect, seed, kind) triples as accepted by `replicate`.
        workers: Number of parallel workers.

    Returns:
        The replicate results in the order of `jobs`.
    """
    from joblib import delayed
    from joblib import Parallel

    return list(
        Parallel(n_jobs=workers)(delayed(replicate)(*job) for job in jobs)
    )


def wilson(successes: int, trials: int) -> tuple[float, float]:
    """95 percent Wilson interval.

    Args:
        successes: Number of successes.
        trials: Number of trials.

    Returns:
        The lower and upper bounds.
    """
    return certificates.bound(successes, trials, 0.95)


def band_99(alpha: float, runs: int) -> tuple[float, float]:
    """99 percent normal-approximation band for a rate of `alpha`.

    Args:
        alpha: The nominal rate.
        runs: Number of replicates the rate is observed over.

    Returns:
        The lower and upper edges of the band.
    """
    return f4_certify.band_99(alpha, runs)


def measure(kind: str, workers: int = 1) -> tuple[Any, dict[str, Any]]:
    """Measures the screen on null worlds and planted leaks for one kind.

    Args:
        kind: `binary`, `multiclass` or `regression`.
        workers: Number of parallel workers.

    Returns:
        (certificate, measurements).
    """
    frame = TABLES[kind][0]()
    jobs: list[tuple[str, int, str]] = [
        ("", s, kind) for s in range(NULL_REPLICATES)
    ]
    jobs += [("grouped", s, kind) for s in range(GROUPED_REPLICATES)]
    for defect in DEFECTS:
        jobs += [(defect, s, kind) for s in range(PLANT_REPLICATES)]
    outcomes = run_parallel(jobs, workers)
    nulls = outcomes[:NULL_REPLICATES]
    grouped = outcomes[NULL_REPLICATES : NULL_REPLICATES + GROUPED_REPLICATES]
    start = NULL_REPLICATES + GROUPED_REPLICATES
    planted_alarms = {
        defect: sum(
            item["alarm"]
            for item in outcomes[start + k * PLANT_REPLICATES :][
                :PLANT_REPLICATES
            ]
        )
        for k, defect in enumerate(DEFECTS)
    }
    null_alarms = sum(item["alarm"] for item in nulls)
    grouped_alarms = sum(item["alarm"] for item in grouped)
    false_alarms = null_alarms + grouped_alarms + sum(planted_alarms.values())
    worlds = len(outcomes)
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
                detector=screen.DETECTORS[kind],
                defect=defect,
                planted=PLANT_REPLICATES,
                detected=detected,
                clean=worlds,
                alarms=false_alarms,
                seeds=randomness.seed_path(0, "screen.certify", 0),
            )
        )
    cert = certificates.certify(
        screen.DETECTORS[kind],
        screen.fingerprint(),
        certificates.checksum([certify.frame_digest(frame)]),
        [0],
        results,
        f4_certify.LEAK_POLICY,
    )
    alpha = leakage.CERTIFIED.alpha
    significant = sum(item["significant"] for item in nulls)
    old_false = sum(item["old"] for item in nulls)
    low, high = band_99(alpha, NULL_REPLICATES)
    measurements = {
        "table": TABLES[kind][0].__name__,
        "rows": len(frame),
        "classes": (
            int(frame[f4_certify.TARGET].nunique())
            if TABLES[kind][1] == "classification"
            else 0
        ),
        "alpha": alpha,
        "rule": "benjamini-yekutieli",
        "null_replicates": NULL_REPLICATES,
        "plant_replicates_per_type": PLANT_REPLICATES,
        "table_columns": TABLE_COLUMNS,
        "family_sizes": sorted({item["family"] for item in nulls}),
        "clean_worlds": worlds,
        "suspect_false_alarms": false_alarms,
        "suspect_false_alarm_wilson_95": list(wilson(false_alarms, worlds)),
        "null_false_alarms": null_alarms,
        "planted_world_false_alarms": planted_alarms,
        "worlds_with_a_significant_column": significant,
        "significant_rate": significant / NULL_REPLICATES,
        "band_99": [low, high],
        "significant_rate_within_alpha": (
            significant / NULL_REPLICATES <= high
        ),
        "grouped_null_replicates": GROUPED_REPLICATES,
        "grouped_suspect_false_alarms": grouped_alarms,
        "grouped_significant_worlds": sum(i["significant"] for i in grouped),
        "old_rule_false_alarms": old_false,
        "old_rule_false_alarm_wilson_95": list(
            wilson(old_false, NULL_REPLICATES)
        ),
        "recall": recall,
        "settings": dataclasses.asdict(leakage.CERTIFIED),
    }
    return cert, measurements


def closing(frame: pandas.DataFrame, seed: int, *, late: bool) -> pandas.Series:
    """A date that moves with the target, after or before the cutoff.

    Args:
        frame: The table, including the target.
        seed: Replicate seed.
        late: A closing date after `START` when True (a leak once the
            cutoff is declared), an opening date before it otherwise.

    Returns:
        One date per row, aligned to `frame`.
    """
    rng = randomness.component_rng(seed, "screen.dated", 0)
    ranks = frame[f4_certify.TARGET].rank(pct=True).to_numpy()
    days = benchmarks.evidence.leakage.SHIFT * ranks + rng.exponential(
        benchmarks.evidence.leakage.JITTER, len(frame)
    )
    offset = pandas.to_timedelta(days, unit="D")
    start = benchmarks.evidence.leakage.START
    return pandas.Series(
        start + offset if late else start - offset, index=frame.index
    )


def dated_replicate(seed: int, kind: str, *, late: bool) -> dict[str, bool]:
    """One dated world: real columns and a planted date, cutoff `START`.

    Args:
        seed: Replicate seed.
        kind: `binary`, `multiclass` or `regression`.
        late: Plant a closing date (True) or an opening date (False).

    Returns:
        Whether the date was flagged (`found`), and whether any other
        column was (`alarm`).
    """
    frame = TABLES[kind][0]()
    names = real_columns(frame, TABLE_COLUMNS - 1, seed)
    table = frame[names].assign(date=closing(frame, seed, late=late))
    flagged = {
        f.feature
        for f in screen.dated(
            table,
            list(table.columns),
            benchmarks.evidence.leakage.START.isoformat(),
        )
    }
    return {"found": "date" in flagged, "alarm": bool(flagged - {"date"})}


def measure_dated() -> tuple[Any, dict[str, Any]]:
    """Measures the dated check on closing and opening dates of real tables.

    Returns:
        (certificate, measurements).
    """
    planted = [
        dated_replicate(seed, kind, late=True)
        for kind in KINDS
        for seed in range(DATED_REPLICATES)
    ]
    clean = [
        dated_replicate(seed, kind, late=False)
        for kind in KINDS
        for seed in range(DATED_REPLICATES)
    ]
    detected = sum(item["found"] for item in planted)
    alarms = sum(item["alarm"] for item in planted)
    alarms += sum(item["found"] or item["alarm"] for item in clean)
    worlds = len(planted) + len(clean)
    result = qualification.Certificate(
        detector=screen.DATED_ID,
        defect=screen.DATED_DEFECT,
        planted=len(planted),
        detected=detected,
        clean=worlds,
        alarms=alarms,
        seeds=randomness.seed_path(0, "screen.dated", 0),
    )
    cert = certificates.certify(
        screen.DATED_ID,
        screen.dated_digest(),
        certificates.checksum(
            [certify.frame_digest(TABLES[kind][0]()) for kind in KINDS]
        ),
        [0],
        [result],
        f4_certify.LEAK_POLICY,
    )
    measurements = {
        "tables": [TABLES[kind][0].__name__ for kind in KINDS],
        "cutoff": benchmarks.evidence.leakage.START.isoformat(),
        "replicates_per_table": DATED_REPLICATES,
        "planted": len(planted),
        "detected": detected,
        "detected_wilson_95": list(wilson(detected, len(planted))),
        "clean_worlds": worlds,
        "false_alarms": alarms,
        "false_alarm_wilson_95": list(wilson(alarms, worlds)),
    }
    return cert, measurements


def certify_screen(workers: int = 1) -> tuple[list[Any], dict[str, Any]]:
    """Measures the screen for every kind of target, and the dated check.

    Args:
        workers: Number of parallel workers.

    Returns:
        (one certificate per kind and one for the dated check, measurements
        keyed by kind and `dated`).
    """
    found = {kind: measure(kind, workers) for kind in KINDS}
    found["dated"] = measure_dated()
    return [c for c, _ in found.values()], {k: m for k, (_, m) in found.items()}


def problems(store: certificates.Store) -> list[str]:
    """Why the certificate file is unusable: missing or stale entries.

    Args:
        store: The loaded certificate store.

    Returns:
        A message per kind and leak type, and for the dated check, whose
        certificate is missing or stale; empty when all are current.
    """
    found = []
    digest = screen.fingerprint()
    for kind in KINDS:
        detector = screen.DETECTORS[kind]
        for defect in DEFECTS:
            report = certificates.audit(store, {detector: (digest, defect)})
            if report.missing:
                found.append(f"{detector}/{defect}: missing")
            if report.stale:
                found.append(f"{detector}/{defect}: stale")
    report = certificates.audit(
        store, {screen.DATED_ID: (screen.dated_digest(), screen.DATED_DEFECT)}
    )
    if report.missing:
        found.append(f"{screen.DATED_ID}/{screen.DATED_DEFECT}: missing")
    if report.stale:
        found.append(f"{screen.DATED_ID}/{screen.DATED_DEFECT}: stale")
    return found


def uncertified(store: certificates.Store) -> list[str]:
    """Kinds and leak types that were measured and did not meet the policy.

    Args:
        store: The loaded certificate store.

    Returns:
        `kind/defect` for every pair whose status is not gating, and
        `dated/leak_dated` when the dated check's is not.
    """
    digest = screen.fingerprint()
    found = [
        f"{kind}/{defect}"
        for kind in KINDS
        for defect in DEFECTS
        if store.status(screen.DETECTORS[kind], digest, defect)
        is not certificates.Authority.GATING
    ]
    status = store.status(
        screen.DATED_ID, screen.dated_digest(), screen.DATED_DEFECT
    )
    if status is not certificates.Authority.GATING:
        found.append(f"dated/{screen.DATED_DEFECT}")
    return found


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
        store = certificates.Store.load(screen.STORE_PATH)
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
    certs, measurements = certify_screen(options.workers)
    certificates.Store(certs).save(screen.STORE_PATH)
    MEASUREMENTS_PATH.write_text(
        json.dumps(measurements, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(measurements, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
