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
  (`screen.MECHANISM_DEFECTS`). The plants are the product's own
  (`scmle.assurance.admissibility.plants`), the same every run probes for: a
  copy of the target, a noisy copy, a field derived from the target with noise,
  a value recorded only for part of the target's range, and a value recorded for
  about 9 percent of the rows, all of one class or band.
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
from sklearn.datasets import load_diabetes
from sklearn.datasets import load_digits

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
            or column.missingness_association(frame[name], target, task)
            >= OLD_MISSINGNESS
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
    cfg = dataclasses.replace(leakage.CERTIFIED, run_seed=seed)
    return screen.screen(
        frame, f4_certify.TARGET, TABLES[kind][1], cfg, groups=groups
    )


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
        Whether the screen found any suspect or any significant column, the
        size of the tested family, and whether the old rule flagged any
        column.
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
        whether it was a suspect by any mechanism, and whether the old rule
        flagged it.
    """
    frame = TABLES[kind][0]()
    names = real_columns(frame, TABLE_COLUMNS - 1, seed)
    table = frame[[*names, f4_certify.TARGET]].assign(
        **{defect: plant_column(frame, defect, seed, kind)}
    )
    report = run_screen(table, seed, kind=kind)
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
    old = old_rule(table, [defect], kind)
    return {
        "found": bool(found),
        "found_any_mechanism": bool(any_mechanism),
        "old": bool(old[defect]),
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
                detector=screen.DETECTORS[kind],
                defect=defect,
                planted=PLANT_REPLICATES,
                detected=detected,
                clean=NULL_REPLICATES,
                alarms=false_alarms,
                seeds=randomness.seed_path(0, "screen.certify", 0),
            )
        )
    cert = certificates.certify(
        screen.DETECTORS[kind],
        screen.code_digest(),
        certificates.corpus_hash([certify.frame_digest(frame)]),
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


def certify_screen(workers: int = 1) -> tuple[list[Any], dict[str, Any]]:
    """Measures the screen for every kind of target.

    Args:
        workers: Number of parallel workers.

    Returns:
        (one certificate per kind, measurements keyed by kind).
    """
    found = {kind: measure(kind, workers) for kind in KINDS}
    return [c for c, _ in found.values()], {k: m for k, (_, m) in found.items()}


def problems(store: certificates.Store) -> list[str]:
    """Why the certificate file is unusable: missing or stale entries.

    Args:
        store: The loaded certificate store.

    Returns:
        A message per kind and leak type whose certificate is missing or
        stale; empty when all are current.
    """
    found = []
    digest = screen.code_digest()
    for kind in KINDS:
        detector = screen.DETECTORS[kind]
        for defect in DEFECTS:
            report = certificates.audit(store, {detector: (digest, defect)})
            if report.missing:
                found.append(f"{detector}/{defect}: missing")
            if report.stale:
                found.append(f"{detector}/{defect}: stale")
    return found


def uncertified(store: certificates.Store) -> list[str]:
    """Kinds and leak types that were measured and did not meet the policy.

    Args:
        store: The loaded certificate store.

    Returns:
        `kind/defect` for every pair whose status is not gating.
    """
    digest = screen.code_digest()
    return [
        f"{kind}/{defect}"
        for kind in KINDS
        for defect in DEFECTS
        if store.status(screen.DETECTORS[kind], digest, defect)
        is not certificates.Authority.GATING
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
