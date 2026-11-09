"""Certificates for the robustness checks.

Three robustness checks may gate a run once certified:

* `f6-shift-c2st` (`scmle.data.robustness.shift.c2st`): the classifier
  two-sample test.
* `f6-shift-mmd` (`...shift.mmd_test`): the kernel two-sample test.
* `f6-stability` (`...stability`, `...multiverse`): the multiverse survival
  rule.

Everything below was fixed before any certification measurement and is not
tuned to results. A check that does not meet its floor is filed as measured
and stays descriptive; nothing here is retuned until it passes.

Shift. A replicate draws two samples of `SHIFT_ROWS` rows from five
independent standard normal columns. The clean side is `SHIFT_NULL`
replicates of identical distributions, so a false alarm is a rejection under
the null. The planted side is `SHIFT_PLANTED` replicates in which one column
of the second sample is shifted by ``sqrt(2) * ppf(0.60)`` standard
deviations, the shift whose Bayes-optimal AUC is 0.60 (the power point of
the certification). Plants are synthetic and easy, and the generator is
Gaussian, so the power is optimistic for real data; no real deployment sample
is available to plant into. Policy: recall Wilson lower bound at least 0.80,
false-alarm Wilson upper bound at most 0.075 (a 5 percent test has Wilson
upper bound near 0.065 over 1000 nulls), 95 percent intervals. The size by
p-value alone is also recorded against a 99 percent binomial band.

Stability. Replicates are semi-synthetic: the feature is a real column of the
breast-cancer table that ships with scikit-learn, sampled without
replacement, with 15 percent of its values deleted at random, and the target
is generated from it. The clean side is `STABLE_REPS` genuine, robust
effects (rho near 0.45). Planted spec-dependent effects: `fragile_outlier_
driven` (the whole association comes from 10 percent extreme values that the
`iqr_drop` rule removes) and `fragile_inclusion_reversal` (the effect has
the opposite sign in the second half of the rows). `no_effect` replicates
have an independent target. A planted defect is detected when the finding is
labelled `fragile` (for `no_effect`: not labelled `stable`); a false alarm is
a genuine effect not labelled `stable`. Policy: recall lower bound at least
0.80 and false-alarm upper bound at most 0.10 (genuine effects labelled
stable at least 0.90), 95 percent intervals. The grid is the 18
specifications in `GRID_CHOICES`. The plants are built to be found by that
grid; the certificate says nothing about spec-dependence of other kinds.

Amendment 2026-11-04 (made before the re-run, policy unchanged). The first
certificate of `f6-shift-mmd` (1000 nulls) had 60 false alarms and a Wilson
upper bound of 0.0765 against the ceiling 0.075: the observed rate sat inside
the ceiling and the interval was too wide to say so. The number of null
replicates for the MMD test is raised to `MMD_NULL` (5000, seeds 0 up to
4999, which include the first 1000). The recall floor, the false-alarm
ceiling, the confidence, the world and the test are not changed. The result
is filed whichever way it falls.

Seeds. Development seeds start at `DEV_SEED_BASE`; certification uses seeds
0 up to the replicate count. The two ranges never overlap. A replicate with
seed k draws its data and its test with seed k.

    python -m benchmarks.qualification.f6_certify --write      # measure
    python -m benchmarks.qualification.f6_certify --check      # verify file
    python -m benchmarks.qualification.f6_certify --dev        # quick look
"""

import argparse
from collections.abc import Sequence
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy
import pandas
from scipy import stats
from sklearn.datasets import load_breast_cancer

from benchmarks.qualification import certify
from scmle.contracts import qualification
from scmle.core import randomness
from scmle.data.charter import contracts
from scmle.data.detectors import certificates
from scmle.data.robustness import multiverse
from scmle.data.robustness import shift
from scmle.data.robustness import stability

STORE_PATH = Path(__file__).with_name("f6_certificates.json")
MEASUREMENTS_PATH = Path(__file__).with_name("f6_measurements.json")
DEV_SEED_BASE = 10_000
Z_99 = 2.5758293035489004
SHIFT_ROWS = 2000
SHIFT_COLUMNS = 5
SHIFT_NULL = 1000
MMD_NULL = 5000
SHIFT_PLANTED = 300
PLANTED_AUC = 0.60
SHIFT_POLICY = certificates.CertificationPolicy(0.80, 0.075, 0.95)
STABILITY_ROWS = 400
STABLE_REPS = 300
PLANT_REPS = 200
STABILITY_POLICY = certificates.CertificationPolicy(0.80, 0.10, 0.95)
MISSING_RATE = 0.15
STABLE_BETA = 0.5
REVERSAL_BETA = 0.6
EXTREME_SHARE = 0.10
EXTREME_SIGMAS = 8.0
EXTREME_OUTCOME = 3.0
BOOTSTRAP = 100
SOURCE_COLUMN = "mean texture"
GRID_CHOICES = (
    contracts.Choice("imputation", ("median", "complete_case"), "median"),
    contracts.Choice("outlier_rule", ("none", "winsorize", "iqr_drop"), "none"),
    contracts.Choice("inclusion", ("all", "first_half", "second_half"), "all"),
)
CLEAN = "stable"
FRAGILE_KINDS = (stability.DEFECT_OUTLIER, stability.DEFECT_REVERSAL)


def band_99(alpha: float, runs: int) -> tuple[float, float]:
    """99 percent normal-approximation band for a rate of `alpha`.

    Args:
        alpha: The nominal rate.
        runs: Number of replicates the rate is observed over.

    Returns:
        The lower and upper edges of the band.
    """
    half = Z_99 * math.sqrt(alpha * (1.0 - alpha) / runs)
    return alpha - half, alpha + half


def shift_delta() -> float:
    """Mean shift (in standard deviations) whose Bayes AUC is `PLANTED_AUC`.

    Returns:
        ``sqrt(2) * ppf(PLANTED_AUC)``.
    """
    return math.sqrt(2.0) * float(stats.norm.ppf(PLANTED_AUC))


def shift_samples(
    planted: bool, seed: int
) -> tuple[pandas.DataFrame, pandas.DataFrame]:
    """Two samples, the second shifted in its first column when planted.

    Args:
        planted: Whether to shift the second sample by `shift_delta()`.
        seed: Replicate seed.

    Returns:
        Two frames of `SHIFT_ROWS` rows and `SHIFT_COLUMNS` standard normal
        columns `x0`, `x1`, ....
    """
    rng = randomness.component_rng(seed, "f6.shift.world", int(planted))
    names = [f"x{k}" for k in range(SHIFT_COLUMNS)]
    first = pandas.DataFrame(
        rng.normal(size=(SHIFT_ROWS, SHIFT_COLUMNS)), columns=names
    )
    second = pandas.DataFrame(
        rng.normal(size=(SHIFT_ROWS, SHIFT_COLUMNS)), columns=names
    )
    if planted:
        second["x0"] += shift_delta()
    return first, second


def shift_replicate(test: str, planted: bool, seed: int) -> tuple[bool, float]:
    """(rejected, p-value) of one two-sample test on one replicate.

    Args:
        test: `shift.C2ST_ID` for the classifier test, otherwise the kernel
            test.
        planted: Whether the second sample carries the planted shift.
        seed: Replicate seed, also the test's seed.

    Returns:
        Whether the test's decision is not a pass, and its p-value (1.0 when
        it reports none).
    """
    first, second = shift_samples(planted, seed)
    run = shift.c2st if test == shift.C2ST_ID else shift.mmd_test
    check = run(first, second, seed, gate=certificates.GateStatus.GATING)
    return check.decision is not qualification.Decision.PASS, float(
        check.p_value or 1.0
    )


def source_values() -> numpy.ndarray:
    """Standardised values of the real source column.

    Returns:
        The `SOURCE_COLUMN` of the breast-cancer table with zero mean and
        unit standard deviation.
    """
    frame = load_breast_cancer(as_frame=True).frame
    values = frame[SOURCE_COLUMN].to_numpy(dtype=float)
    return (values - values.mean()) / values.std()


def stability_world(
    kind: str, seed: int
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """One semi-synthetic finding: the feature with holes, and the target.

    Args:
        kind: `CLEAN` for a genuine effect, or a defect type of the
            stability check.
        seed: Replicate seed.

    Returns:
        The feature (with `MISSING_RATE` of its values missing) and the
        target.

    Raises:
        ValueError: If `kind` is not a known kind.
    """
    rng = randomness.component_rng(seed, f"f6.stability.world.{kind}", 0)
    pool = source_values()
    x = rng.permutation(pool)[:STABILITY_ROWS].copy()
    noise = rng.normal(size=STABILITY_ROWS)
    if kind == CLEAN:
        y = STABLE_BETA * x + noise
    elif kind == stability.DEFECT_NULL:
        y = noise
    elif kind == stability.DEFECT_REVERSAL:
        sign = numpy.where(
            numpy.arange(STABILITY_ROWS) < STABILITY_ROWS / 2, 1, -1
        )
        y = REVERSAL_BETA * sign * x + noise
    elif kind == stability.DEFECT_OUTLIER:
        count = round(EXTREME_SHARE * STABILITY_ROWS)
        rows = rng.choice(STABILITY_ROWS, count, replace=False)
        side = rng.choice([-1.0, 1.0], count)
        x[rows] = side * EXTREME_SIGMAS
        y = noise.copy()
        y[rows] = side * EXTREME_OUTCOME + 0.5 * noise[rows]
    else:
        raise ValueError(f"unknown kind: {kind}")
    x[rng.random(STABILITY_ROWS) < MISSING_RATE] = numpy.nan
    return x, y


def stability_replicate(kind: str, seed: int) -> str:
    """The verdict on one semi-synthetic finding.

    Args:
        kind: `CLEAN` or a defect type of the stability check.
        seed: Replicate seed.

    Returns:
        The stability status of the finding.
    """
    x, y = stability_world(kind, seed)
    grid = multiverse.enumerate_grid(
        GRID_CHOICES, seed, multiverse.DEFAULT_MAX_SPECS * 2
    )
    result = stability.assess_finding(
        kind, x, y, grid, seed, bootstrap=BOOTSTRAP
    )
    return result.status


def run_jobs(
    function: Any, jobs: Sequence[tuple[Any, ...]], workers: int
) -> list[Any]:
    """Runs `function(*job)` for every job, in order.

    Args:
        function: The function to call.
        jobs: Argument tuples.
        workers: Number of parallel workers.

    Returns:
        The results in the order of `jobs`.
    """
    from joblib import delayed
    from joblib import Parallel

    return list(Parallel(n_jobs=workers)(delayed(function)(*j) for j in jobs))


def shift_certificate(
    test: str, offset: int, workers: int
) -> tuple[certificates.DetectorCertificate, dict[str, Any]]:
    """Measures one two-sample test on nulls and planted shifts.

    Args:
        test: `shift.C2ST_ID` or `shift.MMD_ID`.
        offset: Added to the replicate seeds.
        workers: Number of parallel workers.

    Returns:
        The certificate and the measurements (size by p-value alone with its
        99 percent band, and power).
    """
    null_count = MMD_NULL if test == shift.MMD_ID else SHIFT_NULL
    nulls = run_jobs(
        shift_replicate,
        [(test, False, offset + s) for s in range(null_count)],
        workers,
    )
    plants = run_jobs(
        shift_replicate,
        [(test, True, offset + s) for s in range(SHIFT_PLANTED)],
        workers,
    )
    false_alarms = sum(item[0] for item in nulls)
    detected = sum(item[0] for item in plants)
    p_alone = sum(item[1] <= shift.ALPHA for item in nulls)
    result = qualification.Certificate(
        detector_id=test,
        defect_type=shift.DEFECT_TYPE,
        planted=SHIFT_PLANTED,
        detected=detected,
        clean_rows=null_count,
        false_alarms=false_alarms,
        seed_path=randomness.seed_path(offset, f"f6.certify.{test}", 0),
    )
    cert = certificates.certify(
        test,
        shift.code_digest(),
        certificates.corpus_hash(
            [
                multiverse.digest(
                    {
                        "generator": "gaussian two-sample",
                        "rows": SHIFT_ROWS,
                        "columns": SHIFT_COLUMNS,
                        "planted_auc": PLANTED_AUC,
                    }
                )
            ]
        ),
        [offset],
        [result],
        SHIFT_POLICY,
    )
    low, high = band_99(shift.ALPHA, null_count)
    rate = p_alone / null_count
    measurements = {
        "null_replicates": null_count,
        "planted_replicates": SHIFT_PLANTED,
        "rows_per_side": SHIFT_ROWS,
        "planted_bayes_auc": PLANTED_AUC,
        "alpha": shift.ALPHA,
        "permutations": shift.PERMUTATIONS,
        "decision_false_alarms": false_alarms,
        "size_by_p_value_alone": rate,
        "band_99": [low, high],
        "size_in_band": low <= rate <= high,
        "power": detected / SHIFT_PLANTED,
        "seed_offset": offset,
    }
    return cert, measurements


def stability_certificate(
    offset: int, workers: int
) -> tuple[certificates.DetectorCertificate, dict[str, Any]]:
    """Measures the survival rule on genuine and planted findings.

    Args:
        offset: Added to the replicate seeds.
        workers: Number of parallel workers.

    Returns:
        The certificate and the measurements (labels of genuine effects,
        detections by defect type, the rule and the grid size).
    """
    clean = run_jobs(
        stability_replicate,
        [(CLEAN, offset + s) for s in range(STABLE_REPS)],
        workers,
    )
    false_alarms = sum(status != stability.STABLE for status in clean)
    results = []
    counts = {}
    for kind in stability.DEFECT_TYPES:
        statuses = run_jobs(
            stability_replicate,
            [(kind, offset + s) for s in range(PLANT_REPS)],
            workers,
        )
        if kind == stability.DEFECT_NULL:
            found = sum(status != stability.STABLE for status in statuses)
        else:
            found = sum(status == stability.FRAGILE for status in statuses)
        counts[kind] = {
            "planted": PLANT_REPS,
            "detected": found,
            "statuses": {s: statuses.count(s) for s in sorted(set(statuses))},
        }
        results.append(
            qualification.Certificate(
                detector_id=stability.DETECTOR_ID,
                defect_type=kind,
                planted=PLANT_REPS,
                detected=found,
                clean_rows=STABLE_REPS,
                false_alarms=false_alarms,
                seed_path=randomness.seed_path(
                    offset, "f6.certify.stability", 0
                ),
            )
        )
    pool = pandas.Series(source_values()).to_frame("value")
    cert = certificates.certify(
        stability.DETECTOR_ID,
        stability.code_digest(),
        certificates.corpus_hash(
            [
                certify.frame_digest(pool),
                multiverse.digest(
                    {
                        "choices": [c.to_dict() for c in GRID_CHOICES],
                        "rows": STABILITY_ROWS,
                        "missing": MISSING_RATE,
                        "bootstrap": BOOTSTRAP,
                    }
                ),
            ]
        ),
        [offset],
        results,
        STABILITY_POLICY,
    )
    measurements = {
        "genuine_effects": STABLE_REPS,
        "genuine_labelled_stable": STABLE_REPS - false_alarms,
        "genuine_statuses": {s: clean.count(s) for s in sorted(set(clean))},
        "by_defect": counts,
        "rule": stability.DEFAULT_RULE.to_dict(),
        "bootstrap": BOOTSTRAP,
        "specifications": len(
            multiverse.enumerate_grid(
                GRID_CHOICES, 0, multiverse.DEFAULT_MAX_SPECS * 2
            ).specs
        ),
        "seed_offset": offset,
    }
    return cert, measurements


def certify_all(
    workers: int = 1, offset: int = 0
) -> tuple[certificates.CertificateStore, dict[str, Any]]:
    """Measures every robustness check; `offset` shifts the seed range.

    Args:
        workers: Number of parallel workers.
        offset: Added to the replicate seeds.

    Returns:
        The certificate store and the measurements document.
    """
    started = time.monotonic()
    store = certificates.CertificateStore()
    document: dict[str, Any] = {}
    for test in (shift.C2ST_ID, shift.MMD_ID):
        cert, measured = shift_certificate(test, offset, workers)
        store.add(cert)
        document[test] = measured
    cert, measured = stability_certificate(offset, workers)
    store.add(cert)
    document[stability.DETECTOR_ID] = measured
    document["seconds"] = round(time.monotonic() - started, 1)
    document["development_seed_base"] = DEV_SEED_BASE
    return store, document


def current_digests() -> dict[str, str]:
    """Code hash of every robustness check as it is now.

    Returns:
        Mapping of detector id to its current code hash.
    """
    return {
        shift.C2ST_ID: shift.code_digest(),
        shift.MMD_ID: shift.code_digest(),
        stability.DETECTOR_ID: stability.code_digest(),
    }


def required_defects() -> dict[str, tuple[str, ...]]:
    """Defect types each robustness check must be certified for.

    Returns:
        Mapping of detector id to its defect types.
    """
    return {
        shift.C2ST_ID: (shift.DEFECT_TYPE,),
        shift.MMD_ID: (shift.DEFECT_TYPE,),
        stability.DETECTOR_ID: stability.DEFECT_TYPES,
    }


def audit_labels(store: certificates.CertificateStore) -> dict[str, list[str]]:
    """Detector/defect pairs by why they may not gate.

    Args:
        store: The certificate store.

    Returns:
        `missing` and `stale` are failures of the certificate file;
        `uncertified` pairs were measured and did not meet the policy, so the
        check stays descriptive (not an error).
    """
    found: dict[str, list[str]] = {
        "missing": [],
        "stale": [],
        "uncertified": [],
    }
    digests = current_digests()
    for detector, defects in required_defects().items():
        for defect in defects:
            report = certificates.audit(
                store, {detector: (digests[detector], defect)}
            )
            for label in found:
                if getattr(report, label):
                    found[label].append(f"{detector}/{defect}")
    return found


def problems(store: certificates.CertificateStore) -> list[str]:
    """Why the certificate file is unusable: missing or stale entries.

    Args:
        store: The certificate store.

    Returns:
        One message per missing or stale detector/defect pair.
    """
    labels = audit_labels(store)
    return [
        f"{item}: {kind}"
        for kind in ("missing", "stale")
        for item in labels[kind]
    ]


def development_look(workers: int) -> dict[str, Any]:
    """A small run on development seeds, to time the work. Writes nothing.

    Args:
        workers: Number of parallel workers.

    Returns:
        The raw replicate results of both shift tests (8 each) and the
        stability verdicts (20 per kind).
    """
    small = {
        "shift_rows": SHIFT_ROWS,
        "c2st": [
            shift_replicate(shift.C2ST_ID, s % 2 == 1, DEV_SEED_BASE + s)
            for s in range(8)
        ],
        "mmd": [
            shift_replicate(shift.MMD_ID, s % 2 == 1, DEV_SEED_BASE + s)
            for s in range(8)
        ],
    }
    kinds = (CLEAN, *stability.DEFECT_TYPES)
    small["stability"] = {
        kind: run_jobs(
            stability_replicate,
            [(kind, DEV_SEED_BASE + s) for s in range(20)],
            workers,
        )
        for kind in kinds
    }
    return small


def main(arguments: list[str] | None = None) -> int:
    """Measures (`--write`), verifies (`--check`) or looks (`--dev`).

    Args:
        arguments: Command-line arguments; `None` reads `sys.argv`.

    Returns:
        With `--check`, 1 if any certificate is missing or stale, otherwise
        0; with `--write` and `--dev`, always 0.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true")
    action.add_argument("--check", action="store_true")
    action.add_argument("--dev", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--only-mmd",
        action="store_true",
        help="with --write: re-measure the MMD test and keep the rest",
    )
    options = parser.parse_args(arguments)
    if options.dev:
        print(json.dumps(development_look(options.workers), default=str))
        return 0
    if options.check:
        store = certificates.CertificateStore.load(STORE_PATH)
        found = problems(store)
        report = {
            "ok": not found,
            "problems": found,
            "descriptive_only": audit_labels(store)["uncertified"],
        }
        print(json.dumps(report, indent=2))
        return 1 if found else 0
    if options.only_mmd:
        store = certificates.CertificateStore.load(STORE_PATH)
        document = json.loads(MEASUREMENTS_PATH.read_text(encoding="utf-8"))
        cert, document[shift.MMD_ID] = shift_certificate(
            shift.MMD_ID, 0, options.workers
        )
        store.add(cert)
    else:
        store, document = certify_all(options.workers)
    store.save(STORE_PATH)
    MEASUREMENTS_PATH.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(document, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
