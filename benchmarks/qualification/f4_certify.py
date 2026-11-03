"""Certificates for the F4 detectors (canon P1).

Three F4 checks may gate a run once certified:

* `f4-leakage-skill` (`scmle.data.admissibility.leakage`): the single-feature
  skill test. Defect types are leaks planted in the breast-cancer table that
  ships with scikit-learn (a copy of the target, a noisy copy, a threshold
  rule, a value recorded only for one class). The clean side is 1000 *null
  worlds*: the real features with the outcome permuted, one feature tested per
  world, so a "clean row" here means a null replicate. Strong legitimate
  predictors are not in the clean side; the test is expected to call those
  suspects and the certificate says nothing about them.
* `f4-asof-reconstruction` (`...admissibility.asof`): rows whose stored value
  was overwritten after the prediction moment, on a synthetic change log
  (no real change log is available), judged row by row.
* `f4-reconciliation` (`...admissibility.reconciliation`): planted unexplained
  gaps among quantities, some of which carry correct explanations.

The settings and thresholds below were fixed before measuring and not tuned.
Plants are synthetic, so recall is optimistic (canon limits). The leakage
size measured by the permutation p-value alone is also recorded against a 99
percent binomial band, in `f4_measurements.json`.

    python -m benchmarks.qualification.f4_certify --write      # measure
    python -m benchmarks.qualification.f4_certify --check      # verify file
"""

import argparse
from collections.abc import Sequence
import dataclasses
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

from benchmarks.qualification import certify
from benchmarks.qualification import inject
from benchmarks.qualification import simulate
from scmle.contracts import qualification
from scmle.core import randomness
from scmle.data.admissibility import asof
from scmle.data.admissibility import leakage
from scmle.data.admissibility import reconciliation
from scmle.data.detectors import certificates

STORE_PATH = Path(__file__).with_name("f4_certificates.json")
MEASUREMENTS_PATH = Path(__file__).with_name("f4_measurements.json")
TARGET = "target"
NULL_REPLICATES = 1000
PLANT_REPLICATES = 200
SEEDS = (0, 1, 2)
Z_99 = 2.5758293035489004
# Fixed in advance: Wilson lower bound of recall at least 0.9, upper bound of
# the false-alarm rate on clean copies at most 0.05, 95 percent intervals.
LEAK_POLICY = certificates.CertificationPolicy(0.9, 0.05, 0.95)
THRESHOLD_NOISE = 0.25
NOISY_COPY_NOISE = 0.02
LOG_ROWS = 1000
OVERWRITE_RATE = 0.1
GAP_RATE = 0.1
VERSIONS = 3


def breast_cancer() -> pd.DataFrame:
    """The table the leak plants go into."""
    return load_breast_cancer(as_frame=True).frame.reset_index(drop=True)


def plant_leak(frame: pd.DataFrame, defect: str, seed: int) -> pd.Series:
    """Adds one planted leak column and returns it.

    Args:
        frame: Breast-cancer table with the `target` column.
        defect: One of `leakage.LEAK_DEFECTS`.
        seed: Run seed.

    Returns:
        The leak column.
    """
    if defect == "leak_threshold":
        rng = randomness.component_rng(seed, "f4.threshold", 0)
        values = frame[TARGET].to_numpy(dtype=float) + rng.normal(
            scale=THRESHOLD_NOISE, size=len(frame)
        )
        return pd.Series(values, index=frame.index, name=defect)
    kinds = {
        "leak_copy": ("copy", 0.1),
        "leak_noisy_copy": ("noisy_copy", NOISY_COPY_NOISE),
        "leak_missing_for_class": ("missing_for_class", 0.1),
    }
    kind, noise = kinds[defect]
    injection = inject.inject_leak(
        frame, TARGET, kind, seed, name=defect, noise=noise
    )
    return injection.frame[defect]


def leak_detected(
    feature: pd.Series, outcome: pd.Series, run_seed: int
) -> tuple[bool, float]:
    """(suspect, permutation p-value) of one feature."""
    cfg = dataclasses.replace(leakage.CERTIFIED, run_seed=run_seed)
    result = leakage.assess_feature(
        feature,
        pd.DataFrame(index=feature.index),
        outcome,
        "classification",
        cfg,
        gate=certificates.GateStatus.GATING,
    )
    return result.suspect, result.skill.p_value


def null_replicate(seed: int) -> tuple[bool, float]:
    """One null world: permuted outcome, one real feature tested."""
    frame = breast_cancer()
    features = frame.drop(columns=TARGET)
    world = simulate.null_world(features, frame[TARGET], seed)
    column = features.columns[seed % features.shape[1]]
    return leak_detected(world.features[column], world.outcome, seed)


def plant_replicate(defect: str, seed: int) -> tuple[bool, float]:
    """One planted leak, tested on the leak column alone."""
    frame = breast_cancer()
    return leak_detected(plant_leak(frame, defect, seed), frame[TARGET], seed)


def replicate(defect: str, seed: int) -> tuple[bool, float]:
    """A planted replicate, or a null replicate when `defect` is empty."""
    if defect:
        return plant_replicate(defect, seed)
    return null_replicate(seed)


def run_parallel(
    jobs: Sequence[tuple[str, int]], workers: int
) -> list[tuple[bool, float]]:
    """Runs null (`""`) and planted replicates, in order."""
    from joblib import delayed
    from joblib import Parallel

    return list(
        Parallel(n_jobs=workers)(delayed(replicate)(d, s) for d, s in jobs)
    )


def band_99(alpha: float, runs: int) -> tuple[float, float]:
    """99 percent normal-approximation band for a rate of `alpha`."""
    half = Z_99 * math.sqrt(alpha * (1.0 - alpha) / runs)
    return alpha - half, alpha + half


def certify_leakage(workers: int = 1) -> tuple[Any, dict[str, Any]]:
    """Measures the skill test on nulls and planted leaks.

    Returns:
        (certificate, measurements) where measurements holds the size by
        p-value alone and its 99 percent band.
    """
    frame = breast_cancer()
    jobs: list[tuple[str, int]] = [("", s) for s in range(NULL_REPLICATES)]
    for defect in leakage.LEAK_DEFECTS:
        jobs += [(defect, s) for s in range(PLANT_REPLICATES)]
    outcomes = run_parallel(jobs, workers)
    nulls = outcomes[:NULL_REPLICATES]
    false_alarms = sum(item[0] for item in nulls)
    p_alone = sum(item[1] <= leakage.CERTIFIED.alpha for item in nulls)
    results = []
    recalls = {}
    for k, defect in enumerate(leakage.LEAK_DEFECTS):
        start = NULL_REPLICATES + k * PLANT_REPLICATES
        part = outcomes[start : start + PLANT_REPLICATES]
        detected = sum(item[0] for item in part)
        recalls[defect] = detected
        results.append(
            qualification.Certificate(
                detector_id=leakage.DETECTOR_ID,
                defect_type=defect,
                planted=PLANT_REPLICATES,
                detected=detected,
                clean_rows=NULL_REPLICATES,
                false_alarms=false_alarms,
                seed_path=randomness.seed_path(0, "f4.certify.leakage", 0),
            )
        )
    cert = certificates.certify(
        leakage.DETECTOR_ID,
        leakage.code_digest(),
        certificates.corpus_hash([certify.frame_digest(frame)]),
        [0],
        results,
        LEAK_POLICY,
    )
    alpha = leakage.CERTIFIED.alpha
    low, high = band_99(alpha, NULL_REPLICATES)
    measurements = {
        "null_replicates": NULL_REPLICATES,
        "plant_replicates_per_type": PLANT_REPLICATES,
        "suspect_false_alarms": false_alarms,
        "size_by_p_value_alone": p_alone / NULL_REPLICATES,
        "alpha": alpha,
        "band_99": [low, high],
        "size_in_band": low <= p_alone / NULL_REPLICATES <= high,
        "detected": recalls,
        "settings": dataclasses.asdict(leakage.CERTIFIED),
    }
    return cert, measurements


def asof_world(seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A synthetic table and change log whose stored values are correct.

    Each of `LOG_ROWS` entities has `VERSIONS` log entries at increasing
    times; the prediction moment falls between the first and last entry, and
    the stored value is the log value at the moment.
    """
    rng = randomness.component_rng(seed, "f4.asof.world", 0)
    keys = np.arange(LOG_ROWS)
    times = np.sort(rng.uniform(0.0, 100.0, size=(LOG_ROWS, VERSIONS)), axis=1)
    values = rng.integers(0, 10**6, size=(LOG_ROWS, VERSIONS))
    moment = rng.uniform(times[:, 0] + 1e-3, times[:, -1] - 1e-3)
    log = pd.DataFrame(
        {
            "key": np.repeat(keys, VERSIONS),
            "written_at": times.ravel(),
            "value": values.ravel(),
        }
    )
    rows = pd.DataFrame({"key": keys, "moment": moment})
    index = (times <= moment[:, None]).sum(axis=1) - 1
    rows["stored"] = values[keys, index]
    rows["backfill"] = values[:, -1]
    return rows, log


def asof_detector(log: pd.DataFrame) -> certify.Detector:
    """Row detector bound to a change log."""

    def detect(frame: pd.DataFrame) -> pd.Series:
        return asof.reconstruction_mismatch(
            frame, log, "key", "moment", "stored", "written_at", "value"
        )

    return detect


def overwrite_plan(rows: pd.DataFrame) -> certify.DefectPlan:
    """Plants overwrites by replacing stored values with the latest one."""

    def plant(frame: pd.DataFrame, seed: int) -> inject.Injection:
        rng = randomness.component_rng(seed, "f4.asof.plant", 0)
        pick = rng.random(len(frame)) < OVERWRITE_RATE
        changed = pick & (frame["backfill"] != frame["stored"]).to_numpy()
        out = frame.copy()
        out.loc[changed, "stored"] = frame.loc[changed, "backfill"]
        mask = inject.blank_mask(out)
        mask["stored"] = changed
        return inject.finish(
            out, mask, asof.DEFECT_TYPE, "f4.asof", seed, 0, {}
        )

    return certify.DefectPlan(asof.DEFECT_TYPE, plant)


def certify_asof() -> certificates.DetectorCertificate:
    """Certifies the as-of reconstruction on synthetic change logs."""
    rows, log = asof_world(0)
    return certify.certify_detector(
        asof.DETECTOR_ID,
        asof_detector(log),
        certificates.code_hash(asof),
        rows,
        [overwrite_plan(rows)],
        SEEDS,
        LEAK_POLICY,
    )


def gap_frame(seed: int) -> pd.DataFrame:
    """Quantities that all reconcile, a third by a correct explanation."""
    rng = randomness.component_rng(seed, "f4.gap.world", 0)
    source = rng.integers(1_000, 100_000, size=LOG_ROWS).astype(float)
    explained = np.zeros(LOG_ROWS)
    pick = rng.random(LOG_ROWS) < 1 / 3
    explained[pick] = -np.round(
        source[pick] * rng.uniform(0.01, 0.1, pick.sum())
    )
    return pd.DataFrame(
        {
            "source": source,
            "extract": source + explained,
            "explained": explained,
        }
    )


def gap_detector(frame: pd.DataFrame) -> pd.Series:
    """Row detector: True where the reconciliation has an unexplained gap."""
    flags = []
    columns = zip(
        frame["extract"].tolist(),
        frame["source"].tolist(),
        frame["explained"].tolist(),
        strict=True,
    )
    for k, (extract, source, explained) in enumerate(columns):
        steps = (
            (
                reconciliation.Explanation(
                    reconciliation.Mechanism.FILTER,
                    "test accounts excluded",
                    float(explained),
                ),
            )
            if explained != 0
            else ()
        )
        item = reconciliation.Item(
            f"q{k}", float(extract), float(source), "ledger", steps
        )
        flags.append(
            reconciliation.reconcile_item(item).status
            is reconciliation.Status.UNEXPLAINED
        )
    return pd.Series(flags, index=frame.index, dtype=bool)


def gap_plan() -> certify.DefectPlan:
    """Plants unexplained shortfalls of 1 to 10 percent."""

    def plant(frame: pd.DataFrame, seed: int) -> inject.Injection:
        rng = randomness.component_rng(seed, "f4.gap.plant", 0)
        pick = rng.random(len(frame)) < GAP_RATE
        cut = np.round(
            frame["source"].to_numpy() * rng.uniform(0.01, 0.1, len(frame))
        )
        out = frame.copy()
        out.loc[pick, "extract"] = frame.loc[pick, "extract"] - cut[pick]
        mask = inject.blank_mask(out)
        mask["extract"] = pick
        return inject.finish(
            out, mask, reconciliation.DEFECT_TYPE, "f4.gap", seed, 0, {}
        )

    return certify.DefectPlan(reconciliation.DEFECT_TYPE, plant)


def certify_reconciliation() -> certificates.DetectorCertificate:
    """Certifies reconciliation on planted unexplained gaps."""
    return certify.certify_detector(
        reconciliation.DETECTOR_ID,
        gap_detector,
        certificates.code_hash(reconciliation),
        gap_frame(0),
        [gap_plan()],
        SEEDS,
        LEAK_POLICY,
    )


def current_digests() -> dict[str, str]:
    """Code hash of every F4 detector as it is now."""
    return {
        leakage.DETECTOR_ID: leakage.code_digest(),
        asof.DETECTOR_ID: certificates.code_hash(asof),
        reconciliation.DETECTOR_ID: certificates.code_hash(reconciliation),
    }


def required_defects() -> dict[str, tuple[str, ...]]:
    """Defect types each F4 detector must be certified for."""
    return {
        leakage.DETECTOR_ID: leakage.LEAK_DEFECTS,
        asof.DETECTOR_ID: (asof.DEFECT_TYPE,),
        reconciliation.DETECTOR_ID: (reconciliation.DEFECT_TYPE,),
    }


def problems(store: certificates.CertificateStore) -> list[str]:
    """Why the store does not permit gating by every F4 detector."""
    found = []
    digests = current_digests()
    for detector, defects in required_defects().items():
        for defect in defects:
            report = certificates.audit(
                store, {detector: (digests[detector], defect)}
            )
            for label in ("missing", "stale", "uncertified"):
                if getattr(report, label):
                    found.append(f"{detector}/{defect}: {label}")
    return found


def main(arguments: list[str] | None = None) -> int:
    """Measures (`--write`) or verifies (`--check`) the F4 certificates."""
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true")
    action.add_argument("--check", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    options = parser.parse_args(arguments)
    if options.check:
        store = certificates.CertificateStore.load(STORE_PATH)
        found = problems(store)
        print(json.dumps({"ok": not found, "problems": found}, indent=2))
        return 1 if found else 0
    cert, measurements = certify_leakage(options.workers)
    store = certificates.CertificateStore(
        [cert, certify_asof(), certify_reconciliation()]
    )
    store.save(STORE_PATH)
    MEASUREMENTS_PATH.write_text(
        json.dumps(measurements, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(measurements, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
