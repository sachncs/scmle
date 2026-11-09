"""Certification of the data detectors (missingness, labels, structure).

Plants defects in a real table (scikit-learn's breast-cancer data, bundled,
no download), runs each detector on the planted copies and on the clean
data, and files the counts as certificates in
`detector_certificates.json`.

Pre-declared policy, fixed before any measurement and not tuned to results:
recall lower Wilson bound at least 0.80 and false-alarm upper Wilson bound
at most 0.05, 95 percent intervals. Development used seeds
100 and above; certification uses the seeds listed below, which are disjoint.

Units. Row detectors (sentinels, duplicates, label noise, outlier error
class) are judged per row: planted rows found over planted rows, flagged
clean rows over clean rows. Rows within one table are not independent
draws, so these intervals are narrower than a replicate-level interval
would be. The missingness detector makes one decision per table, so it is
judged per replicate: planted = replicate tables with the mechanism,
detected = tables where MCAR is rejected, false alarms = rejections among
MCAR replicates (the null). Its `clean_rows` field therefore counts MCAR
replicates, not rows.

Honest limits. Plants are synthetic and easier than real defects, so recall
is optimistic. The "clean" table is real and may hold native defects (zero
inflation, label noise), which count as false alarms. Outlier provenance is
taken from the clean data's own range, which assumes the provenance is right.
MNAR cannot be identified from data: the missingness certificate for
`missing_mnar` measures rejection of MCAR when missingness depends on the
value itself and correlated columns carry that signal, not detection of MNAR
as such. One corpus is used; other dataset families are uncertified.
"""

from collections.abc import Callable
from collections.abc import Sequence
import dataclasses
from pathlib import Path

import pandas as pd

from benchmarks.qualification import certify
from benchmarks.qualification import inject
from benchmarks.qualification import standins
from scmle.contracts import qualification
from scmle.core import randomness
from scmle.data.detectors import certificates
from scmle.data.detectors import duplicates
from scmle.data.detectors import labels
from scmle.data.detectors import missingness
from scmle.data.detectors import outliers
from scmle.data.detectors import registry
from scmle.data.detectors import sentinels

CERT_PATH = Path(__file__).with_name("detector_certificates.json")
TARGET = standins.TARGET
POLICY = certificates.CertificationPolicy(
    recall_floor=0.80, false_alarm_ceiling=0.05, confidence=0.95
)
ROW_SEEDS = (0, 1, 2, 3, 4)
MISSING_SEEDS = tuple(range(200))
MISSING_COLUMNS = 6
SENTINEL_COLUMNS = ["mean radius", "mean texture"]
OUTLIER_COLUMNS = ["mean radius", "mean texture", "mean perimeter"]
OUTLIER_PAD = 0.10
CUT_GRID = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1)
CLASS_RATES = {0: 0.20, 1: 0.05}


def label_detector(frame: pd.DataFrame) -> pd.Series:
    """Rows both label-noise detectors flag (the gating rule).

    Args:
        frame: Table with feature columns and the `target` label column.

    Returns:
        A boolean Series with the index of `frame`; True for the rows in the
        label-noise report's suspects.
    """
    report = labels.detect(frame.drop(columns=[TARGET]), frame[TARGET], 0)
    flags = pd.Series(False, index=frame.index)
    flags.iloc[list(report.suspects)] = True
    return flags


def provenance_for(frame: pd.DataFrame) -> dict[str, outliers.FieldProvenance]:
    """Instrument ranges from the clean table, padded by `OUTLIER_PAD`.

    Args:
        frame: The clean table.

    Returns:
        For each column in `OUTLIER_COLUMNS`, a provenance whose instrument
        range is the column's minimum and maximum widened by 10 percent of
        the span on each side.
    """
    out = {}
    for name in OUTLIER_COLUMNS:
        low, high = float(frame[name].min()), float(frame[name].max())
        pad = OUTLIER_PAD * (high - low)
        out[name] = outliers.FieldProvenance(
            instrument_range=(low - pad, high + pad)
        )
    return out


def renamed(plan: certify.DefectPlan, name: str) -> certify.DefectPlan:
    """The same plan under a different certificate key.

    Args:
        plan: The plan to rename.
        name: The new defect type.

    Returns:
        A plan that plants as `plan` does but labels the injection `name`.
    """

    def plant(frame: pd.DataFrame, seed: int) -> inject.Injection:
        """Plants with the original plan, then renames the defect type.

        Args:
            frame: Table to plant into.
            seed: Run seed.

        Returns:
            The injection with `defect_type` replaced by the new name.
        """
        return dataclasses.replace(plan.plant(frame, seed), defect_type=name)

    return certify.DefectPlan(name, plant)


def row_plans() -> dict[str, list[certify.DefectPlan]]:
    """Plans per row-level detector id.

    Returns:
        Mapping of detector id to the defect plans it is measured on:
        sentinels, exact and near duplicates, symmetric and class-conditional
        label flips, and outliers.
    """
    return {
        "f5-sentinel": [
            certify.DefectPlan(
                "sentinel",
                lambda f, s: inject.inject_sentinels(
                    f, SENTINEL_COLUMNS, 0.10, s
                ),
            )
        ],
        "f5-duplicates": [
            certify.DefectPlan(
                "duplicate_exact",
                lambda f, s: inject.inject_duplicates(f, 0.10, s),
            ),
            certify.DefectPlan(
                "duplicate_near",
                lambda f, s: inject.inject_duplicates(
                    f, 0.10, s, near=True, scale=1e-3
                ),
            ),
        ],
        "f5-label-noise": [
            renamed(
                certify.DefectPlan(
                    "label_flip",
                    lambda f, s: inject.flip_labels(f, TARGET, 0.10, s),
                ),
                "label_flip_symmetric",
            ),
            renamed(
                certify.DefectPlan(
                    "label_flip",
                    lambda f, s: inject.flip_labels(
                        f,
                        TARGET,
                        0.0,
                        s,
                        mode="class_conditional",
                        class_rates=CLASS_RATES,
                    ),
                ),
                "label_flip_class_conditional",
            ),
        ],
        "f5-outlier-triage": [
            renamed(
                certify.DefectPlan(
                    "outlier",
                    lambda f, s: inject.inject_outliers(
                        f, OUTLIER_COLUMNS, 0.10, s, magnitude=8.0
                    ),
                ),
                "outlier_error",
            )
        ],
    }


def row_detectors(
    frame: pd.DataFrame,
) -> dict[str, Callable[[pd.DataFrame], pd.Series]]:
    """Maps detector id to a function from frame to one flag per row.

    Args:
        frame: The clean table, from which outlier provenance is taken.

    Returns:
        Mapping of detector id to a function returning a boolean Series.
    """
    provenance = provenance_for(frame)
    return {
        "f5-sentinel": sentinels.sentinel_rows,
        "f5-duplicates": duplicates.duplicate_flags,
        "f5-label-noise": label_detector,
        "f5-outlier-triage": lambda f: outliers.error_flags(f, provenance),
    }


def certify_missingness(
    frame: pd.DataFrame,
    seeds: Sequence[int] = MISSING_SEEDS,
    policy: certificates.CertificationPolicy = POLICY,
) -> certificates.DetectorCertificate:
    """Replicate-level certificate for the MCAR rejection decision.

    Plants missing values by each mechanism in the first `MISSING_COLUMNS`
    columns, once per seed, and counts the tables on which MCAR is rejected.

    Args:
        frame: The clean table.
        seeds: Replicate seeds, one table each.
        policy: Pre-declared certification floors.

    Returns:
        The certificate with outcomes for MAR and MNAR plants; the MCAR
        rejections serve as the false alarms.
    """
    base = frame.iloc[:, :MISSING_COLUMNS].copy()
    names = list(base.columns)
    holes, driver = names[:3], names[-1]
    rejected: dict[str, int] = {}
    for mechanism in inject.MECHANISMS:
        count = 0
        for seed in seeds:
            planted = inject.inject_missing(
                base,
                holes,
                0.2,
                seed,
                mechanism=mechanism,
                driver=driver if mechanism == "mar" else None,
            )
            count += int(
                missingness.assess_mechanism(planted.frame, seed).mcar_rejected
            )
        rejected[mechanism] = count
    results = [
        qualification.Certificate(
            detector_id="f5-missingness",
            defect_type=f"missing_{mechanism}",
            planted=len(seeds),
            detected=rejected[mechanism],
            clean_rows=len(seeds),
            false_alarms=rejected["mcar"],
            seed_path=randomness.seed_path(
                seeds[0], "certify.f5-missingness", 0
            ),
        )
        for mechanism in ("mar", "mnar")
    ]
    return certificates.certify(
        "f5-missingness",
        registry.SPECS["f5-missingness"].code_hash(),
        certificates.corpus_hash([certify.frame_digest(base)]),
        seeds,
        results,
        policy,
    )


def select_cut(
    frame: pd.DataFrame,
    policy: certificates.CertificationPolicy = POLICY,
) -> float:
    """Largest grid cut whose clean-copy false-alarm bound meets the policy.

    The shipped default is one decade below this value, so the default keeps
    a margin on data other than the certification corpus.

    Args:
        frame: The clean table, screened for near duplicates against itself.
        policy: Certification floors; the false-alarm ceiling and confidence
            are used.

    Returns:
        The largest cut in `CUT_GRID` whose upper Wilson bound on clean
        rows flagged is within the ceiling; the smallest cut when none is.
    """
    best = CUT_GRID[0]
    columns = list(frame.columns)
    for cut in CUT_GRID:
        flagged = int(duplicates.near_flags(frame, columns, cut).sum())
        upper = certificates.wilson_interval(
            flagged, len(frame), policy.confidence
        )[1]
        if upper <= policy.false_alarm_ceiling:
            best = cut
    return best


def certify_all(
    seeds: Sequence[int] = ROW_SEEDS,
    missing_seeds: Sequence[int] = MISSING_SEEDS,
    frame: pd.DataFrame | None = None,
    only: Sequence[str] | None = None,
) -> certificates.CertificateStore:
    """Certifies the data detectors on the demo table.

    Args:
        seeds: Run seeds for the row detectors; counts are pooled.
        missing_seeds: Replicate seeds for the missingness detector.
        frame: Table to plant into; defaults to the breast-cancer data.
        only: Restrict to these detector ids.

    Returns:
        A store with one certificate per detector.
    """
    data = standins.demo_frame() if frame is None else frame
    store = certificates.CertificateStore()
    plans = row_plans()
    detectors = row_detectors(data)
    for detector_id, detector in detectors.items():
        if only is not None and detector_id not in only:
            continue
        spec = registry.SPECS[detector_id]
        store.add(
            certify.certify_detector(
                detector_id,
                detector,
                spec.code_hash(),
                data,
                plans[detector_id],
                seeds,
                POLICY,
            )
        )
    if only is None or "f5-missingness" in only:
        store.add(certify_missingness(data, missing_seeds))
    return store


def problems(store: certificates.CertificateStore) -> list[str]:
    """Why a store does not hold current certificates for every detector.

    Args:
        store: Certificate store to audit.

    Returns:
        One message per registered detector whose certificate is not current
        for its present code; empty when all are current.
    """
    found = []
    for detector_id, spec in registry.SPECS.items():
        state = store.state(detector_id, spec.code_hash())
        if state is not certificates.CertificateState.CURRENT:
            found.append(f"{detector_id}: certificate is {state.value}")
    return found


def gating(
    store: certificates.CertificateStore,
) -> list[tuple[str, str, str]]:
    """(detector id, code hash, defect type) for every passing outcome.

    Args:
        store: Certificate store to read.

    Returns:
        One tuple per defect type whose current certificate passed its floors.
    """
    registered = []
    for detector_id, spec in registry.SPECS.items():
        digest = spec.code_hash()
        for item in store.current(detector_id, digest):
            for defect in item.certified_defect_types:
                registered.append((detector_id, digest, defect))
    return registered


def audit_problems(store: certificates.CertificateStore) -> list[str]:
    """Registered gating pairs that `mark_gating` would refuse.

    Args:
        store: Certificate store to audit.

    Returns:
        A message for each passing pair that `mark_gating` rejects with a
        `CertificateError`; empty when all are accepted.
    """
    found = []
    for detector_id, digest, defect in gating(store):
        try:
            store.mark_gating(detector_id, digest, defect)
        except certificates.CertificateError as error:
            found.append(f"{detector_id}/{defect}: {error}")
    return found


def summary(store: certificates.CertificateStore) -> list[dict[str, object]]:
    """One line per detector and defect type with the bounds and verdict.

    Args:
        store: Certificate store to read.

    Returns:
        Rows with the detector, defect type, planted and detected counts,
        the lower recall bound, clean rows, false alarms, the upper
        false-alarm bound (both rounded to four decimals) and whether the
        outcome passed its floors.
    """
    rows: list[dict[str, object]] = []
    for item in store.certificates:
        for outcome in item.outcomes:
            cert = outcome.certificate
            rows.append(
                {
                    "detector": item.detector_id,
                    "defect_type": cert.defect_type,
                    "planted": cert.planted,
                    "detected": cert.detected,
                    "recall_lower": round(outcome.recall_interval[0], 4),
                    "clean": cert.clean_rows,
                    "false_alarms": cert.false_alarms,
                    "false_alarm_upper": round(
                        outcome.false_alarm_interval[1], 4
                    ),
                    "gates": outcome.passed,
                }
            )
    return rows
