"""Certification of the data detectors (missingness, labels, structure).

Plants defects in a real table (scikit-learn's breast-cancer data, bundled,
no download), runs each detector on the planted copies and on the clean
data, and files the counts as certificates in
`detector_certificates.json`.

Policy, the plan's rule for planted defects (plan section 7, exploratory
analysis): recall lower bound at least `profile_recall_min` (0.90) and
false-alarm upper bound at most `profile_false_max` (0.05), both read from
`acceptance.toml`, each the one-sided exact (Clopper-Pearson) bound at
`alpha` of that file (the two-sided interval at level 1 - 2 `alpha`), the
rule every other certificate uses. An earlier version used a 0.80 recall
floor, looser than the plan, and then two-sided Wilson intervals at level
1 - `alpha`; Wilson is not exact and can fall below the nominal coverage
near 0 and 1, where these rates sit.
Development used seeds 100 and above; certification uses the seeds listed
below, which are disjoint.

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

import pandas

from benchmarks import acceptance
from benchmarks.qualification import certify
from benchmarks.qualification import inject
from benchmarks.qualification import standins
from scmle.assurance.detectors import certificates
from scmle.assurance.detectors import duplicates
from scmle.assurance.detectors import labels
from scmle.assurance.detectors import missingness
from scmle.assurance.detectors import outliers
from scmle.assurance.detectors import registry
from scmle.assurance.detectors import sentinels
import scmle.certificates
from scmle.contracts import qualification
from scmle.runtime import randomness

CERT_PATH = scmle.certificates.DIRECTORY / "detector_certificates.json"
VARIABLES = acceptance.load()
POLICY = certificates.Standard(
    recall=VARIABLES["profile_recall_min"].value,
    alarms=VARIABLES["profile_false_max"].value,
    confidence=1.0 - 2 * VARIABLES["alpha"].value,
    method="exact",
)
ROW_SEEDS = (0, 1, 2, 3, 4)
MISSING_SEEDS = tuple(range(200))
MISSING_COLUMNS = 6
SENTINEL_COLUMNS = ["mean radius", "mean texture"]
OUTLIER_COLUMNS = ["mean radius", "mean texture", "mean perimeter"]
OUTLIER_PAD = 0.10
CUT_GRID = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1)
CLASS_RATES = {0: 0.20, 1: 0.05}


def label_detector(frame: pandas.DataFrame) -> pandas.Series:
    """Rows both label-noise detectors flag (the gating rule).

    Args:
        frame: Table with feature columns and the `target` label column.

    Returns:
        A boolean Series with the index of `frame`; True for the rows in the
        label-noise report's suspects.
    """
    report = labels.detect(
        frame.drop(columns=[standins.TARGET]), frame[standins.TARGET], 0
    )
    flags = pandas.Series(data=False, index=frame.index)
    flags.iloc[list(report.suspects)] = True
    return flags


def provenance_for(
    frame: pandas.DataFrame,
) -> dict[str, outliers.Pedigree]:
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
        out[name] = outliers.Pedigree(instrument_range=(low - pad, high + pad))
    return out


def renamed(plan: certify.DefectPlan, name: str) -> certify.DefectPlan:
    """The same plan under a different certificate key.

    Args:
        plan: The plan to rename.
        name: The new defect type.

    Returns:
        A plan that plants as `plan` does but labels the injection `name`.
    """

    def plant(frame: pandas.DataFrame, seed: int) -> inject.Injection:
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
                    lambda f, s: inject.flip_labels(
                        f, standins.TARGET, 0.10, s
                    ),
                ),
                "label_flip_symmetric",
            ),
            renamed(
                certify.DefectPlan(
                    "label_flip",
                    lambda f, s: inject.flip_labels(
                        f,
                        standins.TARGET,
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
    frame: pandas.DataFrame,
) -> dict[str, Callable[[pandas.DataFrame], pandas.Series]]:
    """Maps detector id to a function from frame to one flag per row.

    Args:
        frame: The clean table, from which outlier provenance is taken.

    Returns:
        Mapping of detector id to a function returning a boolean Series.
    """
    provenance = provenance_for(frame)
    return {
        "f5-sentinel": sentinels.placeholders,
        "f5-duplicates": duplicates.duplicate_flags,
        "f5-label-noise": label_detector,
        "f5-outlier-triage": lambda f: outliers.misrecorded(f, provenance),
    }


def certify_missingness(
    frame: pandas.DataFrame,
    seeds: Sequence[int] = MISSING_SEEDS,
    policy: certificates.Standard = POLICY,
) -> certificates.Certification:
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
                missingness.investigate(planted.frame, seed).mcar_rejected
            )
        rejected[mechanism] = count
    results = [
        qualification.Certificate(
            detector="f5-missingness",
            defect=f"missing_{mechanism}",
            planted=len(seeds),
            detected=rejected[mechanism],
            clean=len(seeds),
            alarms=rejected["mcar"],
            seeds=randomness.seed_path(seeds[0], "certify.f5-missingness", 0),
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
    frame: pandas.DataFrame,
    policy: certificates.Standard = POLICY,
) -> float:
    """Largest grid cut whose clean-copy false-alarm bound meets the policy.

    The shipped default is one decade below this value, so the default keeps
    a margin on data other than the certification corpus.

    Args:
        frame: The clean table, screened for near duplicates against itself.
        policy: Certification floors; the false-alarm ceiling and confidence
            are used.

    Returns:
        The largest cut in `CUT_GRID` whose upper bound on clean rows
        flagged, by the policy's interval method, is within the ceiling; the
        smallest cut when none is.
    """
    best = CUT_GRID[0]
    columns = list(frame.columns)
    for cut in CUT_GRID:
        flagged = int(duplicates.near_flags(frame, columns, cut).sum())
        upper = certificates.interval(
            flagged, len(frame), policy.confidence, policy.method
        )[1]
        if upper <= policy.alarms:
            best = cut
    return best


def certify_all(
    seeds: Sequence[int] = ROW_SEEDS,
    missing_seeds: Sequence[int] = MISSING_SEEDS,
    frame: pandas.DataFrame | None = None,
    only: Sequence[str] | None = None,
) -> certificates.Store:
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
    store = certificates.Store()
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


def problems(store: certificates.Store) -> list[str]:
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
        if state is not certificates.Freshness.CURRENT:
            found.append(f"{detector_id}: certificate is {state.value}")
    return found


def gating(
    store: certificates.Store,
) -> list[tuple[str, str, str]]:
    """(detector id, code hash, defect type) for every passing outcome.

    Args:
        store: Certificate store to read.

    Returns:
        One tuple per defect type whose current certificate passed its floors.
    """
    registered: list[tuple[str, str, str]] = []
    for detector_id, spec in registry.SPECS.items():
        digest = spec.code_hash()
        registered.extend(
            (detector_id, digest, defect)
            for item in store.current(detector_id, digest)
            for defect in item.certified_defect_types
        )
    return registered


def audit_problems(store: certificates.Store) -> list[str]:
    """Registered gating pairs that `authorise` would refuse.

    Args:
        store: Certificate store to audit.

    Returns:
        A message for each passing pair that `authorise` rejects with a
        `CertificateError`; empty when all are accepted.
    """
    found = []
    for detector_id, digest, defect in gating(store):
        try:
            store.authorise(detector_id, digest, defect)
        except certificates.CertificateError as error:
            found.append(f"{detector_id}/{defect}: {error}")
    return found


def summary(store: certificates.Store) -> list[dict[str, object]]:
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
                    "detector": item.detector,
                    "defect_type": cert.defect,
                    "planted": cert.planted,
                    "detected": cert.detected,
                    "recall_lower": round(outcome.recall[0], 4),
                    "clean": cert.clean,
                    "false_alarms": cert.alarms,
                    "false_alarm_upper": round(outcome.alarms[1], 4),
                    "gates": outcome.passed,
                }
            )
    return rows
