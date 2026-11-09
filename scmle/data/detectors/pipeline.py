"""Runs the detectors in order and reports what each may gate.

Order: sentinels, duplicates, missingness, labels, outliers. Sentinels run
first and their codes are recoded to missing before any later detector sees
the table, because a code such as 999 hides in every other statistic.

Every detector's status is `gating` only if the certificate store holds a
current, passing certificate for the defect type; otherwise it is
`descriptive` and its decisions are limited to warnings.
"""

from collections.abc import Callable
from collections.abc import Mapping
import dataclasses
from typing import Any

import pandas as pd

from scmle.data.detectors import certificates
from scmle.data.detectors import duplicates
from scmle.data.detectors import labels
from scmle.data.detectors import missingness
from scmle.data.detectors import outliers
from scmle.data.detectors import registry
from scmle.data.detectors import sentinels


@dataclasses.dataclass(frozen=True)
class PipelineResult:
    """What the detectors found and how far each may be trusted.

    Attributes:
        order: Detector ids in the order they ran or were skipped.
        results: Detector id -> result object; a detector that did not run is
            absent.
        status: Detector id -> defect type -> gating status.
        recode_log: Sentinel recodes applied before later detectors.
        skipped: Detector id -> reason it did not run.
    """

    order: tuple[str, ...]
    results: Mapping[str, Any]
    status: Mapping[str, Mapping[str, certificates.GateStatus]]
    recode_log: tuple[dict[str, Any], ...]
    skipped: Mapping[str, str]


def gate_status(
    store: certificates.CertificateStore | None, detector_id: str
) -> dict[str, certificates.GateStatus]:
    """Looks up the gating status of each defect type of a detector.

    Args:
        store: Certificates on file, or None.
        detector_id: Key of the detector in `registry.SPECS`.

    Returns:
        Defect type -> gating status for the detector's current code. Every
        status is descriptive when there is no store.
    """
    spec = registry.SPECS[detector_id]
    digest = spec.code_hash()
    return {
        defect: (
            certificates.GateStatus.DESCRIPTIVE
            if store is None
            else store.status(detector_id, digest, defect)
        )
        for defect in spec.defect_types
    }


def run(
    frame: pd.DataFrame,
    target: str | None = None,
    store: certificates.CertificateStore | None = None,
    seed: int = 0,
    provenance: Mapping[str, outliers.FieldProvenance] | None = None,
) -> PipelineResult:
    """Runs all detectors on exploration rows.

    Args:
        frame: Exploration set `X` only.
        target: Label column, if any; label noise needs a categorical one.
        store: Certificates on file, or None (everything descriptive).
        seed: Run seed.
        provenance: Field provenance for outlier triage.

    Returns:
        The results in execution order with gating status per defect type.
    """
    results: dict[str, Any] = {}
    skipped: dict[str, str] = {}
    order: list[str] = []
    steps: dict[str, Callable[[pd.DataFrame], Any]] = {}
    log: list[dict[str, Any]] = []
    working = frame
    report = sentinels.find_sentinels(frame)
    results["f5-sentinel"] = report
    order.append("f5-sentinel")
    working, recoded = sentinels.recode_missing(frame, report)
    log.extend(recoded)
    steps["f5-duplicates"] = lambda f: duplicates.assess_duplicates(
        f, target=target
    )
    steps["f5-missingness"] = lambda f: missingness.assess_mechanism(f, seed)
    steps["f5-outlier-triage"] = lambda f: outliers.triage(f, provenance)
    for detector_id in registry.EXECUTION_ORDER[1:]:
        order.append(detector_id)
        if detector_id == "f5-label-noise":
            outcome = label_step(working, target, seed)
            if isinstance(outcome, str):
                skipped[detector_id] = outcome
            else:
                results[detector_id] = outcome
            continue
        results[detector_id] = steps[detector_id](working)
    status = {d: gate_status(store, d) for d in registry.SPECS}
    return PipelineResult(tuple(order), results, status, tuple(log), skipped)


def label_step(
    frame: pd.DataFrame, target: str | None, seed: int
) -> labels.LabelReport | str:
    """Runs the label-noise detector, or says why it was declined.

    Args:
        frame: Exploration rows, with the target column among them.
        target: Label column, or None.
        seed: Run seed.

    Returns:
        The label-noise report, or the reason it could not run: no target
        column, no numeric features, or the message of the `ValueError`
        the detector raised. Numeric features are median-filled first.
    """
    if target is None or target not in frame.columns:
        return "no target column"
    features = frame.drop(columns=[target])
    numeric = features.select_dtypes("number")
    if numeric.shape[1] == 0:
        return "no numeric features"
    try:
        return labels.detect(
            numeric.fillna(numeric.median()), frame[target], seed
        )
    except ValueError as error:
        return str(error)
