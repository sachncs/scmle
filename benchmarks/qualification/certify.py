"""Measures a detector against planted defects and issues its certificate.

For each defect type the harness plants the defect in a copy of real data,
runs the detector on the planted copy and on the clean data, and counts
planted rows found and clean rows flagged. Detection is judged per row: a
planted row (an appended copy, or a row with a changed cell) counts as found
when the detector flags that row. Counts are pooled over the seeds and judged
by Wilson bounds against the pre-registered policy (see
`scmle.data.detectors.certificates`). The clean data are scored once, not
once per seed: a detector sees the same rows every time, so repeating them
would only shrink the false-alarm interval without adding evidence.

Plants are synthetic, so the recall measured here is optimistic. The result
is kept per defect type, never pooled into one number.
"""

from collections.abc import Callable
from collections.abc import Sequence
import dataclasses
import hashlib

import pandas as pd

from benchmarks.qualification import inject
from scmle.contracts import qualification
from scmle.core import randomness
from scmle.data.detectors import certificates

Detector = Callable[[pd.DataFrame], pd.Series]
Planter = Callable[[pd.DataFrame, int], inject.Injection]
CSV_FLOAT_FORMAT = "%.17g"


@dataclasses.dataclass(frozen=True)
class DefectPlan:
    """How to plant one kind of defect.

    Attributes:
        defect_type: Certificate key; it must equal the injection's
            `defect_type`.
        plant: Plants the defect in a frame for a given run seed.
    """

    defect_type: str
    plant: Planter


def frame_digest(frame: pd.DataFrame) -> str:
    """SHA-256 of a table's values, independent of library hash functions."""
    text = frame.to_csv(index=True, float_format=CSV_FLOAT_FORMAT)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def flags_for(detector: Detector, frame: pd.DataFrame, what: str) -> pd.Series:
    """Runs a detector and checks that it returned one boolean per row.

    Raises:
        ValueError: If the output is not a boolean Series aligned to `frame`.
    """
    flags = detector(frame)
    if (
        not isinstance(flags, pd.Series)
        or flags.dtype != bool
        or not flags.index.equals(frame.index)
    ):
        raise ValueError(
            f"detector flags for the {what} must be a boolean Series "
            "aligned to the frame"
        )
    return flags


def certify_detector(
    detector_id: str,
    detector: Detector,
    code_digest: str,
    frame: pd.DataFrame,
    plans: Sequence[DefectPlan],
    seeds: Sequence[int],
    policy: certificates.CertificationPolicy,
    created: str | None = None,
) -> certificates.DetectorCertificate:
    """Plants defects, runs the detector and issues a certificate.

    Args:
        detector_id: Name the certificate is filed under.
        detector: Maps a frame to one boolean flag per row.
        code_digest: `certificates.code_hash` of the detector's code.
        frame: Clean real data to plant into.
        plans: One plan per defect type.
        seeds: Run seeds; counts are pooled over them.
        policy: Pre-registered floors.
        created: Optional ISO timestamp supplied by the caller.

    Returns:
        The certificate with a verdict per defect type.

    Raises:
        ValueError: On no plans or seeds, a repeated defect type, a plant
            whose type disagrees with its plan, or malformed detector output.
    """
    if not plans:
        raise ValueError("at least one plan is required")
    if not seeds:
        raise ValueError("at least one seed is required")
    kinds = [plan.defect_type for plan in plans]
    if len(set(kinds)) != len(kinds):
        raise ValueError("defect types must be unique")
    clean_flags = flags_for(detector, frame, "clean data")
    results: list[qualification.Certificate] = []
    for plan in plans:
        planted = detected = 0
        for seed in seeds:
            injection = plan.plant(frame, seed)
            if injection.defect_type != plan.defect_type:
                raise ValueError(
                    f"plan '{plan.defect_type}' planted "
                    f"'{injection.defect_type}'"
                )
            flags = flags_for(detector, injection.frame, plan.defect_type)
            truth = injection.row_mask
            planted += int(truth.sum())
            detected += int((flags & truth).sum())
        results.append(
            qualification.Certificate(
                detector_id=detector_id,
                defect_type=plan.defect_type,
                planted=planted,
                detected=detected,
                clean_rows=len(frame),
                false_alarms=int(clean_flags.sum()),
                seed_path=randomness.seed_path(
                    seeds[0], f"certify.{detector_id}", 0
                ),
            )
        )
    return certificates.certify(
        detector_id,
        code_digest,
        certificates.corpus_hash([frame_digest(frame)]),
        seeds,
        results,
        policy,
        created,
    )
