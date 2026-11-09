"""Which code each detector consists of, and what it may gate on.

A certificate is tied to a hash of the detector's source. Each detector here
is hashed over whole modules, so an edit to any helper makes its certificate
stale and the detector falls back to descriptive.
"""

import dataclasses
from types import ModuleType

from scmle.data.detectors import certificates
from scmle.data.detectors import duplicates
from scmle.data.detectors import labels
from scmle.data.detectors import missingness
from scmle.data.detectors import outliers
from scmle.data.detectors import sentinels
from scmle.data.splitting import duplicates as split_duplicates


@dataclasses.dataclass(frozen=True)
class DetectorSpec:
    """A detector's identity for certification and gating.

    Attributes:
        detector_id: Key in the certificate store.
        modules: Modules whose source defines the detector.
        defect_types: Defect types it is certified on; it may gate on a
            type only when that type's outcome passed.
    """

    detector_id: str
    modules: tuple[ModuleType, ...]
    defect_types: tuple[str, ...]

    def code_hash(self) -> str:
        """Hashes the source of the detector's modules.

        Returns:
            The code hash that a certificate for this detector must match.
        """
        return certificates.code_hash(*self.modules)


SPECS: dict[str, DetectorSpec] = {
    spec.detector_id: spec
    for spec in (
        DetectorSpec("f5-sentinel", (sentinels,), ("sentinel",)),
        DetectorSpec(
            "f5-duplicates",
            (duplicates, split_duplicates),
            ("duplicate_exact", "duplicate_near"),
        ),
        DetectorSpec(
            "f5-label-noise",
            (labels,),
            ("label_flip_symmetric", "label_flip_class_conditional"),
        ),
        DetectorSpec(
            "f5-missingness",
            (missingness,),
            ("missing_mar", "missing_mnar"),
        ),
        DetectorSpec("f5-outlier-triage", (outliers,), ("outlier_error",)),
    )
}

EXECUTION_ORDER = (
    "f5-sentinel",
    "f5-duplicates",
    "f5-missingness",
    "f5-label-noise",
    "f5-outlier-triage",
)
