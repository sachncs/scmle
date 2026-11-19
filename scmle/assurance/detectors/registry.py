"""Which code each detector consists of, and what it may gate on.

A certificate is tied to a hash of the detector's source. Each detector here
is hashed over whole modules, so an edit to any helper makes its certificate
stale and the detector falls back to descriptive.
"""

import dataclasses
from types import ModuleType

from scmle.assurance.detectors import certificates
from scmle.assurance.detectors import duplicates
from scmle.assurance.detectors import labels
from scmle.assurance.detectors import minhash
from scmle.assurance.detectors import missingness
from scmle.assurance.detectors import outliers
from scmle.assurance.detectors import sentinels
import scmle.data.splitting.duplicates


@dataclasses.dataclass(frozen=True)
class Identity:
    """A detector's identity for certification and gating.

    Attributes:
        key: Key in the certificate store.
        modules: Modules whose source defines the detector.
        types: Defect types it is certified on; it may gate on a
            type only when that type's outcome passed.
    """

    key: str
    modules: tuple[ModuleType, ...]
    types: tuple[str, ...]

    def digest(self) -> str:
        """Hashes the source of the detector's modules.

        Returns:
            The code hash that a certificate for this detector must match.
        """
        return certificates.fingerprint(*self.modules)


SPECS: dict[str, Identity] = {
    spec.key: spec
    for spec in (
        Identity("f5-sentinel", (sentinels,), ("sentinel",)),
        Identity(
            "f5-duplicates",
            (duplicates, minhash, scmle.data.splitting.duplicates),
            ("duplicate_exact", "duplicate_near"),
        ),
        Identity(
            "f5-label-noise",
            (labels,),
            ("label_flip_symmetric", "label_flip_class_conditional"),
        ),
        Identity(
            "f5-missingness",
            (missingness,),
            ("missing_mar", "missing_mnar"),
        ),
        Identity("f5-outlier-triage", (outliers,), ("outlier_error",)),
    )
}

EXECUTION_ORDER = (
    "f5-sentinel",
    "f5-duplicates",
    "f5-missingness",
    "f5-label-noise",
    "f5-outlier-triage",
)
