"""Which causal-identification gates have a current, passing certificate.

A gate may only count toward `confirmed` if its simulation certificate in
`scmle/certificates/f10_certificates.json` passed its pre-registered
floor and was measured on the current method code. A certificate stores a
`method_hash` over the package functions it measured; this module recomputes
it, so a changed estimator makes its gates descriptive until re-measured.
The certificate file ships with the package; without it no gate
is certified and every causal claim stays at most `attested`.
"""

import dataclasses
import json
from pathlib import Path
from typing import Any

from scmle.assurance.detectors import certificates
from scmle.assurance.ledger import multiplicity
from scmle.causal.identification import aipw
from scmle.causal.identification import battery
from scmle.causal.identification import did
from scmle.causal.identification import iv
from scmle.causal.identification import rd
from scmle.causal.identification import select
from scmle.causal.identification import sensitivity
from scmle.causal.identification import synth
from scmle.causal.identification import tier
import scmle.certificates

STORE_PATH = scmle.certificates.DIRECTORY / "f10_certificates.json"
METHOD_SOURCES: dict[str, list[Any]] = {
    "selector": [select.select, select.judge, select.verify],
    "coverage_aipw": [
        aipw.scores,
        aipw.estimate,
        aipw.prefit,
        aipw.spawn,
        aipw.Design,
    ],
    "coverage_randomized": [aipw.difference_in_means],
    "coverage_did": [did.estimate],
    "coverage_iv": [iv.estimate, iv.anderson_rubin, iv.residualise],
    "coverage_rd": [
        rd.estimate,
        rd.discontinuity,
        rd.polyfit,
        rd.crossvalidate,
    ],
    "size_synth": [synth.estimate, synth.weigh, synth.ratio_for],
    "overlap": [aipw.propensities],
    "iv_strength": [iv.estimate, iv.anderson_rubin],
    "density_rd": [rd.estimate],
    "battery_nco": [battery.run_battery, battery.wald, aipw.estimate],
    "battery_pretrend": [battery.run_battery, did.estimate],
    "battery_rd": [battery.run_battery, rd.estimate],
    "sensitivity_bound": [
        sensitivity.assess,
        sensitivity.regress,
        sensitivity.calibrate,
        sensitivity.robustify,
        sensitivity.ceiling,
    ],
    "sensitivity_did": [did.estimate, did.Estimate.sensitivity],
    "false_confirm": [tier.decide, battery.run_battery],
}
NOT_MEASURED = ("battery_iv", "battery_synth")


def method_hashes() -> dict[str, str]:
    """Hashes the package methods behind each gate.

    Returns:
        Gate name to the code hash of the functions listed for it in
        `METHOD_SOURCES`.
    """
    return {k: certificates.code_hash(*v) for k, v in METHOD_SOURCES.items()}


def certificated(path: Path | None = None) -> set[str]:
    """Lists the gates whose stored certificate passed and is current.

    Args:
        path: Certificate file; the shipped one when omitted.

    Returns:
        The names of the gates whose stored entry passed and whose method
        hash equals the hash of the current code. The set is empty when the
        file does not exist.
    """
    target = STORE_PATH if path is None else path
    if not target.exists():
        return set()
    document = json.loads(target.read_text(encoding="utf-8"))
    current = method_hashes()
    return {
        name
        for name, entry in document.get("gates", {}).items()
        if entry.get("passed") and entry.get("method_hash") == current.get(name)
    }


SUBGROUP_STORE_PATH = STORE_PATH.with_name("subgroup_certificates.json")
SUBGROUP_GATE = "subgroup_holm"
GATING = "gating"
DESCRIPTIVE = "descriptive"


@dataclasses.dataclass(frozen=True, slots=True)
class SubgroupGate:
    """Whether the Holm family of subgroup estimates is certified.

    Attributes:
        status: `gating` when the stored certificate passed its
            pre-registered floors on the current code, else `descriptive`.
        reason: One sentence saying why.
        min_rows: The smallest subgroup size (rows of `C` in the subgroup)
            at which the power floor was met, or None.
    """

    status: str
    reason: str
    min_rows: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Builds the JSON-ready record.

        Returns:
            A mapping of the status, reason and minimum rows.
        """
        return dataclasses.asdict(self)


def fingerprint(report: Any) -> str:
    """Hashes the code a subgroup certificate measured.

    Args:
        report: `scmle.causal.identification.run.stratify`, passed in
            because that module imports this one's loader.

    Returns:
        The code hash of the report function, the Wald test and the
        multiplicity adjustment.
    """
    return certificates.code_hash(report, battery.wald, multiplicity.adjust)


def subgroup_gate(report: Any, path: Path | None = None) -> SubgroupGate:
    """Decides whether subgroup results gate or only describe.

    The subgroup family never changes the tier (the tier rule has no
    subgroup input); this decides only which word the report carries.

    Args:
        report: `scmle.causal.identification.run.stratify`.
        path: Certificate file; the shipped one when omitted.

    Returns:
        `gating` with the smallest certified subgroup size when the stored
        certificate exists, matches the current code and passed; otherwise
        `descriptive` with the reason.
    """
    target = SUBGROUP_STORE_PATH if path is None else path
    if not target.exists():
        return SubgroupGate(DESCRIPTIVE, "no subgroup certificate file")
    document = json.loads(target.read_text(encoding="utf-8"))
    entry = document.get("gates", {}).get(SUBGROUP_GATE)
    if entry is None:
        return SubgroupGate(DESCRIPTIVE, "no subgroup certificate")
    if entry.get("method_hash") != fingerprint(report):
        return SubgroupGate(
            DESCRIPTIVE, "stale: the code changed since it was measured"
        )
    if not entry.get("passed"):
        return SubgroupGate(
            DESCRIPTIVE, "the certificate missed its pre-registered floor"
        )
    return SubgroupGate(
        GATING,
        "family-wise error and power met their pre-registered floors",
        entry.get("min_certified_rows"),
    )
