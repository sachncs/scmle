"""Which F10 gates have a current, passing certificate (canon P1).

A gate may only count toward `confirmed` if its simulation certificate in
`benchmarks/qualification/f10_certificates.json` passed its pre-registered
floor and was measured on the current method code. A certificate stores a
`method_hash` over the package functions it measured; this module recomputes
it, so a changed estimator makes its gates descriptive until re-measured.
The certificate file is not part of the installed package: without it no gate
is certified and every causal claim stays at most `attested`.
"""

import dataclasses
import json
from pathlib import Path
from typing import Any

from scmle.data.detectors import certificates
from scmle.data.identification import aipw
from scmle.data.identification import battery
from scmle.data.identification import did
from scmle.data.identification import iv
from scmle.data.identification import rd
from scmle.data.identification import select
from scmle.data.identification import sensitivity
from scmle.data.identification import synth
from scmle.data.identification import tier
from scmle.data.ledger import multiplicity

STORE_PATH = (
    Path(__file__).resolve().parents[3]
    / "benchmarks"
    / "qualification"
    / "f10_certificates.json"
)
METHOD_SOURCES: dict[str, list[Any]] = {
    "selector": [select.select, select.judge, select.check_adjustment],
    "coverage_aipw": [
        aipw.scores,
        aipw.estimate,
        aipw.fit_nuisances,
        aipw.make_learners,
        aipw.Design,
    ],
    "coverage_randomized": [aipw.difference_in_means],
    "coverage_did": [did.estimate],
    "coverage_iv": [iv.estimate, iv.anderson_rubin, iv.residualise],
    "coverage_rd": [
        rd.estimate,
        rd.discontinuity,
        rd.local_fit,
        rd.cv_bandwidth,
    ],
    "size_synth": [synth.estimate, synth.fit_weights, synth.ratio_for],
    "overlap": [aipw.overlap_report],
    "iv_strength": [iv.estimate, iv.anderson_rubin],
    "density_rd": [rd.estimate],
    "battery_nco": [battery.run_battery, battery.wald_test, aipw.estimate],
    "battery_pretrend": [battery.run_battery, did.estimate],
    "battery_rd": [battery.run_battery, rd.estimate],
    "sensitivity_bound": [
        sensitivity.assess,
        sensitivity.ols_fit,
        sensitivity.benchmark_strength,
        sensitivity.adjusted_interval,
        sensitivity.bias_bound,
    ],
    "sensitivity_did": [did.estimate, did.DidResult.sensitivity],
    "false_confirm": [tier.decide, battery.run_battery],
}
NOT_MEASURED = ("battery_iv", "battery_synth")


def method_hashes() -> dict[str, str]:
    """Code hash of the package methods behind each gate."""
    return {k: certificates.code_hash(*v) for k, v in METHOD_SOURCES.items()}


def certified_gates(path: Path | None = None) -> set[str]:
    """Gates whose stored certificate passed and matches the current code."""
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
        """JSON-ready record."""
        return dataclasses.asdict(self)


def subgroup_method_hash(report: Any) -> str:
    """Hash of the code a subgroup certificate measured.

    Args:
        report: `scmle.data.identification.run.subgroup_report`, passed in
            because that module imports this one's loader.
    """
    return certificates.code_hash(
        report, battery.wald_test, multiplicity.adjust
    )


def subgroup_gate(report: Any, path: Path | None = None) -> SubgroupGate:
    """Whether subgroup results gate or only describe.

    The subgroup family never changes the tier (the tier rule has no
    subgroup input); this decides only which word the report carries.

    Args:
        report: `scmle.data.identification.run.subgroup_report`.
        path: Certificate file; the shipped one when omitted.
    """
    target = SUBGROUP_STORE_PATH if path is None else path
    if not target.exists():
        return SubgroupGate(DESCRIPTIVE, "no subgroup certificate file")
    document = json.loads(target.read_text(encoding="utf-8"))
    entry = document.get("gates", {}).get(SUBGROUP_GATE)
    if entry is None:
        return SubgroupGate(DESCRIPTIVE, "no subgroup certificate")
    if entry.get("method_hash") != subgroup_method_hash(report):
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
