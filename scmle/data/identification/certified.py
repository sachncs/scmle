"""Which F10 gates have a current, passing certificate (canon P1).

A gate may only count toward `confirmed` if its simulation certificate in
`benchmarks/qualification/f10_certificates.json` passed its pre-registered
floor and was measured on the current method code. A certificate stores a
`method_hash` over the package functions it measured; this module recomputes
it, so a changed estimator makes its gates descriptive until re-measured.
The certificate file is not part of the installed package: without it no gate
is certified and every causal claim stays at most `attested`.
"""

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
