"""Command line for the verification harness.

    python -m benchmarks.qualification --smoke            # the Phase 1 gate
    python -m benchmarks.qualification --smoke --full     # nightly, longer
    python -m benchmarks.qualification --check            # certificates
    python -m benchmarks.qualification --check fdr        # claim ledger (F3)
    python -m benchmarks.qualification --check framing    # framing (F8)
    python -m benchmarks.qualification --check ingest     # ingestion (F9)
    python -m benchmarks.qualification --write-sample F   # stand-in sample
    python -m benchmarks.qualification --write-f5 F       # F5 certificates

`--check` fails when a gating detector has no current certificate. The F5
detectors (`scmle.data.detectors`) are checked against
`detector_certificates.json`: each must have a certificate for its current
code, and each outcome that passed must still be accepted for gating. An
outcome that did not pass is simply descriptive and does not fail the check.
The F6 checks (`scmle.data.robustness`) are checked the same way against
`f6_certificates.json`.
The F12 coverage and breadth gates (`uncertainty_certify.py`) are checked
against `uncertainty_certificates.json`: a certificate is stale when the code it
measured changed. A gate that missed its floor is listed as descriptive and
does not fail the check.
The F8 framing methods (`framing_certify.py`) are checked by `--check framing`
against `framing_certificates.json` and `framing_measurements.json`: stale
code, a `MIN_CLUSTERS` that differs from the measured minimum, or an accepted
malformed charter fails; a method that missed its floor is listed as
descriptive.
The leakage screen (`scmle.data.admissibility.screen`,
the test behind `leakage_policy="block"`) ships its certificates inside the
package (`screen_certificates.json`) and is checked the same way.
The F9 ingestion detectors (torn-read check, extract reconciliation) are checked
against `ingest_certificates.json`; `--check ingest` also verifies that
`ingest_measurements.json` was measured on the current code and re-runs a
small live point-in-time and torn-read check.
"""

import argparse
import json
from pathlib import Path

from benchmarks.qualification import f5
from benchmarks.qualification import f6_certify
from benchmarks.qualification import framing_certify
from benchmarks.qualification import ingest_certify
from benchmarks.qualification import ledger_checks
from benchmarks.qualification import screen_certify
from benchmarks.qualification import smoke
from benchmarks.qualification import standins
from benchmarks.qualification import uncertainty_certify
from scmle.data.detectors import certificates

# Detector id -> (current code hash, defect type it gates on).
GATING: dict[str, tuple[str, str]] = {}


def check_certificates() -> int:
    """Validates the sample store and every registered gating detector."""
    store = certificates.CertificateStore.load(standins.SAMPLE_PATH)
    problems = standins.sample_problems(store)
    report = certificates.audit(store, GATING)
    f5_store = certificates.CertificateStore.load(f5.CERT_PATH)
    f5_problems = f5.problems(f5_store) + f5.audit_problems(f5_store)
    f6_store = certificates.CertificateStore.load(f6_certify.STORE_PATH)
    f6_problems = f6_certify.problems(f6_store)
    uncertainty_problems = uncertainty_certify.problems()
    screen_store = certificates.CertificateStore.load(screen_certify.STORE_PATH)
    screen_problems = screen_certify.problems(screen_store)
    ingest_store = certificates.CertificateStore.load(ingest_certify.STORE_PATH)
    ingest_problems = ingest_certify.problems(ingest_store)
    ok = (
        not problems
        and report.ok
        and not f5_problems
        and not f6_problems
        and not uncertainty_problems
        and not screen_problems
        and not ingest_problems
    )
    print(
        json.dumps(
            {
                "ok": ok,
                "sample_problems": problems,
                "missing": list(report.missing),
                "stale": list(report.stale),
                "uncertified": list(report.uncertified),
                "gating_detectors": len(GATING),
                "f5_problems": f5_problems,
                "f5_gating": [f"{d}/{t}" for d, _, t in f5.gating(f5_store)],
                "uncertainty_problems": uncertainty_problems,
                "uncertainty_descriptive_only": (
                    uncertainty_certify.descriptive_only()
                ),
                "f6_problems": f6_problems,
                "f6_descriptive_only": f6_certify.audit_labels(f6_store)[
                    "uncertified"
                ],
                "screen_problems": screen_problems,
                "screen_descriptive_only": screen_certify.uncertified(
                    screen_store
                ),
                "ingest_problems": ingest_problems,
                "ingest_descriptive_only": ingest_certify.audit_labels(
                    ingest_store
                )["uncertified"],
            },
            indent=2,
        )
    )
    return 0 if ok else 1


def main(arguments: list[str] | None = None) -> int:
    """Runs the requested action and returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--smoke", action="store_true")
    action.add_argument(
        "--check",
        nargs="?",
        const="certificates",
        choices=("certificates", "fdr", "framing", "ingest"),
    )
    action.add_argument("--write-sample", type=Path, metavar="FILE")
    action.add_argument("--write-f5", type=Path, metavar="FILE")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--runs", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=400)
    options = parser.parse_args(arguments)
    if options.smoke:
        report = smoke.run_smoke(options.runs, options.seed, options.full)
        print(json.dumps(report.to_dict(), indent=2))
        return 0 if report.ok else 1
    if options.check == "fdr":
        findings = ledger_checks.run_all(
            options.runs, options.seed, options.repeats
        )
        summary = ledger_checks.report(findings)
        print(json.dumps(summary, indent=2))
        return 0 if summary["ok"] else 1
    if options.check == "ingest":
        ingested = ingest_certify.check()
        print(json.dumps(ingested, indent=2))
        return 0 if ingested["ok"] else 1
    if options.check == "framing":
        found = framing_certify.check()
        print(json.dumps(found, indent=2))
        return 0 if found["ok"] else 1
    if options.check:
        return check_certificates()
    if options.write_f5:
        store = f5.certify_all()
        store.save(options.write_f5)
        print(json.dumps(f5.summary(store), indent=2))
        return 0
    standins.certify_all().save(options.write_sample)
    print(json.dumps({"written": str(options.write_sample)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
