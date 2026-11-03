"""Command line for the verification harness.

    python -m benchmarks.qualification --smoke            # the Phase 1 gate
    python -m benchmarks.qualification --smoke --full     # nightly, longer
    python -m benchmarks.qualification --check            # certificates
    python -m benchmarks.qualification --write-sample F   # stand-in sample

`--check` fails when a gating detector has no current certificate. No real
detector may gate yet, so `GATING` is empty; Phase 4 registers detectors
here as they are certified.
"""

import argparse
import json
from pathlib import Path

from benchmarks.qualification import smoke
from benchmarks.qualification import standins
from scmle.data.detectors import certificates

# Detector id -> (current code hash, defect type it gates on).
GATING: dict[str, tuple[str, str]] = {}


def check_certificates() -> int:
    """Validates the sample store and every registered gating detector."""
    store = certificates.CertificateStore.load(standins.SAMPLE_PATH)
    problems = standins.sample_problems(store)
    report = certificates.audit(store, GATING)
    ok = not problems and report.ok
    print(
        json.dumps(
            {
                "ok": ok,
                "sample_problems": problems,
                "missing": list(report.missing),
                "stale": list(report.stale),
                "uncertified": list(report.uncertified),
                "gating_detectors": len(GATING),
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
    action.add_argument("--check", action="store_true")
    action.add_argument("--write-sample", type=Path, metavar="FILE")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--runs", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    options = parser.parse_args(arguments)
    if options.smoke:
        report = smoke.run_smoke(options.runs, options.seed, options.full)
        print(json.dumps(report.to_dict(), indent=2))
        return 0 if report.ok else 1
    if options.check:
        return check_certificates()
    standins.certify_all().save(options.write_sample)
    print(json.dumps({"written": str(options.write_sample)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
