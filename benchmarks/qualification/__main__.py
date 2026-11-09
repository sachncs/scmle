"""Command line for the verification harness.

    python -m benchmarks.qualification --smoke            # the smoke gate
    python -m benchmarks.qualification --smoke --full     # nightly, longer
    python -m benchmarks.qualification --check            # certificates
    python -m benchmarks.qualification --check fdr        # claim ledger
    python -m benchmarks.qualification --check ingest     # ingestion
    python -m benchmarks.qualification --check framing    # framing
    python -m benchmarks.qualification --check loop       # closed loop
    python -m benchmarks.qualification --check f10        # causal gates
    python -m benchmarks.qualification --check f11        # decision layer
    python -m benchmarks.qualification --check all        # every family above
    python -m benchmarks.qualification --write-sample F   # stand-in sample
    python -m benchmarks.qualification --write-f5 F       # detector certs

`--check` fails when a gating detector has no current certificate. The data
detectors (`scmle.data.detectors`) are checked against
`detector_certificates.json`: each must have a certificate for its current
code, and each outcome that passed must still be accepted for gating. An
outcome that did not pass is simply descriptive and does not fail the check.
The robustness checks (`scmle.data.robustness`) are checked the same way
against `f6_certificates.json`. The leakage screen
(`scmle.data.admissibility.screen`, the test behind
`leakage_policy="block"`) ships its certificates inside the package
(`screen_certificates.json`) and is checked the same way. The ingestion
detectors (torn-read check, extract reconciliation) are checked against
`ingest_certificates.json`; `--check ingest` also verifies that
`ingest_measurements.json` was measured on the current code and re-runs a
small live point-in-time and torn-read check.
The coverage and breadth gates (`uncertainty_certify.py`) are checked
against `uncertainty_certificates.json`: a certificate is stale when the code
it measured changed. A gate that missed its floor is listed as descriptive and
does not fail the check.
The Holm family of subgroup hypotheses (`subgroup_certify.py`) is checked
against `subgroup_certificates.json`: a certificate is stale when the method
code changed; a family that missed its floors is listed as descriptive and
does not fail the check, and the causal report then labels subgroups
`descriptive`.
The framing methods (`framing_certify.py`) are checked by `--check framing`
against `framing_certificates.json` and `framing_measurements.json`: stale
code, a `MIN_CLUSTERS` that differs from the measured minimum, or an accepted
malformed charter fails; a method that missed its floor is listed as
descriptive.
The causal gates (`f10_certify.py`) are checked by `--check f10` against
`f10_certificates.json`: a certificate is stale when the code it measured
changed; a gate that missed its floor is listed as descriptive and does not
fail the check, and the package counts it as uncertified.
The decision-layer gates (`f11_certify.py`) are checked by `--check f11`
against `f11_certificates.json`: a certificate is stale when the code it
measured changed; a gate that missed its floor, or could not run for lack of
the Hillstrom file, is listed as descriptive and does not fail the check.
The closed-loop gates (`loop_certify.py`) are checked by `--check loop`
against `loop_certificates.json`: a certificate is stale when the code or the
simulator it measured changed; a gate that missed its floor (today the
detection delay of the alarm) is listed as descriptive and does not
fail the check, and the loop then does not let an alarm retrain by itself.
"""

import argparse
import json
from pathlib import Path
from typing import Any

from benchmarks.qualification import f5
from benchmarks.qualification import f5_corpora
from benchmarks.qualification import f6_certify
from benchmarks.qualification import f10_certify
from benchmarks.qualification import f11_certify
from benchmarks.qualification import framing_certify
from benchmarks.qualification import ingest_certify
from benchmarks.qualification import ledger_checks
from benchmarks.qualification import loop_certify
from benchmarks.qualification import screen_certify
from benchmarks.qualification import smoke
from benchmarks.qualification import standins
from benchmarks.qualification import subgroup_certify
from benchmarks.qualification import uncertainty_certify
import scmle.data.admissibility.screen
from scmle.data.detectors import certificates

# Detector id -> (current code hash, defect type it gates on).
GATING: dict[str, tuple[str, str]] = {}


def check_certificates() -> int:
    """Validates the sample store and every registered gating detector.

    Prints a JSON report of the problems, missing, stale and descriptive-only
    entries of each certificate family it covers.

    Returns:
        0 when no family has a problem, otherwise 1.
    """
    store = certificates.Store.load(standins.SAMPLE_PATH)
    problems = standins.sample_problems(store)
    report = certificates.audit(store, GATING)
    f5_store = certificates.Store.load(f5.CERT_PATH)
    f5_problems = f5.problems(f5_store) + f5.audit_problems(f5_store)
    f6_store = certificates.Store.load(f6_certify.STORE_PATH)
    f6_problems = f6_certify.problems(f6_store)
    screen_store = certificates.Store.load(
        scmle.data.admissibility.screen.STORE_PATH
    )
    screen_problems = screen_certify.problems(screen_store)
    ingest_store = certificates.Store.load(ingest_certify.STORE_PATH)
    ingest_problems = ingest_certify.problems(ingest_store)
    uncertainty_problems = uncertainty_certify.problems()
    subgroup_problems = subgroup_certify.problems()
    corpora_stale = f5_corpora.stale()
    ok = (
        not corpora_stale
        and not problems
        and report.ok
        and not f5_problems
        and not f6_problems
        and not screen_problems
        and not uncertainty_problems
        and not subgroup_problems
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
                "f5_corpora_stale": corpora_stale,
                "uncertainty_problems": uncertainty_problems,
                "uncertainty_descriptive_only": (
                    uncertainty_certify.descriptive_only()
                ),
                "subgroup_problems": subgroup_problems,
                "subgroup_descriptive_only": (
                    subgroup_certify.descriptive_only()
                    if not subgroup_problems
                    else []
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


def check_all() -> int:
    """Runs every certificate family and fails if any of them fails.

    The families are the default certificate check and the causal,
    decision-layer, closed-loop, framing and ingestion checks; each runs even
    when an earlier one failed, and the combined result is printed last.

    Returns:
        0 when every family passes, otherwise 1.
    """
    results: dict[str, bool] = {}
    results["certificates"] = check_certificates() == 0
    issues = f10_certify.problems()
    print(
        json.dumps(
            {
                "ok": not issues,
                "problems": issues,
                "descriptive_only": f10_certify.descriptive_only(),
            },
            indent=2,
        )
    )
    results["f10"] = not issues
    for name, check in (
        ("f11", f11_certify.check),
        ("loop", loop_certify.check),
        ("framing", framing_certify.check),
        ("ingest", ingest_certify.check),
    ):
        found = check()
        print(json.dumps(found, indent=2))
        results[name] = bool(found["ok"])
    ok = all(results.values())
    print(json.dumps({"all": results, "ok": ok}, indent=2))
    return 0 if ok else 1


CHECK_MODULES = {
    "f11": f11_certify,
    "loop": loop_certify,
    "framing": framing_certify,
    "ingest": ingest_certify,
}


def parse_options(arguments: list[str] | None) -> argparse.Namespace:
    """Parses the command line.

    Args:
        arguments: Command-line arguments; `None` reads `sys.argv`.

    Returns:
        The namespace with the requested action and its settings.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--smoke", action="store_true")
    action.add_argument(
        "--check",
        nargs="?",
        const="certificates",
        choices=(
            "certificates",
            "f10",
            "f11",
            "fdr",
            "framing",
            "ingest",
            "loop",
            "all",
        ),
    )
    action.add_argument("--write-sample", type=Path, metavar="FILE")
    action.add_argument("--write-f5", type=Path, metavar="FILE")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--runs", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=400)
    return parser.parse_args(arguments)


def print_verdict(found: dict[str, Any]) -> int:
    """Prints a check's findings and returns its exit code.

    Args:
        found: The findings, with an `ok` entry.

    Returns:
        0 when `ok` is true, otherwise 1.
    """
    print(json.dumps(found, indent=2))
    return 0 if found["ok"] else 1


def run_check(options: argparse.Namespace) -> int:
    """Runs the check named by `--check`.

    Args:
        options: The parsed command line with `check` set.

    Returns:
        0 when the check held, otherwise 1.
    """
    if options.check == "all":
        return check_all()
    if options.check == "fdr":
        findings = ledger_checks.run_all(
            options.runs, options.seed, options.repeats
        )
        return print_verdict(ledger_checks.report(findings))
    if options.check == "f10":
        issues = f10_certify.problems()
        return print_verdict(
            {
                "ok": not issues,
                "problems": issues,
                "descriptive_only": f10_certify.descriptive_only(),
            }
        )
    if options.check in CHECK_MODULES:
        return print_verdict(CHECK_MODULES[options.check].check())
    return check_certificates()


def main(arguments: list[str] | None = None) -> int:
    """Runs the requested action and returns a process exit code.

    Args:
        arguments: Command-line arguments; `None` reads `sys.argv`.

    Returns:
        0 when the requested check or write succeeded, otherwise 1.
    """
    options = parse_options(arguments)
    if options.smoke:
        report = smoke.run_smoke(options.runs, options.seed, options.full)
        print(json.dumps(report.to_dict(), indent=2))
        return 0 if report.ok else 1
    if options.check:
        return run_check(options)
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
