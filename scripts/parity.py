"""Prints the status of every certified gate, as the running copy sees it.

Run it once from the source checkout and once with the interpreter of a
clean environment that installed the built wheel; the two outputs must be
identical (ERR-1, task 11.9). A gate that is `gating` in the checkout and
`descriptive` when installed means a certificate store did not ship.

    uv run python scripts/parity.py > checkout.json
    /tmp/wheel-env/bin/python -I scripts/parity.py > installed.json
    diff checkout.json installed.json

Run with `-I` from the installed environment, so `scmle` is imported from
the wheel and not from the checkout.
"""

import hashlib
import json
import sys

from scmle.assurance import coverage_slices
from scmle.assurance.admissibility import qualify
from scmle.assurance.admissibility import screen
import scmle.certificates
from scmle.data.identification import certified
from scmle.evaluation import task_gate


def survey() -> dict[str, object]:
    """Collects the shipped stores and the status of each gate.

    Returns:
        A JSON-ready mapping: the digest of every shipped store, and for
        each gate family its freshness or the gates it certifies.
    """
    stores = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(scmle.certificates.DIRECTORY.glob("*.json"))
    }
    detectors = qualify.load_store(qualify.F5_STORE)
    robustness = qualify.load_store("f6_certificates.json")
    leakage = screen.load_store()
    return {
        "stores": stores,
        "detectors": detectors.note or "loaded",
        "robustness": robustness.note or "loaded",
        "screen": (
            "missing"
            if leakage is None
            else str(leakage.state(screen.DETECTOR_ID, screen.code_digest()))
        ),
        "identification": sorted(certified.certificated()),
        "slices": str(coverage_slices.flag_gate().status),
        "tasks": task_gate.load(),
    }


def main() -> int:
    """Prints the survey; fails when a store is missing.

    Returns:
        0 when every gate family found its store, 1 otherwise.
    """
    found = survey()
    print(json.dumps(found, indent=2, sort_keys=True))
    absent = [
        name for name in ("detectors", "robustness") if found[name] != "loaded"
    ]
    absent += ["screen"] if found["screen"] == "missing" else []
    return 1 if absent else 0


if __name__ == "__main__":
    sys.exit(main())
