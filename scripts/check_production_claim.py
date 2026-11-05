#!/usr/bin/env python3
"""Refuse the production claim unless both providers' evidence is complete.

    python scripts/check_production_claim.py --root plan/evidence/phase-17
        --key-file pipeline.key --reviewer-key-file reviewer.key

Exit status 0 only if ``<root>/aws/`` and ``<root>/gcp/`` hold every signed
record, from staging, for one commit, passing, and covered by a reviewer's
sign-off that is not the author's (see ``scmle.operations.claim``). The
verdict lists every reason it refuses. ``--releasing`` also checks that the
fixed claim text in ``docs/RELEASING.md`` is the one the checker guards.
"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scmle.operations import claim  # noqa: E402


def read_key(path: str) -> str:
    """Reads a key file; an empty or missing file is an error."""
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"{path} is empty")
    return text


def main(arguments: list[str] | None = None) -> int:
    """Prints the verdict; returns 0 only if the claim may be made."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", default="plan/evidence/phase-17")
    parser.add_argument("--key-file", required=True)
    parser.add_argument("--reviewer-key-file", required=True)
    parser.add_argument("--releasing", default="docs/RELEASING.md")
    options = parser.parse_args(arguments)
    try:
        verdict = claim.check_claim(
            options.root,
            read_key(options.key_file),
            read_key(options.reviewer_key_file),
        )
    except (OSError, ValueError) as error:
        print(json.dumps({"allowed": False, "problems": [str(error)]}))
        return 2
    problems = list(verdict.problems)
    problems.extend(claim.claim_text_problems(options.releasing))
    allowed = verdict.allowed and not problems
    print(
        json.dumps(
            {**verdict.to_dict(), "allowed": allowed, "problems": problems},
            indent=2,
        )
    )
    return 0 if allowed else 1


if __name__ == "__main__":
    sys.exit(main())
