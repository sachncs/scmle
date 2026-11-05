#!/usr/bin/env python3
"""Check the Terraform under deploy/terraform against the written policy.

Reads the files with `scmle.deploy.hcl`; needs no Terraform and no credentials.
Exit 0 when no rule is violated, 1 otherwise, 2 for an unreadable file. A pass
is not `terraform validate` and not a plan (see scripts/terraform_checks.sh).
Implementation: `scmle.deploy.iac_policy`.
"""

import argparse
import json
from pathlib import Path
import sys

from scmle.deploy import hcl
from scmle.deploy import iac_policy


def main(arguments: list[str] | None = None) -> int:
    """Run the policy; print the violations as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root", type=Path, nargs="?", default=Path("deploy/terraform")
    )
    options = parser.parse_args(arguments)
    try:
        found = iac_policy.check_tree(options.root)
    except (OSError, hcl.HclError) as error:
        print(f"policy check failed: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {"passed": not found, "violations": [v.to_dict() for v in found]},
            indent=2,
            sort_keys=True,
        )
    )
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
