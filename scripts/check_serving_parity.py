#!/usr/bin/env python3
"""Check a running service against the golden records.

Sends `deploy/local/golden/records.json` to `<url>/predict` and compares the
answer with the closed-form expected outputs under the declared tolerance.
Exit 0 only when every label matches and every probability is within
tolerance, and the response carries the version and artifact digest.
"""

import argparse
import json
from pathlib import Path
import sys

from scmle.inference import parity

DEFAULT_DIRECTORY = Path("deploy/local/golden")


def run(url: str, directory: Path, level: str, timeout: float) -> dict:
    """Return the parity report for the service at `url`.

    Args:
        url: Base URL of the running service.
        directory: Folder with the golden records and expected outputs.
        level: `same_process` or `cross_environment` tolerance.
        timeout: Seconds to wait for the answer.

    Returns:
        The report from `parity.check_remote`; its `passed` key says whether
        every label matched and every probability was within tolerance.
    """
    return parity.check_remote(url, directory, level, timeout)


def main(arguments: list[str] | None = None) -> int:
    """Run the check; print the report as JSON.

    Args:
        arguments: Command-line arguments; `None` reads `sys.argv`.

    Returns:
        0 when the report passed, otherwise 1.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--directory", type=Path, default=DEFAULT_DIRECTORY)
    parser.add_argument(
        "--level",
        choices=["same_process", "cross_environment"],
        default="cross_environment",
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    options = parser.parse_args(arguments)
    report = run(options.url, options.directory, options.level, options.timeout)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
