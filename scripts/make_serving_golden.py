#!/usr/bin/env python3
"""Write or check the golden expected outputs from the model spec and records.

The expected values are computed in closed form with numpy (no scikit-learn),
so they are an independent reference for every serving path.
"""

import argparse
import json
from pathlib import Path
import sys

from scmle.inference import parity

DEFAULT_DIRECTORY = Path("deploy/local/golden")


def build(directory: Path) -> str:
    """Return the expected-output document for the files in `directory`."""
    golden = parity.load_golden(directory)
    document = {
        "comment": "Closed-form outputs of model_spec.json on records.json. "
        "Regenerate with scripts/make_serving_golden.py; never edit by hand.",
        **parity.expected_outputs(golden.spec, golden.records),
    }
    return json.dumps(document, indent=2) + "\n"


def main(arguments: list[str] | None = None) -> int:
    """Write the file, or with --check fail when it is stale."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=DEFAULT_DIRECTORY)
    parser.add_argument("--check", action="store_true")
    options = parser.parse_args(arguments)
    target = options.directory / "expected.json"
    text = build(options.directory)
    if options.check:
        if not target.exists() or target.read_text(encoding="utf-8") != text:
            print(f"{target} is stale: run scripts/make_serving_golden.py")
            return 1
        print(f"{target} is current")
        return 0
    target.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
