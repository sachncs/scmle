"""Command line: check a bundle's trace, or rebuild it on a fresh environment.

    python -m scmle.trail trace BUNDLE [--run RUN] [--repository REPO]
        [--data FILE]
    python -m scmle.trail rebuild BUNDLE [--run RUN] [--repository REPO]
        [--data FILE] [--workspace DIR] [--tolerance VALUE]

Each prints its JSON record and exits 1 when the bundle does not trace or
is not reproduced, and 2 on an input or artifact error. The signing key is
read from ``$SCMLE_SIGNING_KEY``.
"""

import argparse
import json
import sys

from scmle.contracts import errors
from scmle.trail import origin
from scmle.trail import replica


def parser() -> argparse.ArgumentParser:
    """The command line's parser.

    Returns:
        A parser with the `trace` and `rebuild` actions.
    """
    found = argparse.ArgumentParser(prog="python -m scmle.trail")
    actions = found.add_subparsers(dest="action", required=True)
    for name, text in (
        (
            "trace",
            (
                "check that a bundle traces to data, configuration, commit "
                "and environment"
            ),
        ),
        (
            "rebuild",
            "rebuild the environment from the manifest, retrain and compare",
        ),
    ):
        action = actions.add_parser(name, help=text)
        action.add_argument("bundle", help="bundle directory")
        action.add_argument("--run", help="run directory that built it")
        action.add_argument("--repository", help="git repository of the commit")
        action.add_argument("--data", help="data file, if the source moved")
    rebuild = actions.choices["rebuild"]
    rebuild.add_argument("--workspace", help="keep the rebuild in this folder")
    rebuild.add_argument("--tolerance", type=float, default=replica.TOLERANCE)
    return found


def main(arguments: list[str] | None = None) -> int:
    """Runs one action.

    Args:
        arguments: The command-line arguments; `sys.argv` when None.

    Returns:
        0 when the bundle traces or is reproduced, 1 when not, 2 on error.
    """
    args = parser().parse_args(arguments)
    common = {"run": args.run, "repository": args.repository, "data": args.data}
    try:
        if args.action == "trace":
            report = origin.trace(args.bundle, **common)
            record, passed = report.dump(), report.traced
        else:
            rebuilt = replica.rebuild(
                args.bundle,
                workspace=args.workspace,
                tolerance=args.tolerance,
                **common,
            )
            record, passed = rebuilt.dump(), rebuilt.reproduced
    except (errors.InputError, errors.ArtifactError) as error:
        sys.stderr.write(f"error: {error}\n")
        return 2
    sys.stdout.write(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
