"""Runs the command line as `python -m scmle.cli`."""

import sys

from scmle import cli

if __name__ == "__main__":
    sys.exit(cli.main())
