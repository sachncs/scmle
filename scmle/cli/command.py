"""The sub-command interface, the exit codes and the usage-error parser."""

import abc
import argparse
import sys
from typing import ClassVar
from typing import NoReturn

# Exit codes (ERR-21): each outcome has its own, so a script can tell a
# refused run from a failed one, a wrong command line and a defect. Usage and
# internal errors take the conventional sysexits values.
EXIT_OK = 0
EXIT_FAILED = 1
EXIT_BLOCKED = 2
EXIT_USAGE = 64
EXIT_INTERNAL = 70


class Parser(argparse.ArgumentParser):
    """The argument parser, exiting with `EXIT_USAGE` on a bad command line.

    argparse exits 2 by default, which is the code for a blocked run here.
    """

    def error(self, message: str) -> NoReturn:
        """Reports a usage error and exits with `EXIT_USAGE`.

        Args:
            message: What is wrong with the command line.
        """
        self.print_usage(sys.stderr)
        self.exit(EXIT_USAGE, f"{self.prog}: error: {message}\n")


class Command(abc.ABC):
    """A CLI sub-command.

    Attributes:
        name: Sub-command name.
        help: One-line description.
    """

    name: ClassVar[str]
    help: ClassVar[str]

    @abc.abstractmethod
    def configure(self, parser: argparse.ArgumentParser) -> None:
        """Adds this command's arguments to `parser`.

        Args:
            parser: The argument parser of this command.
        """

    @abc.abstractmethod
    def execute(self, args: argparse.Namespace) -> int:
        """Runs the command.

        Args:
            args: Parsed arguments.

        Returns:
            Process exit code.
        """


def validators(parser: argparse.ArgumentParser) -> None:
    """Adds the validation strategy, fold, test-split and window options.

    Args:
        parser: The argument parser of the command.
    """
    parser.add_argument(
        "--validation",
        choices=[
            "auto",
            "stratified",
            "stratified_events",
            "multilabel_stratified",
            "treatment_stratified",
            "random",
            "group",
            "temporal",
            "rolling",
        ],
        default="auto",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument(
        "--horizon-rows",
        type=int,
        help="rolling validation: rows in each validation window",
    )
    parser.add_argument(
        "--window-rows",
        type=int,
        help="rolling validation: rows in each training window",
    )
