"""The sub-command interface, the exit codes and the usage-error parser."""

import abc
import argparse
from collections.abc import Sequence
import sys
from typing import Any
from typing import ClassVar
from typing import NoReturn
import warnings

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


class Alias(argparse.Action):
    """An option's old spelling: hidden, it warns once and stores as the new.

    Attributes:
        spelling: The option's current spelling.
        warned: Whether the deprecation warning was given.
    """

    def __init__(
        self, option_strings: Sequence[str], dest: str, **kwargs: Any
    ) -> None:
        """Builds the alias.

        Args:
            option_strings: The old spelling.
            dest: The new option's destination.
            **kwargs: `spelling` (the new option) and the new option's
                `nargs`, `const`, `type`, `choices` and `metavar`.
        """
        self.spelling = str(kwargs.pop("spelling"))
        self.warned = False
        super().__init__(option_strings, dest, **kwargs)

    def __call__(
        self,
        parser: argparse.ArgumentParser,  # noqa: ARG002 -- Action interface
        namespace: argparse.Namespace,
        values: Any,
        option_string: str | None = None,
    ) -> None:
        """Warns on first use, then stores the value as the new option does.

        Args:
            parser: The parser.
            namespace: The parsed arguments so far.
            values: The converted value, or None for a flag.
            option_string: The spelling used.
        """
        if not self.warned:
            self.warned = True
            warnings.warn(
                f"{option_string} is deprecated and will be removed in the "
                f"next release; use {self.spelling}",
                FutureWarning,
                stacklevel=2,
            )
        setattr(namespace, self.dest, self.const if self.nargs == 0 else values)


def option(
    parser: argparse.ArgumentParser, flag: str, old: str, **kwargs: Any
) -> argparse.Action:
    """Adds an option and its old spelling as a hidden, deprecated alias.

    Args:
        parser: The argument parser of the command.
        flag: The option's spelling.
        old: The spelling it replaces, accepted for one release.
        **kwargs: The option's `add_argument` keywords.

    Returns:
        The option's action.
    """
    action = parser.add_argument(flag, **kwargs)
    flagged = action.nargs == 0
    parser.add_argument(
        old,
        dest=action.dest,
        action=Alias,
        spelling=flag,
        nargs=action.nargs,
        const=action.const if flagged else None,
        type=action.type or str,
        choices=action.choices,
        metavar=action.metavar,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    return action


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
    option(
        parser,
        "--test",
        "--test-fraction",
        type=float,
        default=0.2,
        help="share of rows locked away for the final test",
    )
    option(
        parser,
        "--rolling-horizon",
        "--horizon-rows",
        type=int,
        help="rolling validation: rows in each validation window",
    )
    option(
        parser,
        "--rolling-window",
        "--window-rows",
        type=int,
        help="rolling validation: rows in each training window",
    )
