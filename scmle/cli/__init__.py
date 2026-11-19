"""Command-line interface.

Each sub-command is a `Command` subclass that registers its own arguments and
executes itself, so adding a command never touches the dispatcher. The
commands live in modules grouped by what they work on; this module holds the
registry and the dispatcher, and re-exports every command.
"""

import argparse
from collections.abc import Sequence
import logging
import os

from scmle.cli.clustering import Clustering
from scmle.cli.command import Command
from scmle.cli.command import EXIT_BLOCKED
from scmle.cli.command import EXIT_FAILED
from scmle.cli.command import EXIT_INTERNAL
from scmle.cli.command import EXIT_OK
from scmle.cli.command import EXIT_USAGE
from scmle.cli.command import Parser
from scmle.cli.command import validators
from scmle.cli.delivery import Batch
from scmle.cli.delivery import Export
from scmle.cli.delivery import Packaging
from scmle.cli.delivery import Prediction
from scmle.cli.delivery import Serve
from scmle.cli.erasure import Deletion
from scmle.cli.erasure import Forgetting
from scmle.cli.erasure import Retirement
from scmle.cli.governance import Claims
from scmle.cli.governance import Loop
from scmle.cli.governance import Trail
from scmle.cli.intake import Cataloging
from scmle.cli.intake import Charter
from scmle.cli.intake import Frame
from scmle.cli.intake import IngestCommand
from scmle.cli.maintenance import Records
from scmle.cli.rollout import Deployment
from scmle.cli.rollout import Operation
from scmle.cli.training import Estimation
from scmle.cli.training import Plan
from scmle.cli.training import ProjectCommand
from scmle.cli.training import Resumption
from scmle.cli.training import Run
from scmle.contracts import errors
from scmle.runtime import logs
from scmle.runtime import tracing

__all__ = [
    "COMMANDS",
    "EXIT_BLOCKED",
    "EXIT_FAILED",
    "EXIT_INTERNAL",
    "EXIT_OK",
    "EXIT_USAGE",
    "LOGGER",
    "Batch",
    "Cataloging",
    "Charter",
    "Claims",
    "Clustering",
    "Command",
    "Deletion",
    "Deployment",
    "Estimation",
    "Export",
    "Forgetting",
    "Frame",
    "IngestCommand",
    "Loop",
    "Operation",
    "Packaging",
    "Parser",
    "Plan",
    "Prediction",
    "ProjectCommand",
    "Records",
    "Resumption",
    "Retirement",
    "Run",
    "Serve",
    "Trail",
    "build_parser",
    "main",
    "report",
    "validators",
]

LOGGER = logging.getLogger(__name__)


COMMANDS: tuple[Command, ...] = (
    Plan(),
    Run(),
    Resumption(),
    Estimation(),
    Cataloging(),
    Prediction(),
    Batch(),
    Serve(),
    Export(),
    Packaging(),
    IngestCommand(),
    Charter(),
    Claims(),
    Frame(),
    Loop(),
    Operation(),
    Deployment(),
    Trail(),
    Records(),
    Clustering(),
    Forgetting(),
    Retirement(),
    Deletion(),
)


def build_parser() -> argparse.ArgumentParser:
    """Builds the argument parser from the registered commands.

    Returns:
        The parser, with a sub-command for each registered command.
    """
    parser = Parser(prog="scmle")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="verbosity of the log written to standard error",
    )
    parser.add_argument(
        "--log-json",
        action="store_true",
        help="write the log as one JSON object per line, with run and "
        "request ids",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="export OpenTelemetry spans of requests, scoring and pipeline "
        "stages over OTLP (needs the tracing extra; also SCMLE_TRACING)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="on failure, show the traceback, the cause chain and details",
    )
    subparsers = parser.add_subparsers(
        dest="command", required=True, parser_class=Parser
    )
    for command in COMMANDS:
        command.configure(
            subparsers.add_parser(command.name, help=command.help)
        )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parses the arguments, runs the chosen command and returns its status.

    Args:
        argv: Arguments; defaults to `sys.argv[1:]`.

    Returns:
        The command's own status on success; `EXIT_BLOCKED` if it was refused
        (a `BlockedError` or `LeakageError`); `EXIT_FAILED` for any other
        `ScmleError`; `EXIT_INTERNAL` for any other exception, which is a
        defect. A bad command line exits with `EXIT_USAGE`. With `--debug`
        every failure is logged with its traceback, causes and details.
    """
    args = build_parser().parse_args(argv)
    level = logging.DEBUG if args.debug else args.log_level
    logs.configure(level, structured=args.log_json)
    if args.trace or os.environ.get(tracing.SWITCH):
        tracing.TRACING.enable()
    command = next(c for c in COMMANDS if c.name == args.command)
    try:
        return command.execute(args)
    except (errors.BlockedError, errors.LeakageError) as error:
        # handled: logged by `report`, exit code EXIT_BLOCKED
        report("blocked", error, debug=args.debug)
        return EXIT_BLOCKED
    except errors.ScmleError as error:
        # handled: logged by `report`, exit code EXIT_FAILED. This also
        # catches `InputError` (and `KindError`, `OrderError`), which were
        # bare ValueError/TypeError/RuntimeError raises exiting 70 before
        # ERR-20. Kept on purpose: docs/CLI.md gives bad data exit 1, and
        # exit 70 means a defect; a bare ValueError still exits 70.
        report(f"{error.category} error", error, debug=args.debug)
        return EXIT_FAILED
    except Exception as error:  # noqa: BLE001 -- the process boundary: any other failure is a defect, reported with its traceback
        # handled: logged with its traceback by `report`, EXIT_INTERNAL
        report("internal error (a defect; please report it)", error, debug=True)
        return EXIT_INTERNAL
    finally:
        tracing.TRACING.disable()


def report(label: str, error: BaseException, *, debug: bool) -> None:
    """Logs a failure: one line, or the whole story with `--debug`.

    Args:
        label: What kind of failure it is.
        error: The failure.
        debug: Whether to add the traceback, the causes and the details.
    """
    if not debug:
        LOGGER.error("%s: %s", label, error)
        return
    failure = errors.Failure.of(error)
    details = getattr(error, "details", None) or {}
    LOGGER.error(
        "%s: %s\ncauses: %s\ndetails: %s\n%s",
        label,
        error,
        list(failure.causes) or "none",
        details or "none",
        failure.trace,
    )
