"""The `forget`, `retire` and `delete` commands.

    scmle forget dataset runs registry --data old.csv --reason "ticket 118"
    scmle forget subject runs --data customers.csv --column customer_id \
        --values-file ids.txt --retrain --reason "erasure request 2026-118"
    scmle forget verify receipts/forget-0123456789abcdef.json
    scmle retire registry --model churn --version 3 --reason "drift"
    scmle delete runs/run_20261118_001 --reason "retention expired"

Each command prints the plan with `--dry-run`, and otherwise carries it out
and writes a signed receipt under `--receipts`. Running the same command
again after an interruption resumes it. A key is read from `--key-file` or
`SCMLE_SIGNING_KEY`; without one nothing is changed.
"""

import argparse
import logging
import pathlib

from scmle.api import erasure
from scmle.cli import command
from scmle.contracts import errors

LOGGER = logging.getLogger(__name__)


def common(parser: argparse.ArgumentParser) -> None:
    """Adds the reason, receipts, key and dry-run options.

    Args:
        parser: The argument parser of an action.
    """
    parser.add_argument(
        "--reason", required=True, help="why: a request or ticket reference"
    )
    parser.add_argument(
        "--receipts",
        default="receipts",
        help="folder for the plan, journal and signed receipt",
    )
    parser.add_argument(
        "--key-file", help="file holding the signing key; default: env"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the plan and change nothing",
    )


def key(path: str | None) -> str | None:
    """Reads the signing key file, if one is named.

    Args:
        path: The file, or None.

    Returns:
        The stripped key, or None to use the environment.

    Raises:
        ConfigError: If the file cannot be read.
    """
    if path is None:
        return None
    try:
        return pathlib.Path(path).read_text(encoding="utf-8").strip()
    except OSError as error:
        raise errors.ConfigError(
            f"cannot read the key file: {error}"
        ) from error


def conclude(job: erasure.Erasure, args: argparse.Namespace) -> int:
    """Prints the plan, or carries it out and prints the receipt.

    Args:
        job: The erasure.
        args: The parsed arguments (`dry_run`, `key_file`).

    Returns:
        EXIT_OK (0).
    """
    if args.dry_run:
        planned = job.preview()
        for step in planned:
            print(f"would {step.action} {step.target}")
        print(f"{len(planned)} step(s) planned for {job.identifier}")
        return command.EXIT_OK
    receipt = job.execute(key(args.key_file))
    for action, count in sorted(receipt.tally().items()):
        print(f"{action}: {count}")
    print(f"receipt: {job.receipts / (receipt.identifier + '.json')}")
    return command.EXIT_OK


class Forgetting(command.Command):
    """Forgets a dataset or a data subject, or verifies a receipt."""

    name = "forget"
    help = "forget a dataset or a data subject everywhere it was used"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        """Adds the `dataset`, `subject` and `verify` actions.

        Args:
            parser: The argument parser of this command.
        """
        actions = parser.add_subparsers(dest="forget_action", required=True)
        dataset = actions.add_parser(
            "dataset", help="forget every artefact derived from a dataset"
        )
        dataset.add_argument("roots", nargs="+", help="folders to search")
        named = dataset.add_mutually_exclusive_group(required=True)
        named.add_argument("--data", help="the data file the runs read")
        named.add_argument(
            "--digest", action="append", help="a dataset digest (repeatable)"
        )
        common(dataset)
        subject = actions.add_parser(
            "subject", help="purge a subject's rows and forget their traces"
        )
        subject.add_argument("roots", nargs="+", help="folders to search")
        subject.add_argument("--data", required=True, help="the source file")
        subject.add_argument("--column", required=True)
        subject.add_argument("--value", action="append", default=[])
        subject.add_argument(
            "--values-file", help="one identifier per line, kept off the shell"
        )
        subject.add_argument(
            "--earlier",
            action="append",
            default=[],
            help="digest of an earlier version of the source (repeatable)",
        )
        subject.add_argument(
            "--retrain",
            action="store_true",
            help="train each affected run again without the rows",
        )
        common(subject)
        verify = actions.add_parser(
            "verify", help="verify the signature of any erasure receipt"
        )
        verify.add_argument("receipt")
        verify.add_argument("--key-file")

    def execute(self, args: argparse.Namespace) -> int:
        """Runs the chosen action.

        Args:
            args: The parsed command-line arguments.

        Returns:
            EXIT_OK (0).

        Raises:
            ArtifactError: If a receipt does not verify.
            InputError: If a subject is named by no identifier or the values
                file cannot be read.
        """
        if args.forget_action == "verify":
            receipt = erasure.Receipt.read(args.receipt)
            if not receipt.verify(key(args.key_file)):
                raise errors.ArtifactError(
                    f"{args.receipt}: the signature does not verify"
                )
            print(f"{receipt.identifier}: verified ({receipt.kind})")
            return command.EXIT_OK
        if args.forget_action == "dataset":
            subject = (
                erasure.Dataset.of(args.data)
                if args.data
                else erasure.Dataset(args.digest)
            )
            job = erasure.Forgetting(
                args.roots, subject, args.receipts, args.reason
            )
            return conclude(job, args)
        values = list(args.value)
        if args.values_file:
            try:
                text = pathlib.Path(args.values_file).read_text("utf-8")
            except OSError as error:
                raise errors.InputError(
                    f"cannot read the values file: {error}"
                ) from error
            values += [line.strip() for line in text.splitlines()]
        values = [v for v in values if v]
        if not values:
            raise errors.InputError("give --value or --values-file")
        job = erasure.Forgetting(
            args.roots,
            erasure.Subject(args.data, args.column, values, args.earlier),
            args.receipts,
            args.reason,
            retrain=args.retrain,
        )
        return conclude(job, args)


class Retirement(command.Command):
    """Retires a model so that nothing serves it again."""

    name = "retire"
    help = "retire a model bundle or a registered version"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        """Adds the target and the registry options.

        Args:
            parser: The argument parser of this command.
        """
        parser.add_argument("target", help="bundle directory or registry root")
        parser.add_argument("--model", default="", help="registered name")
        parser.add_argument(
            "--version", type=int, default=0, help="registered version"
        )
        common(parser)

    def execute(self, args: argparse.Namespace) -> int:
        """Retires the model.

        Args:
            args: The parsed command-line arguments.

        Returns:
            EXIT_OK (0).
        """
        job = erasure.Retirement(
            args.target, args.receipts, args.reason, args.model, args.version
        )
        return conclude(job, args)


class Deletion(command.Command):
    """Deletes runs or project folders, keeping the custody records."""

    name = "delete"
    help = "delete runs or a project folder, keeping custody records"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        """Adds the targets.

        Args:
            parser: The argument parser of this command.
        """
        parser.add_argument(
            "targets", nargs="+", help="run directories or project folders"
        )
        common(parser)

    def execute(self, args: argparse.Namespace) -> int:
        """Deletes the targets.

        Args:
            args: The parsed command-line arguments.

        Returns:
            EXIT_OK (0).
        """
        job = erasure.Deletion(args.targets, args.receipts, args.reason)
        return conclude(job, args)
