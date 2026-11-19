"""Commands that take data in: catalog, ingest, charter and frame."""

import argparse
import json
import logging
import pathlib
import sys
from typing import cast

import pandas

from scmle.assurance.charter import contracts
from scmle.assurance.charter import feasibility
from scmle.assurance.charter import sealing
from scmle.assurance.framing import costs
from scmle.assurance.framing import draft
from scmle.assurance.framing import refusal
from scmle.assurance.ingest import budget
from scmle.assurance.ingest import grain
from scmle.assurance.ingest import joins
import scmle.assurance.ingest.plan
from scmle.cli import command
from scmle.contracts import errors
from scmle.contracts import signing
from scmle.data.catalog import Catalog
from scmle.data.readers import tabular

LOGGER = logging.getLogger(__name__)


class Cataloging(command.Command):
    """Produces model-agnostic dataset curation evidence."""

    name = "catalog"
    help = "catalog a local table or opaque raw bytes"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        """Adds input, label, source and artifact options.

        Args:
            parser: The argument parser of this command.
        """
        parser.add_argument(
            "input",
            nargs="?",
            help=(
                "supported local tabular path, or '-' to read raw bytes "
                "from standard input"
            ),
        )
        parser.add_argument(
            "--raw-bytes",
            nargs="?",
            const="-",
            metavar="PATH",
            help=(
                "read opaque raw bytes from PATH; omit PATH or use '-' "
                "for standard input"
            ),
        )
        parser.add_argument(
            "--label-column",
            help="explicit label field to describe; labels are never inferred",
        )
        parser.add_argument(
            "--source-uri",
            help="declared source URI; defaults to the local input identity",
        )
        parser.add_argument(
            "--output",
            help="optional path for the atomically written JSON artifact",
        )

    def read(self, location: str) -> bytes:
        """Reads raw bytes from a local path or standard input.

        Args:
            location: A file path, or ``-`` for standard input.

        Returns:
            The bytes read.

        Raises:
            DataError: If the file cannot be read or standard input is not a
                binary stream.
        """
        if location == "-":
            stream = getattr(sys.stdin, "buffer", sys.stdin)
            payload = stream.read()
        else:
            try:
                payload = pathlib.Path(location).read_bytes()
            except OSError as error:
                raise errors.DataError(
                    f"failed to read raw bytes from {location}: {error}"
                ) from error
        if not isinstance(payload, bytes):
            raise errors.DataError("raw byte input must be a binary stream")
        return payload

    def execute(self, args: argparse.Namespace) -> int:
        """Catalogs a tabular file or raw bytes and prints the JSON evidence.

        The input is raw bytes when `--raw-bytes` is given or the positional
        input is `-`; otherwise it is loaded as a table. The evidence is also
        written to `--output` when that is given.

        Args:
            args: The parsed command-line arguments.

        Returns:
            The exit status, EXIT_OK (0).

        Raises:
            ArtifactError: If the artifact cannot be written to `--output`.
            ConfigError: If both or neither of a tabular input and
                `--raw-bytes` are given, or the catalog options are invalid.
        """
        if args.input is not None and args.raw_bytes is not None:
            raise errors.ConfigError(
                "provide either a tabular input or --raw-bytes, not both"
            )
        if args.input is None and args.raw_bytes is None:
            raise errors.ConfigError(
                "catalog requires a tabular input or --raw-bytes"
            )

        raw_location: str | None = args.raw_bytes
        if args.input == "-":
            raw_location = "-"
        if raw_location is not None:
            catalog_input: bytes | pandas.DataFrame = self.read(raw_location)
            source = args.source_uri or (
                "memory://stdin" if raw_location == "-" else raw_location
            )
        else:
            loader = tabular.Loader()
            # Not None: the checks above reject neither input being given.
            tabular_input = cast("str", args.input)
            catalog_input = loader.load(tabular_input)
            source = args.source_uri or tabular_input

        try:
            result = Catalog.observe(
                catalog_input,
                source=source,
                label_column=args.label_column,
            )
        except (TypeError, ValueError) as error:
            raise errors.ConfigError(
                f"invalid catalog options: {error}"
            ) from error

        if args.output is not None:
            try:
                result.write(args.output)
            except OSError as error:
                raise errors.ArtifactError(
                    f"failed to write catalog artifact {args.output}: {error}"
                ) from error

        print(json.dumps(result.to_dict(), ensure_ascii=False, sort_keys=True))
        return command.EXIT_OK


class IngestCommand(command.Command):
    """Profiles a SQLite source: catalog, keys, unit verdict and joins."""

    name = "ingest"
    help = "profile a relational source: tables, keys, grain and joins"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        """Adds source, scope, tolerance and budget options.

        Args:
            parser: The argument parser of this command.
        """
        parser.add_argument("--sqlite", required=True, help="database file")
        parser.add_argument(
            "--tables", required=True, help="comma-separated table names"
        )
        parser.add_argument("--output", default="ingest")
        parser.add_argument("--unit", help="charter unit text to check")
        parser.add_argument("--target-table", help="table the unit describes")
        parser.add_argument("--eps-key", type=float, default=0.0)
        parser.add_argument("--eps-null", type=float, default=0.0)
        parser.add_argument("--eps-fk", type=float, default=0.0)
        parser.add_argument("--max-key-size", type=int, default=grain.LEVELS)
        parser.add_argument("--max-queries", type=int)
        parser.add_argument("--max-rows-scanned", type=int)
        parser.add_argument("--max-seconds", type=float)
        parser.add_argument(
            "--foreign-keys",
            help="JSON file: [{child, child_columns, parent, parent_columns}]",
        )

    def execute(self, args: argparse.Namespace) -> int:
        """Runs the ingest stages and prints the JSON summary.

        A blocking unit verdict is raised after the summary is printed, which
        `main` turns into exit status 2.

        Args:
            args: The parsed command-line arguments.

        Returns:
            The exit status, EXIT_OK (0).

        Raises:
            BlockedError: If the unit verdict is one that blocks the run.
        """
        declared: tuple[joins.Reference, ...] = ()
        if args.foreign_keys:
            declared = scmle.assurance.ingest.plan.declarations(
                json.loads(
                    pathlib.Path(args.foreign_keys).read_text(encoding="utf-8")
                )
            )
        settings = scmle.assurance.ingest.plan.Config(
            tables=tuple(t for t in args.tables.split(",") if t),
            unit=args.unit,
            target=args.target_table,
            violation=args.eps_key,
            null=args.eps_null,
            orphans=args.eps_fk,
            width=args.max_key_size,
            declared=declared,
        )
        plan = scmle.assurance.ingest.plan.Plan(
            pathlib.Path(args.output),
            scmle.assurance.ingest.plan.opener(args.sqlite),
            settings,
            budget.Budget(
                args.max_queries, args.max_rows_scanned, args.max_seconds
            ),
        )
        summary = plan.execute()
        print(json.dumps(summary, indent=2))
        if plan.verdict is not None and plan.verdict.blocks:
            raise errors.BlockedError(
                f"unit verdict {plan.verdict.verdict}: {plan.verdict.reason}"
            )
        return command.EXIT_OK


class Charter(command.Command):
    """Creates, checks and signs the analysis charter."""

    name = "charter"
    help = "create, check or sign an analysis charter"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        """Adds the `init`, `check` and `sign` actions.

        Args:
            parser: The argument parser of this command.
        """
        actions = parser.add_subparsers(dest="charter_action", required=True)
        init = actions.add_parser("init", help="write a template charter")
        init.add_argument("path")
        init.add_argument("--force", action="store_true")
        check = actions.add_parser("check", help="validate a charter")
        check.add_argument("path")
        check.add_argument(
            "--rows",
            type=int,
            help="development rows, to report confirmation feasibility",
        )
        sign = actions.add_parser("sign", help="sign a validated charter")
        sign.add_argument("path")
        sign.add_argument("--output", help="default: rewrite the input")
        sign.add_argument(
            "--key-file",
            help="file holding the key; default: SCMLE_SIGNING_KEY",
        )

    def execute(self, args: argparse.Namespace) -> int:
        """Runs the chosen action.

        Args:
            args: The parsed command-line arguments.

        Returns:
            The exit status of the chosen action.
        """
        action = getattr(self, f"run_{args.charter_action}")
        status: int = action(args)
        return status

    @staticmethod
    def run_init(args: argparse.Namespace) -> int:
        """Writes a template that fails `check` until it is filled in.

        Args:
            args: The parsed command-line arguments.

        Returns:
            The exit status, EXIT_OK (0).

        Raises:
            ConfigError: If the file exists and `--force` is not given.
        """
        path = pathlib.Path(args.path)
        if path.exists() and not args.force:
            raise errors.ConfigError(f"{path} exists; pass --force")
        path.write_text(
            json.dumps(contracts.template(), indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        LOGGER.info("wrote template charter %s", path)
        return command.EXIT_OK

    @staticmethod
    def run_check(args: argparse.Namespace) -> int:
        """Validates the charter, its hash and (if present) its signature.

        Args:
            args: The parsed command-line arguments.

        Returns:
            The exit status, EXIT_OK (0).
        """
        loaded = sealing.read(args.path)
        charter = loaded.charter
        LOGGER.info(
            "charter %s %s",
            loaded.digest,
            "signature verified" if loaded.signed else "unsigned or unchecked",
        )
        if args.rows is not None:
            report = feasibility.assess(
                n_confirmation=int(args.rows * charter.sets.confirmation),
                min_effect=charter.effect,
                alpha=float(charter.budget.confirmation.alpha_confirm),
                k_max=charter.quota,
                power=charter.power,
            )
            LOGGER.info("feasibility: %s (%s)", report.verdict, report.reason)
        return command.EXIT_OK

    @staticmethod
    def run_sign(args: argparse.Namespace) -> int:
        """Signs the charter with a key from a file or the environment.

        Args:
            args: The parsed command-line arguments.

        Returns:
            The exit status, EXIT_OK (0).

        Raises:
            ConfigError: If no signing key is available from the key file or
                the environment.
        """
        key = (
            pathlib.Path(args.key_file).read_text(encoding="utf-8").strip()
            if args.key_file
            else None
        )
        if signing.keying(key) is None:
            raise errors.ConfigError(
                "no signing key: set SCMLE_SIGNING_KEY or pass --key-file"
            )
        charter = contracts.peek(args.path)
        output = sealing.inscribe(
            args.output or args.path, charter, signing_key=key
        )
        LOGGER.info("signed %s", output)
        return command.EXIT_OK


class Frame(command.Command):
    """Drafts a charter, cost model and questions from a plain request."""

    name = "frame"
    help = (
        "draft an unsigned charter and the questions only a person can answer"
    )

    def configure(self, parser: argparse.ArgumentParser) -> None:
        """Adds the request, source, data and optional inputs.

        Args:
            parser: The argument parser of this command.
        """
        parser.add_argument("request", help="text file with the request")
        parser.add_argument(
            "--source",
            required=True,
            help="JSON: uri, columns, key, time",
        )
        parser.add_argument(
            "--data", required=True, help="exploration rows only (CSV)"
        )
        parser.add_argument(
            "--parse-dates", default="", help="comma-separated date columns"
        )
        parser.add_argument("--cost-model", help="JSON cost model")
        parser.add_argument("--baseline-column", help="current 0/1 decisions")
        parser.add_argument(
            "--replay",
            help="JSON with model_id, samples and tau: recorded LLM drafts",
        )
        parser.add_argument("--output", required=True, help="output directory")

    def execute(self, args: argparse.Namespace) -> int:
        """Writes the framing artifacts and, if one was drafted, the charter.

        The draft charter goes to `charter.json` in the output directory; it
        is unsigned.

        Args:
            args: The parsed command-line arguments.

        Returns:
            EXIT_BLOCKED (2) if the framing was refused, else EXIT_OK (0).
        """
        source = draft.Offer.from_dict(contracts.deserialize(args.source))
        dates = [c for c in args.parse_dates.split(",") if c]
        frame = pandas.read_csv(args.data, parse_dates=dates)
        text = pathlib.Path(args.request).read_text(encoding="utf-8")
        model = (
            costs.Tariff.from_dict(contracts.deserialize(args.cost_model))
            if args.cost_model
            else None
        )
        sampler = None
        tau = None
        if args.replay:
            recorded = contracts.deserialize(args.replay)
            sampler = refusal.Replayer(
                str(recorded["model_id"]), recorded["samples"]
            )
            tau = recorded.get("tau")
        result = draft.frame(
            draft.Request(text, source),
            frame,
            cost_model=model,
            baseline_column=args.baseline_column,
            sampler=sampler,
            tau=tau,
        )
        output = pathlib.Path(args.output)
        written = draft.emit(result, output)
        if result.charter is not None:
            (output / "charter.json").write_text(
                json.dumps(result.charter.to_dict(), indent=2) + "\n",
                encoding="utf-8",
                newline="\n",
            )
        LOGGER.info("framing %s: wrote %s", result.status, ", ".join(written))
        return (
            command.EXIT_BLOCKED
            if result.status == draft.REFUSED
            else command.EXIT_OK
        )
