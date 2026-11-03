"""Command-line interface.

Each sub-command is a `Command` subclass that registers its own arguments and
executes itself, so adding a command never touches the dispatcher.
"""

import abc
import argparse
from collections.abc import Sequence
import json
import logging
import pathlib
import sys
import threading
from typing import ClassVar

import pandas as pd

from scmle import DatasetCatalog
from scmle import durability
from scmle.api import project
from scmle.artifacts import bundle as bundle_module
from scmle.contracts import errors
from scmle.contracts import signing
from scmle.contracts import tasks
from scmle.core import config
from scmle.data.charter import contracts as charter_contracts
from scmle.data.charter import feasibility as charter_feasibility
from scmle.data.charter import sealing as charter_sealing
from scmle.data.readers import tabular
from scmle.inference import mapping as mapping_module
from scmle.inference import serving
from scmle.optimization import optimizers

LOGGER = logging.getLogger(__name__)
EXIT_OK = 0
EXIT_BLOCKED = 2
EXIT_FAILED = 1


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
        """Adds this command's arguments to `parser`."""

    @abc.abstractmethod
    def execute(self, args: argparse.Namespace) -> int:
        """Runs the command.

        Args:
            args: Parsed arguments.

        Returns:
            Process exit code.
        """


class ProjectCommand(Command):
    """Shared arguments for commands that build an `MLProject`."""

    def configure(self, parser: argparse.ArgumentParser) -> None:
        """Adds data and project options."""
        parser.add_argument("data", help="CSV or Parquet file")
        parser.add_argument("--target", required=True, nargs="+")
        parser.add_argument("--task", default="auto")
        parser.add_argument("--metric")
        parser.add_argument("--time-budget", type=int, default=1800)
        parser.add_argument("--workers", type=int, default=1)
        parser.add_argument("--group-column")
        parser.add_argument("--weight-column")
        parser.add_argument("--event-column")
        parser.add_argument("--query-column")
        parser.add_argument("--treatment-column")
        parser.add_argument("--memory-budget-gb", type=float)
        parser.add_argument(
            "--training", choices=["sample", "stream"], default="sample"
        )
        parser.add_argument("--horizon", type=int)
        parser.add_argument("--seasonal-period", type=int)
        parser.add_argument("--ordinal-order", nargs="+")
        parser.add_argument(
            "--unlabeled",
            choices=["block", "ignore", "pseudo_label"],
            default="block",
        )
        parser.add_argument("--time-column")
        parser.add_argument("--time-as-feature", action="store_true")
        parser.add_argument("--id-columns", nargs="*", default=[])
        parser.add_argument("--exclude", nargs="*", default=[])
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument(
            "--top-models",
            type=int,
            default=2,
            help="models to hyperparameter-tune (0 disables search)",
        )
        parser.add_argument("--configs-per-model", type=int, default=6)
        parser.add_argument(
            "--optimizer",
            choices=sorted(optimizers.OPTIMIZERS),
            default="random",
        )
        parser.add_argument("--output-dir", default="runs")
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
        parser.add_argument(
            "--monotonic",
            nargs="*",
            default=[],
            metavar="COLUMN=SIGN",
            help="predictions must not fall (+1) or rise (-1) as COLUMN grows",
        )
        parser.add_argument(
            "--sensitive-columns",
            nargs="*",
            default=[],
            help="columns whose groups the fairness report compares",
        )
        parser.add_argument(
            "--feature-delay",
            nargs="*",
            default=[],
            metavar="COLUMN=ROWS",
            help="columns known only ROWS rows after the row they describe",
        )
        parser.add_argument(
            "--embargo-rows",
            type=int,
            default=0,
            help="rows dropped between training and validation windows",
        )
        parser.add_argument(
            "--unavailable-at-prediction",
            nargs="*",
            default=[],
            help="columns known only after the outcome (excluded)",
        )
        parser.add_argument(
            "--strategies",
            nargs="*",
            help="data strategies to evaluate (default: all standard ones)",
        )
        parser.add_argument(
            "--enforcement", choices=["hard", "cooperative"], default="hard"
        )
        parser.add_argument(
            "--exclude-models",
            nargs="*",
            default=[],
            help="model names to skip",
        )
        parser.add_argument(
            "--exclude-families",
            nargs="*",
            default=[],
            help="model families to skip (for example deep_learning)",
        )
        parser.add_argument(
            "--leakage-policy", choices=["warn", "block"], default="warn"
        )
        parser.add_argument(
            "--charter",
            help="analysis charter (see `scmle charter init`): splits off a "
            "sealed confirmation set the exploration stages cannot read",
        )
        parser.add_argument(
            "--config",
            help="JSON file with the nested sections `search`, `strategies`, "
            "`ensemble`, `feature_graph`, `selection` and `decision` "
            "(same fields as the specification classes)",
        )

    def build_project(self, args: argparse.Namespace) -> project.MLProject:
        """Creates the project described by the arguments."""
        sections = config.load_sections(args.config)
        sections.setdefault(
            "search",
            tasks.SearchSpec(
                top_models=args.top_models,
                configs_per_model=args.configs_per_model,
                optimizer=args.optimizer,
            ),
        )
        if args.exclude_models or args.exclude_families:
            sections.setdefault(
                "models",
                tasks.ModelSpec(
                    exclude_models=args.exclude_models,
                    exclude_families=args.exclude_families,
                ),
            )
        if args.strategies is not None:
            sections.setdefault(
                "strategies", tasks.StrategySpec(names=args.strategies)
            )
        return project.MLProject(
            args.data,
            args.target[0] if len(args.target) == 1 else args.target,
            task=args.task,
            metric=args.metric,
            time_budget=args.time_budget,
            workers=args.workers,
            id_columns=args.id_columns,
            group_column=args.group_column,
            weight_column=args.weight_column,
            event_column=args.event_column,
            query_column=args.query_column,
            treatment_column=args.treatment_column,
            memory_budget_gb=args.memory_budget_gb,
            training=args.training,
            horizon=args.horizon,
            seasonal_period=args.seasonal_period,
            ordinal_order=args.ordinal_order,
            unlabeled=args.unlabeled,
            time_column=args.time_column,
            time_as_feature=args.time_as_feature,
            excluded_columns=args.exclude,
            seed=args.seed,
            validation=args.validation,
            folds=args.folds,
            test_fraction=args.test_fraction,
            embargo_rows=args.embargo_rows,
            sensitive_columns=args.sensitive_columns,
            monotonic_constraints={
                pair.split("=", 1)[0]: int(pair.split("=", 1)[1])
                for pair in args.monotonic
            },
            horizon_rows=args.horizon_rows,
            window_rows=args.window_rows,
            feature_delays={
                pair.split("=", 1)[0]: int(pair.split("=", 1)[1])
                for pair in args.feature_delay
            },
            unavailable_at_prediction=args.unavailable_at_prediction,
            enforcement=args.enforcement,
            leakage_policy=args.leakage_policy,
            charter=args.charter,
            output_dir=args.output_dir,
            **sections,
        )


class PlanCommand(ProjectCommand):
    """Prints the proposed plan without training."""

    name = "plan"
    help = "analyze the data and print the proposed plan"

    def execute(self, args: argparse.Namespace) -> int:
        """Prints the plan summary."""
        LOGGER.info(self.build_project(args).plan().summary())
        return EXIT_OK


class RunCommand(ProjectCommand):
    """Trains, selects, tests and packages."""

    name = "run"
    help = "execute the full workflow"

    def execute(self, args: argparse.Namespace) -> int:
        """Runs the project and prints the result summary."""
        result = self.build_project(args).run()
        LOGGER.info("%s", result.summary())
        LOGGER.info("artifacts: %s", result.run_dir)
        return EXIT_OK


class ResumeCommand(Command):
    """Continues an interrupted run."""

    name = "resume"
    help = "continue a cancelled, failed or crashed run, reusing its trials"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        """Adds the run directory argument."""
        parser.add_argument("run_directory")

    def execute(self, args: argparse.Namespace) -> int:
        """Resumes the run and prints the result summary."""
        result = project.MLProject.resume(args.run_directory)
        LOGGER.info("%s", result.summary())
        LOGGER.info("artifacts: %s", result.run_dir)
        return EXIT_OK


class CatalogCommand(Command):
    """Produces model-agnostic dataset curation evidence."""

    name = "catalog"
    help = "catalog a local table or opaque raw bytes"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        """Adds input, label, source and artifact options."""
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

    def read_raw_bytes(self, location: str) -> bytes:
        """Reads raw bytes from a local path or standard input."""
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
        """Builds, emits and optionally persists catalog evidence."""
        if args.input is not None and args.raw_bytes is not None:
            raise errors.ConfigError(
                "provide either a tabular input or --raw-bytes, not both"
            )
        if args.input is None and args.raw_bytes is None:
            raise errors.ConfigError(
                "catalog requires a tabular input or --raw-bytes"
            )

        raw_location = args.raw_bytes
        if args.input == "-":
            raw_location = "-"
        if raw_location is not None:
            catalog_input: bytes | pd.DataFrame = self.read_raw_bytes(
                raw_location
            )
            source = args.source_uri or (
                "memory://stdin" if raw_location == "-" else raw_location
            )
        else:
            loader = tabular.DatasetLoader()
            catalog_input = loader.load(args.input)
            source = args.source_uri or args.input

        try:
            result = DatasetCatalog.from_data(
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
        return EXIT_OK


class PredictCommand(Command):
    """Scores a dataset with a saved bundle."""

    name = "predict"
    help = "predict with a saved model bundle (trusted bundles only)"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        """Adds bundle, data and output options."""
        parser.add_argument("bundle", help="bundle directory")
        parser.add_argument("data", help="CSV or Parquet file")
        parser.add_argument("--output", default="predictions.csv")
        parser.add_argument(
            "--rename",
            action="append",
            default=[],
            metavar="INCOMING=EXPECTED",
            help="explicitly map an incoming column name onto the schema",
        )
        parser.add_argument("--case-insensitive", action="store_true")
        parser.add_argument("--strip-whitespace", action="store_true")
        parser.add_argument(
            "--trusted",
            action="store_true",
            help="load an unsigned bundle you vouch for (pickle executes code)",
        )

    def execute(self, args: argparse.Namespace) -> int:
        """Writes predictions (and probabilities for classifiers)."""
        model = bundle_module.ModelBundle.load(
            args.bundle, trusted=args.trusted
        )
        mapping = mapping_module.ColumnMapping(
            dict(pair.split("=", 1) for pair in args.rename),
            args.case_insensitive,
            args.strip_whitespace,
        )
        frame = pd.DataFrame({"prediction": model.predict(args.data, mapping)})
        if model.classes is not None:
            proba = model.predict_proba(args.data, mapping)
            for index, label in enumerate(model.classes):
                frame[f"proba_{label}"] = proba[:, index]
        durability.write_atomically(
            pathlib.Path(args.output), frame.to_csv(index=False).encode("utf-8")
        )
        LOGGER.info("wrote %d predictions to %s", len(frame), args.output)
        return EXIT_OK


class ServeCommand(Command):
    """Serves a saved bundle over HTTP."""

    name = "serve"
    help = "serve a saved model bundle over HTTP (trusted bundles only)"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        """Adds bundle and binding options."""
        parser.add_argument("bundle", help="bundle directory")
        parser.add_argument("--host", default="127.0.0.1")
        parser.add_argument("--port", type=int, default=8000)
        parser.add_argument(
            "--trusted",
            action="store_true",
            help="load an unsigned bundle you vouch for (pickle executes code)",
        )

    def execute(self, args: argparse.Namespace) -> int:
        """Loads the bundle and serves until interrupted."""
        model = bundle_module.ModelBundle.load(
            args.bundle, trusted=args.trusted
        )
        adapter = serving.HttpServing(model, args.host, args.port)
        adapter.start()
        LOGGER.info("serving %s on %s", model.info.model_name, adapter.address)
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            LOGGER.info("stopping")
        finally:
            adapter.stop()
        return EXIT_OK


class ExportOnnxCommand(Command):
    """Exports a saved bundle's pipeline as an ONNX graph."""

    name = "export-onnx"
    help = "export a saved model bundle as ONNX (needs the onnx extra)"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        """Adds bundle, output and parity-check options."""
        parser.add_argument("bundle", help="bundle directory")
        parser.add_argument("--output", default="model.onnx")
        parser.add_argument(
            "--check", help="CSV or Parquet rows to compare graph and pipeline"
        )
        parser.add_argument("--trusted", action="store_true")

    def execute(self, args: argparse.Namespace) -> int:
        """Writes the graph, optionally verifying it on sample rows."""
        model = bundle_module.ModelBundle.load(
            args.bundle, trusted=args.trusted
        )
        check = None if args.check is None else model.read_frame(args.check)
        written = model.export_onnx(args.output, check=check)
        LOGGER.info("wrote %s", written)
        return EXIT_OK


class PackageCommand(Command):
    """Writes a deployable service folder for a saved bundle."""

    name = "package"
    help = "package a saved model bundle as a ready-to-run service folder"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        """Adds bundle and destination options."""
        parser.add_argument("bundle", help="bundle directory")
        parser.add_argument("--output", default="service")
        parser.add_argument("--trusted", action="store_true")

    def execute(self, args: argparse.Namespace) -> int:
        """Writes the folder."""
        model = bundle_module.ModelBundle.load(
            args.bundle, trusted=args.trusted
        )
        folder = model.export_service(args.output)
        LOGGER.info("wrote service folder %s", folder)
        return EXIT_OK


class CharterCommand(Command):
    """Creates, checks and signs the analysis charter."""

    name = "charter"
    help = "create, check or sign an analysis charter"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        """Adds the `init`, `check` and `sign` actions."""
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
        """Runs the chosen action."""
        action = getattr(self, f"run_{args.charter_action}")
        return action(args)

    @staticmethod
    def run_init(args: argparse.Namespace) -> int:
        """Writes a template that fails `check` until it is filled in."""
        path = pathlib.Path(args.path)
        if path.exists() and not args.force:
            raise errors.ConfigError(f"{path} exists; pass --force")
        path.write_text(
            json.dumps(charter_contracts.template(), indent=2) + "\n"
        )
        LOGGER.info("wrote template charter %s", path)
        return EXIT_OK

    @staticmethod
    def run_check(args: argparse.Namespace) -> int:
        """Validates the charter, its hash and (if present) its signature."""
        loaded = charter_sealing.read_verified(args.path)
        charter = loaded.charter
        LOGGER.info(
            "charter %s %s",
            loaded.charter_hash,
            "signature verified" if loaded.signed else "unsigned or unchecked",
        )
        if args.rows is not None:
            report = charter_feasibility.assess(
                n_confirmation=int(
                    args.rows * charter.sets.confirmation_fraction
                ),
                min_effect=charter.min_effect,
                alpha=float(charter.error_budget.confirmation.alpha_confirm),
                k_max=charter.k_max,
                power=charter.power,
            )
            LOGGER.info("feasibility: %s (%s)", report.verdict, report.reason)
        return EXIT_OK

    @staticmethod
    def run_sign(args: argparse.Namespace) -> int:
        """Signs the charter with a key from a file or the environment."""
        key = (
            pathlib.Path(args.key_file).read_text().strip()
            if args.key_file
            else None
        )
        if signing.resolve_key(key) is None:
            raise errors.ConfigError(
                "no signing key: set SCMLE_SIGNING_KEY or pass --key-file"
            )
        charter = charter_contracts.read_charter(args.path)
        output = charter_sealing.write_charter(
            args.output or args.path, charter, signing_key=key
        )
        LOGGER.info("signed %s", output)
        return EXIT_OK


COMMANDS: tuple[Command, ...] = (
    PlanCommand(),
    RunCommand(),
    ResumeCommand(),
    CatalogCommand(),
    PredictCommand(),
    ServeCommand(),
    ExportOnnxCommand(),
    PackageCommand(),
    CharterCommand(),
)


def build_parser() -> argparse.ArgumentParser:
    """Builds the argument parser from the registered commands."""
    parser = argparse.ArgumentParser(prog="scmle")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="verbosity of the log written to standard error",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in COMMANDS:
        command.configure(
            subparsers.add_parser(command.name, help=command.help)
        )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Arguments; defaults to `sys.argv[1:]`.

    Returns:
        0 on success, 2 when the run was blocked, 1 on other failures.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(message)s")
    command = next(c for c in COMMANDS if c.name == args.command)
    try:
        return command.execute(args)
    except errors.BlockedError as error:
        LOGGER.error("blocked: %s", error)
        return EXIT_BLOCKED
    except errors.ScmleError as error:
        LOGGER.error("%s error: %s", error.category, error)
        return EXIT_FAILED


if __name__ == "__main__":
    sys.exit(main())
