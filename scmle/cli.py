"""Command-line interface.

Each sub-command is a `Command` subclass that registers its own arguments and
executes itself, so adding a command never touches the dispatcher.
"""

import abc
import argparse
from collections.abc import Sequence
import json
import logging
import os
import pathlib
import sys
import threading
import types
from typing import ClassVar

import pandas as pd

from scmle import DatasetCatalog
from scmle import durability
from scmle.api import project
from scmle.artifacts import bundle as bundle_module
from scmle.artifacts import claims_audit
from scmle.contracts import errors
from scmle.contracts import signing
from scmle.contracts import tasks
from scmle.core import config
from scmle.data.charter import contracts as charter_contracts
from scmle.data.charter import feasibility as charter_feasibility
from scmle.data.charter import sealing as charter_sealing
from scmle.data.framing import costs as framing_costs
from scmle.data.framing import draft as framing_draft
from scmle.data.framing import refusal as framing_refusal
from scmle.data.ingest import budget as ingest_budget
from scmle.data.ingest import grain as ingest_grain
from scmle.data.ingest import plan as ingest_plan
from scmle.data.readers import tabular
from scmle.deploy import gate as deploy_gate
from scmle.deploy import livetests
from scmle.deploy import record as deploy_record
from scmle.deploy import release as deploy_release
from scmle.governance import deployment_ledger
from scmle.governance import envelope as envelope_module
from scmle.governance import loop as loop_module
from scmle.governance import registry as registry_module
from scmle.governance import release_state
from scmle.inference import mapping as mapping_module
from scmle.inference import parity
from scmle.inference import serving
from scmle.operations import controller as controller_module
from scmle.optimization import optimizers
from scmle.security import access
from scmle.trail import builder as trail_build
from scmle.trail import chain as trail_chain
from scmle.trail import show as trail_show

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
        parser.add_argument("--baseline-column")
        parser.add_argument(
            "--baseline-subject",
            choices=("frozen_rule", "selected_model"),
            default="frozen_rule",
        )
        parser.add_argument("--framing-draft")
        parser.add_argument("--propensity-column")
        parser.add_argument("--logged-action-column")
        parser.add_argument("--ingest-plan")
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
            "--leakage-attestations",
            help="JSON file of per-column attestations that a strong "
            "predictor flagged by the leakage screen is legitimate "
            "(charter runs only)",
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
            leakage_attestations=args.leakage_attestations,
            baseline_column=args.baseline_column,
            baseline_subject=args.baseline_subject,
            framing_draft=args.framing_draft,
            propensity_column=args.propensity_column,
            logged_action_column=args.logged_action_column,
            ingest_plan=args.ingest_plan,
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
        try:
            result = project.MLProject.resume(args.run_directory)
        except FileNotFoundError as error:
            raise errors.ArtifactError(str(error)) from error
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


class BatchCommand(Command):
    """Scores a file through the same path as the online service."""

    name = "batch"
    help = (
        "score a file with the online serving path and write a manifest "
        "with the model version and artifact digest"
    )

    def configure(self, parser: argparse.ArgumentParser) -> None:
        """Adds bundle, data, output and policy options."""
        parser.add_argument("bundle", help="bundle directory")
        parser.add_argument("data", help="CSV or Parquet file")
        parser.add_argument("--output", default="predictions.csv")
        parser.add_argument("--chunk-rows", type=int, default=10_000)
        parser.add_argument(
            "--on-null", choices=["impute", "reject"], default="impute"
        )
        parser.add_argument(
            "--on-unseen", choices=["report", "reject"], default="report"
        )
        parser.add_argument(
            "--trusted",
            action="store_true",
            help="load an unsigned bundle you vouch for (pickle executes code)",
        )

    def execute(self, args: argparse.Namespace) -> int:
        """Writes predictions and `<output>.manifest.json`."""
        model = bundle_module.ModelBundle.load(
            args.bundle, trusted=args.trusted
        )
        adapter = serving.BatchServing(
            model, on_null=args.on_null, on_unseen=args.on_unseen
        )
        rows = adapter.score_file(args.data, args.output, args.chunk_rows)
        LOGGER.info("wrote %d predictions to %s", rows, args.output)
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
        adapter = serving.HttpServing(
            model,
            args.host,
            args.port,
            gateway=access.gateway_from_environment(os.environ),
        )
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


class IngestCommand(Command):
    """Profiles a SQLite source: catalog, keys, unit verdict and joins."""

    name = "ingest"
    help = "profile a relational source: tables, keys, grain and joins"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        """Adds source, scope, tolerance and budget options."""
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
        parser.add_argument(
            "--max-key-size", type=int, default=ingest_grain.LEVELS
        )
        parser.add_argument("--max-queries", type=int)
        parser.add_argument("--max-rows-scanned", type=int)
        parser.add_argument("--max-seconds", type=float)
        parser.add_argument(
            "--foreign-keys",
            help="JSON file: [{child, child_columns, parent, parent_columns}]",
        )

    def execute(self, args: argparse.Namespace) -> int:
        """Runs the stages and prints a summary; 2 if the unit blocks."""
        declared: tuple[ingest_plan.joins.DeclaredKey, ...] = ()
        if args.foreign_keys:
            declared = ingest_plan.declared_from_json(
                json.loads(pathlib.Path(args.foreign_keys).read_text())
            )
        settings = ingest_plan.IngestSettings(
            tables=tuple(t for t in args.tables.split(",") if t),
            unit=args.unit,
            target_table=args.target_table,
            eps_key=args.eps_key,
            eps_null=args.eps_null,
            eps_fk=args.eps_fk,
            max_key_size=args.max_key_size,
            declared=declared,
        )
        plan = ingest_plan.IngestPlan(
            pathlib.Path(args.output),
            ingest_plan.sqlite_factory(args.sqlite),
            settings,
            ingest_budget.IngestBudget(
                args.max_queries, args.max_rows_scanned, args.max_seconds
            ),
        )
        summary = plan.run_all()
        print(json.dumps(summary, indent=2))
        if plan.verdict is not None and plan.verdict.blocks:
            raise errors.BlockedError(
                f"unit verdict {plan.verdict.verdict}: {plan.verdict.reason}"
            )
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
            json.dumps(charter_contracts.template(), indent=2) + "\n",
            newline="\n",
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


class LoopCommand(Command):
    """Drives the closed loop of a deployment (observe, step, ask, reply)."""

    name = "loop"
    help = "observe batches, step the closed loop, list and answer escalations"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        """Adds the loop actions."""
        actions = parser.add_subparsers(dest="loop_action", required=True)
        init = actions.add_parser("init", help="create the deployment ledger")
        init.add_argument("deploy_dir")
        init.add_argument("--policy", required=True, help="policy JSON file")
        init.add_argument("--charter-hash", required=True)
        observe = actions.add_parser("observe", help="route a batch of cases")
        observe.add_argument("deploy_dir")
        observe.add_argument("--batch", required=True, help="CSV of cases")
        observe.add_argument("--predictions-column", default="prediction")
        observe.add_argument("--decisions-column", default="decision")
        mature = actions.add_parser("mature", help="record arrived losses")
        mature.add_argument("deploy_dir")
        mature.add_argument(
            "--labels", required=True, help="CSV with case_id and loss"
        )
        step = actions.add_parser("step", help="apply the decision map")
        step.add_argument("deploy_dir")
        pending = actions.add_parser("pending", help="list open escalations")
        pending.add_argument("deploy_dir")
        reply = actions.add_parser("reply", help="record a signed reply")
        reply.add_argument("deploy_dir")
        reply.add_argument("escalation_id")
        reply.add_argument(
            "--attestation", required=True, help="attestation JSON file"
        )
        reply.add_argument(
            "--key-file",
            required=True,
            help="file with the key that verifies the reply (the person's)",
        )
        verify = actions.add_parser("verify", help="verify the ledger chain")
        verify.add_argument("deploy_dir")

    def execute(self, args: argparse.Namespace) -> int:
        """Runs the chosen action."""
        action = getattr(self, f"run_{args.loop_action}")
        return action(args)

    @staticmethod
    def run_init(args: argparse.Namespace) -> int:
        """Creates the ledger from a policy file and a charter hash."""
        policy = loop_module.LoopPolicy.from_dict(
            json.loads(pathlib.Path(args.policy).read_text())
        )
        charter = types.SimpleNamespace(charter_hash=args.charter_hash)
        loop_module.Loop.open(args.deploy_dir, charter, policy)
        LOGGER.info("opened deployment ledger in %s", args.deploy_dir)
        return EXIT_OK

    @staticmethod
    def run_observe(args: argparse.Namespace) -> int:
        """Routes one batch and prints how many cases went where."""
        loop = loop_module.Loop.resume(args.deploy_dir)
        frame = pd.read_csv(args.batch)
        outcome = loop.observe(
            frame,
            frame[args.predictions_column].tolist(),
            frame[args.decisions_column].tolist(),
        )
        counts = pd.Series(outcome.routing.status).value_counts().to_dict()
        print(json.dumps({"batch": outcome.batch, "routing": counts}))
        return EXIT_OK

    @staticmethod
    def run_mature(args: argparse.Namespace) -> int:
        """Stores arrived losses by case id."""
        loop = loop_module.Loop.resume(args.deploy_dir)
        frame = pd.read_csv(args.labels)
        matched = loop.mature(
            dict(zip(frame["case_id"].astype(str), frame["loss"], strict=True))
        )
        print(json.dumps({"matched": matched}))
        return EXIT_OK

    @staticmethod
    def run_step(args: argparse.Namespace) -> int:
        """Steps the loop and prints what it found."""
        loop = loop_module.Loop.resume(args.deploy_dir)
        result = loop.step()
        print(
            json.dumps(
                {
                    "status": result.status,
                    "mechanism": result.mechanism.value,
                    "events": [list(event) for event in result.events],
                    "escalation": (
                        None
                        if result.escalation is None
                        else result.escalation.id
                    ),
                }
            )
        )
        return EXIT_OK

    @staticmethod
    def run_pending(args: argparse.Namespace) -> int:
        """Lists open escalations with their questions."""
        loop = loop_module.Loop.resume(args.deploy_dir)
        print(
            json.dumps(
                [
                    {
                        "id": e.id,
                        "question": e.question,
                        "question_hash": e.question_hash,
                        "evidence": list(e.evidence),
                        "proposed_default": e.proposed_default,
                    }
                    for e in loop.pending_escalations()
                ],
                indent=2,
            )
        )
        return EXIT_OK

    @staticmethod
    def run_reply(args: argparse.Namespace) -> int:
        """Records a person's signed reply; refused replies fail."""
        loop = loop_module.Loop.resume(args.deploy_dir)
        reply = deployment_ledger.Attestation.from_dict(
            json.loads(pathlib.Path(args.attestation).read_text())
        )
        key = pathlib.Path(args.key_file).read_text().strip()
        try:
            outcome = loop.reply(args.escalation_id, reply, key)
        except loop_module.ReplyRefused as error:
            raise errors.ConfigError(f"reply refused: {error}") from error
        print(
            json.dumps(
                {
                    "tier": outcome.tier.value,
                    "decision": outcome.decision.value,
                    "safe_default_applies": outcome.safe_default_applies,
                }
            )
        )
        return EXIT_OK

    @staticmethod
    def run_verify(args: argparse.Namespace) -> int:
        """Verifies the ledger chain and the e-LOND replay."""
        path = pathlib.Path(args.deploy_dir) / deployment_ledger.FILE_NAME
        try:
            ledger = deployment_ledger.DeploymentLedger.load(path)
        except deployment_ledger.DeploymentLedgerError as error:
            raise errors.ArtifactError(
                f"ledger verification failed: {error}"
            ) from error
        print(
            json.dumps(
                {"ok": True, "rows": len(ledger.rows), "head": ledger.head}
            )
        )
        return EXIT_OK


class OperateCommand(Command):
    """Runs the unattended controller: ticks, envelopes, status and halt."""

    name = "operate"
    help = (
        "run the unattended controller on an inbox; grant or revoke authority"
    )

    def configure(self, parser: argparse.ArgumentParser) -> None:
        """Adds the controller actions."""
        actions = parser.add_subparsers(dest="operate_action", required=True)
        tick = actions.add_parser(
            "tick", help="apply new inbox files once and step the loop"
        )
        tick.add_argument("deploy_dir")
        tick.add_argument(
            "--inbox",
            required=True,
            help="folder with batches/*.csv and labels/*.csv",
        )
        tick.add_argument(
            "--keys-file",
            help="JSON map from key id to the key that verifies it",
        )
        status = actions.add_parser("status", help="show the controller state")
        status.add_argument("deploy_dir")
        grant = actions.add_parser(
            "grant", help="record a person's signed envelope"
        )
        grant.add_argument("deploy_dir")
        grant.add_argument("--envelope", required=True, help="envelope JSON")
        grant.add_argument(
            "--key-file",
            required=True,
            help="file with the key that verifies it (the person's)",
        )
        revoke = actions.add_parser("revoke", help="remove an envelope")
        revoke.add_argument("deploy_dir")
        revoke.add_argument("envelope_id")
        revoke.add_argument("--reason", required=True)
        halt = actions.add_parser(
            "halt", help="stop the loop; every case is escalated"
        )
        halt.add_argument("deploy_dir")
        halt.add_argument("--reason", required=True)

    def execute(self, args: argparse.Namespace) -> int:
        """Runs the chosen action."""
        action = getattr(self, f"run_{args.operate_action}")
        return action(args)

    @staticmethod
    def open(
        deploy_dir: str, keys: dict[str, str] | None = None
    ) -> controller_module.Controller:
        """Opens the controller of a deployment, mapping ledger faults."""
        try:
            return controller_module.Controller(
                deploy_dir, verification_keys=keys
            )
        except deployment_ledger.DeploymentLedgerError as error:
            raise errors.ArtifactError(
                f"cannot open the deployment: {error}"
            ) from error

    @classmethod
    def run_tick(cls, args: argparse.Namespace) -> int:
        """Applies the inbox once and prints what happened."""
        keys = (
            json.loads(pathlib.Path(args.keys_file).read_text())
            if args.keys_file
            else None
        )
        result = cls.open(args.deploy_dir, keys).tick(args.inbox)
        print(json.dumps(result.to_dict()))
        return EXIT_OK

    @classmethod
    def run_status(cls, args: argparse.Namespace) -> int:
        """Prints the controller's state."""
        print(json.dumps(cls.open(args.deploy_dir).status(), indent=2))
        return EXIT_OK

    @classmethod
    def run_grant(cls, args: argparse.Namespace) -> int:
        """Records an envelope after checking its signature."""
        key = pathlib.Path(args.key_file).read_text().strip()
        envelope = deployment_ledger.Envelope.from_dict(
            json.loads(pathlib.Path(args.envelope).read_text())
        )
        controller = cls.open(
            args.deploy_dir, {deployment_ledger.key_fingerprint(key): key}
        )
        try:
            controller.envelopes.grant(envelope)
        except (
            envelope_module.EnvelopeRefused,
            deployment_ledger.DeploymentLedgerError,
        ) as error:
            raise errors.ConfigError(f"envelope refused: {error}") from error
        print(json.dumps(controller.status()["envelopes"]))
        return EXIT_OK

    @classmethod
    def run_revoke(cls, args: argparse.Namespace) -> int:
        """Removes an envelope's authority."""
        controller = cls.open(args.deploy_dir)
        try:
            controller.envelopes.revoke(args.envelope_id, args.reason)
        except deployment_ledger.DeploymentLedgerError as error:
            raise errors.ConfigError(f"revoke refused: {error}") from error
        print(json.dumps(controller.status()["envelopes"]))
        return EXIT_OK

    @classmethod
    def run_halt(cls, args: argparse.Namespace) -> int:
        """Halts the loop; only a passing recalibration releases it."""
        controller = cls.open(args.deploy_dir)
        controller.loop.enter_halt("operator_halt", {"reason": args.reason})
        controller.loop.save_state()
        print(json.dumps(controller.status()))
        return EXIT_OK


class DeployCommand(Command):
    """Release evidence: live checks, the promotion gate, deployment records."""

    name = "deploy"
    help = "live checks of a service, the promotion gate, deployment records"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        """Adds the `live`, `gate` and `verify-record` actions."""
        actions = parser.add_subparsers(dest="deploy_action", required=True)
        live = actions.add_parser(
            "live", help="health, schema, inference and security checks"
        )
        live.add_argument("--url", required=True, help="base URL (https)")
        live.add_argument("--token-file", required=True)
        live.add_argument("--read-only-token-file")
        live.add_argument("--golden", default="deploy/local/golden")
        live.add_argument("--environment", required=True)
        live.add_argument("--commit", required=True)
        live.add_argument("--container-digest", required=True)
        live.add_argument("--model-digest", required=True)
        live.add_argument("--timeout", type=float, default=30.0)
        live.add_argument("--output", required=True, help="results JSON")
        decide = actions.add_parser(
            "gate", help="allow or refuse a release from its evidence"
        )
        decide.add_argument("stage", choices=["admit", "certify"])
        decide.add_argument("manifest", help="release manifest JSON")
        decide.add_argument("--key-file", required=True)
        decide.add_argument("--output", help="signed record (certify)")
        digest = actions.add_parser(
            "model-digest", help="digest a service reports for a version"
        )
        digest.add_argument("registry", help="registry directory")
        digest.add_argument("--model", required=True)
        digest.add_argument("--version", type=int, required=True)
        reconcile = actions.add_parser(
            "reconcile",
            help="compare the registry and ledger with the runtime report",
        )
        reconcile.add_argument("registry", help="registry directory")
        reconcile.add_argument("--model", required=True)
        reconcile.add_argument("--environment", required=True)
        reconcile.add_argument(
            "--observation",
            required=True,
            help="observation JSON (Observation.to_dict)",
        )
        reconcile.add_argument(
            "--ledger", help="deployment ledger file (deployment_ledger.jsonl)"
        )
        reconcile.add_argument("--output", help="write the report here")
        verify = actions.add_parser(
            "verify-record", help="check a deployment record's signature"
        )
        verify.add_argument("record")
        verify.add_argument("--key-file", required=True)

    def execute(self, args: argparse.Namespace) -> int:
        """Runs the chosen action."""
        action = getattr(self, f"run_{args.deploy_action.replace('-', '_')}")
        return action(args)

    @staticmethod
    def read_key(path: str) -> str:
        """Reads a key from a file (never from the command line)."""
        key = pathlib.Path(path).read_text(encoding="utf-8").strip()
        if not key:
            raise errors.ConfigError(f"{path} holds no key")
        return key

    @staticmethod
    def run_live(args: argparse.Namespace) -> int:
        """Runs the four live checks; exit 1 if any failed."""
        read_only = (
            DeployCommand.read_key(args.read_only_token_file)
            if args.read_only_token_file
            else None
        )
        target = livetests.Target(
            environment=args.environment,
            source_commit=args.commit,
            container_digest=args.container_digest,
            model_digest=args.model_digest,
            token=DeployCommand.read_key(args.token_file),
            base_url=args.url,
            read_only_token=read_only,
        )
        golden = parity.load_golden(args.golden)
        expected = parity.read_json(pathlib.Path(args.golden) / "expected.json")
        results = livetests.run_checks(
            livetests.http_fetch(args.url, args.timeout),
            target,
            golden,
            expected,
        )
        payload = [item.to_dict() for item in results]
        pathlib.Path(args.output).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        print(json.dumps({item.name: item.passed for item in results}))
        return EXIT_OK if all(item.passed for item in results) else EXIT_FAILED

    @staticmethod
    def run_gate(args: argparse.Namespace) -> int:
        """Prints the decision; a refusal exits with the blocked status."""
        key = DeployCommand.read_key(args.key_file)
        candidate = deploy_release.load_candidate(args.manifest)
        if args.stage == "admit":
            decision = deploy_gate.admit(candidate, key)
        else:
            decision = deploy_gate.certify(candidate, key)
        print(json.dumps(decision.to_dict(), indent=2, sort_keys=True))
        if not decision.allowed:
            return EXIT_BLOCKED
        if decision.record is not None and args.output:
            deploy_record.write_record(args.output, decision.record)
        return EXIT_OK

    @staticmethod
    def run_reconcile(args: argparse.Namespace) -> int:
        """Exits 1 if the registry, ledger and runtime disagree."""
        store = registry_module.ModelRegistryStore(args.registry)
        ledger = (
            deployment_ledger.DeploymentLedger.load(args.ledger)
            if args.ledger
            else None
        )
        seen = deploy_release.observation_from_dict(
            deploy_release.read_json(pathlib.Path(args.observation))
        )
        report = release_state.ReleaseRegistry(store, ledger).reconcile(
            args.model, args.environment, seen
        )
        text = json.dumps(report.to_dict(), indent=2, sort_keys=True)
        if args.output:
            pathlib.Path(args.output).write_text(
                text + "\n", encoding="utf-8", newline="\n"
            )
        print(text)
        return EXIT_OK if report.consistent else EXIT_FAILED

    @staticmethod
    def run_model_digest(args: argparse.Namespace) -> int:
        """Prints the digest a service running this version reports."""
        store = registry_module.ModelRegistryStore(args.registry)
        version = store.resolve(args.model, args.version)
        print(deploy_gate.model_digest_of(store, version))
        return EXIT_OK

    @staticmethod
    def run_verify_record(args: argparse.Namespace) -> int:
        """Exits 0 only if the record's signature verifies."""
        record = deploy_record.read_record(args.record)
        valid = record.verify(DeployCommand.read_key(args.key_file))
        print(json.dumps({"valid": valid, "digest": record.digest()}))
        return EXIT_OK if valid else EXIT_FAILED


class ClaimsCommand(Command):
    """Lists, shows and verifies the claims of a charter run."""

    name = "claims"
    help = "list, show or verify the claims and ledger of a charter run"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        """Adds the `list`, `show` and `verify` actions."""
        actions = parser.add_subparsers(dest="claims_action", required=True)
        listing = actions.add_parser("list", help="one row per claim")
        listing.add_argument("run_dir")
        listing.add_argument("--tier", choices=["confirmed", "unverified"])
        listing.add_argument("--json", action="store_true")
        show = actions.add_parser("show", help="everything about one claim")
        show.add_argument("run_dir")
        show.add_argument("claim_id")
        verify = actions.add_parser(
            "verify",
            help="check the ledger chain and replay, the claims file, the "
            "second-stage accounting, the custody log and the charter",
        )
        verify.add_argument("run_dir")
        verify.add_argument("--json", action="store_true")
        verify.add_argument(
            "--key-file",
            help="file holding the signing key; default: SCMLE_SIGNING_KEY",
        )

    def execute(self, args: argparse.Namespace) -> int:
        """Runs the chosen action."""
        action = getattr(self, f"run_{args.claims_action}")
        return action(args)

    @staticmethod
    def run_list(args: argparse.Namespace) -> int:
        """Prints the claims of a run."""
        rows = claims_audit.list_claims(args.run_dir)
        if args.tier:
            rows = [r for r in rows if r["tier"] == args.tier]
        if args.json:
            print(json.dumps(rows, indent=2))
            return EXIT_OK
        for row in rows:
            if row.get("kind"):
                print(
                    f"{row['claim_id']}  {row['tier']:<10}  "
                    f"registered {row['kind']}  status={row['status']}"
                    + (
                        f"  identification={row['identification']}"
                        if row["identification"]
                        else ""
                    )
                )
                continue
            adjusted = row["adjusted_p_value"]
            print(
                f"{row['claim_id']}  {row['tier']:<10}  "
                f"{row['feature']!s:<24}  "
                f"adj.p={'-' if adjusted is None else f'{adjusted:.4g}'}  "
                f"effect={row['effect']:.4g}"
                + ("  withdrawn before C" if row["withdrawn"] else "")
            )
        print(f"{len(rows)} claim(s)")
        return EXIT_OK

    @staticmethod
    def run_show(args: argparse.Namespace) -> int:
        """Prints one claim with its hypothesis and checks."""
        print(
            json.dumps(
                claims_audit.show_claim(args.run_dir, args.claim_id),
                indent=2,
                sort_keys=True,
            )
        )
        return EXIT_OK

    @staticmethod
    def run_verify(args: argparse.Namespace) -> int:
        """Verifies the run's artifacts; a failed step makes the exit code 1."""
        key = (
            pathlib.Path(args.key_file).read_text().strip()
            if args.key_file
            else None
        )
        result = claims_audit.verify(args.run_dir, key)
        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            for found in result.findings:
                print(f"{found.status.upper():<8}{found.name}: {found.detail}")
            print(
                f"{result.verified} of {len(result.findings)} steps passed"
                + ("" if result.ok else "; verification FAILED")
            )
        if not result.ok:
            raise errors.ArtifactError("claims verification failed")
        return EXIT_OK


class FrameCommand(Command):
    """Drafts a charter, cost model and questions from a plain request."""

    name = "frame"
    help = (
        "draft an unsigned charter and the questions only a person can answer"
    )

    def configure(self, parser: argparse.ArgumentParser) -> None:
        """Adds the request, source, data and optional inputs."""
        parser.add_argument("request", help="text file with the request")
        parser.add_argument(
            "--source",
            required=True,
            help="JSON: uri, columns, primary_key, target_time_column",
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
        """Writes the framing artifacts and the unsigned draft charter."""
        source = framing_draft.SourceDescription.from_dict(
            charter_contracts.load_json(args.source)
        )
        dates = [c for c in args.parse_dates.split(",") if c]
        frame = pd.read_csv(args.data, parse_dates=dates)
        text = pathlib.Path(args.request).read_text(encoding="utf-8")
        model = (
            framing_costs.DecisionCostModel.from_dict(
                charter_contracts.load_json(args.cost_model)
            )
            if args.cost_model
            else None
        )
        sampler = None
        tau = None
        if args.replay:
            recorded = charter_contracts.load_json(args.replay)
            sampler = framing_refusal.ReplaySampler(
                str(recorded["model_id"]), recorded["samples"]
            )
            tau = recorded.get("tau")
        result = framing_draft.frame(
            framing_draft.FramingRequest(text, source),
            frame,
            cost_model=model,
            baseline_column=args.baseline_column,
            sampler=sampler,
            tau=tau,
        )
        output = pathlib.Path(args.output)
        written = framing_draft.write_artifacts(result, output)
        if result.charter is not None:
            (output / "charter.json").write_text(
                json.dumps(result.charter.to_dict(), indent=2) + "\n",
                encoding="utf-8",
                newline="\n",
            )
        LOGGER.info("framing %s: wrote %s", result.status, ", ".join(written))
        return (
            EXIT_BLOCKED if result.status == framing_draft.REFUSED else EXIT_OK
        )


class TrailCommand(Command):
    """Builds, verifies and prints the trail of one deployed model."""

    name = "trail"
    help = "build, verify or show the business trail of a deployed model"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        """Adds the `build`, `verify` and `show` actions."""
        actions = parser.add_subparsers(dest="trail_action", required=True)
        build = actions.add_parser(
            "build", help="copy the retained records and seal the trail"
        )
        build.add_argument("output", help="trail folder to write")
        build.add_argument("--run", required=True, help="charter run folder")
        build.add_argument("--framing", required=True, help="frame output")
        build.add_argument("--registry", required=True, help="registry root")
        build.add_argument(
            "--deployment",
            required=True,
            help="loop folder with observed_outcomes.json",
        )
        build.add_argument("--inputs", required=True, help="signed inputs")
        verify = actions.add_parser(
            "verify", help="check a trail folder against its own records"
        )
        verify.add_argument("trail_dir")
        show = actions.add_parser("show", help="print the trail")
        show.add_argument("trail_dir")
        for sub in (build, verify):
            sub.add_argument(
                "--keys",
                help="JSON file mapping analyst, stakeholder and reviewer "
                "to keys; default: SCMLE_SIGNING_KEY for the analyst only",
            )
            sub.add_argument("--json", action="store_true")

    def execute(self, args: argparse.Namespace) -> int:
        """Runs the chosen action."""
        if args.trail_action == "show":
            print(trail_show.render(args.trail_dir))
            return EXIT_OK
        keys = self.keys(args.keys)
        if args.trail_action == "build":
            report = trail_build.build(
                args.output,
                run_dir=args.run,
                framing_dir=args.framing,
                registry_dir=args.registry,
                deployment_dir=args.deployment,
                inputs_file=args.inputs,
                keys=keys,
            )
        else:
            report = trail_chain.verify(args.trail_dir, keys)
        if args.json:
            print(json.dumps(report.to_dict(), indent=2))
        else:
            for found in report.findings:
                print(f"{found.kind:<30}{found.step}: {found.detail}")
            print(
                f"{report.steps} steps, head {report.head[:12]}; "
                + ("verified" if report.ok else "NOT verified")
            )
        if not report.ok:
            raise errors.ArtifactError("trail verification failed")
        return EXIT_OK

    @staticmethod
    def keys(path: str | None) -> dict[str, str | None]:
        """Keys by role from a file, else the environment key as analyst."""
        if path:
            loaded = json.loads(pathlib.Path(path).read_text("utf-8"))
            return {str(k): str(v) for k, v in loaded.items()}
        configured = signing.key_from_environment(os.environ)
        return {"analyst": configured}


COMMANDS: tuple[Command, ...] = (
    PlanCommand(),
    RunCommand(),
    ResumeCommand(),
    CatalogCommand(),
    PredictCommand(),
    BatchCommand(),
    ServeCommand(),
    ExportOnnxCommand(),
    PackageCommand(),
    IngestCommand(),
    CharterCommand(),
    ClaimsCommand(),
    FrameCommand(),
    LoopCommand(),
    OperateCommand(),
    DeployCommand(),
    TrailCommand(),
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
