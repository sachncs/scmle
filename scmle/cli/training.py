"""Commands that build a project from a data file: plan, run and resume."""

import argparse
import logging

from scmle.api import project
from scmle.cli import command
from scmle.contracts import errors
from scmle.contracts import tasks
from scmle.optimization import optimizers
from scmle.runtime import config

LOGGER = logging.getLogger(__name__)


def parse(pairs: list[str]) -> dict[str, int]:
    """Reads `COLUMN=NUMBER` pairs given on the command line.

    Args:
        pairs: The pairs, each split at its first `=`.

    Returns:
        Each column mapped to its integer.
    """
    return {pair.split("=", 1)[0]: int(pair.split("=", 1)[1]) for pair in pairs}


def constrain(parser: argparse.ArgumentParser) -> None:
    """Adds the monotonic, fairness, timing, strategy and enforcement options.

    Args:
        parser: The argument parser of the command.
    """
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


def govern(parser: argparse.ArgumentParser) -> None:
    """Adds the model exclusion, leakage, charter and config file options.

    Args:
        parser: The argument parser of the command.
    """
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


class ProjectCommand(command.Command):
    """Shared arguments for commands that build an `Project`."""

    def configure(self, parser: argparse.ArgumentParser) -> None:
        """Adds data and project options.

        Args:
            parser: The argument parser of this command.
        """
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
        command.validators(parser)
        constrain(parser)
        govern(parser)

    def scaffold(self, args: argparse.Namespace) -> project.Project:
        """Creates the project described by the arguments.

        Args:
            args: The parsed command-line arguments, including the optional
                `--config` sections file.

        Returns:
            The project; search, model and strategy settings given on the
            command line are used unless the config file sets that section.
        """
        sections = config.overlay(args.config)
        sections.setdefault(
            "search",
            tasks.Search(
                top_models=args.top_models,
                configs_per_model=args.configs_per_model,
                optimizer=args.optimizer,
            ),
        )
        if args.exclude_models or args.exclude_families:
            sections.setdefault(
                "models",
                tasks.Candidates(
                    exclude_models=args.exclude_models,
                    exclude_families=args.exclude_families,
                ),
            )
        if args.strategies is not None:
            sections.setdefault(
                "strategies", tasks.Strategies(names=args.strategies)
            )
        return project.Project(
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
            monotonic_constraints=parse(args.monotonic),
            horizon_rows=args.horizon_rows,
            window_rows=args.window_rows,
            feature_delays=parse(args.feature_delay),
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


class Plan(ProjectCommand):
    """Prints the proposed plan without training."""

    name = "plan"
    help = "analyze the data and print the proposed plan"

    def execute(self, args: argparse.Namespace) -> int:
        """Logs the summary of the execution plan at INFO level.

        Args:
            args: The parsed command-line arguments.

        Returns:
            The exit status, EXIT_OK (0).
        """
        LOGGER.info(self.scaffold(args).plan().summary())
        return command.EXIT_OK


class Run(ProjectCommand):
    """Trains, selects, tests and packages."""

    name = "run"
    help = "execute the full workflow"

    def execute(self, args: argparse.Namespace) -> int:
        """Runs the project and logs the result summary and run directory.

        Args:
            args: The parsed command-line arguments.

        Returns:
            The exit status, EXIT_OK (0).
        """
        result = self.scaffold(args).run()
        LOGGER.info("%s", result.summary())
        LOGGER.info("artifacts: %s", result.run_dir)
        return command.EXIT_OK


class Resumption(command.Command):
    """Continues an interrupted run."""

    name = "resume"
    help = "continue a cancelled, failed or crashed run, reusing its trials"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        """Adds the run directory argument.

        Args:
            parser: The argument parser of this command.
        """
        parser.add_argument("run_directory")

    def execute(self, args: argparse.Namespace) -> int:
        """Resumes the run and logs the result summary and run directory.

        Args:
            args: The parsed command-line arguments.

        Returns:
            The exit status, EXIT_OK (0).

        Raises:
            ArtifactError: If the run directory, or a file the resume reads
                from it, cannot be found.
        """
        try:
            result = project.Project.resume(args.run_directory)
        except FileNotFoundError as error:
            raise errors.ArtifactError(str(error)) from error
        LOGGER.info("%s", result.summary())
        LOGGER.info("artifacts: %s", result.run_dir)
        return command.EXIT_OK
