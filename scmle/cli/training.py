"""Commands that build a project from a data file: plan, run and resume."""

import argparse
from collections.abc import Mapping
import logging
import pathlib
from typing import Any

from scmle.api import estimation
from scmle.api import project
from scmle.cli import command
from scmle.contracts import errors
from scmle.contracts import tasks
from scmle.evaluation import parameter_estimation
from scmle.optimization import optimizers
from scmle.pipeline import kinds
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
    command.option(
        parser,
        "--sensitive",
        "--sensitive-columns",
        nargs="*",
        default=[],
        help="columns whose groups the fairness report compares",
    )
    parser.add_argument(
        "--mitigation",
        choices=["none", "reweighing", "group_thresholds"],
        default="none",
        help="close the gap of the first sensitive column: `reweighing` "
        "(binary, multiclass, regression) or `group_thresholds` (binary)",
    )
    command.option(
        parser,
        "--delay",
        "--feature-delay",
        nargs="*",
        default=[],
        metavar="COLUMN=ROWS",
        help="columns known only ROWS rows after the row they describe",
    )
    command.option(
        parser,
        "--embargo",
        "--embargo-rows",
        type=int,
        default=0,
        help="rows dropped between training and validation windows",
    )
    command.option(
        parser,
        "--unavailable",
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
    command.option(
        parser,
        "--leakage",
        "--leakage-policy",
        choices=["warn", "block"],
        default="warn",
    )
    parser.add_argument(
        "--charter",
        help="analysis charter (see `scmle charter init`): splits off a "
        "sealed confirmation set the exploration stages cannot read",
    )
    command.option(
        parser,
        "--attestations",
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
    parser.add_argument(
        "--estimation",
        help="JSON file declaring a parameter estimate (`estimand` and "
        "`policy`), made beside model selection and kept in the run",
    )


def roles(parser: argparse.ArgumentParser) -> None:
    """Adds the options naming the columns that play a role in the data.

    Args:
        parser: The argument parser of the command.
    """
    command.option(parser, "--group", "--group-column")
    command.option(parser, "--weight", "--weight-column")
    command.option(parser, "--event", "--event-column")
    command.option(parser, "--query", "--query-column")
    command.option(parser, "--treatment", "--treatment-column")
    command.option(parser, "--baseline", "--baseline-column")
    command.option(
        parser,
        "--comparator",
        "--baseline-subject",
        choices=("frozen_rule", "selected_model"),
        default="frozen_rule",
    )
    command.option(parser, "--draft", "--framing-draft")
    command.option(parser, "--propensity", "--propensity-column")
    command.option(parser, "--action", "--logged-action-column")
    command.option(parser, "--ingest", "--ingest-plan")
    command.option(parser, "--time", "--time-column")
    command.option(
        parser, "--temporal", "--time-as-feature", action="store_true"
    )
    command.option(
        parser, "--identifiers", "--id-columns", nargs="*", default=[]
    )


def declarations(args: argparse.Namespace) -> Mapping[str, Any]:
    """Reads the fairness mitigation and the `--estimation` file.

    Args:
        args: The parsed command-line arguments.

    Returns:
        `fairness` when a mitigation is requested, and `estimand` and
        `estimation_policy` when a declaration file is given.
    """
    fairness = (
        {}
        if args.mitigation == "none"
        else {"fairness": tasks.Parity(mitigation=args.mitigation)}
    )
    if args.estimation is None:
        return fairness
    estimand, policy = estimation.load(args.estimation)
    return {**fairness, "estimand": estimand, "estimation_policy": policy}


class ProjectCommand(command.Command):
    """Shared arguments for commands that build an `Project`."""

    def configure(self, parser: argparse.ArgumentParser) -> None:
        """Adds data and project options.

        Args:
            parser: The argument parser of this command.
        """
        parser.add_argument(
            "data",
            help="CSV or Parquet file, or the source of the declared "
            "modality (a manifest, a directory of label folders, a graph "
            "directory or a vector file)",
        )
        parser.add_argument("--target", required=True, nargs="+")
        parser.add_argument(
            "--modality",
            choices=sorted(kinds.KINDS),
            default="tabular",
            help="the kind of data the source holds; every kind but "
            "tabular is experimental and needs `models.experimental_models` "
            "in the --config file",
        )
        parser.add_argument("--task", default="auto")
        parser.add_argument("--metric")
        command.option(
            parser,
            "--budget",
            "--time-budget",
            type=int,
            default=1800,
            help="time budget in seconds",
        )
        parser.add_argument("--workers", type=int, default=1)
        command.option(
            parser,
            "--memory",
            "--memory-budget-gb",
            type=float,
            help="largest in-memory size of the data in GB",
        )
        parser.add_argument(
            "--training", choices=["sample", "stream"], default="sample"
        )
        parser.add_argument("--horizon", type=int)
        command.option(parser, "--period", "--seasonal-period", type=int)
        command.option(parser, "--order", "--ordinal-order", nargs="+")
        parser.add_argument(
            "--unlabeled",
            choices=["block", "ignore", "pseudo_label"],
            default="block",
        )
        roles(parser)
        parser.add_argument("--exclude", nargs="*", default=[])
        parser.add_argument("--seed", type=int, default=42)
        command.option(
            parser,
            "--leaders",
            "--top-models",
            type=int,
            default=2,
            help="models to hyperparameter-tune (0 disables search)",
        )
        command.option(
            parser,
            "--configurations",
            "--configs-per-model",
            type=int,
            default=6,
        )
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
        sections = {**declarations(args), **config.overlay(args.config)}
        sections.setdefault(
            "search",
            tasks.Search(
                top_models=args.leaders,
                configs_per_model=args.configurations,
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
            modality=args.modality,
            task=args.task,
            metric=args.metric,
            budget=args.budget,
            workers=args.workers,
            identifiers=args.identifiers,
            group=args.group,
            weight=args.weight,
            event=args.event,
            query=args.query,
            treatment=args.treatment,
            memory_budget_gb=args.memory,
            training=args.training,
            horizon=args.horizon,
            period=args.period,
            order=args.order,
            unlabeled=args.unlabeled,
            timestamp=args.time,
            time_as_feature=args.temporal,
            excluded=args.exclude,
            seed=args.seed,
            validation=args.validation,
            folds=args.folds,
            holdout=args.test,
            rows=args.embargo,
            sensitive=args.sensitive,
            constraints=parse(args.monotonic),
            span=args.rolling_horizon,
            window=args.rolling_window,
            delays=parse(args.delay),
            unavailable=args.unavailable,
            enforcement=args.enforcement,
            leakage=args.leakage,
            charter=args.charter,
            attestations=args.attestations,
            baseline=args.baseline,
            subject=args.comparator,
            draft=args.draft,
            propensity=args.propensity,
            action=args.action,
            ingest=args.ingest,
            directory=args.output_dir,
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
        if result.estimation is not None:
            announce(result.estimation)
        LOGGER.info("artifacts: %s", result.directory)
        return command.EXIT_OK


def announce(found: parameter_estimation.Estimation) -> None:
    """Logs each estimated parameter with its interval.

    Args:
        found: The verified estimation report.
    """
    level = found.policy.confidence
    for item in found.parameters:
        LOGGER.info(
            "%s %s = %.6g, %g%% interval [%.6g, %.6g], p = %.3g",
            found.estimand.name,
            item.name,
            item.estimate,
            100 * level,
            item.interval[0],
            item.interval[1],
            item.p_value,
        )


class Estimation(command.Command):
    """Estimates a declared parameter without training a model."""

    name = "estimate"
    help = "estimate a declared parameter with its interval, or say why not"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        """Adds the data, declaration and output options.

        Args:
            parser: The argument parser of this command.
        """
        parser.add_argument("data", help="CSV or Parquet file")
        parser.add_argument(
            "--estimation",
            required=True,
            help="JSON file declaring the parameter (`estimand` and `policy`)",
        )
        command.option(parser, "--treatment", "--treatment-column")
        parser.add_argument(
            "--output-dir",
            default="estimates",
            help="where parameter_estimation.json, or "
            "parameter_estimation_refusal.json with why, is written",
        )

    def execute(self, args: argparse.Namespace) -> int:
        """Writes the estimate and logs each parameter with its interval.

        Args:
            args: The parsed command-line arguments.

        Returns:
            The exit status, EXIT_OK (0).

        Raises:
            BlockedError: If the estimate is refused; the reasons are in
                `parameter_estimation_refusal.json` in `--output-dir`.
        """
        estimand, policy = estimation.load(args.estimation)
        declared = project.Project(
            args.data,
            estimand.outcome,
            treatment=args.treatment,
            estimand=estimand,
            policy=policy,
            directory=args.output_dir,
        )
        folder = pathlib.Path(args.output_dir)
        try:
            found = declared.estimate(folder)
        except estimation.REFUSALS as error:
            raise errors.BlockedError(
                f"parameter estimation refused: {error}",
                {"record": str(folder / estimation.REFUSAL)},
            ) from error
        announce(found)
        LOGGER.info("estimate: %s", folder / estimation.REPORT)
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
        LOGGER.info("artifacts: %s", result.directory)
        return command.EXIT_OK
