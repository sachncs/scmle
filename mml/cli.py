"""Command-line interface.

Each sub-command is a `Command` subclass that registers its own arguments and
executes itself, so adding a command never touches the dispatcher.
"""

import abc
import argparse
from collections.abc import Sequence
import pathlib
import sys
from typing import ClassVar

import pandas as pd

from mml.api import project
from mml.artifacts import bundle as bundle_module
from mml.contracts import errors
from mml.contracts import tasks
from mml.optimization import optimizers

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
        parser.add_argument("--target", required=True)
        parser.add_argument("--task", default="auto")
        parser.add_argument("--metric")
        parser.add_argument("--time-budget", type=int, default=1800)
        parser.add_argument("--group-column")
        parser.add_argument("--time-column")
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

    def build_project(self, args: argparse.Namespace) -> project.MLProject:
        """Creates the project described by the arguments."""
        return project.MLProject(
            args.data,
            args.target,
            task=args.task,
            metric=args.metric,
            time_budget=args.time_budget,
            id_columns=args.id_columns,
            group_column=args.group_column,
            time_column=args.time_column,
            excluded_columns=args.exclude,
            seed=args.seed,
            search=tasks.SearchSpec(
                top_models=args.top_models,
                configs_per_model=args.configs_per_model,
                optimizer=args.optimizer,
            ),
            output_dir=args.output_dir,
        )


class PlanCommand(ProjectCommand):
    """Prints the proposed plan without training."""

    name = "plan"
    help = "analyze the data and print the proposed plan"

    def execute(self, args: argparse.Namespace) -> int:
        """Prints the plan summary."""
        print(self.build_project(args).plan().summary())
        return EXIT_OK


class RunCommand(ProjectCommand):
    """Trains, selects, tests and packages."""

    name = "run"
    help = "execute the full workflow"

    def execute(self, args: argparse.Namespace) -> int:
        """Runs the project and prints the result summary."""
        result = self.build_project(args).run()
        print(result.summary())
        print(f"artifacts: {result.run_dir}")
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

    def execute(self, args: argparse.Namespace) -> int:
        """Writes predictions (and probabilities for classifiers)."""
        model = bundle_module.ModelBundle.load(args.bundle)
        frame = pd.DataFrame({"prediction": model.predict(args.data)})
        if model.classes is not None:
            proba = model.predict_proba(args.data)
            for index, label in enumerate(model.classes):
                frame[f"proba_{label}"] = proba[:, index]
        frame.to_csv(pathlib.Path(args.output), index=False)
        print(f"wrote {len(frame)} predictions to {args.output}")
        return EXIT_OK


COMMANDS: tuple[Command, ...] = (PlanCommand(), RunCommand(), PredictCommand())


def build_parser() -> argparse.ArgumentParser:
    """Builds the argument parser from the registered commands."""
    parser = argparse.ArgumentParser(prog="magicml")
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
    command = next(c for c in COMMANDS if c.name == args.command)
    try:
        return command.execute(args)
    except errors.BlockedError as error:
        print(f"blocked: {error}", file=sys.stderr)
        return EXIT_BLOCKED
    except errors.MagicMLError as error:
        print(f"{error.category} error: {error}", file=sys.stderr)
        return EXIT_FAILED


if __name__ == "__main__":
    sys.exit(main())
