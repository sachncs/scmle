"""The `cluster` command: groups in unlabeled rows, saved as a bundle."""

import argparse
import logging

from scmle.api import clustering
from scmle.cli import command

LOGGER = logging.getLogger(__name__)


class Clustering(command.Command):
    """Finds groups in unlabeled rows and saves a bundle that assigns them."""

    name = "cluster"
    help = (
        "find groups in unlabeled rows and save a bundle that predicts "
        "cluster labels"
    )

    def configure(self, parser: argparse.ArgumentParser) -> None:
        """Adds data, output and search options.

        Args:
            parser: The argument parser of this command.
        """
        parser.add_argument("data", help="CSV or Parquet file")
        parser.add_argument(
            "--output", required=True, help="new bundle directory"
        )
        parser.add_argument(
            "--counts",
            type=int,
            nargs="+",
            default=list(range(2, 9)),
            metavar="K",
            help="cluster counts to try (each at least 2)",
        )
        parser.add_argument(
            "--methods",
            nargs="+",
            choices=clustering.METHODS,
            default=list(clustering.METHODS),
        )
        parser.add_argument(
            "--rounds",
            type=int,
            default=10,
            help="bootstrap refits per candidate for the stability score",
        )
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument(
            "--exclude",
            action="append",
            default=[],
            metavar="COLUMN",
            help="a numeric column that is not a feature (repeatable)",
        )
        parser.add_argument(
            "--truth",
            metavar="COLUMN",
            help="column of known groups to measure agreement against",
        )

    def execute(self, args: argparse.Namespace) -> int:
        """Writes the bundle and its `clustering.json` report to `--output`.

        Args:
            args: The parsed command-line arguments.

        Returns:
            The exit status, EXIT_OK (0).
        """
        result = clustering.cluster(
            args.data,
            tuple(args.counts),
            tuple(args.methods),
            args.rounds,
            args.seed,
            tuple(args.exclude),
            args.truth,
        )
        folder = result.save(args.output)
        best = result.report.best
        LOGGER.info(
            "%s with %d clusters (silhouette %.3f, stability %.3f, %s); "
            "bundle and %s written to %s",
            best.method,
            best.clusters,
            best.silhouette,
            best.stability,
            "stable" if result.report.stable else "NOT stable",
            clustering.REPORT_FILE,
            folder,
        )
        return command.EXIT_OK
