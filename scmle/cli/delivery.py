"""Commands that use a trained bundle: predict, batch, serve, export."""

import argparse
import logging
import os
import pathlib

import pandas

from scmle import durability
from scmle.artifacts import bundle
from scmle.cli import command
from scmle.deploy import packaging
from scmle.inference import onnx_export
import scmle.inference.mapping
from scmle.operations import signals
from scmle.security import access
from scmle.serving import runtime
from scmle.serving import server

LOGGER = logging.getLogger(__name__)


class Prediction(command.Command):
    """Scores a dataset with a saved bundle."""

    name = "predict"
    help = "predict with a saved model bundle (trusted bundles only)"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        """Adds bundle, data and output options.

        Args:
            parser: The argument parser of this command.
        """
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
        """Writes a predictions CSV to `--output`.

        The file has a `prediction` column and, when the bundle has class
        labels, one `proba_<label>` column per class.

        Args:
            args: The parsed command-line arguments.

        Returns:
            The exit status, EXIT_OK (0).
        """
        model = bundle.Bundle.load(args.bundle, trusted=args.trusted)
        mapping = scmle.inference.mapping.Mapper(
            dict(pair.split("=", 1) for pair in args.rename),
            args.case_insensitive,
            args.strip_whitespace,
        )
        frame = pandas.DataFrame(
            {"prediction": model.predict(args.data, mapping)}
        )
        if model.classes is not None:
            proba = model.predict_proba(args.data, mapping)
            for index, label in enumerate(model.classes):
                frame[f"proba_{label}"] = proba[:, index]
        durability.write(
            pathlib.Path(args.output), frame.to_csv(index=False).encode("utf-8")
        )
        LOGGER.info("wrote %d predictions to %s", len(frame), args.output)
        return command.EXIT_OK


class Batch(command.Command):
    """Scores a file through the same path as the online service."""

    name = "batch"
    help = (
        "score a file with the online serving path, streamed in chunks, and "
        "write a manifest with the model version and artifact digest; an "
        "interrupted run resumes"
    )

    def configure(self, parser: argparse.ArgumentParser) -> None:
        """Adds bundle, data, output and policy options.

        Args:
            parser: The argument parser of this command.
        """
        parser.add_argument("bundle", help="bundle directory")
        parser.add_argument("data", help="CSV or Parquet file")
        parser.add_argument("--output", default="predictions.csv")
        parser.add_argument(
            "--chunk-rows",
            type=int,
            default=server.BATCH_CHUNK_ROWS,
            help="rows read and scored at a time (bounds the memory used)",
        )
        parser.add_argument(
            "--restart",
            action="store_true",
            help="score from the first row even if an interrupted run of the "
            "same input, bundle and policies left a journal",
        )
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
        """Writes predictions and `<output>.manifest.json`.

        The input is read `--chunk-rows` rows at a time. An interrupted run
        leaves `<output>.partial` and `<output>.progress.json`; running the
        same command again continues after the last scored chunk.

        Args:
            args: The parsed command-line arguments.

        Returns:
            The exit status, EXIT_OK (0).
        """
        model = bundle.Bundle.load(args.bundle, trusted=args.trusted)
        adapter = server.Batcher(
            model, on_null=args.on_null, on_unseen=args.on_unseen
        )
        rows = adapter.score(
            args.data, args.output, args.chunk_rows, resume=not args.restart
        )
        LOGGER.info("wrote %d predictions to %s", rows, args.output)
        return command.EXIT_OK


class Serve(command.Command):
    """Serves a saved bundle over HTTP.

    The serving loop is the packaged service's (`runtime.hold`): SIGTERM or
    SIGINT drains the requests in flight and exits, SIGHUP reloads the
    bundle directory without downtime, and the operations signals over the
    last `--signals-window` seconds are written every `--signals-interval`
    seconds, drift judged against the served bundle's training reference.
    """

    name = "serve"
    help = (
        "serve a saved model bundle over HTTP (trusted bundles only); "
        "SIGHUP reloads it, SIGTERM drains and stops"
    )

    def configure(self, parser: argparse.ArgumentParser) -> None:
        """Adds bundle, binding, drain and signals options.

        Args:
            parser: The argument parser of this command.
        """
        parser.add_argument("bundle", help="bundle directory")
        parser.add_argument("--host", default="127.0.0.1")
        parser.add_argument("--port", type=int, default=8000)
        parser.add_argument(
            "--trusted",
            action="store_true",
            help="load an unsigned bundle you vouch for (pickle executes code)",
        )
        parser.add_argument(
            "--drain-seconds",
            type=float,
            help="time requests in flight get to finish on SIGTERM (default: "
            "the longest an admitted request can take)",
        )
        parser.add_argument(
            "--signals-file",
            help="append the evaluated operations signals here as JSON lines "
            "(default: the log)",
        )
        parser.add_argument(
            "--signals-interval",
            type=float,
            default=runtime.INTERVAL,
            help="seconds between two evaluations of the signals",
        )
        parser.add_argument(
            "--signals-window",
            type=float,
            default=signals.Window().window,
            help="seconds of traffic each evaluation of the signals covers",
        )

    def execute(self, args: argparse.Namespace) -> int:
        """Loads the bundle and serves until SIGTERM or SIGINT.

        Args:
            args: The parsed command-line arguments.

        Returns:
            The exit status, EXIT_OK (0), once serving has drained and
            stopped.
        """
        path = pathlib.Path(args.bundle)
        model = bundle.Bundle.load(path, trusted=args.trusted)
        monitor = signals.Monitor(signals.Window(window=args.signals_window))
        adapter = server.Service(
            model,
            args.host,
            args.port,
            gateway=access.assemble(os.environ),
            monitor=monitor,
            drain=args.drain_seconds,
        )
        LOGGER.info("serving %s", model.info.model)
        runtime.hold(
            adapter,
            monitor,
            args.signals_file,
            args.signals_interval,
            runtime.fetch(path, args.trusted),
        )
        return command.EXIT_OK


class Export(command.Command):
    """Exports a saved bundle's pipeline as an ONNX graph."""

    name = "export-onnx"
    help = "export a saved model bundle as ONNX (needs the onnx extra)"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        """Adds bundle, output, check, tolerance and skip options.

        Args:
            parser: The argument parser of this command.
        """
        parser.add_argument("bundle", help="bundle directory")
        parser.add_argument("--output", default="model.onnx")
        parser.add_argument(
            "--check",
            help=(
                "CSV or Parquet rows held out from training; the graph is "
                f"exported only when it matches the pipeline on at least "
                f"{onnx_export.ROWS} of them"
            ),
        )
        parser.add_argument(
            "--tolerance",
            type=float,
            default=onnx_export.DEFAULT_TOLERANCE,
            help="largest allowed probability / scaled value difference",
        )
        parser.add_argument(
            "--unverified",
            action="store_true",
            help="skip the check (recorded in the export manifest)",
        )
        parser.add_argument("--trusted", action="store_true")

    def execute(self, args: argparse.Namespace) -> int:
        """Exports the ONNX graph, verified on held-out rows by default.

        Args:
            args: The parsed command-line arguments.

        Returns:
            The exit status, EXIT_OK (0).
        """
        model = bundle.Bundle.load(args.bundle, trusted=args.trusted)
        check = None if args.check is None else model.read(args.check)
        written = model.export(
            args.output,
            check=check,
            tolerance=args.tolerance,
            unverified=args.unverified,
        )
        LOGGER.info("wrote %s", written)
        return command.EXIT_OK


class Packaging(command.Command):
    """Writes a deployable service folder for a saved bundle."""

    name = "package"
    help = "package a saved model bundle as a ready-to-run service folder"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        """Adds bundle and destination options.

        Args:
            parser: The argument parser of this command.
        """
        parser.add_argument("bundle", help="bundle directory")
        parser.add_argument("--output", default="service")
        parser.add_argument("--trusted", action="store_true")

    def execute(self, args: argparse.Namespace) -> int:
        """Writes the deployable service folder for the bundle.

        Args:
            args: The parsed command-line arguments.

        Returns:
            The exit status, EXIT_OK (0).
        """
        model = bundle.Bundle.load(args.bundle, trusted=args.trusted)
        folder = packaging.scaffold(model, args.output)
        LOGGER.info("wrote service folder %s", folder)
        return command.EXIT_OK
