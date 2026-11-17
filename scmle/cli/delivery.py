"""Commands that use a trained bundle: predict, batch, serve, export."""

import argparse
import logging
import os
import pathlib
import threading

import pandas

from scmle import durability
from scmle.artifacts import bundle
from scmle.cli import command
from scmle.deploy import packaging
import scmle.inference.mapping
from scmle.security import access
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
        durability.write_atomically(
            pathlib.Path(args.output), frame.to_csv(index=False).encode("utf-8")
        )
        LOGGER.info("wrote %d predictions to %s", len(frame), args.output)
        return command.EXIT_OK


class Batch(command.Command):
    """Scores a file through the same path as the online service."""

    name = "batch"
    help = (
        "score a file with the online serving path and write a manifest "
        "with the model version and artifact digest"
    )

    def configure(self, parser: argparse.ArgumentParser) -> None:
        """Adds bundle, data, output and policy options.

        Args:
            parser: The argument parser of this command.
        """
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
        """Writes predictions and `<output>.manifest.json`.

        Args:
            args: The parsed command-line arguments.

        Returns:
            The exit status, EXIT_OK (0).
        """
        model = bundle.Bundle.load(args.bundle, trusted=args.trusted)
        adapter = server.Batcher(
            model, on_null=args.on_null, on_unseen=args.on_unseen
        )
        rows = adapter.score_file(args.data, args.output, args.chunk_rows)
        LOGGER.info("wrote %d predictions to %s", rows, args.output)
        return command.EXIT_OK


class Serve(command.Command):
    """Serves a saved bundle over HTTP."""

    name = "serve"
    help = "serve a saved model bundle over HTTP (trusted bundles only)"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        """Adds bundle and binding options.

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

    def execute(self, args: argparse.Namespace) -> int:
        """Loads the bundle and serves until interrupted.

        Args:
            args: The parsed command-line arguments.

        Returns:
            The exit status, EXIT_OK (0), once serving has stopped.
        """
        model = bundle.Bundle.load(args.bundle, trusted=args.trusted)
        adapter = server.HttpServing(
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
        return command.EXIT_OK


class Export(command.Command):
    """Exports a saved bundle's pipeline as an ONNX graph."""

    name = "export-onnx"
    help = "export a saved model bundle as ONNX (needs the onnx extra)"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        """Adds bundle, output and parity-check options.

        Args:
            parser: The argument parser of this command.
        """
        parser.add_argument("bundle", help="bundle directory")
        parser.add_argument("--output", default="model.onnx")
        parser.add_argument(
            "--check", help="CSV or Parquet rows to compare graph and pipeline"
        )
        parser.add_argument("--trusted", action="store_true")

    def execute(self, args: argparse.Namespace) -> int:
        """Exports the ONNX graph, optionally comparing it on sample rows.

        Args:
            args: The parsed command-line arguments.

        Returns:
            The exit status, EXIT_OK (0).
        """
        model = bundle.Bundle.load(args.bundle, trusted=args.trusted)
        check = None if args.check is None else model.read_frame(args.check)
        written = model.export_onnx(args.output, check=check)
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
