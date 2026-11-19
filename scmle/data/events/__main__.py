"""Command line: event-log features at prediction cutoffs.

    python -m scmle.data.events features EVENTS... --queries FILE
        --window SPEC [--window SPEC ...] [--lateness 1h|auto]
        [--policy drop|accept|reject] [--output FILE]
    python -m scmle.data.events follow EVENTS.jsonl --queries FILE
        --window SPEC ... [--lateness 1h|auto] [--poll 1] [--idle 30]
    python -m scmle.data.events lateness EVENTS... [--coverage 0.99]
        [--confidence 0.95]

`features` reads whole event files (`-` is JSON lines on standard input,
such as another program's output piped in) and writes one row per query (CSV on
standard output, or CSV, Parquet or JSON lines by the `--output` suffix).
`follow` reads a JSON-lines file as another process appends to it and
prints each query as one JSON line the moment its cutoff passes the
watermark; when no event has arrived for `--idle` seconds it prints the
queries still open, marked `"final": false`, and stops. Under
`--policy accept` no answer is ever final, since an event of any age may
still revise it, so `follow` answers every query only once the log goes
idle, each marked `"final": false`. `lateness`
estimates a lateness bound from the files' delays. `--lateness auto` uses
that estimate, made from the events read before the first query is
answered. Each command writes a JSON summary to standard error and exits
2 on an input or data error.
"""

import argparse
from collections.abc import Iterable
from collections.abc import Mapping
from collections.abc import Sequence
import json
import math
import pathlib
import sys
import time

import pandas

from scmle.contracts import errors
from scmle.data.events import event
from scmle.data.events import lateness
from scmle.data.events import sources
from scmle.data.events import store
from scmle.data.events import windows
from scmle.data.readers import tabular
from scmle.data.readers.incremental.timeline import Lateness

# Exit codes, kept equal to `scmle.cli.command.EXIT_OK` and `EXIT_BLOCKED`
# (the source of truth). The layer contract forbids `scmle.data` importing
# `scmle.cli`, so the values are restated here and a test pins them equal.
EXIT_OK = 0
EXIT_INPUT = 2


def parser() -> argparse.ArgumentParser:
    """The command line's parser.

    Returns:
        A parser with the `features`, `follow` and `lateness` actions.
    """
    found = argparse.ArgumentParser(prog="python -m scmle.data.events")
    actions = found.add_subparsers(dest="action", required=True)
    for name, text in (
        ("features", "windowed features and snapshots at query cutoffs"),
        ("follow", "answer queries as a growing JSON-lines log passes them"),
        ("lateness", "estimate a lateness bound from observed delays"),
    ):
        action = actions.add_parser(name, help=text, description=text)
        action.add_argument(
            "events", nargs=1 if name == "follow" else "+", type=pathlib.Path
        )
        for flag, default in (
            ("--entity", "entity"),
            ("--time", "time"),
            ("--type", "type"),
            ("--payload", "payload"),
        ):
            action.add_argument(
                flag, default=default, help=f"event field (default {default})"
            )
        action.add_argument(
            "--lenient",
            action="store_true",
            help="skip malformed records instead of stopping",
        )
        if name == "lateness":
            action.add_argument(
                "--coverage", type=float, default=lateness.COVERAGE
            )
            action.add_argument(
                "--confidence", type=float, default=lateness.CONFIDENCE
            )
            continue
        action.add_argument("--queries", type=pathlib.Path, required=True)
        action.add_argument(
            "--cutoff", default="cutoff", help="query column of the cutoff"
        )
        action.add_argument(
            "--window",
            action="append",
            required=True,
            help="KIND[.FIELD]:STATISTIC:SPAN, as in purchase.amount:total:7d",
        )
        action.add_argument("--lateness", default="0s")
        action.add_argument(
            "--policy", choices=[p.value for p in Lateness], default="drop"
        )
        if name == "features":
            action.add_argument("--output", type=pathlib.Path)
        else:
            action.add_argument("--poll", type=float, default=1.0)
            action.add_argument("--idle", type=float, default=30.0)
    return found


def emit(text: str, *, error: bool = False) -> None:
    """Writes one line and flushes it, so a reader of a pipe sees it now.

    Args:
        text: The line, without its newline.
        error: Whether it goes to standard error instead of standard output.
    """
    stream = sys.stderr if error else sys.stdout
    stream.write(text + "\n")
    stream.flush()


def plain(value: object) -> object:
    """Makes a feature value JSON-ready: NaN becomes null.

    Args:
        value: A feature or state value.

    Returns:
        The value, with NaN as None and numpy scalars as Python ones.
    """
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def bounded(
    options: argparse.Namespace, backlog: Sequence[event.Event]
) -> store.Store:
    """Builds the store the options describe and fills it with a backlog.

    Args:
        options: The parsed command line.
        backlog: The events read before any query is answered.

    Returns:
        The store holding the backlog.
    """
    spans = [windows.Window.parse(text) for text in options.window]
    if options.lateness == "auto":
        bound = lateness.bound([item.time for item in backlog])
    else:
        bound = event.span(options.lateness)
    found = store.Store(spans, bound, Lateness(options.policy))
    found.extend(backlog)
    return found


def summary(found: store.Store, skipped: int) -> Mapping[str, object]:
    """Describes what a store took in.

    Args:
        found: The store.
        skipped: Malformed records skipped.

    Returns:
        The arrival counts, lateness bound, watermark and skipped count.
    """
    watermark = found.watermark
    return {
        "arrivals": {a.value: found.tally[a] for a in store.Arrival},
        "lateness_seconds": found.lateness / windows.NANOSECONDS,
        "watermark": (
            None
            if watermark is None
            else pandas.Timestamp(watermark, tz="UTC").isoformat()
        ),
        "skipped": skipped,
    }


def write(frame: pandas.DataFrame, path: pathlib.Path | None) -> None:
    """Writes the answered queries by the output's suffix.

    Args:
        frame: The answers.
        path: CSV, Parquet or JSON-lines file; None writes CSV to stdout.

    Raises:
        InputError: If the suffix is none of those.
    """
    if path is None:
        frame.to_csv(sys.stdout, index=False)
    elif path.suffix == ".csv":
        frame.to_csv(path, index=False)
    elif path.suffix == ".parquet":
        frame.to_parquet(path, index=False)
    elif path.suffix in {".jsonl", ".ndjson"}:
        frame.to_json(path, orient="records", lines=True, date_format="iso")
    else:
        raise errors.InputError(
            f"--output must end in .csv, .parquet or .jsonl: {path}"
        )


def supply(
    path: pathlib.Path, columns: event.Columns, *, strict: bool
) -> sources.Source:
    """The source a command-line path names.

    Args:
        path: An event file, or `-` for JSON lines on standard input.
        columns: Where each record keeps each part of an event.
        strict: Whether a malformed record raises.

    Returns:
        A `Lines` source over standard input for `-`, else a `File`.
    """
    if str(path) == "-":
        return sources.Lines(sys.stdin, columns, strict=strict)
    return sources.File(path, columns, strict=strict)


def features(options: argparse.Namespace, columns: event.Columns) -> int:
    """Runs `features`.

    Args:
        options: The parsed command line.
        columns: Where each record keeps each part of an event.

    Returns:
        The exit code.
    """
    files = [
        supply(path, columns, strict=not options.lenient)
        for path in options.events
    ]
    backlog = [item for source in files for item in source.events()]
    found = bounded(options, backlog)
    queries = tabular.read_table(options.queries)
    write(found.frame(queries, options.entity, options.cutoff), options.output)
    report(found, sum(len(source.skipped) for source in files))
    return EXIT_OK


def answer(found: store.Store, entity: str, cutoff: object) -> str:
    """One query's answer as a JSON line.

    Args:
        found: The store.
        entity: The entity.
        cutoff: The cutoff as the queries file holds it.

    Returns:
        The JSON object text.
    """
    snapshot = found.snapshot(entity, event.instant(cutoff))
    return json.dumps(
        {
            "entity": entity,
            "cutoff": pandas.Timestamp(snapshot.cutoff, tz="UTC").isoformat(),
            **{k: plain(v) for k, v in snapshot.features.items()},
            **{k: plain(v) for k, v in snapshot.state.items()},
            "final": snapshot.final,
        }
    )


def follow(options: argparse.Namespace, columns: event.Columns) -> int:
    """Runs `follow`.

    Args:
        options: The parsed command line.
        columns: Where each record keeps each part of an event.

    Returns:
        The exit code.
    """
    source = sources.Tail(
        options.events[0], columns, strict=not options.lenient
    )
    found = bounded(options, list(source.events()))
    queries = tabular.read_table(options.queries)
    for column in (options.entity, options.cutoff):
        if column not in queries.columns:
            raise event.EventError(f"the queries need a {column!r} column")
    pending = sorted(
        (
            (event.instant(when), event.identity(who), when)
            for who, when in zip(
                queries[options.entity].tolist(),
                queries[options.cutoff].tolist(),
                strict=True,
            )
        ),
        key=lambda query: query[0],
    )
    quiet = time.monotonic()
    while pending:
        while pending and found.final(pending[0][0]):
            emit(answer(found, pending[0][1], pending[0][2]))
            pending.pop(0)
        if not pending or time.monotonic() - quiet > options.idle:
            break
        time.sleep(options.poll)
        if sum(found.extend(source.events()).values()):
            quiet = time.monotonic()
    for query in pending:
        emit(answer(found, query[1], query[2]))
    report(found, len(source.skipped))
    return EXIT_OK


def estimate(options: argparse.Namespace, columns: event.Columns) -> int:
    """Runs `lateness`.

    Args:
        options: The parsed command line.
        columns: Where each record keeps each part of an event.

    Returns:
        The exit code.
    """
    times: list[int] = []
    skipped = 0
    for path in options.events:
        source = supply(path, columns, strict=not options.lenient)
        times.extend(item.time for item in source.events())
        skipped += len(source.skipped)
    bound = lateness.bound(times, options.coverage, options.confidence)
    emit(
        json.dumps(
            {
                "lateness": str(pandas.Timedelta(bound)),
                "lateness_seconds": bound / windows.NANOSECONDS,
                "coverage": options.coverage,
                "confidence": options.confidence,
                "events": len(times),
                "skipped": skipped,
            }
        )
    )
    return EXIT_OK


def report(found: store.Store, skipped: int) -> None:
    """Writes a store's summary to standard error.

    Args:
        found: The store.
        skipped: Malformed records skipped.
    """
    emit(json.dumps(summary(found, skipped)), error=True)


def main(arguments: Iterable[str] | None = None) -> int:
    """Runs the command line.

    Args:
        arguments: The arguments; None reads `sys.argv`.

    Returns:
        The exit code.
    """
    options = parser().parse_args(
        None if arguments is None else list(arguments)
    )
    columns = event.Columns(
        options.entity, options.time, options.type, options.payload
    )
    actions = {"features": features, "follow": follow, "lateness": estimate}
    try:
        return actions[options.action](options, columns)
    except errors.DataError as error:
        emit(json.dumps({"error": str(error)}), error=True)
        return EXIT_INPUT


if __name__ == "__main__":
    sys.exit(main())
