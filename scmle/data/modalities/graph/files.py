"""Reading graph edges from CSV and JSONL records."""

from collections.abc import Iterable
from collections.abc import Mapping
from collections.abc import Sequence
import json
from pathlib import Path
from typing import Any

import pandas

from scmle.data.modalities.graph.elements import Edge
from scmle.data.modalities.graph.elements import normalize
from scmle.data.modalities.graph.elements import ParseError
from scmle.data.modalities.graph.elements import TemporalError


def lines(input_path: Path) -> Sequence[Mapping[str, Any]]:
    """Reads the objects of a JSONL file, skipping blank lines.

    Args:
        input_path: The JSONL file.

    Returns:
        One mapping per non-blank line, in file order.

    Raises:
        ParseError: If the file cannot be read, a line is not valid JSON, or
            a line is not an object.
    """
    records: list[dict[str, Any]] = []
    try:
        with input_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                content = line.strip()
                if not content:
                    continue
                try:
                    record = json.loads(content)
                except json.JSONDecodeError as error:
                    raise ParseError(
                        f"malformed JSONL at line {line_number}: {error.msg}"
                    ) from error
                if not isinstance(record, dict):
                    raise ParseError(
                        f"JSONL line {line_number} must be an object"
                    )
                records.append(record)
    except OSError as error:
        raise ParseError(f"failed to read JSONL edges: {error}") from error
    return records


def edgelist(
    path: str | Path,
    source_column: str = "source",
    target_column: str = "target",
    weight_column: str = "weight",
    relation_column: str = "relation",
    time_column: str | None = None,
) -> tuple[Edge, ...]:
    """Reads CSV or JSONL edge records with bounded schema handling.

    The format is chosen from the file name: ``.csv``, ``.jsonl`` or
    ``.ndjson`` (case-insensitive). Blank JSONL lines are skipped.

    Args:
        path: Edge file.
        source_column: Name of the source column or key.
        target_column: Name of the target column or key.
        weight_column: Name of the weight column or key; a missing weight
            defaults to 1.0.
        relation_column: Name of the relation column or key; a missing
            relation defaults to ``default``.
        time_column: Name of the event-time column or key, or None for
            untimed edges.

    Returns:
        The edges in file order.

    Raises:
        ParseError: If the file does not exist, has another extension,
            cannot be read or parsed, a JSONL line is not an object, or a
            record is invalid (see `link`).
    """
    input_path = Path(path)
    if not input_path.exists():
        raise ParseError(f"graph edge file not found: {input_path}")
    suffix = input_path.name.lower()
    if suffix.endswith(".csv"):
        try:
            # Endpoints are read as written: `007` stays `007` and `NA` or
            # `null` stay node ids, so CSV and JSONL give one graph (MM-7).
            frame = pandas.read_csv(
                input_path,
                converters={source_column: str, target_column: str},
            )
        except (
            OSError,
            pandas.errors.ParserError,
            UnicodeDecodeError,
        ) as error:
            raise ParseError(f"failed to read CSV edges: {error}") from error
        csv_records = [
            {str(key): value for key, value in record.items()}
            for record in frame.to_dict(orient="records")
        ]
        return link(
            csv_records,
            source_column=source_column,
            target_column=target_column,
            weight_column=weight_column,
            relation_column=relation_column,
            time_column=time_column,
        )
    if suffix.endswith((".jsonl", ".ndjson")):
        records = lines(input_path)
        return link(
            records,
            source_column=source_column,
            target_column=target_column,
            weight_column=weight_column,
            relation_column=relation_column,
            time_column=time_column,
        )
    raise ParseError("graph edges must use a .csv, .jsonl, or .ndjson file")


def link(
    records: Iterable[Mapping[str, Any]],
    source_column: str = "source",
    target_column: str = "target",
    weight_column: str = "weight",
    relation_column: str = "relation",
    time_column: str | None = None,
) -> tuple[Edge, ...]:
    """Converts mappings into immutable edges with deterministic errors.

    Args:
        records: Mappings holding one edge each.
        source_column: Key of the source identifier.
        target_column: Key of the target identifier.
        weight_column: Key of the weight; a missing key defaults to 1.0.
        relation_column: Key of the relation; a missing or null value
            becomes ``default``.
        time_column: Key of the event time, or None for untimed edges. A
            missing or null value leaves the edge untimed.

    Returns:
        The edges in record order; identifiers are converted to text.

    Raises:
        ParseError: If a record lacks the source or target key or has a
            null endpoint, a null or non-numeric weight, an invalid event
            time, or values that `Edge` rejects. The message gives the
            1-based record number.
    """
    edges: list[Edge] = []
    for index, record in enumerate(records, start=1):
        if source_column not in record or target_column not in record:
            raise ParseError(
                f"edge record {index} must contain '{source_column}' and "
                f"'{target_column}'"
            )
        source = record[source_column]
        target = record[target_column]
        if any(
            pandas.isna(value) or not str(value).strip()
            for value in (source, target)
        ):
            raise ParseError(f"edge record {index} has a missing endpoint")
        weight_value = record.get(weight_column, 1.0)
        if pandas.isna(weight_value):
            raise ParseError(f"edge record {index} has an invalid weight")
        try:
            weight = float(weight_value)
        except (TypeError, ValueError) as error:
            raise ParseError(
                f"edge record {index} has a non-numeric weight"
            ) from error
        relation_value = record.get(relation_column, "default")
        if pandas.isna(relation_value):
            relation_value = "default"
        event_time = None
        time_value = None if time_column is None else record.get(time_column)
        if time_value is not None and not pandas.isna(time_value):
            try:
                event_time = normalize(time_value)
            except TemporalError as error:
                raise ParseError(
                    f"edge record {index} has an invalid event time"
                ) from error
        try:
            edges.append(
                Edge(
                    source=str(source),
                    target=str(target),
                    weight=weight,
                    relation=str(relation_value),
                    time=event_time,
                )
            )
        except (TypeError, ValueError) as error:
            raise ParseError(f"invalid edge record {index}: {error}") from error
    return tuple(edges)
