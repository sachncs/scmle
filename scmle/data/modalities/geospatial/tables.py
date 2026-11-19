"""Vector files read as rows: one row per feature, properties and geometry.

`load` reads a GeoJSON FeatureCollection (`.geojson`, `.json`), a GeoJSON
text sequence (`.geojsonl`, `.geojsons`, `.ndjson`, `.jsonl`; RFC 8142's
record separators are accepted) or, when the optional `geopandas` library is
installed, any format it reads (GeoPackage, Shapefile, FlatGeobuf, ...).
`tabulate` turns the features into a table: each scalar property becomes a
column (a list or object property is kept as compact JSON text) and the
geometry becomes the `COLUMNS` measures, all in the declared CRS units.

Only the first two values of a position count, so an elevation or measure
never enters a centroid, an extent, an area or a length (MM-2).
"""

from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Mapping
import importlib
import json
import pathlib
from typing import Any

import numpy
import pandas

from scmle.contracts import capabilities
from scmle.contracts import errors
from scmle.data.modalities.geospatial.geometry import GeometryError
from scmle.data.modalities.geospatial.geometry import Placemark
from scmle.data.modalities.geospatial.geometry import placemarks

DEFAULT = "OGC:CRS84"
"""The CRS of GeoJSON that declares none (RFC 7946, section 4)."""

LIBRARY = "geopandas"
"""The optional library that reads the formats other than GeoJSON."""

COLLECTIONS = frozenset({".geojson", ".json"})
"""Suffixes read as one GeoJSON FeatureCollection."""

SEQUENCES = frozenset({".geojsonl", ".geojsons", ".ndjson", ".jsonl"})
"""Suffixes read as a GeoJSON text sequence, one Feature per record."""

PREFIX = "geo_"
"""The prefix of the columns the kind derives; no property may start with
it."""

SEPARATOR = "\x1e"
"""The record separator RFC 8142 puts before each sequence record."""

COLUMNS = (
    "geo_x",
    "geo_y",
    "geo_width",
    "geo_height",
    "geo_area",
    "geo_length",
    "geo_vertices",
    "geo_parts",
    "geo_type",
)
"""The geometry columns: the centre of the x/y extent, its width and height,
the polygon area, the line length or ring perimeter, the vertex and part
counts, and the GeoJSON geometry type."""


PARTS: Mapping[str, Callable[[Any], list[Any]]] = {
    "Point": lambda position: [[[position]]],
    "MultiPoint": lambda positions: [[[item]] for item in positions],
    "LineString": lambda line: [[line]],
    "MultiLineString": lambda lines: [[line] for line in lines],
    "Polygon": lambda rings: [rings],
    "MultiPolygon": list,
}
"""Per geometry type, its coordinates as parts, each a list of lines (a
point is a line of one position; a polygon's lines are its rings)."""


def load(path: str | pathlib.Path) -> tuple[Mapping[str, Any], str]:
    """Reads a vector file as a FeatureCollection and its declared CRS.

    Args:
        path: The file.

    Returns:
        The FeatureCollection mapping and its CRS: the `crs` member's name
        when the document declares one (the 2008 GeoJSON form), the CRS
        `geopandas` reports for its formats, else `DEFAULT`.

    Raises:
        DataError: If the file is missing or is not valid JSON.
        GeometryError: If the document is not a FeatureCollection, or a
            sequence record is not a Feature.
        UnsupportedError: If the format needs `LIBRARY` and it is not
            installed.
    """
    source = pathlib.Path(path)
    suffix = source.suffix.lower()
    if suffix not in COLLECTIONS | SEQUENCES:
        return foreign(source)
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as error:
        raise errors.DataError(f"cannot read vector file {source}") from error
    try:
        if suffix in SEQUENCES:
            document: Any = {
                "type": "FeatureCollection",
                "features": [
                    json.loads(line.strip(SEPARATOR))
                    for line in text.splitlines()
                    if line.strip(SEPARATOR + " \t")
                ],
            }
        else:
            document = json.loads(text)
    except json.JSONDecodeError as error:
        raise errors.DataError(
            f"vector file {source} is not valid JSON: {error.msg}"
        ) from error
    if not isinstance(document, Mapping) or document.get("type") != (
        "FeatureCollection"
    ):
        raise GeometryError(
            "a GeoJSON file must hold a FeatureCollection",
            {"code": "unsupported_geojson_root", "uri": str(source)},
        )
    return document, declared(document)


def declared(document: Mapping[str, Any]) -> str:
    """The CRS a GeoJSON document names in its `crs` member.

    Args:
        document: The FeatureCollection.

    Returns:
        The named CRS, or `DEFAULT` when the document names none.
    """
    member = document.get("crs")
    if isinstance(member, Mapping):
        properties = member.get("properties")
        if (
            isinstance(properties, Mapping)
            and str(properties.get("name", "")).strip()
        ):
            return str(properties["name"]).strip()
    return DEFAULT


def foreign(source: pathlib.Path) -> tuple[Mapping[str, Any], str]:
    """Reads a non-GeoJSON vector format through the optional library.

    Args:
        source: The file.

    Returns:
        The FeatureCollection and the CRS the library reports, or `DEFAULT`
        when the file declares none.

    Raises:
        UnsupportedError: If `LIBRARY` is not installed.
        DataError: If the library cannot read the file.
    """
    try:
        library = importlib.import_module(LIBRARY)
    except ImportError as error:
        raise capabilities.UnsupportedError(
            f"reading {source.suffix or 'this'} vector files needs the "
            "optional geopandas library; GeoJSON is read without it",
            {
                "status": "unsupported",
                "reason": "optional vector reader is not installed",
                "library": LIBRARY,
                "uri": str(source),
            },
        ) from error
    try:
        frame = library.read_file(source)
        document = json.loads(frame.to_json())
    except Exception as error:  # the library's errors vary by driver
        raise errors.DataError(
            f"{LIBRARY} cannot read vector file {source}: {error}"
        ) from error
    crs = frame.crs.to_string() if frame.crs is not None else DEFAULT
    return document, crs


def tabulate(document: Mapping[str, Any]) -> pandas.DataFrame:
    """One row per feature: its properties, then its geometry measures.

    Args:
        document: A GeoJSON FeatureCollection.

    Returns:
        The rows, in document order. Property columns come in the order
        they first appear; a property a feature lacks is missing.

    Raises:
        GeometryError: If a feature or geometry is invalid
            (`geometry.placemarks`), or a property is named like one of
            `COLUMNS`.
    """
    features = placemarks(document)
    names = list(
        dict.fromkeys(key for item in features for key in item.properties)
    )
    clashes = sorted(name for name in names if name.startswith(PREFIX))
    if clashes:
        raise GeometryError(
            f"properties {clashes} use the reserved '{PREFIX}' prefix of the "
            "geometry columns; rename them in the source",
            {"code": "column_collision", "columns": clashes},
        )
    rows = {
        name: [scalar(item.properties.get(name)) for item in features]
        for name in names
    }
    measures = numpy.asarray([measure(item) for item in features], dtype=object)
    frame = pandas.DataFrame(rows)
    for index, name in enumerate(COLUMNS):
        values: list[Any] = list(measures[:, index]) if len(features) else []
        frame[name] = pandas.Series(values, dtype=dtype(name))
    return frame


def dtype(name: str) -> str:
    """The column type of a geometry measure.

    Args:
        name: One of `COLUMNS`.

    Returns:
        "object" for the type name, "int64" for counts, else "float64".
    """
    if name == "geo_type":
        return "object"
    if name in {"geo_vertices", "geo_parts"}:
        return "int64"
    return "float64"


def scalar(value: Any) -> Any:
    """A property value as one table cell.

    Args:
        value: A JSON-native property value.

    Returns:
        A scalar unchanged; a list or object as compact, key-sorted JSON.
    """
    if isinstance(value, list | dict):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def measure(item: Placemark) -> tuple[Any, ...]:
    """The geometry measures of one feature, in `COLUMNS` order.

    Args:
        item: The validated feature.

    Returns:
        Centre x and y, width, height, area, length, vertices, parts and
        the geometry type.
    """
    minimum, low, maximum, high = item.bounds
    area = length = 0.0
    vertices = parts = 0
    for closed, lines in members(item.geometry):
        parts += 1
        for order, line in enumerate(lines):
            vertices += len(line)
            length += float(numpy.hypot(*numpy.diff(line, axis=0).T).sum())
            if closed:
                surface = abs(shoelace(line))
                area += -surface if order else surface
    return (
        (minimum + maximum) / 2,
        (low + high) / 2,
        maximum - minimum,
        high - low,
        area,
        length,
        vertices,
        parts,
        item.kind,
    )


def members(
    geometry: Mapping[str, Any],
) -> Iterable[tuple[bool, list[numpy.ndarray]]]:
    """The parts of a geometry, each a list of x/y position arrays.

    Args:
        geometry: A validated GeoJSON geometry.

    Yields:
        Per part, whether its lines are polygon rings (the first the
        exterior, the rest holes) and the lines as (n, 2) arrays.
    """
    kind = geometry["type"]
    if kind == "GeometryCollection":
        for nested in geometry["geometries"]:
            yield from members(nested)
        return
    closed = kind in {"Polygon", "MultiPolygon"}
    for part in PARTS[kind](geometry["coordinates"]):
        yield closed, [planar(line) for line in part]


def planar(positions: list[Any]) -> numpy.ndarray:
    """The x and y of each position (MM-2: further values are dropped).

    Args:
        positions: GeoJSON positions.

    Returns:
        An (n, 2) float array.
    """
    return numpy.asarray(
        [position[:2] for position in positions], dtype=numpy.float64
    )


def shoelace(ring: numpy.ndarray) -> float:
    """The signed area a closed ring encloses (the shoelace formula).

    Args:
        ring: An (n, 2) array of ring positions.

    Returns:
        The signed area; zero for fewer than three positions.
    """
    x, y = ring[:, 0], ring[:, 1]
    return (
        float(numpy.dot(x, numpy.roll(y, -1)) - numpy.dot(y, numpy.roll(x, -1)))
        / 2
    )
