"""Claim C5, real part: point-in-time features on public relational data.

`DATASETS` lists five public multi-table datasets under free licences, each
pinned to an exact file (a commit-pinned URL and its SHA-256). `fetch`
downloads them into `CACHE`, the benchmark cache other real data uses, and
refuses a file whose checksum differs.

Every dataset is read twice, independently:

* the product: SQLite through the product's read-only database connector
  (`DbApiConnector` with the snapshot-checked `profile.untorn`), CSV through
  `read_table`, Stack Exchange XML through `pandas.read_xml` (the product has
  no XML reader); times stay as the source gives them (text or numbers) and
  are converted by the product's as-of view;
* the reference: the standard library only (`sqlite3`, `csv`,
  `xml.etree`), times converted to days exactly (`days`: integer
  microseconds over a day, rounded once).

Targets: `ENTITIES` entities of the child table, spread evenly over its
row-count order (light and heavy ones), each asked at the hard cutoffs
`cutoffs` builds from its own rows: ties with an event time and with a
recorded time, a window's edge, before the first row and after the last.

Features: every primitive of the grammar over each window (none and
`WINDOWS_DAYS`), and, where the dataset has a leaf table, every nested
inner x outer pair of the grammar, rotated over the time rules (recorded
time where the dataset has one, else event time plus a delay), delays and
lags. Each is computed by `synthesize.evaluate` over `asof_view.View` on the
whole tables and by `ingest_naive` on the sampled entities' rows, and
compared within `reference_rel_tol` relative and absolute.

A dataset counts only when it was fetched, read and every feature agrees;
a download, checksum, reading or evaluation failure counts against it.
"""

import csv
import dataclasses
import datetime
import fractions
import hashlib
import itertools
import math
from pathlib import Path
import sqlite3
import tempfile
from typing import Any
from urllib import request
from xml.etree import ElementTree

import numpy
import pandas

from benchmarks.qualification import ingest_naive
from scmle.assurance.ingest import asof_view
from scmle.assurance.ingest import profile
from scmle.assurance.ingest import synthesize
import scmle.assurance.ingest.plan
from scmle.data.readers import connectors
from scmle.data.readers import tabular

# The benchmark cache of real data (as `benchmarks.real_tasks.CACHE`).
CACHE = Path(tempfile.gettempdir()) / "scmle_real_data" / "relational"
# Entities asked about per dataset, spread from the lightest to the
# heaviest. The reference loops over every mid row and every leaf row per
# target, so the sample is bounded by its cost (about two minutes for all
# datasets at 16); the cutoffs, not the count, carry the hard cases.
ENTITIES = 16
# Rows per snapshot-checked batch of the product's database read.
BATCH = 100_000
LAGS = (0.0, 1.0, 30.0)
DELAYS = (0.0, 1.0)
EPOCH = datetime.datetime(1970, 1, 1, tzinfo=datetime.UTC)
MICROSECONDS = 86_400 * 10**6
TIMEOUT = 300


@dataclasses.dataclass(frozen=True)
class File:
    """One pinned source file.

    Attributes:
        name: Name in the cache.
        url: Commit- or snapshot-pinned download URL.
        sha256: Expected SHA-256 of the bytes.
    """

    name: str
    url: str
    sha256: str


@dataclasses.dataclass(frozen=True)
class Table:
    """One table of a dataset and how it is timed.

    Attributes:
        name: Table name (SQLite) or file name (CSV, XML).
        key: Column joining it to its parent (the entity, or the mid id).
        event: Event time column; None takes the mid row's times.
        known: Recorded time column, or None.
        measures: Numeric value columns.
        ident: Identifier column the leaf rows point to (mid table only).
        order: SQLite ORDER BY clause.
    """

    name: str
    key: str
    event: str | None
    known: str | None
    measures: tuple[str, ...]
    ident: str | None = None
    order: str = ""


@dataclasses.dataclass(frozen=True)
class Dataset:
    """One public relational dataset.

    Attributes:
        name: Identifier.
        licence: Licence as the source states it.
        home: Source page.
        form: `sqlite`, `csv` or `xml`.
        files: Pinned files; the first is the database for `sqlite`.
        child: The entity's child table.
        leaf: The child's own child table, or None.
        numeric: Whether times are plain numbers (years), not timestamps.
    """

    name: str
    licence: str
    home: str
    form: str
    files: tuple[File, ...]
    child: Table
    leaf: Table | None = None
    numeric: bool = False


RAW = "https://raw.githubusercontent.com"
DATASETS = (
    Dataset(
        "chinook",
        "MIT",
        "https://github.com/lerocha/chinook-database",
        "sqlite",
        (
            File(
                "chinook.sqlite",
                f"{RAW}/lerocha/chinook-database/"
                "7f67772503d71ba90f19283c38e93923addb43fa/ChinookDatabase/"
                "DataSources/Chinook_Sqlite.sqlite",
                "7651ba378ac2fcd0dfc3c66fb101f7a7"
                "eed3ba39a612ec642b96e20702061f15",
            ),
        ),
        Table(
            "Invoice",
            "CustomerId",
            "InvoiceDate",
            None,
            ("Total",),
            "InvoiceId",
            "InvoiceId",
        ),
        Table(
            "InvoiceLine",
            "InvoiceId",
            None,
            None,
            ("UnitPrice", "Quantity"),
            order="InvoiceLineId",
        ),
    ),
    Dataset(
        "northwind",
        "MIT",
        "https://github.com/jpwhite3/northwind-SQLite3",
        "sqlite",
        (
            File(
                "northwind.db",
                f"{RAW}/jpwhite3/northwind-SQLite3/"
                "4f56e7f5906dfd23b25244c5bfe8fb5da6402efd/dist/northwind.db",
                "2f4f5c68dfcd33ba27373eae48c7a486"
                "9800c68095ee0f9f0da494f83382a877",
            ),
        ),
        Table(
            "Orders",
            "CustomerID",
            "OrderDate",
            "ShippedDate",
            ("Freight", "ShipVia"),
            "OrderID",
            "OrderID",
        ),
        Table(
            "Order Details",
            "OrderID",
            None,
            None,
            ("UnitPrice", "Quantity", "Discount"),
            order="OrderID, ProductID",
        ),
    ),
    Dataset(
        "sakila",
        "BSD-3-Clause",
        "https://github.com/bradleygrant/sakila-sqlite3",
        "sqlite",
        (
            File(
                "sakila.db",
                f"{RAW}/bradleygrant/sakila-sqlite3/"
                "9394b42d13888c3d3d3d56cd7e9c84fadafb71c7/sakila_master.db",
                "88c91a4a1a6b61f9d3f35904c0a173c8"
                "87b25e73f20c3c2fdb073818c06f4268",
            ),
        ),
        Table(
            "rental",
            "customer_id",
            "rental_date",
            "last_update",
            ("inventory_id", "staff_id"),
            "rental_id",
            "rental_id",
        ),
        Table(
            "payment",
            "rental_id",
            "payment_date",
            "last_update",
            ("amount",),
            order="payment_id",
        ),
    ),
    Dataset(
        "baseball",
        "CC BY-SA 3.0",
        "https://github.com/xorq-labs/baseballdatabank",
        "csv",
        (
            File(
                "Batting.csv",
                f"{RAW}/xorq-labs/baseballdatabank/"
                "7c6d48bae384c0b2669d0ee1133b99ed083dace6/core/Batting.csv",
                "6628e132e5e77cd558545bd8773ac415"
                "c00eae1831d6a3fbacd58cf4dd94bd31",
            ),
        ),
        Table("Batting.csv", "playerID", "yearID", None, ("H", "AB", "IBB")),
        numeric=True,
    ),
    Dataset(
        "coffee",
        "CC BY-SA 4.0",
        "https://archive.org/details/stackexchange",
        "xml",
        (
            File(
                "Posts.xml",
                "https://archive.org/download/stackexchange/"
                "coffee.stackexchange.com.7z/Posts.xml",
                "713f20db132c55294fba92617c9f647d"
                "73e4e8ab3b3b522b2eb3b503c399f6b6",
            ),
            File(
                "Comments.xml",
                "https://archive.org/download/stackexchange/"
                "coffee.stackexchange.com.7z/Comments.xml",
                "00595f089eafb1810918451675fe06a1"
                "2e16b8afb9d88b68b8128a54cfe73d16",
            ),
        ),
        Table(
            "Posts.xml",
            "OwnerUserId",
            "CreationDate",
            "LastEditDate",
            ("Score", "ViewCount"),
            "Id",
        ),
        Table("Comments.xml", "PostId", "CreationDate", None, ("Score",)),
    ),
)


def fetch(dataset: Dataset) -> Path:
    """Downloads a dataset's files into the cache unless they are there.

    Args:
        dataset: The dataset.

    Returns:
        Its cache folder.

    Raises:
        OSError: If a download fails or a file's checksum differs.
    """
    folder = CACHE / dataset.name
    folder.mkdir(parents=True, exist_ok=True)
    for file in dataset.files:
        path = folder / file.name
        if not path.exists():
            with request.urlopen(file.url, timeout=TIMEOUT) as response:
                body: bytes = response.read()
            path.write_bytes(body)
        found = hashlib.sha256(path.read_bytes()).hexdigest()
        if found != file.sha256:
            path.unlink()
            raise OSError(f"{file.name}: SHA-256 {found}, not {file.sha256}")
    return folder


def days(cell: Any, numeric: bool) -> float:
    """A time as float days, exactly, or NaN when missing.

    Args:
        cell: Text (ISO 8601) or a number.
        numeric: Whether times are plain numbers.

    Returns:
        Days since 1970-01-01 rounded once from integer microseconds, the
        number itself for numeric times, NaN for a missing cell.
    """
    if cell is None or cell == "":
        return math.nan
    if numeric:
        return float(cell)
    moment = datetime.datetime.fromisoformat(str(cell))
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=datetime.UTC)
    return float(fractions.Fraction(micro(moment), MICROSECONDS))


def micro(moment: datetime.datetime) -> int:
    """Microseconds since the epoch.

    Args:
        moment: A timezone-aware time.

    Returns:
        The exact integer count.
    """
    return (moment - EPOCH) // datetime.timedelta(microseconds=1)


def number(cell: Any) -> Any:
    """A cell as the reference reads it: a number, text, or NaN.

    Args:
        cell: Raw cell.

    Returns:
        NaN for missing, an int or float for numeric text, else the cell.
    """
    if cell is None or cell == "":
        return math.nan
    if isinstance(cell, int | float):
        return cell
    for kind in (int, float):
        try:
            return kind(cell)
        except ValueError:
            continue
    return cell


def raw(dataset: Dataset, folder: Path, table: Table) -> list[dict[str, Any]]:
    """A table's rows read with the standard library only.

    Args:
        dataset: The dataset.
        folder: Its cache folder.
        table: The table.

    Returns:
        Rows as dicts in source order.
    """
    if dataset.form == "sqlite":
        connection = sqlite3.connect(folder / dataset.files[0].name)
        connection.row_factory = sqlite3.Row
        try:
            query = f'SELECT * FROM "{table.name}" ORDER BY {table.order}'
            return [dict(row) for row in connection.execute(query)]
        finally:
            connection.close()
    path = folder / table.name
    if dataset.form == "csv":
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    return [dict(row.attrib) for row in ElementTree.parse(path).getroot()]


def product(dataset: Dataset, folder: Path, table: Table) -> pandas.DataFrame:
    """A table as the product reads it.

    Args:
        dataset: The dataset.
        folder: Its cache folder.
        table: The table.

    Returns:
        The frame.
    """
    if dataset.form == "sqlite":
        query = f'SELECT * FROM "{table.name}" ORDER BY {table.order}'
        connector = connectors.DbApiConnector(
            f"sqlite://{dataset.name}",
            query,
            scmle.assurance.ingest.plan.opener(folder / dataset.files[0].name),
        )
        return profile.untorn(connector, BATCH).frame
    if dataset.form == "csv":
        return tabular.read(folder / table.name)
    return pandas.read_xml(folder / table.name, parser="etree")


@dataclasses.dataclass
class Tables:
    """A dataset's child and leaf tables, as each side read them.

    Attributes:
        child: Product frame of the child table.
        leaf: Product frame of the leaf table, with event and known time.
        rows: Reference child rows, times in days.
        leaves: Reference leaf rows, times in days.
    """

    child: pandas.DataFrame
    leaf: pandas.DataFrame | None
    rows: pandas.DataFrame
    leaves: pandas.DataFrame | None


EVENT = "event"
KNOWN = "known"


def reference(
    rows: list[dict[str, Any]], table: Table, numeric: bool
) -> pandas.DataFrame:
    """Reference rows: key, id, event and known days, numeric values.

    Args:
        rows: Rows from `raw`.
        table: The table; without an event column the times are NaN.
        numeric: Whether times are plain numbers.

    Returns:
        An object frame in source order.
    """
    out = []
    for row in rows:
        entry = {
            "key": number(row.get(table.key)),
            EVENT: days(row.get(table.event), numeric)
            if table.event
            else math.nan,
            KNOWN: days(row.get(table.known), numeric)
            if table.known
            else math.nan,
        }
        if table.ident:
            entry["ident"] = number(row.get(table.ident))
        for column in table.measures:
            entry[column] = number(row.get(column))
        out.append(entry)
    return pandas.DataFrame(out, dtype=object)


def load(dataset: Dataset) -> Tables:
    """Fetches and reads a dataset on both sides.

    A leaf table without its own times takes its mid row's event and known
    times (an invoice line happens with its invoice).

    Args:
        dataset: The dataset.

    Returns:
        The tables.
    """
    folder = fetch(dataset)
    child = dataset.child
    frame = product(dataset, folder, child)
    rows = reference(raw(dataset, folder, child), child, dataset.numeric)
    if dataset.leaf is None:
        return Tables(frame, None, rows, None)
    leaf = dataset.leaf
    side = product(dataset, folder, leaf)
    lines = raw(dataset, folder, leaf)
    if leaf.event is None:
        times = [child.ident, child.event] + (
            [child.known] if child.known else []
        )
        renamed = {
            child.ident: leaf.key,
            child.event: EVENT,
            child.known: KNOWN,
        }
        side = side.merge(
            frame[times].rename(columns=renamed), on=leaf.key, how="left"
        )
        lookup = {
            ident: (event, known)
            for ident, event, known in zip(
                rows["ident"], rows[EVENT], rows[KNOWN], strict=True
            )
        }
        leaves = reference(lines, leaf, dataset.numeric)
        paired = [lookup.get(k, (math.nan, math.nan)) for k in leaves["key"]]
        leaves[EVENT] = [p[0] for p in paired]
        leaves[KNOWN] = [p[1] for p in paired]
    else:
        side = side.rename(columns={leaf.event: EVENT, leaf.known: KNOWN})
        leaves = reference(lines, leaf, dataset.numeric)
    return Tables(frame, side, rows, leaves)


def entities(rows: pandas.DataFrame) -> list[Any]:
    """`ENTITIES` keys spread evenly over the row-count order.

    Args:
        rows: Reference child rows.

    Returns:
        Keys, from the lightest to the heaviest entity.
    """
    keyed = rows.loc[[not ingest_naive.is_missing(k) for k in rows["key"]]]
    counts = keyed.groupby("key").size()
    ordered = sorted(counts.index, key=lambda k: (counts[k], str(k)))
    picks = numpy.linspace(0, len(ordered) - 1, ENTITIES).round()
    return [ordered[int(i)] for i in dict.fromkeys(picks)]


def cutoffs(own: pandas.DataFrame, window: float) -> list[float]:
    """Hard cutoffs for one entity, in days.

    Args:
        own: The entity's reference child rows.
        window: A window length in days.

    Returns:
        A tie with the middle event time, a tie with a recorded time, an
        event time plus `window` (the window's edge), one day before the
        first event and one after the last known time.
    """
    events = sorted(e for e in own[EVENT] if not math.isnan(e))
    known = sorted(k for k in own[KNOWN] if not math.isnan(k))
    middle = events[len(events) // 2]
    out = [middle, middle + window, events[0] - 1.0]
    if known:
        out.append(known[len(known) // 2])
    out.append(max(events + known) + 1.0)
    return out


def instant(value: float, numeric: bool) -> Any:
    """A cutoff in days as the product's moment column holds it.

    Args:
        value: Days since the epoch (or a plain number).
        numeric: Whether times are plain numbers.

    Returns:
        The number, or the timestamp to the microsecond.
    """
    if numeric:
        return value
    return pandas.Timestamp(round(value * MICROSECONDS), unit="us")


@dataclasses.dataclass(frozen=True)
class Spec:
    """One feature compared.

    Attributes:
        primitive: Primitive (outer one when nested).
        value: Value column, or None.
        window: Window in days, or None.
        known: Whether the recorded-time rule is used.
        delay: Recording delay in days.
        lag: Gap between moment and cutoff.
        inner: Inner primitive when nested, else None.
    """

    primitive: str
    value: str | None
    window: float | None
    known: bool
    delay: float
    lag: float
    inner: str | None = None

    def label(self) -> str:
        """Short description for the stored document.

        Returns:
            The fields in one line.
        """
        rule = "recorded" if self.known else f"delay {self.delay}"
        head = (
            f"{self.inner}->{self.primitive}" if self.inner else self.primitive
        )
        return (
            f"{head}({self.value}) window {self.window}, {rule}, lag {self.lag}"
        )


def specs(dataset: Dataset) -> list[Spec]:
    """Every feature compared on a dataset.

    Args:
        dataset: The dataset.

    Returns:
        Each primitive over each window, then each nested inner x outer
        pair of the grammar over each window when there is a leaf table;
        rules, delays, lags and value columns rotate.
    """
    windows = (None, *synthesize.WINDOWS_DAYS)
    out = []
    pairs = itertools.product(synthesize.PRIMITIVES, windows)
    for i, (name, window) in enumerate(pairs):
        free = name in synthesize.VALUE_FREE
        values = dataset.child.measures
        out.append(
            Spec(
                name,
                None if free else values[i % len(values)],
                window,
                bool(dataset.child.known) and i % 2 == 0,
                DELAYS[i // 2 % len(DELAYS)],
                LAGS[i % len(LAGS)],
            )
        )
    if dataset.leaf is None:
        return out
    nested = itertools.product(
        synthesize.NESTED_INNER, synthesize.NESTED_OUTER, windows
    )
    for i, (inner, outer, window) in enumerate(nested):
        values = dataset.leaf.measures
        out.append(
            Spec(
                outer,
                None if inner == synthesize.COUNT else values[i % len(values)],
                window,
                bool(dataset.child.known) and i % 2 == 0,
                DELAYS[i // 2 % len(DELAYS)],
                LAGS[i % len(LAGS)],
                inner,
            )
        )
    return out


def fast(
    spec: Spec,
    dataset: Dataset,
    tables: Tables,
    targets: pandas.DataFrame,
) -> numpy.ndarray:
    """The feature by the product over the whole tables.

    Args:
        spec: The feature.
        dataset: The dataset.
        tables: Both sides' tables.
        targets: Product targets: `key` and `moment`.

    Returns:
        One float per target.
    """
    child = dataset.child

    def rule(known: str | None) -> asof_view.Timing:
        if spec.known and known:
            return asof_view.Timing(asof_view.TRANSACTION, known, spec.delay)
        return asof_view.Timing(asof_view.EVENT_DELAY, None, spec.delay)

    view = asof_view.View(
        tables.child, child.key, rule(child.known), child.event, child.name
    )
    source = synthesize.Source(view, "key")
    if spec.inner is None:
        feature = synthesize.Feature(
            "f", "aggregate", 0, spec.primitive, spec.value, spec.window
        )
        nests = []
    else:
        if tables.leaf is None or dataset.leaf is None or child.ident is None:
            raise ValueError("a nested feature needs a leaf table")
        own = dataset.leaf.event is not None
        known = KNOWN if (dataset.leaf.known if own else child.known) else None
        leaf = asof_view.View(
            tables.leaf, dataset.leaf.key, rule(known), EVENT, "leaf"
        )
        nests = [synthesize.Nest(source, leaf, child.ident)]
        feature = synthesize.Feature(
            "f",
            "nested",
            0,
            spec.primitive,
            spec.value,
            spec.window,
            nest=0,
            inner=spec.inner,
        )
    values = synthesize.evaluate(
        feature, targets, "moment", [source], nests, spec.lag
    )
    return values.to_numpy(dtype=float)


def naive(spec: Spec, tables: Tables, asked: pandas.DataFrame) -> numpy.ndarray:
    """The feature by the row-by-row reference, one entity at a time.

    Each entity's targets see only that entity's child rows and their leaf
    rows; the reference still filters by key and time itself.

    Args:
        spec: The feature.
        tables: Both sides' tables.
        asked: Reference targets: `key` and `moment` in days.

    Returns:
        One float per target, in `asked` order.
    """
    known = KNOWN if spec.known else None
    out = numpy.full(len(asked), numpy.nan)
    for key, group in asked.groupby("key", sort=False):
        own = tables.rows.loc[tables.rows["key"] == key]
        if spec.inner is None:
            found = ingest_naive.aggregate_feature(
                group, "key", "moment", own, "key", EVENT, known,
                spec.value, spec.primitive, spec.window, spec.lag, spec.delay,
            )  # fmt: skip
        else:
            leaves = tables.leaves
            if leaves is None:
                raise ValueError("a nested feature needs a leaf table")
            links = set(own["ident"])
            mine = leaves.loc[[k in links for k in leaves["key"]]]
            found = ingest_naive.nested_feature(
                group, "key", "moment", own, "key", "ident", EVENT, known,
                mine, "key", EVENT, known, spec.value, spec.inner,
                spec.primitive, spec.window, spec.window, spec.lag,
                spec.delay,
            )  # fmt: skip
        out[group.index.to_numpy()] = found
    return out


def compare(dataset: Dataset, tolerance: float) -> dict[str, Any]:
    """Fetches, reads and compares every feature of one dataset.

    Args:
        dataset: The dataset.
        tolerance: Relative and absolute tolerance.

    Returns:
        The dataset's name, licence, source, file checksums, entities,
        targets, features compared and equal, the largest gap, the
        unequal features (with the first differing target) and `error`,
        empty unless something failed before the comparison ended.
    """
    out: dict[str, Any] = {
        "name": dataset.name,
        "licence": dataset.licence,
        "home": dataset.home,
        "files": {f.name: f.sha256 for f in dataset.files},
        "features": 0,
        "equal": 0,
        "targets": 0,
        "max_gap": 0.0,
        "unequal": [],
        "error": "",
    }
    try:
        tables = load(dataset)
        picked = entities(tables.rows)
        window = synthesize.WINDOWS_DAYS[0]
        asked = pandas.DataFrame(
            [
                {"key": key, "moment": cut}
                for key in picked
                for cut in cutoffs(
                    tables.rows.loc[tables.rows["key"] == key], window
                )
            ]
        )
        out["entities"] = len(picked)
        out["targets"] = len(asked)
        for spec in specs(dataset):
            moved = asked.assign(moment=asked["moment"] + spec.lag)
            targets = pandas.DataFrame(
                {
                    "key": moved["key"].to_numpy(),
                    "moment": [
                        instant(m, dataset.numeric) for m in moved["moment"]
                    ],
                }
            )
            got = fast(spec, dataset, tables, targets)
            want = naive(spec, tables, moved)
            out["features"] += 1
            equal = got.shape == want.shape and numpy.allclose(
                got, want, rtol=tolerance, atol=tolerance, equal_nan=True
            )
            both = ~(numpy.isnan(got) | numpy.isnan(want))
            gap = float(numpy.max(numpy.abs(got - want)[both], initial=0.0))
            out["max_gap"] = max(out["max_gap"], gap)
            if equal:
                out["equal"] += 1
                continue
            close = numpy.isclose(
                got, want, rtol=tolerance, atol=tolerance, equal_nan=True
            )
            first = int(numpy.flatnonzero(~close)[0])
            out["unequal"].append(
                {
                    "feature": spec.label(),
                    "target": str(asked.iloc[first].to_dict()),
                    "product": float(got[first]),
                    "reference": float(want[first]),
                }
            )
    except Exception as error:  # noqa: BLE001 -- any failure counts against the dataset
        out["error"] = f"{type(error).__name__}: {error}"[:300]
    return out


def agrees(found: dict[str, Any]) -> bool:
    """Whether a dataset counts toward the criterion.

    Args:
        found: One `compare` result.

    Returns:
        True only when it was read without error and every one of at least
        one feature equals the reference.
    """
    return (
        not found.get("error")
        and found.get("features", 0) > 0
        and found.get("equal", 0) == found.get("features", 0)
    )
