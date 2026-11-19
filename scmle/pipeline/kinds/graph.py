"""The graph kind: the labelled nodes of a graph as rows of the shared stages.

A graph source is a directory holding one node table (`nodes.csv`, `.tsv`,
`.jsonl`, `.json` or `.parquet`) and one edge file (`edges.csv`, `.jsonl` or
`.ndjson`). The node table names each node under `node` (read as text, so
`007` and `NA` stay node ids), holds the target and any node features; the
edge file has `source`, `target`, optional `weight`, `relation` and `time`
(ISO 8601 with a zone) columns.

Each node becomes one row: its features with four structural features
(out-degree, in-degree, strength, the summed weight of its edges, and the
mean degree of its neighbours). A node is predicted when it is first seen,
so in a timed graph its structural features count only the edges known by
then: its first timed edge (`graph.debut`), or the node table's declared
`data.time_column`. No feature counts an edge from the future of the node
it describes. The split key (`rows.KEY`) is that time, so the shared
splitter holds out the latest-seen nodes and trains only on earlier ones
(MM-11); an untimed graph's nodes are exchangeable and split as declared.
"""

from collections.abc import Sequence
import dataclasses
from datetime import datetime
import math
import pathlib

import numpy
import pandas

from scmle.contracts import errors
from scmle.contracts import tasks
from scmle.data.modalities import graph
from scmle.data.readers import streaming
from scmle.pipeline.kinds import rows

IDENTITY = "node"
"""The node table's column of node ids."""

TIME = "time"
"""The edge file's column of event times."""

TABLES = (".csv", ".tsv", ".jsonl", ".json", ".parquet")
"""Suffixes of the node table, in the order they are looked for."""

LISTS = (".csv", ".jsonl", ".ndjson")
"""Suffixes of the edge file, in the order they are looked for."""

STRUCTURE = ("outdegree", "indegree", "strength", "neighbourhood")
"""The structural columns each node row gets."""

SECONDS = 2e-4
"""Upper bound on the seconds reading and describing take per node.

Observed on a 20,000-node, 100,000-edge timed graph (one thread,
2026-11-19): reading both files and computing the as-of structure took
0.06 ms a node with five edges each; the bound allows three times that
edge density.
"""

EXAMPLES = 5
"""Identities quoted in an error, so a message stays readable."""


def locate(
    root: pathlib.Path, stem: str, suffixes: Sequence[str]
) -> pathlib.Path:
    """Finds the one file of a stem in a graph directory.

    Args:
        root: The graph directory.
        stem: `nodes` or `edges`.
        suffixes: The suffixes looked for, in order.

    Returns:
        The first file `<stem><suffix>` that exists.

    Raises:
        DataError: If none exists.
    """
    for suffix in suffixes:
        path = root / f"{stem}{suffix}"
        if path.is_file():
            return path
    raise errors.DataError(
        f"graph directory {root} has no {stem} file "
        f"({', '.join(stem + suffix for suffix in suffixes)})",
        {"uri": str(root), "stem": stem},
    )


def moment(value: datetime | None) -> float:
    """A time as seconds since the epoch; an unknown time is the latest.

    Args:
        value: The time, or None for an untimed node or edge.

    Returns:
        Its POSIX timestamp, or infinity.
    """
    return math.inf if value is None else value.timestamp()


@dataclasses.dataclass(frozen=True, slots=True)
class Structure:
    """The edges of a graph indexed for as-of structural features.

    An untimed edge is known at every time; a timed edge from its event
    time on.

    Attributes:
        outward: Each node's out-edge times, sorted.
        inward: Each node's in-edge times, sorted.
        weights: Each node's incident edges as (time, weight, neighbour),
            sorted by time.
    """

    outward: dict[str, numpy.ndarray]
    inward: dict[str, numpy.ndarray]
    weights: dict[str, list[tuple[float, float, str]]]

    @classmethod
    def index(cls, edges: Sequence[graph.Edge]) -> "Structure":
        """Indexes edges by node.

        Args:
            edges: The edges.

        Returns:
            The index.
        """
        outward: dict[str, list[float]] = {}
        inward: dict[str, list[float]] = {}
        weights: dict[str, list[tuple[float, float, str]]] = {}
        for edge in edges:
            at = -math.inf if edge.time is None else moment(edge.time)
            outward.setdefault(edge.source, []).append(at)
            inward.setdefault(edge.target, []).append(at)
            weights.setdefault(edge.source, []).append(
                (at, edge.weight, edge.target)
            )
            weights.setdefault(edge.target, []).append(
                (at, edge.weight, edge.source)
            )
        return cls(
            {node: numpy.sort(times) for node, times in outward.items()},
            {node: numpy.sort(times) for node, times in inward.items()},
            {node: sorted(items) for node, items in weights.items()},
        )

    def degree(self, node: str, at: float) -> int:
        """A node's edges known at a time.

        Args:
            node: The node id.
            at: The time, in POSIX seconds.

        Returns:
            Its out- plus in-edges with an event time at or before `at`.
        """
        empty = numpy.empty(0)
        return int(
            numpy.searchsorted(self.outward.get(node, empty), at, "right")
            + numpy.searchsorted(self.inward.get(node, empty), at, "right")
        )

    def describe(self, node: str, at: float) -> tuple[float, ...]:
        """The structural features of a node as of a time.

        Args:
            node: The node id.
            at: The time, in POSIX seconds.

        Returns:
            Out-degree, in-degree, strength (summed weight) and the mean
            degree of its neighbours, all counting only the edges known at
            `at`; the mean is 0 for a node without a known neighbour.
        """
        empty = numpy.empty(0)
        known = [item for item in self.weights.get(node, []) if item[0] <= at]
        neighbours = sorted({item[2] for item in known})
        return (
            float(
                numpy.searchsorted(self.outward.get(node, empty), at, "right")
            ),
            float(
                numpy.searchsorted(self.inward.get(node, empty), at, "right")
            ),
            float(sum(item[1] for item in known)),
            float(
                numpy.mean([self.degree(other, at) for other in neighbours])
                if neighbours
                else 0.0
            ),
        )


class Graph(rows.Rows):
    """The labelled nodes of a graph, one row each.

    Attributes:
        name: "graph".
    """

    name = "graph"
    seconds = SECONDS
    role = "time"
    evidence = "tests/unit/test_kind_graph.py"

    def load(
        self, spec: tasks.Config
    ) -> tuple[pandas.DataFrame, tuple[graph.Edge, ...]]:
        """Reads the node table and the edges.

        Args:
            spec: The run's specification; its URI is the graph directory.

        Returns:
            The node table and the edges.

        Raises:
            DataError: If the directory or a file is missing or unreadable,
                a node id is missing or repeated, a node table column
                clashes with a column the rows get, or an edge names a node
                the table does not list.
        """
        root = pathlib.Path(spec.data.uri)
        if not root.is_dir():
            raise errors.DataError(
                f"a graph source is a directory with a node table and an "
                f"edge file: {root}",
                {"uri": str(root)},
            )
        listing = rows.catalogue(locate(root, "nodes", TABLES), IDENTITY)
        if IDENTITY not in listing.columns:
            raise errors.DataError(
                f"the node table needs a '{IDENTITY}' column",
                {"columns": [str(column) for column in listing.columns]},
            )
        identities = listing[IDENTITY]
        if identities.isna().any() or (identities.str.strip() == "").any():
            raise errors.DataError("the node table has a missing node id")
        repeated = sorted(set(identities[identities.duplicated()]))
        if repeated:
            raise errors.DataError(
                f"{len(repeated)} node id(s) repeat in the node table",
                {"repeated": repeated[:EXAMPLES]},
            )
        reserved = {*STRUCTURE, rows.KEY} & set(map(str, listing.columns))
        if reserved:
            raise errors.DataError(
                f"node table columns clash with the graph rows: "
                f"{sorted(reserved)}",
                {"reserved": sorted(reserved)},
            )
        try:
            edges = graph.edgelist(
                locate(root, "edges", LISTS), time_column=TIME
            )
        except graph.GraphError as error:
            raise errors.DataError(str(error), {"uri": str(root)}) from error
        known = set(identities)
        unknown = sorted(
            {
                endpoint
                for edge in edges
                for endpoint in (edge.source, edge.target)
                if endpoint not in known
            }
        )
        if unknown:
            raise errors.DataError(
                f"{len(unknown)} edge endpoint(s) are not in the node table",
                {"unknown": unknown[:EXAMPLES]},
            )
        return listing, edges

    def times(
        self,
        spec: tasks.Config,
        listing: pandas.DataFrame,
        edges: Sequence[graph.Edge],
    ) -> list[datetime | None]:
        """When each node is predicted: when it is first seen.

        Args:
            spec: The run's specification.
            listing: The node table.
            edges: The edges.

        Returns:
            Per node, the declared time column's value, else in a timed
            graph its first timed edge, else None (an untimed graph).

        Raises:
            DataError: If a declared time is not a time, or a timed graph
                is partly timed or has a node without an edge
                (`graph.chronicle`).
        """
        column = spec.data.time_column
        if column:
            parsed = pandas.to_datetime(
                listing[column], errors="coerce", utc=True
            )
            if parsed.isna().any():
                raise errors.DataError(
                    f"time column '{column}' has missing or unparseable values"
                )
            return [stamp.to_pydatetime() for stamp in parsed]
        if all(edge.time is None for edge in edges):
            return [None] * len(listing)
        dataset = graph.Dataset.connect(edges, nodes=listing[IDENTITY].tolist())
        try:
            graph.chronicle(dataset)
        except graph.TemporalError as error:
            raise errors.DataError(
                f"{error}; declare data.time_column on the node table to "
                "say when each node is predicted",
                {"uri": spec.data.uri},
            ) from error
        seen = graph.debut(dataset)
        return [seen[node] for node in listing[IDENTITY]]

    def rows(
        self,
        listing: pandas.DataFrame,
        edges: Sequence[graph.Edge],
        times: Sequence[datetime | None],
    ) -> pandas.DataFrame:
        """Turns nodes into the rows a model reads.

        Serving uses it on new nodes with the graph known at their time.

        Args:
            listing: The node table (its `node` column names the nodes).
            edges: The edges.
            times: When each node is predicted; None counts every edge.

        Returns:
            The node table without its ids, with the structural columns.
        """
        structure = Structure.index(edges)
        values = numpy.asarray(
            [
                structure.describe(node, moment(at))
                for node, at in zip(listing[IDENTITY], times, strict=True)
            ],
            dtype=float,
        ).reshape(len(listing), len(STRUCTURE))
        described = pandas.DataFrame(values, columns=list(STRUCTURE))
        return pandas.concat(
            [
                listing.drop(columns=[IDENTITY]).reset_index(drop=True),
                described,
            ],
            axis=1,
        )

    def read(
        self, spec: tasks.Config
    ) -> tuple[
        pandas.DataFrame, tuple[streaming.Feed, streaming.Totals] | None
    ]:
        """Reads the nodes into rows, keyed by when each is first seen.

        Args:
            spec: The run's specification.

        Returns:
            The node rows (see `rows`), with the first-seen time as the
            split key in a timed graph without a declared time column, and
            no stream.

        Raises:
            DataError: See `load` and `times`.
        """
        listing, edges = self.load(spec)
        times = self.times(spec, listing, edges)
        frame = self.rows(listing, edges, times)
        if spec.data.time_column or all(at is None for at in times):
            return frame, None
        stamps = pandas.to_datetime(
            [at.isoformat() for at in times if at], utc=True
        )
        return frame.assign(**{rows.KEY: stamps}), None

    def notice(self, frame: pandas.DataFrame) -> list[str]:
        """The nodes without a known edge and the order the split follows.

        Args:
            frame: The rows, with the split key when they have one.

        Returns:
            A finding counting the nodes with no edge known when they are
            predicted, and one stating that a timed graph's latest-seen
            nodes are held out, each when it applies.
        """
        found: list[str] = []
        if set(STRUCTURE) <= set(frame.columns):
            alone = int(((frame["outdegree"] + frame["indegree"]) == 0).sum())
            if alone:
                found.append(
                    f"{alone} nodes have no edge known when they are "
                    "predicted; their structural features are zero"
                )
        if rows.KEY in frame.columns:
            found.append(
                "the graph is timed: the latest-seen nodes are held out and "
                "structural features count only the edges known by then"
            )
        return found

    def limits(self) -> tuple[str, ...]:
        """The boundaries the graph capability states.

        Returns:
            Plain statements of what the kind does not cover.
        """
        return (
            (
                "node-level tasks on node features and as-of degree "
                "features; no graph neural network"
            ),
            "a served model reads rows built by `Graph.rows`",
        )


GRAPH = Graph()
"""The graph kind."""
