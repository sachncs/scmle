"""Leakage-safe node splits and bounded numeric node features."""

from datetime import datetime
import math
from typing import TYPE_CHECKING

import numpy

from scmle.contracts import errors
from scmle.data.modalities.graph.dataset import Dataset
from scmle.data.modalities.graph.elements import TemporalError
from scmle.data.modalities.graph.evidence import Features
from scmle.data.modalities.graph.evidence import Partition

if TYPE_CHECKING:
    from scmle.data.modalities.graph.elements import Edge


def apportion(
    fractions: tuple[float, float, float], node_count: int
) -> list[int]:
    """Divides nodes among the partitions by largest remainder.

    Args:
        fractions: Train, validation and test shares, summing to one.
        node_count: Number of nodes to divide.

    Returns:
        The node count of each partition. When there are enough nodes,
        every partition with a positive share gets at least one; otherwise
        the partitions with the largest shares get one each.
    """
    positive_indices = [index for index, value in enumerate(fractions) if value]
    counts = [0, 0, 0]
    if node_count >= len(positive_indices):
        raw_counts = [node_count * value for value in fractions]
        floor_counts = [int(value) for value in raw_counts]
        counts = list(floor_counts)
        remainder = node_count - sum(floor_counts)
        order = sorted(
            range(len(fractions)),
            key=lambda index: (
                raw_counts[index] - floor_counts[index],
                fractions[index],
                -index,
            ),
            reverse=True,
        )
        for index in order[:remainder]:
            counts[index] += 1
        empty = sorted(
            (index for index in positive_indices if not counts[index]),
            key=lambda index: (fractions[index], -index),
            reverse=True,
        )
        for index in empty:
            donor = max(
                range(len(fractions)),
                key=lambda other: (counts[other], fractions[other], -other),
            )
            counts[donor] -= 1
            counts[index] = 1
    else:
        order = sorted(
            range(len(fractions)),
            key=lambda index: (fractions[index], -index),
            reverse=True,
        )
        for index in order[:node_count]:
            counts[index] = 1
    return counts


def debut(graph: Dataset) -> dict[str, datetime]:
    """The time each node is first seen: its earliest timed edge.

    Args:
        graph: The graph.

    Returns:
        For each node with at least one timed edge, the earliest event time
        of its edges; nodes without one are absent.
    """
    seen: dict[str, datetime] = {}
    for edge in graph.edges:
        if edge.event_time is None:
            continue
        for node in (edge.source, edge.target):
            if node not in seen or edge.event_time < seen[node]:
                seen[node] = edge.event_time
    return seen


def chronicle(graph: Dataset) -> list[str]:
    """Orders the nodes of a timed graph by when they are first seen.

    Args:
        graph: The graph; every edge carries an event time.

    Returns:
        The node ids, earliest first seen first; ties by id.

    Raises:
        TemporalError: If only some edges are timed (the order of the
            others is unknown), or a node has no edge to be seen by.
    """
    timed = sum(edge.event_time is not None for edge in graph.edges)
    if timed != len(graph.edges):
        raise TemporalError(
            f"graph is partly timed ({timed} of {len(graph.edges)} edges): "
            "a node split cannot order its nodes in time"
        )
    seen = debut(graph)
    unseen = sorted(set(graph.node_ids) - set(seen))
    if unseen:
        raise TemporalError(
            f"{len(unseen)} node(s) have no timed edge, so when they are "
            f"first seen is unknown: {unseen[:5]}"
        )
    return sorted(graph.node_ids, key=lambda node: (seen[node], node))


def arrange(graph: Dataset, seed: int) -> numpy.ndarray:
    """The order `induct` deals the nodes out in.

    Args:
        graph: The graph.
        seed: Seed of the shuffle of an untimed graph.

    Returns:
        A timed graph's nodes earliest first seen first (`chronicle`), else
        the nodes shuffled with the seed.
    """
    if any(edge.event_time is not None for edge in graph.edges):
        return numpy.asarray(chronicle(graph), dtype=object)
    node_ids = numpy.asarray(graph.node_ids, dtype=object)
    generator = numpy.random.default_rng(seed)
    return node_ids[generator.permutation(len(node_ids))]


def induct(
    graph: Dataset,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 0,
) -> Partition:
    """Creates deterministic node partitions with no retained edge leakage.

    An untimed graph's nodes are shuffled with the seed; a timed graph's
    nodes are ordered by when they are first seen (`chronicle`), so the
    validation and test nodes are the latest and no node is trained on that
    appears after a held-out one (MM-11). The nodes are divided by the
    fractions; every partition with a positive fraction receives at least
    one node when there are enough nodes. Edges inside a partition are
    retained for it and edges between partitions are withheld.

    Args:
        graph: The graph; it must be valid.
        train_fraction: Share of nodes for training.
        validation_fraction: Share of nodes for validation.
        test_fraction: Share of nodes for testing.
        seed: Seed of the node shuffle of an untimed graph.

    Returns:
        The node and edge partitions with the seed.

    Raises:
        ValueError: If a fraction is negative or not finite, or the
            fractions do not sum to one. An invalid graph raises
            ValidationError from `Dataset.ensure`; a partly timed graph
            raises TemporalError from `chronicle`.
    """
    graph.ensure()
    fractions = (train_fraction, validation_fraction, test_fraction)
    if any(value < 0 or not math.isfinite(value) for value in fractions):
        raise errors.InputError(
            "split fractions must be finite and non-negative"
        )
    if not math.isclose(sum(fractions), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise errors.InputError("split fractions must sum to one")
    shuffled = arrange(graph, seed)
    counts = apportion(fractions, len(shuffled))
    train_end = counts[0]
    validation_end = train_end + counts[1]
    train_nodes = tuple(sorted(str(value) for value in shuffled[:train_end]))
    validation_nodes = tuple(
        sorted(str(value) for value in shuffled[train_end:validation_end])
    )
    test_nodes = tuple(
        sorted(str(value) for value in shuffled[validation_end:])
    )
    partitions = {
        node_id: index
        for index, nodes in enumerate(
            (train_nodes, validation_nodes, test_nodes)
        )
        for node_id in nodes
    }
    retained: list[list[Edge]] = [[], [], []]
    withheld: list[Edge] = []
    for edge in graph.edges:
        source_partition = partitions[edge.source]
        target_partition = partitions[edge.target]
        if source_partition == target_partition:
            retained[source_partition].append(edge)
        else:
            withheld.append(edge)
    return Partition(
        train_nodes=train_nodes,
        validation_nodes=validation_nodes,
        test_nodes=test_nodes,
        train_edges=tuple(retained[0]),
        validation_edges=tuple(retained[1]),
        test_edges=tuple(retained[2]),
        withheld_edges=tuple(withheld),
        seed=seed,
    )


def valence(graph: Dataset) -> dict[str, list[float]]:
    """Computes the out-degree, in-degree and weighted degree of each node.

    Args:
        graph: The graph.

    Returns:
        For each node id, the three degrees in that order; the weighted
        degree is the summed weight of incident edges. Edges with an
        endpoint outside the node set are not counted at that endpoint.
    """
    degrees = {node_id: [0.0, 0.0, 0.0] for node_id in graph.node_ids}
    for edge in graph.edges:
        if edge.source in degrees:
            degrees[edge.source][0] += 1.0
            degrees[edge.source][2] += edge.weight
        if edge.target in degrees:
            degrees[edge.target][1] += 1.0
            degrees[edge.target][2] += edge.weight
    return degrees


def featurize(
    rows: list[list[float]], width: int, *, standardize: bool
) -> tuple[tuple[float, ...], ...]:
    """Turns feature rows into the immutable values of a feature matrix.

    Args:
        rows: One list of features per node, already cut to ``width``.
        width: Number of columns kept.
        standardize: Whether to scale each column to zero mean and unit
            variance (constant columns are only centred).

    Returns:
        The values as nested tuples; empty rows when there is no row or no
        column.

    Raises:
        ValueError: If a feature value is not finite.
    """
    if not (rows and width):
        return ((),) * len(rows)
    matrix = numpy.asarray(rows, dtype=numpy.float64)
    if not numpy.isfinite(matrix).all():
        raise errors.InputError("prepared features must be finite")
    if standardize:
        means = matrix.mean(axis=0)
        scales = matrix.std(axis=0)
        scales[scales == 0] = 1.0
        matrix = (matrix - means) / scales
    return tuple(tuple(float(value) for value in row) for row in matrix)


def prepare_features(
    graph: Dataset,
    max_nodes: int = 100_000,
    max_features: int = 512,
    standardize: bool = False,
    include_structural: bool = True,
) -> Features:
    """Prepares bounded numeric node features for a baseline model.

    User-provided features are followed by deterministic degree features when
    ``include_structural`` is true. Nodes and features are truncated in
    canonical order, and no graph-neural-network dependency is introduced.
    Structural features are out-degree, in-degree and weighted degree (the
    summed weight of incident edges).

    Args:
        graph: The graph; it must be valid.
        max_nodes: Largest number of nodes kept, in canonical order.
        max_features: Largest number of columns kept.
        standardize: Whether to scale each column to zero mean and unit
            variance (constant columns are only centred).
        include_structural: Whether to append the degree features.

    Returns:
        The prepared features. Without any column the rows are empty.

    Raises:
        ValueError: If ``max_nodes`` or ``max_features`` is below one, or a
            feature value is not finite. An invalid graph raises
            ValidationError from `Dataset.ensure`.
    """
    graph.ensure()
    if max_nodes < 1 or max_features < 1:
        raise errors.InputError(
            "max_nodes and max_features must be at least one"
        )
    nodes = graph.nodes[:max_nodes]
    feature_dimension = max((len(node.features) for node in nodes), default=0)
    names = list(graph.feature_names)
    if feature_dimension and not names:
        names = [f"feature_{index}" for index in range(feature_dimension)]
    rows = [list(node.features) for node in nodes]
    if include_structural:
        degrees = valence(graph)
        for index, node in enumerate(nodes):
            rows[index].extend(degrees[node.node_id])
        names.extend(("out_degree", "in_degree", "weighted_degree"))
    width = min(max_features, max((len(row) for row in rows), default=0))
    rows = [row[:width] for row in rows]
    names = names[:width]
    return Features(
        node_ids=tuple(node.node_id for node in nodes),
        feature_names=tuple(names),
        values=featurize(rows, width, standardize=standardize),
        standardized=standardize,
    )
