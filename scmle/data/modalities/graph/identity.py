"""The canonical, order-independent identity of graph content."""

from collections.abc import Iterable
import hashlib
import json
from typing import Any

from scmle.data.modalities.graph.elements import Edge
from scmle.data.modalities.graph.elements import Node


def graph_identity(
    nodes: Iterable[Node],
    edges: Iterable[Edge],
    directed: bool = True,
    feature_names: Iterable[str] = (),
) -> str:
    """Returns a canonical SHA-256 identity independent of input ordering.

    For an undirected graph the endpoints of each edge are sorted before
    hashing.

    Args:
        nodes: The nodes.
        edges: The edges.
        directed: Whether edges are directed.
        feature_names: Names of the node feature columns; hashed in sorted
            order.

    Returns:
        The SHA-256 hex digest of the canonical JSON of the content.
    """
    node_records: list[dict[str, Any]] = [
        {
            "node_id": node.node_id,
            "features": list(node.features),
            "label": node.label,
        }
        for node in nodes
    ]
    edge_records: list[dict[str, Any]] = [
        {
            "source": edge.source,
            "target": edge.target,
            "weight": edge.weight,
            "relation": edge.relation,
            "event_time": (
                edge.event_time.isoformat() if edge.event_time else None
            ),
        }
        for edge in edges
    ]
    if not directed:
        for record in edge_records:
            source = str(record["source"])
            target = str(record["target"])
            record["source"], record["target"] = sorted((source, target))
    payload = {
        "directed": directed,
        "feature_names": sorted(str(name) for name in feature_names),
        "nodes": sorted(
            node_records, key=lambda record: str(record["node_id"])
        ),
        "edges": sorted(
            edge_records,
            key=lambda record: (
                str(record["source"]),
                str(record["target"]),
                str(record["relation"]),
                float(record["weight"]),
                str(record["event_time"]),
            ),
        ),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
