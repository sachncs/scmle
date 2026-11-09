"""A causal DAG held as a hashed hypothesis, with graph criteria.

The graph comes from domain experts or an LLM draft. It is stored as an input
marked `hypothesis`, with the source and prompt of the draft, and its hash
covers every one of those fields, so any edit changes the hash. Nothing in
this package confirms a DAG: conditional-independence tests (see
`independence`) can only falsify an implied independence.

Graph criteria implemented here, with the references they follow:

* d-separation by moralising the ancestral graph (Lauritzen, Dawid,
  Larsen and Leimer, 1990, Networks 20);
* the back-door criterion (Pearl, 2009, Causality, 2nd ed., Definition
  3.3.1 and Theorem 3.3.2), checked as: no member of the set descends from
  the treatment, and the treatment and outcome are d-separated by the set in
  the graph without the treatment's outgoing edges.

Tests compare both against brute-force enumeration of paths using the
textbook blocking rules. R `dagitty` was not available here, so agreement
with it is unverified.
"""

from collections.abc import Collection
from collections.abc import Iterable
from collections.abc import Mapping
import dataclasses
import hashlib
import itertools
import json
from typing import Any
from typing import Self

from scmle.contracts import evidence
from scmle.contracts import qualification

FORMAT_VERSION = 1
STATUS = "hypothesis"
MAX_SEARCH_CANDIDATES = 16

Edge = tuple[str, str]


def children_map(edges: Iterable[Edge]) -> dict[str, set[str]]:
    """Maps each node to the set of its children.

    Args:
        edges: Directed edges (parent, child).

    Returns:
        A dict from parent to the set of its children. Nodes without
        children are absent.
    """
    out: dict[str, set[str]] = {}
    for parent, child in edges:
        out.setdefault(parent, set()).add(child)
    return out


def parents_map(edges: Iterable[Edge]) -> dict[str, set[str]]:
    """Maps each node to the set of its parents.

    Args:
        edges: Directed edges (parent, child).

    Returns:
        A dict from child to the set of its parents. Nodes without parents
        are absent.
    """
    out: dict[str, set[str]] = {}
    for parent, child in edges:
        out.setdefault(child, set()).add(parent)
    return out


def reach(start: Collection[str], step: Mapping[str, set[str]]) -> set[str]:
    """Finds the nodes reachable by following a step relation.

    Args:
        start: Nodes to start from.
        step: Mapping from a node to the nodes one step away.

    Returns:
        Nodes reachable from `start` by one or more steps. A start node is
        included only if a cycle leads back to it.
    """
    seen: set[str] = set()
    frontier = list(start)
    while frontier:
        node = frontier.pop()
        for nxt in step.get(node, ()):
            if nxt not in seen:
                seen.add(nxt)
                frontier.append(nxt)
    return seen


def has_cycle(nodes: Collection[str], edges: Iterable[Edge]) -> bool:
    """Tests whether a directed graph contains a cycle.

    Args:
        nodes: All nodes.
        edges: Directed edges (parent, child).

    Returns:
        True if some node can reach itself by one or more edges.
    """
    kids = children_map(edges)
    return any(n in reach([n], kids) for n in nodes)


def separated(
    nodes: Collection[str],
    edges: Collection[Edge],
    xs: Collection[str],
    ys: Collection[str],
    zs: Collection[str],
) -> bool:
    """Tests d-separation of `xs` and `ys` given `zs` in a DAG.

    Moralises the subgraph induced by the ancestors of the three sets, deletes
    `zs`, and asks whether `xs` still reaches `ys`.

    Args:
        nodes: All nodes.
        edges: Directed edges (parent, child).
        xs: First set.
        ys: Second set.
        zs: Conditioning set.

    Returns:
        True when every path between `xs` and `ys` is blocked by `zs`.

    Raises:
        ValueError: If the three sets overlap or name unknown nodes.
    """
    sets = [set(xs), set(ys), set(zs)]
    if not set().union(*sets) <= set(nodes):
        raise ValueError("sets name nodes that are not in the graph")
    if (sets[0] & sets[1]) or (sets[0] & sets[2]) or (sets[1] & sets[2]):
        raise ValueError("the three sets must be disjoint")
    parents = parents_map(edges)
    keep = set().union(*sets)
    keep |= reach(keep, parents)
    adjacent: dict[str, set[str]] = {n: set() for n in keep}
    for child in keep:
        mine = sorted(parents.get(child, set()) & keep)
        for parent in mine:
            adjacent[parent].add(child)
            adjacent[child].add(parent)
        for a, b in itertools.combinations(mine, 2):
            adjacent[a].add(b)
            adjacent[b].add(a)
    blocked = sets[2]
    seen = set(sets[0])
    frontier = list(sets[0])
    while frontier:
        node = frontier.pop()
        for nxt in adjacent[node]:
            if nxt in blocked or nxt in seen:
                continue
            if nxt in sets[1]:
                return False
            seen.add(nxt)
            frontier.append(nxt)
    return True


@dataclasses.dataclass(frozen=True, slots=True)
class Dag:
    """A directed acyclic graph offered as a hypothesis.

    Attributes:
        nodes: Variable names, sorted. Latent variables are listed too.
        edges: Directed edges (parent, child), sorted.
        latent: Nodes that are not columns of the data.
        source: Who or what drafted the graph.
        prompt: The prompt or brief behind a draft, or "none".
    """

    nodes: tuple[str, ...]
    edges: tuple[Edge, ...]
    latent: tuple[str, ...]
    source: str
    prompt: str

    def __post_init__(self) -> None:
        """Validates the graph.

        Raises:
            ValueError: If `nodes`, `latent` or `edges` is not a sorted,
                duplicate-free tuple, a node name is empty, an edge is not
                a pair of known nodes or is a self loop, a latent node is
                not a node, the graph has a cycle, or `source` or `prompt`
                is empty.
        """
        for name, values in (("nodes", self.nodes), ("latent", self.latent)):
            qualification.require_tuple(values, name)
            for item in values:
                evidence.require_text(item, name)
            if list(values) != sorted(set(values)):
                raise ValueError(f"{name} must be sorted and unique")
        qualification.require_tuple(self.edges, "edges")
        if list(self.edges) != sorted(set(self.edges)):
            raise ValueError("edges must be sorted and unique")
        names = set(self.nodes)
        for edge in self.edges:
            if len(edge) != 2 or not set(edge) <= names:
                raise ValueError(f"edge {edge} names an unknown node")
            if edge[0] == edge[1]:
                raise ValueError("self loops are not allowed")
        if not set(self.latent) <= names:
            raise ValueError("latent nodes must be nodes")
        if has_cycle(self.nodes, self.edges):
            raise ValueError("the graph has a cycle")
        evidence.require_text(self.source, "source")
        evidence.require_text(self.prompt, "prompt")

    @classmethod
    def build(
        cls,
        edges: Iterable[Edge],
        source: str,
        prompt: str = "none",
        latent: Iterable[str] = (),
        extra_nodes: Iterable[str] = (),
    ) -> Self:
        """Builds a graph, sorting and de-duplicating its parts.

        Args:
            edges: Directed edges (parent, child).
            source: Who or what drafted the graph.
            prompt: Prompt or brief of an LLM draft, or "none".
            latent: Unobserved nodes.
            extra_nodes: Nodes with no edges.

        Returns:
            The graph. Its nodes are those of the edges, `latent` and
            `extra_nodes`, converted to strings. A graph the constructor
            rejects (self loop, cycle, empty name, `source` or `prompt`)
            raises its `ValueError` here.
        """
        pairs = sorted({(str(a), str(b)) for a, b in edges})
        names = {n for pair in pairs for n in pair}
        names |= {str(n) for n in extra_nodes} | {str(n) for n in latent}
        return cls(
            tuple(sorted(names)),
            tuple(pairs),
            tuple(sorted({str(n) for n in latent})),
            source,
            prompt,
        )

    @property
    def status(self) -> str:
        """Always `hypothesis`: the graph is never confirmed here."""
        return STATUS

    @property
    def observed(self) -> tuple[str, ...]:
        """Nodes that are columns of the data."""
        return tuple(n for n in self.nodes if n not in self.latent)

    def parents(self, node: str) -> set[str]:
        """Finds the direct causes of a node.

        Args:
            node: The node.

        Returns:
            The set of its parents; empty if it has none.
        """
        return parents_map(self.edges).get(node, set())

    def children(self, node: str) -> set[str]:
        """Finds the direct effects of a node.

        Args:
            node: The node.

        Returns:
            The set of its children; empty if it has none.
        """
        return children_map(self.edges).get(node, set())

    def ancestors(self, node: str) -> set[str]:
        """Finds the nodes with a directed path to a node.

        Args:
            node: The node.

        Returns:
            The set of its ancestors, not including `node`.
        """
        return reach([node], parents_map(self.edges))

    def descendants(self, node: str) -> set[str]:
        """Finds the nodes reachable from a node by a directed path.

        Args:
            node: The node.

        Returns:
            The set of its descendants, not including `node`.
        """
        return reach([node], children_map(self.edges))

    def d_separated(
        self,
        xs: Collection[str],
        ys: Collection[str],
        zs: Collection[str] = (),
    ) -> bool:
        """Tests whether `zs` d-separates `xs` and `ys`.

        Args:
            xs: First set of nodes.
            ys: Second set of nodes.
            zs: Conditioning set.

        Returns:
            True when every path between `xs` and `ys` is blocked by `zs`.
            Overlapping sets and unknown nodes are rejected with a
            `ValueError` from `separated`.
        """
        return separated(self.nodes, self.edges, xs, ys, zs)

    def backdoor_valid(
        self, treatment: str, outcome: str, adjustment: Collection[str]
    ) -> bool:
        """Tests whether `adjustment` satisfies the back-door criterion.

        Args:
            treatment: The treatment node.
            outcome: The outcome node.
            adjustment: Candidate adjustment set.

        Returns:
            True when the set holds no descendant of the treatment (and not
            the treatment or outcome) and blocks every back-door path.

        Raises:
            ValueError: If a node is unknown or `treatment` equals
                `outcome`.
        """
        chosen = set(adjustment)
        self.require_nodes([treatment, outcome, *chosen])
        if treatment == outcome:
            raise ValueError("treatment and outcome must differ")
        if chosen & ({treatment, outcome} | self.descendants(treatment)):
            return False
        cut = tuple(e for e in self.edges if e[0] != treatment)
        return separated(self.nodes, cut, [treatment], [outcome], chosen)

    def require_nodes(self, names: Iterable[str]) -> None:
        """Checks that every name is a node of the graph.

        Args:
            names: Names to check.

        Raises:
            ValueError: If any name is not a node.
        """
        unknown = sorted(set(names) - set(self.nodes))
        if unknown:
            raise ValueError(f"not nodes of the graph: {unknown}")

    def candidates(self, treatment: str, outcome: str) -> list[str]:
        """Lists the observed nodes that may be adjusted for.

        Args:
            treatment: The treatment node.
            outcome: The outcome node.

        Returns:
            Observed nodes other than the treatment, the outcome and the
            descendants of the treatment, in node order.
        """
        banned = {treatment, outcome} | self.descendants(treatment)
        return [n for n in self.observed if n not in banned]

    def adjustment_set(
        self, treatment: str, outcome: str
    ) -> tuple[str, ...] | None:
        """Finds the smallest valid back-door set of observed nodes, if any.

        Sets are tried by size, then in sorted order, so the answer is
        deterministic.

        Args:
            treatment: The treatment node.
            outcome: The outcome node.

        Returns:
            The set, possibly empty, or None when no observed set blocks the
            back-door paths.

        Raises:
            ValueError: If more than `MAX_SEARCH_CANDIDATES` candidates make
                the search too large, a node is unknown, or `treatment`
                equals `outcome`.
        """
        self.require_nodes([treatment, outcome])
        pool = self.candidates(treatment, outcome)
        if len(pool) > MAX_SEARCH_CANDIDATES:
            raise ValueError(
                f"more than {MAX_SEARCH_CANDIDATES} candidate covariates"
            )
        for size in range(len(pool) + 1):
            for subset in itertools.combinations(pool, size):
                if self.backdoor_valid(treatment, outcome, subset):
                    return subset
        return None

    def to_dict(self) -> dict[str, Any]:
        """Converts the graph to a JSON-ready record, without the hash.

        Returns:
            A dict with the format version, status, nodes, edges (as
            lists), latent nodes, source and prompt.
        """
        return {
            "format_version": FORMAT_VERSION,
            "status": STATUS,
            "nodes": list(self.nodes),
            "edges": [list(e) for e in self.edges],
            "latent": list(self.latent),
            "source": self.source,
            "prompt": self.prompt,
        }

    def digest(self) -> str:
        """Hashes the graph; any edit changes the hash.

        Returns:
            The hex SHA-256 of the record's JSON with sorted keys and no
            whitespace.
        """
        text = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def artifact(self) -> dict[str, Any]:
        """Builds the stored DAG document.

        Returns:
            The record from `to_dict` plus its `hash`.
        """
        return {**self.to_dict(), "hash": self.digest()}

    @classmethod
    def from_artifact(cls, document: Mapping[str, Any]) -> Self:
        """Rebuilds a graph and verifies its stored hash and status.

        Args:
            document: A mapping in the form `artifact` returns.

        Returns:
            The graph.

        Raises:
            ValueError: On another status, version or field set, a hash
                that does not match the content, or content that is not a
                valid graph.
        """
        names = {
            "format_version",
            "status",
            "nodes",
            "edges",
            "latent",
            "source",
            "prompt",
            "hash",
        }
        if set(document) != names:
            raise ValueError("unexpected or missing field(s) in dag artifact")
        if document["format_version"] != FORMAT_VERSION:
            raise ValueError("unsupported format_version")
        if document["status"] != STATUS:
            raise ValueError("a DAG can only be stored as a hypothesis")
        graph = cls(
            tuple(document["nodes"]),
            tuple((str(a), str(b)) for a, b in document["edges"]),
            tuple(document["latent"]),
            document["source"],
            document["prompt"],
        )
        if graph.digest() != document["hash"]:
            raise ValueError("hash does not match the DAG content")
        return graph
