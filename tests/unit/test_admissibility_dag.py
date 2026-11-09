"""The DAG hypothesis, d-separation and back-door criteria vs brute force."""

import itertools
import json

import numpy
import pytest

from scmle.data.admissibility import dag


def simple_paths(edges, nodes, start, end):
    """All simple undirected paths, as lists of nodes."""
    nbr = {n: set() for n in nodes}
    for a, b in edges:
        nbr[a].add(b)
        nbr[b].add(a)
    out = []

    def walk(path):
        last = path[-1]
        if last == end:
            out.append(list(path))
            return
        for nxt in sorted(nbr[last]):
            if nxt not in path:
                walk([*path, nxt])

    walk([start])
    return out


def descendants(edges, node):
    seen, todo = set(), [node]
    while todo:
        cur = todo.pop()
        for a, b in edges:
            if a == cur and b not in seen:
                seen.add(b)
                todo.append(b)
    return seen


def path_blocked(path, edges, z):
    """Textbook rule: a collider blocks unless it or a descendant is in z."""
    eset = set(edges)
    for i in range(1, len(path) - 1):
        a, m, b = path[i - 1], path[i], path[i + 1]
        collider = (a, m) in eset and (b, m) in eset
        if collider:
            if m not in z and not (descendants(edges, m) & z):
                return True
        elif m in z:
            return True
    return False


def brute_dsep(nodes, edges, x, y, z):
    return all(
        path_blocked(p, edges, set(z)) for p in simple_paths(edges, nodes, x, y)
    )


def brute_backdoor(nodes, edges, t, y, z):
    z = set(z)
    if z & ({t, y} | descendants(edges, t)):
        return False
    eset = set(edges)
    for p in simple_paths(edges, nodes, t, y):
        into_t = len(p) > 1 and (p[1], t) in eset
        if into_t and not path_blocked(p, edges, z):
            return False
    return True


def random_dag(seed, n=6, density=0.35):
    rng = numpy.random.default_rng(seed)
    names = [f"v{i}" for i in range(n)]
    order = list(rng.permutation(n))
    edges = [
        (names[order[i]], names[order[j]])
        for i in range(n)
        for j in range(i + 1, n)
        if rng.random() < density
    ]
    return names, edges


@pytest.mark.parametrize("seed", range(60))
def test_d_separation_matches_path_enumeration(seed):
    names, edges = random_dag(seed)
    graph = dag.Dag.build(edges, "test", extra_nodes=names)
    for x, y in itertools.combinations(names, 2):
        others = [n for n in names if n not in (x, y)]
        for size in range(len(others) + 1):
            for z in itertools.combinations(others, size):
                assert dag.separated(
                    graph.nodes, graph.edges, [x], [y], z
                ) == brute_dsep(names, graph.edges, x, y, z), (edges, x, y, z)


@pytest.mark.parametrize("seed", range(60))
def test_backdoor_criterion_matches_the_definition(seed):
    names, edges = random_dag(seed, n=6, density=0.45)
    graph = dag.Dag.build(edges, "test", extra_nodes=names)
    t, y = names[0], names[1]
    others = [n for n in names if n not in (t, y)]
    for size in range(len(others) + 1):
        for z in itertools.combinations(others, size):
            assert graph.backdoor_valid(t, y, z) == brute_backdoor(
                names, graph.edges, t, y, z
            ), (edges, z)


def test_textbook_confounder_mediator_collider():
    # C -> T, C -> Y, T -> M -> Y, T -> K <- Y
    graph = dag.Dag.build(
        [
            ("C", "T"),
            ("C", "Y"),
            ("T", "M"),
            ("M", "Y"),
            ("T", "K"),
            ("Y", "K"),
        ],
        "expert",
    )
    assert graph.adjustment_set("T", "Y") == ("C",)
    assert graph.backdoor_valid("T", "Y", {"C"})
    assert not graph.backdoor_valid("T", "Y", set())  # confounded
    assert not graph.backdoor_valid("T", "Y", {"C", "M"})  # mediator
    assert not graph.backdoor_valid("T", "Y", {"C", "K"})  # descendant
    assert graph.candidates("T", "Y") == ["C"]


def test_m_bias_collider_is_not_adjusted_for_and_empty_set_is_valid():
    graph = dag.Dag.build(
        [("U1", "T"), ("U1", "Z"), ("U2", "Z"), ("U2", "Y"), ("T", "Y")],
        "expert",
        latent=["U1", "U2"],
    )
    assert graph.adjustment_set("T", "Y") == ()
    assert not graph.backdoor_valid("T", "Y", {"Z"})
    assert graph.backdoor_valid("T", "Y", set())


def test_no_adjustment_set_when_the_confounder_is_latent():
    graph = dag.Dag.build(
        [("U", "T"), ("U", "Y"), ("T", "Y")], "expert", latent=["U"]
    )
    assert graph.adjustment_set("T", "Y") is None


def test_search_is_bounded():
    count = dag.MAX_SEARCH_CANDIDATES + 1
    edges = [(f"c{i}", "Y") for i in range(count)]
    graph = dag.Dag.build([*edges, ("T", "Y")], "expert")
    with pytest.raises(ValueError, match="candidate"):
        graph.adjustment_set("T", "Y")


def test_ancestors_descendants_parents_children():
    graph = dag.Dag.build([("a", "b"), ("b", "c"), ("a", "c")], "t")
    assert graph.parents("c") == {"a", "b"}
    assert graph.children("a") == {"b", "c"}
    assert graph.descendants("a") == {"b", "c"}
    assert graph.ancestors("c") == {"a", "b"}
    assert graph.descendants("c") == set()
    assert graph.status == "hypothesis"


def test_validation_rejects_cycles_loops_and_unsorted_input():
    with pytest.raises(ValueError, match="cycle"):
        dag.Dag.build([("a", "b"), ("b", "a")], "t")
    with pytest.raises(ValueError, match="self loops"):
        dag.Dag.build([("a", "a")], "t")
    with pytest.raises(ValueError, match="sorted"):
        dag.Dag(("b", "a"), (), (), "t", "none")
    with pytest.raises(ValueError, match="unknown node"):
        dag.Dag(("a",), (("a", "z"),), (), "t", "none")
    with pytest.raises(ValueError, match="latent"):
        dag.Dag(("a",), (), ("q",), "t", "none")
    with pytest.raises(ValueError, match="source"):
        dag.Dag.build([("a", "b")], " ")


def test_separation_rejects_overlapping_or_unknown_sets():
    graph = dag.Dag.build([("a", "b")], "t")
    with pytest.raises(ValueError, match="disjoint"):
        dag.separated(graph.nodes, graph.edges, ["a"], ["a"], [])
    with pytest.raises(ValueError, match="not in the graph"):
        dag.separated(graph.nodes, graph.edges, ["a"], ["q"], [])
    with pytest.raises(ValueError, match="differ"):
        graph.backdoor_valid("a", "a", [])
    with pytest.raises(ValueError, match="not nodes"):
        graph.backdoor_valid("a", "b", ["q"])


def test_hash_is_stable_and_every_field_changes_it():
    base = dag.Dag.build([("a", "b")], "expert", "none")
    assert (
        base.digest() == dag.Dag.build([("a", "b")], "expert", "none").digest()
    )
    assert len(base.digest()) == 64
    variants = [
        dag.Dag.build([("a", "b"), ("b", "c")], "expert", "none"),
        dag.Dag.build([("b", "a")], "expert", "none"),
        dag.Dag.build([("a", "b")], "llm", "none"),
        dag.Dag.build([("a", "b")], "expert", "draft a DAG"),
        dag.Dag.build([("a", "b")], "expert", "none", latent=["a"]),
        dag.Dag.build([("a", "b")], "expert", "none", extra_nodes=["z"]),
    ]
    digests = {v.digest() for v in variants} | {base.digest()}
    assert len(digests) == len(variants) + 1


def test_artifact_round_trips_and_detects_edits():
    graph = dag.Dag.build([("a", "b")], "llm draft", "prompt text")
    doc = json.loads(json.dumps(graph.artifact()))
    assert doc["status"] == "hypothesis"
    assert doc["hash"] == graph.digest()
    assert dag.Dag.from_artifact(doc) == graph
    edited = {**doc, "edges": [["a", "b"], ["b", "a"]]}
    with pytest.raises(ValueError):
        dag.Dag.from_artifact(edited)
    tampered = {**doc, "source": "someone else"}
    with pytest.raises(ValueError, match="hash"):
        dag.Dag.from_artifact(tampered)
    with pytest.raises(ValueError, match="hypothesis"):
        dag.Dag.from_artifact({**doc, "status": "confirmed"})
    with pytest.raises(ValueError, match="unexpected or missing"):
        dag.Dag.from_artifact({k: v for k, v in doc.items() if k != "prompt"})
    with pytest.raises(ValueError, match="format_version"):
        dag.Dag.from_artifact({**doc, "format_version": 2})


def test_a_search_of_exactly_the_maximum_size_is_allowed():
    edges = [(f"c{i}", "Y") for i in range(dag.MAX_SEARCH_CANDIDATES)]
    graph = dag.Dag.build([*edges, ("T", "Y")], "expert")
    assert len(graph.candidates("T", "Y")) == dag.MAX_SEARCH_CANDIDATES
    assert graph.adjustment_set("T", "Y") == ()
