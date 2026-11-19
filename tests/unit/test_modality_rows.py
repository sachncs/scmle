"""Register rows closed in the document and graph data foundations.

Each test names its row and failed before the fix it pins:

- MM-1: the email pattern was quadratic on long dotted runs without `@`.
- MM-7: CSV edge files read `007` as `7` and `NA` as missing.
- MM-8: group splits took fractions of groups with no label stratification.
- MM-10: exact and near-duplicate documents straddled train and test.
- MM-11: the graph node split ignored event time.
- MM-16: duplicate document ids collapsed, hiding conflicting labels.
- MM-22: one group split serves every modality (`cohorts.divide`).
"""

from datetime import datetime
from datetime import timedelta
from datetime import UTC
import json
import time

import numpy
import pytest

from scmle.data.modalities import cohorts
from scmle.data.modalities import curation
from scmle.data.modalities import document
from scmle.data.modalities import graph
from scmle.data.modalities.document import likeness

EMAIL_BUDGET = 0.1
"""Seconds the email pattern may take on the 80 KB adversarial run (MM-1)."""


def test_mm1_the_email_pattern_is_linear_on_dotted_runs_without_at():
    content = "a." * 40_000
    started = time.perf_counter()
    found = document.pii(content)
    assert time.perf_counter() - started < EMAIL_BUDGET
    assert found == {}
    assert document.pii("write to ann.lee@example.org today") == {"email": 1}


def test_mm7_csv_edge_ids_keep_their_text(tmp_path):
    rows = [
        {"source": "007", "target": "NA"},
        {"source": "NA", "target": "null"},
    ]
    csv = tmp_path / "edges.csv"
    csv.write_text("source,target\n007,NA\nNA,null\n", encoding="utf-8")
    jsonl = tmp_path / "edges.jsonl"
    jsonl.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    from_csv = graph.edgelist(csv)
    assert [(edge.source, edge.target) for edge in from_csv] == [
        ("007", "NA"),
        ("NA", "null"),
    ]
    assert from_csv == graph.edgelist(jsonl)
    assert (
        graph.Dataset.from_edge_file(csv).identity
        == graph.Dataset.from_edge_file(jsonl).identity
    )


def test_mm8_a_stratified_group_split_holds_out_every_class():
    cohorts = [f"g{index}" for index in range(30)]
    strata = {
        name: ("rare" if index < 4 else "common")
        for index, name in enumerate(cohorts)
    }
    for seed in range(25):
        validation, test = curation.allocate(cohorts, 0.2, 0.2, seed, strata)
        assert {strata[name] for name in test} == {"rare", "common"}
        assert {strata[name] for name in validation} == {"rare", "common"}
        train = set(cohorts) - validation - test
        assert {strata[name] for name in train} == {"rare", "common"}


def test_mm8_a_class_with_one_group_stays_in_training():
    strata = {"a": "x", "b": "y", "c": "y", "d": "y", "e": "y"}
    for seed in range(10):
        validation, test = curation.allocate(strata, 0.4, 0.4, seed, strata)
        assert "a" not in validation | test


def test_mm8_an_unstratified_allocation_is_unchanged():
    cohorts = [f"g{index}" for index in range(10)]
    assert curation.allocate(cohorts, 0.2, 0.2, 3) == (
        {"g6", "g0"},
        {"g5", "g1"},
    )
    assert curation.allocate(cohorts, 0.1, 0.3, 8) == (
        {"g4"},
        {"g2", "g9", "g8"},
    )


def test_mm22_one_group_split_keeps_groups_whole_and_strata_present():
    samples = [
        (f"s{index}", f"g{index // 2}", index % 3) for index in range(60)
    ]
    train, validation, test = cohorts.divide(
        samples,
        [sample[1] for sample in samples],
        [sample[2] % 2 for sample in samples],
        0.2,
        0.2,
        5,
    )
    assert len(train) + len(validation) + len(test) == len(samples)
    sides = [
        {sample[1] for sample in part} for part in (train, validation, test)
    ]
    assert not sides[0] & sides[1]
    assert not sides[0] & sides[2]
    assert not sides[1] & sides[2]
    with pytest.raises(ValueError, match="test_fraction"):
        cohorts.divide(samples, ["g"] * 60, [0] * 60, 0.2, 1.2, 0)


VOCABULARY = tuple(f"word{index}" for index in range(400))


def prose(seed, count=80):
    """A text of `count` words drawn from the vocabulary."""
    generator = numpy.random.default_rng(seed)
    return " ".join(generator.choice(VOCABULARY, count))


def corpus():
    samples = [
        document.Sample(f"doc-{index}", prose(index), label=index % 2)
        for index in range(40)
    ]
    words = samples[3].content.split()
    words[40] = "edited"
    samples.append(document.Sample("copy-3", " ".join(words), label=1))
    samples.append(document.Sample("copy-7", samples[7].content, label=1))
    return document.Dataset(tuple(samples))


def test_mm10_near_duplicate_documents_share_a_side_of_the_split():
    dataset = corpus()
    resemblance = likeness.Resemblance()
    keys = resemblance.cluster([sample.content for sample in dataset.samples])
    by_id = {
        sample.document_id: key
        for sample, key in zip(dataset.samples, keys, strict=True)
    }
    assert by_id["copy-7"] == by_id["doc-7"]
    assert by_id["copy-3"] == by_id["doc-3"]
    for seed in range(20):
        split = dataset.split(0.2, 0.2, seed, resemblance)
        ids = split.document_ids()
        for twin, original in (("copy-7", "doc-7"), ("copy-3", "doc-3")):
            side = [name for name, part in ids.items() if original in part]
            assert twin in ids[side[0]]


def test_mm10_unrelated_documents_are_not_merged():
    texts = [
        "apples and pears grow in the orchard behind the house",
        "the train to the coast leaves at nine every morning",
        "quantum chromodynamics describes the strong interaction",
    ]
    assert len(set(likeness.Resemblance().cluster(texts))) == 3


def test_mm16_duplicate_document_ids_with_conflicting_labels_are_found():
    dataset = document.Dataset(
        (
            document.Sample("same", "identical words", label="x"),
            document.Sample("same", "identical words", label="y"),
            document.Sample("other", "different words", label="x"),
        )
    )
    found = {finding.identifier for finding in dataset.curate().risk_findings}
    assert "risk.conflicting-duplicate-labels" in found


def stamp(hours):
    return datetime(2026, 1, 1, tzinfo=UTC) + timedelta(hours=hours)


def test_mm11_a_timed_graph_holds_out_the_latest_nodes():
    edges = [
        graph.Edge(f"n{index}", f"n{index + 1}", event_time=stamp(index))
        for index in range(20)
    ]
    timed = graph.Dataset.connect(edges)
    for seed in range(5):
        part = graph.induct(timed, 0.6, 0.2, 0.2, seed=seed)
        seen = graph.debut(timed)
        latest_train = max(seen[node] for node in part.train_nodes)
        assert all(seen[node] >= latest_train for node in part.test_nodes)
        assert all(seen[node] >= latest_train for node in part.validation_nodes)


def test_mm11_a_partly_timed_graph_is_refused_for_the_node_split():
    edges = [
        graph.Edge("a", "b", event_time=stamp(1)),
        graph.Edge("b", "c"),
        graph.Edge("c", "d", event_time=stamp(2)),
    ]
    with pytest.raises(graph.TemporalError, match="partly timed"):
        graph.induct(graph.Dataset.connect(edges), 0.5, 0.25, 0.25)
