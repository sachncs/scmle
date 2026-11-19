"""A9: the document and graph splits never let related rows straddle.

Properties over generated data, for the splitter each kind states
(`Kind.constrain`) applied to the rows it reads:

- documents: no declared group, no exact duplicate and no near duplicate
  (one word of eighty edited) lies on both sides of the test split or of
  any cross-validation fold;
- graphs: no node is trained on that is first seen after a held-out node,
  in the test split and in every fold.
"""

from datetime import datetime
from datetime import timedelta
from datetime import UTC

from hypothesis import given
from hypothesis import HealthCheck
from hypothesis import settings
from hypothesis import strategies
import numpy
import pandas

from scmle.contracts import tasks
from scmle.pipeline.kinds import document
from scmle.pipeline.kinds import graph
from scmle.pipeline.kinds import rows

VOCABULARY = tuple(f"term{index}" for index in range(500))
EXAMPLES = settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)


def sides(manifest, rows):
    """Each held-out side of a split as sets of row positions."""
    train = numpy.asarray(manifest.training)
    found = [set(manifest.test)]
    found += [set(train[list(fold[1])].tolist()) for fold in manifest.folds]
    return [side & rows for side in found]


def together(manifest, rows):
    """Whether rows are wholly in or wholly out of every held-out side."""
    return all(not side or side == rows for side in sides(manifest, rows))


@EXAMPLES
@given(
    seed=strategies.integers(0, 10_000),
    count=strategies.integers(70, 100),
    copies=strategies.integers(1, 6),
    groups=strategies.integers(0, 8),
)
def test_documents_keep_groups_and_near_duplicates_on_one_side(
    tmp_path_factory, seed, count, copies, groups
):
    generator = numpy.random.default_rng(seed)
    drawn = generator.choice(VOCABULARY, (count, 80))
    texts = [" ".join(words) for words in drawn]
    twins = []
    for index in range(copies):
        words = texts[index].split()
        if index % 2:
            words[int(generator.integers(0, 80))] = "edited"
        texts.append(" ".join(words))
        twins.append({index, len(texts) - 1})
    labels = [position % 2 for position in range(count)]
    labels += [labels[index] for index in range(copies)]
    teams = [
        f"t{position % groups}" if groups and position < 2 * groups else None
        for position in range(len(texts))
    ]
    path = tmp_path_factory.mktemp("listing") / "listing.csv"
    pandas.DataFrame({"text": texts, "label": labels, "team": teams}).to_csv(
        path, index=False
    )
    spec = tasks.Config(
        data=tasks.Dataset(
            uri=str(path),
            target="label",
            modality="document",
            group_column="team" if groups else None,
        ),
        validation=tasks.Validation(seed=seed),
    )
    frame = document.DOCUMENT.read(spec)[0]
    split = document.DOCUMENT.constrain(spec, tasks.Task.BINARY).split(frame)
    assert rows.KEY not in split.features.columns
    for pair in twins:
        assert together(split.manifest, pair)
    for team in {name for name in teams if name}:
        members = {i for i, name in enumerate(teams) if name == team}
        assert together(split.manifest, members)


@EXAMPLES
@given(
    seed=strategies.integers(0, 10_000),
    count=strategies.integers(60, 100),
    ties=strategies.integers(1, 4),
)
def test_graphs_never_train_on_a_node_seen_after_a_held_out_one(
    tmp_path_factory, seed, count, ties
):
    generator = numpy.random.default_rng(seed)
    root = tmp_path_factory.mktemp("graph")
    order = generator.permutation(count)
    rank = numpy.argsort(order)
    pandas.DataFrame(
        {
            "node": [f"n{index}" for index in range(count)],
            "x": generator.normal(size=count),
            "label": [int(rank[index]) % 2 for index in range(count)],
        }
    ).to_csv(root / "nodes.csv", index=False)
    start = datetime(2026, 1, 1, tzinfo=UTC)
    edges = [
        {
            "source": f"n{order[position]}",
            "target": f"n{order[int(generator.integers(0, position))]}",
            "time": (start + timedelta(hours=position // ties)).isoformat(),
        }
        for position in range(1, count)
    ]
    edges.insert(
        0,
        {
            "source": f"n{order[0]}",
            "target": f"n{order[1]}",
            "time": start.isoformat(),
        },
    )
    pandas.DataFrame(edges).to_csv(root / "edges.csv", index=False)
    spec = tasks.Config(
        data=tasks.Dataset(uri=str(root), target="label", modality="graph"),
        validation=tasks.Validation(seed=seed),
    )
    frame = graph.GRAPH.read(spec)[0]
    seen = frame[rows.KEY].to_numpy()
    split = graph.GRAPH.constrain(spec, tasks.Task.BINARY).split(frame)
    train = numpy.asarray(split.manifest.training)
    assert seen[train].max() < seen[list(split.manifest.test)].min()
    for fit, check in split.manifest.folds:
        assert seen[train[list(fit)]].max() < seen[train[list(check)]].min()
