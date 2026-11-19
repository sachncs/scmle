"""A9 split leakage for media: no copy, edited copy or group on both sides.

Rows are drawn as pictures (random thumbnails) and copies of them: exact
copies and brightness/contrast edits, which keep a picture's structure. A
row's family is the picture it was copied from, joined with the group the
data declares. Whatever the protocol, every family must sit wholly on one
side of the held-out split and of every fold. The families are known from
how the rows were drawn, not from the linker under test.
"""

import hypothesis
from hypothesis import strategies
import numpy
import pandas
import pytest

from scmle.contracts import errors
from scmle.contracts import tasks
from scmle.data.modalities.image import descriptor
from scmle.pipeline.kinds import image
from scmle.pipeline.kinds import rows

NAMES = [name for name in descriptor.names() if name.startswith("thumb_")]


def draw(seed: int, pictures: int, copies: int, groups: int):
    """Rows of pictures and copies, their family and declared group."""
    rng = numpy.random.default_rng(seed)
    base = rng.uniform(0.1, 0.8, (pictures, 64, 3))
    parents = numpy.concatenate(
        [numpy.arange(pictures), rng.integers(0, pictures, copies)]
    )
    scale = numpy.concatenate(
        [numpy.ones(pictures), rng.uniform(0.8, 1.0, copies)]
    )
    shift = numpy.concatenate(
        [numpy.zeros(pictures), rng.uniform(0.0, 0.1, copies)]
    )
    drawn = base[parents] * scale[:, None, None] + shift[:, None, None]
    frame = pandas.DataFrame(
        drawn.transpose(0, 2, 1).reshape(len(drawn), -1), columns=NAMES
    )
    frame["group"] = [
        f"g{index % groups}" for index in rng.permutation(len(drawn))
    ]
    frame["y"] = (parents % 2).astype(float) + rng.normal(0, 0.1, len(drawn))
    frame["when"] = rng.permutation(len(drawn)).astype(float)
    return frame, parents


def families(
    parents: numpy.ndarray, groups: numpy.ndarray | None
) -> numpy.ndarray:
    """The family of each row: its picture, joined with its declared group."""
    owner = {int(parent): int(parent) for parent in parents}

    def root(node: int) -> int:
        while owner[node] != node:
            node = owner[node]
        return node

    if groups is not None:
        first: dict[str, int] = {}
        for parent, group in zip(parents, groups, strict=True):
            other = first.setdefault(group, int(parent))
            owner[root(int(parent))] = root(other)
    return numpy.array([root(int(parent)) for parent in parents])


def divide(spec: tasks.Config, frame: pandas.DataFrame):
    """Links the rows as the image kind's reader does, then splits them."""
    linked = frame.assign(
        **{rows.KEY: rows.unite(len(frame), image.IMAGE.pairs([], frame))}
    )
    return image.IMAGE.constrain(spec, tasks.Task.REGRESSION).split(linked)


def straddles(family: numpy.ndarray, left, right) -> set[int]:
    """Families with rows on both sides."""
    return set(family[numpy.asarray(left, dtype=int)]) & set(
        family[numpy.asarray(right, dtype=int)]
    )


@hypothesis.settings(max_examples=40, deadline=None)
@hypothesis.given(
    seed=strategies.integers(0, 2**31),
    pictures=strategies.integers(12, 40),
    copies=strategies.integers(0, 30),
    declared=strategies.booleans(),
    strategy=strategies.sampled_from(["auto", "random", "group", "temporal"]),
    entities=strategies.sampled_from(["known", "unseen"]),
)
def test_no_copy_edited_copy_or_group_is_on_both_sides(
    seed, pictures, copies, declared, strategy, entities
):
    hypothesis.assume(declared or strategy != "group")
    frame, parents = draw(seed, pictures, copies, max(8, pictures // 2))
    data = tasks.Dataset(
        uri="<in-memory>",
        target="y",
        group_column="group" if declared else None,
        time_column="when" if strategy == "temporal" else None,
    )
    validation = tasks.Validation(
        strategy=strategy,
        folds=3,
        seed=seed % 1000,
        entities=entities if declared else "known",
    )
    spec = tasks.Config(data=data, validation=validation)
    try:
        split = divide(spec, frame)
    except errors.BlockedError:
        hypothesis.reject()
    timed = strategy == "temporal"
    joined = declared and (not timed or entities == "unseen")
    family = families(parents, frame["group"].to_numpy() if joined else None)
    manifest = split.manifest
    assert not straddles(family, manifest.training, manifest.test)
    development = numpy.asarray(manifest.training)
    for fit, check in manifest.folds:
        assert not straddles(
            family, development[list(fit)], development[list(check)]
        )
    assert rows.KEY not in split.features.columns
    assert rows.KEY not in split.test.features.columns


def test_without_cohorts_the_declared_protocol_applies_unchanged():
    frame = draw(7, 40, 0, 40)[0]
    spec = tasks.Config(
        data=tasks.Dataset(uri="<in-memory>", target="y"),
        validation=tasks.Validation(folds=3),
    )
    split = divide(spec, frame)
    assert split.manifest.strategy == "random"
    assert split.manifest.notes[-1].startswith("0 cohorts")


def test_with_cohorts_a_random_protocol_becomes_a_group_split():
    frame = draw(7, 30, 10, 30)[0]
    spec = tasks.Config(
        data=tasks.Dataset(uri="<in-memory>", target="y"),
        validation=tasks.Validation(strategy="random", folds=3),
    )
    split = divide(spec, frame)
    assert split.manifest.strategy == "group"


def test_too_few_cohorts_for_the_folds_is_blocked():
    frame = draw(3, 12, 30, 12)[0]
    frame["group"] = "one"
    spec = tasks.Config(
        data=tasks.Dataset(uri="<in-memory>", target="y", group_column="group"),
    )
    with pytest.raises(errors.BlockedError, match="groups"):
        divide(spec, frame)
