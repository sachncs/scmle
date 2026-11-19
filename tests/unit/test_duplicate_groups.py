"""Repeated records are exchangeable draws: counted, not moved."""

import numpy
import pandas

from scmle.contracts import tasks
from scmle.data.splitting import duplicates
from scmle.data.splitting import splitters


def spec(task="binary_classification"):
    return tasks.Config(data=tasks.Dataset(uri="x", target="y"), task=task)


def frame_with_copies(rows=600, copies=200, seed=0):
    rng = numpy.random.default_rng(seed)
    base = pandas.DataFrame(
        {
            "a": rng.normal(size=rows),
            "b": rng.choice(["p", "q", "r"], rows),
            "y": rng.integers(0, 2, rows),
        }
    )
    extra = base.sample(copies, random_state=seed, replace=True)
    return pandas.concat([base, extra], ignore_index=True)


def test_rows_with_the_same_features_share_a_group():
    frame = pandas.DataFrame(
        {"a": [1.0, 1.0, 2.0, 3.0], "b": ["x", "x", "x", "y"]}
    )

    groups = duplicates.group(frame, ["a", "b"])

    assert groups[0] == groups[1] != duplicates.NOT_DUPLICATED
    assert groups[2] == groups[3] == duplicates.NOT_DUPLICATED


def test_numbers_are_compared_to_six_decimals():
    frame = pandas.DataFrame({"a": [1.0000001, 1.0000002, 1.01]})

    groups = duplicates.group(frame, ["a"])

    assert groups[0] == groups[1] != duplicates.NOT_DUPLICATED
    assert groups[2] == duplicates.NOT_DUPLICATED


def test_repeats_are_exchangeable_draws_and_are_counted():
    frame = frame_with_copies()
    manager = splitters.Splitter(spec(), tasks.Task.BINARY)
    plain = splitters.STRATEGIES[manager.strategy](
        spec(), tasks.Task.BINARY
    ).partition(frame)
    groups = duplicates.group(frame, ["a", "b"])

    split = manager.split(frame)

    assert split.manifest.training == tuple(int(i) for i in plain.development)
    assert split.manifest.test == tuple(int(i) for i in plain.test)
    seen = set(groups[list(split.manifest.training)])
    repeated = sum(
        groups[row] != duplicates.NOT_DUPLICATED and groups[row] in seen
        for row in split.manifest.test
    )
    assert repeated > 0
    assert (
        f"{repeated} of {len(split.manifest.test)} test rows repeat a "
        "development record" in " ".join(split.manifest.notes)
    )


def test_data_without_copies_keeps_the_plain_split():
    frame = frame_with_copies(copies=0)
    manager = splitters.Splitter(spec(), tasks.Task.BINARY)

    plain = splitters.STRATEGIES[manager.strategy](
        spec(), tasks.Task.BINARY
    ).partition(frame)
    split = manager.split(frame)

    assert split.manifest.training == tuple(int(i) for i in plain.development)
    assert split.manifest.test == tuple(int(i) for i in plain.test)
    assert not any("duplicate" in note for note in split.manifest.notes)


def test_every_row_still_lands_in_exactly_one_place():
    frame = frame_with_copies()

    split = splitters.Splitter(spec(), tasks.Task.BINARY).split(frame)

    dev, test = split.manifest.training, split.manifest.test
    assert sorted(dev + test) == list(range(len(frame)))
