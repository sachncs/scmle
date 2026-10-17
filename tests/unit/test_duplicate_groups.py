"""Copies of one record never straddle a split."""

import numpy as np
import pandas as pd

from mml.contracts import tasks
from mml.data.splitting import duplicates
from mml.data.splitting import splitters


def spec(task="binary_classification"):
    return tasks.ProjectSpec(
        data=tasks.DataSpec(uri="x", target="y"), task=task
    )


def frame_with_copies(rows=600, copies=200, seed=0):
    rng = np.random.default_rng(seed)
    base = pd.DataFrame(
        {
            "a": rng.normal(size=rows),
            "b": rng.choice(["p", "q", "r"], rows),
            "y": rng.integers(0, 2, rows),
        }
    )
    extra = base.sample(copies, random_state=seed, replace=True)
    return pd.concat([base, extra], ignore_index=True)


def test_rows_with_the_same_features_share_a_group():
    frame = pd.DataFrame({"a": [1.0, 1.0, 2.0, 3.0], "b": ["x", "x", "x", "y"]})

    groups = duplicates.duplicate_groups(frame, ["a", "b"])

    assert groups[0] == groups[1] != duplicates.NOT_DUPLICATED
    assert groups[2] == groups[3] == duplicates.NOT_DUPLICATED


def test_numbers_are_compared_to_six_decimals():
    frame = pd.DataFrame({"a": [1.0000001, 1.0000002, 1.01]})

    groups = duplicates.duplicate_groups(frame, ["a"])

    assert groups[0] == groups[1] != duplicates.NOT_DUPLICATED
    assert groups[2] == duplicates.NOT_DUPLICATED


def test_no_duplicate_group_straddles_the_holdout_or_any_fold():
    frame = frame_with_copies()
    groups = duplicates.duplicate_groups(frame, ["a", "b"])

    split = splitters.SplitManager(spec(), tasks.TaskType.BINARY).split(frame)

    dev = set(split.manifest.train_ids)
    test = set(split.manifest.test_ids)
    for group in set(groups) - {duplicates.NOT_DUPLICATED}:
        rows = set(np.flatnonzero(groups == group).tolist())
        assert rows <= dev or rows <= test
    dev_rows = np.array(split.manifest.train_ids)
    for fit, held in split.manifest.folds:
        fit_groups = set(groups[dev_rows[list(fit)]])
        held_groups = set(groups[dev_rows[list(held)]])
        shared = (fit_groups & held_groups) - {duplicates.NOT_DUPLICATED}
        assert not shared
    assert any("duplicate rows" in note for note in split.manifest.notes)


def test_data_without_copies_keeps_the_plain_split():
    frame = frame_with_copies(copies=0)
    manager = splitters.SplitManager(spec(), tasks.TaskType.BINARY)

    plain = splitters.STRATEGIES[manager.strategy_name](
        spec(), tasks.TaskType.BINARY
    ).partition(frame)
    split = manager.split(frame)

    assert split.manifest.train_ids == tuple(int(i) for i in plain.dev_ids)
    assert split.manifest.test_ids == tuple(int(i) for i in plain.test_ids)
    assert not any("duplicate" in note for note in split.manifest.notes)


def test_every_row_still_lands_in_exactly_one_place():
    frame = frame_with_copies()

    split = splitters.SplitManager(spec(), tasks.TaskType.BINARY).split(frame)

    dev, test = split.manifest.train_ids, split.manifest.test_ids
    assert sorted(dev + test) == list(range(len(frame)))
