import types

import numpy as np
import pandas as pd

from scmle.governance import lifecycle


def frame(rows=100):
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "when": pd.date_range("2026-01-01", periods=rows)[::-1],
            "who": np.repeat(np.arange(rows // 5), 5),
            "x": rng.normal(size=rows),
        }
    )


def test_temporal_rows_are_held_out_by_time():
    schema = types.SimpleNamespace(time_column="when", group_column=None)
    train, holdout = lifecycle.split_fresh(frame(), schema)
    assert len(holdout) == 30
    assert train["when"].max() < holdout["when"].min()


def test_grouped_rows_never_share_a_group_across_the_split():
    schema = types.SimpleNamespace(time_column=None, group_column="who")
    train, holdout = lifecycle.split_fresh(frame(), schema)
    assert not set(train["who"]) & set(holdout["who"])
    assert len(train) + len(holdout) == 100
    assert len(holdout) > 0


def test_plain_rows_are_shuffled_as_before():
    schema = types.SimpleNamespace(time_column=None, group_column=None)
    train, holdout = lifecycle.split_fresh(frame(), schema)
    assert (len(train), len(holdout)) == (70, 30)
    assert not train["when"].is_monotonic_decreasing


def test_a_single_group_falls_back_to_the_shuffle():
    schema = types.SimpleNamespace(time_column=None, group_column="g")
    one = frame().assign(g="only")
    parts = lifecycle.split_fresh(one, schema)
    assert len(parts[1]) == 30
