"""Shared fixtures."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def binary_frame():
    rng = np.random.default_rng(0)
    rows = 500
    frame = pd.DataFrame(
        {
            "a": rng.normal(size=rows),
            "b": rng.normal(size=rows),
            "c": rng.choice(list("xyz"), rows),
        }
    )
    frame.loc[::13, "a"] = np.nan
    signal = frame.a.fillna(0) + 0.5 * frame.b + (frame.c == "x")
    noise = rng.normal(scale=0.5, size=rows)
    frame["y"] = (signal + noise > 0.5).astype(int)
    return frame


@pytest.fixture
def multiclass_frame():
    rng = np.random.default_rng(1)
    rows = 600
    frame = pd.DataFrame(
        {"a": rng.normal(size=rows), "b": rng.normal(size=rows)}
    )
    frame["y"] = pd.cut(
        frame.a + frame.b, [-9, -0.7, 0.7, 9], labels=["lo", "mid", "hi"]
    ).astype(str)
    return frame


@pytest.fixture
def regression_frame():
    rng = np.random.default_rng(2)
    rows = 500
    frame = pd.DataFrame(
        {
            "a": rng.normal(size=rows),
            "b": rng.normal(size=rows),
            "c": rng.choice(list("pq"), rows),
        }
    )
    frame["y"] = (
        3 * frame.a
        - 2 * frame.b
        + (frame.c == "p")
        + rng.normal(scale=0.3, size=rows)
    )
    return frame
