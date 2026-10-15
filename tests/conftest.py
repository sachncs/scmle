import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def binary_df():
    rng = np.random.default_rng(0)
    n = 500
    df = pd.DataFrame(
        {"a": rng.normal(size=n), "b": rng.normal(size=n), "c": rng.choice(list("xyz"), n)}
    )
    df.loc[::13, "a"] = np.nan
    df["y"] = (
        (df.a.fillna(0) + 0.5 * df.b + (df.c == "x")) + rng.normal(scale=0.5, size=n) > 0.5
    ).astype(int)
    return df


@pytest.fixture
def multiclass_df():
    rng = np.random.default_rng(1)
    n = 600
    df = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    df["y"] = pd.cut(df.a + df.b, [-9, -0.7, 0.7, 9], labels=["lo", "mid", "hi"]).astype(str)
    return df


@pytest.fixture
def regression_df():
    rng = np.random.default_rng(2)
    n = 500
    df = pd.DataFrame(
        {"a": rng.normal(size=n), "b": rng.normal(size=n), "c": rng.choice(list("pq"), n)}
    )
    df["y"] = 3 * df.a - 2 * df.b + (df.c == "p") + rng.normal(scale=0.3, size=n)
    return df
