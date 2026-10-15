import numpy as np
import pandas as pd
import pytest

from mml.contracts.errors import BlockedError, LeakageError
from mml.contracts.tasks import DataSpec, ProjectSpec, TaskType, ValidationSpec
from mml.data.splitting import make_split


def spec(strategy="auto", **data):
    return ProjectSpec(
        data=DataSpec(uri="m", target="y", **data), validation=ValidationSpec(strategy=strategy)
    )


@pytest.fixture
def df():
    rng = np.random.default_rng(0)
    n = 400
    return pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "g": rng.integers(0, 40, n),
            "t": pd.date_range("2024-01-01", periods=n, freq="h"),
            "y": (rng.random(n) < 0.2).astype(int),
        }
    )


def test_stratified_preserves_ratio_and_disjoint(df):
    m, X, y, locked = make_split(df, spec(), TaskType.BINARY)
    assert m.strategy == "stratified"
    assert set(m.train_ids).isdisjoint(m.test_ids)
    assert len(m.train_ids) + len(m.test_ids) == len(df)
    assert abs(y.mean() - df.y.mean()) < 0.03
    for tr, va in m.folds:
        assert set(tr).isdisjoint(va)
        assert abs(y.iloc[list(va)].mean() - df.y.mean()) < 0.08
    assert "y" not in X.columns


def test_group_split_has_no_group_overlap(df):
    m, X, y, locked = make_split(df, spec(group_column="g"), TaskType.BINARY)
    assert m.strategy == "group"
    tr_g = set(df.g.iloc[list(m.train_ids)])
    te_g = set(df.g.iloc[list(m.test_ids)])
    assert tr_g.isdisjoint(te_g)
    dev_g = df.g.iloc[list(m.train_ids)].to_numpy()
    for tr, va in m.folds:
        assert set(dev_g[list(tr)]).isdisjoint(dev_g[list(va)])


def test_temporal_split_respects_order(df):
    shuffled = df.sample(frac=1, random_state=1).reset_index(drop=True)
    m, *_ = make_split(shuffled, spec(time_column="t"), TaskType.BINARY)
    t = shuffled.t
    assert t.iloc[list(m.train_ids)].max() <= t.iloc[list(m.test_ids)].min()
    dev_t = t.iloc[list(m.train_ids)].to_numpy()
    for tr, va in m.folds:
        assert dev_t[list(tr)].max() <= dev_t[list(va)].min()


def test_temporal_with_bad_time_blocks(df):
    bad = df.assign(t=df.t.astype(str))
    bad.loc[3, "t"] = "not a date"
    with pytest.raises(BlockedError):
        make_split(bad, spec(time_column="t"), TaskType.BINARY)


def test_manifest_deterministic(df):
    a, *_ = make_split(df, spec(), TaskType.BINARY)
    b, *_ = make_split(df.copy(), spec(), TaskType.BINARY)
    assert a.fingerprint == b.fingerprint


def test_tiny_minority_blocks(df):
    d = df.copy()
    d["y"] = 0
    d.loc[0, "y"] = 1
    with pytest.raises(BlockedError):
        make_split(d, spec(), TaskType.BINARY)


def test_regression_uses_random(df):
    m, *_ = make_split(df.assign(y=df.x * 2), spec(), TaskType.REGRESSION)
    assert m.strategy == "random"


def test_locked_test_set_single_use(df):
    _, _, _, locked = make_split(df, spec(), TaskType.BINARY)
    with pytest.raises(LeakageError):
        locked.open("guess")
    tok = locked.issue_token()
    Xt, yt = locked.open(tok)
    assert len(Xt) == len(locked) == len(yt)
    with pytest.raises(LeakageError):
        locked.open(tok)
