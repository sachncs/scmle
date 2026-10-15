import numpy as np
import pandas as pd
import pytest

from mml.contracts.errors import BlockedError
from mml.contracts.tasks import DataSpec, ProjectSpec, TaskType
from mml.data.profiling import profile_dataset, resolve_task
from mml.data.readers import build_manifest


def make(df, **kw):
    task = kw.pop("task", "auto")
    data = kw.pop("data", {})
    return ProjectSpec(data=DataSpec(uri="m", target="y", **data), task=task, **kw)


def run(df, **kw):
    spec = make(df, **kw)
    t = resolve_task(spec, df["y"])
    return t, profile_dataset(df, spec, build_manifest(df, "y"), t)


@pytest.fixture
def clf():
    rng = np.random.default_rng(0)
    n = 300
    return pd.DataFrame(
        {
            "num": rng.normal(size=n),
            "cat": rng.choice(list("abc"), n),
            "const": 1,
            "uid": [f"u{i}" for i in range(n)],
            "y": rng.integers(0, 2, n),
        }
    )


def test_task_resolution(clf):
    assert resolve_task(make(clf), clf["y"]) is TaskType.BINARY
    assert resolve_task(make(clf, task="classification"), clf["y"]) is TaskType.BINARY
    multi = clf.assign(y=np.arange(len(clf)) % 4)
    assert resolve_task(make(multi, task="classification"), multi["y"]) is TaskType.MULTICLASS
    reg = clf.assign(y=np.random.default_rng(1).normal(size=len(clf)))
    assert resolve_task(make(reg), reg["y"]) is TaskType.REGRESSION


def test_ambiguous_integer_target_blocks(clf):
    amb = clf.assign(y=np.arange(len(clf)) % 5)
    with pytest.raises(BlockedError, match="ambiguous"):
        resolve_task(make(amb), amb["y"])


def test_explicit_mismatch_blocks(clf):
    with pytest.raises(BlockedError):
        resolve_task(make(clf, task="multiclass_classification"), clf["y"])
    s = pd.Series(["a", "b"] * 5, name="y")
    with pytest.raises(BlockedError):
        resolve_task(make(clf, task="regression"), s)


def test_feature_plan_drops_const_and_id_like(clf):
    _, p = run(clf)
    assert p.feature_plan.numeric == ("num",)
    assert p.feature_plan.categorical == ("cat",)
    assert p.feature_plan.dropped["const"] == "constant"
    assert "id-like" in p.feature_plan.dropped["uid"]


def test_leaky_feature_flagged(clf):
    leaky = clf.assign(leak=clf["y"] * 10 + 1)
    _, p = run(leaky)
    assert any("'leak'" in r for r in p.risk_findings)
    assert not any("'num'" in r for r in p.risk_findings)


def test_missing_target_blocks(clf):
    bad = clf.copy()
    bad.loc[0, "y"] = np.nan
    with pytest.raises(BlockedError, match="missing"):
        run(bad)


def test_imbalance_and_duplicates_reported(clf):
    imb = pd.concat([clf[clf.y == 0], clf[clf.y == 1].head(5)])
    _, p = run(imb)
    assert any("imbalance" in r for r in p.risk_findings)
    d = pd.concat([clf, clf.head(5)])
    _, p = run(d)
    assert any("duplicate" in q for q in p.quality_findings)
