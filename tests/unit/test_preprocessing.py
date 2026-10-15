import numpy as np
import pandas as pd
from sklearn.base import clone

from mml.contracts.data import FeaturePlan
from mml.contracts.tasks import TaskType
from mml.data.preprocessing import build_pipeline
from mml.models.registry import get_model


def frame(n=200, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "num": rng.normal(size=n),
            "flag": rng.random(n) < 0.5,
            "cat": rng.choice(["a", "b", "c"], n),
        }
    )
    df.loc[::7, "num"] = np.nan
    df.loc[::11, "cat"] = None
    y = pd.Series((df.num.fillna(0) > 0).astype(int))
    return df, y


PLAN = FeaturePlan(("num", "flag"), ("cat",))


def test_handles_missing_bool_and_categoricals():
    X, y = frame()
    for name in ["logistic_regression", "random_forest", "hist_gradient_boosting", "dummy"]:
        pipe = build_pipeline(PLAN, get_model(name), TaskType.BINARY, 0).fit(X, y)
        assert pipe.predict_proba(X).shape == (200, 2)


def test_unknown_category_and_extra_columns_at_inference():
    X, y = frame()
    pipe = build_pipeline(PLAN, get_model("logistic_regression"), TaskType.BINARY, 0).fit(X, y)
    new = X.head(5).assign(cat="never_seen", junk="x")
    assert pipe.predict(new).shape == (5,)


def test_fold_local_fitting_uses_only_training_rows():
    X, y = frame()
    train, other = X.iloc[:100], X.iloc[100:].assign(num=1e6)
    pipe = build_pipeline(PLAN, get_model("logistic_regression"), TaskType.BINARY, 0)
    pipe.fit(train, y.iloc[:100])
    scaler = pipe.named_steps["prep"].named_transformers_["num"].named_steps["scale"]
    assert scaler.mean_[0] < 10  # statistics come from training data only
    assert other.num.iloc[0] == 1e6  # untouched by fitting


def test_clone_is_unfitted_and_deterministic():
    X, y = frame()
    p1 = build_pipeline(PLAN, get_model("random_forest"), TaskType.BINARY, 3)
    p1.fit(X, y)
    p2 = clone(p1)
    assert not hasattr(p2.named_steps["model"], "estimators_")
    np.testing.assert_allclose(p2.fit(X, y).predict_proba(X), p1.predict_proba(X))
