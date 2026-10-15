import json

import numpy as np
import pandas as pd
import pytest

from mml.artifacts.bundle import ModelBundle
from mml.contracts.data import FeaturePlan
from mml.contracts.errors import ArtifactError, SchemaError
from mml.contracts.tasks import TaskType
from mml.data.preprocessing import build_pipeline
from mml.models.registry import get_model

PLAN = FeaturePlan(("num",), ("cat",))


@pytest.fixture
def bundle():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"num": rng.normal(size=200), "cat": rng.choice(["a", "b"], 200)})
    y = pd.Series((X.num > 0).astype(int))
    pipe = build_pipeline(PLAN, get_model("random_forest"), TaskType.BINARY, 0).fit(X, y)
    return ModelBundle.create(
        pipe, PLAN, X, "y", "binary_classification", "roc_auc", "rf", [0, 1]
    ), X


def test_roundtrip_prediction_equivalence(bundle, tmp_path):
    b, X = bundle
    b.save(tmp_path / "m")
    loaded = ModelBundle.load(tmp_path / "m")
    np.testing.assert_array_equal(b.predict(X), loaded.predict(X))
    np.testing.assert_allclose(b.predict_proba(X), loaded.predict_proba(X))


def test_predict_from_file_and_extra_columns(bundle, tmp_path):
    b, X = bundle
    X.assign(junk=1).to_csv(tmp_path / "n.csv", index=False)
    assert len(b.predict(tmp_path / "n.csv")) == len(X)
    assert b.validate(X.assign(junk=1)).extra == ("junk",)


def test_schema_errors_are_structured(bundle):
    b, X = bundle
    with pytest.raises(SchemaError) as e:
        b.predict(X.drop(columns=["num"]))
    assert e.value.details["missing"] == ["num"]
    with pytest.raises(SchemaError, match="non-numeric"):
        b.predict(X.assign(num="oops"))


def test_unknown_categories_reported_not_fatal(bundle):
    b, X = bundle
    new = X.head(3).assign(cat="zzz")
    assert b.validate(new).unseen_categories == {"cat": ["zzz"]}
    assert len(b.predict(new)) == 3


def test_tamper_detected(bundle, tmp_path):
    b, _ = bundle
    d = b.save(tmp_path / "m")
    with (d / "pipeline.joblib").open("ab") as fh:
        fh.write(b"x")
    with pytest.raises(ArtifactError, match="checksum"):
        ModelBundle.load(d)


def test_version_mismatch_blocks_unless_allowed(bundle, tmp_path):
    b, X = bundle
    d = b.save(tmp_path / "m")
    mf = json.loads((d / "model_manifest.json").read_text())
    mf["sklearn"] = "0.1.0"
    (d / "model_manifest.json").write_text(json.dumps(mf))
    with pytest.raises(ArtifactError, match="different dependencies"):
        ModelBundle.load(d)
    assert ModelBundle.load(d, allow_version_mismatch=True).predict(X).shape[0] == len(X)


def test_not_a_bundle(tmp_path):
    with pytest.raises(ArtifactError):
        ModelBundle.load(tmp_path)
