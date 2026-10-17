"""A project can encode its text columns with sentence embeddings."""

import importlib.util

import numpy as np
import pandas as pd
import pytest
import support

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("sentence_transformers") is None,
    reason="sentence-transformers is not installed",
)

VEHICLES = ["a fast car", "the truck is loud", "a slow bus", "the red train"]
FOOD = ["sweet apple", "the bread is warm", "ripe banana", "fresh cheese"]


def reviews(rows=400, seed=0):
    rng = np.random.default_rng(seed)
    label = rng.integers(0, 2, rows)
    texts = [
        f"{rng.choice(VEHICLES if y else FOOD)} number {i}"
        for i, y in enumerate(label)
    ]
    return pd.DataFrame(
        {"text": texts, "size": rng.normal(size=rows), "y": label}
    )


def test_embedded_text_trains_packages_and_predicts(tmp_path):
    frame = reviews()

    result = support.make_project(
        frame,
        "y",
        time_budget=300,
        output_dir=tmp_path,
        text_encoding="embedding",
    ).run()

    assert result.evaluation["test_metrics"]["roc_auc"] > 0.9
    fresh = reviews(50, seed=4).drop(columns="y")
    assert len(result.bundle.predict(fresh)) == 50
    assert "text" in result.bundle.features
