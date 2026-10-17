"""Text embeddings computed in the worker, with a TF-IDF fallback."""

import importlib.util
import logging
import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest
from sklearn import linear_model
from sklearn import metrics

from mml.data.curation import embedding
from mml.data.curation import transformers

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("sentence_transformers") is None
    or importlib.util.find_spec("torch") is None,
    reason="sentence-transformers is not installed",
)

TRAIN_VEHICLES = ["car", "truck", "bus"]
TRAIN_FOOD = ["apple", "banana", "bread"]
TEST_VEHICLES = ["automobile", "lorry", "coach"]
TEST_FOOD = ["pear", "mango", "loaf"]


def labelled(vehicles, food, copies=10):
    texts = (vehicles + food) * copies
    labels = ([1] * len(vehicles) + [0] * len(food)) * copies
    return pd.Series(texts), np.array(labels)


def test_the_framework_process_never_imports_torch():
    script = textwrap.dedent(
        """
        import sys
        from mml.data.curation import embedding
        model = embedding.TextEmbedder().fit(["a car", "an apple"])
        model.transform(["a truck"])
        assert model.mode_ == "embedding", model.mode_
        assert "torch" not in sys.modules
        assert "sentence_transformers" not in sys.modules
        """
    )

    done = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )

    assert done.returncode == 0, done.stderr


def test_identical_strings_are_encoded_once():
    model = embedding.TextEmbedder().fit(["warm up", "the model"])
    texts = ["a unique sentence about cats"] * 40 + ["another about dogs"] * 40

    first = model.transform(texts)
    again = model.transform(texts)

    assert model.encoded_last_ == 0
    assert first.shape == again.shape == (80, model.width_)
    np.testing.assert_allclose(first, again)
    assert np.allclose(first[0], first[1])


def test_embeddings_generalise_to_wording_tfidf_never_saw():
    train_texts, train_labels = labelled(TRAIN_VEHICLES, TRAIN_FOOD)
    test_texts, test_labels = labelled(TEST_VEHICLES, TEST_FOOD)

    def auc(encoder):
        encoder.fit(train_texts)
        model = linear_model.LogisticRegression(max_iter=1000).fit(
            encoder.transform(train_texts), train_labels
        )
        scores = model.predict_proba(encoder.transform(test_texts))[:, 1]
        return metrics.roc_auc_score(test_labels, scores)

    embedded = auc(embedding.TextEmbedder())
    terms = auc(transformers.TextVectorizer())

    assert embedded > 0.9
    assert embedded > terms + 0.2


def test_an_unavailable_model_falls_back_to_terms_and_says_so(caplog):
    caplog.set_level(logging.WARNING)
    texts = pd.Series(["red apple pie", "blue car wash"] * 5)

    model = embedding.TextEmbedder("no-such-org/no-such-model").fit(texts)

    assert model.mode_ == "tfidf"
    assert "falling back to TF-IDF" in caplog.text
    assert model.transform(texts).shape[0] == len(texts)
    assert len(model.get_feature_names_out()) == model.transform(texts).shape[1]
