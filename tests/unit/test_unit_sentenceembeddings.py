"""The worker's sentence-embedding cache, on a stand-in model library."""

import sys
import types

import numpy
import pytest

from scmle.models.deeplearning import embeddings


class Model:
    """A stand-in `SentenceTransformer`: one row per text, its length twice.

    Attributes:
        name: The model identifier it was loaded with.
        device: The device it was loaded on.
        calls: The texts and options of every `encode` call.
    """

    loaded: list[str] = []

    def __init__(self, name, device):
        self.name = name
        self.device = device
        self.calls = []
        Model.loaded.append(name)

    def encode(self, texts, batch_size, show_progress_bar):
        self.calls.append((list(texts), batch_size, show_progress_bar))
        return [[float(len(text)), float(len(text)) * 2] for text in texts]


@pytest.fixture
def library(monkeypatch):
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = Model
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    monkeypatch.setattr(embeddings, "MODELS", {})
    Model.loaded = []
    return module


def test_the_cache_and_batch_limits_are_fixed():
    assert embeddings.CACHE_STRINGS == 200_000
    assert embeddings.BATCH == 64


def test_an_embedder_loads_its_model_on_the_cpu(library):
    embedder = embeddings.Embedder("mini")

    assert embedder.name == "mini"
    assert embedder.model.name == "mini"
    assert embedder.model.device == "cpu"
    assert len(embedder.cache) == 0


def test_an_embedder_encodes_each_new_string_once_in_order(library):
    embedder = embeddings.Embedder("mini")

    matrix, encoded = embedder.embed(["ab", "c", "ab"])

    assert encoded == 2
    assert matrix.dtype == numpy.float32
    assert matrix.tolist() == [[2.0, 4.0], [1.0, 2.0], [2.0, 4.0]]
    assert embedder.model.calls == [(["ab", "c"], 64, False)]


def test_an_embedder_reads_seen_strings_from_the_cache(library):
    embedder = embeddings.Embedder("mini")
    embedder.embed(["ab", "c"])

    matrix, encoded = embedder.embed(["c", "xyz", "ab"])

    assert encoded == 1
    assert matrix.tolist() == [[1.0, 2.0], [3.0, 6.0], [2.0, 4.0]]
    assert embedder.model.calls[-1] == (["xyz"], 64, False)
    assert list(embedder.cache) == ["c", "xyz", "ab"]


def test_an_embedder_does_not_call_the_model_when_everything_is_cached(library):
    embedder = embeddings.Embedder("mini")
    embedder.embed(["a"])

    matrix, encoded = embedder.embed(["a", "a"])

    assert encoded == 0
    assert matrix.tolist() == [[1.0, 2.0], [1.0, 2.0]]
    assert len(embedder.model.calls) == 1


def test_an_embedder_evicts_the_least_recently_used_strings(
    library, monkeypatch
):
    monkeypatch.setattr(embeddings, "CACHE_STRINGS", 2)
    embedder = embeddings.Embedder("mini")
    embedder.embed(["a", "bb"])
    embedder.embed(["a"])

    matrix, encoded = embedder.embed(["ccc"])

    assert encoded == 1
    assert matrix.tolist() == [[3.0, 6.0]]
    assert list(embedder.cache) == ["a", "ccc"]


def test_an_embedder_keeps_exactly_the_cache_limit(library, monkeypatch):
    monkeypatch.setattr(embeddings, "CACHE_STRINGS", 2)
    embedder = embeddings.Embedder("mini")

    embedder.embed(["a", "bb"])

    assert list(embedder.cache) == ["a", "bb"]


def test_an_embedder_needs_the_library(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)

    with pytest.raises(ImportError):
        embeddings.Embedder("mini")


def test_embed_loads_each_model_once_and_describe_counts_its_cache(library):
    first, encoded = embeddings.embed("mini", ["a", "bb"])
    again, cached = embeddings.embed("mini", ["bb"])
    other, fresh = embeddings.embed("large", ["a"])

    assert (encoded, cached, fresh) == (2, 0, 1)
    assert first.tolist() == [[1.0, 2.0], [2.0, 4.0]]
    assert again.tolist() == [[2.0, 4.0]]
    assert other.tolist() == [[1.0, 2.0]]
    assert Model.loaded == ["mini", "large"]
    assert embeddings.describe() == {"mini": 2, "large": 1}


def test_describe_is_empty_before_any_model_loads(library):
    assert embeddings.describe() == {}


def test_available_reports_whether_the_library_imports(library, monkeypatch):
    assert embeddings.available() is True

    monkeypatch.setitem(sys.modules, "sentence_transformers", None)

    assert embeddings.available() is False
