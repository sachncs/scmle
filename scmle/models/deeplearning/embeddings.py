"""Sentence embeddings. Runs only inside the worker process.

Importing this module needs `sentence-transformers` (and PyTorch), so nothing
outside `worker` loads it. A string is embedded once per worker: results are
kept in a bounded cache, so repeated and unseen-before strings are the only
ones that cost a forward pass.
"""

import collections
import importlib
from typing import Any

import numpy

CACHE_STRINGS = 200_000
BATCH = 64


class Embedder:
    """A loaded sentence-embedding model with a string cache.

    Attributes:
        name: Model identifier.
        model: The loaded model.
        cache: Embedding per string, least recently used first.
    """

    def __init__(self, name: str) -> None:
        """Loads the model on the CPU.

        Importing `sentence_transformers` raises ImportError if the package
        is not installed.

        Args:
            name: Model identifier passed to `SentenceTransformer`.
        """
        module = importlib.import_module("sentence_transformers")
        self.name = name
        self.model = module.SentenceTransformer(name, device="cpu")
        self.cache: collections.OrderedDict[str, numpy.ndarray] = (
            collections.OrderedDict()
        )

    def embed(self, texts: list[str]) -> tuple[numpy.ndarray, int]:
        """Embeds texts, encoding only those not seen before.

        Args:
            texts: Strings to embed; duplicates are encoded once.

        Returns:
            A float32 matrix, one row per text in the given order, and how many
            texts had to be encoded (the rest came from the cache).
        """
        missing = list(dict.fromkeys(t for t in texts if t not in self.cache))
        if missing:
            vectors = self.model.encode(
                missing, batch_size=BATCH, show_progress_bar=False
            )
            for text, vector in zip(missing, vectors, strict=True):
                self.cache[text] = numpy.asarray(vector, dtype=numpy.float32)
        for text in texts:
            self.cache.move_to_end(text)
        while len(self.cache) > CACHE_STRINGS:
            self.cache.popitem(last=False)
        return numpy.vstack([self.cache[t] for t in texts]), len(missing)


MODELS: dict[str, Embedder] = {}


def embed(name: str, texts: list[str]) -> tuple[numpy.ndarray, int]:
    """Embeds `texts` with the (cached) model called `name`.

    Args:
        name: Model identifier; the model is loaded on first use.
        texts: Strings to embed.

    Returns:
        A float32 matrix with one row per text, and how many texts had to be
        encoded rather than read from the cache.
    """
    if name not in MODELS:
        MODELS[name] = Embedder(name)
    return MODELS[name].embed(texts)


def available() -> bool:
    """Whether the embedding library can be imported here.

    Returns:
        True if `sentence_transformers` can be imported, else False.
    """
    try:
        importlib.import_module("sentence_transformers")
    except ImportError:
        return False
    return True


def describe() -> dict[str, Any]:
    """Which models are loaded and how many strings each has cached.

    Returns:
        Mapping from model identifier to its number of cached strings.
    """
    return {name: len(model.cache) for name, model in MODELS.items()}
