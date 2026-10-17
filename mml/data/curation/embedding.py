"""Text columns as sentence embeddings, computed in the PyTorch worker.

The model's vectors place texts with similar meaning close together, so a
classifier can generalise to wording it never saw in training, which TF-IDF
terms cannot. The framework process never imports PyTorch: unique strings go
to the worker, which keeps a cache. If the embedding model cannot be used
(library missing, no network for the first download), the transformer falls
back to TF-IDF and says so in the log.
"""

import logging
from typing import Any
from typing import Self

import numpy as np
from sklearn import base

from mml.data.curation import transformers
from mml.models.deeplearning import worker

LOGGER = logging.getLogger(__name__)
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FALLBACK_TERMS = 200


class TextEmbedder(base.TransformerMixin, base.BaseEstimator):
    """Embeds a text column with a sentence-embedding model.

    Attributes:
        model: Embedding model identifier.
        fallback_terms: TF-IDF vocabulary size if embeddings are unavailable.
        mode_: `embedding` or `tfidf` after fitting.
        width_: Number of output columns.
        fallback_: The TF-IDF vectorizer when `mode_` is `tfidf`.
        encoded_last_: Strings the worker had to encode on the latest call.
    """

    def __init__(
        self, model: str = DEFAULT_MODEL, fallback_terms: int = FALLBACK_TERMS
    ) -> None:
        """Stores the model name and fallback vocabulary size."""
        self.model = model
        self.fallback_terms = fallback_terms

    @staticmethod
    def flatten(texts: Any) -> list[str]:
        """One string per row from a column, frame or array."""
        return [str(v) for v in np.asarray(texts, dtype=object).ravel()]

    def request(self, strings: list[str]) -> np.ndarray:
        """Embeds unique strings in the worker and maps them back to rows."""
        unique = list(dict.fromkeys(strings))
        matrix, encoded = worker.client().request("embed", (self.model, unique))
        self.encoded_last_ = int(encoded)
        position = {text: index for index, text in enumerate(unique)}
        return np.asarray(matrix)[[position[t] for t in strings]]

    def fit(self, texts: Any, target: Any = None) -> Self:
        """Checks the model works; otherwise falls back to TF-IDF."""
        strings = self.flatten(texts)
        try:
            self.width_ = int(self.request(strings[:2] or [""]).shape[1])
            self.mode_ = "embedding"
        except Exception as error:  # The worker reports many kinds.
            LOGGER.warning(
                "embedding model '%s' is unavailable (%s); falling back to "
                "TF-IDF terms",
                self.model,
                error,
            )
            self.mode_ = "tfidf"
            self.fallback_ = transformers.TextVectorizer(
                max_terms=self.fallback_terms
            ).fit(strings)
        return self

    def transform(self, texts: Any) -> Any:
        """Embedding matrix, or TF-IDF terms in fallback mode."""
        strings = self.flatten(texts)
        if self.mode_ == "tfidf":
            return self.fallback_.transform(strings)
        return self.request(strings)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Column names."""
        if self.mode_ == "tfidf":
            return self.fallback_.get_feature_names_out()
        return np.array([f"embedding_{i}" for i in range(self.width_)])
