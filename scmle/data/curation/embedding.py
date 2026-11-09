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

from scmle.data.curation import transformers
from scmle.models.deeplearning import worker

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
        """Stores the model name and fallback vocabulary size.

        Args:
            model: Embedding model identifier.
            fallback_terms: TF-IDF vocabulary size used if embeddings are
                unavailable.
        """
        self.model = model
        self.fallback_terms = fallback_terms

    @staticmethod
    def flatten(texts: Any) -> list[str]:
        """Converts a column, frame or array to one string per element.

        Args:
            texts: A column, frame or array of texts; it is flattened in
                row-major order.

        Returns:
            One string per element, in flattened order.
        """
        return [str(v) for v in np.asarray(texts, dtype=object).ravel()]

    def request(self, strings: list[str]) -> np.ndarray:
        """Embeds unique strings in the worker and maps them back to rows.

        Sets `encoded_last_` to the number of strings the worker had to
        encode (the rest came from its cache).

        Args:
            strings: The texts to embed, one per row; duplicates are sent to
                the worker once.

        Returns:
            A matrix with one embedding row per input string, in input order.
        """
        unique = list(dict.fromkeys(strings))
        matrix, encoded = worker.client().request("embed", (self.model, unique))
        self.encoded_last_ = int(encoded)
        position = {text: index for index, text in enumerate(unique)}
        return np.asarray(matrix)[[position[t] for t in strings]]

    def fit(self, texts: Any, target: Any = None) -> Self:
        """Checks the model works; otherwise falls back to TF-IDF.

        Any failure of the worker while embedding up to two probe strings
        is logged as a warning and switches to a TF-IDF vectorizer fitted on
        all the texts.

        Args:
            texts: The training texts.
            target: Ignored; present for scikit-learn compatibility.

        Returns:
            The fitted transformer.
        """
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
        """Embeds the texts, or computes TF-IDF terms in fallback mode.

        Args:
            texts: The texts to transform.

        Returns:
            A dense embedding matrix with `width_` columns, or the sparse
            TF-IDF matrix in fallback mode; one row per text.
        """
        strings = self.flatten(texts)
        if self.mode_ == "tfidf":
            return self.fallback_.transform(strings)
        return self.request(strings)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Names of the output columns.

        Args:
            input_features: Ignored; the names do not depend on the input.

        Returns:
            `embedding_<i>` for each embedding dimension, or the TF-IDF terms
            in fallback mode.
        """
        if self.mode_ == "tfidf":
            return self.fallback_.get_feature_names_out()
        return np.array([f"embedding_{i}" for i in range(self.width_)])
