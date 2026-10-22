"""Public proxy for the optional tabular JEPA representation trainer."""

from typing import Any

import numpy as np

from scml.models.deeplearning import estimators


class JepaEstimator(estimators.NetworkEstimator):
    """Worker-backed tabular JEPA representation estimator.

    This is a self-supervised numeric-tabular representation path. It is not a
    multimodal JEPA implementation and is not registered as a supervised
    candidate in the default model search.
    """

    engine_path = "scml.models.deeplearning.jepa_engine.JepaEngine"

    def __init__(
        self,
        hidden_size: int = 128,
        embedding_size: int = 64,
        mask_probability: float = 0.3,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        epochs: int = 50,
        batch_size: int | str = 128,
        ema_decay: float = 0.99,
        variance_weight: float = 0.1,
        device: str = "auto",
        random_state: int = 0,
    ) -> None:
        """Store the explicit JEPA contract and resource policy."""
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.mask_probability = mask_probability
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.ema_decay = ema_decay
        self.variance_weight = variance_weight
        self.device = device
        self.random_state = random_state

    def fit(
        self,
        features: Any,
        target: Any = None,
        sample_weight: Any = None,
    ) -> "JepaEstimator":
        """Learn a representation without using a target label."""
        return super().fit(
            features,
            (
                np.zeros(len(features), dtype=np.float32)
                if target is None
                else target
            ),
            sample_weight,
        )

    def transform(self, features: Any) -> np.ndarray:
        """Return context-encoder embeddings for new rows."""
        return self.call("encode", features)


__all__ = ["JepaEstimator"]
