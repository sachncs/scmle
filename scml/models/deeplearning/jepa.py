"""Public proxy for the optional tabular JEPA representation trainer."""

import os
import pathlib
import tempfile
from typing import Any

import joblib
import numpy as np
from sklearn import base
from sklearn import utils

from scml.models.deeplearning import estimators


class JepaEstimator(base.TransformerMixin, estimators.NetworkEstimator):
    """Worker-backed tabular JEPA representation estimator.

    This is a self-supervised numeric-tabular representation path. It is not a
    multimodal JEPA implementation and is not registered as a supervised
    candidate in the default model search.
    """

    engine_path = "scml.models.deeplearning.jepa_engine.JepaEngine"
    CHECKPOINT_FORMAT = "scml.tabular-jepa.v1"

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

    def save_checkpoint(self, path: str | pathlib.Path) -> pathlib.Path:
        """Save a fitted JEPA representation checkpoint.

        The checkpoint contains the estimator configuration, training report,
        and NumPy-only worker state. It is intended for trusted local storage;
        loading it uses joblib's pickle-based format.

        Args:
            path: Destination file. Its parent directory must already exist.

        Returns:
            The checkpoint path.

        Raises:
            sklearn.exceptions.NotFittedError: If the estimator is unfitted.
        """
        utils.validation.check_is_fitted(self, attributes=["state_"])
        destination = pathlib.Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary: pathlib.Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = pathlib.Path(handle.name)
            joblib.dump(
                {"format": self.CHECKPOINT_FORMAT, "estimator": self},
                temporary,
            )
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
            descriptor = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except BaseException:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            raise
        return destination

    @classmethod
    def load_checkpoint(cls, path: str | pathlib.Path) -> "JepaEstimator":
        """Load a trusted fitted tabular JEPA checkpoint.

        Args:
            path: Checkpoint produced by :meth:`save_checkpoint`.

        Returns:
            A fitted estimator whose transform output matches the saved model.

        Raises:
            ValueError: If the checkpoint format or payload is invalid.
        """
        payload = joblib.load(path)
        if (
            not isinstance(payload, dict)
            or payload.get("format") != cls.CHECKPOINT_FORMAT
        ):
            raise ValueError("unsupported tabular JEPA checkpoint format")
        estimator = payload.get("estimator")
        if not isinstance(estimator, cls):
            raise ValueError("checkpoint does not contain a JepaEstimator")
        utils.validation.check_is_fitted(estimator, attributes=["state_"])
        return estimator


__all__ = ["JepaEstimator"]
