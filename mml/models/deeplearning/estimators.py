"""Scikit-learn estimators whose computation runs in the PyTorch worker.

These classes never import torch. `fit` ships the (dense, float32) training
matrix to the worker and keeps the returned weights as plain numpy arrays;
`predict` ships rows plus a state id, and the worker loads the weights once
and caches them. Pickling therefore stores only numpy data.
"""

from typing import Any
import uuid

import numpy as np
from scipy import sparse
from sklearn import base

from mml.models.deeplearning import worker

NEED_STATE = worker.NEED_STATE


class NetworkEstimator(base.BaseEstimator):
    """Common behaviour of the worker-backed estimators.

    Attributes:
        architecture: Name in `architectures.ARCHITECTURES`.
        hidden_sizes: Layer widths (residual: width, then block count).
        dropout: Dropout probability.
        learning_rate: Adam step size.
        weight_decay: L2 penalty.
        epochs: Maximum epochs.
        batch_size: Rows per step.
        patience: Epochs without validation improvement before stopping.
        validation_fraction: Share of rows held out for early stopping.
        device: `auto` (CUDA when available), `cpu` or `cuda`.
        random_state: Seed for initialization, shuffling and the split.
    """

    engine_path = ""

    def __init__(
        self,
        architecture: str = "mlp",
        hidden_sizes: tuple[int, ...] = (64,),
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        epochs: int = 100,
        batch_size: int = 128,
        patience: int = 10,
        validation_fraction: float = 0.1,
        device: str = "auto",
        random_state: int = 0,
    ) -> None:
        """Stores the hyperparameters (scikit-learn convention)."""
        self.architecture = architecture
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.device = device
        self.random_state = random_state

    @staticmethod
    def dense(features: Any) -> np.ndarray:
        """Returns a float32 matrix, densifying sparse input."""
        if sparse.issparse(features):
            features = features.toarray()
        return np.asarray(features, dtype=np.float32)

    def fit(
        self, features: Any, target: Any, sample_weight: Any = None
    ) -> "NetworkEstimator":
        """Trains the network in the worker.

        Args:
            features: Numeric matrix.
            target: Labels or values.
            sample_weight: Optional per-row weights.

        Returns:
            The fitted estimator.
        """
        weights = (
            None
            if sample_weight is None
            else np.asarray(sample_weight, dtype=np.float32)
        )
        state = worker.client().request(
            "fit",
            (
                self.engine_path,
                self.get_params(),
                self.dense(features),
                np.asarray(target),
                weights,
            ),
        )
        self.state_ = state
        self.state_id_ = uuid.uuid4().hex
        self.n_iter_ = state["n_iter_"]
        self.remember(state)
        return self

    def remember(self, state: dict[str, Any]) -> None:
        """Copies the attributes scikit-learn expects from the fitted state."""

    def call(self, method: str, features: Any) -> np.ndarray:
        """Runs an engine method in the worker on new rows."""
        matrix = self.dense(features)
        client = worker.client()
        for state in (None, self.state_):
            reply = client.request(
                "call",
                (self.engine_path, self.state_id_, state, method, matrix),
            )
            if not (isinstance(reply, str) and reply == NEED_STATE):
                return np.asarray(reply)
        raise worker.WorkerError("the worker could not load the model state")


class TorchClassifier(base.ClassifierMixin, NetworkEstimator):
    """Classifier: softmax over class scores, optional class balancing."""

    engine_path = "mml.models.deeplearning.engines.ClassifierEngine"

    def __init__(
        self,
        architecture: str = "mlp",
        hidden_sizes: tuple[int, ...] = (64,),
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        epochs: int = 100,
        batch_size: int = 128,
        patience: int = 10,
        validation_fraction: float = 0.1,
        device: str = "auto",
        random_state: int = 0,
        class_weight: str | None = None,
    ) -> None:
        """Stores the hyperparameters, plus `class_weight` (`balanced`)."""
        super().__init__(
            architecture,
            hidden_sizes,
            dropout,
            learning_rate,
            weight_decay,
            epochs,
            batch_size,
            patience,
            validation_fraction,
            device,
            random_state,
        )
        self.class_weight = class_weight

    def remember(self, state: dict[str, Any]) -> None:
        """Exposes `classes_` after fitting."""
        self.classes_ = state["classes_"]

    def predict_proba(self, features: Any) -> np.ndarray:
        """Class probabilities, columns ordered like `classes_`."""
        return self.call("predict_proba", features)

    def predict(self, features: Any) -> np.ndarray:
        """Most probable class."""
        return self.call("predict", features)


class TorchRegressor(base.RegressorMixin, NetworkEstimator):
    """Regressor trained on the standardized target with squared error."""

    engine_path = "mml.models.deeplearning.engines.RegressorEngine"

    def predict(self, features: Any) -> np.ndarray:
        """Predicted values in the original units."""
        return self.call("predict", features)
