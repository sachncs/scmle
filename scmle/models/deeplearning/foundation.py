"""Pretrained tabular foundation model (TabPFN) as a worker engine.

TabPFN predicts by in-context learning: "fitting" stores the training rows
and prediction runs them, with the test rows, through a pretrained network.
Its weights are downloaded once and require accepting the Prior Labs
license; set `TABPFN_TOKEN` (or accept the license interactively) before use.
This module runs only inside the PyTorch worker process.
"""

import importlib
from typing import Any

import numpy as np
from scipy import sparse

MAX_ROWS = 10000
MAX_FEATURES = 500


class TabPfnEngine:
    """Common behaviour of the TabPFN engines.

    Attributes:
        device: `auto`, `cpu` or `cuda`.
        random_state: Seed of TabPFN's internal ensemble.
        n_estimators: Members of TabPFN's internal ensemble.
    """

    class_name = ""

    def __init__(
        self,
        device: str = "auto",
        random_state: int = 0,
        n_estimators: int = 8,
    ) -> None:
        """Stores the settings."""
        self.device = device
        self.random_state = random_state
        self.n_estimators = n_estimators

    def build(self) -> Any:
        """Creates the TabPFN model from the installed package."""
        library = importlib.import_module("tabpfn")
        return getattr(library, self.class_name)(
            device="cuda" if self.device == "cuda" else self.device,
            random_state=self.random_state,
            n_estimators=self.n_estimators,
        )

    def fit(
        self, features: Any, target: Any, sample_weight: Any = None
    ) -> "TabPfnEngine":
        """Stores the training rows with the model (in-context learning).

        Raises:
            ValueError: If the data exceeds what TabPFN supports, or if
                `sample_weight` is given (in-context learning has no way to
                weight rows, and ignoring the weights silently would be wrong).
        """
        if sample_weight is not None:
            raise ValueError("TabPFN does not support sample weights")
        if sparse.issparse(features):
            features = features.toarray()
        matrix = np.asarray(features, dtype=np.float32)
        if len(matrix) > MAX_ROWS or matrix.shape[1] > MAX_FEATURES:
            raise ValueError(
                f"TabPFN supports up to {MAX_ROWS} rows and {MAX_FEATURES} "
                f"features, got {matrix.shape}"
            )
        self.train_features_ = matrix
        self.train_target_ = np.asarray(target)
        self.model_ = self.build().fit(matrix, self.train_target_)
        return self

    def __getstate__(self) -> dict[str, Any]:
        """The data and settings from which the model is rebuilt."""
        return {
            "device": self.device,
            "random_state": self.random_state,
            "n_estimators": self.n_estimators,
            "train_features_": self.train_features_,
            "train_target_": self.train_target_,
            "n_iter_": 1,
            "training_report_": {"verdict": "ok", "steps": 0},
            **self.extra_state(),
        }

    def extra_state(self) -> dict[str, Any]:
        """Task-specific entries of the state."""
        return {}

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Rebuilds the model from the stored rows (cheap: no training)."""
        self.device = state["device"]
        self.random_state = state["random_state"]
        self.n_estimators = state["n_estimators"]
        self.train_features_ = state["train_features_"]
        self.train_target_ = state["train_target_"]
        self.model_ = self.build().fit(self.train_features_, self.train_target_)

    def rows(self, features: Any) -> np.ndarray:
        """Float32 matrix of the rows to score."""
        if sparse.issparse(features):
            features = features.toarray()
        return np.asarray(features, dtype=np.float32)


class ClassifierEngine(TabPfnEngine):
    """TabPFN classifier."""

    class_name = "TabPFNClassifier"

    def extra_state(self) -> dict[str, Any]:
        """The class labels."""
        return {"classes_": np.asarray(self.model_.classes_)}

    def predict_proba(self, features: Any) -> np.ndarray:
        """Class probabilities."""
        return np.asarray(self.model_.predict_proba(self.rows(features)))

    def predict(self, features: Any) -> np.ndarray:
        """Most probable class."""
        return np.asarray(self.model_.predict(self.rows(features)))


class RegressorEngine(TabPfnEngine):
    """TabPFN regressor."""

    class_name = "TabPFNRegressor"

    def predict(self, features: Any) -> np.ndarray:
        """Predicted values."""
        return np.asarray(self.model_.predict(self.rows(features)))
