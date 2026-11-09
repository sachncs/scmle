"""Pretrained tabular foundation model (TabPFN) as a worker engine.

TabPFN predicts by in-context learning: "fitting" stores the training rows
and prediction runs them, with the test rows, through a pretrained network.
Its weights are downloaded once and require accepting the Prior Labs
license; set `TABPFN_TOKEN` (or accept the license interactively) before use.
This module runs only inside the PyTorch worker process.
"""

import importlib
from typing import Any

import numpy
from scipy import sparse

MAX_ROWS = 10000
MAX_FEATURES = 500


class Learner:
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
        """Stores the settings.

        Args:
            device: `auto`, `cpu` or `cuda`.
            random_state: Seed of TabPFN's internal ensemble.
            n_estimators: Members of TabPFN's internal ensemble.
        """
        self.device = device
        self.random_state = random_state
        self.n_estimators = n_estimators

    def build(self) -> Any:
        """Creates the TabPFN model from the installed package.

        Returns:
            An unfitted TabPFN estimator of this engine's `class_name`.
        """
        library = importlib.import_module("tabpfn")
        return getattr(library, self.class_name)(
            device="cuda" if self.device == "cuda" else self.device,
            random_state=self.random_state,
            n_estimators=self.n_estimators,
        )

    def fit(
        self, features: Any, target: Any, sample_weight: Any = None
    ) -> "Learner":
        """Stores the training rows with the model (in-context learning).

        Args:
            features: Training matrix, dense or sparse.
            target: Training labels or values.
            sample_weight: Must be None.

        Returns:
            This fitted engine.

        Raises:
            ValueError: If the data exceeds what TabPFN supports, or if
                `sample_weight` is given (in-context learning has no way to
                weight rows, and ignoring the weights silently would be wrong).
        """
        if sample_weight is not None:
            raise ValueError("TabPFN does not support sample weights")
        if sparse.issparse(features):
            features = features.toarray()
        matrix = numpy.asarray(features, dtype=numpy.float32)
        if len(matrix) > MAX_ROWS or matrix.shape[1] > MAX_FEATURES:
            raise ValueError(
                f"TabPFN supports up to {MAX_ROWS} rows and {MAX_FEATURES} "
                f"features, got {matrix.shape}"
            )
        self.train_features_ = matrix
        self.train_target_ = numpy.asarray(target)
        self.model_ = self.build().fit(matrix, self.train_target_)
        return self

    def __getstate__(self) -> dict[str, Any]:
        """The data and settings from which the model is rebuilt.

        Returns:
            Mapping of the settings, the stored training rows, a fixed training
            report and any task-specific entries.
        """
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
        """Task-specific entries of the state.

        Returns:
            Extra entries merged into the pickled state; empty by default.
        """
        return {}

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Rebuilds the model from the stored rows (cheap: no training).

        Args:
            state: Mapping produced by `__getstate__`.
        """
        self.device = state["device"]
        self.random_state = state["random_state"]
        self.n_estimators = state["n_estimators"]
        self.train_features_ = state["train_features_"]
        self.train_target_ = state["train_target_"]
        self.model_ = self.build().fit(self.train_features_, self.train_target_)

    def rows(self, features: Any) -> numpy.ndarray:
        """Float32 matrix of the rows to score.

        Args:
            features: Matrix to score.

        Returns:
            Dense float32 array of the rows.
        """
        if sparse.issparse(features):
            features = features.toarray()
        return numpy.asarray(features, dtype=numpy.float32)


class Classifier(Learner):
    """TabPFN classifier."""

    class_name = "TabPFNClassifier"

    def extra_state(self) -> dict[str, Any]:
        """The class labels.

        Returns:
            Mapping with the fitted model's `classes_`.
        """
        return {"classes_": numpy.asarray(self.model_.classes_)}

    def predict_proba(self, features: Any) -> numpy.ndarray:
        """Class probabilities.

        Args:
            features: Matrix to score.

        Returns:
            Array of shape `(n_rows, n_classes)`.
        """
        return numpy.asarray(self.model_.predict_proba(self.rows(features)))

    def predict(self, features: Any) -> numpy.ndarray:
        """Most probable class.

        Args:
            features: Matrix to score.

        Returns:
            Array of shape `(n_rows,)` of class labels.
        """
        return numpy.asarray(self.model_.predict(self.rows(features)))


class Regressor(Learner):
    """TabPFN regressor."""

    class_name = "TabPFNRegressor"

    def predict(self, features: Any) -> numpy.ndarray:
        """Predicted values.

        Args:
            features: Matrix to score.

        Returns:
            Array of shape `(n_rows,)`.
        """
        return numpy.asarray(self.model_.predict(self.rows(features)))
