"""Keras training engines. They run only inside the worker process.

Keras 3 runs on the PyTorch backend the worker already has, so no second
deep-learning framework is installed. Importing this module needs Keras; the
estimators users see are the proxies in `estimators`.
"""

import importlib
import os
from typing import Any

import numpy as np
from sklearn import base

STALL_IMPROVEMENT = 0.01
MIN_ROWS_FOR_VALIDATION = 30


def load_keras() -> Any:
    """Imports Keras on the PyTorch backend (it reads the variable once).

    Returns:
        The imported `keras` module.
    """
    os.environ.setdefault("KERAS_BACKEND", "torch")
    return importlib.import_module("keras")


class KerasEngine(base.BaseEstimator):
    """A dense network trained with Keras, Adam and early stopping.

    Attributes:
        hidden_sizes: Layer widths.
        dropout: Dropout probability after each hidden layer.
        learning_rate: Adam step size.
        l2: L2 penalty on the dense kernels.
        epochs: Maximum epochs.
        batch_size: Rows per step.
        patience: Epochs without validation improvement before stopping.
        validation_fraction: Share of rows held out for early stopping.
        random_state: Seed.
    """

    def __init__(
        self,
        hidden_sizes: tuple[int, ...] = (64,),
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        l2: float = 1e-5,
        epochs: int = 100,
        batch_size: int = 128,
        patience: int = 10,
        validation_fraction: float = 0.1,
        random_state: int = 0,
    ) -> None:
        """Stores the hyperparameters (scikit-learn convention).

        Args:
            hidden_sizes: Layer widths.
            dropout: Dropout probability after each hidden layer.
            learning_rate: Adam step size.
            l2: L2 penalty on the dense kernels.
            epochs: Maximum epochs.
            batch_size: Rows per step.
            patience: Epochs without validation improvement before stopping.
            validation_fraction: Share of rows held out for early stopping.
            random_state: Seed.
        """
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.l2 = l2
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.random_state = random_state

    def output_units(self) -> int:
        """Network outputs (overridden by the classifier).

        Returns:
            The number of output units: 1 for the base engine.
        """
        return 1

    def loss_name(self) -> str:
        """Keras loss identifier.

        Returns:
            The loss passed to `compile`: `"mse"` for the base engine.
        """
        return "mse"

    def encode_target(self, target: Any) -> np.ndarray:
        """Converts the target to what the loss consumes.

        Args:
            target: Training labels or values.

        Returns:
            The target as a float32 array.
        """
        return np.asarray(target, dtype=np.float32)

    def build_network(self, keras: Any, width: int) -> Any:
        """Creates the untrained network.

        Args:
            keras: The imported keras module.
            width: Number of input features.

        Returns:
            A sequential model of ReLU dense layers with dropout and a final
            dense layer of `output_units()` units.
        """
        regularizer = keras.regularizers.l2(self.l2)
        layers = [keras.layers.Input((width,))]
        for size in self.hidden_sizes:
            layers.append(
                keras.layers.Dense(
                    size, activation="relu", kernel_regularizer=regularizer
                )
            )
            layers.append(keras.layers.Dropout(self.dropout))
        layers.append(keras.layers.Dense(self.output_units()))
        return keras.Sequential(layers)

    def fit(
        self, features: Any, target: Any, sample_weight: Any = None
    ) -> "KerasEngine":
        """Trains the network and keeps its weights as numpy arrays.

        Early stopping on a validation split is used only when there are at
        least 30 rows.

        Args:
            features: Training matrix.
            target: Training labels or values.
            sample_weight: Per-row weights; None weights all rows equally.

        Returns:
            This fitted engine.
        """
        keras = load_keras()
        keras.utils.set_random_seed(self.random_state)
        matrix = np.asarray(features, dtype=np.float32)
        encoded = self.encode_target(target)
        self.input_width_ = matrix.shape[1]
        network = self.build_network(keras, self.input_width_)
        network.compile(
            optimizer=keras.optimizers.Adam(self.learning_rate),
            loss=self.loss_name(),
        )
        held = len(matrix) >= MIN_ROWS_FOR_VALIDATION
        callbacks = [keras.callbacks.TerminateOnNaN()]
        if held:
            callbacks.append(
                keras.callbacks.EarlyStopping(
                    patience=self.patience, restore_best_weights=True
                )
            )
        history = network.fit(
            matrix,
            encoded,
            sample_weight=None
            if sample_weight is None
            else np.asarray(sample_weight, dtype=np.float32),
            epochs=self.epochs,
            batch_size=int(self.batch_size),
            validation_split=self.validation_fraction if held else 0.0,
            callbacks=callbacks,
            verbose=0,
            shuffle=True,
        )
        losses = history.history["loss"]
        self.n_iter_ = len(losses)
        self.network_ = network
        self.training_report_ = self.verdict(losses)
        return self

    @staticmethod
    def verdict(losses: list[float]) -> dict[str, Any]:
        """Names a failed run: non-finite loss, or no progress at all.

        Args:
            losses: Training loss of each epoch.

        Returns:
            A mapping with the verdict (`diverged`, `stalled` or `ok`) and the
            number of epochs; a run that did not diverge also carries the
            relative loss improvement.
        """
        if not np.isfinite(losses).all():
            return {"verdict": "diverged", "steps": len(losses)}
        improvement = 0.0
        if len(losses) > 2 and abs(losses[0]) > 0:
            improvement = (losses[0] - losses[-1]) / abs(losses[0])
        stalled = len(losses) > 2 and improvement < STALL_IMPROVEMENT
        return {
            "verdict": "stalled" if stalled else "ok",
            "steps": len(losses),
            "improvement": float(improvement),
        }

    def __getstate__(self) -> dict[str, Any]:
        """Pickles the weights as plain arrays, not the Keras model.

        Returns:
            The instance state with `network_` replaced by `weights_`, a list of
            numpy arrays.
        """
        state = dict(super().__getstate__())
        network = state.pop("network_", None)
        if network is not None:
            state["weights_"] = [
                np.asarray(weight) for weight in network.get_weights()
            ]
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Rebuilds the network and loads the pickled weights.

        Args:
            state: Mapping produced by `__getstate__`.
        """
        state = dict(state)
        weights = state.pop("weights_", None)
        super().__setstate__(state)
        if weights is not None:
            keras = load_keras()
            network = self.build_network(keras, self.input_width_)
            network.set_weights(weights)
            self.network_ = network

    def raw_outputs(self, features: Any) -> np.ndarray:
        """Network outputs for new rows.

        Args:
            features: Numeric matrix with the width seen at fit.

        Returns:
            Array of shape `(n_rows, output_units)`.
        """
        return np.asarray(
            self.network_.predict(
                np.asarray(features, dtype=np.float32), verbose=0
            )
        )


class ClassifierEngine(base.ClassifierMixin, KerasEngine):
    """Classifier: softmax over class scores."""

    def output_units(self) -> int:
        """One unit per class.

        Returns:
            The number of classes seen by `encode_target`.
        """
        return len(self.classes_)

    def loss_name(self) -> str:
        """Cross entropy over integer class codes, from logits.

        Returns:
            A Keras sparse categorical cross-entropy loss with `from_logits`.
        """
        keras = load_keras()
        return keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    def encode_target(self, target: Any) -> np.ndarray:
        """Maps labels to 0..K-1 and remembers the classes.

        Args:
            target: Class labels.

        Returns:
            Integer codes into `classes_`, as int64.
        """
        self.classes_, codes = np.unique(
            np.asarray(target), return_inverse=True
        )
        return codes.astype(np.int64)

    def predict_proba(self, features: Any) -> np.ndarray:
        """Class probabilities, columns ordered like `classes_`.

        Args:
            features: Numeric matrix with the width seen at fit.

        Returns:
            Array of shape `(n_rows, n_classes)`; each row sums to one.
        """
        scores = self.raw_outputs(features)
        scores = scores - scores.max(axis=1, keepdims=True)
        exp = np.exp(scores)
        return exp / exp.sum(axis=1, keepdims=True)

    def predict(self, features: Any) -> np.ndarray:
        """Most probable class.

        Args:
            features: Numeric matrix with the width seen at fit.

        Returns:
            Array of shape `(n_rows,)` of class labels.
        """
        return self.classes_[self.predict_proba(features).argmax(axis=1)]


class RegressorEngine(base.RegressorMixin, KerasEngine):
    """Regressor trained on the standardized target."""

    def encode_target(self, target: Any) -> np.ndarray:
        """Standardizes the target and remembers its scale.

        Args:
            target: Training values.

        Returns:
            The values minus their mean and divided by their standard deviation
            (1 when it is zero), as float32.
        """
        values = np.asarray(target, dtype=np.float64)
        self.target_mean_ = float(values.mean())
        self.target_scale_ = float(values.std()) or 1.0
        return ((values - self.target_mean_) / self.target_scale_).astype(
            np.float32
        )

    def predict(self, features: Any) -> np.ndarray:
        """Predicted values in the original units.

        Args:
            features: Numeric matrix with the width seen at fit.

        Returns:
            Array of shape `(n_rows,)`.
        """
        scaled = self.raw_outputs(features)[:, 0]
        return scaled * self.target_scale_ + self.target_mean_
