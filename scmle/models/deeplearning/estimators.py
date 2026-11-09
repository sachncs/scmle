"""Scikit-learn estimators whose computation runs in the PyTorch worker.

These classes never import torch. `fit` ships the (dense, float32) training
matrix to the worker and keeps the returned weights as plain numpy arrays;
`predict` ships rows plus a state id, and the worker loads the weights once
and caches them. Pickling therefore stores only numpy data.
"""

import hashlib
import hmac
import io
import os
import pathlib
import tempfile
from typing import Any
import uuid

import joblib
import numpy as np
from scipy import sparse
from sklearn import base
from sklearn import utils

from scmle import durability
from scmle.models.deeplearning import contracts
from scmle.models.deeplearning import worker

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
        batch_size: Rows per step, or `auto` to size it from the data.
        schedule: Learning-rate schedule, `cosine` or `constant`.
        warmup_epochs: Epochs of linear learning-rate warm-up.
        max_grad_norm: Gradient clipping threshold (None disables it).
        patience: Epochs without validation improvement before stopping.
        validation_fraction: Share of rows held out for early stopping.
        device: `auto` (CUDA when available), `cpu` or `cuda`.
        random_state: Seed for initialization, shuffling and the split.
        optimizer: `adam`, `adamw`, `sgd`, `rmsprop` or `lbfgs`.
        momentum: Momentum of `sgd`.
        loss: `default`, or `focal` (classifier) / `huber` (regressor).
        label_smoothing: Share of probability moved off the true class.
        focal_gamma: Focusing strength of the focal loss.
        huber_delta: Width of the Huber loss's squared region.
        accumulation_steps: Mini-batches summed per parameter update.
    """

    engine_path = ""
    CHECKPOINT_SCHEMA_VERSION = 1

    def __init__(
        self,
        architecture: str = "mlp",
        hidden_sizes: tuple[int, ...] = (64,),
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        epochs: int = 100,
        batch_size: int | str = 128,
        schedule: str = "cosine",
        warmup_epochs: int = 1,
        max_grad_norm: float | None = 5.0,
        patience: int = 10,
        validation_fraction: float = 0.1,
        device: str = "auto",
        random_state: int = 0,
        optimizer: str = "adam",
        momentum: float = 0.9,
        loss: str = "default",
        label_smoothing: float = 0.0,
        focal_gamma: float = 2.0,
        huber_delta: float = 1.0,
        accumulation_steps: int = 1,
    ) -> None:
        """Stores the hyperparameters (scikit-learn convention).

        Args:
            architecture: Name in `architectures.ARCHITECTURES`.
            hidden_sizes: Layer widths (residual: width, then block count).
            dropout: Dropout probability.
            learning_rate: Adam step size.
            weight_decay: L2 penalty.
            epochs: Maximum epochs.
            batch_size: Rows per step, or `auto` to size it from the data.
            schedule: Learning-rate schedule, `cosine` or `constant`.
            warmup_epochs: Epochs of linear learning-rate warm-up.
            max_grad_norm: Gradient clipping threshold (None disables it).
            patience: Epochs without validation improvement before stopping.
            validation_fraction: Share of rows held out for early stopping.
            device: `auto` (CUDA when available), `cpu` or `cuda`.
            random_state: Seed for initialization, shuffling and the split.
            optimizer: `adam`, `adamw`, `sgd` (with `momentum`), `rmsprop` or
                `lbfgs` (full-batch second-order steps).
            momentum: Momentum of `sgd`.
            loss: `default`, or `focal` (classifier) / `huber` (regressor).
            label_smoothing: Share of probability moved off the true class.
            focal_gamma: Focusing strength of the focal loss.
            huber_delta: Where the Huber loss turns from squared to linear (in
                standardized target units).
            accumulation_steps: Mini-batches whose gradients are summed before
                one parameter update (a larger effective batch).
        """
        self.optimizer = optimizer
        self.momentum = momentum
        self.loss = loss
        self.label_smoothing = label_smoothing
        self.focal_gamma = focal_gamma
        self.huber_delta = huber_delta
        self.accumulation_steps = accumulation_steps
        self.architecture = architecture
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.schedule = schedule
        self.warmup_epochs = warmup_epochs
        self.max_grad_norm = max_grad_norm
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.device = device
        self.random_state = random_state

    @staticmethod
    def dense(features: Any) -> np.ndarray:
        """Returns a float32 matrix, densifying sparse input.

        Args:
            features: Dense or sparse array-like of model inputs.

        Returns:
            Dense float32 array.
        """
        if sparse.issparse(features):
            features = features.toarray()
        return np.asarray(features, dtype=np.float32)

    def model_contract(self) -> contracts.ModelFamilyContract:
        """Return the explicit model-family boundary for this estimator."""
        return contracts.model_family_contract(self.architecture)

    def inference_contract(self) -> contracts.InferenceContract:
        """Return the input, output, and inference-method contract."""
        return self.model_contract().inference

    def explanation_contract(self) -> contracts.ExplanationContract:
        """Return the local feature-attribution contract.

        Raises:
            ValueError: If the model family declares no explanation contract.
        """
        explanation = self.model_contract().explanation
        if explanation is None:
            raise ValueError(
                f"{self.model_contract().model_family} has no explanation "
                "contract"
            )
        return explanation

    def checkpoint_manifest(self) -> dict[str, object]:
        """Return the fitted checkpoint metadata without serializing weights."""
        utils.validation.check_is_fitted(self, attributes=["state_"])
        family = self.model_contract()
        return {
            "format": family.checkpoint.format,
            "schema_version": self.CHECKPOINT_SCHEMA_VERSION,
            "model_family": family.model_family,
            "architecture": self.architecture,
            "modality": family.inference.modality.value,
            "input_width": self.state_["input_width_"],
            "inference": family.inference.to_dict(),
            "explanation": (
                None
                if family.explanation is None
                else family.explanation.to_dict()
            ),
        }

    def fit(
        self, features: Any, target: Any, sample_weight: Any = None
    ) -> "NetworkEstimator":
        """Trains the network in the worker.

        Features that fail the contract validation (not a finite numeric 2-D
        matrix) raise ValueError.

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
        matrix = self.model_contract().validate_features(features)
        state = worker.client().request(
            "fit",
            (
                self.engine_path,
                self.get_params(),
                matrix,
                np.asarray(target),
                weights,
            ),
        )
        self.state_ = state
        self.state_id_ = uuid.uuid4().hex
        self.n_iter_ = state["n_iter_"]
        self.training_report_ = state["training_report_"]
        self.remember(state)
        return self

    def remember(self, state: dict[str, Any]) -> None:
        """Copies the attributes scikit-learn expects from the fitted state.

        Args:
            state: Fitted state returned by the worker.
        """

    def call(self, method: str, features: Any) -> np.ndarray:
        """Runs an engine method in the worker on new rows.

        If the worker does not hold the fitted state it is sent once and the
        call is retried. Features that are not a finite numeric matrix of the
        fitted width raise ValueError from the contract validation.

        Args:
            method: Name of the engine method to call, for example `predict`.
            features: Numeric matrix with the width seen at fit.

        Returns:
            The method's result as a numpy array.

        Raises:
            WorkerError: If the worker still lacks the model state after it was
                sent.
        """
        utils.validation.check_is_fitted(self, attributes=["state_"])
        matrix = self.model_contract().validate_features(
            features, expected_width=self.state_["input_width_"]
        )
        client = worker.client()
        for state in (None, self.state_):
            reply = client.request(
                "call",
                (self.engine_path, self.state_id_, state, method, matrix),
            )
            if not (isinstance(reply, str) and reply == NEED_STATE):
                return np.asarray(reply)
        raise worker.WorkerError("the worker could not load the model state")

    def explain(self, features: Any) -> contracts.FeatureExplanation:
        """Return local gradient-times-input feature attributions.

        The result explains the model output selected by the fitted worker:
        the predicted class logit for classification, the averaged ensemble
        value for regression, or the embedding energy for a representation
        estimator. These are sensitivity scores, not causal effects.

        Args:
            features: Numeric matrix with the width seen at fit.

        Returns:
            A `FeatureExplanation` with one signed attribution per row and input
            feature.
        """
        values = self.call("explain", features)
        return contracts.FeatureExplanation(
            values=values,
            method=self.explanation_contract().method,
            target="predicted_output",
            feature_count=self.state_["input_width_"],
        )

    def save_checkpoint(self, path: str | pathlib.Path) -> pathlib.Path:
        """Atomically save a verified, trusted-local fitted checkpoint.

        An unfitted estimator raises scikit-learn's `NotFittedError`.

        Args:
            path: Destination file. Missing parent directories are created.

        Returns:
            The checkpoint path.
        """
        utils.validation.check_is_fitted(self, attributes=["state_"])
        destination = pathlib.Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        family = self.model_contract()
        serialized = io.BytesIO()
        joblib.dump(
            {"format": family.checkpoint.format, "estimator": self},
            serialized,
        )
        serialized_payload = serialized.getvalue()
        envelope = {
            **self.checkpoint_manifest(),
            "payload_sha256": hashlib.sha256(serialized_payload).hexdigest(),
            "payload": serialized_payload,
        }
        temporary: pathlib.Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = pathlib.Path(handle.name)
            joblib.dump(envelope, temporary)
            durability.sync_file(temporary)
            os.replace(temporary, destination)
            durability.sync_directory(destination.parent)
        except BaseException:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            raise
        return destination

    @classmethod
    def load_checkpoint(cls, path: str | pathlib.Path) -> "NetworkEstimator":
        """Load a verified checkpoint for this estimator class.

        Checkpoints are trusted-local joblib payloads. The digest detects
        truncation and accidental modification; it is not a protection
        against a malicious pickle.

        Args:
            path: Checkpoint produced by `save_checkpoint`.

        Returns:
            The fitted estimator stored in the checkpoint.

        Raises:
            ValueError: If the file cannot be read, the envelope, metadata or
                digest check fails, or the payload does not hold a fitted
                estimator of this class that matches the metadata.
        """
        try:
            envelope = joblib.load(path)
        except Exception as error:
            raise ValueError(
                "could not read deep-learning checkpoint"
            ) from error
        if not isinstance(envelope, dict):
            raise ValueError("deep-learning checkpoint envelope is invalid")
        required = {
            "format",
            "schema_version",
            "model_family",
            "architecture",
            "modality",
            "input_width",
            "inference",
            "explanation",
            "payload_sha256",
            "payload",
        }
        if not required.issubset(envelope):
            raise ValueError("deep-learning checkpoint metadata is incomplete")
        serialized_payload = envelope["payload"]
        digest = envelope["payload_sha256"]
        if (
            envelope["schema_version"] != cls.CHECKPOINT_SCHEMA_VERSION
            or not isinstance(serialized_payload, bytes)
            or not isinstance(digest, str)
            or not hmac.compare_digest(
                hashlib.sha256(serialized_payload).hexdigest(), digest
            )
        ):
            raise ValueError("deep-learning checkpoint integrity check failed")
        try:
            serialized = joblib.load(io.BytesIO(serialized_payload))
        except Exception as error:
            raise ValueError(
                "deep-learning checkpoint payload is invalid"
            ) from error
        estimator = (
            serialized.get("estimator")
            if isinstance(serialized, dict)
            else None
        )
        if not isinstance(estimator, cls):
            raise ValueError(f"checkpoint does not contain a {cls.__name__}")
        utils.validation.check_is_fitted(estimator, attributes=["state_"])
        family = estimator.model_contract()
        if (
            serialized.get("format") != family.checkpoint.format
            or envelope["format"] != family.checkpoint.format
            or envelope["model_family"] != family.model_family
            or envelope["architecture"] != estimator.architecture
            or envelope["modality"] != family.inference.modality.value
            or envelope["input_width"] != estimator.state_["input_width_"]
            or envelope["inference"] != family.inference.to_dict()
            or envelope["explanation"]
            != (
                None
                if family.explanation is None
                else family.explanation.to_dict()
            )
        ):
            raise ValueError("deep-learning checkpoint metadata does not match")
        return estimator


class TorchClassifier(base.ClassifierMixin, NetworkEstimator):
    """Classifier: softmax over class scores, optional class balancing."""

    engine_path = "scmle.models.deeplearning.engines.ClassifierEngine"

    def __init__(
        self,
        architecture: str = "mlp",
        hidden_sizes: tuple[int, ...] = (64,),
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        epochs: int = 100,
        batch_size: int | str = 128,
        schedule: str = "cosine",
        warmup_epochs: int = 1,
        max_grad_norm: float | None = 5.0,
        patience: int = 10,
        validation_fraction: float = 0.1,
        device: str = "auto",
        random_state: int = 0,
        optimizer: str = "adam",
        momentum: float = 0.9,
        loss: str = "default",
        label_smoothing: float = 0.0,
        focal_gamma: float = 2.0,
        huber_delta: float = 1.0,
        accumulation_steps: int = 1,
        class_weight: str | None = None,
    ) -> None:
        """Stores the hyperparameters, plus `class_weight` (`balanced`).

        Args:
            architecture: Name in `architectures.ARCHITECTURES`.
            hidden_sizes: Layer widths (residual: width, then block count).
            dropout: Dropout probability.
            learning_rate: Adam step size.
            weight_decay: L2 penalty.
            epochs: Maximum epochs.
            batch_size: Rows per step, or `auto` to size it from the data.
            schedule: Learning-rate schedule, `cosine` or `constant`.
            warmup_epochs: Epochs of linear learning-rate warm-up.
            max_grad_norm: Gradient clipping threshold (None disables it).
            patience: Epochs without validation improvement before stopping.
            validation_fraction: Share of rows held out for early stopping.
            device: `auto` (CUDA when available), `cpu` or `cuda`.
            random_state: Seed for initialization, shuffling and the split.
            optimizer: `adam`, `adamw`, `sgd` (with `momentum`), `rmsprop` or
                `lbfgs` (full-batch second-order steps).
            momentum: Momentum of `sgd`.
            loss: `default`, or `focal` (classifier) / `huber` (regressor).
            label_smoothing: Share of probability moved off the true class.
            focal_gamma: Focusing strength of the focal loss.
            huber_delta: Where the Huber loss turns from squared to linear (in
                standardized target units).
            accumulation_steps: Mini-batches whose gradients are summed before
                one parameter update (a larger effective batch).
            class_weight: `balanced` for inverse-class-frequency row weights, or
                None for equal weights.
        """
        super().__init__(
            architecture,
            hidden_sizes,
            dropout,
            learning_rate,
            weight_decay,
            epochs,
            batch_size,
            schedule,
            warmup_epochs,
            max_grad_norm,
            patience,
            validation_fraction,
            device,
            random_state,
            optimizer,
            momentum,
            loss,
            label_smoothing,
            focal_gamma,
            huber_delta,
            accumulation_steps,
        )
        self.class_weight = class_weight

    def remember(self, state: dict[str, Any]) -> None:
        """Exposes `classes_` after fitting.

        Args:
            state: Fitted state returned by the worker.
        """
        self.classes_ = state["classes_"]

    def predict_proba(self, features: Any) -> np.ndarray:
        """Class probabilities, columns ordered like `classes_`.

        Args:
            features: Numeric matrix with the width seen at fit.

        Returns:
            Array of shape `(n_rows, n_classes)`.
        """
        return self.call("predict_proba", features)

    def predict(self, features: Any) -> np.ndarray:
        """Most probable class.

        Args:
            features: Numeric matrix with the width seen at fit.

        Returns:
            Array of shape `(n_rows,)` of class labels.
        """
        return self.call("predict", features)


class TorchRegressor(base.RegressorMixin, NetworkEstimator):
    """Regressor trained on the standardized target with squared error."""

    engine_path = "scmle.models.deeplearning.engines.RegressorEngine"

    def predict(self, features: Any) -> np.ndarray:
        """Predicted values in the original units.

        Args:
            features: Numeric matrix with the width seen at fit.

        Returns:
            Array of shape `(n_rows,)`.
        """
        return self.call("predict", features)


class TabPfnEstimator(base.BaseEstimator):
    """Worker-backed TabPFN with the scikit-learn estimator interface.

    Attributes:
        device: `auto`, `cpu` or `cuda`.
        random_state: Seed of TabPFN's internal ensemble.
        n_estimators: Members of TabPFN's internal ensemble.
    """

    engine_path = ""

    def __init__(
        self,
        device: str = "auto",
        random_state: int = 0,
        n_estimators: int = 8,
    ) -> None:
        """Stores the settings (scikit-learn convention).

        Args:
            device: `auto`, `cpu` or `cuda`.
            random_state: Seed of TabPFN's internal ensemble.
            n_estimators: Members of TabPFN's internal ensemble.
        """
        self.device = device
        self.random_state = random_state
        self.n_estimators = n_estimators

    def fit(self, features: Any, target: Any) -> "TabPfnEstimator":
        """Stores the rows in the worker's TabPFN model.

        Args:
            features: Training matrix.
            target: Labels or values.

        Returns:
            The fitted estimator.
        """
        matrix = NetworkEstimator.dense(features)
        self.state_ = worker.client().request(
            "fit",
            (
                self.engine_path,
                self.get_params(),
                matrix,
                np.asarray(target),
                None,
            ),
        )
        self.state_id_ = uuid.uuid4().hex
        self.remember(self.state_)
        return self

    def remember(self, state: dict[str, Any]) -> None:
        """Copies attributes scikit-learn expects from the fitted state.

        Args:
            state: Fitted state returned by the worker.
        """

    def call(self, method: str, features: Any) -> np.ndarray:
        """Runs an engine method in the worker on new rows.

        If the worker does not hold the fitted state it is sent once and the
        call is retried.

        Args:
            method: Name of the engine method to call, for example `predict`.
            features: Numeric matrix with the width seen at fit.

        Returns:
            The method's result as a numpy array.

        Raises:
            WorkerError: If the worker still lacks the model state after it was
                sent.
        """
        matrix = NetworkEstimator.dense(features)
        client = worker.client()
        for state in (None, self.state_):
            reply = client.request(
                "call",
                (self.engine_path, self.state_id_, state, method, matrix),
            )
            if not (isinstance(reply, str) and reply == NEED_STATE):
                return np.asarray(reply)
        raise worker.WorkerError("the worker could not load the model state")


class TabPfnClassifier(base.ClassifierMixin, TabPfnEstimator):
    """TabPFN classifier."""

    engine_path = "scmle.models.deeplearning.foundation.ClassifierEngine"

    def remember(self, state: dict[str, Any]) -> None:
        """Exposes `classes_` after fitting.

        Args:
            state: Fitted state returned by the worker.
        """
        self.classes_ = state["classes_"]

    def predict_proba(self, features: Any) -> np.ndarray:
        """Class probabilities.

        Args:
            features: Numeric matrix with the width seen at fit.

        Returns:
            Array of shape `(n_rows, n_classes)`.
        """
        return self.call("predict_proba", features)

    def predict(self, features: Any) -> np.ndarray:
        """Most probable class.

        Args:
            features: Numeric matrix with the width seen at fit.

        Returns:
            Array of shape `(n_rows,)` of class labels.
        """
        return self.call("predict", features)


class TabPfnRegressor(base.RegressorMixin, TabPfnEstimator):
    """TabPFN regressor."""

    engine_path = "scmle.models.deeplearning.foundation.RegressorEngine"

    def predict(self, features: Any) -> np.ndarray:
        """Predicted values.

        Args:
            features: Numeric matrix with the width seen at fit.

        Returns:
            Array of shape `(n_rows,)`.
        """
        return self.call("predict", features)


class KerasEstimator(NetworkEstimator):
    """Worker-backed Keras network with the scikit-learn estimator interface.

    Attributes:
        hidden_sizes: Layer widths.
        dropout: Dropout probability.
        learning_rate: Adam step size.
        l2: L2 penalty on the kernels.
        epochs: Maximum epochs.
        batch_size: Rows per step.
        patience: Epochs without validation improvement before stopping.
        validation_fraction: Share of rows held out for early stopping.
        random_state: Seed.
    """

    architecture = "mlp"

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
            dropout: Dropout probability.
            learning_rate: Adam step size.
            l2: L2 penalty on the kernels.
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


class KerasClassifier(base.ClassifierMixin, KerasEstimator):
    """Keras classifier."""

    engine_path = "scmle.models.deeplearning.keras_engine.ClassifierEngine"

    def remember(self, state: dict[str, Any]) -> None:
        """Exposes `classes_` after fitting.

        Args:
            state: Fitted state returned by the worker.
        """
        self.classes_ = state["classes_"]

    def predict_proba(self, features: Any) -> np.ndarray:
        """Class probabilities.

        Args:
            features: Numeric matrix with the width seen at fit.

        Returns:
            Array of shape `(n_rows, n_classes)`.
        """
        return self.call("predict_proba", features)

    def predict(self, features: Any) -> np.ndarray:
        """Most probable class.

        Args:
            features: Numeric matrix with the width seen at fit.

        Returns:
            Array of shape `(n_rows,)` of class labels.
        """
        return self.call("predict", features)


class KerasRegressor(base.RegressorMixin, KerasEstimator):
    """Keras regressor."""

    engine_path = "scmle.models.deeplearning.keras_engine.RegressorEngine"

    def predict(self, features: Any) -> np.ndarray:
        """Predicted values.

        Args:
            features: Numeric matrix with the width seen at fit.

        Returns:
            Array of shape `(n_rows,)`.
        """
        return self.call("predict", features)
