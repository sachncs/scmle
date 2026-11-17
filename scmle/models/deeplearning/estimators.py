"""Scikit-learn estimators whose computation runs in the PyTorch worker.

These classes never import torch. `fit` ships the (dense, float32) training
matrix to the worker and keeps the returned weights as plain numpy arrays;
`predict` ships rows plus a state id, and the worker loads the weights once
and caches them. Pickling therefore stores only numpy data.
"""

import io
import json
import pathlib
from typing import Any
import uuid

import joblib
import numpy
from scipy import sparse
from sklearn import base
from sklearn.utils import validation

from scmle import envelope
from scmle.contracts import errors
from scmle.models.deeplearning import contracts
from scmle.models.deeplearning import worker


class NetworkEstimator(base.BaseEstimator):  # type: ignore[misc]  # scikit-learn ships no type information
    """Common behaviour of the worker-backed estimators.

    Attributes:
        architecture: Name in `architectures.ARCHITECTURES`.
        hidden_sizes: Layer widths. `residual` uses the first entry as the
            block width and adds one block per non-zero entry;
            `ft_transformer` uses it as the embedding width and builds
            one layer per entry.
        dropout: Dropout probability.
        learning_rate: Peak step size (not used by `lbfgs`).
        weight_decay: L2 penalty (not used by `lbfgs`).
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
            hidden_sizes: Layer widths. `residual` uses the first entry as the
                block width and adds one block per non-zero entry;
                `ft_transformer` uses it as the embedding width and builds
                one layer per entry.
            dropout: Dropout probability.
            learning_rate: Peak step size (not used by `lbfgs`).
            weight_decay: L2 penalty (not used by `lbfgs`).
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
    def dense(features: Any) -> numpy.ndarray:
        """Returns a float32 matrix, densifying sparse input.

        Args:
            features: Dense or sparse array-like of model inputs.

        Returns:
            Dense float32 array.
        """
        if sparse.issparse(features):
            features = features.toarray()
        return numpy.asarray(features, dtype=numpy.float32)

    def model_contract(self) -> contracts.Boundary:
        """Returns the explicit model-family boundary for this estimator."""
        return contracts.resolve(self.architecture)

    def inference_contract(self) -> contracts.Contract:
        """Returns the input, output, and inference-method contract."""
        return self.model_contract().inference

    def explanation_contract(self) -> contracts.Explainability:
        """Returns the local feature-attribution contract.

        Raises:
            ValueError: If the model family declares no explanation contract.
        """
        explanation = self.model_contract().explanation
        if explanation is None:
            raise errors.InputError(
                f"{self.model_contract().model_family} has no explanation "
                "contract"
            )
        return explanation

    def inventory(self) -> dict[str, object]:
        """Returns the fitted checkpoint metadata, without the weights."""
        validation.check_is_fitted(self, attributes=["state_"])
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
            else numpy.asarray(sample_weight, dtype=numpy.float32)
        )
        matrix = self.model_contract().validate(features)
        state = worker.client().request(
            "fit",
            (
                self.engine_path,
                self.get_params(),
                matrix,
                numpy.asarray(target),
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

    def call(self, method: str, features: Any) -> numpy.ndarray:
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
        validation.check_is_fitted(self, attributes=["state_"])
        matrix = self.model_contract().validate(
            features, expected_width=self.state_["input_width_"]
        )
        client = worker.client()
        for state in (None, self.state_):
            reply = client.request(
                "call",
                (self.engine_path, self.state_id_, state, method, matrix),
            )
            if not (isinstance(reply, str) and reply == worker.NEED_STATE):
                return numpy.asarray(reply)
        raise worker.WorkerError("the worker could not load the model state")

    def explain(self, features: Any) -> contracts.Contribution:
        """Returns local gradient-times-input feature attributions.

        The result explains the model output selected by the fitted worker:
        the predicted class logit for classification, the averaged ensemble
        value for regression, or the embedding energy for a representation
        estimator. These are sensitivity scores, not causal effects.

        Args:
            features: Numeric matrix with the width seen at fit.

        Returns:
            A `Contribution` with one signed attribution per row and input
            feature.
        """
        values = self.call("explain", features)
        return contracts.Contribution(
            values=values,
            method=self.explanation_contract().method,
            target="predicted_output",
            feature_count=self.state_["input_width_"],
        )

    def stash(
        self,
        path: str | pathlib.Path,
        signing_key: str | bytes | None = None,
    ) -> pathlib.Path:
        """Atomically saves the fitted estimator as an authenticated envelope.

        An unfitted estimator raises scikit-learn's `NotFittedError`.

        Args:
            path: Destination file. Missing parent directories are created.
            signing_key: HMAC key; the configured key is used when None.
                Without any key the checkpoint can be loaded only with
                `trusted=True` (SEC-1).

        Returns:
            The checkpoint path.
        """
        validation.check_is_fitted(self, attributes=["state_"])
        family = self.model_contract()
        serialized = io.BytesIO()
        joblib.dump(
            {"format": family.checkpoint.format, "estimator": self},
            serialized,
        )
        return envelope.seal(
            path, self.inventory(), serialized.getvalue(), signing_key
        )

    @classmethod
    def restore(
        cls,
        path: str | pathlib.Path,
        signing_key: str | bytes | None = None,
        trusted: bool = False,
    ) -> "NetworkEstimator":
        """Loads a checkpoint, authenticating it before anything is unpickled.

        Args:
            path: Checkpoint produced by `stash`.
            signing_key: HMAC key; the configured key is used when None.
            trusted: Whether the caller vouches for an unsigned checkpoint.

        Returns:
            The fitted estimator stored in the checkpoint.

        Raises:
            ValueError: If the file cannot be read, is not an envelope, fails
                its digest, or does not hold a fitted estimator of this class
                that matches the metadata.
            ArtifactError: If the checkpoint is unsigned (or no key is set)
                and `trusted` is false, its signature does not verify, or
                `trusted` is true and `$SCMLE_ENV` is `production`.
        """
        try:
            meta, payload = envelope.read(path, signing_key, trusted)
        except OSError as error:
            raise errors.InputError(
                "could not read deep-learning checkpoint"
            ) from error
        if meta.get("schema_version") != cls.CHECKPOINT_SCHEMA_VERSION:
            raise errors.InputError(
                "deep-learning checkpoint integrity check failed"
            )
        try:
            serialized = joblib.load(io.BytesIO(payload))
        except Exception as error:
            raise errors.InputError(
                "deep-learning checkpoint payload is invalid"
            ) from error
        estimator = (
            serialized.get("estimator")
            if isinstance(serialized, dict)
            else None
        )
        if not isinstance(estimator, cls):
            raise errors.InputError(
                f"checkpoint does not contain a {cls.__name__}"
            )
        validation.check_is_fitted(estimator, attributes=["state_"])
        expected = json.loads(json.dumps(estimator.inventory()))
        if (
            serialized.get("format")
            != estimator.model_contract().checkpoint.format
            or meta != expected
        ):
            raise errors.InputError(
                "deep-learning checkpoint metadata does not match"
            )
        return estimator


class TorchClassifier(base.ClassifierMixin, NetworkEstimator):  # type: ignore[misc]  # scikit-learn ships no type information
    """Classifier: softmax over class scores, optional class balancing."""

    engine_path = "scmle.models.deeplearning.engines.Classifier"

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
            hidden_sizes: Layer widths. `residual` uses the first entry as the
                block width and adds one block per non-zero entry;
                `ft_transformer` uses it as the embedding width and builds
                one layer per entry.
            dropout: Dropout probability.
            learning_rate: Peak step size (not used by `lbfgs`).
            weight_decay: L2 penalty (not used by `lbfgs`).
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

    def predict_proba(self, features: Any) -> numpy.ndarray:
        """Class probabilities, columns ordered like `classes_`.

        Args:
            features: Numeric matrix with the width seen at fit.

        Returns:
            Array of shape `(n_rows, n_classes)`.
        """
        return self.call("predict_proba", features)

    def predict(self, features: Any) -> numpy.ndarray:
        """Most probable class.

        Args:
            features: Numeric matrix with the width seen at fit.

        Returns:
            Array of shape `(n_rows,)` of class labels.
        """
        return self.call("predict", features)


class TorchRegressor(base.RegressorMixin, NetworkEstimator):  # type: ignore[misc]  # scikit-learn ships no type information
    """Regressor trained on the standardized target with squared error."""

    engine_path = "scmle.models.deeplearning.engines.Regressor"

    def predict(self, features: Any) -> numpy.ndarray:
        """Predicted values in the original units.

        Args:
            features: Numeric matrix with the width seen at fit.

        Returns:
            Array of shape `(n_rows,)`.
        """
        return self.call("predict", features)


class TabPfn(base.BaseEstimator):  # type: ignore[misc]  # scikit-learn ships no type information
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

    def fit(self, features: Any, target: Any) -> "TabPfn":
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
                numpy.asarray(target),
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

    def call(self, method: str, features: Any) -> numpy.ndarray:
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
            if not (isinstance(reply, str) and reply == worker.NEED_STATE):
                return numpy.asarray(reply)
        raise worker.WorkerError("the worker could not load the model state")


class TabPfnClassifier(base.ClassifierMixin, TabPfn):  # type: ignore[misc]  # scikit-learn ships no type information
    """TabPFN classifier."""

    engine_path = "scmle.models.deeplearning.foundation.Classifier"

    def remember(self, state: dict[str, Any]) -> None:
        """Exposes `classes_` after fitting.

        Args:
            state: Fitted state returned by the worker.
        """
        self.classes_ = state["classes_"]

    def predict_proba(self, features: Any) -> numpy.ndarray:
        """Class probabilities.

        Args:
            features: Numeric matrix with the width seen at fit.

        Returns:
            Array of shape `(n_rows, n_classes)`.
        """
        return self.call("predict_proba", features)

    def predict(self, features: Any) -> numpy.ndarray:
        """Most probable class.

        Args:
            features: Numeric matrix with the width seen at fit.

        Returns:
            Array of shape `(n_rows,)` of class labels.
        """
        return self.call("predict", features)


class TabPfnRegressor(base.RegressorMixin, TabPfn):  # type: ignore[misc]  # scikit-learn ships no type information
    """TabPFN regressor."""

    engine_path = "scmle.models.deeplearning.foundation.Regressor"

    def predict(self, features: Any) -> numpy.ndarray:
        """Predicted values.

        Args:
            features: Numeric matrix with the width seen at fit.

        Returns:
            Array of shape `(n_rows,)`.
        """
        return self.call("predict", features)


class Keras(NetworkEstimator):
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


class KerasClassifier(base.ClassifierMixin, Keras):  # type: ignore[misc]  # scikit-learn ships no type information
    """Keras classifier."""

    engine_path = "scmle.models.deeplearning.keras_engine.Classifier"

    def remember(self, state: dict[str, Any]) -> None:
        """Exposes `classes_` after fitting.

        Args:
            state: Fitted state returned by the worker.
        """
        self.classes_ = state["classes_"]

    def predict_proba(self, features: Any) -> numpy.ndarray:
        """Class probabilities.

        Args:
            features: Numeric matrix with the width seen at fit.

        Returns:
            Array of shape `(n_rows, n_classes)`.
        """
        return self.call("predict_proba", features)

    def predict(self, features: Any) -> numpy.ndarray:
        """Most probable class.

        Args:
            features: Numeric matrix with the width seen at fit.

        Returns:
            Array of shape `(n_rows,)` of class labels.
        """
        return self.call("predict", features)


class KerasRegressor(base.RegressorMixin, Keras):  # type: ignore[misc]  # scikit-learn ships no type information
    """Keras regressor."""

    engine_path = "scmle.models.deeplearning.keras_engine.Regressor"

    def predict(self, features: Any) -> numpy.ndarray:
        """Predicted values.

        Args:
            features: Numeric matrix with the width seen at fit.

        Returns:
            Array of shape `(n_rows,)`.
        """
        return self.call("predict", features)
