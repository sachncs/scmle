"""Public proxy for the optional tabular JEPA representation trainer."""

import hashlib
import hmac
import io
import os
import pathlib
import tempfile
from typing import Any

import joblib
import numpy as np
from sklearn import base
from sklearn import utils

from scmle.models.deeplearning import contracts
from scmle.models.deeplearning import estimators


class JepaEstimator(base.TransformerMixin, estimators.NetworkEstimator):
    """Worker-backed tabular JEPA representation estimator.

    This is a self-supervised numeric-tabular representation path. It is not a
    multimodal JEPA implementation and is not registered as a supervised
    candidate in the default model search.
    """

    engine_path = "scmle.models.deeplearning.jepa_engine.JepaEngine"
    CHECKPOINT_FORMAT = "scmle.tabular-jepa.v2"
    CHECKPOINT_SCHEMA_VERSION = 1

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
        """Learn a representation without using a target label.

        Pipeline.fit forwards the supervised target to every intermediate
        transformer. JEPA must not consume that target, so this method
        deliberately replaces it with a constant placeholder before the
        worker call. This keeps the representation path self-supervised even
        when it is composed with a downstream classifier or regressor.
        """
        del target
        return super().fit(
            features,
            np.zeros(len(features), dtype=np.float32),
            sample_weight,
        )

    def transform(self, features: Any) -> np.ndarray:
        """Return context-encoder embeddings for new rows."""
        return self.call("encode", features)

    def model_contract(self) -> contracts.ModelFamilyContract:
        """Return the explicit tabular JEPA family boundary."""
        return contracts.TABULAR_JEPA_CONTRACT

    def inference_contract(self) -> contracts.InferenceContract:
        """Return the embedding inference contract."""
        return self.model_contract().inference

    def explanation_contract(self) -> contracts.ExplanationContract:
        """Return the embedding-sensitivity explanation contract."""
        explanation = self.model_contract().explanation
        if explanation is None:
            raise ValueError("tabular JEPA has no explanation contract")
        return explanation

    def explain(self, features: Any) -> contracts.FeatureExplanation:
        """Explain local sensitivity of the learned embedding."""
        values = self.call("explain", features)
        return contracts.FeatureExplanation(
            values=values,
            method=self.explanation_contract().method,
            target="embedding_energy",
            feature_count=self.state_["input_width_"],
        )

    def checkpoint_manifest(self) -> dict[str, object]:
        """Return the explicit fitted-checkpoint contract metadata."""
        utils.validation.check_is_fitted(self, attributes=["state_"])
        family = self.model_contract()
        return {
            "format": self.CHECKPOINT_FORMAT,
            "schema_version": self.CHECKPOINT_SCHEMA_VERSION,
            "model_family": family.model_family,
            "scope": "tabular_numeric_representation",
            "input_width": self.state_["input_width_"],
            "embedding_size": self.embedding_size,
            "inference": family.inference.to_dict(),
            "explanation": family.explanation.to_dict()
            if family.explanation is not None
            else None,
        }

    def save_checkpoint(self, path: str | pathlib.Path) -> pathlib.Path:
        """Save a fitted JEPA representation checkpoint.

        The checkpoint contains a versioned envelope around the estimator
        configuration, training report, and NumPy-only worker state. The
        envelope includes a SHA-256 digest so truncated or accidentally
        modified files fail closed before the estimator is trusted. It is
        intended for trusted local storage; loading it uses joblib's
        pickle-based format and is not a security boundary.

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
            serialized = io.BytesIO()
            joblib.dump(
                {"format": self.CHECKPOINT_FORMAT, "estimator": self},
                serialized,
            )
            serialized_payload = serialized.getvalue()
            joblib.dump(
                {
                    **self.checkpoint_manifest(),
                    "payload_sha256": hashlib.sha256(
                        serialized_payload
                    ).hexdigest(),
                    "payload": serialized_payload,
                },
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
        try:
            payload = joblib.load(path)
        except (EOFError, OSError, ValueError, TypeError) as error:
            raise ValueError(
                "could not read tabular JEPA checkpoint"
            ) from error
        if (
            not isinstance(payload, dict)
            or payload.get("format") != cls.CHECKPOINT_FORMAT
        ):
            raise ValueError("unsupported tabular JEPA checkpoint format")
        required = {
            "schema_version",
            "scope",
            "model_family",
            "inference",
            "explanation",
            "payload_sha256",
            "input_width",
            "embedding_size",
            "payload",
        }
        if not required.issubset(payload):
            raise ValueError("tabular JEPA checkpoint metadata is incomplete")
        serialized_payload = payload["payload"]
        digest = payload["payload_sha256"]
        explanation = contracts.TABULAR_JEPA_CONTRACT.explanation
        if explanation is None:
            raise ValueError("tabular JEPA explanation contract is missing")
        if (
            payload["schema_version"] != cls.CHECKPOINT_SCHEMA_VERSION
            or payload["scope"] != "tabular_numeric_representation"
            or payload["model_family"]
            != contracts.TABULAR_JEPA_CONTRACT.model_family
            or payload["inference"]
            != contracts.TABULAR_JEPA_CONTRACT.inference.to_dict()
            or payload["explanation"] != explanation.to_dict()
            or not isinstance(serialized_payload, bytes)
            or not isinstance(digest, str)
            or not hmac.compare_digest(
                hashlib.sha256(serialized_payload).hexdigest(), digest
            )
        ):
            raise ValueError("tabular JEPA checkpoint integrity check failed")
        try:
            serialized = joblib.load(io.BytesIO(serialized_payload))
        except (EOFError, OSError, ValueError, TypeError) as error:
            raise ValueError(
                "tabular JEPA checkpoint payload is invalid"
            ) from error
        if (
            not isinstance(serialized, dict)
            or serialized.get("format") != cls.CHECKPOINT_FORMAT
        ):
            raise ValueError(
                "tabular JEPA checkpoint payload format is invalid"
            )
        estimator = serialized.get("estimator")
        if not isinstance(estimator, cls):
            raise ValueError("checkpoint does not contain a JepaEstimator")
        utils.validation.check_is_fitted(estimator, attributes=["state_"])
        if (
            payload["input_width"] != estimator.state_.get("input_width_")
            or payload["embedding_size"] != estimator.embedding_size
        ):
            raise ValueError("tabular JEPA checkpoint metadata does not match")
        return estimator


__all__ = ["JepaEstimator"]
