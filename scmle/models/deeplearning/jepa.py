"""Public proxy for the optional tabular JEPA representation trainer."""

import io
import json
import pathlib
from typing import Any
from typing import cast

import joblib
import numpy
from sklearn import base
from sklearn.utils import validation

from scmle import envelope
from scmle.contracts import errors
from scmle.models.deeplearning import contracts
from scmle.models.deeplearning import estimators


class Embedder(base.TransformerMixin, estimators.NetworkEstimator):  # type: ignore[misc]  # scikit-learn ships no type information
    """Worker-backed tabular JEPA representation estimator.

    This is a self-supervised numeric-tabular representation path. It is not a
    multimodal JEPA implementation and is not registered as a supervised
    candidate in the default model search.
    """

    engine_path = "scmle.models.deeplearning.jepa_engine.Trainer"
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
        """Stores the explicit JEPA contract and resource policy.

        Args:
            hidden_size: Width of the hidden layers.
            embedding_size: Width of the learned embedding.
            mask_probability: Probability that an input value is masked in the
                context view.
            learning_rate: AdamW learning rate.
            weight_decay: AdamW weight decay.
            epochs: Number of training epochs.
            batch_size: Rows per step, or `auto` for the full data in one batch.
            ema_decay: Decay of the exponential moving average that updates the
                target encoder.
            variance_weight: Weight of the variance-preservation penalty.
            device: `auto`, `cpu` or `cuda`.
            random_state: Seed for initialisation, masking and shuffling.
        """
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
    ) -> "Embedder":
        """Learns a representation without using a target label.

        Pipeline.fit forwards the supervised target to every intermediate
        transformer. JEPA must not consume that target, so this method
        deliberately replaces it with a constant placeholder before the
        worker call. This keeps the representation path self-supervised even
        when it is composed with a downstream classifier or regressor.

        Args:
            features: Numeric training matrix.
            target: Ignored.
            sample_weight: Passed to the worker, whose JEPA engine ignores it.

        Returns:
            This fitted estimator.
        """
        del target
        return cast(
            "Embedder",
            super().fit(
                features,
                numpy.zeros(len(features), dtype=numpy.float32),
                sample_weight,
            ),
        )

    def transform(self, features: Any) -> numpy.ndarray:
        """Returns context-encoder embeddings for new rows.

        Args:
            features: Numeric matrix with the width seen at fit.

        Returns:
            Embedding matrix of shape `(n_rows, embedding_size)`.
        """
        return self.call("encode", features)

    def model_contract(self) -> contracts.Boundary:
        """Returns the explicit tabular JEPA family boundary."""
        return contracts.TABULAR_JEPA_CONTRACT

    def inference_contract(self) -> contracts.Contract:
        """Returns the embedding inference contract."""
        return self.model_contract().inference

    def explanation_contract(self) -> contracts.Explainability:
        """Returns the embedding-sensitivity explanation contract.

        Raises:
            ValueError: If the JEPA family declares no explanation contract.
        """
        explanation = self.model_contract().explanation
        if explanation is None:
            raise errors.InputError("tabular JEPA has no explanation contract")
        return explanation

    def explain(self, features: Any) -> contracts.Contribution:
        """Explains local sensitivity of the learned embedding.

        Args:
            features: Numeric matrix with the width seen at fit.

        Returns:
            Gradient-times-input attributions of the embedding energy, one value
            per row and input feature.
        """
        values = self.call("explain", features)
        return contracts.Contribution(
            values=values,
            method=self.explanation_contract().method,
            target="embedding_energy",
            feature_count=self.state_["input_width_"],
        )

    def inventory(self) -> dict[str, object]:
        """Returns the explicit fitted-checkpoint contract metadata."""
        validation.check_is_fitted(self, attributes=["state_"])
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

    def stash(
        self,
        path: str | pathlib.Path,
        signing_key: str | bytes | None = None,
    ) -> pathlib.Path:
        """Saves the fitted representation as an authenticated envelope.

        The payload holds the estimator configuration, training report and
        NumPy-only worker state; the header holds `inventory`, the
        payload digest and an HMAC (`scmle.envelope`), so nothing is
        unpickled before it is authenticated (SEC-1). An unfitted estimator
        raises scikit-learn's `NotFittedError`.

        Args:
            path: Destination file. Missing parent directories are created.
            signing_key: HMAC key; the configured key is used when None.
                Without any key the checkpoint loads only with `trusted=True`.

        Returns:
            The checkpoint path.
        """
        validation.check_is_fitted(self, attributes=["state_"])
        serialized = io.BytesIO()
        joblib.dump(
            {"format": self.CHECKPOINT_FORMAT, "estimator": self}, serialized
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
    ) -> "Embedder":
        """Loads a checkpoint, authenticating it before anything is unpickled.

        Args:
            path: Checkpoint produced by `stash`.
            signing_key: HMAC key; the configured key is used when None.
            trusted: Whether the caller vouches for an unsigned checkpoint.

        Returns:
            A fitted estimator whose transform output matches the saved model.

        Raises:
            ValueError: If the file cannot be read, is not an envelope, fails
                its digest, or does not hold a matching fitted
                `Embedder`.
            ArtifactError: If the checkpoint is unsigned (or no key is set)
                and `trusted` is false, its signature does not verify, or
                `trusted` is true and `$SCMLE_ENV` is `production`.
        """
        estimators.vouch(trusted)
        try:
            meta, payload = envelope.read(path, signing_key, trusted)
        except OSError as error:
            raise errors.InputError(
                "could not read tabular JEPA checkpoint"
            ) from error
        if (
            meta.get("format") != cls.CHECKPOINT_FORMAT
            or meta.get("schema_version") != cls.CHECKPOINT_SCHEMA_VERSION
        ):
            raise errors.InputError(
                "unsupported tabular JEPA checkpoint format"
            )
        try:
            serialized = joblib.load(io.BytesIO(payload))
        except (EOFError, OSError, ValueError, TypeError) as error:
            raise errors.InputError(
                "tabular JEPA checkpoint payload is invalid"
            ) from error
        if (
            not isinstance(serialized, dict)
            or serialized.get("format") != cls.CHECKPOINT_FORMAT
        ):
            raise errors.InputError(
                "tabular JEPA checkpoint payload format is invalid"
            )
        estimator = serialized.get("estimator")
        if not isinstance(estimator, cls):
            raise errors.InputError("checkpoint does not contain a Embedder")
        validation.check_is_fitted(estimator, attributes=["state_"])
        expected = json.loads(json.dumps(estimator.inventory()))
        if meta != expected:
            raise errors.InputError(
                "tabular JEPA checkpoint metadata does not match"
            )
        return estimator


__all__ = ["Embedder"]
