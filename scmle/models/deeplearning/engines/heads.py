"""The classification and regression engines."""

from typing import Any

import numpy
from sklearn import base

from scmle.models.deeplearning.engines.trainer import Trainer


class Classifier(base.ClassifierMixin, Trainer):  # type: ignore[misc]  # scikit-learn ships no type information
    """Classifier: softmax over class scores, optional class balancing."""

    losses = ("default", "focal")

    __module__ = "scmle.models.deeplearning.engines"

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
            accumulation_steps: Mini-batches summed into one update.
            class_weight: `balanced` for inverse-frequency weights, else None.
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

    def vectorize(self, target: Any) -> numpy.ndarray:
        """Maps labels to 0..K-1 and remembers the classes.

        Args:
            target: Class labels.

        Returns:
            Integer codes into `classes_`, as int64.
        """
        self.classes_, codes = numpy.unique(
            numpy.asarray(target), return_inverse=True
        )
        return codes.astype(numpy.int64)

    def dimensionality(self) -> int:
        """One unit per class.

        Returns:
            The number of classes seen by `vectorize`.
        """
        return len(self.classes_)

    def penalize(self, torch: Any, outputs: Any, targets: Any) -> Any:
        """Cross entropy per row, optionally smoothed or focal.

        Args:
            torch: The imported torch module.
            outputs: Network outputs, one row per (example, member) pair.
            targets: Encoded targets repeated once per member.

        Returns:
            Tensor with one loss per (example, member) pair.
        """
        entropy = torch.nn.functional.cross_entropy(
            outputs,
            targets,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        if self.loss == "focal":
            confidence = torch.exp(-entropy)
            return (1.0 - confidence) ** self.focal_gamma * entropy
        return entropy

    def weighting(self, targets: numpy.ndarray) -> numpy.ndarray:
        """Inverse class frequency when `class_weight` is `balanced`.

        Args:
            targets: Integer class codes.

        Returns:
            Float32 array with one weight per row; all ones unless balanced.
        """
        weights = numpy.ones(len(targets), dtype=numpy.float32)
        if self.class_weight == "balanced":
            counts = numpy.bincount(targets, minlength=len(self.classes_))
            per_class = len(targets) / (len(counts) * numpy.maximum(counts, 1))
            weights = per_class[targets].astype(numpy.float32)
        return weights

    def predict_proba(self, features: Any) -> numpy.ndarray:
        """Class probabilities, columns ordered like `classes_`.

        Args:
            features: Numeric matrix (dense or sparse) of the width seen at fit.

        Returns:
            Array of shape `(n_rows, n_classes)`: the softmax probabilities
            averaged over ensemble members.
        """
        scores = self.raw_outputs(features)
        scores = scores - scores.max(axis=2, keepdims=True)
        exp = numpy.exp(scores)
        probabilities = exp / exp.sum(axis=2, keepdims=True)
        averaged: numpy.ndarray = probabilities.mean(axis=1)
        return averaged

    def predict(self, features: Any) -> numpy.ndarray:
        """Most probable class.

        Args:
            features: Numeric matrix (dense or sparse) of the width seen at fit.

        Returns:
            Array of shape `(n_rows,)` of class labels.
        """
        return numpy.asarray(
            self.classes_[self.predict_proba(features).argmax(axis=1)]
        )


class Regressor(base.RegressorMixin, Trainer):  # type: ignore[misc]  # scikit-learn ships no type information
    """Regressor trained on the standardized target (squared or Huber)."""

    losses = ("default", "huber")

    __module__ = "scmle.models.deeplearning.engines"

    def vectorize(self, target: Any) -> numpy.ndarray:
        """Standardizes the target and remembers its scale.

        Args:
            target: Training values.

        Returns:
            The values minus their mean and divided by their standard deviation
            (1 when it is zero), as float32.
        """
        values = numpy.asarray(target, dtype=numpy.float64)
        self.target_mean_ = float(values.mean())
        self.target_scale_ = float(values.std()) or 1.0
        return ((values - self.target_mean_) / self.target_scale_).astype(
            numpy.float32
        )

    def dimensionality(self) -> int:
        """A single value.

        Returns:
            1.
        """
        return 1

    def penalize(self, torch: Any, outputs: Any, targets: Any) -> Any:
        """Squared error per row, or Huber when configured.

        Args:
            torch: The imported torch module.
            outputs: Network outputs, one row per (example, member) pair.
            targets: Encoded targets repeated once per member.

        Returns:
            Tensor with one loss per (example, member) pair.
        """
        if self.loss == "huber":
            return torch.nn.functional.huber_loss(
                outputs.squeeze(-1),
                targets,
                reduction="none",
                delta=self.huber_delta,
            )
        return (outputs.squeeze(-1) - targets) ** 2

    def weighting(self, targets: numpy.ndarray) -> numpy.ndarray:
        """Unit weights.

        Args:
            targets: Encoded targets.

        Returns:
            Float32 array of ones, one per row.
        """
        return numpy.ones(len(targets), dtype=numpy.float32)

    def predict(self, features: Any) -> numpy.ndarray:
        """Predicted values in the original units.

        Args:
            features: Numeric matrix (dense or sparse) of the width seen at fit.

        Returns:
            Array of shape `(n_rows,)`, averaged over ensemble members.
        """
        scaled = self.raw_outputs(features)[:, :, 0].mean(axis=1)
        return numpy.asarray(scaled * self.target_scale_ + self.target_mean_)
