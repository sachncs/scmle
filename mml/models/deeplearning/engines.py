"""PyTorch training engines. They run only inside the worker process.

Importing this module needs PyTorch, so nothing outside `worker` loads it;
the estimators users see are the proxies in `estimators`.
"""

import abc
import copy
import importlib
import math
from typing import Any

import numpy as np
from scipy import sparse
from sklearn import base

from mml.contracts import errors
from mml.models.deeplearning import architectures
from mml.models.deeplearning import training as training_helpers

MIN_ROWS_FOR_VALIDATION = 30


class TabularEngine(base.BaseEstimator, abc.ABC):
    """Trains a network with Adam, early stopping and weighted losses.

    The matrix is expected to be numeric, complete and scaled (the standard
    pipeline provides that); sparse input is densified.

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
    """

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
    ) -> None:
        """Stores the hyperparameters (scikit-learn convention)."""
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

    @abc.abstractmethod
    def encode_target(self, target: Any) -> np.ndarray:
        """Converts the target to the array the loss consumes."""

    @abc.abstractmethod
    def output_units(self) -> int:
        """Number of network outputs."""

    @abc.abstractmethod
    def row_losses(self, torch: Any, outputs: Any, targets: Any) -> Any:
        """Per-row loss (no reduction)."""

    @abc.abstractmethod
    def row_weights(self, targets: np.ndarray) -> np.ndarray:
        """Per-row weight implied by the estimator's own settings."""

    def resolve_device(self, torch: Any) -> str:
        """Chooses where to train."""
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device

    @staticmethod
    def dense(features: Any) -> np.ndarray:
        """Returns a float32 matrix, densifying sparse input."""
        if sparse.issparse(features):
            features = features.toarray()
        return np.asarray(features, dtype=np.float32)

    def fit(
        self, features: Any, target: Any, sample_weight: Any = None
    ) -> "TabularEngine":
        """Trains the network.

        Args:
            features: Numeric matrix.
            target: Labels or values.
            sample_weight: Optional per-row weights.

        Returns:
            The fitted estimator.

        Raises:
            ConfigError: For an unknown architecture.
        """
        if self.architecture not in architectures.ARCHITECTURES:
            raise errors.ConfigError(
                f"unknown architecture '{self.architecture}'"
            )
        torch = importlib.import_module("torch")
        matrix = self.dense(features)
        encoded = self.encode_target(target)
        weights = self.row_weights(encoded)
        if sample_weight is not None:
            weights = weights * np.asarray(sample_weight, dtype=np.float32)
        order = np.random.default_rng(self.random_state).permutation(
            len(matrix)
        )
        held = (
            int(len(matrix) * self.validation_fraction)
            if len(matrix) >= MIN_ROWS_FOR_VALIDATION
            else 0
        )
        validation, training = order[:held], order[held:]
        torch.manual_seed(self.random_state)
        device = self.resolve_device(torch)
        self.input_width_ = matrix.shape[1]
        network = self.new_network(torch).to(device)
        tensors = {
            "x": torch.from_numpy(matrix).to(device),
            "y": torch.from_numpy(encoded).to(device),
            "w": torch.from_numpy(weights.astype(np.float32)).to(device),
        }
        optimizer = torch.optim.Adam(
            network.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        batch_rows = training_helpers.resolve_batch_size(
            self.batch_size, len(training)
        )
        steps_per_epoch = max(math.ceil(len(training) / batch_rows), 1)
        total_steps = steps_per_epoch * self.epochs
        warmup_steps = steps_per_epoch * self.warmup_epochs
        monitor = training_helpers.TrainingMonitor()
        generator = np.random.default_rng(self.random_state + 1)
        best, best_state, stale, step = np.inf, None, 0, 0
        self.n_iter_ = 0
        for epoch in range(self.epochs):
            network.train()
            shuffled = generator.permutation(training)
            epoch_loss, alive = 0.0, True
            for start in range(0, len(shuffled), batch_rows):
                batch = torch.from_numpy(
                    shuffled[start : start + batch_rows]
                ).to(device)
                for group in optimizer.param_groups:
                    group["lr"] = training_helpers.learning_rate_at(
                        step,
                        total_steps,
                        self.learning_rate,
                        warmup_steps,
                        self.schedule,
                    )
                optimizer.zero_grad()
                loss = self.weighted_loss(torch, network, tensors, batch)
                loss.backward()
                norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        network.parameters(), self.max_grad_norm or math.inf
                    )
                )
                alive = monitor.step(float(loss), norm, self.max_grad_norm)
                if not alive:
                    break
                optimizer.step()
                epoch_loss += float(loss)
                step += 1
            if not alive:
                break
            monitor.epoch(epoch_loss / steps_per_epoch)
            self.n_iter_ = epoch + 1
            if not held:
                continue
            network.eval()
            with torch.no_grad():
                score = float(
                    self.weighted_loss(
                        torch,
                        network,
                        tensors,
                        torch.from_numpy(validation).to(device),
                    )
                )
            if score < best:
                best, stale = score, 0
                best_state = copy.deepcopy(network.state_dict())
            else:
                stale += 1
                if stale >= self.patience:
                    break
        self.training_report_ = monitor.report()
        if best_state is not None:
            network.load_state_dict(best_state)
        network.eval()
        self.network_ = network.cpu()
        return self

    def new_network(self, torch: Any) -> Any:
        """Creates an untrained network of the configured architecture."""
        return architectures.ARCHITECTURES[self.architecture].build(
            torch,
            self.input_width_,
            self.output_units(),
            tuple(self.hidden_sizes),
            self.dropout,
        )

    def __getstate__(self) -> dict[str, Any]:
        """Pickles the weights as plain arrays, not the module objects."""
        state = dict(super().__getstate__())
        network = state.pop("network_", None)
        if network is not None:
            state["weights_"] = {
                name: tensor.numpy()
                for name, tensor in network.state_dict().items()
            }
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Rebuilds the network from the pickled weights."""
        state = dict(state)
        weights = state.pop("weights_", None)
        super().__setstate__(state)
        if weights is not None:
            torch = importlib.import_module("torch")
            network = self.new_network(torch)
            network.load_state_dict(
                {name: torch.from_numpy(w) for name, w in weights.items()}
            )
            self.network_ = network.eval()

    def weighted_loss(
        self, torch: Any, network: Any, tensors: dict[str, Any], rows: Any
    ) -> Any:
        """Weighted mean loss over the given row indices."""
        outputs = network(tensors["x"][rows])
        losses = self.row_losses(torch, outputs, tensors["y"][rows])
        weights = tensors["w"][rows]
        return (losses * weights).sum() / weights.sum()

    def raw_outputs(self, features: Any) -> np.ndarray:
        """Network outputs for new rows (evaluation mode, CPU)."""
        torch = importlib.import_module("torch")
        with torch.no_grad():
            tensor = torch.from_numpy(self.dense(features))
            return self.network_(tensor).numpy()


class ClassifierEngine(base.ClassifierMixin, TabularEngine):
    """Classifier: softmax over class scores, optional class balancing."""

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
            schedule,
            warmup_epochs,
            max_grad_norm,
            patience,
            validation_fraction,
            device,
            random_state,
        )
        self.class_weight = class_weight

    def encode_target(self, target: Any) -> np.ndarray:
        """Maps labels to 0..K-1 and remembers the classes."""
        self.classes_, codes = np.unique(
            np.asarray(target), return_inverse=True
        )
        return codes.astype(np.int64)

    def output_units(self) -> int:
        """One unit per class."""
        return len(self.classes_)

    def row_losses(self, torch: Any, outputs: Any, targets: Any) -> Any:
        """Cross entropy per row."""
        return torch.nn.functional.cross_entropy(
            outputs, targets, reduction="none"
        )

    def row_weights(self, targets: np.ndarray) -> np.ndarray:
        """Inverse class frequency when `class_weight` is `balanced`."""
        weights = np.ones(len(targets), dtype=np.float32)
        if self.class_weight == "balanced":
            counts = np.bincount(targets, minlength=len(self.classes_))
            per_class = len(targets) / (len(counts) * np.maximum(counts, 1))
            weights = per_class[targets].astype(np.float32)
        return weights

    def predict_proba(self, features: Any) -> np.ndarray:
        """Class probabilities, columns ordered like `classes_`."""
        scores = self.raw_outputs(features)
        scores = scores - scores.max(axis=1, keepdims=True)
        exp = np.exp(scores)
        return exp / exp.sum(axis=1, keepdims=True)

    def predict(self, features: Any) -> np.ndarray:
        """Most probable class."""
        return self.classes_[self.predict_proba(features).argmax(axis=1)]


class RegressorEngine(base.RegressorMixin, TabularEngine):
    """Regressor trained on the standardized target with squared error."""

    def encode_target(self, target: Any) -> np.ndarray:
        """Standardizes the target and remembers its scale."""
        values = np.asarray(target, dtype=np.float64)
        self.target_mean_ = float(values.mean())
        self.target_scale_ = float(values.std()) or 1.0
        return ((values - self.target_mean_) / self.target_scale_).astype(
            np.float32
        )

    def output_units(self) -> int:
        """A single value."""
        return 1

    def row_losses(self, torch: Any, outputs: Any, targets: Any) -> Any:
        """Squared error per row."""
        return (outputs.squeeze(-1) - targets) ** 2

    def row_weights(self, targets: np.ndarray) -> np.ndarray:
        """Unit weights."""
        return np.ones(len(targets), dtype=np.float32)

    def predict(self, features: Any) -> np.ndarray:
        """Predicted values in the original units."""
        scaled = self.raw_outputs(features).reshape(-1)
        return scaled * self.target_scale_ + self.target_mean_
