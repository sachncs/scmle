"""The network every engine trains: hyperparameters, losses, persistence."""

import abc
import dataclasses
import importlib
from typing import Any
from typing import cast

import numpy
from scipy import sparse
from sklearn import base

from scmle.models.deeplearning import architectures

MIN_ROWS_FOR_VALIDATION = 30
SINGLE_MEMBER_OUTPUT_DIMS = 2


@dataclasses.dataclass(frozen=True, slots=True)
class Plan:
    """How one training run cuts its epochs into optimizer updates.

    Attributes:
        rows: Rows per mini-batch.
        order: Whether the optimizer takes full-batch L-BFGS steps.
        accumulate: Mini-batches whose gradients are summed per update.
        updates: Optimizer updates in one epoch.
        steps: Optimizer updates over all epochs.
        warmup: Optimizer updates of the learning-rate warm-up.
    """

    rows: int
    order: bool
    accumulate: int
    updates: int
    steps: int
    warmup: int


class Network(base.BaseEstimator, abc.ABC):  # type: ignore[misc]  # scikit-learn ships no type information
    """Hyperparameters, the per-task loss interface and the network weights.

    `Trainer` documents the hyperparameters; this layer stores them, builds
    the network, computes the weighted loss and pickles the weights.
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

    @abc.abstractmethod
    def vectorize(self, target: Any) -> numpy.ndarray:
        """Converts the target to the array the loss consumes.

        Args:
            target: Training labels or values.

        Returns:
            The encoded target, one entry per row.
        """

    @abc.abstractmethod
    def dimensionality(self) -> int:
        """Number of network outputs.

        Returns:
            The number of output units the network needs.
        """

    @abc.abstractmethod
    def penalize(self, torch: Any, outputs: Any, targets: Any) -> Any:
        """Per-row loss (no reduction).

        Args:
            torch: The imported torch module.
            outputs: Network outputs, one row per (example, member) pair.
            targets: Encoded targets repeated once per member.

        Returns:
            Tensor with one loss per (example, member) pair.
        """

    @abc.abstractmethod
    def weighting(self, targets: numpy.ndarray) -> numpy.ndarray:
        """Per-row weight implied by the estimator's own settings.

        Args:
            targets: Encoded targets.

        Returns:
            Float32 array with one weight per row.
        """

    def place(self, torch: Any) -> str:
        """Chooses where to train.

        Args:
            torch: The imported torch module.

        Returns:
            `cuda` or `cpu` when `device` is `auto` (by CUDA availability),
            otherwise the configured device.
        """
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device

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

    def construct(self, torch: Any) -> Any:
        """Creates an untrained network of the configured architecture.

        Args:
            torch: The imported torch module.

        Returns:
            The untrained module, sized by `input_width_` and
            `dimensionality()`.
        """
        return architectures.ARCHITECTURES[self.architecture].build(
            torch,
            self.input_width_,
            self.dimensionality(),
            tuple(self.hidden_sizes),
            self.dropout,
        )

    def __getstate__(self) -> dict[str, Any]:
        """Pickles the weights as plain arrays, not the module objects.

        Returns:
            The instance state with `network_` replaced by `weights_`, a mapping
            of parameter names to numpy arrays.
        """
        state = dict(cast("dict[str, Any]", super().__getstate__()))
        network = state.pop("network_", None)
        if network is not None:
            state["weights_"] = {
                name: tensor.numpy()
                for name, tensor in network.state_dict().items()
            }
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Rebuilds the network from the pickled weights.

        Args:
            state: Mapping produced by `__getstate__`.
        """
        state = dict(state)
        weights = state.pop("weights_", None)
        super().__setstate__(state)
        if weights is not None:
            torch = importlib.import_module("torch")
            network = self.construct(torch)
            network.load_state_dict(
                {name: torch.from_numpy(w) for name, w in weights.items()}
            )
            self.network_ = network.eval()

    def weigh(
        self, torch: Any, network: Any, tensors: dict[str, Any], rows: Any
    ) -> Any:
        """Weighted mean loss over the given row indices.

        Args:
            torch: The imported torch module.
            network: The network to evaluate.
            tensors: Device tensors `x`, `y` and `w` (inputs, targets, weights).
            rows: Indices of the rows to use.

        Returns:
            Scalar tensor: the member-averaged row losses weighted by `w`.
        """
        outputs = network(tensors["x"][rows])
        if outputs.dim() == SINGLE_MEMBER_OUTPUT_DIMS:
            outputs = outputs.unsqueeze(1)
        batch, members = outputs.shape[0], outputs.shape[1]
        losses = self.penalize(
            torch,
            outputs.reshape(batch * members, -1),
            tensors["y"][rows].repeat_interleave(members),
        )
        losses = losses.reshape(batch, members).mean(dim=1)
        weights = tensors["w"][rows]
        return (losses * weights).sum() / weights.sum()
