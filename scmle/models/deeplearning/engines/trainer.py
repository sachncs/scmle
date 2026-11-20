"""The training loop with early stopping, and inference on the network."""

import copy
import importlib
from typing import Any

import numpy

from scmle.contracts import errors
from scmle.models.deeplearning import architectures
from scmle.models.deeplearning.engines.descent import Descent
from scmle.models.deeplearning.engines.network import MIN_ROWS_FOR_VALIDATION
from scmle.models.deeplearning.engines.network import SINGLE_MEMBER_OUTPUT_DIMS
import scmle.models.deeplearning.training


class Trainer(Descent):
    """Trains a network with early stopping and weighted losses.

    The matrix is expected to be numeric, complete and scaled (the standard
    pipeline provides that); sparse input is densified.

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

    __module__ = "scmle.models.deeplearning.engines"

    def evaluate(
        self, torch: Any, network: Any, tensors: dict[str, Any], rows: Any
    ) -> float:
        """Scores the held-out rows in evaluation mode.

        Args:
            torch: The imported torch module.
            network: The network being trained.
            tensors: Device tensors `x`, `y` and `w` (inputs, targets, weights).
            rows: Indices of the held-out rows on the device.

        Returns:
            The weighted loss on the held-out rows.
        """
        network.eval()
        with torch.no_grad():
            return float(self.weigh(torch, network, tensors, rows))

    def divide(self, rows: int) -> tuple[numpy.ndarray, numpy.ndarray]:
        """Shuffles the row indices and holds some out for early stopping.

        Args:
            rows: Number of training rows.

        Returns:
            The held-out validation indices (none below
            `MIN_ROWS_FOR_VALIDATION` rows, otherwise `validation_fraction` of
            them) and the remaining training indices.
        """
        order = numpy.random.default_rng(self.random_state).permutation(rows)
        held = (
            int(rows * self.validation_fraction)
            if rows >= MIN_ROWS_FOR_VALIDATION
            else 0
        )
        return order[:held], order[held:]

    def fit(
        self, features: Any, target: Any, sample_weight: Any = None
    ) -> "Trainer":
        """Trains the network.

        An unknown optimizer or loss, or a non-positive accumulation count,
        raises `ConfigError` from `vet`.

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
        self.vet()
        torch = importlib.import_module("torch")
        matrix, encoded, weights = self.prepare(features, target, sample_weight)
        validation, training = self.divide(len(matrix))
        held = len(validation)
        torch.manual_seed(self.random_state)
        device = self.place(torch)
        self.input_width_ = matrix.shape[1]
        network = self.construct(torch).to(device)
        tensors = {
            "x": torch.from_numpy(matrix).to(device),
            "y": torch.from_numpy(encoded).to(device),
            "w": torch.from_numpy(weights.astype(numpy.float32)).to(device),
        }
        optimizer = self.instantiate(torch, network)
        plan = self.apportion(len(training))
        monitor = scmle.models.deeplearning.training.Monitor()
        generator = numpy.random.default_rng(self.random_state + 1)
        best, best_state, stale, step = numpy.inf, None, 0, 0
        self.n_iter_ = 0
        for epoch in range(self.epochs):
            network.train()
            shuffled = torch.from_numpy(generator.permutation(training)).to(
                device
            )
            epoch_loss, alive, step = self.train(
                torch,
                network,
                optimizer,
                tensors,
                monitor,
                plan,
                (shuffled, step),
            )
            if not alive:
                break
            monitor.losses.append(epoch_loss / plan.updates)
            self.n_iter_ = epoch + 1
            if not held:
                continue
            score = self.evaluate(
                torch, network, tensors, torch.from_numpy(validation).to(device)
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

    def emit(self, features: Any) -> numpy.ndarray:
        """Member outputs for new rows, shape (rows, members, outputs).

        Single networks have one member. Evaluation mode, on the CPU.

        Args:
            features: Numeric matrix (dense or sparse) of the width seen at fit.

        Returns:
            Array of shape `(n_rows, n_members, n_outputs)`.
        """
        torch = importlib.import_module("torch")
        with torch.no_grad():
            tensor = torch.from_numpy(self.dense(features))
            outputs = self.network_(tensor)
            if outputs.dim() == SINGLE_MEMBER_OUTPUT_DIMS:
                outputs = outputs.unsqueeze(1)
            raw: numpy.ndarray = outputs.numpy()
            return raw

    def explain(self, features: Any) -> numpy.ndarray:
        """Returns local gradient-times-input attributions.

        Classification selects the predicted class logit for each row.
        Regression selects the ensemble-mean output. The worker returns one
        signed attribution per numeric input feature, preserving the model's
        actual input space rather than inventing raw-column semantics.

        Args:
            features: Numeric matrix (dense or sparse) of the width seen at fit.

        Returns:
            Array of shape `(n_rows, input_width)`.
        """
        torch = importlib.import_module("torch")
        matrix = self.dense(features)
        tensor = torch.from_numpy(matrix).requires_grad_(requires_grad=True)
        outputs = self.network_(tensor)
        if outputs.dim() == SINGLE_MEMBER_OUTPUT_DIMS:
            outputs = outputs.unsqueeze(1)
        scores = outputs.mean(dim=1)
        if scores.shape[1] == 1:
            selected = scores[:, 0]
        else:
            selected = scores.gather(
                1, scores.argmax(dim=1, keepdim=True)
            ).squeeze(1)
        gradient = torch.autograd.grad(selected.sum(), tensor)[0]
        attribution: numpy.ndarray = (gradient * tensor).detach().numpy()
        return attribution
