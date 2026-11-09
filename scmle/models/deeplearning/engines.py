"""PyTorch training engines. They run only inside the worker process.

Importing this module needs PyTorch, so nothing outside `worker` loads it;
the estimators users see are the proxies in `estimators`.
"""

import abc
import copy
import importlib
import math
from typing import Any

import numpy
from scipy import sparse
from sklearn import base

from scmle.contracts import errors
from scmle.models.deeplearning import architectures
from scmle.models.deeplearning import training as training_helpers

MIN_ROWS_FOR_VALIDATION = 30


class TabularEngine(base.BaseEstimator, abc.ABC):
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
    def encode_target(self, target: Any) -> numpy.ndarray:
        """Converts the target to the array the loss consumes.

        Args:
            target: Training labels or values.

        Returns:
            The encoded target, one entry per row.
        """

    @abc.abstractmethod
    def output_units(self) -> int:
        """Number of network outputs.

        Returns:
            The number of output units the network needs.
        """

    @abc.abstractmethod
    def row_losses(self, torch: Any, outputs: Any, targets: Any) -> Any:
        """Per-row loss (no reduction).

        Args:
            torch: The imported torch module.
            outputs: Network outputs, one row per (example, member) pair.
            targets: Encoded targets repeated once per member.

        Returns:
            Tensor with one loss per (example, member) pair.
        """

    @abc.abstractmethod
    def row_weights(self, targets: numpy.ndarray) -> numpy.ndarray:
        """Per-row weight implied by the estimator's own settings.

        Args:
            targets: Encoded targets.

        Returns:
            Float32 array with one weight per row.
        """

    def resolve_device(self, torch: Any) -> str:
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

    def fit(
        self, features: Any, target: Any, sample_weight: Any = None
    ) -> "TabularEngine":
        """Trains the network.

        An unknown optimizer or loss, or a non-positive accumulation count,
        raises `ConfigError` from `check_options`.

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
        self.check_options()
        torch = importlib.import_module("torch")
        matrix = self.dense(features)
        encoded = self.encode_target(target)
        weights = self.row_weights(encoded)
        if sample_weight is not None:
            weights = weights * numpy.asarray(
                sample_weight, dtype=numpy.float32
            )
        order = numpy.random.default_rng(self.random_state).permutation(
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
            "w": torch.from_numpy(weights.astype(numpy.float32)).to(device),
        }
        optimizer = self.new_optimizer(torch, network)
        second_order = self.optimizer == "lbfgs"
        batch_rows = (
            len(training)
            if second_order
            else training_helpers.resolve_batch_size(
                self.batch_size, len(training)
            )
        )
        accumulate = 1 if second_order else self.accumulation_steps
        batches_per_epoch = max(math.ceil(len(training) / batch_rows), 1)
        updates_per_epoch = max(math.ceil(batches_per_epoch / accumulate), 1)
        total_steps = updates_per_epoch * self.epochs
        warmup_steps = updates_per_epoch * self.warmup_epochs
        monitor = training_helpers.TrainingMonitor()
        generator = numpy.random.default_rng(self.random_state + 1)
        best, best_state, stale, step = numpy.inf, None, 0, 0
        self.n_iter_ = 0
        for epoch in range(self.epochs):
            network.train()
            shuffled = generator.permutation(training)
            epoch_loss, alive = 0.0, True
            pending, running = 0, 0.0
            for start in range(0, len(shuffled), batch_rows):
                batch = torch.from_numpy(
                    shuffled[start : start + batch_rows]
                ).to(device)
                if second_order:
                    value, norm = self.lbfgs_step(
                        torch, network, optimizer, tensors, batch
                    )
                    alive = monitor.step(value, norm, None)
                    if not alive:
                        break
                    epoch_loss += value
                    step += 1
                    continue
                if pending == 0:
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
                (loss / accumulate).backward()
                running += float(loss.detach())
                pending += 1
                last = start + batch_rows >= len(shuffled)
                if pending < accumulate and not last:
                    continue
                norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        network.parameters(), self.max_grad_norm or math.inf
                    )
                )
                value = running / pending
                alive = monitor.step(value, norm, self.max_grad_norm)
                if not alive:
                    break
                optimizer.step()
                epoch_loss += value
                step += 1
                pending, running = 0, 0.0
            if not alive:
                break
            monitor.epoch(epoch_loss / updates_per_epoch)
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

    OPTIMIZERS = ("adam", "adamw", "sgd", "rmsprop", "lbfgs")
    CLASSIFIER_LOSSES = ("default", "focal")
    REGRESSOR_LOSSES = ("default", "huber")
    losses: tuple[str, ...] = ("default",)

    def check_options(self) -> None:
        """Rejects unknown optimizers and losses before any work is done.

        Raises:
            ConfigError: For an unknown optimizer, a loss the task does not
                have, or a non-positive accumulation count.
        """
        if self.optimizer not in self.OPTIMIZERS:
            raise errors.ConfigError(
                f"unknown optimizer '{self.optimizer}'; use one of "
                f"{list(self.OPTIMIZERS)}"
            )
        if self.loss not in self.losses:
            raise errors.ConfigError(
                f"unknown loss '{self.loss}' for this task; use one of "
                f"{list(self.losses)}"
            )
        if self.accumulation_steps < 1:
            raise errors.ConfigError("accumulation_steps must be at least 1")

    def new_optimizer(self, torch: Any, network: Any) -> Any:
        """Creates the configured optimizer for the network's parameters.

        Args:
            torch: The imported torch module.
            network: The network whose parameters are optimised.

        Returns:
            The torch optimizer; Adam for any unrecognised name.
        """
        parameters = network.parameters()
        rate, decay = self.learning_rate, self.weight_decay
        if self.optimizer == "adamw":
            return torch.optim.AdamW(parameters, lr=rate, weight_decay=decay)
        if self.optimizer == "sgd":
            return torch.optim.SGD(
                parameters,
                lr=rate,
                momentum=self.momentum,
                nesterov=self.momentum > 0,
                weight_decay=decay,
            )
        if self.optimizer == "rmsprop":
            return torch.optim.RMSprop(parameters, lr=rate, weight_decay=decay)
        if self.optimizer == "lbfgs":
            return torch.optim.LBFGS(
                parameters, lr=1.0, max_iter=20, line_search_fn="strong_wolfe"
            )
        return torch.optim.Adam(parameters, lr=rate, weight_decay=decay)

    def lbfgs_step(
        self,
        torch: Any,
        network: Any,
        optimizer: Any,
        tensors: dict[str, Any],
        batch: Any,
    ) -> tuple[float, float]:
        """Runs one L-BFGS step; returns the starting loss and gradient norm.

        Args:
            torch: The imported torch module.
            network: The network being trained.
            optimizer: The L-BFGS optimizer.
            tensors: Device tensors `x`, `y` and `w` (inputs, targets, weights).
            batch: Indices of the rows to use (all training rows).

        Returns:
            The loss at the start of the step (the first closure evaluation)
            and the norm of the gradients left after it, without clipping.
        """

        def closure() -> Any:
            optimizer.zero_grad()
            loss = self.weighted_loss(torch, network, tensors, batch)
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        norm = float(
            torch.nn.utils.clip_grad_norm_(network.parameters(), math.inf)
        )
        return float(loss.detach()), norm

    def new_network(self, torch: Any) -> Any:
        """Creates an untrained network of the configured architecture.

        Args:
            torch: The imported torch module.

        Returns:
            The untrained module, sized by `input_width_` and `output_units()`.
        """
        return architectures.ARCHITECTURES[self.architecture].build(
            torch,
            self.input_width_,
            self.output_units(),
            tuple(self.hidden_sizes),
            self.dropout,
        )

    def __getstate__(self) -> dict[str, Any]:
        """Pickles the weights as plain arrays, not the module objects.

        Returns:
            The instance state with `network_` replaced by `weights_`, a mapping
            of parameter names to numpy arrays.
        """
        state = dict(super().__getstate__())
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
            network = self.new_network(torch)
            network.load_state_dict(
                {name: torch.from_numpy(w) for name, w in weights.items()}
            )
            self.network_ = network.eval()

    def weighted_loss(
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
        if outputs.dim() == 2:
            outputs = outputs.unsqueeze(1)
        batch, members = outputs.shape[0], outputs.shape[1]
        losses = self.row_losses(
            torch,
            outputs.reshape(batch * members, -1),
            tensors["y"][rows].repeat_interleave(members),
        )
        losses = losses.reshape(batch, members).mean(dim=1)
        weights = tensors["w"][rows]
        return (losses * weights).sum() / weights.sum()

    def raw_outputs(self, features: Any) -> numpy.ndarray:
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
            if outputs.dim() == 2:
                outputs = outputs.unsqueeze(1)
            return outputs.numpy()

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
        tensor = torch.from_numpy(matrix).requires_grad_(True)
        outputs = self.network_(tensor)
        if outputs.dim() == 2:
            outputs = outputs.unsqueeze(1)
        scores = outputs.mean(dim=1)
        if scores.shape[1] == 1:
            selected = scores[:, 0]
        else:
            selected = scores.gather(
                1, scores.argmax(dim=1, keepdim=True)
            ).squeeze(1)
        gradient = torch.autograd.grad(selected.sum(), tensor)[0]
        return (gradient * tensor).detach().numpy()


class ClassifierEngine(base.ClassifierMixin, TabularEngine):
    """Classifier: softmax over class scores, optional class balancing."""

    losses = ("default", "focal")

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

    def encode_target(self, target: Any) -> numpy.ndarray:
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

    def output_units(self) -> int:
        """One unit per class.

        Returns:
            The number of classes seen by `encode_target`.
        """
        return len(self.classes_)

    def row_losses(self, torch: Any, outputs: Any, targets: Any) -> Any:
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

    def row_weights(self, targets: numpy.ndarray) -> numpy.ndarray:
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
        return probabilities.mean(axis=1)

    def predict(self, features: Any) -> numpy.ndarray:
        """Most probable class.

        Args:
            features: Numeric matrix (dense or sparse) of the width seen at fit.

        Returns:
            Array of shape `(n_rows,)` of class labels.
        """
        return self.classes_[self.predict_proba(features).argmax(axis=1)]


class RegressorEngine(base.RegressorMixin, TabularEngine):
    """Regressor trained on the standardized target (squared or Huber)."""

    losses = ("default", "huber")

    def encode_target(self, target: Any) -> numpy.ndarray:
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

    def output_units(self) -> int:
        """A single value.

        Returns:
            1.
        """
        return 1

    def row_losses(self, torch: Any, outputs: Any, targets: Any) -> Any:
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

    def row_weights(self, targets: numpy.ndarray) -> numpy.ndarray:
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
        return scaled * self.target_scale_ + self.target_mean_
