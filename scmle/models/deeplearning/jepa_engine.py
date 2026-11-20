"""Tabular JEPA training engine, imported only inside the PyTorch worker."""

import copy
import importlib
import itertools
import operator
from typing import Any

import numpy

from scmle.contracts import errors

FEATURE_MATRIX_RANK = 2
MIN_FIT_ROWS = 8


class Network:
    """Context encoder, target encoder, and predictor for tabular JEPA."""

    def __init__(
        self,
        torch: Any,
        inputs: int,
        hidden_size: int,
        embedding_size: int,
    ) -> None:
        """Builds the two-tower representation model.

        Args:
            torch: The imported torch module.
            inputs: Number of input features.
            hidden_size: Width of the hidden layers.
            embedding_size: Width of the embedding.
        """
        module = torch.nn
        self.module = module.Module()
        self.module.context_encoder = module.Sequential(
            module.Linear(inputs, hidden_size),
            module.GELU(),
            module.Linear(hidden_size, embedding_size),
        )
        self.module.target_encoder = copy.deepcopy(self.module.context_encoder)
        self.module.predictor = module.Sequential(
            module.Linear(embedding_size, hidden_size),
            module.GELU(),
            module.Linear(hidden_size, embedding_size),
        )
        for parameter in self.module.target_encoder.parameters():
            parameter.requires_grad_(requires_grad=False)

    def parameters(self) -> Any:
        """Returns trainable context and predictor parameters."""
        return list(self.module.context_encoder.parameters()) + list(
            self.module.predictor.parameters()
        )

    def state_dict(self) -> dict[str, Any]:
        """Returns the complete network state."""
        state: dict[str, Any] = self.module.state_dict()
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore the complete network state.

        Args:
            state: Mapping of parameter names to tensors, as returned by
                `state_dict`.
        """
        self.module.load_state_dict(state)


class Settings:
    """Self-supervised training settings and their validation."""

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
        """Stores self-supervised training settings.

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
        self.hidden = hidden_size
        self.embedding = embedding_size
        self.probability = mask_probability
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.decay = ema_decay
        self.weight = variance_weight
        self.device = device
        self.random_state = random_state

    @staticmethod
    def validate(features: Any) -> numpy.ndarray:
        """Converts the features to a finite float32 matrix with enough rows.

        Args:
            features: Candidate numeric 2-D feature matrix.

        Returns:
            The float32 matrix.

        Raises:
            ValueError: If the features are not a finite numeric 2-D matrix
                with at least 8 rows and one column.
        """
        try:
            matrix = numpy.asarray(features, dtype=numpy.float32)
        except (TypeError, ValueError) as error:
            raise errors.InputError(
                "tabular JEPA requires a numeric 2-D feature matrix"
            ) from error
        if matrix.ndim != FEATURE_MATRIX_RANK:
            raise errors.InputError(
                "tabular JEPA requires a 2-D feature matrix"
            )
        if len(matrix) < MIN_FIT_ROWS:
            raise errors.InputError("tabular JEPA needs at least 8 rows")
        if matrix.shape[1] < 1:
            raise errors.InputError("tabular JEPA needs at least one feature")
        if not numpy.isfinite(matrix).all():
            raise errors.InputError(
                "tabular JEPA requires finite numeric features"
            )
        return matrix

    def vet(self) -> None:
        """Rejects hyperparameters outside their valid range.

        Raises:
            ValueError: If a layer size, the mask probability, the EMA
                decay, the learning rate, the weight decay, the epochs, the
                batch size, the variance weight or the device name is
                invalid.
        """
        hidden_size: object = self.hidden
        embedding_size: object = self.embedding
        epochs: object = self.epochs
        if (
            not isinstance(hidden_size, (int, numpy.integer))
            or isinstance(hidden_size, bool)
            or hidden_size < 1
            or not isinstance(embedding_size, (int, numpy.integer))
            or isinstance(embedding_size, bool)
            or embedding_size < 1
        ):
            raise errors.InputError(
                "JEPA layer sizes must be positive integers"
            )
        if (
            not numpy.isfinite(self.probability)
            or not 0.0 < self.probability < 1.0
        ):
            raise errors.InputError(
                "mask_probability must be finite and lie strictly between "
                "0 and 1"
            )
        if not numpy.isfinite(self.decay) or not 0.0 < self.decay < 1.0:
            raise errors.InputError(
                "ema_decay must be finite and lie strictly between 0 and 1"
            )
        if not numpy.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise errors.InputError("learning_rate must be finite and positive")
        if not numpy.isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise errors.InputError(
                "weight_decay must be finite and non-negative"
            )
        if (
            not isinstance(epochs, (int, numpy.integer))
            or isinstance(epochs, bool)
            or epochs < 1
        ):
            raise errors.InputError("epochs must be a positive integer")
        if self.batch_size != "auto" and (
            not isinstance(self.batch_size, (int, numpy.integer))
            or isinstance(self.batch_size, bool)
            or self.batch_size < 1
        ):
            raise errors.InputError(
                "batch_size must be auto or a positive integer"
            )
        if not numpy.isfinite(self.weight) or self.weight < 0.0:
            raise errors.InputError(
                "variance_weight must be finite and non-negative"
            )
        if self.device not in {"auto", "cpu"} and not str(
            self.device
        ).startswith("cuda"):
            raise errors.InputError("device must be auto, cpu, or cuda")

    def place(self, torch: Any) -> str:
        """Chooses the device the training runs on.

        Args:
            torch: The imported torch module.

        Returns:
            `cuda` for `auto` when CUDA is available, `cpu` for `auto`
            otherwise, and the requested device in any other case.
        """
        device = (
            "cuda"
            if self.device == "auto" and torch.cuda.is_available()
            else self.device
        )
        return "cpu" if device == "auto" else device

    def report(
        self, losses: list[float], device: str
    ) -> dict[str, str | int | float | list[float]]:
        """Summarises a finished training run.

        Args:
            losses: Mean loss of each epoch.
            device: Device the training ran on.

        Returns:
            The objective, the settings that shaped it and the losses.
        """
        return {
            "objective": "masked_context_prediction",
            "epochs": self.epochs,
            "losses": losses,
            "final_loss": losses[-1],
            "embedding_size": self.embedding,
            "mask_probability": self.probability,
            "ema_decay": self.decay,
            "variance_regularization": self.weight,
            "device": device,
        }


class Trainer(Settings):
    """Train a bounded tabular JEPA representation."""

    def learn(
        self,
        torch: Any,
        network: Network,
        optimizer: Any,
        batch: Any,
        mask: Any,
    ) -> float:
        """Takes one optimizer step on a masked-context prediction batch.

        Args:
            torch: The imported torch module.
            network: The two-tower network being trained.
            optimizer: Optimizer over the network's trainable parameters.
            batch: The unmasked rows.
            mask: Boolean tensor of the values hidden from the context view.

        Returns:
            The batch loss.
        """
        context = batch.masked_fill(mask, 0.0)
        context_embedding = network.module.context_encoder(context)
        prediction = network.module.predictor(context_embedding)
        with torch.no_grad():
            target_embedding = network.module.target_encoder(batch)
        prediction = torch.nn.functional.normalize(prediction, dim=1)
        target_embedding = torch.nn.functional.normalize(
            target_embedding, dim=1
        )
        alignment = (prediction - target_embedding).pow(2).mean()
        variance = torch.relu(1.0 - prediction.std(dim=0)).mean()
        loss = alignment + self.weight * variance
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), 5.0)
        optimizer.step()
        with torch.no_grad():
            for target, source in zip(
                network.module.target_encoder.parameters(),
                network.module.context_encoder.parameters(),
                strict=True,
            ):
                target.mul_(self.decay).add_(source, alpha=1.0 - self.decay)
        return float(loss.detach().cpu())

    def train(
        self,
        torch: Any,
        network: Network,
        optimizer: Any,
        tensor: Any,
        generator: Any,
        device: str,
        batch_rows: int,
    ) -> float:
        """Runs one shuffled pass over the rows.

        Args:
            torch: The imported torch module.
            network: The two-tower network being trained.
            optimizer: Optimizer over the network's trainable parameters.
            tensor: All rows, on the training device.
            generator: Random generator for the shuffle and the masks.
            device: Device the rows and the generator live on.
            batch_rows: Rows per step.

        Returns:
            The mean batch loss of the epoch.
        """
        rows = len(tensor)
        order = torch.randperm(rows, generator=generator, device=device)
        epoch_losses: list[float] = []
        for start in range(0, rows, batch_rows):
            batch = tensor[order[start : start + batch_rows]]
            mask = (
                torch.rand(batch.shape, generator=generator, device=device)
                < self.probability
            )
            epoch_losses.append(
                self.learn(torch, network, optimizer, batch, mask)
            )
        return float(numpy.mean(epoch_losses))

    def fit(
        self,
        features: Any,
        target: Any = None,
        sample_weight: Any = None,
    ) -> "Trainer":
        """Fits context prediction without consuming labels.

        Args:
            features: Finite numeric matrix with at least 8 rows and 1 column.
            target: Ignored.
            sample_weight: Ignored.

        Returns:
            This fitted engine.

        Raises:
            ValueError: If a CUDA device is requested but unavailable. The
                features must also be a finite numeric 2-D matrix with at
                least 8 rows and one column, and every hyperparameter inside
                its valid range; `validate` and
                `vet` raise `ValueError` otherwise.
        """
        del target, sample_weight
        matrix = self.validate(features)
        self.vet()
        torch = importlib.import_module("torch")
        if (
            str(self.device).startswith("cuda")
            and not torch.cuda.is_available()
        ):
            raise errors.InputError("requested CUDA device is not available")
        torch.manual_seed(self.random_state)
        device = self.place(torch)
        self.input_width_ = matrix.shape[1]
        network = Network(torch, self.input_width_, self.hidden, self.embedding)
        network.module.to(device)
        optimizer = torch.optim.AdamW(
            network.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        tensor = torch.from_numpy(matrix).to(device)
        batch_rows = (
            len(matrix)
            if self.batch_size == "auto"
            else min(int(self.batch_size), len(matrix))
        )
        generator = torch.Generator(device=device).manual_seed(
            self.random_state + 1
        )
        losses = list(
            map(
                operator.call,
                itertools.repeat(
                    lambda: self.train(
                        torch,
                        network,
                        optimizer,
                        tensor,
                        generator,
                        device,
                        batch_rows,
                    ),
                    self.epochs,
                ),
            )
        )
        network.module.eval().cpu()
        self.network_ = network
        self.n_iter_ = self.epochs
        self.training_report_ = self.report(losses, device)
        return self

    def conform(self, features: Any) -> numpy.ndarray:
        """Converts rows to encode to a float32 matrix of the fitted width.

        Args:
            features: Numeric matrix with the width seen at fit.

        Returns:
            The float32 matrix.

        Raises:
            ValueError: If the features are not a finite numeric matrix of
                shape `(rows, input_width)`.
        """
        try:
            matrix = numpy.asarray(features, dtype=numpy.float32)
        except (TypeError, ValueError) as error:
            raise errors.InputError(
                "tabular JEPA requires a numeric 2-D feature matrix"
            ) from error
        if (
            matrix.ndim != FEATURE_MATRIX_RANK
            or matrix.shape[1] != self.input_width_
            or not numpy.isfinite(matrix).all()
        ):
            raise errors.InputError(
                "features must be finite and have shape "
                f"(rows, {self.input_width_})"
            )
        return matrix

    def encode(self, features: Any) -> numpy.ndarray:
        """Encodes rows with the learned context encoder.

        Args:
            features: Numeric matrix with the width seen at fit.

        Returns:
            Embedding matrix of shape `(n_rows, embedding_size)`.

        Raises:
            ValueError: If the features are not a finite numeric matrix of
                shape `(rows, input_width)`.
        """
        torch = importlib.import_module("torch")
        matrix = self.conform(features)
        with torch.no_grad():
            embedding: numpy.ndarray = self.network_.module.context_encoder(
                torch.from_numpy(matrix)
            ).numpy()
            return embedding

    def explain(self, features: Any) -> numpy.ndarray:
        """Returns embedding-energy gradient-times-input attributions.

        This explains local sensitivity of the learned representation. It is
        intentionally not presented as a label explanation or causal effect.

        Args:
            features: Numeric matrix with the width seen at fit.

        Returns:
            Array of shape `(n_rows, input_width)`.

        Raises:
            ValueError: If the features are not a finite numeric matrix of
                shape `(rows, input_width)`.
        """
        torch = importlib.import_module("torch")
        matrix = self.conform(features)
        tensor = torch.from_numpy(matrix).requires_grad_(requires_grad=True)
        embedding = self.network_.module.context_encoder(tensor)
        energy = embedding.pow(2).mean(dim=1)
        gradient = torch.autograd.grad(energy.sum(), tensor)[0]
        attribution: numpy.ndarray = (gradient * tensor).detach().numpy()
        return attribution

    def __getstate__(self) -> dict[str, Any]:
        """Serialize weights as NumPy arrays for the parent process.

        Returns:
            The instance state with `network_` replaced by `weights_`, a mapping
            of parameter names to numpy arrays.
        """
        state = dict(self.__dict__)
        network = state.pop("network_", None)
        if network is not None:
            state["weights_"] = {
                name: tensor.numpy()
                for name, tensor in network.state_dict().items()
            }
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Rebuilds the PyTorch network from serialized arrays.

        Args:
            state: Mapping produced by `__getstate__`.
        """
        state = dict(state)
        weights = state.pop("weights_", None)
        self.__dict__.update(state)
        if weights is not None:
            torch = importlib.import_module("torch")
            network = Network(
                torch,
                self.input_width_,
                self.hidden,
                self.embedding,
            )
            network.load_state_dict(
                {
                    name: torch.from_numpy(value)
                    for name, value in weights.items()
                }
            )
            network.module.eval()
            self.network_ = network
