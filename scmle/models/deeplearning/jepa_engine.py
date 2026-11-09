"""Tabular JEPA training engine, imported only inside the PyTorch worker."""

import copy
import importlib
from typing import Any

import numpy as np


class JepaNetwork:
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
            parameter.requires_grad_(False)

    def parameters(self) -> Any:
        """Returns trainable context and predictor parameters."""
        return list(self.module.context_encoder.parameters()) + list(
            self.module.predictor.parameters()
        )

    def state_dict(self) -> dict[str, Any]:
        """Returns the complete network state."""
        return self.module.state_dict()

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore the complete network state.

        Args:
            state: Mapping of parameter names to tensors, as returned by
                `state_dict`.
        """
        self.module.load_state_dict(state)


class JepaEngine:
    """Train a bounded tabular JEPA representation."""

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
    ) -> "JepaEngine":
        """Fits context prediction without consuming labels.

        Args:
            features: Finite numeric matrix with at least 8 rows and 1 column.
            target: Ignored.
            sample_weight: Ignored.

        Returns:
            This fitted engine.

        Raises:
            ValueError: If the features are not a finite numeric 2-D matrix with
                at least 8 rows and one column, a hyperparameter is outside its
                valid range, or a CUDA device is requested but unavailable.
        """
        del target, sample_weight
        try:
            matrix = np.asarray(features, dtype=np.float32)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "tabular JEPA requires a numeric 2-D feature matrix"
            ) from error
        if matrix.ndim != 2:
            raise ValueError("tabular JEPA requires a 2-D feature matrix")
        if len(matrix) < 8:
            raise ValueError("tabular JEPA needs at least 8 rows")
        if matrix.shape[1] < 1:
            raise ValueError("tabular JEPA needs at least one feature")
        if not np.isfinite(matrix).all():
            raise ValueError("tabular JEPA requires finite numeric features")
        if (
            not isinstance(self.hidden_size, (int, np.integer))
            or isinstance(self.hidden_size, bool)
            or self.hidden_size < 1
            or not isinstance(self.embedding_size, (int, np.integer))
            or isinstance(self.embedding_size, bool)
            or self.embedding_size < 1
        ):
            raise ValueError("JEPA layer sizes must be positive integers")
        if (
            not np.isfinite(self.mask_probability)
            or not 0.0 < self.mask_probability < 1.0
        ):
            raise ValueError(
                "mask_probability must be finite and lie strictly between "
                "0 and 1"
            )
        if not np.isfinite(self.ema_decay) or not 0.0 < self.ema_decay < 1.0:
            raise ValueError(
                "ema_decay must be finite and lie strictly between 0 and 1"
            )
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        if not np.isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise ValueError("weight_decay must be finite and non-negative")
        if (
            not isinstance(self.epochs, (int, np.integer))
            or isinstance(self.epochs, bool)
            or self.epochs < 1
        ):
            raise ValueError("epochs must be a positive integer")
        if self.batch_size != "auto" and (
            not isinstance(self.batch_size, (int, np.integer))
            or isinstance(self.batch_size, bool)
            or self.batch_size < 1
        ):
            raise ValueError("batch_size must be auto or a positive integer")
        if not np.isfinite(self.variance_weight) or self.variance_weight < 0.0:
            raise ValueError("variance_weight must be finite and non-negative")
        if (
            self.device != "auto"
            and self.device != "cpu"
            and not str(self.device).startswith("cuda")
        ):
            raise ValueError("device must be auto, cpu, or cuda")
        torch = importlib.import_module("torch")
        if (
            str(self.device).startswith("cuda")
            and not torch.cuda.is_available()
        ):
            raise ValueError("requested CUDA device is not available")
        torch.manual_seed(self.random_state)
        device = (
            "cuda"
            if self.device == "auto" and torch.cuda.is_available()
            else self.device
        )
        if device == "auto":
            device = "cpu"
        self.input_width_ = matrix.shape[1]
        network = JepaNetwork(
            torch, self.input_width_, self.hidden_size, self.embedding_size
        )
        network.module.to(device)
        optimizer = torch.optim.AdamW(
            network.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        tensor = torch.from_numpy(matrix).to(device)
        rows = len(matrix)
        batch_rows = (
            rows
            if self.batch_size == "auto"
            else min(int(self.batch_size), rows)
        )
        generator = torch.Generator(device=device).manual_seed(
            self.random_state + 1
        )
        losses: list[float] = []
        epoch_number = 0
        while epoch_number < self.epochs:
            order = torch.randperm(rows, generator=generator, device=device)
            epoch_losses: list[float] = []
            for start in range(0, rows, batch_rows):
                batch = tensor[order[start : start + batch_rows]]
                mask = (
                    torch.rand(batch.shape, generator=generator, device=device)
                    < self.mask_probability
                )
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
                loss = alignment + self.variance_weight * variance
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
                        target.mul_(self.ema_decay).add_(
                            source, alpha=1.0 - self.ema_decay
                        )
                epoch_losses.append(float(loss.detach().cpu()))
            losses.append(float(np.mean(epoch_losses)))
            epoch_number += 1
        network.module.eval().cpu()
        self.network_ = network
        self.n_iter_ = self.epochs
        self.training_report_ = {
            "objective": "masked_context_prediction",
            "epochs": self.epochs,
            "losses": losses,
            "final_loss": losses[-1],
            "embedding_size": self.embedding_size,
            "mask_probability": self.mask_probability,
            "ema_decay": self.ema_decay,
            "variance_regularization": self.variance_weight,
            "device": device,
        }
        return self

    def encode(self, features: Any) -> np.ndarray:
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
        try:
            matrix = np.asarray(features, dtype=np.float32)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "tabular JEPA requires a numeric 2-D feature matrix"
            ) from error
        if (
            matrix.ndim != 2
            or matrix.shape[1] != self.input_width_
            or not np.isfinite(matrix).all()
        ):
            raise ValueError(
                "features must be finite and have shape "
                f"(rows, {self.input_width_})"
            )
        with torch.no_grad():
            return self.network_.module.context_encoder(
                torch.from_numpy(matrix)
            ).numpy()

    def explain(self, features: Any) -> np.ndarray:
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
        try:
            matrix = np.asarray(features, dtype=np.float32)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "tabular JEPA requires a numeric 2-D feature matrix"
            ) from error
        if (
            matrix.ndim != 2
            or matrix.shape[1] != self.input_width_
            or not np.isfinite(matrix).all()
        ):
            raise ValueError(
                "features must be finite and have shape "
                f"(rows, {self.input_width_})"
            )
        tensor = torch.from_numpy(matrix).requires_grad_(True)
        embedding = self.network_.module.context_encoder(tensor)
        energy = embedding.pow(2).mean(dim=1)
        gradient = torch.autograd.grad(energy.sum(), tensor)[0]
        return (gradient * tensor).detach().numpy()

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
            network = JepaNetwork(
                torch,
                self.input_width_,
                self.hidden_size,
                self.embedding_size,
            )
            network.load_state_dict(
                {
                    name: torch.from_numpy(value)
                    for name, value in weights.items()
                }
            )
            network.module.eval()
            self.network_ = network
