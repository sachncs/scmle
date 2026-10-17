"""PyTorch model-family adapters (`deeplearning` extra)."""

from collections.abc import Mapping
import dataclasses
from typing import Any
from typing import ClassVar

from mml.contracts import errors
from mml.contracts import models
from mml.contracts import search
from mml.contracts import tasks
from mml.models.deeplearning import estimators
from mml.models.deeplearning import foundation
from mml.models.deeplearning import worker

ALL_TASKS = tasks.STANDARD_TASKS
COMMON_SPACE = (
    search.FloatParameter("dropout", 0.0, 0.4),
    search.FloatParameter("learning_rate", 1e-4, 1e-2, log=True),
    search.FloatParameter("weight_decay", 1e-6, 1e-2, log=True),
    search.ChoiceParameter("batch_size", ("auto", 64, 128, 256)),
    search.ChoiceParameter("schedule", ("cosine", "constant")),
)


class TorchAdapter(models.ModelAdapter):
    """A PyTorch tabular network family.

    Attributes:
        architecture: Name in `architectures.ARCHITECTURES`.
    """

    family = "deep_learning"
    requires: ClassVar[tuple[str, ...]] = ("torch",)
    architecture: ClassVar[str]
    capabilities = models.Capabilities(
        ALL_TASKS,
        probabilities=True,
        early_stopping=True,
        needs_scaling=True,
        gpu=True,
        max_rows=500_000,
        cost=4,
        memory_factor=4.0,
    )

    def gpu_usable(self) -> bool:
        """Whether PyTorch can see a CUDA device (asked of the worker)."""
        return self.is_available() and bool(
            worker.client().request("probe", ())
        )

    def build(
        self,
        task: tasks.TaskType,
        seed: int,
        params: Mapping[str, Any] | None = None,
    ) -> models.Estimator:
        """Creates an unfitted network estimator.

        Args:
            task: Resolved task type.
            seed: Random seed.
            params: Search-space overrides.

        Returns:
            A scikit-learn compatible estimator.

        Raises:
            ConfigError: If PyTorch is not installed.
        """
        if not self.is_available():
            raise errors.ConfigError(
                f"{self.name}: install the 'deeplearning' extra (torch)"
            )
        settings = {
            "architecture": self.architecture,
            "random_state": seed,
            **(params or {}),
        }
        if task.is_classification:
            return estimators.TorchClassifier(**settings)
        return estimators.TorchRegressor(**settings)


class TorchMlpAdapter(TorchAdapter):
    """Fully connected network."""

    name = "torch_mlp"
    architecture = "mlp"
    search_space = search.SearchSpace(
        (
            search.ChoiceParameter(
                "hidden_sizes", ((64,), (128,), (128, 64), (256, 128))
            ),
            *COMMON_SPACE,
        )
    )


class TorchResidualAdapter(TorchAdapter):
    """Residual network with layer normalization."""

    name = "torch_residual"
    architecture = "residual"
    search_space = search.SearchSpace(
        (
            search.ChoiceParameter(
                "hidden_sizes", ((64,), (64, 64), (128, 128), (128, 128, 128))
            ),
            *COMMON_SPACE,
        )
    )


class TorchEnsembleAdapter(TorchAdapter):
    """BatchEnsemble MLP (TabM-style): many cheap members, averaged."""

    name = "torch_tabm"
    architecture = "tabm"
    search_space = search.SearchSpace(
        (
            search.ChoiceParameter(
                "hidden_sizes", ((64,), (128, 64), (256, 128))
            ),
            *COMMON_SPACE,
        )
    )


class TorchTransformerAdapter(TorchAdapter):
    """FT-Transformer: attention over feature tokens."""

    name = "torch_ft_transformer"
    architecture = "ft_transformer"
    capabilities = dataclasses.replace(
        TorchAdapter.capabilities, max_rows=100_000, cost=5, memory_factor=8.0
    )
    search_space = search.SearchSpace(
        (
            search.ChoiceParameter("hidden_sizes", ((32,), (64,), (64, 64))),
            *COMMON_SPACE,
        )
    )


class TorchRobustMlpAdapter(TorchAdapter):
    """MLP with learned scaling, smooth clipping and PReLU (RealMLP-style)."""

    name = "torch_realmlp"
    architecture = "robust_mlp"
    search_space = search.SearchSpace(
        (
            search.ChoiceParameter(
                "hidden_sizes", ((128,), (256, 128), (256, 256))
            ),
            *COMMON_SPACE,
        )
    )


class KerasMlpAdapter(TorchAdapter):
    """Keras dense network, run on the PyTorch backend in the same worker."""

    name = "keras_mlp"
    requires = ("torch", "keras")
    search_space = search.SearchSpace(
        (
            search.ChoiceParameter(
                "hidden_sizes", ((64,), (128,), (128, 64), (256, 128))
            ),
            search.FloatParameter("dropout", 0.0, 0.4),
            search.FloatParameter("learning_rate", 1e-4, 1e-2, log=True),
            search.FloatParameter("l2", 1e-6, 1e-2, log=True),
            search.ChoiceParameter("batch_size", (64, 128, 256)),
        )
    )

    def build(
        self,
        task: tasks.TaskType,
        seed: int,
        params: Mapping[str, Any] | None = None,
    ) -> models.Estimator:
        """Creates an unfitted Keras estimator.

        Raises:
            ConfigError: If PyTorch or Keras is not installed.
        """
        if not self.is_available():
            raise errors.ConfigError(
                f"{self.name}: install the 'deeplearning' extra (torch, keras)"
            )
        settings = {"random_state": seed, **(params or {})}
        if task.is_classification:
            return estimators.KerasClassifier(**settings)
        return estimators.KerasRegressor(**settings)


class TabPfnAdapter(models.ModelAdapter):
    """Pretrained tabular foundation model (`tabpfn` extra, license required).

    Needs the `tabpfn` package and an accepted Prior Labs license for the
    weights (`TABPFN_TOKEN`). Without them its trials fail with the library's
    own message and the rest of the run continues.
    """

    name = "tabpfn"
    family = "foundation"
    requires = ("tabpfn",)
    capabilities = models.Capabilities(
        ALL_TASKS,
        probabilities=True,
        gpu=True,
        max_rows=foundation.MAX_ROWS,
        cost=5,
        memory_factor=6.0,
    )

    def gpu_usable(self) -> bool:
        """Whether PyTorch can see a CUDA device (asked of the worker)."""
        return self.is_available() and bool(
            worker.client().request("probe", ())
        )

    def build(
        self,
        task: tasks.TaskType,
        seed: int,
        params: Mapping[str, Any] | None = None,
    ) -> models.Estimator:
        """Creates an unfitted TabPFN estimator.

        Raises:
            ConfigError: If the `tabpfn` package is not installed.
        """
        if not self.is_available():
            raise errors.ConfigError(
                f"{self.name}: install the 'tabpfn' extra (tabpfn)"
            )
        settings = {"random_state": seed, **(params or {})}
        if task.is_classification:
            return estimators.TabPfnClassifier(**settings)
        return estimators.TabPfnRegressor(**settings)


DEEP_LEARNING_ADAPTERS: tuple[type[models.ModelAdapter], ...] = (
    TabPfnAdapter,
    TorchMlpAdapter,
    TorchResidualAdapter,
    TorchEnsembleAdapter,
    TorchTransformerAdapter,
    TorchRobustMlpAdapter,
    KerasMlpAdapter,
)
