"""PyTorch model-family adapters (`deeplearning` extra)."""

from collections.abc import Mapping
from typing import Any
from typing import ClassVar

from mml.contracts import errors
from mml.contracts import models
from mml.contracts import search
from mml.contracts import tasks
from mml.models.deeplearning import estimators

ALL_TASKS = frozenset(tasks.TaskType)
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

    family = "neural"
    requires = ("torch",)
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


DEEP_LEARNING_ADAPTERS: tuple[type[models.ModelAdapter], ...] = (
    TorchMlpAdapter,
    TorchResidualAdapter,
)
