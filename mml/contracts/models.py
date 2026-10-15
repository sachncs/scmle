"""Model adapter contract.

Adapters are polymorphic: the planner, trial runner and conformance suite
treat every `ModelAdapter` identically and read declared `Capabilities`
instead of hard-coding library-specific conditions.
"""

import abc
import dataclasses
import importlib.util
from typing import Any, ClassVar, Protocol

from mml.contracts import tasks


class Estimator(Protocol):
    """Scikit-learn compatible estimator protocol."""

    def fit(self, features: Any, target: Any) -> Any:
        """Fits the estimator."""

    def predict(self, features: Any) -> Any:
        """Predicts targets."""

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Returns hyper-parameters."""

    def set_params(self, **params: Any) -> Any:
        """Sets hyper-parameters."""


@dataclasses.dataclass(frozen=True, slots=True)
class Capabilities:
    """What an adapter can do; consumed by the planner.

    Attributes:
        tasks: Supported task types.
        sparse_input: Accepts sparse matrices.
        probabilities: Provides `predict_proba` for classification.
        native_categorical: Handles categorical features natively.
        handles_missing: Handles missing values natively.
        early_stopping: Supports early stopping.
        needs_scaling: Benefits from standardized inputs.
        serializable: Can be persisted in a bundle.
        max_rows: Largest dataset it is practical on, if any.
        cost: Relative training cost from 1 (cheap) to 5 (expensive).
    """

    tasks: frozenset[tasks.TaskType]
    sparse_input: bool = False
    probabilities: bool = True
    native_categorical: bool = False
    handles_missing: bool = False
    early_stopping: bool = False
    needs_scaling: bool = False
    serializable: bool = True
    max_rows: int | None = None
    cost: int = 1


class ModelAdapter(abc.ABC):
    """A model family the framework can plan, train and package.

    Subclasses declare class-level metadata and implement `build`.

    Attributes:
        name: Unique registry name.
        family: Coarse family used for reporting (linear, tree, ...).
        capabilities: Declared capabilities.
        requires: Importable modules needed at build time.
    """

    name: ClassVar[str]
    family: ClassVar[str]
    capabilities: ClassVar[Capabilities]
    requires: ClassVar[tuple[str, ...]] = ()

    @abc.abstractmethod
    def build(self, task: tasks.TaskType, seed: int) -> Estimator:
        """Creates an unfitted estimator.

        Args:
            task: Resolved task type.
            seed: Random seed for deterministic behaviour.

        Returns:
            A fresh estimator.
        """

    def is_available(self) -> bool:
        """Whether every required module is importable (checked lazily)."""
        return all(
            importlib.util.find_spec(module) is not None
            for module in self.requires
        )

    def rejection_reason(
        self, task: tasks.TaskType, row_count: int | None = None
    ) -> str | None:
        """Explains why the adapter cannot be used, if it cannot.

        Args:
            task: Resolved task type.
            row_count: Dataset size, if known.

        Returns:
            A reason string, or None when the adapter is compatible.
        """
        caps = self.capabilities
        if task not in caps.tasks:
            return f"does not support {task.value}"
        if not self.is_available():
            return f"missing dependency: {', '.join(self.requires)}"
        if (
            row_count is not None
            and caps.max_rows is not None
            and row_count > caps.max_rows
        ):
            return f"dataset too large ({row_count} > {caps.max_rows} rows)"
        return None
