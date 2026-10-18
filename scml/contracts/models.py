"""Model adapter contract.

Adapters are polymorphic: the planner, trial runner and conformance suite
treat every `ModelAdapter` identically and read declared `Capabilities`
instead of hard-coding library-specific conditions.
"""

import abc
from collections.abc import Mapping
import dataclasses
import importlib.util
from typing import Any
from typing import ClassVar
from typing import Protocol

from scml.contracts import data as data_contracts
from scml.contracts import search
from scml.contracts import tasks


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
        native_categorical: The pipeline can hand it high-cardinality
            columns as integer codes marked categorical (the adapter
            implements `categorical_params`).
        handles_missing: Handles missing values natively.
        early_stopping: Supports early stopping.
        needs_scaling: Benefits from standardized inputs.
        serializable: Can be persisted in a bundle.
        incremental: Can keep learning from batches (`partial_fit`), so it
            can train on data larger than memory.
        stream_only: Built for streamed training; planned only when
            `data.training` is `stream`.
        gpu: Can train on a GPU.
        standalone: Can be a screening candidate on its own (False for
            meta-models assembled from other candidates).
        max_rows: Largest dataset it is practical on, if any.
        cost: Relative training cost from 1 (cheap) to 5 (expensive).
        memory_factor: Peak training memory as a multiple of the size of the
            numeric training matrix (rows x features x 8 bytes); the planner
            rejects models whose estimate exceeds the memory limit.
    """

    tasks: frozenset[tasks.TaskType]
    sparse_input: bool = False
    probabilities: bool = True
    native_categorical: bool = False
    handles_missing: bool = False
    early_stopping: bool = False
    needs_scaling: bool = False
    serializable: bool = True
    incremental: bool = False
    stream_only: bool = False
    gpu: bool = False
    standalone: bool = True
    max_rows: int | None = None
    cost: int = 1
    memory_factor: float = 1.0


class ModelAdapter(abc.ABC):
    """A model family the framework can plan, train and package.

    Subclasses declare class-level metadata and implement `build`.

    Attributes:
        name: Unique registry name.
        family: Coarse family used for reporting (linear, tree, ...).
        capabilities: Declared capabilities.
        requires: Importable modules needed at build time.
        search_space: Tunable hyperparameters, or None if not tunable.
        conformance_params: Build parameters the conformance suite should use
            for adapters that cannot build without them (meta-models).
        monotonic_param: Name of the estimator parameter that takes one
            monotonic sign per input column, or None when the model cannot
            enforce monotonic constraints.
    """

    name: ClassVar[str]
    family: ClassVar[str]
    capabilities: ClassVar[Capabilities]
    requires: ClassVar[tuple[str, ...]] = ()
    search_space: ClassVar[search.SearchSpace | None] = None
    conformance_params: ClassVar[Mapping[str, Any] | None] = None
    monotonic_param: ClassVar[str | None] = None

    @abc.abstractmethod
    def build(
        self,
        task: tasks.TaskType,
        seed: int,
        params: Mapping[str, Any] | None = None,
    ) -> Estimator:
        """Creates an unfitted estimator.

        Args:
            task: Resolved task type.
            seed: Random seed for deterministic behaviour.
            params: Hyperparameter overrides from the search space.

        Returns:
            A fresh estimator.
        """

    def assemble(
        self,
        plan: data_contracts.FeaturePlan,
        task: tasks.TaskType,
        seed: int,
        params: Mapping[str, Any] | None = None,
    ) -> Estimator | None:
        """Builds a complete pipeline itself, if the model needs to.

        Meta-models such as stacking build their members' pipelines and run
        structure-aware inner cross-validation on the raw frame; they return
        the finished candidate here. Ordinary adapters return None and get
        the standard preprocessing-plus-estimator pipeline.
        """
        return None

    def categorical_params(self, indices: list[int]) -> Mapping[str, Any]:
        """Estimator parameters that mark columns as native categoricals.

        Adapters declaring `native_categorical` override this; the pipeline
        then feeds high-cardinality columns as integer codes and passes
        their positions here instead of one-hot encoding them.

        Args:
            indices: Positions of the categorical columns in the matrix.

        Returns:
            Parameter overrides, empty if the adapter cannot use them.
        """
        return {}

    def gpu_usable(self) -> bool:
        """Whether this adapter can use a GPU on this machine right now.

        The default trusts the declared capability; adapters whose GPU
        support depends on the installed hardware override it.
        """
        return self.capabilities.gpu

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


class AdapterResolver(Protocol):
    """Anything that resolves a model name to its adapter (a registry)."""

    def get(self, name: str) -> ModelAdapter:
        """Returns the adapter registered under `name`."""
