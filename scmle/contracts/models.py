"""Model adapter contract.

Adapters are polymorphic: the planner, trial runner and conformance suite
treat every `Model` identically and read declared `Capabilities`
instead of hard-coding library-specific conditions.
"""

import abc
from collections.abc import Mapping
import dataclasses
import importlib.util
from typing import Any
from typing import Protocol

from scmle.contracts import data
from scmle.contracts import search
from scmle.contracts import tasks


class Estimator(Protocol):
    """Scikit-learn compatible estimator protocol."""

    def fit(self, features: Any, target: Any, /) -> Any:
        """Fits the estimator.

        Args:
            features: The training features.
            target: The training target.

        Returns:
            The fitted estimator.
        """

    def predict(self, features: Any, /) -> Any:
        """Predicts targets.

        Args:
            features: The features to predict for.

        Returns:
            The predictions, one per row.
        """

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Returns hyper-parameters.

        Args:
            deep: Whether to include the parameters of nested estimators.

        Returns:
            The hyper-parameters by name.
        """
        ...

    def set_params(self, **params: Any) -> Any:
        """Sets hyper-parameters.

        Args:
            **params: The hyper-parameters to set, by name.

        Returns:
            The estimator.
        """


@dataclasses.dataclass(frozen=True, slots=True)
class Capabilities:
    """What an adapter can do; consumed by the planner.

    Attributes:
        tasks: Supported task types.
        sparse: Accepts sparse matrices.
        probabilities: Provides `predict_proba` for classification.
        categorical: The pipeline can hand it high-cardinality
            columns as integer codes marked categorical (the adapter
            implements `nominals`).
        missing: Handles missing values natively.
        early_stopping: Supports early stopping.
        scaling: Benefits from standardized inputs.
        serializable: Can be persisted in a bundle.
        incremental: Can keep learning from batches (`partial_fit`), so it
            can train on data larger than memory.
        streaming: Built for streamed training; planned only when
            `data.training` is `stream`.
        gpu: Can train on a GPU.
        standalone: Can be a screening candidate on its own (False for
            meta-models assembled from other candidates).
        scale: Largest dataset it is practical on, if any.
        cost: Relative training cost from 1 (cheap) to 5 (expensive).
        factor: Peak training memory as a multiple of the size of the
            numeric training matrix (rows x features x 8 bytes); the planner
            rejects models whose estimate exceeds the memory limit.
    """

    tasks: frozenset[tasks.Task]
    sparse: bool = False
    probabilities: bool = True
    categorical: bool = False
    missing: bool = False
    early_stopping: bool = False
    scaling: bool = False
    serializable: bool = True
    incremental: bool = False
    streaming: bool = False
    gpu: bool = False
    standalone: bool = True
    scale: int | None = None
    cost: int = 1
    factor: float = 1.0


class Model(abc.ABC):
    """A model family the framework can plan, train and package.

    The metadata below can be declared two ways: as class attributes of a
    subclass that implements `build` (the original style, still supported
    for user adapters), or as instance attributes set by a configurable
    subclass such as `scmle.models.sklearn.classical.Learner`. Code that
    consumes adapters only reads the attributes, so both work identically.

    Attributes:
        name: Unique registry name.
        family: Coarse family used for reporting (linear, tree, ...).
        boundary: Optional release-boundary identity. When absent,
            the coarse family is used for capability resolution.
        candidate: Whether the adapter enters automatic model search.
            Experimental adapters can be selected through an explicit project
            opt-in without silently changing the beta candidate set.
        capabilities: Declared capabilities.
        requires: Importable modules needed at build time.
        space: Tunable hyperparameters, or None if not tunable.
        params: Build parameters the conformance suite should use
            for adapters that cannot build without them (meta-models).
        param: Name of the estimator parameter that takes one
            monotonic sign per input column, or None when the model cannot
            enforce monotonic constraints.
    """

    name: str
    family: str
    boundary: str | None = None
    candidate: bool = True
    capabilities: Capabilities
    requires: tuple[str, ...] = ()
    space: search.Space | None = None
    params: Mapping[str, Any] | None = None
    param: str | None = None

    @abc.abstractmethod
    def build(
        self,
        task: tasks.Task,
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
        plan: data.Blueprint,  # noqa: ARG002 -- overridden by meta-models
        task: tasks.Task,  # noqa: ARG002 -- overridden by meta-models
        seed: int,  # noqa: ARG002 -- overridden by meta-models
        params: Mapping[str, Any] | None = None,  # noqa: ARG002 -- overridden
    ) -> Estimator | None:
        """Builds a complete pipeline itself, if the model needs to.

        Meta-models such as stacking build their members' pipelines and run
        structure-aware inner cross-validation on the raw frame; they return
        the finished candidate here. Ordinary adapters return None and get
        the standard preprocessing-plus-estimator pipeline.

        Args:
            plan: The feature plan.
            task: Resolved task type.
            seed: Random seed for deterministic behaviour.
            params: Hyperparameter overrides from the search space.

        Returns:
            The finished candidate, or None for an ordinary adapter.
        """
        return None

    def nominals(
        self,
        indices: list[int],  # noqa: ARG002 -- overridden by adapters
    ) -> Mapping[str, Any]:
        """Returns estimator parameters marking columns as native categoricals.

        Adapters declaring `native_categorical` override this; the pipeline
        then feeds high-cardinality columns as integer codes and passes
        their positions here instead of one-hot encoding them.

        Args:
            indices: Positions of the categorical columns in the matrix.

        Returns:
            Parameter overrides, empty if the adapter cannot use them.
        """
        return {}

    def accelerated(self) -> bool:
        """Returns whether this adapter can use a GPU on this machine right now.

        The default trusts the declared capability; adapters whose GPU
        support depends on the installed hardware override it.

        Returns:
            True if the adapter can use a GPU here; by default the declared
            GPU capability.
        """
        return self.capabilities.gpu

    def is_available(self) -> bool:
        """Returns whether every module in `requires` is importable.

        Returns:
            True if every module in `requires` can be found.
        """
        return all(
            importlib.util.find_spec(module) is not None
            for module in self.requires
        )

    def objection(
        self, task: tasks.Task, row_count: int | None = None
    ) -> str | None:
        """Returns why the adapter cannot be used, if it cannot.

        Checks, in order, task support, availability of the required modules
        and the `max_rows` limit, and reports the first failure.

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
            and caps.scale is not None
            and row_count > caps.scale
        ):
            return f"dataset too large ({row_count} > {caps.scale} rows)"
        return None


class Resolver(Protocol):
    """Anything that resolves a model name to its adapter (a registry)."""

    def get(self, name: str) -> Model:
        """Returns the adapter registered under `name`.

        Args:
            name: The adapter's registered name.

        Returns:
            The adapter.
        """
        ...
