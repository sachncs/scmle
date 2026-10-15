"""Model adapter contract. Adapters declare capabilities; the planner reads them."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from mml.contracts.tasks import TaskType


class Estimator(Protocol):
    """Scikit-learn compatible estimator protocol."""

    def fit(self, X: Any, y: Any) -> Any: ...
    def predict(self, X: Any) -> Any: ...
    def get_params(self, deep: bool = True) -> dict[str, Any]: ...
    def set_params(self, **params: Any) -> Any: ...


@dataclass(frozen=True)
class Capabilities:
    tasks: frozenset[TaskType]
    sparse_input: bool = False
    probabilities: bool = True
    native_categorical: bool = False
    handles_missing: bool = False
    early_stopping: bool = False
    needs_scaling: bool = False
    serializable: bool = True
    max_rows: int | None = None  # None = no practical limit
    cost: int = 1  # relative training cost, 1 (cheap) .. 5 (expensive)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    capabilities: Capabilities
    factory: Callable[[TaskType, int], Estimator]  # (task, seed) -> estimator
    requires: tuple[str, ...] = ()  # importable modules; checked lazily
    meta: dict[str, Any] = field(default_factory=dict)
