"""Typed, versioned project specification (pydantic)."""

import enum
import hashlib
import json
from typing import Literal
from typing import Self

import pydantic

CONFIG_VERSION = 1


class TaskType(enum.StrEnum):
    """Supervised learning task kinds."""

    BINARY = "binary_classification"
    MULTICLASS = "multiclass_classification"
    REGRESSION = "regression"

    @property
    def is_classification(self) -> bool:
        """Whether the task predicts discrete classes."""
        return self is not TaskType.REGRESSION


TaskRequest = Literal[
    "auto",
    "classification",
    "regression",
    "binary_classification",
    "multiclass_classification",
]
"""Public task requests; `classification` is resolved from the target."""


class FrozenModel(pydantic.BaseModel):
    """Immutable pydantic base that rejects unknown fields."""

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)


class DataSpec(FrozenModel):
    """Where the data lives and what its columns mean."""

    uri: str
    target: str
    id_columns: list[str] = pydantic.Field(default_factory=list)
    group_column: str | None = None
    time_column: str | None = None
    excluded_columns: list[str] = pydantic.Field(default_factory=list)


class ValidationSpec(FrozenModel):
    """How data is partitioned for model selection and final testing."""

    strategy: Literal["auto", "stratified", "random", "group", "temporal"] = (
        "auto"
    )
    test_fraction: float = pydantic.Field(default=0.20, gt=0, lt=1)
    folds: int = pydantic.Field(default=5, ge=2)
    seed: int = 42


class ResourceSpec(FrozenModel):
    """Compute limits for a run."""

    time_budget_seconds: int = pydantic.Field(default=1800, gt=0)
    memory_limit_gb: float = pydantic.Field(default=8.0, gt=0)
    cpu_limit: int | None = pydantic.Field(default=None, ge=1)
    workers: int = pydantic.Field(default=1, ge=1)
    gpu: Literal["auto", "disabled", "required"] = "auto"


class SearchSpec(FrozenModel):
    """Hyperparameter search settings (disabled when `top_models` is 0)."""

    top_models: int = pydantic.Field(default=2, ge=0)
    configs_per_model: int = pydantic.Field(default=6, ge=1)
    optimizer: Literal["random", "optuna"] = "random"
    prune_fraction: float = pydantic.Field(default=0.5, ge=0, lt=1)


class StrategySpec(FrozenModel):
    """Which data strategies to evaluate (imbalance and feature variants)."""

    names: list[str] = pydantic.Field(
        default_factory=lambda: [
            "class_weight",
            "random_oversample",
            "random_undersample",
            "log_skewed",
            "interactions",
        ]
    )
    top_models: int = pydantic.Field(default=2, ge=0)


class ProjectSpec(FrozenModel):
    """The explicit, versioned specification of an ML project."""

    version: int = CONFIG_VERSION
    data: DataSpec
    task: TaskRequest = "auto"
    metric: str | None = None
    validation: ValidationSpec = pydantic.Field(default_factory=ValidationSpec)
    resources: ResourceSpec = pydantic.Field(default_factory=ResourceSpec)
    search: SearchSpec = pydantic.Field(default_factory=SearchSpec)
    strategies: StrategySpec = pydantic.Field(default_factory=StrategySpec)
    leakage_policy: Literal["warn", "block"] = "warn"

    @pydantic.model_validator(mode="after")
    def check_cross_field_rules(self) -> Self:
        """Validates constraints that span several fields.

        Returns:
            The validated specification.

        Raises:
            ValueError: If the combination of fields is inconsistent.
        """
        data, validation = self.data, self.validation
        if validation.strategy == "group" and not data.group_column:
            raise ValueError("strategy 'group' requires data.group_column")
        if validation.strategy == "temporal" and not data.time_column:
            raise ValueError("strategy 'temporal' requires data.time_column")
        if validation.strategy == "stratified" and self.task == "regression":
            raise ValueError(
                "stratified validation is undefined for regression"
            )
        for name in ("group_column", "time_column"):
            if getattr(data, name) == data.target:
                raise ValueError(f"data.{name} must differ from the target")
        if data.target in data.id_columns:
            raise ValueError("the target cannot be an id column")
        if data.target in data.excluded_columns:
            raise ValueError("the target cannot be excluded")
        return self

    def canonical_json(self) -> str:
        """Returns a key-sorted, whitespace-free JSON rendering."""
        return json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        )

    def config_hash(self) -> str:
        """Returns a short deterministic hash of the canonical configuration."""
        digest = hashlib.sha256(self.canonical_json().encode()).hexdigest()
        return digest[:16]
