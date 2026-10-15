"""Versioned, typed project specification."""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

CONFIG_VERSION = 1


class TaskType(StrEnum):
    BINARY = "binary_classification"
    MULTICLASS = "multiclass_classification"
    REGRESSION = "regression"

    @property
    def is_classification(self) -> bool:
        return self is not TaskType.REGRESSION


# Public task requests; "classification" is resolved to binary/multiclass from the target.
TaskRequest = Literal[
    "auto",
    "classification",
    "regression",
    "binary_classification",
    "multiclass_classification",
]


class _Spec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DataSpec(_Spec):
    uri: str
    target: str
    id_columns: list[str] = Field(default_factory=list)
    group_column: str | None = None
    time_column: str | None = None
    excluded_columns: list[str] = Field(default_factory=list)


class ValidationSpec(_Spec):
    strategy: Literal["auto", "stratified", "random", "group", "temporal"] = "auto"
    test_fraction: float = Field(default=0.20, gt=0, lt=1)
    folds: int = Field(default=5, ge=2)
    seed: int = 42


class ResourceSpec(_Spec):
    time_budget_seconds: int = Field(default=1800, gt=0)
    memory_limit_gb: float = Field(default=8.0, gt=0)
    cpu_limit: int | None = Field(default=None, ge=1)
    gpu: Literal["auto", "disabled", "required"] = "auto"


class ProjectSpec(_Spec):
    version: int = CONFIG_VERSION
    data: DataSpec
    task: TaskRequest = "auto"
    metric: str | None = None
    validation: ValidationSpec = Field(default_factory=ValidationSpec)
    resources: ResourceSpec = Field(default_factory=ResourceSpec)
    leakage_policy: Literal["warn", "block"] = "warn"

    @model_validator(mode="after")
    def _cross_field(self) -> Self:
        d, v = self.data, self.validation
        if v.strategy == "group" and not d.group_column:
            raise ValueError("validation.strategy='group' requires data.group_column")
        if v.strategy == "temporal" and not d.time_column:
            raise ValueError("validation.strategy='temporal' requires data.time_column")
        if v.strategy == "stratified" and self.task == "regression":
            raise ValueError("stratified validation is not defined for regression")
        special = {"target": d.target, "group_column": d.group_column, "time_column": d.time_column}
        for name, col in special.items():
            if name != "target" and col == d.target:
                raise ValueError(f"data.{name} must differ from the target")
            if col is not None and col in d.excluded_columns and name == "target":
                raise ValueError("the target cannot be excluded")
        if d.target in d.id_columns:
            raise ValueError("the target cannot be an id column")
        return self

    def canonical_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))

    def config_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()[:16]
