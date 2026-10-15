"""Dataset identity and profile contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DatasetManifest:
    dataset_id: str
    schema_hash: str
    data_hash: str
    row_count: int
    column_count: int
    target: str
    uri: str = ""


@dataclass(frozen=True)
class FeaturePlan:
    """Which columns become model features, and why others were dropped."""

    numeric: tuple[str, ...]
    categorical: tuple[str, ...]
    dropped: Mapping[str, str] = field(default_factory=dict)

    @property
    def features(self) -> tuple[str, ...]:
        return self.numeric + self.categorical


@dataclass(frozen=True)
class DatasetProfile:
    manifest: DatasetManifest
    column_statistics: Mapping[str, Any]
    quality_findings: tuple[str, ...]
    risk_findings: tuple[str, ...]
    target_summary: Mapping[str, Any] = field(default_factory=dict)
    feature_plan: FeaturePlan | None = None
