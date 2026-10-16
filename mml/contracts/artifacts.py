"""Artifact contracts: the input schema and validation report of a bundle."""

import dataclasses
from typing import Any
from typing import Self


@dataclasses.dataclass(frozen=True, slots=True)
class BundleSchema:
    """Input contract of a trained model.

    Attributes:
        target: Name of the target column.
        task: Task type value.
        numeric: Numeric feature columns.
        categorical: Categorical feature columns.
        dtypes: Training dtype per feature.
        known_categories: Categories seen in training per categorical column.
        classes: Class labels for classification, else None.
        extra_inputs: Raw columns the caller must supply because the feature
            graph reads them, although the model does not use them directly.
        series: Ordered (time-series) feature nodes recomputed on the input
            batch at inference, as plain dictionaries.
        series_columns: Columns those nodes produce that the pipeline reads.
        graph_fingerprint: Hash of the feature graph, empty without one.
        group_column: Entity column the series nodes group by, if any.
        time_column: Column ordering the rows for series nodes, if any.
    """

    target: str
    task: str
    numeric: tuple[str, ...]
    categorical: tuple[str, ...]
    dtypes: dict[str, str]
    known_categories: dict[str, list[str]]
    classes: tuple[Any, ...] | None
    extra_inputs: tuple[str, ...] = ()
    series: tuple[dict[str, Any], ...] = ()
    series_columns: tuple[str, ...] = ()
    graph_fingerprint: str = ""
    group_column: str | None = None
    time_column: str | None = None

    @property
    def features(self) -> list[str]:
        """Columns the caller must supply: model features, then graph inputs."""
        return list(
            dict.fromkeys(self.numeric + self.categorical + self.extra_inputs)
        )

    @property
    def pipeline_columns(self) -> list[str]:
        """Columns the fitted pipeline reads (supplied plus computed series)."""
        return list(dict.fromkeys([*self.features, *self.series_columns]))

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-serializable mapping."""
        record = dataclasses.asdict(self)
        record["numeric"] = list(self.numeric)
        record["categorical"] = list(self.categorical)
        record["classes"] = None if self.classes is None else list(self.classes)
        record["extra_inputs"] = list(self.extra_inputs)
        record["series"] = list(self.series)
        record["series_columns"] = list(self.series_columns)
        return record

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> Self:
        """Rebuilds a schema from `to_dict` output."""
        classes = record["classes"]
        return cls(
            target=record["target"],
            task=record["task"],
            numeric=tuple(record["numeric"]),
            categorical=tuple(record["categorical"]),
            dtypes=record["dtypes"],
            known_categories=record["known_categories"],
            classes=None if classes is None else tuple(classes),
            extra_inputs=tuple(record.get("extra_inputs", ())),
            series=tuple(record.get("series", ())),
            series_columns=tuple(record.get("series_columns", ())),
            graph_fingerprint=record.get("graph_fingerprint", ""),
            group_column=record.get("group_column"),
            time_column=record.get("time_column"),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class SchemaReport:
    """Result of validating input data against a bundle schema.

    Attributes:
        missing: Required columns absent from the input.
        extra: Input columns the model ignores.
        missing_value_counts: Missing values per present feature.
        unseen_categories: Categories never seen in training.
    """

    missing: tuple[str, ...]
    extra: tuple[str, ...]
    missing_value_counts: dict[str, int]
    unseen_categories: dict[str, list[str]]

    @property
    def valid(self) -> bool:
        """Whether every required column is present."""
        return not self.missing
