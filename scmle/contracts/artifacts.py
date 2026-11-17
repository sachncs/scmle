"""Artifact contracts: the input schema and validation report of a bundle."""

import dataclasses
from typing import Any
from typing import ClassVar
from typing import Self

from scmle.contracts import records


@dataclasses.dataclass(frozen=True, slots=True)
class Interface:
    """Input contract of a trained model.

    Attributes:
        target: Name of the target column.
        task: Task type value.
        numeric: Numeric feature columns.
        categorical: Categorical feature columns.
        dtypes: Training dtype name per numeric and categorical feature.
        categories: Sorted categories seen in training per categorical
            column (when built from training data, capped in number).
        classes: Class labels for classification, else None.
        extras: Raw columns the caller must supply because the feature
            graph reads them, although the model does not use them directly.
        series: Ordered (time-series) feature nodes recomputed on the input
            batch at inference, as plain dictionaries.
        derived: Columns those nodes produce that the pipeline reads.
        digests: Digests by the hashed thing; ``graph`` is the hash of the
            feature graph, absent without one.
        group: Entity column the series nodes group by, if any.
        time: Column ordering the rows for series nodes, if any.
        targets: Output columns of a multi-target model, in order.
        optional: Columns used when supplied but not required (the
            sensitive column behind group-specific thresholds).
        migrations: Version steps of the stored schema.
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = (
        records.nest(
            renames={
                "known_categories": "categories",
                "extra_inputs": "extras",
                "series_columns": "derived",
                "group_column": "group",
                "time_column": "time",
                "optional_inputs": "optional",
            },
            groups={"digests": {"graph_fingerprint": "graph"}},
        ),
    )

    target: str
    task: str
    numeric: tuple[str, ...]
    categorical: tuple[str, ...]
    dtypes: dict[str, str]
    categories: dict[str, list[str]]
    classes: tuple[Any, ...] | None
    extras: tuple[str, ...] = ()
    series: tuple[dict[str, Any], ...] = ()
    derived: tuple[str, ...] = ()
    digests: dict[str, str] = dataclasses.field(default_factory=dict)
    group: str | None = None
    time: str | None = None
    optional: tuple[str, ...] = ()
    targets: tuple[str, ...] = ()

    @property
    def features(self) -> list[str]:
        """Columns the caller must supply: model features, then graph inputs."""
        return list(
            dict.fromkeys(self.numeric + self.categorical + self.extras)
        )

    @property
    def consumed(self) -> list[str]:
        """Columns the fitted pipeline reads (supplied plus computed series)."""
        return list(dict.fromkeys([*self.features, *self.derived]))

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-serializable mapping."""
        record = dataclasses.asdict(self)
        record["numeric"] = list(self.numeric)
        record["categorical"] = list(self.categorical)
        record["classes"] = None if self.classes is None else list(self.classes)
        record["extras"] = list(self.extras)
        record["series"] = list(self.series)
        record["derived"] = list(self.derived)
        record[records.KEY] = records.ORIGIN + len(self.migrations)
        return record

    def anchor(self) -> records.Layout:
        """Returns the schema in its version-1 layout, for identity digests.

        A digest that names the schema across runs (the serving schema
        digest) is computed on this layout, so renaming a stored key never
        changes it.
        """
        record = self.to_dict()
        del record[records.KEY]
        layout = records.downgrade(self.migrations, record)
        # Version 1 always wrote the fingerprint, empty without a graph.
        layout.setdefault("graph_fingerprint", "")
        return layout

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> Self:
        """Rebuilds a schema from `to_dict` output.

        A schema stored at an older schema version is migrated first.

        Args:
            record: A mapping as produced by `to_dict`, at any supported
                schema version; the later fields default to empty when
                absent.

        Returns:
            The schema.
        """
        record = records.load(cls.__name__, cls.migrations, record)
        classes = record["classes"]
        digests = {
            str(key): str(value)
            for key, value in record.get("digests", {}).items()
        }
        return cls(
            target=record["target"],
            task=record["task"],
            numeric=tuple(record["numeric"]),
            categorical=tuple(record["categorical"]),
            dtypes=record["dtypes"],
            categories=record["categories"],
            classes=None if classes is None else tuple(classes),
            extras=tuple(record.get("extras", ())),
            series=tuple(record.get("series", ())),
            derived=tuple(record.get("derived", ())),
            digests=digests,
            group=record.get("group"),
            time=record.get("time"),
            optional=tuple(record.get("optional", ())),
            targets=tuple(record.get("targets", ())),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Conformance:
    """Result of validating input data against a bundle schema.

    Attributes:
        missing: Required columns absent from the input.
        extra: Input columns that are neither required features nor the
            target column.
        missing_value_counts: Count of missing values for each present feature
            that has at least one.
        unseen_categories: Per categorical column with any, the sorted values
            never seen in training (at most the first 20).
    """

    missing: tuple[str, ...]
    extra: tuple[str, ...]
    missing_value_counts: dict[str, int]
    unseen_categories: dict[str, list[str]]

    @property
    def valid(self) -> bool:
        """Whether every required column is present."""
        return not self.missing
