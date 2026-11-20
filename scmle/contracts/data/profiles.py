"""Feature plans, profiles, curation and datasheets of a dataset."""

from collections.abc import Mapping
import dataclasses
from typing import Any
from typing import ClassVar
from typing import Self

from scmle.contracts import errors
from scmle.contracts import events
from scmle.contracts import records
from scmle.contracts.data import findings
from scmle.contracts.data import provenance

HIGH_CARDINALITY_LEVELS = 20

BLUEPRINT = records.Migration(
    renames={"extra_inputs": "extras", "high_cardinality": "cardinal"}
)
"""Version 2 of a stored feature plan: the two multi-word keys it writes."""

FINDINGS = {
    "findings": {"quality_findings": "quality", "risk_findings": "risk"}
}
"""Quality and risk findings, stored together under `findings`."""


@dataclasses.dataclass(frozen=True, slots=True)
class Blueprint:
    """Which columns become features and why the others were dropped.

    Attributes:
        numeric: Numeric feature columns.
        categorical: Categorical feature columns.
        text: Free-text columns (encoded as TF-IDF terms; also listed in
            `extras` so inference requires them).
        dropped: Mapping of dropped column to the reason.
        derived: Row-wise feature-graph nodes computed inside the pipeline.
        series: Ordered (time-series) feature-graph nodes.
        extras: Raw columns only the feature graph reads.
        monotonic: Numeric feature to +1 / -1 monotonic constraint.
        cardinal: Categorical features with many distinct levels
            (`HIGH_CARDINALITY_LEVELS` or more), which one-hot encoding
            handles poorly.
        group_column: Entity column, so inner cross-validation keeps entities
            together.
        time_column: Ordering column, so inner cross-validation keeps time
            order.
        weight_column: Column of per-row training weights, if any.
        event_column: Survival event indicator column, if any.
        ordinal_levels: Number of ordered levels of an ordinal target.
        query_column: Ranking query identifier column, if any.
        target_columns: Every target column of a multi-target task.
        value_column: The series value column of a forecasting task.
        treatment_column: Uplift treatment indicator column, if any.
        binary_outcome: Whether an uplift outcome is a 0/1 label.
        text_encoding: `tfidf` or `embedding` for free-text columns.
        embedding_model: Sentence-embedding model for `embedding`.
        informative_missing: Features whose gaps are associated with the
            target; `None` when missingness was not assessed.
        migrations: Version steps of the stored plan; version 2 writes
            `extra_inputs` as `extras` and `high_cardinality` as
            `cardinal`. The other fields are not written.
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = (BLUEPRINT,)

    numeric: tuple[str, ...]
    categorical: tuple[str, ...]
    dropped: Mapping[str, str] = dataclasses.field(default_factory=dict)
    text: tuple[str, ...] = ()
    monotonic: Mapping[str, int] = dataclasses.field(default_factory=dict)
    derived: tuple[Any, ...] = ()
    series: tuple[Any, ...] = ()
    extras: tuple[str, ...] = ()
    cardinal: tuple[str, ...] = ()
    group_column: str | None = None
    time_column: str | None = None
    weight_column: str | None = None
    event_column: str | None = None
    ordinal_levels: int | None = None
    query_column: str | None = None
    target_columns: tuple[str, ...] = ()
    value_column: str | None = None
    treatment_column: str | None = None
    binary_outcome: bool = True
    text_encoding: str = "tfidf"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    informative_missing: tuple[str, ...] | None = None

    @property
    def features(self) -> tuple[str, ...]:
        """All feature columns the model consumes, numeric first."""
        return self.numeric + self.categorical

    @property
    def engineered(self) -> frozenset[str]:
        """Columns computed by the feature graph (row-wise or ordered)."""
        return frozenset(n.name for n in (*self.derived, *self.series))

    @property
    def sourced(self) -> tuple[str, ...]:
        """Raw columns the fitted pipeline reads (plus series features).

        These exist in the frame a pipeline is fitted on: model features that
        are not computed inside the pipeline, graph-only inputs, and the
        ordered (series) features computed at ingestion.
        """
        rowwise = frozenset(n.name for n in self.derived)
        columns = [c for c in self.features if c not in rowwise]
        return tuple(dict.fromkeys([*columns, *self.extras]))

    def pick(self, kind_name: str) -> tuple[str, ...]:
        """Returns the feature columns of one kind.

        The name is looked up with `getattr`, so any attribute name returns
        that attribute's value unchanged.

        Args:
            kind_name: Name of a column kind such as `numeric` or
                `categorical`.

        Returns:
            The attribute stored under that name, or an empty tuple if there
            is no such attribute.
        """
        return getattr(self, kind_name, ())

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-serializable mapping."""
        return {
            "numeric": list(self.numeric),
            "categorical": list(self.categorical),
            "text": list(self.text),
            "dropped": dict(self.dropped),
            "derived": [n.model_dump(mode="json") for n in self.derived],
            "series": [n.model_dump(mode="json") for n in self.series],
            "extras": list(self.extras),
            "cardinal": list(self.cardinal),
        }


@dataclasses.dataclass(frozen=True, slots=True)
class Portrait:
    """Statistical profile and findings for a dataset.

    Attributes:
        manifest: Identity of the profiled dataset.
        statistics: Per-column statistics.
        quality: Data-quality observations.
        risk_findings: Leakage / imbalance / shift hypotheses (not verdicts).
        summary: Target distribution summary.
        plan: Resulting feature selection.
        overview: Dataset-level fingerprint grouped by dimension
            (shape, types, missingness, target, features, relationships,
            structure, quality, operational, risk).
        missingness: Per-column tests of whether gaps relate to the target.
        evidence: JSON-native records for reproducible quality checks.
        leakage: The leakage screen (`screen.Screen.to_dict`): its level,
            family size, suspects and the suspects that may block a run. Empty
            when the screen did not apply.
        migrations: Version steps of the stored profile; version 2 writes
            the findings as `findings.{quality, risk}`, the feature plan as
            `plan` (in its version-2 layout), the dataset statistics as
            `statistics` and the quality evidence as `evidence`.
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = (
        records.nest(
            renames={
                "feature_plan": "plan",
                "dataset": "statistics",
                "quality_evidence": "evidence",
            },
            parts={"plan": BLUEPRINT},
            groups=FINDINGS,
        ),
    )

    manifest: provenance.Manifest
    statistics: Mapping[str, Any]
    quality: tuple[str, ...]
    risk_findings: tuple[str, ...]
    summary: Mapping[str, Any]
    plan: Blueprint
    overview: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    missingness: tuple[Mapping[str, Any], ...] = ()
    evidence: tuple[dict[str, object], ...] = ()
    leakage: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def blockers(self) -> tuple[str, ...]:
        """Columns whose leakage suspicion is certified and may block a run."""
        return tuple(self.leakage.get("blocking", ()))

    @property
    def untested(self) -> tuple[str, ...]:
        """Columns the leakage screen declined to test (not clean)."""
        return tuple(map(str, self.leakage.get("unchecked", ())))

    @property
    def admonition(self) -> str:
        """The loud statement of an incomplete screen, or an empty string."""
        return str(self.leakage.get("warning", ""))

    def __post_init__(self) -> None:
        """Validates and detaches structured quality evidence."""
        normalized = tuple(
            findings.normalise(item, f"quality_evidence[{index}]")
            for index, item in enumerate(self.evidence)
        )
        object.__setattr__(self, "evidence", normalized)

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-serializable mapping."""
        return {
            "manifest": self.manifest.to_dict(),
            "columns": dict(self.statistics),
            "findings": {
                "quality": list(self.quality),
                "risk": list(self.risk_findings),
            },
            "target": dict(self.summary),
            "plan": self.plan.to_dict(),
            "statistics": dict(self.overview),
            "missingness": [dict(item) for item in self.missingness],
            "evidence": [dict(item) for item in self.evidence],
            "leakage": dict(self.leakage),
        }


@dataclasses.dataclass(frozen=True, slots=True)
class Curation:
    """Auditable record of what happened before modelling.

    This report deliberately contains no model scores. It describes the
    source observed at ingestion, the rows used for profiling, repairs made,
    quality and risk findings, and the resulting feature decisions.

    Version 2 of the stored report writes single-word keys:
    `rows.{source, profiled}`, `read` (seconds), `notes`,
    `findings.{quality, risk}`, `features`, `dropped`, `ready` and
    `blockers`; the fields keep their names. `migrations` moves a stored
    report between the versions.
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = (
        records.nest(
            renames={
                "read_seconds": "read",
                "cleaning_notes": "notes",
                "feature_columns": "features",
                "dropped_columns": "dropped",
                "release_ready": "ready",
                "unresolved_blocking_findings": "blockers",
            },
            groups={
                "rows": {"source_rows": "source", "profiled_rows": "profiled"},
                **FINDINGS,
            },
        ),
    )

    manifest: provenance.Manifest
    source_rows: int
    rows: int
    read: float
    streamed: bool
    notes: tuple[str, ...]
    quality: tuple[findings.Finding, ...]
    risk_findings: tuple[findings.Finding, ...]
    features: Mapping[str, tuple[str, ...]]
    dropped: Mapping[str, str]
    schema_version: int = records.ORIGIN + len(migrations)

    @property
    def blockers(self) -> tuple[findings.Finding, ...]:
        """The blocking findings without an explicit safe disposition."""
        safe_dispositions = {
            findings.Disposition.RESOLVED,
            findings.Disposition.WAIVED,
            findings.Disposition.FALSE_POSITIVE,
        }
        return tuple(
            finding
            for finding in (*self.quality, *self.risk_findings)
            if finding.blocking and finding.disposition not in safe_dispositions
        )

    @property
    def release(self) -> bool:
        """Whether curation has no unresolved blocking findings."""
        return not self.blockers

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-serializable curation record."""
        return {
            "schema_version": self.schema_version,
            "manifest": self.manifest.to_dict(),
            "read": self.read,
            "streamed": self.streamed,
            "notes": list(self.notes),
            "features": {
                name: list(columns) for name, columns in self.features.items()
            },
            "dropped": dict(self.dropped),
            "ready": self.release,
            "blockers": [finding.identifier for finding in self.blockers],
            "rows": {
                "source": self.source_rows,
                "profiled": self.rows,
            },
            "findings": {
                "quality": [finding.to_dict() for finding in self.quality],
                "risk": [finding.to_dict() for finding in self.risk_findings],
            },
        }


@dataclasses.dataclass(frozen=True, slots=True)
class Datasheet:
    """Human-reviewable summary of a dataset snapshot before modelling.

    A card is generated from the curation report, but it is not a release
    approval. Its explicit intended use, exclusions, limitations, privacy
    context, and open findings are the minimum context a reviewer needs before
    trusting a model result derived from the snapshot.

    Version 2 of the stored card writes single-word keys: `dataset`,
    `snapshot`, `use`, `excluded`, `features`, `findings.{quality, risk}`,
    `review` and `generated`; the fields keep their names. `migrations`
    moves a stored card between the versions.
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = (
        records.nest(
            renames={
                "dataset_id": "dataset",
                "snapshot_id": "snapshot",
                "intended_use": "use",
                "out_of_scope": "excluded",
                "feature_summary": "features",
                "review_status": "review",
                "generated_at": "generated",
            },
            groups=FINDINGS,
        ),
    )

    dataset: str
    snapshot: str
    modality: str
    task: str
    target: str
    use: str
    excluded: tuple[str, ...]
    composition: Mapping[str, int]
    summary: Mapping[str, tuple[str, ...]]
    limitations: tuple[str, ...]
    quality: tuple[findings.Finding, ...]
    risk_findings: tuple[findings.Finding, ...]
    source: provenance.Origin
    lineage: provenance.Ancestry
    privacy: provenance.Confidentiality
    owner: str
    status: str = "machine_generated_unreviewed"
    schema_version: int = records.ORIGIN + len(migrations)
    generated: str = dataclasses.field(default_factory=events.stamp)

    def __post_init__(self) -> None:
        """Validates the fields that make the card useful to reviewers.

        Raises:
            ValueError: If a required text field (dataset id, modality,
                intended use, owner, review status or generation time) is
                blank, the schema version is below one, a composition count
                is negative, or the out-of-scope or limitations list is
                empty.
        """
        for name, value in (
            ("dataset_id", self.dataset),
            ("modality", self.modality),
            ("intended_use", self.use),
            ("owner", self.owner),
            ("review_status", self.status),
            ("generated_at", self.generated),
        ):
            if not value.strip():
                raise errors.InputError(f"{name} must not be empty")
        if self.schema_version < 1:
            raise errors.InputError("schema_version must be positive")
        if any(value < 0 for value in self.composition.values()):
            raise errors.InputError("composition counts must not be negative")
        if not self.excluded:
            raise errors.InputError("out_of_scope must not be empty")
        if not self.limitations:
            raise errors.InputError("limitations must not be empty")

    @classmethod
    def distill(cls, report: Curation) -> Self:
        """Builds a card from the immutable curation evidence.

        Args:
            report: The curation report.

        Returns:
            The card, with limitations that include the report's cleaning notes.
        """
        manifest = report.manifest
        task = manifest.task or "the declared learning task"
        limitations = [
            "This machine-generated card requires human review before release.",
            "Curation findings are observations, not proof of data quality.",
        ]
        limitations.extend(report.notes)
        if not report.risk_findings:
            limitations.append("No risk finding was emitted by the profiler.")
        return cls(
            dataset=manifest.identifier,
            snapshot=manifest.snapshot,
            modality=manifest.modality,
            task=manifest.task,
            target=manifest.target,
            use=(
                f"Training and evaluation for {task} using the declared "
                "target and feature decisions."
            ),
            excluded=(
                "It is not evidence of production-population coverage.",
                "It is not a release approval or a causal claim.",
                (
                    "It must not be used outside the declared task and "
                    "privacy policy."
                ),
            ),
            composition={
                "source_rows": report.source_rows,
                "profiled_rows": report.rows,
                "feature_columns": sum(
                    len(columns) for columns in report.features.values()
                ),
                "dropped_columns": len(report.dropped),
            },
            summary=report.features,
            limitations=tuple(limitations),
            quality=report.quality,
            risk_findings=report.risk_findings,
            source=manifest.source,
            lineage=manifest.lineage,
            privacy=manifest.privacy,
            owner=manifest.source.owner or "unassigned",
        )

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-serializable dataset card."""
        return {
            "schema_version": self.schema_version,
            "dataset": self.dataset,
            "snapshot": self.snapshot,
            "modality": self.modality,
            "task": self.task,
            "target": self.target,
            "use": self.use,
            "excluded": list(self.excluded),
            "composition": dict(self.composition),
            "features": {
                name: list(columns) for name, columns in self.summary.items()
            },
            "limitations": list(self.limitations),
            "source": self.source.to_dict(),
            "lineage": self.lineage.to_dict(),
            "privacy": self.privacy.to_dict(),
            "owner": self.owner,
            "review": self.status,
            "generated": self.generated,
            "findings": {
                "quality": [finding.to_dict() for finding in self.quality],
                "risk": [finding.to_dict() for finding in self.risk_findings],
            },
        }


@dataclasses.dataclass(frozen=True, slots=True)
class Preprocessing:
    """Switches that shape the preprocessing transformers.

    Attributes:
        scale: Standardize numeric features.
        skewed: Log-transform strongly right-skewed numeric features.
        interactions: Add pairwise numeric interaction features.
        impute: Impute missing numeric values (off for models that handle
            missing values natively).
        cardinality: How high-cardinality categorical features are
            encoded: `onehot`, `target` (regularized, cross-fitted target
            encoding) or `ordinal` (integer codes for native categorical
            handling).
        sparse: Emit sparse one-hot output (for sparse-capable models).
        construct: Append differences and ratios of the numeric features
            most associated with the target.
        select: Keep only the columns an L1-penalised linear model uses.
        missing: How missing numeric values are filled: `median`,
            `indicator` (median plus a flag column per feature that has
            missing values) or `iterative` (each feature predicted from the
            others).
        encoding: `tfidf` or `embedding` for free-text columns.
        model: Sentence-embedding model for `embedding`.
    """

    scale: bool = False
    skewed: bool = False
    interactions: bool = False
    impute: bool = True
    cardinality: str = "onehot"
    sparse: bool = False
    construct: bool = False
    select: bool = False
    missing: str = "median"
    encoding: str = "tfidf"
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
