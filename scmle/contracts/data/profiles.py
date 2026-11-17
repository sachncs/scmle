"""Feature plans, profiles, curation and datasheets of a dataset."""

from collections.abc import Mapping
import dataclasses
from typing import Any
from typing import Self

from scmle.contracts import errors
from scmle.contracts import events
from scmle.contracts.data import findings
from scmle.contracts.data import provenance

HIGH_CARDINALITY_LEVELS = 20


@dataclasses.dataclass(frozen=True, slots=True)
class Blueprint:
    """Which columns become features and why the others were dropped.

    Attributes:
        numeric: Numeric feature columns.
        categorical: Categorical feature columns.
        text: Free-text columns (encoded as TF-IDF terms; also listed in
            `extra_inputs` so inference requires them).
        dropped: Mapping of dropped column to the reason.
        derived: Row-wise feature-graph nodes computed inside the pipeline.
        series: Ordered (time-series) feature-graph nodes.
        extra_inputs: Raw columns only the feature graph reads.
        monotonic: Numeric feature to +1 / -1 monotonic constraint.
        high_cardinality: Categorical features with many distinct levels
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
    """

    numeric: tuple[str, ...]
    categorical: tuple[str, ...]
    dropped: Mapping[str, str] = dataclasses.field(default_factory=dict)
    text: tuple[str, ...] = ()
    monotonic: Mapping[str, int] = dataclasses.field(default_factory=dict)
    derived: tuple[Any, ...] = ()
    series: tuple[Any, ...] = ()
    extra_inputs: tuple[str, ...] = ()
    high_cardinality: tuple[str, ...] = ()
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
        return tuple(dict.fromkeys([*columns, *self.extra_inputs]))

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
            "extra_inputs": list(self.extra_inputs),
            "high_cardinality": list(self.high_cardinality),
        }


@dataclasses.dataclass(frozen=True, slots=True)
class Portrait:
    """Statistical profile and findings for a dataset.

    Attributes:
        manifest: Identity of the profiled dataset.
        column_statistics: Per-column statistics.
        quality_findings: Data-quality observations.
        risk_findings: Leakage / imbalance / shift hypotheses (not verdicts).
        target_summary: Target distribution summary.
        feature_plan: Resulting feature selection.
        dataset_statistics: Dataset-level fingerprint grouped by dimension
            (shape, types, missingness, target, features, relationships,
            structure, quality, operational, risk).
        missingness: Per-column tests of whether gaps relate to the target.
        quality_evidence: JSON-native records for reproducible quality checks.
        leakage: The leakage screen (`screen.Screen.to_dict`): its level,
            family size, suspects and the suspects that may block a run. Empty
            when the screen did not apply.
    """

    manifest: provenance.DatasetManifest
    column_statistics: Mapping[str, Any]
    quality_findings: tuple[str, ...]
    risk_findings: tuple[str, ...]
    target_summary: Mapping[str, Any]
    feature_plan: Blueprint
    dataset_statistics: Mapping[str, Any] = dataclasses.field(
        default_factory=dict
    )
    missingness: tuple[Mapping[str, Any], ...] = ()
    quality_evidence: tuple[dict[str, object], ...] = ()
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
            for index, item in enumerate(self.quality_evidence)
        )
        object.__setattr__(self, "quality_evidence", normalized)

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-serializable mapping."""
        return {
            "manifest": self.manifest.to_dict(),
            "columns": dict(self.column_statistics),
            "quality_findings": list(self.quality_findings),
            "risk_findings": list(self.risk_findings),
            "target": dict(self.target_summary),
            "feature_plan": self.feature_plan.to_dict(),
            "dataset": dict(self.dataset_statistics),
            "missingness": [dict(item) for item in self.missingness],
            "quality_evidence": [dict(item) for item in self.quality_evidence],
            "leakage": dict(self.leakage),
        }


@dataclasses.dataclass(frozen=True, slots=True)
class Curation:
    """Auditable record of what happened before modelling.

    This report deliberately contains no model scores. It describes the
    source observed at ingestion, the rows used for profiling, repairs made,
    quality and risk findings, and the resulting feature decisions.
    """

    manifest: provenance.DatasetManifest
    source_rows: int
    profiled_rows: int
    read_seconds: float
    streamed: bool
    cleaning_notes: tuple[str, ...]
    quality_findings: tuple[findings.Finding, ...]
    risk_findings: tuple[findings.Finding, ...]
    feature_columns: Mapping[str, tuple[str, ...]]
    dropped_columns: Mapping[str, str]
    schema_version: int = 1

    @property
    def unresolved_blocking_findings(self) -> tuple[findings.Finding, ...]:
        """The blocking findings without an explicit safe disposition."""
        safe_dispositions = {
            findings.Disposition.RESOLVED,
            findings.Disposition.WAIVED,
            findings.Disposition.FALSE_POSITIVE,
        }
        return tuple(
            finding
            for finding in (*self.quality_findings, *self.risk_findings)
            if finding.blocking and finding.disposition not in safe_dispositions
        )

    @property
    def release_ready(self) -> bool:
        """Whether curation has no unresolved blocking findings."""
        return not self.unresolved_blocking_findings

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-serializable curation record."""
        return {
            "schema_version": self.schema_version,
            "manifest": self.manifest.to_dict(),
            "source_rows": self.source_rows,
            "profiled_rows": self.profiled_rows,
            "read_seconds": self.read_seconds,
            "streamed": self.streamed,
            "cleaning_notes": list(self.cleaning_notes),
            "quality_findings": [
                finding.to_dict() for finding in self.quality_findings
            ],
            "risk_findings": [
                finding.to_dict() for finding in self.risk_findings
            ],
            "feature_columns": {
                name: list(columns)
                for name, columns in self.feature_columns.items()
            },
            "dropped_columns": dict(self.dropped_columns),
            "release_ready": self.release_ready,
            "unresolved_blocking_findings": [
                finding.identifier
                for finding in self.unresolved_blocking_findings
            ],
        }


@dataclasses.dataclass(frozen=True, slots=True)
class Datasheet:
    """Human-reviewable summary of a dataset snapshot before modelling.

    A card is generated from the curation report, but it is not a release
    approval. Its explicit intended use, exclusions, limitations, privacy
    context, and open findings are the minimum context a reviewer needs before
    trusting a model result derived from the snapshot.
    """

    dataset_id: str
    snapshot_id: str
    modality: str
    task: str
    target: str
    intended_use: str
    out_of_scope: tuple[str, ...]
    composition: Mapping[str, int]
    feature_summary: Mapping[str, tuple[str, ...]]
    limitations: tuple[str, ...]
    quality_findings: tuple[findings.Finding, ...]
    risk_findings: tuple[findings.Finding, ...]
    source: provenance.Origin
    lineage: provenance.Ancestry
    privacy: provenance.Confidentiality
    owner: str
    review_status: str = "machine_generated_unreviewed"
    schema_version: int = 1
    generated_at: str = dataclasses.field(default_factory=events.stamp)

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
            ("dataset_id", self.dataset_id),
            ("modality", self.modality),
            ("intended_use", self.intended_use),
            ("owner", self.owner),
            ("review_status", self.review_status),
            ("generated_at", self.generated_at),
        ):
            if not value.strip():
                raise errors.InputError(f"{name} must not be empty")
        if self.schema_version < 1:
            raise errors.InputError("schema_version must be positive")
        if any(value < 0 for value in self.composition.values()):
            raise errors.InputError("composition counts must not be negative")
        if not self.out_of_scope:
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
        limitations.extend(report.cleaning_notes)
        if not report.risk_findings:
            limitations.append("No risk finding was emitted by the profiler.")
        return cls(
            dataset_id=manifest.identifier,
            snapshot_id=manifest.snapshot,
            modality=manifest.modality,
            task=manifest.task,
            target=manifest.target,
            intended_use=(
                f"Training and evaluation for {task} using the declared "
                "target and feature decisions."
            ),
            out_of_scope=(
                "It is not evidence of production-population coverage.",
                "It is not a release approval or a causal claim.",
                (
                    "It must not be used outside the declared task and "
                    "privacy policy."
                ),
            ),
            composition={
                "source_rows": report.source_rows,
                "profiled_rows": report.profiled_rows,
                "feature_columns": sum(
                    len(columns) for columns in report.feature_columns.values()
                ),
                "dropped_columns": len(report.dropped_columns),
            },
            feature_summary=report.feature_columns,
            limitations=tuple(limitations),
            quality_findings=report.quality_findings,
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
            "dataset_id": self.dataset_id,
            "snapshot_id": self.snapshot_id,
            "modality": self.modality,
            "task": self.task,
            "target": self.target,
            "intended_use": self.intended_use,
            "out_of_scope": list(self.out_of_scope),
            "composition": dict(self.composition),
            "feature_summary": {
                name: list(columns)
                for name, columns in self.feature_summary.items()
            },
            "limitations": list(self.limitations),
            "quality_findings": [
                finding.to_dict() for finding in self.quality_findings
            ],
            "risk_findings": [
                finding.to_dict() for finding in self.risk_findings
            ],
            "source": self.source.to_dict(),
            "lineage": self.lineage.to_dict(),
            "privacy": self.privacy.to_dict(),
            "owner": self.owner,
            "review_status": self.review_status,
            "generated_at": self.generated_at,
        }


@dataclasses.dataclass(frozen=True, slots=True)
class Preprocessing:
    """Switches that shape the preprocessing transformers.

    Attributes:
        scale: Standardize numeric features.
        log_skewed: Log-transform strongly right-skewed numeric features.
        interactions: Add pairwise numeric interaction features.
        impute: Impute missing numeric values (off for models that handle
            missing values natively).
        high_cardinality: How high-cardinality categorical features are
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
        text_encoding: `tfidf` or `embedding` for free-text columns.
        embedding_model: Sentence-embedding model for `embedding`.
    """

    scale: bool = False
    log_skewed: bool = False
    interactions: bool = False
    impute: bool = True
    high_cardinality: str = "onehot"
    sparse: bool = False
    construct: bool = False
    select: bool = False
    missing: str = "median"
    text_encoding: str = "tfidf"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
