"""Quality gates before training and before inference, with a declared policy.

A `QualityGate` holds rules. Each rule has a `policy` that says, in advance,
what a violation means:

``fail``
    The run stops: `QualityGateError`, carrying the report.
``quarantine``
    The offending rows are removed from the data that continues, and written
    with the report under ``quarantine/<snapshot id>/`` in the store. A
    rule that is about the whole dataset (schema, freshness, row count,
    null fraction) has no rows to remove, so its quarantine holds the whole
    snapshot: a marker is written, the run stops with
    `QualityGateError(quarantined=True)` and a later run of the same
    snapshot is refused without evaluation until the marker is removed.
``warn``
    The finding is recorded in the report and the data continues unchanged.

Every rule is evaluated before anything acts, so a report is complete. The
report is written to ``quality/`` in the store whatever the outcome. Rules
apply to the stages they name (`train`, `inference`).
"""

import abc
from collections.abc import Collection
from collections.abc import Sequence
import dataclasses
import enum
import json
from typing import Any

import pandas as pd

from scmle.contracts import errors
from scmle.data.store import objectstore
from scmle.data.store import snapshot as snapshot_module

QUARANTINE_PREFIX = "quarantine"
QUALITY_PREFIX = "quality"
MARKER = "DATASET.json"


class Policy(enum.StrEnum):
    """What a violation of a rule means."""

    FAIL = "fail"
    QUARANTINE = "quarantine"
    WARN = "warn"


class Stage(enum.StrEnum):
    """When a gate runs."""

    TRAIN = "train"
    INFERENCE = "inference"


BOTH = frozenset(Stage)


class QualityGateError(errors.DataError):
    """The data failed a rule whose policy stops the run.

    Attributes:
        report: The complete gate report.
        quarantined: Whether the snapshot was held back (not just failed).
    """

    category = "quality"

    def __init__(
        self, message: str, report: "GateReport", *, quarantined: bool = False
    ) -> None:
        """Initializes the error.

        Args:
            message: What failed.
            report: The gate report.
            quarantined: Whether the snapshot is quarantined.
        """
        super().__init__(message, {"decision": report.decision})
        self.report = report
        self.quarantined = quarantined


@dataclasses.dataclass(frozen=True, slots=True)
class Finding:
    """What one rule found.

    Attributes:
        rule: Rule name.
        policy: The rule's declared policy.
        passed: Whether the data satisfied it.
        violations: Violating rows, or 1 for a dataset-level violation.
        row_level: Whether `rows` identify the violations.
        rows: Positions of violating rows (empty for dataset-level rules).
        detail: Human-readable explanation.
    """

    rule: str
    policy: Policy
    passed: bool
    violations: int
    row_level: bool
    rows: tuple[int, ...]
    detail: str

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form (row positions are capped at 20)."""
        record = dataclasses.asdict(self)
        record["policy"] = self.policy.value
        record["rows"] = list(self.rows[:20])
        return record


@dataclasses.dataclass(frozen=True, slots=True)
class GateReport:
    """The outcome of one gate run.

    Attributes:
        stage: `train` or `inference`.
        snapshot_id: The snapshot checked.
        dataset: The dataset's contract name.
        findings: One per rule that applies to the stage.
        decision: `passed`, `passed_with_warnings`, `rows_quarantined`,
            `failed` or `quarantined`.
        rows_in: Rows before the gate.
        rows_out: Rows that continue.
    """

    stage: str
    snapshot_id: str
    dataset: str
    findings: tuple[Finding, ...]
    decision: str
    rows_in: int
    rows_out: int

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form."""
        return {
            "stage": self.stage,
            "snapshot_id": self.snapshot_id,
            "dataset": self.dataset,
            "decision": self.decision,
            "rows_in": self.rows_in,
            "rows_out": self.rows_out,
            "findings": [finding.to_dict() for finding in self.findings],
        }


@dataclasses.dataclass(frozen=True, slots=True)
class GateResult:
    """The data that may continue, what was held back, and the report."""

    frame: pd.DataFrame
    quarantined: pd.DataFrame
    report: GateReport


class Rule(abc.ABC):
    """A check with a declared failure policy."""

    name = "rule"

    def __init__(
        self,
        policy: Policy = Policy.FAIL,
        stages: Collection[Stage] = BOTH,
    ) -> None:
        """Initializes the rule.

        Args:
            policy: What a violation means.
            stages: The stages the rule applies to.
        """
        self.policy = Policy(policy)
        self.stages = frozenset(stages)

    @abc.abstractmethod
    def evaluate(
        self,
        frame: pd.DataFrame,
        record: snapshot_module.SnapshotRecord,
        contract: snapshot_module.DataContract,
    ) -> Finding:
        """Checks the data."""

    def dataset_finding(self, ok: bool, detail: str) -> Finding:
        """A finding about the dataset as a whole."""
        return Finding(
            self.name, self.policy, ok, 0 if ok else 1, False, (), detail
        )

    def row_finding(self, bad: pd.Series, detail: str) -> Finding:
        """A finding about the rows where `bad` is true."""
        rows = tuple(int(i) for i in bad.to_numpy().nonzero()[0])
        return Finding(
            self.name, self.policy, not rows, len(rows), True, rows, detail
        )


class SchemaRule(Rule):
    """The observed schema equals the contract's."""

    name = "schema"

    def evaluate(
        self,
        frame: pd.DataFrame,
        record: snapshot_module.SnapshotRecord,
        contract: snapshot_module.DataContract,
    ) -> Finding:
        """See `Rule.evaluate`."""
        problems = contract.mismatches(frame)
        return self.dataset_finding(not problems, "; ".join(problems[:5]))


class FreshnessRule(Rule):
    """The data is not older than the contract allows."""

    name = "freshness"

    def evaluate(
        self,
        frame: pd.DataFrame,
        record: snapshot_module.SnapshotRecord,
        contract: snapshot_module.DataContract,
    ) -> Finding:
        """See `Rule.evaluate`."""
        stale = record.freshness == snapshot_module.STALE
        return self.dataset_finding(
            not stale,
            f"{record.age_seconds:.0f}s old, limit {record.max_age_seconds}s",
        )


class MinRowsRule(Rule):
    """The dataset has at least `minimum` rows."""

    name = "min_rows"

    def __init__(self, minimum: int, **options: Any) -> None:
        """Initializes the rule.

        Args:
            minimum: Fewest acceptable rows.
            **options: `policy` and `stages` (see `Rule`).
        """
        super().__init__(**options)
        self.minimum = minimum

    def evaluate(
        self,
        frame: pd.DataFrame,
        record: snapshot_module.SnapshotRecord,
        contract: snapshot_module.DataContract,
    ) -> Finding:
        """See `Rule.evaluate`."""
        return self.dataset_finding(
            len(frame) >= self.minimum,
            f"{len(frame)} rows, need {self.minimum}",
        )


class NullFractionRule(Rule):
    """At most `maximum` of a column is null."""

    name = "null_fraction"

    def __init__(self, column: str, maximum: float, **options: Any) -> None:
        """Initializes the rule.

        Args:
            column: Column to check.
            maximum: Largest acceptable fraction of nulls (0 to 1).
            **options: `policy` and `stages` (see `Rule`).
        """
        super().__init__(**options)
        self.column = column
        self.maximum = maximum

    def evaluate(
        self,
        frame: pd.DataFrame,
        record: snapshot_module.SnapshotRecord,
        contract: snapshot_module.DataContract,
    ) -> Finding:
        """See `Rule.evaluate`."""
        if self.column not in frame.columns:
            return self.dataset_finding(False, f"no column {self.column}")
        fraction = (
            float(frame[self.column].isna().mean()) if len(frame) else 1.0
        )
        return self.dataset_finding(
            fraction <= self.maximum,
            f"{self.column}: {fraction:.3f} null, limit {self.maximum}",
        )


class NoNullsRule(Rule):
    """No row has a null in the column."""

    name = "no_nulls"

    def __init__(self, column: str, **options: Any) -> None:
        """Initializes the rule.

        Args:
            column: Column that must be complete.
            **options: `policy` and `stages` (see `Rule`).
        """
        super().__init__(**options)
        self.column = column

    def evaluate(
        self,
        frame: pd.DataFrame,
        record: snapshot_module.SnapshotRecord,
        contract: snapshot_module.DataContract,
    ) -> Finding:
        """See `Rule.evaluate`."""
        if self.column not in frame.columns:
            return self.dataset_finding(False, f"no column {self.column}")
        return self.row_finding(
            frame[self.column].isna().reset_index(drop=True),
            f"null {self.column}",
        )


class UniqueKeyRule(Rule):
    """The contract's key columns identify rows (repeats are violations)."""

    name = "unique_key"

    def evaluate(
        self,
        frame: pd.DataFrame,
        record: snapshot_module.SnapshotRecord,
        contract: snapshot_module.DataContract,
    ) -> Finding:
        """See `Rule.evaluate`."""
        keys = list(contract.key_columns)
        if not keys or any(k not in frame.columns for k in keys):
            return self.dataset_finding(False, "no usable key columns")
        repeated = frame.duplicated(subset=keys, keep="first")
        return self.row_finding(
            repeated.reset_index(drop=True), f"repeated key {keys}"
        )


class RangeRule(Rule):
    """Values of a numeric column lie in `[minimum, maximum]`."""

    name = "range"

    def __init__(
        self,
        column: str,
        minimum: float | None = None,
        maximum: float | None = None,
        **options: Any,
    ) -> None:
        """Initializes the rule.

        Args:
            column: Numeric column.
            minimum: Smallest acceptable value, if any.
            maximum: Largest acceptable value, if any.
            **options: `policy` and `stages` (see `Rule`).
        """
        super().__init__(**options)
        self.column = column
        self.minimum = minimum
        self.maximum = maximum

    def evaluate(
        self,
        frame: pd.DataFrame,
        record: snapshot_module.SnapshotRecord,
        contract: snapshot_module.DataContract,
    ) -> Finding:
        """See `Rule.evaluate`; nulls are left to the null rules."""
        if self.column not in frame.columns:
            return self.dataset_finding(False, f"no column {self.column}")
        values = pd.to_numeric(frame[self.column], errors="coerce")
        bad = pd.Series(False, index=values.index)
        if self.minimum is not None:
            bad |= values < self.minimum
        if self.maximum is not None:
            bad |= values > self.maximum
        return self.row_finding(
            bad.reset_index(drop=True),
            f"{self.column} outside [{self.minimum}, {self.maximum}]",
        )


class AllowedValuesRule(Rule):
    """Values of a column belong to a declared set."""

    name = "allowed_values"

    def __init__(
        self, column: str, allowed: Sequence[Any], **options: Any
    ) -> None:
        """Initializes the rule.

        Args:
            column: Column to check.
            allowed: The acceptable values.
            **options: `policy` and `stages` (see `Rule`).
        """
        super().__init__(**options)
        self.column = column
        self.allowed = tuple(allowed)

    def evaluate(
        self,
        frame: pd.DataFrame,
        record: snapshot_module.SnapshotRecord,
        contract: snapshot_module.DataContract,
    ) -> Finding:
        """See `Rule.evaluate`; nulls are left to the null rules."""
        if self.column not in frame.columns:
            return self.dataset_finding(False, f"no column {self.column}")
        column = frame[self.column]
        bad = (~column.isin(self.allowed)) & column.notna()
        return self.row_finding(
            bad.reset_index(drop=True),
            f"{self.column} not in {list(self.allowed)}",
        )


class QualityGate:
    """Applies rules to a snapshot and acts on each rule's policy."""

    def __init__(
        self,
        rules: Sequence[Rule],
        store: objectstore.ObjectStore | None = None,
    ) -> None:
        """Initializes the gate.

        Args:
            rules: The rules, evaluated in order.
            store: Where reports and quarantined rows are written. Without
                it nothing is persisted, and a quarantine of a whole
                snapshot is not remembered.

        Raises:
            ConfigError: If there are no rules.
        """
        if not rules:
            raise errors.ConfigError("a gate needs at least one rule")
        self.rules = tuple(rules)
        self.store = store

    def marker_key(self, snapshot_id: str) -> str:
        """Where a whole-snapshot quarantine is recorded."""
        return f"{QUARANTINE_PREFIX}/{snapshot_id}/{MARKER}"

    def release(self, snapshot_id: str) -> None:
        """Lifts a whole-snapshot quarantine (an explicit human decision)."""
        if self.store is not None:
            self.store.delete(self.marker_key(snapshot_id))

    def run(
        self,
        frame: pd.DataFrame,
        record: snapshot_module.SnapshotRecord,
        contract: snapshot_module.DataContract,
        stage: Stage,
    ) -> GateResult:
        """Checks the data for a stage.

        Args:
            frame: The data read.
            record: Its snapshot record.
            contract: What the consumer expects.
            stage: `train` or `inference`.

        Returns:
            The rows that may continue, the rows held back, and the report.

        Raises:
            QualityGateError: If a `fail` rule is violated, or a dataset
                level `quarantine` rule is (or the snapshot was quarantined
                before).
        """
        stage = Stage(stage)
        applicable = [rule for rule in self.rules if stage in rule.stages]
        if self.store is not None and self.store.exists(
            self.marker_key(record.snapshot_id)
        ):
            report = GateReport(
                stage.value,
                record.snapshot_id,
                record.dataset,
                (),
                "quarantined",
                len(frame),
                0,
            )
            raise QualityGateError(
                f"snapshot {record.snapshot_id} is quarantined",
                report,
                quarantined=True,
            )
        findings = tuple(
            rule.evaluate(frame.reset_index(drop=True), record, contract)
            for rule in applicable
        )
        failed = [f for f in findings if not f.passed]
        stops = [f for f in failed if f.policy is Policy.FAIL]
        holds = [
            f
            for f in failed
            if f.policy is Policy.QUARANTINE and not f.row_level
        ]
        removed = sorted(
            {
                row
                for f in failed
                if f.policy is Policy.QUARANTINE and f.row_level
                for row in f.rows
            }
        )
        decision = self.decide(failed, stops, holds, removed)
        kept = frame.reset_index(drop=True).drop(index=removed)
        held = frame.reset_index(drop=True).iloc[removed]
        report = GateReport(
            stage.value,
            record.snapshot_id,
            record.dataset,
            findings,
            decision,
            len(frame),
            0 if stops or holds else len(kept),
        )
        self.persist(report, held, record, bool(holds))
        if stops:
            raise QualityGateError(
                "; ".join(f"{f.rule}: {f.detail}" for f in stops), report
            )
        if holds:
            raise QualityGateError(
                "snapshot quarantined: "
                + "; ".join(f"{f.rule}: {f.detail}" for f in holds),
                report,
                quarantined=True,
            )
        return GateResult(kept, held, report)

    def decide(
        self,
        failed: list[Finding],
        stops: list[Finding],
        holds: list[Finding],
        removed: list[int],
    ) -> str:
        """The overall decision word for a report."""
        if stops:
            return "failed"
        if holds:
            return "quarantined"
        if removed:
            return "rows_quarantined"
        return "passed_with_warnings" if failed else "passed"

    def persist(
        self,
        report: GateReport,
        held: pd.DataFrame,
        record: snapshot_module.SnapshotRecord,
        whole_snapshot: bool,
    ) -> None:
        """Writes the report, quarantined rows and the snapshot marker."""
        if self.store is None:
            return
        body = json.dumps(report.to_dict(), indent=2, sort_keys=True).encode()
        self.store.put(
            f"{QUALITY_PREFIX}/{record.snapshot_id}-{report.stage}.json", body
        )
        if len(held):
            self.store.put(
                f"{QUARANTINE_PREFIX}/{record.snapshot_id}/rows-{report.stage}.csv",
                held.to_csv(index=False).encode(),
            )
        if whole_snapshot:
            marker = {"snapshot_id": record.snapshot_id, "stage": report.stage}
            self.store.put(
                self.marker_key(record.snapshot_id),
                json.dumps(marker, sort_keys=True).encode(),
            )
