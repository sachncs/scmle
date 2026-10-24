"""Drift detection, delayed-label performance and retraining advice."""

import abc
import dataclasses
from typing import Any

import numpy as np
import pandas as pd

from scmle.artifacts import bundle as bundle_module
from scmle.contracts import errors
from scmle.contracts import tasks
from scmle.evaluation import metrics
from scmle.monitoring import reference as reference_module

STABLE = "stable"
MODERATE = "moderate"
HIGH = "high"
SEVERITY = {STABLE: 0, MODERATE: 1, HIGH: 2}


@dataclasses.dataclass(frozen=True)
class MonitoringThresholds:
    """Decision thresholds for monitoring.

    Attributes:
        psi_moderate: PSI at which drift is noteworthy.
        psi_high: PSI at which drift is large.
        high_drift_feature_share: Share of features with large drift that
            triggers a retraining recommendation.
        unseen_category_rate: Unseen-category rate that needs investigation.
        min_labeled_rows: Labels needed before performance is judged.
        performance_tolerance: Relative metric degradation tolerated.
    """

    psi_moderate: float = 0.1
    psi_high: float = 0.25
    high_drift_feature_share: float = 0.3
    unseen_category_rate: float = 0.2
    min_labeled_rows: int = 30
    performance_tolerance: float = 0.10

    def status(self, psi: float) -> str:
        """Classifies a PSI value."""
        if psi >= self.psi_high:
            return HIGH
        return MODERATE if psi >= self.psi_moderate else STABLE


@dataclasses.dataclass(frozen=True)
class FeatureDrift:
    """Drift of one feature.

    Attributes:
        feature: Column name.
        psi: Noise-adjusted population stability index vs training.
        status: stable / moderate / high.
        missing_rate: Missing share in the new data.
        reference_missing_rate: Missing share in training.
        unseen_rate: Share of never-seen categories (categoricals).
        raw_psi: PSI before removing the sampling-noise floor.
    """

    feature: str
    psi: float
    status: str
    missing_rate: float
    reference_missing_rate: float
    unseen_rate: float
    raw_psi: float = 0.0


@dataclasses.dataclass(frozen=True)
class DriftReport:
    """Input and output drift of a batch.

    Attributes:
        rows: Rows assessed.
        features: Per-feature drift, worst first.
        output_psi: Noise-adjusted PSI of the model's outputs.
        output_status: stable / moderate / high.
        output_raw_psi: Output PSI before removing the noise floor.
    """

    rows: int
    features: list[FeatureDrift]
    output_psi: float
    output_status: str
    output_raw_psi: float = 0.0

    @property
    def high_share(self) -> float:
        """Share of features with large drift."""
        flagged = sum(1 for f in self.features if f.status == HIGH)
        return flagged / max(len(self.features), 1)


@dataclasses.dataclass(frozen=True)
class PerformanceReport:
    """Delayed-label performance.

    Attributes:
        rows: Rows assessed.
        labeled_rows: Rows whose outcome is known.
        labeled_fraction: Outcome coverage.
        metrics: Metrics on labeled rows (empty if too few labels).
        reference_metrics: The same metrics at training time.
        relative_change: Oriented relative change per metric (negative is
            worse).
        status: `insufficient_labels`, `ok` or `degraded`.
    """

    rows: int
    labeled_rows: int
    labeled_fraction: float
    metrics: dict[str, float]
    reference_metrics: dict[str, float]
    relative_change: dict[str, float]
    status: str


@dataclasses.dataclass(frozen=True)
class Recommendation:
    """What to do about the model.

    Attributes:
        action: `none`, `investigate` or `retrain`.
        reasons: Evidence behind the action.
    """

    action: str
    reasons: list[str]


@dataclasses.dataclass(frozen=True)
class MonitoringReport:
    """Everything monitoring concluded about a batch.

    Attributes:
        drift: Input and output drift.
        performance: Performance on labeled rows, if outcomes were given.
        recommendation: Advice derived from the evidence.
    """

    drift: DriftReport
    performance: PerformanceReport | None
    recommendation: Recommendation

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-serializable mapping."""
        return dataclasses.asdict(self)


class AdvisoryRule(abc.ABC):
    """One piece of evidence that can recommend action."""

    @abc.abstractmethod
    def evaluate(
        self,
        drift: DriftReport,
        performance: PerformanceReport | None,
        thresholds: MonitoringThresholds,
    ) -> tuple[str, str] | None:
        """Judges the evidence.

        Returns:
            `(action, reason)` when the rule fires, otherwise None.
        """


class PerformanceRule(AdvisoryRule):
    """Retrain when labeled performance degraded beyond tolerance."""

    def evaluate(
        self,
        drift: DriftReport,
        performance: PerformanceReport | None,
        thresholds: MonitoringThresholds,
    ) -> tuple[str, str] | None:
        """Fires on degraded performance."""
        if performance is not None and performance.status == "degraded":
            worst = min(
                performance.relative_change.items(), key=lambda kv: kv[1]
            )
            return (
                "retrain",
                f"{worst[0]} degraded {abs(worst[1]):.0%} vs training on "
                f"{performance.labeled_rows} labeled rows",
            )
        return None


class FeatureDriftRule(AdvisoryRule):
    """Retrain on widespread large drift; investigate localized drift."""

    def evaluate(
        self,
        drift: DriftReport,
        performance: PerformanceReport | None,
        thresholds: MonitoringThresholds,
    ) -> tuple[str, str] | None:
        """Fires on feature drift."""
        if drift.high_share >= thresholds.high_drift_feature_share:
            names = ", ".join(
                f"'{f.feature}' (PSI {f.psi:.2f})"
                for f in drift.features
                if f.status == HIGH
            )
            return (
                "retrain",
                f"{drift.high_share:.0%} of features drifted strongly: {names}",
            )
        worst = [f for f in drift.features if f.status != STABLE]
        if worst:
            top = worst[0]
            return (
                "investigate",
                f"feature '{top.feature}' drifted (PSI {top.psi:.2f})",
            )
        return None


class OutputDriftRule(AdvisoryRule):
    """Investigate shifts in the model's own output distribution."""

    def evaluate(
        self,
        drift: DriftReport,
        performance: PerformanceReport | None,
        thresholds: MonitoringThresholds,
    ) -> tuple[str, str] | None:
        """Fires on output drift."""
        if drift.output_status == HIGH:
            return (
                "retrain" if performance is None else "investigate",
                (f"model output drifted strongly (PSI {drift.output_psi:.2f})"),
            )
        if drift.output_status == MODERATE:
            return (
                "investigate",
                f"model output drifted (PSI {drift.output_psi:.2f})",
            )
        return None


class UnseenCategoryRule(AdvisoryRule):
    """Investigate when many categories were never seen in training."""

    def evaluate(
        self,
        drift: DriftReport,
        performance: PerformanceReport | None,
        thresholds: MonitoringThresholds,
    ) -> tuple[str, str] | None:
        """Fires on unseen categories."""
        for feature in drift.features:
            if feature.unseen_rate >= thresholds.unseen_category_rate:
                return (
                    "investigate",
                    f"{feature.unseen_rate:.0%} of '{feature.feature}' values "
                    "are categories unseen in training",
                )
        return None


DEFAULT_RULES: tuple[AdvisoryRule, ...] = (
    PerformanceRule(),
    FeatureDriftRule(),
    OutputDriftRule(),
    UnseenCategoryRule(),
)


def advise(
    drift: DriftReport,
    performance: PerformanceReport | None,
    thresholds: MonitoringThresholds,
    rules: tuple[AdvisoryRule, ...] = DEFAULT_RULES,
) -> Recommendation:
    """Combines the rules; the strongest action wins, all reasons are kept."""
    order = {"none": 0, "investigate": 1, "retrain": 2}
    fired = [
        verdict
        for rule in rules
        if (verdict := rule.evaluate(drift, performance, thresholds))
    ]
    if not fired:
        return Recommendation("none", [])
    action = max((a for a, r in fired), key=order.__getitem__)
    return Recommendation(action, [reason for a, reason in fired])


class ModelMonitor:
    """Compares production data and outcomes with a bundle's training state."""

    def __init__(
        self,
        bundle: bundle_module.ModelBundle,
        thresholds: MonitoringThresholds | None = None,
    ) -> None:
        """Initializes the monitor.

        Args:
            bundle: The serving model (must carry a reference profile).
            thresholds: Decision thresholds.

        Raises:
            ArtifactError: If the bundle has no reference profile.
        """
        if bundle.reference is None:
            raise errors.ArtifactError(
                "this bundle has no monitoring reference profile"
            )
        self.bundle = bundle
        self.reference: reference_module.ReferenceProfile = bundle.reference
        self.thresholds = thresholds or MonitoringThresholds()

    def model_output(self, frame: pd.DataFrame) -> pd.Series:
        """Computes the output the reference describes."""
        if self.bundle.schema.task == tasks.TaskType.BINARY.value:
            return pd.Series(self.bundle.predict_proba(frame)[:, 1])
        return pd.Series(self.bundle.predict(frame))

    def drift(self, data: pd.DataFrame) -> DriftReport:
        """Measures feature and output drift of a batch."""
        frame = self.bundle.prepare_frame(data)
        features = []
        for name, ref in self.reference.features.items():
            values = frame[name]
            psi = ref.adjusted_psi(values)
            features.append(
                FeatureDrift(
                    name,
                    psi,
                    self.thresholds.status(psi),
                    float(values.isna().mean()),
                    ref.missing_rate,
                    ref.unseen_rate(values),
                    ref.psi(values),
                )
            )
        features.sort(key=lambda f: -f.psi)
        output = self.model_output(frame)
        output_psi = self.reference.output.adjusted_psi(output)
        return DriftReport(
            len(frame),
            features,
            output_psi,
            self.thresholds.status(output_psi),
            self.reference.output.psi(output),
        )

    def performance(
        self, data: pd.DataFrame, outcomes: pd.Series
    ) -> PerformanceReport:
        """Scores the model on rows whose outcome is already known.

        Args:
            data: Served rows.
            outcomes: Observed outcomes aligned to `data`; missing values
                mean the label has not arrived yet.
        """
        frame = self.bundle.prepare_frame(data)
        known = outcomes.reset_index(drop=True).notna().to_numpy()
        labeled = int(known.sum())
        rows = len(frame)
        coverage = labeled / max(rows, 1)
        reference_metrics = dict(self.reference.reference_metrics)
        if labeled < self.thresholds.min_labeled_rows:
            return PerformanceReport(
                rows,
                labeled,
                coverage,
                {},
                reference_metrics,
                {},
                "insufficient_labels",
            )
        task = tasks.TaskType(self.bundle.schema.task)
        target = outcomes.reset_index(drop=True)[known].reset_index(drop=True)
        features = frame.reset_index(drop=True)[known].reset_index(drop=True)
        scores = metrics.CATALOG.evaluate(
            self.bundle.pipeline, features, target, task
        )
        change = {}
        for name, value in scores.items():
            expected = reference_metrics.get(name)
            if expected is None or np.isnan(value) or expected == 0:
                continue
            sign = 1 if metrics.CATALOG.metrics[name].greater_is_better else -1
            change[name] = sign * (value - expected) / abs(expected)
        primary = self.bundle.info.metric
        degraded = (
            primary in change
            and change[primary] < -self.thresholds.performance_tolerance
        )
        return PerformanceReport(
            rows,
            labeled,
            coverage,
            scores,
            reference_metrics,
            change,
            "degraded" if degraded else "ok",
        )

    def assess(
        self, data: pd.DataFrame, outcomes: pd.Series | None = None
    ) -> MonitoringReport:
        """Produces the full monitoring report for a batch.

        Args:
            data: Served rows.
            outcomes: Optional (possibly partial) observed outcomes.
        """
        drift = self.drift(data)
        performance = (
            self.performance(data, outcomes) if outcomes is not None else None
        )
        return MonitoringReport(
            drift,
            performance,
            advise(drift, performance, self.thresholds),
        )
