"""Deterministic quality checks for explanation evidence."""

from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
import math
from typing import Any


def validate_score(value: float, field_name: str) -> float:
    """Return a finite score in the inclusive unit interval."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be numeric")
    score = float(value)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"{field_name} must be finite and between 0 and 1")
    return score


@dataclasses.dataclass(frozen=True, slots=True)
class QualityThresholds:
    """Declared release thresholds for explanation quality measures."""

    stability_minimum: float | None = None
    fidelity_minimum: float | None = None

    def __post_init__(self) -> None:
        """Validate every declared threshold."""
        if self.stability_minimum is None and self.fidelity_minimum is None:
            raise ValueError("at least one explanation threshold is required")
        if self.stability_minimum is not None:
            validate_score(self.stability_minimum, "stability_minimum")
        if self.fidelity_minimum is not None:
            validate_score(self.fidelity_minimum, "fidelity_minimum")

    def to_dict(self) -> dict[str, float | None]:
        """Return JSON-compatible threshold declarations."""
        return {
            "stability_minimum": self.stability_minimum,
            "fidelity_minimum": self.fidelity_minimum,
        }


@dataclasses.dataclass(frozen=True, slots=True)
class QualityEvaluation:
    """Reviewable result of explanation stability and fidelity checks."""

    method: str
    sample_count: int
    stability_score: float | None
    fidelity_score: float | None
    thresholds: QualityThresholds
    passed: bool
    findings: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate the evaluation record before it is persisted."""
        if not self.method.strip():
            raise ValueError("method must not be empty")
        if self.sample_count < 1:
            raise ValueError("sample_count must be at least 1")
        if self.stability_score is not None:
            validate_score(self.stability_score, "stability_score")
        if self.fidelity_score is not None:
            validate_score(self.fidelity_score, "fidelity_score")
        if not isinstance(self.findings, tuple):
            raise TypeError("findings must be a tuple")

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible evaluation."""
        return {
            "method": self.method,
            "sample_count": self.sample_count,
            "stability_score": self.stability_score,
            "fidelity_score": self.fidelity_score,
            "thresholds": self.thresholds.to_dict(),
            "passed": self.passed,
            "findings": list(self.findings),
        }


def explanation_ranking(report: Mapping[str, Any]) -> tuple[str, ...]:
    """Extract a validated feature ranking from one explanation report."""
    features = report.get("features")
    if not isinstance(features, list) or not features:
        raise ValueError("explanation report needs a non-empty features list")
    ranking: list[str] = []
    for item in features:
        if not isinstance(item, Mapping) or not isinstance(
            item.get("feature"), str
        ):
            raise ValueError("explanation features need string feature names")
        feature = item["feature"].strip()
        if not feature or feature in ranking:
            raise ValueError("explanation feature names must be unique and set")
        ranking.append(feature)
    return tuple(ranking)


def ranking_stability(
    reports: Sequence[Mapping[str, Any]], top_k: int | None = None
) -> float:
    """Return mean pairwise top-k overlap across repeated explanations.

    Top-k overlap is intentionally transparent: a score of ``1`` means every
    repeated explanation selected the same features, while ``0`` means no
    selected features overlapped. It does not claim causal validity.
    """
    if len(reports) < 2:
        raise ValueError("at least two explanation reports are required")
    rankings = [explanation_ranking(report) for report in reports]
    limit = top_k or min(len(ranking) for ranking in rankings)
    if limit < 1:
        raise ValueError("top_k must be at least 1")
    if any(limit > len(ranking) for ranking in rankings):
        raise ValueError("top_k cannot exceed a report's feature count")
    selected = [set(ranking[:limit]) for ranking in rankings]
    overlaps: list[float] = []
    for left in range(len(selected)):
        for right in range(left + 1, len(selected)):
            union = selected[left] | selected[right]
            overlaps.append(len(selected[left] & selected[right]) / len(union))
    return sum(overlaps) / len(overlaps)


def fidelity_score(
    reference: Sequence[float], reconstruction: Sequence[float]
) -> float:
    """Return normalized reconstruction fidelity against reference outputs.

    The score is ``1 - RMSE / baseline_RMSE``, where the baseline predicts the
    mean reference output. A perfect reconstruction scores ``1``; a model
    worse than that baseline scores ``0``. This is a predictive agreement
    measure, not evidence that a feature causes the prediction.
    """
    if len(reference) == 0 or len(reference) != len(reconstruction):
        raise ValueError("reference and reconstruction must have equal values")
    reference_values = [float(value) for value in reference]
    reconstruction_values = [float(value) for value in reconstruction]
    if any(
        not math.isfinite(value)
        for value in (*reference_values, *reconstruction_values)
    ):
        raise ValueError("fidelity values must be finite")
    mean = sum(reference_values) / len(reference_values)
    baseline_error = math.sqrt(
        sum((value - mean) ** 2 for value in reference_values)
        / len(reference_values)
    )
    reconstruction_error = math.sqrt(
        sum(
            (actual - predicted) ** 2
            for actual, predicted in zip(
                reference_values, reconstruction_values, strict=True
            )
        )
        / len(reference_values)
    )
    if baseline_error == 0.0:
        return 1.0 if reconstruction_error == 0.0 else 0.0
    return max(0.0, min(1.0, 1.0 - reconstruction_error / baseline_error))


def evaluate(
    method: str,
    sample_count: int,
    thresholds: QualityThresholds,
    *,
    stability: float | None = None,
    fidelity: float | None = None,
) -> QualityEvaluation:
    """Evaluate declared explanation thresholds without inventing measures."""
    if stability is not None:
        validate_score(stability, "stability")
    if fidelity is not None:
        validate_score(fidelity, "fidelity")
    findings: list[str] = []
    passed = True
    if thresholds.stability_minimum is not None:
        if stability is None:
            passed = False
            findings.append("stability evidence is missing")
        elif stability < thresholds.stability_minimum:
            passed = False
            findings.append("stability is below its declared threshold")
    if thresholds.fidelity_minimum is not None:
        if fidelity is None:
            passed = False
            findings.append("fidelity evidence is missing")
        elif fidelity < thresholds.fidelity_minimum:
            passed = False
            findings.append("fidelity is below its declared threshold")
    if passed:
        findings.append("all declared explanation thresholds passed")
    return QualityEvaluation(
        method=method,
        sample_count=sample_count,
        stability_score=stability,
        fidelity_score=fidelity,
        thresholds=thresholds,
        passed=passed,
        findings=tuple(findings),
    )
