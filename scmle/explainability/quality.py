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


@dataclasses.dataclass(frozen=True, slots=True)
class ExplanationGroundTruth:
    """Independent reference ranking for one explanation contract.

    The reference is caller-owned evidence. This type deliberately stores a
    ranking of declared feature names rather than pretending that a feature
    ranking proves raw-modality causality.
    """

    method: str
    reference: str
    expected_features: tuple[str, ...]
    minimum_overlap: float = 1.0
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate the independent reference declaration."""
        if not self.method.strip():
            raise ValueError("method must not be empty")
        if not self.reference.strip():
            raise ValueError("reference must not be empty")
        if not self.expected_features:
            raise ValueError("expected_features must not be empty")
        if len(set(self.expected_features)) != len(self.expected_features):
            raise ValueError("expected_features must be unique")
        if any(not feature.strip() for feature in self.expected_features):
            raise ValueError("expected_features must not contain empty values")
        validate_score(self.minimum_overlap, "minimum_overlap")
        if any(not limitation.strip() for limitation in self.limitations):
            raise ValueError("limitations must not contain empty values")

    def to_dict(self) -> dict[str, object]:
        """Return the reference declaration as JSON-native evidence."""
        return {
            "method": self.method,
            "reference": self.reference,
            "expected_features": list(self.expected_features),
            "minimum_overlap": self.minimum_overlap,
            "limitations": list(self.limitations),
        }


@dataclasses.dataclass(frozen=True, slots=True)
class GroundTruthEvaluation:
    """Reviewable comparison between an explanation and its reference."""

    sample_id: str
    method: str
    reference: str
    expected_features: tuple[str, ...]
    observed_features: tuple[str, ...]
    overlap_score: float
    minimum_overlap: float
    passed: bool
    findings: tuple[str, ...]
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate the comparison before it is persisted."""
        if not self.sample_id.strip():
            raise ValueError("sample_id must not be empty")
        if not self.method.strip():
            raise ValueError("method must not be empty")
        if not self.reference.strip():
            raise ValueError("reference must not be empty")
        validate_score(self.overlap_score, "overlap_score")
        validate_score(self.minimum_overlap, "minimum_overlap")
        if not isinstance(self.findings, tuple):
            raise TypeError("findings must be a tuple")

    def to_dict(self) -> dict[str, object]:
        """Return the comparison as deterministic JSON-native evidence."""
        return {
            "status": "available",
            "sample_id": self.sample_id,
            "method": self.method,
            "reference": self.reference,
            "expected_features": list(self.expected_features),
            "observed_features": list(self.observed_features),
            "overlap_score": self.overlap_score,
            "minimum_overlap": self.minimum_overlap,
            "passed": self.passed,
            "findings": list(self.findings),
            "limitations": list(self.limitations),
        }


@dataclasses.dataclass(frozen=True, slots=True)
class GroundTruthSuiteEvaluation:
    """Bounded suite result for independent explanation references."""

    sample_count: int
    reference_count: int
    evaluations: tuple[GroundTruthEvaluation, ...]
    passed: bool
    findings: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate suite cardinality and evaluation consistency."""
        if self.sample_count < 1:
            raise ValueError("sample_count must be at least 1")
        if self.reference_count < 0 or self.reference_count > self.sample_count:
            raise ValueError("reference_count must be within sample_count")
        if self.reference_count != len(self.evaluations):
            raise ValueError("reference_count must match evaluations")
        if not isinstance(self.findings, tuple):
            raise TypeError("findings must be a tuple")

    @property
    def status(self) -> str:
        """Return whether the suite is unavailable, partial, or complete."""
        if self.reference_count == 0:
            return "not_available"
        if self.reference_count < self.sample_count:
            return "partial"
        return "available"

    def to_dict(self) -> dict[str, object]:
        """Return deterministic suite evidence."""
        overlaps = tuple(
            evaluation.overlap_score for evaluation in self.evaluations
        )
        return {
            "status": self.status,
            "sample_count": self.sample_count,
            "reference_count": self.reference_count,
            "minimum_overlap_score": min(overlaps) if overlaps else None,
            "mean_overlap_score": (
                sum(overlaps) / len(overlaps) if overlaps else None
            ),
            "passed": self.passed,
            "findings": list(self.findings),
            "evaluations": [
                evaluation.to_dict() for evaluation in self.evaluations
            ],
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


def evaluate_ground_truth(
    ground_truth: ExplanationGroundTruth,
    report: Mapping[str, Any],
    sample_id: str,
) -> GroundTruthEvaluation:
    """Compare an explanation ranking with an independent reference ranking.

    The score is the fraction of expected features found in the observed
    top-k ranking, where k is the number of expected features. It is a
    transparent agreement measure, not a causal or semantic validity claim.
    """
    if not isinstance(ground_truth, ExplanationGroundTruth):
        raise TypeError("ground_truth must be ExplanationGroundTruth")
    ranking = explanation_ranking(report)
    observed = ranking[: len(ground_truth.expected_features)]
    expected = set(ground_truth.expected_features)
    overlap = len(expected & set(observed)) / len(expected)
    passed = overlap >= ground_truth.minimum_overlap
    findings = (
        ("independent explanation reference passed",)
        if passed
        else ("independent explanation reference overlap is below threshold",)
    )
    return GroundTruthEvaluation(
        sample_id=sample_id,
        method=ground_truth.method,
        reference=ground_truth.reference,
        expected_features=ground_truth.expected_features,
        observed_features=observed,
        overlap_score=overlap,
        minimum_overlap=ground_truth.minimum_overlap,
        passed=passed,
        findings=findings,
        limitations=ground_truth.limitations,
    )


def evaluate_ground_truth_suite(
    sample_count: int,
    references: Sequence[tuple[ExplanationGroundTruth, Mapping[str, Any], str]],
) -> GroundTruthSuiteEvaluation:
    """Evaluate independent references across a bounded held-out suite.

    A partial reference set fails closed. This prevents a caller from
    receiving a green suite result after annotating only convenient samples.
    """
    if sample_count < 1:
        raise ValueError("sample_count must be at least 1")
    if len(references) > sample_count:
        raise ValueError("references cannot exceed sample_count")
    evaluations = tuple(
        evaluate_ground_truth(ground_truth, report, sample_id)
        for ground_truth, report, sample_id in references
    )
    if not evaluations:
        findings = ("independent explanation references are missing",)
        passed = False
    elif len(evaluations) < sample_count:
        findings = ("independent explanation references are incomplete",)
        passed = False
    elif all(evaluation.passed for evaluation in evaluations):
        findings = ("all independent explanation references passed",)
        passed = True
    else:
        findings = ("one or more independent explanation references failed",)
        passed = False
    return GroundTruthSuiteEvaluation(
        sample_count=sample_count,
        reference_count=len(evaluations),
        evaluations=evaluations,
        passed=passed,
        findings=findings,
    )


def linear_contribution_report(
    sample_id: str,
    feature_names: Sequence[str],
    values: Sequence[float],
    coefficients: Sequence[float],
    top_k: int = 10,
) -> dict[str, object]:
    """Build a deterministic local ranking from linear contributions.

    The returned ranking is an explanation of the supplied feature
    representation. It deliberately does not infer meaning for the features
    or make a causal claim about the raw input.

    Args:
        sample_id: Stable identity of the explained sample.
        feature_names: Feature names in matrix-column order.
        values: One feature row in the same order as ``feature_names``.
        coefficients: Model coefficients for the explained class.
        top_k: Maximum number of ranked terms to return.

    Returns:
        A JSON-native local explanation report.

    Raises:
        ValueError: If identities, values, or bounds are inconsistent.
    """
    if not sample_id.strip():
        raise ValueError("sample_id must not be empty")
    if not 1 <= top_k <= len(feature_names):
        raise ValueError("top_k must be within the feature-name count")
    if len(feature_names) != len(values) or len(values) != len(coefficients):
        raise ValueError("feature names, values, and coefficients must align")
    names = tuple(feature_names)
    if len(set(names)) != len(names) or any(not name.strip() for name in names):
        raise ValueError("feature names must be unique and non-empty")
    contributions = []
    for name, value, coefficient in zip(
        names, values, coefficients, strict=True
    ):
        contribution = float(value) * float(coefficient)
        if not math.isfinite(contribution):
            raise ValueError("linear contributions must be finite")
        contributions.append((name, contribution))
    ranked = sorted(
        contributions,
        key=lambda item: (-abs(item[1]), item[0]),
    )[:top_k]
    return {
        "sample_id": sample_id,
        "features": [
            {"feature": name, "importance": contribution}
            for name, contribution in ranked
        ],
        "terms": [[name, contribution] for name, contribution in ranked],
    }


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
