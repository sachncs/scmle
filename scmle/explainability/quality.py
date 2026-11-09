"""Deterministic quality checks for explanation evidence."""

from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
import math
from typing import Any

MIN_REPORTS = 2


def validate_score(value: float, field_name: str) -> float:
    """Returns a finite score in the inclusive unit interval.

    Args:
        value: The candidate score.
        field_name: Name used in the error message.

    Returns:
        `value` as a float.

    Raises:
        TypeError: If `value` is a bool or not an int or float.
        ValueError: If `value` is not finite or lies outside [0, 1].
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be numeric")
    score = float(value)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"{field_name} must be finite and between 0 and 1")
    return score


@dataclasses.dataclass(frozen=True, slots=True)
class QualityThresholds:
    """Declared release thresholds for explanation quality measures.

    Attributes:
        stability_minimum: Smallest acceptable stability score in [0, 1], or
            None when stability is not required.
        fidelity_minimum: Smallest acceptable fidelity score in [0, 1], or
            None when fidelity is not required.
    """

    stability_minimum: float | None = None
    fidelity_minimum: float | None = None

    def __post_init__(self) -> None:
        """Validates every declared threshold.

        A declared threshold that is not a number is rejected by
        `validate_score` with a `TypeError`.

        Raises:
            ValueError: If neither threshold is declared, or a declared
                threshold is not finite or lies outside [0, 1].
        """
        if self.stability_minimum is None and self.fidelity_minimum is None:
            raise ValueError("at least one explanation threshold is required")
        if self.stability_minimum is not None:
            validate_score(self.stability_minimum, "stability_minimum")
        if self.fidelity_minimum is not None:
            validate_score(self.fidelity_minimum, "fidelity_minimum")

    def to_dict(self) -> dict[str, float | None]:
        """Returns JSON-compatible threshold declarations.

        Returns:
            A mapping with the keys `stability_minimum` and
            `fidelity_minimum`, each a float or None.
        """
        return {
            "stability_minimum": self.stability_minimum,
            "fidelity_minimum": self.fidelity_minimum,
        }


@dataclasses.dataclass(frozen=True, slots=True)
class QualityEvaluation:
    """Reviewable result of explanation stability and fidelity checks.

    Attributes:
        method: Name of the explanation method that was checked.
        sample_count: Number of samples the checks covered.
        stability_score: Measured stability in [0, 1], or None if not
            measured.
        fidelity_score: Measured fidelity in [0, 1], or None if not
            measured.
        thresholds: The declared thresholds the scores were compared with.
        passed: Whether every declared threshold was met.
        findings: Human-readable statements explaining the outcome.
    """

    method: str
    sample_count: int
    stability_score: float | None
    fidelity_score: float | None
    thresholds: QualityThresholds
    passed: bool
    findings: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validates the evaluation record before it is persisted.

        Raises:
            ValueError: If `method` is blank, `sample_count` is below 1, or
                a present score is not finite or lies outside [0, 1].
            TypeError: If a present score is not numeric or `findings` is
                not a tuple.
        """
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
        """Returns a deterministic JSON-compatible evaluation.

        Returns:
            A mapping of every field, with thresholds as a mapping and
            findings as a list.
        """
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

    Attributes:
        method: Name of the explanation method the reference applies to.
        reference: Description of where the reference ranking comes from.
        expected_features: Feature names that the explanation is expected to
            rank at the top; unique and non-empty.
        minimum_overlap: Smallest acceptable fraction of the expected
            features found in the observed top ranking, in [0, 1].
        limitations: Stated limits of the reference.
    """

    method: str
    reference: str
    expected_features: tuple[str, ...]
    minimum_overlap: float = 1.0
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validates the independent reference declaration.

        A `minimum_overlap` that is not a number is rejected by
        `validate_score` with a `TypeError`.

        Raises:
            ValueError: If `method`, `reference`, or any expected feature or
                limitation is blank, if no expected features are given or
                they repeat, or if `minimum_overlap` is not finite or lies
                outside [0, 1].
        """
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
        """Returns the reference declaration as JSON-native evidence.

        Returns:
            A mapping of every field, with sequences as lists.
        """
        return {
            "method": self.method,
            "reference": self.reference,
            "expected_features": list(self.expected_features),
            "minimum_overlap": self.minimum_overlap,
            "limitations": list(self.limitations),
        }


@dataclasses.dataclass(frozen=True, slots=True)
class GroundTruthEvaluation:
    """Reviewable comparison between an explanation and its reference.

    Attributes:
        sample_id: Identity of the explained sample.
        method: Name of the explanation method.
        reference: Description of the reference ranking's source.
        expected_features: Feature names expected at the top.
        observed_features: Top features the explanation actually ranked, as
            many as there are expected features.
        overlap_score: Fraction of the expected features that were observed,
            in [0, 1].
        minimum_overlap: Smallest overlap that passes, in [0, 1].
        passed: Whether `overlap_score` reached `minimum_overlap`.
        findings: Human-readable statements explaining the outcome.
        limitations: Stated limits of the reference.
    """

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
        """Validates the comparison before it is persisted.

        A score that is not a number is rejected by `validate_score` with
        a `TypeError`.

        Raises:
            ValueError: If `sample_id`, `method` or `reference` is blank, or
                a score is not finite or lies outside [0, 1].
            TypeError: If `findings` is not a tuple.
        """
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
        """Returns the comparison as deterministic JSON-native evidence.

        Returns:
            A mapping of every field plus `status` set to "available", with
            sequences as lists.
        """
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
    """Bounded suite result for independent explanation references.

    Attributes:
        sample_count: Number of samples in the held-out suite.
        reference_count: Number of samples that have a reference; between 0
            and `sample_count`.
        evaluations: One comparison per referenced sample.
        passed: Whether every sample has a reference and every comparison
            passed.
        findings: Human-readable statements explaining the outcome.
    """

    sample_count: int
    reference_count: int
    evaluations: tuple[GroundTruthEvaluation, ...]
    passed: bool
    findings: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validates suite cardinality and evaluation consistency.

        Raises:
            ValueError: If `sample_count` is below 1, `reference_count` is
                negative or exceeds `sample_count`, or it differs from the
                number of evaluations.
            TypeError: If `findings` is not a tuple.
        """
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
        """The suite status.

        It is "not_available" when no sample has a reference, "partial" when
        only some do, and "available" when all do.
        """
        if self.reference_count == 0:
            return "not_available"
        if self.reference_count < self.sample_count:
            return "partial"
        return "available"

    def to_dict(self) -> dict[str, object]:
        """Returns deterministic suite evidence.

        Returns:
            A mapping with the status, counts, the smallest and mean overlap
            score (None when there are no evaluations), the pass flag, the
            findings and the per-sample evaluations as mappings.
        """
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
    """Extracts a validated feature ranking from one explanation report.

    Args:
        report: An explanation report whose `features` entry is a list of
            mappings, each with a string `feature` name, in rank order.

    Returns:
        The stripped feature names in the order of the report.

    Raises:
        ValueError: If `features` is missing, not a list or empty, an item
            is not a mapping with a string `feature`, or a name is blank or
            repeated.
    """
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
    """Compares an explanation ranking with an independent reference ranking.

    The score is the fraction of expected features found in the observed
    top-k ranking, where k is the number of expected features. It is a
    transparent agreement measure, not a causal or semantic validity claim.

    Args:
        ground_truth: The independent reference ranking.
        report: The explanation report to check, in the form read by
            `explanation_ranking`.
        sample_id: Identity of the explained sample.

    Returns:
        The comparison, with the observed top features, the overlap score
        and whether it reached the reference's minimum overlap.

    An invalid report is rejected by `explanation_ranking` with a
    `ValueError`, and a blank `sample_id` by the result record with a
    `ValueError`.

    Raises:
        TypeError: If `ground_truth` is not an `ExplanationGroundTruth`.
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
    """Evaluates independent references across a bounded held-out suite.

    A partial reference set fails closed. This prevents a caller from
    receiving a green suite result after annotating only convenient samples.

    Args:
        sample_count: Number of samples in the held-out suite.
        references: One (reference, explanation report, sample id) triple
            for each sample that has a reference.

    Returns:
        The suite result. It passes only when every sample has a reference
        and every comparison passed.

    A reference that is not an `ExplanationGroundTruth` is rejected with a
    `TypeError`, and an invalid report or blank sample id with a
    `ValueError`, by `evaluate_ground_truth`.

    Raises:
        ValueError: If `sample_count` is below 1 or there are more
            references than samples.
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
    """Builds a deterministic local ranking from linear contributions.

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
        A mapping with the `sample_id`, `features` (the `top_k` largest
        absolute contributions, as `feature` and signed `importance`) and
        `terms` (the same pairs as lists), strongest first, ties broken by
        name.

    Raises:
        ValueError: If `sample_id` is blank, `top_k` is outside 1 to the
            number of features, the three sequences differ in length, names
            are blank or repeated, or a contribution is not finite.
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
    """Returns mean pairwise top-k overlap across repeated explanations.

    Top-k overlap is intentionally transparent: a score of ``1`` means every
    repeated explanation selected the same features, while ``0`` means no
    selected features overlapped. It does not claim causal validity.

    Args:
        reports: Two or more explanation reports in the form read by
            `explanation_ranking`.
        top_k: Number of leading features compared. When None or 0, the
            smallest feature count among the reports is used.

    Returns:
        The mean Jaccard overlap of the top-k feature sets over all pairs of
        reports, in [0, 1].

    Raises:
        ValueError: If fewer than two reports are given, a report is not a
            valid explanation report, `top_k` is negative, or `top_k`
            exceeds the feature count of any report.
    """
    if len(reports) < MIN_REPORTS:
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
    """Returns normalized reconstruction fidelity against reference outputs.

    The score is ``1 - RMSE / baseline_RMSE``, where the baseline predicts the
    mean reference output. A perfect reconstruction scores ``1``; a model
    worse than that baseline scores ``0``. This is a predictive agreement
    measure, not evidence that a feature causes the prediction.

    Args:
        reference: Reference outputs; must not be empty.
        reconstruction: Reconstructed outputs, one per reference output.

    Returns:
        The score clipped to [0, 1]. When the reference is constant, 1.0 for
        an exact reconstruction and 0.0 otherwise.

    Raises:
        ValueError: If the sequences are empty or differ in length, or any
            value is not finite.
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
    """Evaluates declared explanation thresholds without inventing measures.

    A declared threshold whose measure is missing fails.

    Args:
        method: Name of the explanation method.
        sample_count: Number of samples the measures cover.
        thresholds: The declared thresholds.
        stability: Measured stability in [0, 1], or None if not measured.
        fidelity: Measured fidelity in [0, 1], or None if not measured.

    Returns:
        The evaluation, with the findings that explain each failure or the
        statement that all thresholds passed.

    A measure that is not a number is rejected by `validate_score` with a
    `TypeError`. A measure that is not finite or lies outside [0, 1], a
    blank `method` and a `sample_count` below 1 are rejected with a
    `ValueError`.
    """
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
