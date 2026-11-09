"""Focused tests for the bounded longitudinal causal contract."""

import json

import numpy
import pandas
import pytest

from scmle.evaluation.longitudinal import LongitudinalEstimand
from scmle.evaluation.longitudinal import LongitudinalEstimationPolicy
from scmle.evaluation.longitudinal import LongitudinalEstimationResult
from scmle.evaluation.longitudinal import LongitudinalIPWEstimator


def make_estimand() -> LongitudinalEstimand:
    """Return the fully identified longitudinal declaration."""
    return LongitudinalEstimand(
        name="period_specific_treatment_effect",
        subject_identity="subject_id",
        time_column="time",
        treatment_column="treatment",
        outcome="outcome",
        covariates=("baseline_score",),
        population="eligible subjects observed at each declared period",
        assumptions=(
            "consistency",
            "positivity",
            "conditional_exchangeability",
            "no_unmeasured_time_varying_confounding",
            "no_time_varying_covariates",
        ),
    )


def make_frame() -> pandas.DataFrame:
    """Return a deterministic repeated-subject long-format fixture."""
    rng = numpy.random.default_rng(17)
    rows = []
    for subject_id in range(60):
        baseline_score = float(rng.normal())
        for time in range(3):
            probability = 1.0 / (
                1.0 + numpy.exp(-(-0.15 + 0.6 * baseline_score + 0.15 * time))
            )
            treatment = int(rng.random() < probability)
            outcome = (
                1.0
                + 1.5 * treatment
                + 0.4 * baseline_score
                + 0.2 * time
                + rng.normal(scale=0.5)
            )
            rows.append(
                {
                    "subject_id": subject_id,
                    "time": time,
                    "treatment": treatment,
                    "outcome": outcome,
                    "baseline_score": baseline_score,
                }
            )
    return pandas.DataFrame(rows)


def test_period_specific_ipw_is_deterministic_and_auditable() -> None:
    """The result contains period diagnostics, covariance, and reference evidence."""
    estimator = LongitudinalIPWEstimator(
        make_estimand(), LongitudinalEstimationPolicy(seed=23)
    )
    first = estimator.fit(make_frame())
    second = estimator.fit(make_frame())

    assert first.to_dict() == second.to_dict()
    assert len(first.periods) == 3
    assert [period.period for period in first.periods] == ["0", "1", "2"]
    assert numpy.asarray(first.covariance).shape == (3, 3)
    assert first.reference["passed"] is True
    assert first.diagnostics["method"] == (
        "period_specific_covariate_adjusted_hajek_ipw"
    )
    assert "not a full sequential" in first.limitations["longitudinal_scope"]
    json.dumps(first.to_dict())


def test_report_round_trip_is_digest_bound_and_atomic(tmp_path) -> None:
    """Longitudinal reports round-trip and reject content tampering."""
    result = LongitudinalIPWEstimator(make_estimand()).fit(make_frame())
    destination = tmp_path / "longitudinal.json"

    digest = result.write(destination)
    restored = LongitudinalEstimationResult.read(destination)

    assert digest == result.digest() == restored.digest()
    assert restored.to_dict() == result.to_dict()
    envelope = json.loads(destination.read_text(encoding="utf-8"))
    envelope["report"]["rows_used"] += 1
    destination.write_text(json.dumps(envelope), encoding="utf-8")
    with pytest.raises(ValueError, match="digest mismatch"):
        LongitudinalEstimationResult.read(destination)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("missing_subject", "missing subject"),
        ("duplicate_time", "duplicate subject-time"),
        ("non_binary", "binary"),
        ("invalid_order", "strictly increasing"),
        ("non_finite", "non-finite"),
        ("insufficient_support", "treatment support"),
    ],
)
def test_invalid_longitudinal_inputs_are_rejected(
    change: str, message: str
) -> None:
    """The input contract rejects malformed identities, support, and values."""
    frame = make_frame()
    if change == "missing_subject":
        frame.loc[0, "subject_id"] = numpy.nan
    elif change == "duplicate_time":
        frame.loc[1, "time"] = 0
    elif change == "non_binary":
        frame.loc[0, "treatment"] = 2
    elif change == "invalid_order":
        frame.loc[1, "time"] = -1
    elif change == "non_finite":
        frame.loc[0, "outcome"] = numpy.inf
    elif change == "insufficient_support":
        frame["treatment"] = 0
    else:
        raise AssertionError(f"unhandled change {change}")

    with pytest.raises(ValueError, match=message):
        LongitudinalIPWEstimator(make_estimand()).fit(frame)


def test_complete_case_policy_records_dropped_rows() -> None:
    """Complete-case deletion is explicit in row accounting and limitations."""
    frame = make_frame()
    frame.loc[0, "outcome"] = numpy.nan
    policy = LongitudinalEstimationPolicy(missingness="complete_case")

    result = LongitudinalIPWEstimator(make_estimand(), policy).fit(frame)

    assert result.rows_dropped == 1
    assert result.rows_used == result.rows_input - 1
    assert result.limitations["missingness"]["policy"] == "complete_case"


def test_required_assumptions_are_not_implicit() -> None:
    """Omitting any identification assumption blocks estimation declaration."""
    with pytest.raises(ValueError, match="required assumptions"):
        LongitudinalEstimand(
            name="period_specific_treatment_effect",
            subject_identity="subject_id",
            time_column="time",
            treatment_column="treatment",
            outcome="outcome",
            covariates=(),
            population="population",
            assumptions=("consistency", "positivity"),
        )
