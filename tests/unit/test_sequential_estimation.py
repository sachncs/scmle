"""Focused tests for the bounded sequential causal contract."""

import dataclasses
import json

import numpy
import pandas
import pytest

from scmle.evaluation.sequential import Config
from scmle.evaluation.sequential import Estimand
from scmle.evaluation.sequential import Estimation
from scmle.evaluation.sequential import Estimator
from scmle.evaluation.sequential import PeriodDiagnostics
from scmle.evaluation.sequential import RegimeEstimate


def make_estimand() -> Estimand:
    """Return an identified sustained-treatment declaration."""
    return Estimand(
        name="sustained_treatment_effect",
        subject_identity="subject_id",
        time_column="time",
        treatment_column="treatment",
        outcome="outcome",
        baseline_covariates=("baseline_score",),
        time_varying_covariates=("state",),
        population="subjects with complete follow-up across declared periods",
        assumptions=(
            "consistency",
            "positivity",
            "sequential_exchangeability",
            "no_unmeasured_time_varying_confounding",
            "time_varying_covariates_are_pre_treatment",
            "complete_follow_up",
        ),
    )


def make_frame() -> pandas.DataFrame:
    """Return a deterministic panel with treatment-confounded state."""
    rng = numpy.random.default_rng(31)
    rows = []
    for subject_id in range(180):
        baseline_score = float(rng.normal(scale=0.8))
        previous_treatment = 0
        for time in range(3):
            state = float(
                0.35 * baseline_score
                + 0.45 * previous_treatment
                + rng.normal(scale=0.35)
            )
            logit = (
                -0.1
                + 0.35 * baseline_score
                + 0.3 * state
                + 0.25 * previous_treatment
            )
            probability = 1.0 / (1.0 + numpy.exp(-logit))
            treatment = int(rng.random() < probability)
            outcome = (
                1.0
                + 1.4 * treatment
                + 0.3 * state
                + 0.2 * baseline_score
                + 0.25 * time
                + rng.normal(scale=0.45)
            )
            rows.append(
                {
                    "subject_id": subject_id,
                    "time": time,
                    "treatment": treatment,
                    "outcome": outcome,
                    "baseline_score": baseline_score,
                    "state": state,
                }
            )
            previous_treatment = treatment
    return pandas.DataFrame(rows)


def make_policy() -> Config:
    """Return a small but auditable policy for the fixture."""
    return Config(
        seed=17,
        minimum_subjects=100,
        minimum_arm_rows=2,
        minimum_effective_sample_size=2.0,
        maximum_weight=12.0,
    )


def test_sequential_ipw_is_deterministic_and_auditable() -> None:
    """Sequential weights produce stable regime and period evidence."""
    estimator = Estimator(make_estimand(), make_policy())
    first = estimator.fit(make_frame())
    second = estimator.fit(make_frame())

    assert first.to_dict() == second.to_dict()
    assert len(first.periods) == 3
    assert [item.period for item in first.periods] == ["0", "1", "2"]
    assert {item.regime for item in first.regimes} == {
        "always_treated",
        "always_control",
    }
    assert first.reference["passed"] is True
    assert first.diagnostics["method"] == "stabilized_treatment_history_ipw"
    assert first.diagnostics["weight_truncation"]["maximum_weight"] == 12.0
    assert "Dynamic regimes" in first.limitations["unsupported"]
    json.dumps(first.to_dict())


def test_sequential_report_round_trip_is_digest_bound(tmp_path) -> None:
    """Sequential reports round-trip and reject content tampering."""
    result = Estimator(make_estimand(), make_policy()).fit(make_frame())
    destination = tmp_path / "sequential.json"

    digest = result.write(destination)
    restored = Estimation.read(destination)

    assert digest == result.digest() == restored.digest()
    assert restored.to_dict() == result.to_dict()
    envelope = json.loads(destination.read_text(encoding="utf-8"))
    envelope["report"]["rows_used"] += 1
    destination.write_text(json.dumps(envelope), encoding="utf-8")
    with pytest.raises(ValueError, match="digest mismatch"):
        Estimation.read(destination)


def test_unsorted_rows_are_ordered_by_subject_and_period() -> None:
    """Long-format ingestion does not depend on source row ordering."""
    frame = make_frame()
    baseline = Estimator(make_estimand(), make_policy()).fit(frame)
    shuffled = frame.sample(frac=1.0, random_state=9).reset_index(drop=True)
    result = Estimator(make_estimand(), make_policy()).fit(shuffled)

    assert result.rows_used == len(frame)
    assert result.reference["passed"] is True
    assert result.contrast_estimate == baseline.contrast_estimate
    assert result.periods == baseline.periods


def test_complete_case_drops_subjects_as_whole_sequences() -> None:
    """Missingness cannot silently create intermittent follow-up."""
    frame = make_frame()
    frame.loc[frame["subject_id"] == 0, "state"] = numpy.nan
    policy = dataclasses.replace(make_policy(), missingness="complete_case")

    result = Estimator(make_estimand(), policy).fit(frame)

    assert result.subjects_dropped == 1
    assert result.rows_dropped == 3
    assert (
        result.subjects_used + result.subjects_dropped == result.subjects_input
    )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("missing_column", "missing columns"),
        ("duplicate", "duplicate subject-time"),
        ("non_binary", "binary"),
        ("incomplete", "one row at every period"),
        ("baseline_changes", "baseline covariate"),
        ("one_period", "at least two periods"),
    ],
)
def test_invalid_sequential_panels_are_rejected(
    change: str, message: str
) -> None:
    """The panel contract rejects malformed identities and follow-up."""
    frame = make_frame()
    if change == "missing_column":
        frame = frame.drop(columns=["state"])
    elif change == "duplicate":
        frame.loc[1, "time"] = frame.loc[0, "time"]
    elif change == "non_binary":
        frame.loc[0, "treatment"] = 2
    elif change == "incomplete":
        frame = frame.loc[frame["subject_id"] != 0].copy()
        frame = frame.loc[~((frame["subject_id"] == 1) & (frame["time"] == 2))]
    elif change == "baseline_changes":
        frame.loc[
            (frame["subject_id"] == 1) & (frame["time"] == 1),
            "baseline_score",
        ] += 1.0
    elif change == "one_period":
        frame = frame.loc[frame["time"] == 0].copy()
    else:
        raise AssertionError(f"unhandled change {change}")

    with pytest.raises(ValueError, match=message):
        Estimator(make_estimand(), make_policy()).fit(frame)


def test_required_assumptions_are_not_implicit() -> None:
    """Omitting sequential identification assumptions blocks declaration."""
    with pytest.raises(ValueError, match="required assumptions"):
        Estimand(
            name="sustained_treatment_effect",
            subject_identity="subject_id",
            time_column="time",
            treatment_column="treatment",
            outcome="outcome",
            baseline_covariates=(),
            time_varying_covariates=("state",),
            population="population",
            assumptions=("consistency", "positivity"),
        )


@pytest.mark.parametrize(
    "change",
    [
        "name",
        "column",
        "duplicate_column",
        "population",
        "assumptions",
        "target",
        "comparator",
        "same_regimes",
    ],
)
def test_estimand_rejects_ambiguous_declarations(change: str) -> None:
    """Estimand metadata cannot leave identification or columns implicit."""
    values = dataclasses.asdict(make_estimand())
    if change == "name":
        values["name"] = ""
    elif change == "column":
        values["outcome"] = ""
    elif change == "duplicate_column":
        values["baseline_covariates"] = ("state",)
    elif change == "population":
        values["population"] = ""
    elif change == "assumptions":
        values["assumptions"] = ()
    elif change == "target":
        values["target_treatment"] = 2
    elif change == "comparator":
        values["comparator_treatment"] = 2
    elif change == "same_regimes":
        values["comparator_treatment"] = 1
    else:
        raise AssertionError(f"unhandled change {change}")
    values["baseline_covariates"] = tuple(values["baseline_covariates"])
    values["time_varying_covariates"] = tuple(values["time_varying_covariates"])
    values["assumptions"] = tuple(values["assumptions"])

    with pytest.raises(ValueError):
        Estimand(**values)


def test_contract_dataclasses_reject_invalid_evidence() -> None:
    """Period, regime, and result contracts reject inconsistent evidence."""
    period = PeriodDiagnostics(
        period="0",
        rows=4,
        treated_rows=2,
        control_rows=2,
        denominator_propensity_min=0.2,
        denominator_propensity_max=0.8,
        numerator_propensity_min=0.4,
        numerator_propensity_max=0.6,
        fraction_weight_truncated=0.0,
        maximum_untruncated_weight=2.0,
        maximum_weight=2.0,
        effective_sample_size=4.0,
    )
    with pytest.raises(ValueError, match="arm counts"):
        dataclasses.replace(period, treated_rows=1)
    with pytest.raises(ValueError, match="strictly between"):
        dataclasses.replace(period, denominator_propensity_min=0.0)
    regime = RegimeEstimate(
        regime="always_treated",
        treatment=1,
        estimate=1.0,
        standard_error=0.1,
        interval=(0.8, 1.2),
        rows=2,
        weight_sum=2.0,
        effective_sample_size=2.0,
    )
    with pytest.raises(ValueError, match="regime treatment"):
        dataclasses.replace(regime, treatment=2)
    result = Estimator(make_estimand(), make_policy()).fit(make_frame())
    with pytest.raises(ValueError, match="at least two periods"):
        dataclasses.replace(result, periods=(result.periods[0],))
    with pytest.raises(ValueError, match="symmetric"):
        dataclasses.replace(result, covariance=((1.0, 2.0), (0.0, 1.0)))
    with pytest.raises(ValueError, match="SHA-256"):
        dataclasses.replace(result, raw_input_fingerprint="invalid")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("confidence_level", 1.0),
        ("propensity_clip", 0.0),
        ("maximum_weight", 1.0),
        ("seed", -1),
        ("tolerance", 0.0),
        ("minimum_subjects", 0),
        ("minimum_effective_sample_size", 0.0),
        ("missingness", "impute"),
    ],
)
def test_policy_rejects_unbounded_or_ambiguous_values(
    field: str, value: object
) -> None:
    """Policy validation prevents silent changes to the estimand."""
    with pytest.raises(ValueError):
        Config(**{field: value})
