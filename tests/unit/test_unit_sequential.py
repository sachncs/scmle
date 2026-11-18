"""Stabilized sequential IPW: declaration, panel, weighting, estimator, report."""

import dataclasses
import math

import numpy
import pandas
import pytest
from scipy import stats

from scmle.contracts import errors
from scmle.evaluation.envelope import Counts
from scmle.evaluation.sequential import declaration
from scmle.evaluation.sequential import estimator
from scmle.evaluation.sequential import panel
from scmle.evaluation.sequential import report
from scmle.evaluation.sequential import weighting
from tests.unit.test_sequential_estimation import make_estimand
from tests.unit.test_sequential_estimation import make_frame


@pytest.fixture(scope="module")
def fitted():
    return estimator.Estimator(make_estimand()).fit(make_frame())


# declaration


def test_the_declaration_constants_are_fixed():
    assert declaration.PROPENSITY_CLIP_CEILING == 0.5
    assert declaration.MIN_PERIODS == 2
    assert declaration.REGIME_COUNT == 2
    assert declaration.COVARIANCE_DIMENSION == 2
    assert declaration.SHA256_HEX_LENGTH == 64
    assert declaration.MATRIX_NDIM == 2
    assert {
        "consistency",
        "positivity",
        "sequential_exchangeability",
        "no_unmeasured_time_varying_confounding",
        "time_varying_covariates_are_pre_treatment",
        "complete_follow_up",
    } == declaration.REQUIRED_ASSUMPTIONS


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"name": ""}, "estimand name must not be empty"),
        ({"time_varying_covariates": ("",)}, "column names must be non-empty"),
        (
            {"time_varying_covariates": ("baseline_score",)},
            "columns must be unique",
        ),
        ({"population": " "}, "population must not be empty"),
        ({"assumptions": ()}, "needs explicit assumptions"),
        (
            {"assumptions": ("consistency",)},
            (
                "missing required assumptions: complete_follow_up, "
                "no_unmeasured_time_varying_confounding, positivity, "
                "sequential_exchangeability, "
                "time_varying_covariates_are_pre_treatment"
            ),
        ),
        ({"target_treatment": 2}, "target_treatment must be 0 or 1"),
        ({"comparator_treatment": -1}, "comparator_treatment must be 0 or 1"),
        (
            {"comparator_treatment": 1},
            "target and comparator treatments must differ",
        ),
    ],
)
def test_an_invalid_estimand_is_rejected(changes, message):
    with pytest.raises(errors.InputError, match=message):
        dataclasses.replace(make_estimand(), **changes)


def test_an_estimand_serialises_grouped_keys():
    record = make_estimand().to_dict()
    assert record["covariates"] == {
        "baseline": ("baseline_score",),
        "varying": ("state",),
    }
    assert record["regimes"] == {"target": 1, "comparator": 0}
    assert record["subject"] == "subject_id"


def test_a_default_config_has_the_documented_policy():
    assert declaration.Config().to_dict() == {
        "confidence": 0.95,
        "clip": 0.01,
        "weight": 20.0,
        "missingness": "reject",
        "seed": 0,
        "tolerance": 1e-08,
        "iterations": 100,
        "reference": 1e-06,
        "minimums": {"subjects": 30, "arm": 2, "ess": 5.0},
    }


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"confidence_level": 1.0}, "confidence_level must lie strictly"),
        ({"propensity_clip": 0.5}, "propensity_clip must lie strictly"),
        ({"maximum_weight": 1.0}, "maximum_weight must be finite and greater"),
        (
            {"maximum_weight": math.inf},
            "maximum_weight must be finite and greater",
        ),
        ({"missingness": "x"}, "missingness must be reject or complete_case"),
        ({"seed": 1.0}, "seed must be a non-negative integer"),
        ({"tolerance": -1.0}, "tolerance must be positive and finite"),
        (
            {"minimum_subjects": 0},
            "minimum_subjects must be a positive integer",
        ),
        (
            {"minimum_arm_rows": False},
            "minimum_arm_rows must be a positive integer",
        ),
        ({"minimum_effective_sample_size": -1.0}, "must be positive"),
    ],
)
def test_an_invalid_config_is_rejected(changes, message):
    with pytest.raises(errors.InputError, match=message):
        declaration.Config(**changes)


# panel


def test_a_conformance_names_every_declared_column():
    assert panel.conform(make_frame(), make_estimand()) == [
        "subject_id",
        "time",
        "treatment",
        "outcome",
        "baseline_score",
        "state",
    ]
    with pytest.raises(
        errors.InputError, match=r"missing columns: \['state'\]"
    ):
        panel.conform(make_frame().drop(columns="state"), make_estimand())


def test_a_situation_rejects_missing_non_finite_and_duplicate_keys():
    estimand = make_estimand()
    gap = make_frame().astype({"time": float})
    gap.loc[0, "time"] = numpy.nan
    with pytest.raises(errors.InputError, match="must not be missing"):
        panel.situate(gap, estimand)
    endless = make_frame().astype({"time": float})
    endless.loc[0, "time"] = numpy.inf
    with pytest.raises(
        errors.InputError, match="column 'time' contains non-finite"
    ):
        panel.situate(endless, estimand)
    doubled = pandas.concat([make_frame(), make_frame().iloc[[0]]])
    with pytest.raises(errors.InputError, match="duplicate subject-time rows"):
        panel.situate(doubled, estimand)


def test_a_chronology_rejects_only_equal_neighbours():
    estimand = make_estimand()
    descending = pandas.DataFrame({"subject_id": [1, 1], "time": [2, 1]})
    assert panel.chronologize(descending, estimand) is None
    with pytest.raises(errors.InputError, match="strictly increasing"):
        panel.nonrepeating([1, 2, 2])
    assert panel.nonrepeating([3, 1, 3]) is None


def test_an_order_sorts_distinct_periods_or_rejects_mixed_types():
    frame = pandas.DataFrame({"time": [3, 1, 3, 2]})
    assert panel.order(frame, "time") == [1, 2, 3]
    with pytest.raises(errors.InputError, match="invalid ordering"):
        panel.order(pandas.DataFrame({"time": [1, "a"]}), "time")


def test_a_preparation_counts_subjects_and_drops_whole_subjects():
    raw = make_frame()
    prepared, rows, subjects, dropped = panel.prepare(
        raw, make_estimand(), declaration.Config()
    )
    assert (rows, subjects, dropped) == (0, 180, 0)
    assert len(prepared) == len(raw)
    raw.loc[0, "state"] = numpy.nan
    with pytest.raises(errors.InputError, match="missing treatment, outcome"):
        panel.prepare(raw, make_estimand(), declaration.Config())
    prepared, rows, subjects, dropped = panel.prepare(
        raw, make_estimand(), declaration.Config(missingness="complete_case")
    )
    assert (rows, subjects, dropped) == (3, 180, 1)
    assert raw.loc[0, "subject_id"] not in set(prepared["subject_id"])


def test_a_preparation_rejects_unbalanced_and_small_panels():
    estimand = make_estimand()
    with pytest.raises(errors.KindError, match="must be a pandas DataFrame"):
        panel.prepare({}, estimand, declaration.Config())
    with pytest.raises(errors.InputError, match="must not be empty"):
        panel.prepare(pandas.DataFrame(), estimand, declaration.Config())
    one = make_frame()
    one = one[one.time == 0]
    with pytest.raises(errors.InputError, match="at least two periods"):
        panel.prepare(one, estimand, declaration.Config())
    holes = make_frame().drop(index=[0])
    with pytest.raises(errors.InputError, match="one row at every period"):
        panel.prepare(holes, estimand, declaration.Config())
    few = make_frame()
    few = few[few.subject_id < 29]
    with pytest.raises(errors.InputError, match="insufficient subjects"):
        panel.prepare(few, estimand, declaration.Config())
    binary = make_frame()
    binary.loc[0, "treatment"] = 3
    with pytest.raises(errors.InputError, match="treatment must be binary"):
        panel.prepare(binary, estimand, declaration.Config())
    gone = make_frame().assign(outcome=numpy.nan)
    with pytest.raises(errors.InputError, match="left no subjects"):
        panel.prepare(
            gone, estimand, declaration.Config(missingness="complete_case")
        )


def test_a_pretreatment_check_rejects_varying_baselines():
    varying = make_frame()
    varying.loc[1, "baseline_score"] += 1
    with pytest.raises(
        errors.InputError, match="baseline covariate 'baseline_score' varies"
    ):
        panel.pretreat(varying, make_estimand())


def test_a_tabulation_builds_subject_by_period_arrays():
    frame = pandas.DataFrame(
        {
            "subject_id": [2, 2, 1, 1],
            "time": [0, 1, 0, 1],
            "treatment": [1.0, 0.0, 0.0, 1.0],
            "outcome": [5.0, 6.0, 7.0, 8.0],
            "baseline_score": [0.5, 0.5, 0.25, 0.25],
            "state": [1.0, 2.0, 3.0, 4.0],
        }
    )
    found = panel.tabulate(frame, make_estimand(), [0, 1], [1, 2])
    assert found["treatment"].tolist() == [[0.0, 1.0], [1.0, 0.0]]
    assert found["outcome"].tolist() == [[7.0, 8.0], [5.0, 6.0]]
    assert found["baseline"].tolist() == [[0.25], [0.5]]
    assert found["time_varying"].tolist() == [[[3.0], [4.0]], [[1.0], [2.0]]]


# weighting


def test_a_regression_rejects_bad_shapes():
    with pytest.raises(errors.InputError, match="two-dimensional"):
        weighting.regress(numpy.zeros(3), numpy.zeros(3), declaration.Config())
    with pytest.raises(errors.InputError, match="matching rows"):
        weighting.regress(
            numpy.zeros((3, 1)), numpy.zeros(2), declaration.Config()
        )


def test_a_regression_of_a_constant_treatment_does_not_converge():
    with pytest.raises(
        errors.InputError, match="did not converge within policy bound"
    ):
        weighting.regress(
            numpy.zeros((10, 0)),
            numpy.ones(10),
            declaration.Config(max_iterations=3),
        )


def test_a_regression_of_no_features_is_the_mean():
    t = numpy.array([1.0, 0.0, 0.0, 1.0, 1.0])
    probabilities, ignored = weighting.regress(
        numpy.zeros((5, 0)), t, declaration.Config()
    )
    numpy.testing.assert_allclose(probabilities, 0.6, atol=1e-7)


def test_an_overlap_check_enforces_the_positivity_support():
    weighting.overlap(numpy.array([0.1, 0.9]), 0, 0.1)
    with pytest.raises(
        errors.InputError,
        match=r"period 4 violates declared positivity support \[0.1, 0.9\]",
    ):
        weighting.overlap(numpy.array([0.09, 0.5]), 4, 0.1)
    with pytest.raises(errors.InputError, match="positivity"):
        weighting.overlap(numpy.array([0.5, 0.91]), 4, 0.1)


def test_an_effective_sample_size_is_kish():
    assert weighting.effective_sample_size(numpy.array([2.0, 2.0])) == 2.0
    assert weighting.effective_sample_size(numpy.array([0.0])) == 0.0


def policy(**changes):
    values = {"minimum_effective_sample_size": 1.0, "minimum_arm_rows": 2}
    values.update(changes)
    return declaration.Config(**values)


def test_an_average_is_the_weighted_mean_with_influence_error():
    mask = numpy.array([True, True, False, True])
    weights = numpy.array([1.0, 3.0, 9.0, 2.0])
    outcome = numpy.array([2.0, 4.0, 100.0, 1.0])
    found = weighting.average(
        "always_treated", 1, mask, weights, outcome, policy(), 2.0
    )
    mean = (2 + 12 + 2) / 6
    influence = (
        numpy.array([1.0, 3.0, 2.0]) * (numpy.array([2.0, 4.0, 1.0]) - mean) / 6
    )
    se = math.sqrt((influence**2).sum())
    assert found.estimate == pytest.approx(mean)
    assert found.standard_error == pytest.approx(se)
    assert found.interval == pytest.approx((mean - 2 * se, mean + 2 * se))
    assert found.rows == 3
    assert found.weight_sum == 6.0
    assert found.effective_sample_size == pytest.approx(36 / 14)
    assert found.treatment == 1
    assert found.regime == "always_treated"


def test_an_average_rejects_thin_or_weak_support():
    weights = numpy.array([1.0, 1.0, 1.0])
    outcome = numpy.zeros(3)
    with pytest.raises(
        errors.InputError, match="regime 'r' has insufficient treatment-history"
    ):
        weighting.average(
            "r",
            1,
            numpy.array([True, False, False]),
            weights,
            outcome,
            policy(),
            2.0,
        )
    with pytest.raises(errors.InputError, match="no weighted support"):
        weighting.average(
            "r",
            1,
            numpy.array([True, True, False]),
            numpy.zeros(3),
            outcome,
            policy(),
            2.0,
        )
    with pytest.raises(
        errors.InputError, match="insufficient effective sample size"
    ):
        weighting.average(
            "r",
            1,
            numpy.array([True, True, False]),
            weights,
            outcome,
            policy(minimum_effective_sample_size=2.5),
            2.0,
        )


def regimes():
    target = report.RegimeEstimate(
        "always_treated", 1, 2.0, 0.1, (1.8, 2.2), 2, 3.0, 1.8
    )
    comparator = report.RegimeEstimate(
        "always_control", 0, 1.0, 0.1, (0.8, 1.2), 2, 2.0, 2.0
    )
    target_mask = numpy.array([True, True, False, False])
    comparator_mask = numpy.array([False, False, True, True])
    weights = numpy.array([1.0, 2.0, 1.0, 1.0])
    outcome = numpy.array([1.0, 2.5, 0.0, 2.0])
    return target_mask, comparator_mask, weights, outcome, target, comparator


def test_a_covariance_is_built_from_regime_influences():
    tm, cm, w, y, target, comparator = regimes()
    cov = weighting.covary(tm, cm, w, y, target, comparator)
    ti = numpy.array([1.0 * (1 - 2) / 3, 2.0 * (2.5 - 2) / 3])
    ci = numpy.array([(0 - 1) / 2, (2 - 1) / 2])
    assert cov[0, 0] == pytest.approx((ti**2).sum())
    assert cov[1, 1] == pytest.approx((ci**2).sum())
    assert cov[0, 1] == 0.0 and cov[1, 0] == 0.0


def test_a_referee_recomputes_both_regimes():
    tm, cm, w, y, target, comparator = regimes()
    out = weighting.referee(tm, cm, w, y, target, comparator, 1e-9)
    assert out["method"] == "independent_direct_normalized_history_weight_means"
    assert out["passed"] is True
    assert [r["regime"] for r in out["regimes"]] == [
        "always_treated",
        "always_control",
    ]
    assert out["regimes"][0]["estimate"] == pytest.approx(2.0)
    assert out["regimes"][1]["rows"] == 2
    assert out["contrast"] == pytest.approx(1.0)
    off = report.RegimeEstimate(
        "always_treated", 1, 2.5, 0.1, (1.8, 2.7), 2, 3.0, 1.8
    )
    failed = weighting.referee(tm, cm, w, y, off, comparator, 0.4)
    assert failed["passed"] is False
    assert failed["regimes"][0]["absolute_difference"] == pytest.approx(0.5)
    assert failed["regimes"][0]["passed"] is False
    assert failed["regimes"][1]["passed"] is True


def test_a_period_label_and_imprint_are_stable():
    assert (
        weighting.periodize(pandas.Timestamp("2026-03-04"))
        == "2026-03-04T00:00:00"
    )
    assert weighting.periodize(numpy.int32(7)) == "7"
    frame = pandas.DataFrame({"a": [1.0]})
    assert weighting.imprint(frame) == weighting.imprint(frame.copy())
    assert weighting.imprint(frame) != weighting.imprint(
        frame.rename(columns={"a": "b"})
    )
    with pytest.raises(errors.InputError, match="hashable"):
        weighting.imprint(pandas.DataFrame({"a": [{1}]}))


# estimator


def test_an_estimator_checks_its_arguments():
    with pytest.raises(errors.KindError, match="estimand must be Estimand"):
        estimator.Estimator(None)
    with pytest.raises(errors.KindError, match="policy must be Config"):
        estimator.Estimator(make_estimand(), {})
    assert estimator.Estimator(make_estimand()).policy == declaration.Config()


def test_a_fit_contrasts_the_regimes_with_their_covariance(fitted):
    target, comparator = fitted.regimes
    assert (
        target.regime == "always_treated"
        and comparator.regime == "always_control"
    )
    assert fitted.contrast.estimate == pytest.approx(
        target.estimate - comparator.estimate
    )
    se = math.sqrt(sum(sum(row) for row in fitted.covariance))
    assert fitted.contrast.se == pytest.approx(se)
    z = stats.norm.ppf(0.975)
    assert fitted.contrast.interval[1] == pytest.approx(
        fitted.contrast.estimate + z * se
    )
    assert fitted.subjects == Counts(180, 180, 0)
    assert fitted.rows == Counts(540, 540, 0)
    assert [p.period for p in fitted.periods] == ["0", "1", "2"]
    assert fitted.reference["passed"] is True
    assert fitted.reference["contrast"] == pytest.approx(
        fitted.contrast.estimate
    )


def test_a_fit_summary_records_the_regime_history(fitted):
    summary = fitted.diagnostics
    assert summary["method"] == "stabilized_treatment_history_ipw"
    assert summary["period_count"] == 3
    assert summary["subject_count"] == 180
    assert summary["regime_history"] == {
        "target": [1, 1, 1],
        "comparator": [0, 0, 0],
    }
    assert summary["weight_truncation"]["maximum_weight"] == 20.0
    assert summary["weight_truncation"]["fraction_truncated_final"] == (
        fitted.periods[-1].fraction_weight_truncated
    )
    assert set(summary["fit_iterations_final_period"]) == {
        "numerator",
        "denominator",
    }
    assert fitted.limitations["missingness"]["subjects_dropped"] == 0


def test_a_reversed_regime_names_the_target_always_control():
    reversed_estimand = dataclasses.replace(
        make_estimand(), target_treatment=0, comparator_treatment=1
    )
    found = estimator.Estimator(reversed_estimand).fit(make_frame())
    assert [r.regime for r in found.regimes] == [
        "always_control",
        "always_treated",
    ]


def test_a_record_truncates_and_checks_the_effective_size():
    cumulative = numpy.array([1.0, 30.0, 2.0, 1.0])
    found = estimator.record(
        5,
        4,
        (2, 2),
        numpy.array([0.4, 0.6]),
        numpy.array([0.2, 0.7]),
        cumulative,
        policy(maximum_weight=20.0),
    )
    assert found.period == "5"
    assert found.fraction_weight_truncated == 0.25
    assert found.maximum_untruncated_weight == 30.0
    assert found.maximum_weight == 20.0
    assert found.denominator_propensity_min == 0.2
    assert found.numerator_propensity_max == 0.6
    truncated = numpy.array([1.0, 20.0, 2.0, 1.0])
    assert found.effective_sample_size == pytest.approx(
        truncated.sum() ** 2 / (truncated**2).sum()
    )
    with pytest.raises(
        errors.InputError,
        match="period 5 has insufficient effective sample size",
    ):
        estimator.record(
            5,
            4,
            (2, 2),
            numpy.array([0.4]),
            numpy.array([0.2]),
            cumulative,
            policy(minimum_effective_sample_size=2.0),
        )


def test_an_accumulation_needs_both_arms_in_every_period():
    matrices = {
        "treatment": numpy.array([[1.0], [1.0], [1.0], [0.0]]),
        "baseline": numpy.zeros((4, 0)),
        "time_varying": numpy.zeros((4, 1, 0)),
    }
    with pytest.raises(
        errors.InputError, match="period 'p' has insufficient treatment support"
    ):
        estimator.accumulate(matrices, ["p"], 4, policy())


def test_the_limitations_name_the_bounded_contract():
    limits = estimator.qualify(declaration.Config(), 2)
    assert (
        limits["regimes"]
        == "static always-treated versus always-control histories only"
    )
    assert limits["missingness"]["subjects_dropped"] == 2
    assert limits["missingness"]["policy"] == "reject"
    assert (
        "truncation" in limits
        and "uncertainty" in limits
        and "unsupported" in limits
    )


# report


def test_a_report_round_trips(fitted):
    assert report.Estimation.from_dict(fitted.to_dict()) == fitted


def period(**changes):
    values = {
        "period": "0",
        "rows": 4,
        "treated_rows": 2,
        "control_rows": 2,
        "denominator_propensity_min": 0.2,
        "denominator_propensity_max": 0.8,
        "numerator_propensity_min": 0.3,
        "numerator_propensity_max": 0.7,
        "fraction_weight_truncated": 0.0,
        "maximum_untruncated_weight": 2.0,
        "maximum_weight": 2.0,
        "effective_sample_size": 3.0,
    }
    values.update(changes)
    return report.PeriodDiagnostics(**values)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"period": ""}, "period label must not be empty"),
        ({"control_rows": 0, "rows": 2}, "row counts must be positive"),
        ({"rows": 5}, "reconcile to rows"),
        ({"maximum_weight": math.inf}, "maximum_weight must be finite"),
        (
            {"fraction_weight_truncated": 1.5},
            "fraction_weight_truncated must lie",
        ),
        (
            {"denominator_propensity_min": 0.0},
            "denominator_propensity_min must lie strictly",
        ),
        (
            {"numerator_propensity_max": 1.0},
            "numerator_propensity_max must lie strictly",
        ),
        ({"maximum_untruncated_weight": 0.0}, "weight bounds must be positive"),
        (
            {"effective_sample_size": 0.0},
            "effective_sample_size must be positive",
        ),
    ],
)
def test_an_invalid_period_diagnostic_is_rejected(changes, message):
    with pytest.raises(errors.InputError, match=message):
        period(**changes)


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (
            ("", 1, 1.0, 0.1, (0.9, 1.1), 2, 1.0, 1.0),
            "regime name must not be empty",
        ),
        (
            ("r", 2, 1.0, 0.1, (0.9, 1.1), 2, 1.0, 1.0),
            "regime treatment must be 0 or 1",
        ),
        (
            ("r", 1, 1.0, 0.1, (0.9, 1.1), 0, 1.0, 1.0),
            "positive rows and weight sum",
        ),
        (
            ("r", 1, 1.0, 0.1, (0.9, 1.1), 2, 0.0, 1.0),
            "positive rows and weight sum",
        ),
        (
            ("r", 1, 1.0, 0.1, (0.9, 1.1), 2, 1.0, 0.0),
            "effective sample size must be positive",
        ),
        (
            ("r", 1, math.nan, 0.1, (0.9, 1.1), 2, 1.0, 1.0),
            "estimate must be finite",
        ),
        (
            ("r", 1, 1.0, -0.1, (0.9, 1.1), 2, 1.0, 1.0),
            "ordered and non-negative",
        ),
        (
            ("r", 1, 1.0, 0.1, (1.1, 0.9), 2, 1.0, 1.0),
            "ordered and non-negative",
        ),
    ],
)
def test_an_invalid_regime_estimate_is_rejected(values, message):
    with pytest.raises(errors.InputError, match=message):
        report.RegimeEstimate(*values)


def test_an_invalid_contrast_is_rejected():
    with pytest.raises(errors.InputError, match="se must not be negative"):
        report.Contrast(1.0, -0.1, (0.0, 2.0))
    with pytest.raises(errors.InputError, match="interval must be ordered"):
        report.Contrast(1.0, 0.1, (2.0, 0.0))
    with pytest.raises(errors.InputError, match="must be finite"):
        report.Contrast(math.inf, 0.1, (0.0, 2.0))


@pytest.mark.parametrize(
    ("changes", "error", "message"),
    [
        ({"periods": ()}, errors.InputError, "at least two periods"),
        ({"covariance": ((1.0,),)}, errors.InputError, "2 by 2 matrix"),
        (
            {"covariance": ((1.0, 2.0), (0.0, 1.0))},
            errors.InputError,
            "finite and symmetric",
        ),
        ({"contrast": 1.0}, errors.KindError, "contrast must be Contrast"),
        ({"subjects": (1, 1, 0)}, errors.KindError, "subjects must be Counts"),
        (
            {"limitations": ()},
            errors.KindError,
            "limitations must be a mapping",
        ),
        (
            {"digests": {"frame": "x", "y": "z"}},
            errors.InputError,
            "exactly the frame",
        ),
    ],
)
def test_an_invalid_estimation_is_rejected(fitted, changes, error, message):
    with pytest.raises(error, match=message):
        dataclasses.replace(fitted, **changes)


def test_an_estimation_needs_exactly_two_regimes(fitted):
    with pytest.raises(
        errors.InputError, match="target and comparator regimes"
    ):
        dataclasses.replace(fitted, regimes=fitted.regimes[:1])
