"""Period-specific Hajek IPW: declaration, panel, weighting, estimator, report."""

import dataclasses
import datetime
import hashlib
import json
import math

import numpy
import pandas
import pytest
from scipy import special
from scipy import stats

from scmle.contracts import errors
from scmle.evaluation.envelope import Counts
from scmle.evaluation.longitudinal import declaration
from scmle.evaluation.longitudinal import estimator
from scmle.evaluation.longitudinal import panel
from scmle.evaluation.longitudinal import report
from scmle.evaluation.longitudinal import weighting
from tests.unit.test_longitudinal_estimation import make_estimand
from tests.unit.test_longitudinal_estimation import make_frame


def period(**changes):
    values = {
        "period": "0",
        "estimate": 1.0,
        "standard_error": 0.5,
        "interval": (0.0, 2.0),
        "rows": 10,
        "treated_rows": 4,
        "control_rows": 6,
        "propensity_min": 0.1,
        "propensity_max": 0.9,
        "fraction_clipped": 0.0,
        "effective_sample_size_treated": 3.0,
        "effective_sample_size_control": 5.0,
        "converged": True,
        "iterations": 3,
    }
    values.update(changes)
    return declaration.PeriodATE(**values)


# declaration


def test_the_declaration_constants_are_fixed():
    assert declaration.PROPENSITY_CLIP_CEILING == 0.5
    assert declaration.SHA256_HEX_LENGTH == 64
    assert declaration.MATRIX_NDIM == 2
    assert {
        "consistency",
        "positivity",
        "conditional_exchangeability",
        "no_unmeasured_time_varying_confounding",
        "no_time_varying_covariates",
    } == declaration.REQUIRED_ASSUMPTIONS


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"name": " "}, "estimand name must not be empty"),
        ({"outcome": ""}, "estimand column names must be non-empty strings"),
        (
            {"covariates": (3,)},
            "estimand column names must be non-empty strings",
        ),
        ({"outcome": "time"}, "estimand columns must be unique"),
        ({"population": ""}, "estimand population must not be empty"),
        ({"assumptions": ()}, "estimand needs explicit assumptions"),
        (
            {"assumptions": ("consistency", "positivity")},
            (
                "estimand is missing required assumptions: "
                "conditional_exchangeability, "
                "no_time_varying_covariates, "
                "no_unmeasured_time_varying_confounding"
            ),
        ),
    ],
)
def test_an_invalid_estimand_is_rejected(changes, message):
    with pytest.raises(errors.InputError, match=f"^{message}$"):
        dataclasses.replace(make_estimand(), **changes)


def test_an_estimand_serialises_with_short_column_keys():
    record = make_estimand().to_dict()
    assert record["subject"] == "subject_id"
    assert record["time"] == "time"
    assert record["treatment"] == "treatment"
    assert "subject_identity" not in record


def test_a_default_config_has_the_documented_policy():
    assert declaration.Config().to_dict() == {
        "confidence": 0.95,
        "clip": 0.01,
        "missingness": "reject",
        "seed": 0,
        "tolerance": 1e-08,
        "iterations": 100,
        "reference": 1e-06,
        "minimums": {"period": 8, "arm": 2, "ess": 2.0},
    }


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"confidence_level": 0.0}, "confidence_level must lie strictly"),
        ({"confidence_level": 1.0}, "confidence_level must lie strictly"),
        ({"propensity_clip": 0.0}, "propensity_clip must lie strictly"),
        ({"propensity_clip": 0.5}, "propensity_clip must lie strictly"),
        (
            {"missingness": "impute"},
            "missingness must be reject or complete_case",
        ),
        ({"seed": -1}, "seed must be a non-negative integer"),
        ({"seed": True}, "seed must be a non-negative integer"),
        ({"tolerance": 0.0}, "tolerance must be positive and finite"),
        (
            {"reference_tolerance": math.inf},
            "reference_tolerance must be positive",
        ),
        ({"max_iterations": 0}, "max_iterations must be a positive integer"),
        (
            {"minimum_rows_per_arm": 1.5},
            "minimum_rows_per_arm must be a positive",
        ),
        (
            {"minimum_rows_per_period": True},
            "minimum_rows_per_period must be a",
        ),
        (
            {"minimum_effective_sample_size": 0.0},
            "minimum_effective_sample_size",
        ),
        (
            {"minimum_effective_sample_size": math.nan},
            "minimum_effective_sample_size",
        ),
        (
            {"minimum_rows_per_period": 3, "minimum_rows_per_arm": 2},
            "minimum_rows_per_period must cover both treatment arms",
        ),
    ],
)
def test_an_invalid_config_is_rejected(changes, message):
    with pytest.raises(errors.InputError, match=message):
        declaration.Config(**changes)


def test_a_config_accepts_its_boundaries():
    found = declaration.Config(
        minimum_rows_per_period=4, minimum_rows_per_arm=2, seed=0
    )
    assert found.minimum_rows_per_period == 4


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"period": ""}, "period label must not be empty"),
        (
            {"rows": 0, "treated_rows": 0, "control_rows": 0},
            "row counts must be positive",
        ),
        ({"treated_rows": 0, "rows": 6}, "row counts must be positive"),
        ({"rows": 11}, "arm counts must reconcile to rows"),
        ({"iterations": 0}, "iterations must be positive"),
        ({"converged": False}, "must be converged"),
        ({"estimate": math.inf}, "estimate must be finite"),
        (
            {"effective_sample_size_control": math.nan},
            "effective_sample_size_control must be finite",
        ),
        ({"standard_error": -0.1}, "standard_error must not be negative"),
        ({"interval": (2.0, 1.0)}, "period interval must be ordered"),
        (
            {"fraction_clipped": 1.1},
            "fraction_clipped must lie between 0 and 1",
        ),
        (
            {"fraction_clipped": -0.1},
            "fraction_clipped must lie between 0 and 1",
        ),
        ({"propensity_min": -0.1}, "propensities must lie between 0 and 1"),
        ({"propensity_max": 1.1}, "propensities must lie between 0 and 1"),
    ],
)
def test_an_invalid_period_is_rejected(changes, message):
    with pytest.raises(errors.InputError, match=message):
        period(**changes)


def test_a_period_accepts_boundary_values_and_serialises_nested():
    edge = period(
        standard_error=0.0,
        interval=(1.0, 1.0),
        fraction_clipped=1.0,
        propensity_min=0.0,
        propensity_max=1.0,
        iterations=1,
    )
    assert edge.to_dict() == {
        "period": "0",
        "estimate": 1.0,
        "se": 0.0,
        "interval": (1.0, 1.0),
        "rows": 10,
        "treated": 4,
        "control": 6,
        "clipped": 1.0,
        "converged": True,
        "iterations": 1,
        "propensity": {"low": 0.0, "high": 1.0},
        "ess": {"treated": 3.0, "control": 5.0},
    }


# panel


def frame():
    return make_frame()


def test_a_conformance_lists_the_declared_columns():
    assert panel.conform(frame(), make_estimand()) == [
        "subject_id",
        "time",
        "treatment",
        "outcome",
        "baseline_score",
    ]
    with pytest.raises(
        errors.InputError,
        match=r"missing columns: \['outcome', 'baseline_score'\]",
    ):
        panel.conform(
            frame().drop(columns=["outcome", "baseline_score"]), make_estimand()
        )


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (
            lambda f: f.assign(
                subject_id=f.subject_id.astype(object).where(f.index != 0, None)
            ),
            "missing subject identities",
        ),
        (
            lambda f: f.assign(
                subject_id=f.subject_id.astype(str).where(f.index != 0, "  ")
            ),
            "missing subject identities",
        ),
        (
            lambda f: f.assign(
                time=f.time.astype(float).where(f.index != 0, numpy.nan)
            ),
            "missing time values",
        ),
        (
            lambda f: f.assign(
                subject_id=f.subject_id.astype(float).where(
                    f.index != 0, numpy.inf
                )
            ),
            "column 'subject_id' contains non-finite values",
        ),
        (
            lambda f: pandas.concat([f, f.iloc[[0]]], ignore_index=True),
            "duplicate subject-time rows",
        ),
        (
            lambda f: f.iloc[::-1].reset_index(drop=True),
            "strictly increasing per subject",
        ),
    ],
)
def test_a_situation_rejects_bad_identities_and_times(edit, message):
    with pytest.raises(errors.InputError, match=message):
        panel.situate(edit(frame()), make_estimand())


def test_a_monotone_check_rejects_decreasing_or_repeated_times():
    panel.monotone(pandas.Series([1, 2, 5]))
    with pytest.raises(errors.InputError, match="strictly increasing"):
        panel.monotone(pandas.Series([2, 1]))
    with pytest.raises(errors.InputError, match="strictly increasing"):
        panel.monotone(pandas.Series([1, 1]))


def test_a_chronology_turns_a_comparison_error_into_invalid_ordering(
    monkeypatch,
):
    def fail(times):
        raise TypeError("cannot compare")

    monkeypatch.setattr(panel, "monotone", fail)
    with pytest.raises(
        errors.InputError, match="time values have invalid ordering"
    ):
        panel.chronologize(make_frame(), make_estimand())


def test_a_chronology_reports_the_first_unordered_subject():
    rows = pandas.DataFrame({"subject_id": [1, 1, 2], "time": [2, 1, 0]})
    with pytest.raises(
        errors.InputError, match=r"^time values must be strictly"
    ):
        panel.chronologize(rows, make_estimand())


def test_a_numericisation_converts_in_place_and_rejects_bad_values():
    prepared = pandas.DataFrame({"a": ["1", "2.5"]})
    panel.numericize(prepared, ["a"])
    assert list(prepared["a"]) == [1.0, 2.5]
    with pytest.raises(errors.InputError, match="column 'a' must be numeric"):
        panel.numericize(pandas.DataFrame({"a": ["x"]}), ["a"])
    with pytest.raises(
        errors.InputError, match="column 'a' contains non-finite"
    ):
        panel.numericize(pandas.DataFrame({"a": [numpy.inf]}), ["a"])


def test_a_pretreatment_check_rejects_time_varying_covariates():
    varying = frame()
    varying.loc[0, "baseline_score"] += 1.0
    with pytest.raises(
        errors.InputError, match="covariate 'baseline_score' varies"
    ):
        panel.pretreat(varying, make_estimand())
    bare = dataclasses.replace(make_estimand(), covariates=())
    assert panel.pretreat(varying, bare) is None


def test_a_preparation_keeps_only_declared_numeric_columns():
    raw = frame().assign(extra="x")
    prepared, dropped = panel.prepare(
        raw, make_estimand(), declaration.Config()
    )
    assert dropped == 0
    assert list(prepared.columns) == panel.conform(raw, make_estimand())
    assert len(prepared) == 180


def test_a_preparation_drops_or_rejects_missing_values():
    raw = frame()
    raw.loc[[0, 1], "outcome"] = numpy.nan
    with pytest.raises(errors.InputError, match="missing treatment, outcome"):
        panel.prepare(raw, make_estimand(), declaration.Config())
    prepared, dropped = panel.prepare(
        raw, make_estimand(), declaration.Config(missingness="complete_case")
    )
    assert dropped == 2
    assert len(prepared) == 178


@pytest.mark.parametrize(
    ("value", "error", "message"),
    [
        ([], errors.KindError, "frame must be a pandas DataFrame"),
        (pandas.DataFrame(), errors.InputError, "must not be empty"),
    ],
)
def test_a_preparation_rejects_non_frames_and_empty_frames(
    value, error, message
):
    with pytest.raises(error, match=message):
        panel.prepare(value, make_estimand(), declaration.Config())


def test_a_preparation_rejects_non_binary_treatment_and_thin_frames():
    raw = frame()
    raw.loc[0, "treatment"] = 2
    with pytest.raises(errors.InputError, match="treatment must be binary"):
        panel.prepare(raw, make_estimand(), declaration.Config())
    with pytest.raises(errors.InputError, match="insufficient rows"):
        panel.prepare(frame().iloc[:7], make_estimand(), declaration.Config())
    all_missing = frame().assign(outcome=numpy.nan)
    with pytest.raises(errors.InputError, match="left no rows"):
        panel.prepare(
            all_missing,
            make_estimand(),
            declaration.Config(missingness="complete_case"),
        )


# weighting


def test_a_regression_matches_an_unpenalised_logistic_fit():
    rng = numpy.random.default_rng(0)
    x = rng.normal(size=(300, 1))
    t = (rng.random(300) < special.expit(0.5 + x[:, 0])).astype(float)
    probabilities, iterations = weighting.regress(x, t, declaration.Config())
    assert 1 < iterations <= 100
    design = numpy.column_stack([numpy.ones(300), x])
    gradient = design.T @ (probabilities - t)
    assert numpy.abs(gradient).max() < 1e-6


def test_a_regression_rejects_a_vector_and_a_slow_fit():
    with pytest.raises(errors.InputError, match="two-dimensional matrix"):
        weighting.regress(numpy.zeros(4), numpy.zeros(4), declaration.Config())
    rng = numpy.random.default_rng(0)
    x = rng.normal(size=(50, 1))
    t = (x[:, 0] > 0).astype(float)
    with pytest.raises(errors.InputError, match="did not converge"):
        weighting.regress(x, t, declaration.Config(max_iterations=2))


def test_an_effective_sample_size_is_kish():
    assert (
        weighting.effective_sample_size(numpy.array([1.0, 1.0, 2.0])) == 16 / 6
    )
    assert weighting.effective_sample_size(numpy.zeros(3)) == 0.0


def test_a_weighing_is_the_hajek_contrast():
    p = numpy.array([0.2, 0.4, 0.6, 0.8, 0.3, 0.7])
    t = numpy.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    y = numpy.array([3.0, 2.0, 4.0, 1.0, 0.0, 2.0])
    policy = declaration.Config(minimum_effective_sample_size=1.0)
    out = weighting.weigh(p, p, t, y, policy, 0)
    wt = t / p
    wc = (1 - t) / (1 - p)
    treated = (wt * y).sum() / wt.sum()
    control = (wc * y).sum() / wc.sum()
    assert out["estimate"] == pytest.approx(treated - control)
    assert out["reference_treated_mean"] == pytest.approx(treated)
    assert out["reference_control_mean"] == pytest.approx(control)
    assert out["common_support_lower"] == 0.3
    assert out["common_support_upper"] == 0.6
    assert out["effective_sample_size_treated"] == pytest.approx(
        weighting.effective_sample_size(wt)
    )
    expected = wt * (y - treated) / (wt.sum() / 6) - wc * (y - control) / (
        wc.sum() / 6
    )
    numpy.testing.assert_allclose(out["influence"], expected)


def test_a_weighing_rejects_no_overlap_and_low_effective_size():
    t = numpy.array([1.0, 1.0, 0.0, 0.0])
    y = numpy.zeros(4)
    apart = numpy.array([0.8, 0.9, 0.1, 0.2])
    with pytest.raises(
        errors.InputError, match="period 3 has insufficient propensity overlap"
    ):
        weighting.weigh(apart, apart, t, y, declaration.Config(), 3)
    touching = numpy.array([0.5, 0.9, 0.1, 0.5])
    with pytest.raises(errors.InputError, match="overlap"):
        weighting.weigh(touching, touching, t, y, declaration.Config(), 3)
    mixed = numpy.array([0.3, 0.6, 0.4, 0.5])
    with pytest.raises(
        errors.InputError, match="insufficient effective sample size"
    ):
        weighting.weigh(
            mixed,
            mixed,
            t,
            y,
            declaration.Config(minimum_effective_sample_size=2.5),
            "x",
        )


def test_a_sandwich_aligns_influence_by_subject():
    fitted = [
        {"subjects": ["a", "b"], "influence": [2.0, -2.0], "rows": 2},
        {"subjects": ["b", "c"], "influence": [4.0, 0.0], "rows": 2},
        {"subjects": ["z"], "influence": [1.0], "rows": 1},
    ]
    cov = weighting.sandwich(fitted)
    # period 0 centred: a 1, b -1; period 1 centred: b 1, c -1
    assert cov[0, 0] == pytest.approx(2.0 * 2 / 1)
    assert cov[1, 1] == pytest.approx(2.0 * 2 / 1)
    assert cov[0, 1] == pytest.approx(-1.0)
    assert cov[1, 0] == pytest.approx(-1.0)
    assert cov[2, 2] == 0.0
    assert cov[0, 2] == 0.0


def wave():
    return {
        "period": 2,
        "estimate": 1.5,
        "rows": 10,
        "treated_rows": 4,
        "control_rows": 6,
        "propensity_min": 0.1,
        "propensity_max": 0.9,
        "fraction_clipped": 0.2,
        "effective_sample_size_treated": 3.5,
        "effective_sample_size_control": 5.5,
        "iterations": 4,
        "propensity": numpy.array([0.2, 0.4, 0.6, 0.8, 0.3, 0.7]),
        "clipped_propensity": numpy.array([0.2, 0.4, 0.6, 0.8, 0.3, 0.7]),
        "treatment": numpy.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]),
        "outcome": numpy.array([3.0, 2.0, 4.0, 1.0, 0.0, 2.0]),
    }


def test_a_periodisation_builds_the_interval_from_the_variance():
    found = weighting.periodise(wave(), 0.25, 2.0)
    assert found.period == "2"
    assert found.standard_error == 0.5
    assert found.interval == (0.5, 2.5)
    assert found.converged is True
    assert found.iterations == 4
    assert weighting.periodise(wave(), -1.0, 2.0).standard_error == 0.0


def test_a_referee_recomputes_and_compares_each_period():
    item = wave()
    p, t, y = item["clipped_propensity"], item["treatment"], item["outcome"]
    wt = numpy.where(t == 1, 1 / p, 0)
    wc = numpy.where(t == 0, 1 / (1 - p), 0)
    direct = (wt * y).sum() / wt.sum() - (wc * y).sum() / wc.sum()
    item["estimate"] = direct
    out = weighting.referee([item], 1e-9)
    assert out["method"] == "independent_direct_hajek_recalculation"
    assert out["tolerance"] == 1e-9
    assert out["passed"] is True
    record = out["periods"][0]
    assert record["period"] == "2"
    assert record["estimate"] == pytest.approx(direct)
    assert record["propensity_range"] == [0.2, 0.8]
    assert record["independent_method"] == "direct_normalized_weighted_means"
    item["estimate"] = direct + 0.1
    off = weighting.referee([item], 0.1 - 1e-9)
    assert off["passed"] is False
    assert off["periods"][0]["absolute_difference"] == pytest.approx(0.1)


def test_a_period_label_is_stable_text():
    assert (
        weighting.periodize(pandas.Timestamp("2026-01-02"))
        == "2026-01-02T00:00:00"
    )
    assert weighting.periodize(datetime.date(2026, 1, 2)) == "2026-01-02"
    assert weighting.periodize(numpy.int64(3)) == "3"
    assert weighting.periodize(numpy.float64(1.5)) == "1.5"
    assert weighting.periodize("q1") == "q1"


def test_an_imprint_changes_with_values_columns_and_index():
    base = pandas.DataFrame({"a": [1, 2]})
    digest = weighting.imprint(base)
    assert len(digest) == 64
    assert digest == weighting.imprint(base.copy())
    assert digest != weighting.imprint(pandas.DataFrame({"a": [1, 3]}))
    assert digest != weighting.imprint(pandas.DataFrame({"b": [1, 2]}))
    assert digest != weighting.imprint(base.set_axis([5, 6]))
    assert digest != weighting.imprint(base.astype(float))
    descriptor = json.dumps(
        {"columns": ["a"], "dtypes": ["int64"], "shape": [2, 1]},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    expected = hashlib.sha256()
    expected.update(len(descriptor).to_bytes(8, "big"))
    expected.update(descriptor)
    expected.update(
        pandas.util.hash_pandas_object(base, index=True).to_numpy().tobytes()
    )
    assert digest == expected.hexdigest()


def test_an_imprint_of_unhashable_values_is_an_input_error():
    with pytest.raises(errors.InputError, match="must be hashable"):
        weighting.imprint(pandas.DataFrame({"a": [[1], [2]]}))


# estimator


def test_an_estimator_needs_an_estimand_and_defaults_its_policy():
    with pytest.raises(errors.KindError, match="estimand must be Estimand"):
        estimator.Estimator("x")
    assert estimator.Estimator(make_estimand()).policy == declaration.Config()


def test_a_fit_reports_each_period_with_covariance_and_reference():
    result = estimator.Estimator(make_estimand()).fit(make_frame())
    assert [p.period for p in result.periods] == ["0", "1", "2"]
    assert result.rows == Counts(180, 180, 0)
    assert result.reference["passed"] is True
    z = stats.norm.ppf(0.975)
    for index, found in enumerate(result.periods):
        assert found.standard_error == pytest.approx(
            math.sqrt(result.covariance[index][index])
        )
        assert found.interval[1] - found.estimate == pytest.approx(
            z * found.standard_error
        )
        assert 0.8 < found.estimate < 2.2
    assert result.diagnostics["period_count"] == 3
    assert result.diagnostics["subject_count"] == 60
    assert set(result.diagnostics["overlap"]) == {"0", "1", "2"}
    assert result.limitations["missingness"]["rows_dropped"] == 0
    assert result.digests == {"frame": weighting.imprint(make_frame())}


def test_a_wave_rejects_thin_periods_and_arms():
    fitter = estimator.Estimator(make_estimand())
    rows = panel.prepare(make_frame(), make_estimand(), declaration.Config())[0]
    first = rows[rows.time == 0]
    with pytest.raises(
        errors.InputError, match="period 0 has insufficient rows: 7 < 8"
    ):
        fitter.wave(first.iloc[:7], 0)
    treated = first[first.treatment == 1].iloc[:9]
    one = pandas.concat([treated, first[first.treatment == 0].iloc[:1]])
    with pytest.raises(
        errors.InputError, match="insufficient treatment support"
    ):
        fitter.wave(one, 0)


def test_a_wave_records_clipping_and_the_subjects():
    fitter = estimator.Estimator(
        make_estimand(), declaration.Config(propensity_clip=0.3)
    )
    rows = panel.prepare(make_frame(), make_estimand(), declaration.Config())[0]
    first = rows[rows.time == 0]
    found = fitter.wave(first, numpy.int64(0))
    assert found["period"] == "0"
    assert found["subjects"] == first["subject_id"].tolist()
    assert found["clipped_propensity"].min() >= 0.3
    assert found["clipped_propensity"].max() <= 0.7
    assert found["fraction_clipped"] == pytest.approx(
        numpy.mean(found["propensity"] != found["clipped_propensity"])
    )
    assert found["fraction_clipped"] > 0


def test_the_summary_and_limitations_name_the_method():
    limits = estimator.qualify(
        declaration.Config(missingness="complete_case"), 4
    )
    assert limits["missingness"] == {
        "policy": "complete_case",
        "rows_dropped": 4,
        "message": "Complete-case deletion is not a missing-at-random proof; "
        "the declared assumptions must cover missingness.",
    }
    assert limits["prior"]["status"] == "not_applicable"
    assert limits["distribution_shift"]["status"] == "not_evaluated"
    result = estimator.Estimator(make_estimand()).fit(make_frame())
    assert result.diagnostics["method"] == (
        "period_specific_covariate_adjusted_hajek_ipw"
    )
    assert result.diagnostics["covariance_method"] == (
        "subject-aligned_influence_function_sandwich"
    )


# report


@pytest.fixture(scope="module")
def fitted():
    return estimator.Estimator(make_estimand()).fit(make_frame())


def test_a_report_round_trips_through_its_dictionary(fitted):
    record = fitted.to_dict()
    assert record["rows"] == {"received": 180, "used": 180, "dropped": 0}
    assert report.Estimation.from_dict(record) == fitted
    assert report.NESTED["digests"] == {"raw_input_fingerprint": "frame"}


@pytest.mark.parametrize(
    ("changes", "error", "message"),
    [
        ({"periods": ()}, errors.InputError, "needs at least one period"),
        ({"covariance": ((1.0,),)}, errors.InputError, "square over periods"),
        ({"rows": (1, 1, 0)}, errors.KindError, "rows must be Counts"),
        (
            {"diagnostics": []},
            errors.KindError,
            "diagnostics must be a mapping",
        ),
        ({"reference": None}, errors.KindError, "reference must be a mapping"),
        (
            {"digests": {}},
            errors.InputError,
            "digests must hold exactly the frame",
        ),
    ],
)
def test_an_invalid_report_is_rejected(fitted, changes, error, message):
    with pytest.raises(error, match=message):
        dataclasses.replace(fitted, **changes)


def test_a_report_rejects_repeated_labels_and_bad_covariance(fitted):
    first = fitted.periods[0]
    with pytest.raises(errors.InputError, match="period labels must be unique"):
        dataclasses.replace(fitted, periods=(first, first, first))
    cov = [list(row) for row in fitted.covariance]
    cov[0][1] += 1.0
    with pytest.raises(errors.InputError, match="must be symmetric"):
        dataclasses.replace(fitted, covariance=tuple(map(tuple, cov)))
    cov[0][1] = math.nan
    with pytest.raises(errors.InputError, match="finite values"):
        dataclasses.replace(fitted, covariance=tuple(map(tuple, cov)))
