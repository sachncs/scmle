"""AIPW nuisance models fitted on all rows or cross-fitted by fold."""

import numpy
import pytest
from scipy import special

from scmle.contracts import errors
from scmle.evaluation.parameter_estimation import config
from scmle.evaluation.parameter_estimation import crossfitting
from scmle.evaluation.parameter_estimation import linear
from scmle.evaluation.parameter_estimation import weighting

CAUSAL = ("consistency", "conditional_exchangeability", "positivity")


def policy(folds=1, clip=0.01):
    return config.Config(
        estimator="aipw_ate",
        covariance="iid",
        treatment_position=0,
        causal_assumptions=CAUSAL,
        cross_fitting_folds=folds,
        propensity_clip=clip,
    )


def world(rows=120, seed=0):
    rng = numpy.random.default_rng(seed)
    x = rng.normal(size=rows)
    design = numpy.column_stack([numpy.ones(rows), x])
    t = (rng.random(rows) < special.expit(0.5 * x)).astype(float)
    y = 1 + 2 * t + x + rng.normal(scale=0.3, size=rows)
    return design, t, y


def test_an_interleave_assigns_rows_modulo_the_fold_count():
    assert crossfitting.interleave(5, 2).tolist() == [0, 1, 0, 1, 0]
    assert crossfitting.interleave(3, 3).tolist() == [0, 1, 2]
    with pytest.raises(errors.InputError, match="at least two folds"):
        crossfitting.interleave(5, 1)
    with pytest.raises(errors.InputError, match="must not exceed rows"):
        crossfitting.interleave(3, 4)


def test_a_support_check_counts_each_fold():
    treatment = numpy.array([1, 0, 1, 0, 1, 0, 1, 0.0])
    folds = crossfitting.interleave(8, 2)
    shifted = numpy.array([0, 0, 1, 1, 0, 0, 1, 1])
    support = crossfitting.validate_cross_fitting_support(treatment, shifted, 2)
    assert support[0] == {
        "fold": 0,
        "training_treated_rows": 2,
        "training_control_rows": 2,
        "validation_treated_rows": 2,
        "validation_control_rows": 2,
    }
    with pytest.raises(
        errors.InputError, match="both treatment arms in every validation fold"
    ):
        crossfitting.validate_cross_fitting_support(treatment, folds, 2)
    thin = numpy.array([1, 0, 1, 0, 0, 0.0])
    with pytest.raises(
        errors.InputError,
        match="at least two rows per treatment arm in every training",
    ):
        crossfitting.validate_cross_fitting_support(
            thin, crossfitting.interleave(6, 3), 3
        )


def test_the_caveats_add_the_same_row_scoring_note_without_cross_fitting():
    crossfit = crossfitting.caveats(crossfit=True)
    assert len(crossfit) == 4
    assert (
        crossfit[0]
        == "propensity and outcome nuisance uncertainty is not propagated"
    )
    assert crossfitting.caveats(crossfit=False) == [
        *crossfit,
        "nuisance models are fitted on the same rows used for scoring",
    ]


def test_propensities_are_the_clipped_logistic_scores():
    design, t, ignored = world()
    clipped, raw, diagnostics = crossfitting.propensities(
        design, t, design[:5], policy(clip=0.45)
    )
    coefficients, ignored, expected = linear.logit(design, t, 100, 1e-8)
    numpy.testing.assert_allclose(raw, special.expit(design[:5] @ coefficients))
    numpy.testing.assert_allclose(clipped, numpy.clip(raw, 0.45, 0.55))
    assert diagnostics == expected


def test_a_nuisance_fit_needs_two_rows_per_arm():
    design, ignored, y = world(rows=6)
    with pytest.raises(
        errors.InputError, match="each training fold needs at least two rows"
    ):
        crossfitting.fit_aipw_nuisance_models(
            design, numpy.array([1, 0, 0, 0, 0, 0.0]), y, design, policy()
        )


def test_a_nuisance_fit_predicts_both_outcomes_and_reports_ranks():
    design, t, y = world()
    propensity, raw, control, treated, diagnostics = (
        crossfitting.fit_aipw_nuisance_models(design, t, y, design, policy())
    )
    expected_control, expected_treated = weighting.outcomes(
        design, y, (t == 1, t == 0), design, "x"
    )
    numpy.testing.assert_allclose(control, expected_control)
    numpy.testing.assert_allclose(treated, expected_treated)
    assert diagnostics["treated_rank"] == 2
    assert diagnostics["control_rank"] == 2
    assert propensity.shape == raw.shape == (120,)


def test_a_single_fit_nuisance_records_disabled_cross_fitting():
    design, t, y = world()
    fitted = crossfitting.nuisance(design, t, y, policy())
    assert fitted.folds.tolist() == [0] * 120
    assert fitted.crossfitting == {
        "status": "disabled",
        "fold_count": 1,
        "assignment_method": "not_applicable",
        "limitations": crossfitting.caveats(crossfit=False),
    }
    records = fitted.records()
    assert records["outcome_models"] == {
        "method": "separate_ordinary_least_squares",
        "treated_rank": 2,
        "control_rank": 2,
    }
    assert records["propensity_model"]["log_likelihood"] == pytest.approx(
        fitted.diagnostics["propensity_log_likelihood"]
    )


def test_a_cross_fitted_nuisance_scores_each_row_from_its_held_out_fold():
    design, t, y = world()
    fitted = crossfitting.nuisance(design, t, y, policy(folds=3))
    assert fitted.crossfitting["status"] == "enabled"
    assert fitted.crossfitting["assignment_method"] == "row_index_modulo"
    assert fitted.crossfitting["training_rows_by_fold"] == {
        "0": 80,
        "1": 80,
        "2": 80,
    }
    assert fitted.crossfitting["validation_rows_by_fold"] == {
        "0": 40,
        "1": 40,
        "2": 40,
    }
    validation = fitted.folds == 1
    expected = crossfitting.fit_aipw_nuisance_models(
        design[~validation],
        t[~validation],
        y[~validation],
        design[validation],
        policy(folds=3),
    )
    numpy.testing.assert_allclose(fitted.propensity[validation], expected[0])
    numpy.testing.assert_allclose(fitted.treated[validation], expected[3])
    records = fitted.records()
    assert records["outcome_models"]["treated_rank"] == "recorded_per_fold"
    assert records["propensity_model"]["log_likelihood"] == "recorded_per_fold"
    assert fitted.diagnostics["propensity_iterations"] == max(
        f["propensity_iterations"] for f in fitted.diagnostics["folds"]
    )
    assert [f["validation_rows"] for f in fitted.diagnostics["folds"]] == [
        40,
        40,
        40,
    ]
