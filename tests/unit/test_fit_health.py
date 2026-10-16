"""Tests for the fit-health verdict."""

from mml.evaluation import fit_health


def test_a_model_that_memorises_is_overfit():
    result = fit_health.assess(
        [0.99, 0.99, 0.99], [0.70, 0.71, 0.69], baseline_score=0.5
    )

    assert result.verdict == "overfit"
    assert result.retained < 0.5 and "regularisation" in result.note


def test_a_model_that_never_leaves_the_baseline_is_underfit():
    result = fit_health.assess(
        [0.51, 0.52, 0.51], [0.50, 0.51, 0.52], baseline_score=0.5
    )

    assert result.verdict == "underfit"


def test_training_gain_that_carries_over_is_healthy():
    result = fit_health.assess(
        [0.86, 0.87, 0.86], [0.85, 0.84, 0.86], baseline_score=0.5
    )

    assert result.verdict == "healthy" and result.retained > 0.9


def test_a_gain_drowned_by_fold_noise_is_unstable():
    result = fit_health.assess(
        [0.65, 0.66, 0.64], [0.80, 0.52, 0.66], baseline_score=0.5
    )

    assert result.verdict == "unstable"


def test_an_oriented_error_metric_works_the_same_way():
    # RMSE negated: baseline -100; training error 5, validation error 60.
    result = fit_health.assess(
        [-5.0, -6.0, -5.0], [-60.0, -62.0, -58.0], baseline_score=-100.0
    )

    assert result.verdict == "overfit"
    assert result.to_dict()["verdict"] == "overfit"
