"""Tests for the fit-health verdict."""

from mml.contracts import experiments
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


def make_trial(name, scores, train, strategy="default"):
    return experiments.TrialResult(
        "t",
        name,
        experiments.TrialStatus.OK,
        "m",
        fold_scores=scores,
        train_scores=train,
        strategy=strategy,
    )


def test_a_trial_is_judged_against_the_baseline_trial_of_its_run():
    baseline = make_trial("dummy", [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    memoriser = make_trial("tree", [0.7, 0.7, 0.7], [1.0, 1.0, 1.0])

    health = fit_health.assess_trial(memoriser, [baseline, memoriser], True)

    assert health is not None and health.verdict == "overfit"


def test_error_metrics_are_oriented_before_judging():
    baseline = make_trial("dummy", [100.0, 100.0, 100.0], [100.0] * 3)
    overfit = make_trial("tree", [60.0, 62.0, 58.0], [5.0, 6.0, 5.0])

    health = fit_health.assess_trial(overfit, [baseline, overfit], False)

    assert health is not None and health.verdict == "overfit"


def test_without_a_baseline_or_training_scores_there_is_no_verdict():
    alone = make_trial("tree", [0.7, 0.7, 0.7], [1.0, 1.0, 1.0])
    no_train = make_trial("tree", [0.7, 0.7, 0.7], [])
    baseline = make_trial("dummy", [0.5] * 3, [0.5] * 3)

    assert fit_health.assess_trial(alone, [alone], True) is None
    assert fit_health.assess_trial(no_train, [baseline, no_train], True) is None
