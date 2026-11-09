"""Uplift metrics and meta-learners."""

import types

import numpy
import pandas
import pytest

from scmle.contracts import models
from scmle.contracts import tasks
from scmle.evaluation import uplift
from scmle.models.uplift import adapters
from scmle.models.uplift import estimators


def experiment(rows=4000, seed=0):
    """A randomized trial whose effect depends on the first covariate."""
    rng = numpy.random.default_rng(seed)
    x = rng.normal(size=(rows, 3))
    arm = rng.integers(0, 2, rows)
    effect = 0.35 * (x[:, 0] > 0)
    outcome = (rng.random(rows) < 0.25 + arm * effect).astype(int)
    return types.SimpleNamespace(x=x, arm=arm, outcome=outcome, effect=effect)


def test_a_perfect_ranking_has_a_positive_qini_and_a_random_one_does_not():
    d = experiment()
    random_score = numpy.random.default_rng(1).normal(size=len(d.arm))

    good = uplift.qini_coefficient(d.outcome, d.arm, d.effect)
    chance = uplift.qini_coefficient(d.outcome, d.arm, random_score)

    assert good > 0.02
    assert abs(chance) < good / 3


def test_uplift_at_k_compares_treated_and_control_in_the_top_share():
    d = experiment()

    top = uplift.uplift_at(d.outcome, d.arm, d.effect, share=0.25)
    overall = uplift.average_effect(d.outcome, d.arm)

    assert top > overall + 0.1


@pytest.mark.parametrize(
    "learner",
    [
        estimators.TwoModelLearner,
        estimators.SingleModelLearner,
        estimators.CrossLearner,
    ],
)
def test_meta_learners_recover_who_responds(learner):
    d = experiment()

    model = learner(classify=True, random_state=0).fit(
        d.x[:3000], d.outcome[:3000], d.arm[:3000]
    )
    estimated = model.predict(d.x[3000:])

    assert numpy.corrcoef(estimated, d.effect[3000:])[0, 1] > 0.5


def test_the_zero_baseline_has_no_signal():
    d = experiment()

    scores = estimators.ZeroEffect().fit(d.x, d.outcome, d.arm).predict(d.x)

    assert not scores.any()


def test_randomized_treatment_looks_randomized():
    d = experiment()

    found = uplift.check_propensity(pandas.DataFrame(d.x), d.arm)

    assert found.randomized and found.auc < 0.56
    assert found.balance_p_value is not None and found.balance_p_value > 0.05
    assert "consistent with randomization" in found.note


def test_assignment_driven_by_covariates_is_flagged():
    d = experiment()
    noise = numpy.random.default_rng(2).normal(0, 0.3, len(d.arm))
    chosen = ((d.x[:, 1] + noise) > 0).astype(int)

    found = uplift.check_propensity(pandas.DataFrame(d.x), chosen)

    assert not found.randomized and found.auc > 0.8
    assert found.balance_p_value is not None and found.balance_p_value <= 0.05
    assert "confounded" in found.note


def test_a_tiny_arm_is_reported_not_modelled():
    d = experiment()
    sparse = (numpy.arange(len(d.arm)) < 20).astype(int)

    found = uplift.check_propensity(pandas.DataFrame(d.x), sparse)

    assert not found.randomized and numpy.isnan(found.auc)
    assert found.balance_p_value is None


def test_average_effect_is_nan_without_warning_when_an_arm_is_empty():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert numpy.isnan(uplift.average_effect(numpy.ones(4), numpy.ones(4)))
        assert numpy.isnan(uplift.average_effect(numpy.ones(4), numpy.zeros(4)))


class ClassStyleUplift(adapters.Uplift):
    """A subclass declaring its data as class attributes (the old style)."""

    name = "class_style_uplift"
    capabilities = models.Capabilities(adapters.UPLIFT)
    learner = estimators.SingleModelLearner


def test_configured_instances_and_class_style_subclasses_build_learners():
    built = [
        (adapter.name, type(adapter.build(tasks.Task.UPLIFT, 0)))
        for adapter in adapters.UPLIFT_ADAPTERS
    ]
    assert built == [
        ("uplift_t_learner", estimators.TwoModelLearner),
        ("uplift_s_learner", estimators.SingleModelLearner),
        ("uplift_x_learner", estimators.CrossLearner),
    ]
    legacy = ClassStyleUplift()
    assert (legacy.name, legacy.family) == ("class_style_uplift", "causal")
    assert isinstance(
        legacy.build(tasks.Task.UPLIFT, 0), estimators.SingleModelLearner
    )
    assert adapters.Uplift.learner is estimators.TwoModelLearner
