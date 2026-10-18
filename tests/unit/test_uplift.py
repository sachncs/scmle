"""Uplift metrics and meta-learners."""

import types

import numpy as np
import pandas as pd
import pytest

from scml.evaluation import uplift
from scml.models.uplift import estimators


def experiment(rows=4000, seed=0):
    """A randomized trial whose effect depends on the first covariate."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(rows, 3))
    arm = rng.integers(0, 2, rows)
    effect = 0.35 * (x[:, 0] > 0)
    outcome = (rng.random(rows) < 0.25 + arm * effect).astype(int)
    return types.SimpleNamespace(x=x, arm=arm, outcome=outcome, effect=effect)


def test_a_perfect_ranking_has_a_positive_qini_and_a_random_one_does_not():
    d = experiment()
    random_score = np.random.default_rng(1).normal(size=len(d.arm))

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

    assert np.corrcoef(estimated, d.effect[3000:])[0, 1] > 0.5


def test_the_zero_baseline_has_no_signal():
    d = experiment()

    scores = estimators.ZeroEffect().fit(d.x, d.outcome, d.arm).predict(d.x)

    assert not scores.any()


def test_randomized_treatment_looks_randomized():
    d = experiment()

    found = uplift.check_propensity(pd.DataFrame(d.x), d.arm)

    assert found.randomized and found.auc < 0.56
    assert "consistent with randomization" in found.note


def test_assignment_driven_by_covariates_is_flagged():
    d = experiment()
    noise = np.random.default_rng(2).normal(0, 0.3, len(d.arm))
    chosen = ((d.x[:, 1] + noise) > 0).astype(int)

    found = uplift.check_propensity(pd.DataFrame(d.x), chosen)

    assert not found.randomized and found.auc > 0.8
    assert "confounded" in found.note


def test_a_tiny_arm_is_reported_not_modelled():
    d = experiment()
    sparse = (np.arange(len(d.arm)) < 20).astype(int)

    found = uplift.check_propensity(pd.DataFrame(d.x), sparse)

    assert not found.randomized and np.isnan(found.auc)
