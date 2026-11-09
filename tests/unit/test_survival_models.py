"""Survival estimators and the concordance index."""

import numpy
import pytest

from scmle.evaluation import concordance
from scmle.models.survival import estimators


def cohort(rows=1200, seed=0):
    rng = numpy.random.default_rng(seed)
    features = rng.normal(size=(rows, 3))
    rate = numpy.exp(1.0 * features[:, 0] - 0.7 * features[:, 1])
    happens = rng.exponential(1 / rate)
    stops = rng.exponential(1.5, rows)
    return (
        features,
        numpy.minimum(happens, stops),
        (happens <= stops).astype(int),
    )


def test_concordance_is_one_for_a_perfect_ordering_and_half_for_ties():
    time = numpy.array([1.0, 2.0, 3.0, 4.0])
    event = numpy.ones(4)

    assert concordance.concordance_index(time, event, -time) == 1.0
    assert concordance.concordance_index(time, event, time) == 0.0
    assert concordance.concordance_index(time, event, numpy.zeros(4)) == 0.5


def test_pairs_whose_earlier_row_was_censored_are_not_comparable():
    time = numpy.array([1.0, 2.0])
    event = numpy.array([0, 1])

    value = concordance.concordance_index(time, event, numpy.array([1.0, 0.0]))

    assert numpy.isnan(value)


def test_kaplan_meier_matches_a_hand_computation():
    moments, curve = estimators.kaplan_meier(
        numpy.array([1.0, 2.0, 3.0, 4.0]), numpy.array([1, 0, 1, 1])
    )

    assert list(moments) == [1.0, 3.0, 4.0]
    assert curve == pytest.approx([0.75, 0.375, 0.0])


@pytest.mark.parametrize(
    "model",
    [estimators.CoxRegressor(), estimators.DiscreteTimeBoosting()],
    ids=["cox", "discrete_time"],
)
def test_models_order_held_out_risk_and_give_falling_curves(model):
    features, time, event = cohort()

    model.fit(features[:900], time[:900], event[:900])
    risk = model.predict(features[900:])
    curves = model.predict_survival(features[900:910], [0.1, 0.5, 1.0, 3.0])

    assert concordance.concordance_index(time[900:], event[900:], risk) > 0.7
    assert curves.shape == (10, 4)
    assert (numpy.diff(curves, axis=1) <= 1e-9).all()
    assert ((curves >= 0) & (curves <= 1)).all()


def test_the_cox_model_recovers_the_sign_of_each_effect():
    features, time, event = cohort(3000)

    model = estimators.CoxRegressor(penalty=0.1).fit(features, time, event)

    assert model.coefficients_[0] > 0.7
    assert model.coefficients_[1] < -0.4
    assert abs(model.coefficients_[2]) < 0.15


def test_the_marginal_model_cannot_rank_rows():
    features, time, event = cohort()

    model = estimators.MarginalSurvival().fit(features, time, event)

    assert (
        concordance.concordance_index(time, event, model.predict(features))
        == 0.5
    )
