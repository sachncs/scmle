from hypothesis import given
from hypothesis import settings
from hypothesis import strategies
import numpy

from scmle.data.framing import costs

HUMAN = (costs.CostSource.HUMAN,) * 4


@settings(max_examples=200, deadline=None)
@given(
    strategies.integers(0, 10_000),
    strategies.floats(0.1, 5.0),
    strategies.floats(0.1, 5.0),
    strategies.floats(-3.0, 3.0),
    strategies.floats(-3.0, 3.0),
)
def test_cost_is_constant_minus_scaled_net_benefit(seed, fp, fn, c00, c11):
    model = costs.DecisionCostModel(
        c00,
        c11 + fn,
        c00 + fp,
        c11,
        0.001,
        0.999,
        HUMAN,
        costs.CostSource.HUMAN,
    )
    rng = numpy.random.default_rng(seed)
    y = (rng.random(30) < 0.4).astype(float)
    a = (rng.random(30) < 0.5).astype(float)
    direct = costs.expected_cost(y, a, model)
    via = costs.cost_constant(y, model) - model.c_fn * costs.net_benefit(
        y, a, model.threshold
    )
    assert abs(direct - via) <= 1e-9


@settings(max_examples=100, deadline=None)
@given(strategies.integers(0, 10_000), strategies.floats(0.01, 0.99))
def test_treat_all_equals_all_ones_rule(seed, t):
    rng = numpy.random.default_rng(seed)
    y = (rng.random(25) < 0.5).astype(float)
    assert (
        abs(
            costs.net_benefit_all(y, t)
            - costs.net_benefit(y, numpy.ones(25), t)
        )
        <= 1e-12
    )
