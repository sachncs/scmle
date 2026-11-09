"""Properties every online rule must keep for every evidence sequence."""

from fractions import Fraction

from hypothesis import given
from hypothesis import HealthCheck
from hypothesis import settings
from hypothesis import strategies
import pytest

from scmle.data.ledger import online_fdr

SETTINGS = settings(
    max_examples=80,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)

alphas = strategies.fractions(
    min_value=Fraction(1, 100), max_value=Fraction(1, 2), max_denominator=100
)
horizons = strategies.one_of(
    strategies.none(), strategies.integers(min_value=1, max_value=12)
)
pvalues = strategies.lists(
    strategies.one_of(
        strategies.just(0.0),
        strategies.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    ),
    min_size=1,
    max_size=25,
)
RULES = strategies.sampled_from(list(online_fdr.RULE_NAMES))


def config(rule, alpha, horizon):
    return online_fdr.RuleConfig(rule, alpha, horizon=horizon)


def evidence(values):
    return [online_fdr.Evidence(p=v) for v in values]


@SETTINGS
@given(RULES, alphas, horizons, pvalues)
def test_wealth_is_never_negative_and_levels_stay_in_range(
    rule, alpha, horizon, values
):
    steps = online_fdr.replay(config(rule, alpha, horizon), evidence(values))
    for step in steps:
        assert step.wealth_after >= 0
        assert step.wealth_before >= 0
        assert 0 <= step.level <= 1


@SETTINGS
@given(RULES, alphas, horizons, pvalues)
def test_replay_is_deterministic_and_matches_the_dict_round_trip(
    rule, alpha, horizon, values
):
    first = config(rule, alpha, horizon)
    second = online_fdr.RuleConfig.from_dict(first.to_dict())
    assert online_fdr.replay(first, evidence(values)) == online_fdr.replay(
        second, evidence(values)
    )


@SETTINGS
@given(
    strategies.sampled_from([online_fdr.ELOND, online_fdr.LORD_PLUS_PLUS]),
    alphas,
    horizons,
    pvalues,
    strategies.data(),
)
def test_smaller_p_values_never_lose_rejections(
    rule, alpha, horizon, values, data
):
    # Monotone rules: lowering any p-value cannot remove a rejection.
    index = data.draw(strategies.integers(0, len(values) - 1))
    lower = data.draw(strategies.floats(0.0, values[index], allow_nan=False))
    changed = [*values[:index], lower, *values[index + 1 :]]
    cfg = config(rule, alpha, horizon)
    before = [s.rejected for s in online_fdr.replay(cfg, evidence(values))]
    after = [s.rejected for s in online_fdr.replay(cfg, evidence(changed))]
    assert all(a or not b for a, b in zip(after, before, strict=True))


@SETTINGS
@given(alphas, horizons, pvalues)
def test_elond_never_rejects_more_than_it_has_levels_for(
    alpha, horizon, values
):
    steps = online_fdr.replay(
        config(online_fdr.ELOND, alpha, horizon), evidence(values)
    )
    rejected = sum(s.rejected for s in steps)
    assert rejected <= len(values)
    # Without a rejection the level only follows the discount sequence.
    quiet = online_fdr.replay(
        config(online_fdr.ELOND, alpha, horizon),
        [online_fdr.Evidence(e=0.0)] * len(values),
    )
    assert not any(s.rejected for s in quiet)


@SETTINGS
@given(alphas, pvalues)
def test_saffron_never_rejects_above_lambda(alpha, values):
    steps = online_fdr.replay(
        config(online_fdr.SAFFRON, alpha, None), evidence(values)
    )
    for step, value in zip(steps, values, strict=True):
        if value > 0.5:
            assert not step.rejected
            assert not step.candidate


@pytest.mark.parametrize("rule", online_fdr.RULE_NAMES)
def test_all_ones_never_reject(rule):
    steps = online_fdr.replay(
        config(rule, Fraction(1, 20), None), evidence([1.0] * 30)
    )
    assert not any(s.rejected for s in steps)
