"""Off-policy evaluation: unbiasedness, gates and policy fitting."""

import numpy as np
import pytest

from scmle.decision import ope


def bandit(n, seed, eps=0.5, shift=0.0):
    """Two actions; true value of the target is known analytically."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 1))
    mu1 = np.clip(0.5 + eps * np.tanh(x[:, 0]) * 0.5 + shift, 0.05, 0.95)
    behaviour = np.column_stack([1 - mu1, mu1])
    action = (rng.random(n) < mu1).astype(int)
    mean = np.column_stack([0.3 + 0 * x[:, 0], 0.3 + 0.2 * (x[:, 0] > 0)])
    reward = (rng.random(n) < mean[np.arange(n), action]).astype(float)
    treat = (x[:, 0] > 0).astype(float)
    target = np.column_stack([1 - treat, treat])
    truth = float(np.mean((target * mean).sum(axis=1)))
    return x, action, reward, behaviour, target, mean, truth


def build(n, seed, **kwargs):
    parts = bandit(n, seed, **kwargs)
    ignored, action, reward, behaviour, target, mean, truth = parts
    log = ope.LoggedBandit(action, reward, behaviour, target, mean)
    return log, truth


def test_all_three_estimators_are_close_to_the_known_value():
    log, truth = build(60000, 0)
    for name in ope.ESTIMATORS:
        result = ope.estimate_value(log, name)
        assert result.status == ope.ISSUED and result.tier == "confirmed"
        assert result.value == pytest.approx(truth, abs=0.01)
        low, high = result.interval
        assert low < truth < high


def test_the_doubly_robust_value_is_exact_with_a_perfect_outcome_model():
    log, truth = build(2000, 1)
    perfect = ope.estimate_value(log, "dr")
    assert perfect.value == pytest.approx(truth, abs=0.03)
    wrong = ope.LoggedBandit(
        log.actions, log.rewards, log.behaviour, log.target, np.zeros((2000, 2))
    )
    # a wrong q falls back on the importance weights and stays unbiased
    assert ope.estimate_value(wrong, "dr").value == pytest.approx(
        ope.estimate_value(wrong, "ipw").value
    )


def test_a_support_violation_is_refused_with_the_reason():
    log, ignored = build(3000, 2)
    behaviour = log.behaviour.copy()
    behaviour[:, 1] = 0.0
    behaviour[:, 0] = 1.0
    bad = ope.LoggedBandit(
        np.zeros(3000, dtype=int),
        log.rewards,
        behaviour,
        log.target,
        log.outcome_model,
    )
    result = ope.estimate_value(bad, "dr")
    assert result.status == ope.REFUSED and result.value is None
    assert any("support" in r for r in result.reasons)


def test_a_low_effective_sample_size_is_refused_and_clipping_is_labelled():
    log, ignored = build(4000, 3, eps=1.9, shift=-0.45)
    strict = ope.estimate_value(log, "ipw", ess_floor=0.95)
    assert strict.status == ope.REFUSED
    assert any("effective sample size" in r for r in strict.reasons)
    clipped = ope.estimate_value(log, "ipw", clip=3.0, ess_floor=0.01)
    assert clipped.clipped and clipped.clipped_share > 0
    assert any("clipped" in r for r in clipped.reasons)


def test_observational_logs_are_not_confirmed():
    log, ignored = build(2000, 4)
    observational = ope.LoggedBandit(
        log.actions,
        log.rewards,
        log.behaviour,
        log.target,
        log.outcome_model,
        randomized=False,
    )
    assert ope.estimate_value(observational).tier == "unverified"
    assert ope.estimate_value(observational, eligible_strategy=True).tier == (
        "attested"
    )


def test_a_difference_uses_the_paired_influence_function():
    log = build(60000, 5)[0]
    never = np.column_stack([np.ones(60000), np.zeros(60000)])
    diff = ope.difference_value(log, never, "dr")
    single = ope.estimate_value(log, "dr")
    base = ope.estimate_value(
        ope.LoggedBandit(
            log.actions, log.rewards, log.behaviour, never, log.outcome_model
        ),
        "dr",
    )
    assert diff.value == pytest.approx(single.value - base.value, abs=1e-9)
    assert diff.value == pytest.approx(0.1, abs=0.02)
    assert diff.interval[0] > 0


def test_clustered_intervals_are_wider_when_clusters_share_noise():
    log, ignored = build(3000, 6)
    labels = np.repeat(np.arange(30), 100)
    shared = log.rewards + np.repeat(
        np.random.default_rng(0).normal(0, 0.3, 30), 100
    )
    base = ope.LoggedBandit(
        log.actions, shared, log.behaviour, log.target, log.outcome_model
    )
    grouped = ope.LoggedBandit(
        log.actions,
        shared,
        log.behaviour,
        log.target,
        log.outcome_model,
        clusters=labels,
    )
    a = ope.estimate_value(base).interval
    b = ope.estimate_value(grouped).interval
    assert b[1] - b[0] > a[1] - a[0]


def test_outcome_model_and_threshold_policy_fitting():
    x, action, reward, behaviour, ignored, ignored, ignored = bandit(8000, 7)
    predict = ope.fit_outcome_model(x, action, reward, 2)
    q = predict(x)
    assert q.shape == (8000, 2)
    log = ope.LoggedBandit(action, reward, behaviour, behaviour, q)
    tau = x[:, 0]
    policy = ope.fit_threshold_policy(log, tau, penalty=0.03)
    assert abs(policy.cut) < 0.6
    probs = policy.probabilities(tau)
    assert probs.shape == (8000, 2) and np.allclose(probs.sum(axis=1), 1)
    costly = ope.fit_threshold_policy(log, tau, penalty=5.0)
    assert costly.probabilities(tau)[:, 1].sum() == 0
    with pytest.raises(ValueError):
        ope.fit_outcome_model(x, np.zeros(8000, dtype=int), reward, 2)


@pytest.mark.parametrize(
    "build_log",
    [
        lambda: ope.LoggedBandit(
            np.array([0, 2]),
            np.zeros(2),
            np.full((2, 2), 0.5),
            np.full((2, 2), 0.5),
        ),
        lambda: ope.LoggedBandit(
            np.array([0, 1]),
            np.zeros(2),
            np.full((2, 2), 0.7),
            np.full((2, 2), 0.5),
        ),
        lambda: ope.LoggedBandit(
            np.array([0, 1]),
            np.zeros(3),
            np.full((2, 2), 0.5),
            np.full((2, 2), 0.5),
        ),
    ],
)
def test_bad_logs_are_refused(build_log):
    with pytest.raises(ValueError):
        build_log()


def test_estimator_and_level_are_validated():
    log, ignored = build(500, 8)
    with pytest.raises(ValueError):
        ope.estimate_value(log, "magic")
    with pytest.raises(ValueError):
        ope.estimate_value(log, "dr", level=1.0)
    bare = ope.LoggedBandit(log.actions, log.rewards, log.behaviour, log.target)
    with pytest.raises(ValueError, match="outcome model"):
        ope.estimate_value(bare, "dr")
