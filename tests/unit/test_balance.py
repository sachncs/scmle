"""Permutation balance test of covariates across treatment arms."""

import numpy as np
import pandas as pd
import pytest

from scmle.evaluation import balance


def world(seed, n=400, columns=4, confound=0.0, q=0.5):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, columns))
    logit = confound * x[:, 0]
    probability = 1.0 / (1.0 + np.exp(-(np.log(q / (1 - q)) + logit)))
    arm = (rng.uniform(size=n) < probability).astype(int)
    return pd.DataFrame(x, columns=[f"x{k}" for k in range(columns)]), arm


def test_randomized_assignment_is_flagged_at_most_at_the_level():
    # Pre-registered before the first run: the permutation test is exact
    # for a randomized arm, so over 1000 worlds at level 0.05 the rate stays
    # within three standard errors of 0.05.
    worlds = 1000
    flagged = sum(
        not balance.balance_test(*world(seed), seed=seed)["balanced"]
        for seed in range(worlds)
    )
    assert flagged / worlds <= 0.05 + 3 * np.sqrt(0.05 * 0.95 / worlds)


def test_assignment_that_follows_a_covariate_is_flagged():
    # Pre-registered: at least 90 percent of 200 confounded worlds.
    found = sum(
        not balance.balance_test(*world(seed, confound=1.0), seed=seed)[
            "balanced"
        ]
        for seed in range(200)
    )
    assert found / 200 >= 0.90


def test_the_result_carries_its_statistic_and_levels():
    features, arm = world(0)
    found = balance.balance_test(features, arm, seed=0, permutations=99)
    assert found["permutations"] == 99 and found["level"] == 0.05
    assert 1 / 100 <= found["p_value"] <= 1.0
    assert found["covariates"] == 4
    assert found["statistic"] >= 0.0
    assert found["balanced"] == (found["p_value"] > 0.05)


def test_same_seed_same_result_and_constant_columns_are_dropped():
    features, arm = world(1)
    features["flat"] = 3.0
    first = balance.balance_test(features, arm, seed=5)
    assert first == balance.balance_test(features, arm, seed=5)
    assert first["covariates"] == 4


def test_missing_values_and_text_columns_are_handled():
    features, arm = world(2)
    features.loc[::7, "x1"] = np.nan
    features["label"] = "a"
    found = balance.balance_test(features, arm, seed=0)
    assert found["covariates"] == 4


def test_no_numeric_covariate_or_one_arm_cannot_be_tested():
    features = pd.DataFrame({"label": ["a", "b", "a", "b"]})
    none = balance.balance_test(features, np.array([0, 1, 0, 1]))
    assert none["balanced"] is None and none["p_value"] is None
    features, arm = world(3)
    single = balance.balance_test(features, np.zeros_like(arm))
    assert single["balanced"] is None


def test_inputs_are_checked():
    features, arm = world(4)
    with pytest.raises(ValueError):
        balance.balance_test(features, arm[:-1])
    with pytest.raises(ValueError):
        balance.balance_test(features, arm, permutations=0)
    with pytest.raises(ValueError):
        balance.balance_test(features, arm, level=1.5)
