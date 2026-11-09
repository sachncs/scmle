"""The simulator keeps real features and plants (or removes) known effects."""

import math

import numpy
import pandas
import pytest
from scipy import stats
from sklearn.datasets import load_diabetes

from benchmarks.qualification import simulate


def real_features() -> pandas.DataFrame:
    """A real feature matrix that ships with scikit-learn (no download)."""
    return load_diabetes(as_frame=True).data


def correlated(rows: int = 6000, seed: int = 3) -> pandas.DataFrame:
    rng = numpy.random.default_rng(seed)
    base = rng.normal(size=(rows, 1))
    noise = rng.normal(size=(rows, 4))
    values = 0.6 * base + 0.8 * noise
    return pandas.DataFrame(values, columns=["a", "b", "c", "d"])


def slope(world: simulate.World, column: str) -> float:
    z = (world.features - world.features.mean()) / world.features.std()
    design = numpy.column_stack([numpy.ones(len(z)), z.to_numpy()])
    solution = numpy.linalg.lstsq(
        design, world.outcome.to_numpy(float), rcond=None
    )[0]
    return float(solution[1 + list(z.columns).index(column)])


def test_a_planted_world_keeps_the_features_and_recovers_the_effects():
    features = correlated()
    world = simulate.planted_world(features, {"a": 0.5, "c": -0.3}, seed=1)
    pandas.testing.assert_frame_equal(world.features, features)
    assert world.kind == "alternative"
    assert world.signal_columns == ("a", "c")
    assert world.null_columns == ("b", "d")
    assert slope(world, "a") == pytest.approx(0.5, abs=0.05)
    assert slope(world, "c") == pytest.approx(-0.3, abs=0.05)
    assert abs(slope(world, "b")) < 0.05
    assert abs(slope(world, "d")) < 0.05


def test_the_noise_level_scales_the_outcome_spread():
    features = correlated()
    quiet = simulate.planted_world(features, {"a": 1.0}, seed=1, noise_sd=0.5)
    loud = simulate.planted_world(features, {"a": 1.0}, seed=1, noise_sd=2.0)
    assert loud.outcome.std() > quiet.outcome.std()


def test_a_binary_world_draws_bernoulli_outcomes_from_the_logit():
    features = correlated(rows=8000)
    world = simulate.planted_world(features, {"a": 1.5}, seed=2, task="binary")
    assert set(world.outcome.unique()) == {0, 1}
    z = (features["a"] - features["a"].mean()) / features["a"].std()
    expected = float((1.0 / (1.0 + numpy.exp(-1.5 * z))).mean())
    assert float(world.outcome.mean()) == pytest.approx(expected, abs=0.02)
    assert world.outcome[z > 1].mean() > world.outcome[z < -1].mean() + 0.4


def test_worlds_are_reproducible_and_seed_dependent():
    features = correlated(rows=500)
    first = simulate.planted_world(features, {"a": 0.5}, seed=4)
    again = simulate.planted_world(features, {"a": 0.5}, seed=4)
    other = simulate.planted_world(features, {"a": 0.5}, seed=5)
    pandas.testing.assert_series_equal(first.outcome, again.outcome)
    assert not first.outcome.equals(other.outcome)
    assert first.seed_path[0] == 4


@pytest.mark.parametrize(
    "call",
    [
        lambda f: simulate.planted_world(f, {}, seed=1),
        lambda f: simulate.planted_world(f, {"a": 0.0}, seed=1),
        lambda f: simulate.planted_world(f, {"zzz": 0.5}, seed=1),
        lambda f: simulate.planted_world(f, {"a": 0.5}, seed=1, task="other"),
        lambda f: simulate.planted_world(f, {"a": 0.5}, seed=1, noise_sd=0.0),
        lambda f: simulate.planted_world(
            f.assign(a=numpy.nan), {"b": 0.5}, seed=1
        ),
        lambda f: simulate.planted_world(f.assign(s="x"), {"b": 0.5}, seed=1),
        lambda f: simulate.planted_world(f.assign(a=1.0), {"b": 0.5}, seed=1),
    ],
)
def test_invalid_worlds_are_refused(call):
    with pytest.raises(ValueError):
        call(correlated(rows=200))


def test_random_effects_pick_signal_columns_with_a_fixed_size():
    columns = list("abcdefgh")
    effects = simulate.random_effects(columns, 3, 0.4, seed=7)
    assert len(effects) == 3
    assert set(effects) <= set(columns)
    assert all(abs(value) == pytest.approx(0.4) for value in effects.values())
    assert effects == simulate.random_effects(columns, 3, 0.4, seed=7)
    with pytest.raises(ValueError):
        simulate.random_effects(columns, 0, 0.4, seed=1)
    with pytest.raises(ValueError):
        simulate.random_effects(columns, 9, 0.4, seed=1)
    with pytest.raises(ValueError):
        simulate.random_effects(columns, 2, 0.0, seed=1)


def test_a_null_world_keeps_features_and_the_outcome_multiset():
    features = real_features()
    outcome = load_diabetes(as_frame=True).target
    world = simulate.null_world(features, outcome, seed=1)
    assert world.kind == "null"
    assert world.signal_columns == ()
    pandas.testing.assert_frame_equal(world.features, features)
    pandas.testing.assert_frame_equal(
        world.features.corr(), features.corr(), check_exact=True
    )
    assert numpy.array_equal(numpy.sort(world.outcome), numpy.sort(outcome))
    assert not world.outcome.equals(outcome)
    bad = stats.pearsonr(features["bmi"], outcome).statistic
    kept = stats.pearsonr(features["bmi"], world.outcome).statistic
    assert abs(kept) < abs(bad) / 3


def test_null_worlds_can_permute_within_groups_and_time_blocks():
    features = correlated(rows=300)
    outcome = pandas.Series(numpy.arange(300, dtype=float))
    groups = pandas.Series(numpy.repeat(numpy.arange(30), 10))
    world = simulate.null_world(features, outcome, seed=2, groups=groups)
    for rows in groups.groupby(groups).groups.values():
        assert sorted(world.outcome.loc[rows]) == sorted(outcome.loc[rows])
    assert not world.outcome.equals(outcome)

    blocked = simulate.null_world(features, outcome, seed=2, block_size=50)
    for start in range(0, 300, 50):
        window = slice(start, start + 50)
        assert sorted(blocked.outcome.iloc[window]) == sorted(
            outcome.iloc[window]
        )
    with pytest.raises(ValueError):
        simulate.null_world(
            features, outcome, seed=2, groups=groups, block_size=50
        )
    with pytest.raises(ValueError):
        simulate.null_world(features, outcome, seed=2, block_size=1)
    with pytest.raises(ValueError):
        simulate.null_world(features, outcome.head(10), seed=2)


def test_permuting_within_groups_keeps_a_group_level_association():
    rng = numpy.random.default_rng(0)
    groups = pandas.Series(numpy.repeat(numpy.arange(60), 20))
    level = rng.normal(size=60)[groups]
    features = pandas.DataFrame({"x": level + 0.3 * rng.normal(size=1200)})
    outcome = pandas.Series(level + 0.3 * rng.normal(size=1200))
    original = stats.pearsonr(features["x"], outcome).statistic
    within = simulate.null_world(features, outcome, seed=1, groups=groups)
    globally = simulate.null_world(features, outcome, seed=1)
    kept = stats.pearsonr(features["x"], within.outcome).statistic
    broken = stats.pearsonr(features["x"], globally.outcome).statistic
    assert original > 0.8
    assert kept > 0.7
    assert abs(broken) < 0.1


def test_the_empirical_size_of_a_valid_test_is_inside_the_band():
    features = real_features()
    outcome = load_diabetes(as_frame=True).target
    worlds = simulate.null_worlds(features, outcome, runs=2000, seed=0)

    def test(world: simulate.World) -> float:
        return float(
            stats.pearsonr(world.features["bmi"], world.outcome).pvalue
        )

    result = simulate.rejection_rate(test, worlds, alpha=0.05)
    assert result.runs == 2000
    assert result.within_size_band
    low, high = simulate.size_band(0.05, 2000)
    assert low == pytest.approx(0.05 - 2 * math.sqrt(0.05 * 0.95 / 2000))
    assert high == pytest.approx(0.05 + 2 * math.sqrt(0.05 * 0.95 / 2000))
    assert result.interval[0] <= result.rate <= result.interval[1]


def test_a_biased_test_falls_outside_the_size_band():
    features = real_features()
    outcome = load_diabetes(as_frame=True).target
    worlds = simulate.null_worlds(features, outcome, runs=1000, seed=0)

    def liberal(world: simulate.World) -> float:
        # Treats every p-value as if it were half its size.
        return float(
            stats.pearsonr(world.features["bmi"], world.outcome).pvalue / 2
        )

    result = simulate.rejection_rate(liberal, worlds, alpha=0.05)
    assert not result.within_size_band
    assert result.rate > 0.07


def test_power_rises_with_the_effect_size():
    features = correlated(rows=300)

    def test(world: simulate.World) -> float:
        return float(stats.pearsonr(world.features["a"], world.outcome).pvalue)

    curve = simulate.power_curve(
        test,
        features,
        "a",
        grid=(0.05, 0.2, 0.8),
        runs=200,
        alpha=0.05,
        seed=3,
    )
    rates = [curve[size].rate for size in sorted(curve)]
    assert rates == sorted(rates)
    assert curve[0.8].rate > 0.99
    assert curve[0.05].rate < 0.4
    assert tuple(simulate.EFFECT_GRID) == tuple(sorted(simulate.EFFECT_GRID))


def test_false_discovery_proportions_and_their_summary():
    features = correlated(rows=100)
    world = simulate.planted_world(features, {"a": 0.5}, seed=1)
    assert simulate.false_discovery_proportion({"a"}, world) == 0.0
    assert simulate.false_discovery_proportion({"a", "b"}, world) == 0.5
    assert simulate.false_discovery_proportion({"b"}, world) == 1.0
    assert simulate.false_discovery_proportion(set(), world) == 0.0
    summary = simulate.summarize_false_discoveries(
        [0.0, 0.5, 0.0, 0.5], confidence=0.95
    )
    assert summary.mean == pytest.approx(0.25)
    assert summary.upper > summary.mean
    assert summary.runs == 4
    with pytest.raises(ValueError):
        simulate.summarize_false_discoveries([])
    with pytest.raises(ValueError):
        simulate.summarize_false_discoveries([1.2])


def test_coefficient_p_values_match_scipy_on_a_single_feature():
    features = correlated(rows=300)[["a"]]
    world = simulate.planted_world(features, {"a": 0.3}, seed=1)
    ours = simulate.coefficient_pvalues(world)
    reference = stats.linregress(features["a"], world.outcome)
    assert list(ours.index) == ["a"]
    assert float(ours["a"]) == pytest.approx(reference.pvalue, rel=1e-8)


def test_coefficient_p_values_need_more_rows_than_columns():
    features = correlated(rows=300).head(3)
    world = simulate.planted_world(features, {"a": 0.5}, seed=1)
    with pytest.raises(ValueError, match="rows"):
        simulate.coefficient_pvalues(world)


def test_marginal_tests_do_not_test_the_planted_truth_when_features_correlate():
    """A zero coefficient is not a zero correlation in a correlated design.

    The planted truth is conditional on the other columns. A marginal test
    on a null-coefficient column rejects often because the column shares
    signal with a planted one; the conditional test keeps its level.
    """
    features = correlated(rows=1000)
    marginal = conditional = 0
    runs = 300
    for run in range(runs):
        world = simulate.planted_world(features, {"a": 0.5}, seed=2, index=run)
        marginal += (
            stats.pearsonr(world.features["b"], world.outcome).pvalue <= 0.05
        )
        conditional += float(simulate.coefficient_pvalues(world)["b"]) <= 0.05
    assert marginal / runs > 0.5
    assert conditional / runs < 0.10
