"""Negative controls: permutation scope, percentile gate and bookkeeping."""

import numpy
import pandas
import pytest

from scmle.data.ledger import controls


def rng(seed=0):
    return numpy.random.default_rng(seed)


def test_percentile_is_the_ceil_q_b_plus_1_th_order_statistic():
    assert (
        controls.percentile(tuple(range(20))) == 19
    )  # ceil(19.95) = 20: the max
    assert (
        controls.percentile(tuple(range(21))) == 20
    )  # ceil(20.9) = 21: the max
    assert controls.percentile(tuple(range(100))) == 95  # ceil(95.95) = 96th
    assert controls.percentile(tuple(range(19, -1, -1))) == 19  # unsorted input
    assert controls.percentile((5,) * 20) == 5
    assert controls.quantile_index(20) == 19
    assert controls.quantile_index(100) == 95
    assert controls.quantile_index(19) == 18
    with pytest.raises(ValueError):
        controls.percentile(())


def test_the_gate_is_strict_at_the_percentile():
    counts = tuple([0] * 19 + [3])  # p95 = 3, the largest
    assert (
        controls.summarize("permuted_target", "strata", 3, counts).exceeds
        is False
    )
    assert (
        controls.summarize("permuted_target", "strata", 4, counts).exceeds
        is True
    )
    assert controls.summarize("permuted_target", "strata", 0, counts).p95 == 3
    flat = tuple([0] * 20)
    assert (
        controls.summarize("permuted_target", "strata", 1, flat).exceeds is True
    )
    assert (
        controls.summarize("permuted_target", "strata", 0, flat).exceeds
        is False
    )


def test_exceedance_p_counts_ties_and_adds_one():
    counts = tuple([0] * 15 + [2] * 5)
    assert controls.summarize(
        "x", "strata", 3, counts
    ).exceedance_p == pytest.approx(1 / 21)
    assert controls.summarize(
        "x", "strata", 2, counts
    ).exceedance_p == pytest.approx(6 / 21)
    assert controls.summarize("x", "strata", 0, counts).exceedance_p == 1.0


def test_fewer_than_twenty_reruns_are_refused():
    with pytest.raises(ValueError, match="at least 20"):
        controls.summarize("x", "strata", 1, tuple(range(19)))
    with pytest.raises(ValueError, match="b"):
        controls.permuted_target_control(
            lambda f, t: 0,
            pandas.DataFrame({"a": [1.0]}),
            pandas.Series([1.0]),
            observed=0,
            b=19,
            seed=0,
            strata=None,
        )


def test_the_percentile_interval_is_ordered_and_covers_the_percentile():
    counts = tuple(range(40))
    low, high = controls.percentile_interval(counts)
    assert low <= controls.percentile(counts) <= high
    assert high <= max(counts)
    with pytest.raises(ValueError):
        controls.percentile_interval(())


def test_permute_within_never_crosses_strata():
    strata = numpy.array([0, 0, 0, 1, 1, 1, 1, 2])
    values = numpy.arange(8.0)
    for seed in range(20):
        out = controls.permute_within(values, strata, rng(seed))
        for label in (0, 1, 2):
            inside = strata == label
            assert sorted(out[inside]) == sorted(values[inside])
    assert out[7] == 7.0  # a singleton stratum cannot move


def test_permute_within_actually_shuffles_and_global_crosses():
    values = numpy.arange(200.0)
    strata = numpy.repeat([0, 1], 100)
    inside = controls.permute_within(values, strata, rng(1))
    assert not numpy.array_equal(inside, values)
    assert set(inside[:100]) == set(values[:100])
    everywhere = controls.permute_within(values, None, rng(1))
    assert set(everywhere[:100]) != set(values[:100])
    assert sorted(everywhere) == sorted(values)
    with pytest.raises(ValueError, match="one label"):
        controls.permute_within(values, strata[:5], rng())


def test_permute_within_does_not_modify_its_input():
    values = numpy.arange(10.0)
    keep = values.copy()
    controls.permute_within(values, numpy.repeat([0, 1], 5), rng())
    assert numpy.array_equal(values, keep)


def test_block_labels_cut_ordered_rows_into_blocks():
    order = numpy.array([5, 3, 1, 4, 2, 0])
    labels = controls.block_labels(order, 2)
    # sorted order 0..5 -> ranks; rows with the 2 smallest keys share block 0
    assert list(labels) == [2, 1, 0, 2, 1, 0]
    with pytest.raises(ValueError):
        controls.block_labels(order, 1)


def test_target_control_permutes_only_the_target_within_strata():
    features = pandas.DataFrame({"a": numpy.arange(40.0)})
    target = pandas.Series(numpy.arange(40.0))
    strata = numpy.repeat([0, 1], 20)
    seen = []

    def spy(frame, shuffled):
        assert frame.equals(features)
        for label in (0, 1):
            inside = strata == label
            assert sorted(shuffled[inside]) == sorted(target[inside])
        seen.append(shuffled.to_numpy().copy())
        return 0

    result = controls.permuted_target_control(
        spy, features, target, observed=1, b=20, seed=3, strata=strata
    )
    assert len(seen) == 20 and result.b == 20
    assert result.scope == "strata" and result.kind == "permuted_target"
    assert len({tuple(s) for s in seen}) == 20  # distinct streams
    assert result.exceeds is True  # 1 > p95 of all zeros
    assert (
        controls.permuted_target_control(
            lambda f, t: 0,
            features,
            target,
            observed=0,
            b=20,
            seed=3,
            strata=None,
        ).scope
        == "global"
    )


def test_controls_are_deterministic_for_a_seed():
    features = pandas.DataFrame({"a": numpy.arange(30.0)})
    target = pandas.Series(numpy.arange(30.0))

    def count(f, t):
        return int(t.iloc[0] % 3)

    one = controls.permuted_target_control(
        count, features, target, observed=2, b=25, seed=9, strata=None
    )
    two = controls.permuted_target_control(
        count, features, target, observed=2, b=25, seed=9, strata=None
    )
    other = controls.permuted_target_control(
        count, features, target, observed=2, b=25, seed=10, strata=None
    )
    assert one == two and one != other


def test_target_control_validates_strata_length():
    with pytest.raises(ValueError, match="one label"):
        controls.permuted_target_control(
            lambda f, t: 0,
            pandas.DataFrame({"a": numpy.arange(5.0)}),
            pandas.Series(numpy.arange(5.0)),
            observed=0,
            b=20,
            seed=0,
            strata=numpy.array([0, 1]),
        )


def test_column_shuffle_keeps_marginals_and_breaks_association():
    r = rng(5)
    x = r.normal(size=400)
    features = pandas.DataFrame(
        {"a": x, "b": 2 * x + 0.01 * r.normal(size=400)}
    )
    assert abs(features.corr().iloc[0, 1]) > 0.99
    shuffled = controls.shuffle_columns(features, rng(6), None)
    for name in features:
        assert sorted(shuffled[name]) == sorted(features[name])
    assert abs(shuffled.corr().iloc[0, 1]) < 0.2
    grouped = controls.shuffle_columns(
        features, rng(6), numpy.repeat([0, 1], 200)
    )
    assert set(grouped["a"][:200]) == set(features["a"][:200])


def test_block_shuffle_keeps_blocks_intact_and_breaks_drift():
    n = 120
    drift = numpy.arange(n, dtype=float)
    features = pandas.DataFrame({"a": drift})
    blocks = numpy.repeat(numpy.arange(12), 10)
    shuffled = controls.shuffle_blocks(features, blocks, rng(2))
    values = shuffled["a"].to_numpy()
    for start in range(0, n, 10):
        piece = values[start : start + 10]
        assert list(piece) == list(range(int(piece[0]), int(piece[0]) + 10))
    assert not numpy.array_equal(values, drift)
    assert sorted(values) == sorted(drift)
    assert abs(numpy.corrcoef(values, drift)[0, 1]) < 0.9


def test_resampling_control_modes_and_validation():
    features = pandas.DataFrame(
        {"a": numpy.arange(50.0), "b": numpy.arange(50.0)}
    )
    target = pandas.Series(numpy.arange(50.0))
    received = []

    def spy(frame, shuffled):
        assert shuffled.equals(target)  # the target is never touched
        received.append(frame)
        return 0

    result = controls.resampling_control(
        spy, features, target, observed=2, b=20, seed=1, mode="columns"
    )
    assert result.kind == "resampling" and result.scope == "global"
    assert len(received) == 20
    blocks = numpy.repeat(numpy.arange(5), 10)
    result = controls.resampling_control(
        spy,
        features,
        target,
        observed=2,
        b=20,
        seed=1,
        mode="blocks",
        blocks=blocks,
    )
    assert result.scope == "blocks"
    with pytest.raises(ValueError, match="block labels"):
        controls.resampling_control(
            spy, features, target, observed=0, b=20, seed=1, mode="blocks"
        )
    with pytest.raises(ValueError, match="mode"):
        controls.resampling_control(
            spy, features, target, observed=0, b=20, seed=1, mode="x"
        )


def test_result_round_trips_and_states_the_floor_message():
    counts = tuple([0] * 20)
    result = controls.summarize("permuted_target", "strata", 0, counts)
    assert "do not clearly exceed" in result.message
    assert controls.ControlResult.from_dict(result.to_dict()) == result
    good = controls.summarize("permuted_target", "strata", 4, counts)
    assert "exceed the 95th percentile" in good.message
    assert good.to_dict()["note"]
