"""Exact-value and exact-message tests that pin the ledger's arithmetic.

Each test here exists because a mutation of the code (a flipped comparison,
a changed constant, a dropped argument or a renamed field) survived the
broader tests. They check numbers worked out by hand or by scipy and the
literal text of validation errors.
"""

from fractions import Fraction
import re

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from scmle.data.ledger import bivariate as bv
from scmle.data.ledger import budget as b
from scmle.data.ledger import controls as c
from scmle.data.ledger import multiplicity as m
from scmle.data.ledger import online_fdr as o

F = Fraction
ALPHA = F(1, 10)
R = b.MultiplicityRule


def raises(error, message):
    """`pytest.raises` that matches the whole message."""
    return pytest.raises(error, match=f"^{re.escape(message)}$")


# --- online_fdr -----------------------------------------------------------


def test_validation_messages_are_exact():
    with raises(ValueError, "horizon must be at least 1"):
        o.Discount(horizon=0)
    with raises(ValueError, "index must be at least 1"):
        o.Discount().weight(0)
    with raises(ValueError, "count must be non-negative"):
        o.Discount().cumulative(-1)
    with raises(ValueError, "evidence needs a p-value or an e-value"):
        o.Evidence()
    with raises(ValueError, "p must be within [0, 1]"):
        o.Evidence(p=2.0)
    with raises(ValueError, "e must be a finite number"):
        o.Evidence(e=float("inf"))
    with raises(ValueError, "e must be non-negative"):
        o.Evidence(e=-1.0)
    with raises(ValueError, "p must be within [0, 1]"):
        o.calibrate(-0.1)
    with raises(ValueError, "kappa must be a finite number"):
        o.calibrate(0.5, kappa=float("nan"))
    with raises(ValueError, "kappa must be strictly between 0 and 1"):
        o.calibrate(0.5, kappa=1.0)
    with raises(ValueError, "alpha must be within [0, 1]"):
        o.ELond(2)
    with raises(ValueError, "kappa must be a finite number"):
        o.ELond(ALPHA, kappa=float("nan"))
    with raises(ValueError, "kappa must be strictly between 0 and 1"):
        o.ELond(ALPHA, kappa=0.0)
    with raises(ValueError, "w0 must be within [0, 1]"):
        o.LordPlusPlus(ALPHA, w0=2)
    with raises(ValueError, "w0 must be within [0, alpha]"):
        o.LordPlusPlus(ALPHA, w0=F(1, 5))
    with raises(ValueError, "lambda must be a finite number"):
        o.Saffron(ALPHA, lam=float("nan"))
    with raises(ValueError, "lambda must be strictly between 0 and 1"):
        o.Saffron(ALPHA, lam=1)
    with raises(ValueError, "unexpected or missing rule fields"):
        o.RuleConfig.from_dict({"rule": "e-lond"})
    with raises(
        ValueError, "rule must be one of ('e-lond', 'lord++', 'saffron')"
    ):
        o.RuleConfig("x", ALPHA)
    with raises(ValueError, "lord++ needs a p-value"):
        o.LordPlusPlus(ALPHA).step(o.State(), o.Evidence(e=1.0))


def test_config_parse_errors_name_the_field():
    good = o.RuleConfig(o.SAFFRON, ALPHA, w0=F(1, 25)).to_dict()
    for field, label in (
        ("alpha", "alpha"),
        ("w0", "w0"),
        ("lambda", "lambda"),
    ):
        bad = {**good, field: "not-a-fraction"}
        with raises(ValueError, f"{label}: invalid fraction 'not-a-fraction'"):
            o.RuleConfig.from_dict(bad)


def test_alpha_one_is_a_valid_level():
    rule = o.ELond(1)
    assert rule.alpha == 1
    assert o.LordPlusPlus(1).w0 == F(1, 2)


def test_only_saffron_reports_candidates():
    items = [o.Evidence(p=0.0)]
    elond = o.ELond(ALPHA).step(o.State(), o.Evidence(p=0.0))
    lord = o.LordPlusPlus(ALPHA).step(o.State(), items[0])
    assert elond.candidate is False and lord.candidate is False
    assert o.Saffron(ALPHA).step(o.State(), items[0]).candidate is True


def test_saffron_boundary_and_wealth_by_hand():
    rule = o.Saffron(F(1, 4))  # W_0 = 1/8; first level = (1/2)(1/8)(1/2)
    level = rule.level(rule.initial())
    assert level == F(1, 32)
    assert rule.step(rule.initial(), o.Evidence(p=1 / 32)).rejected is True
    assert (
        rule.step(rule.initial(), o.Evidence(p=1 / 32 * 1.001)).rejected
        is False
    )
    # One rejected candidate at lambda = 2/5: credit = W_0 + (alpha - W_0).
    custom = o.Saffron(ALPHA, lam=F(2, 5))
    step = custom.step(custom.initial(), o.Evidence(p=0.0))
    assert step.wealth_after == F(3, 5) * ALPHA - 0
    assert step.wealth_before == F(3, 5) * F(1, 20)
    # A non-candidate spends its level: wealth falls by exactly that level.
    plain = o.Saffron(ALPHA)
    miss = plain.step(plain.initial(), o.Evidence(p=0.9))
    assert miss.wealth_before - miss.wealth_after == miss.level


def test_config_parameters_reach_the_rule():
    elond = o.RuleConfig(o.ELOND, ALPHA, horizon=4, kappa=0.25).build()
    assert elond.kappa == 0.25 and elond.discount.horizon == 4
    assert elond.level(elond.initial()) == ALPHA / 4
    lord = o.RuleConfig(
        o.LORD_PLUS_PLUS, ALPHA, horizon=4, w0=F(1, 100)
    ).build()
    assert lord.w0 == F(1, 100) and lord.discount.horizon == 4
    assert lord.level(lord.initial()) == F(1, 100) / 4
    saffron = o.RuleConfig(
        o.SAFFRON, ALPHA, horizon=4, w0=F(1, 100), lam=F(1, 5)
    ).build()
    assert saffron.lam == F(1, 5) and saffron.w0 == F(1, 100)
    assert saffron.discount.horizon == 4
    assert saffron.level(saffron.initial()) == F(4, 5) * F(1, 100) / 4


def test_calibrator_exact_and_general_forms():
    assert o.calibrate(0.25, kappa=0.5) == 1.0
    assert o.calibrate(0.0625, kappa=0.5) == 2.0
    assert o.calibrate(0.5, kappa=0.25) == pytest.approx(0.25 * 0.5**-0.75)


# --- multiplicity -----------------------------------------------------------


def test_adjustment_carries_its_rule_and_level():
    full = m.adjust([0.001, 0.2], R.BH, F(1, 20))
    assert full.rule is R.BH and full.alpha == F(1, 20)
    empty = m.adjust([], R.HOLM, F(1, 10))
    assert empty.rule is R.HOLM and empty.alpha == F(1, 10)
    assert empty.adjusted == ()


def test_a_single_hypothesis_family():
    for rule in R:
        out = m.adjust([0.04], rule, F(1, 20))
        assert out.rejected == (True,)
        assert out.adjusted == (0.04,)
        assert m.adjust([0.06], rule, F(1, 20)).rejected == (False,)


def test_holm_adjusted_values_are_monotone_in_the_rank():
    out = m.adjust([0.02, 0.021], R.HOLM, F(1, 20))
    assert out.adjusted == (0.04, 0.04)
    assert m.adjust([0.021, 0.02], R.HOLM, F(1, 20)).adjusted == (0.04, 0.04)


def test_multiplicity_messages_are_exact():
    with raises(ValueError, "rule must be a MultiplicityRule"):
        m.adjust([0.1], "bh", F(1, 20))  # type: ignore[arg-type]
    with raises(ValueError, "alpha must be strictly between 0 and 1"):
        m.adjust([0.1], R.BH, F(1))
    with raises(ValueError, "p-value must be within [0, 1]"):
        m.adjust([1.5], R.BH, F(1, 20))
    assert m.harmonic_exact(0) == 0 and m.harmonic_exact(1) == 1


# --- controls ---------------------------------------------------------------


def test_the_quantile_index_is_exact_when_q_times_b_plus_one_is_an_integer():
    assert c.quantile_index(99) == 94  # 0.95 * 100 = 95 -> the 95th
    assert c.percentile(tuple(range(99))) == 94
    assert c.quantile_index(100) == 95  # ceil(95.95) = 96 -> the 96th


@pytest.mark.parametrize(
    ("size", "confidence", "expected"),
    [
        (20, 0.9, (16, 19)),
        (40, 0.9, (35, 39)),
        (99, 0.9, (89, 97)),
        (100, 0.9, (90, 98)),
        (40, 0.5, (36, 39)),
        (100, 0.5, (93, 97)),
    ],
)
def test_percentile_interval_values(size, confidence, expected):
    # Expected values come from scipy's binom.ppf(tail, B, 0.95) and
    # binom.ppf(1 - tail, B, 0.95) + 1, mapped to order statistics (counts
    # equal to their rank, 0-based) and clipped to the range.
    assert (
        c.percentile_interval(tuple(range(size)), confidence=confidence)
        == expected
    )


def test_percentile_interval_ignores_the_input_order():
    counts = tuple(range(40))
    assert c.percentile_interval(counts[::-1]) == c.percentile_interval(counts)
    with raises(ValueError, "counts must not be empty"):
        c.percentile_interval(())
    with raises(ValueError, "counts must not be empty"):
        c.percentile(())


def test_permute_within_moves_values_inside_every_stratum():
    values = np.arange(60.0)
    strata = np.repeat([0, 1, 2], 20)
    moved = {0: False, 1: False, 2: False}
    for seed in range(5):
        out = c.permute_within(values, strata, np.random.default_rng(seed))
        for label in moved:
            inside = strata == label
            assert sorted(out[inside]) == sorted(values[inside])
            moved[label] |= not np.array_equal(out[inside], values[inside])
    assert all(moved.values())  # a cut at the wrong place would pin a stratum


def test_permute_within_accepts_unsorted_labels():
    values = np.arange(12.0)
    strata = np.array([2, 0, 1, 0, 2, 1, 0, 1, 2, 0, 1, 2])
    out = c.permute_within(values, strata, np.random.default_rng(3))
    for label in (0, 1, 2):
        assert sorted(out[strata == label]) == sorted(values[strata == label])


def test_block_labels_are_integers():
    labels = c.block_labels(np.arange(10.0), 5)
    assert labels.dtype.kind == "i"
    assert list(labels) == [0] * 5 + [1] * 5


def test_control_messages_are_exact():
    with raises(ValueError, "block_size must be at least 2"):
        c.block_labels(np.arange(4.0), 1)
    with raises(ValueError, "observed must be non-negative"):
        c.summarize("x", "strata", -1, (0,) * 20)
    with raises(ValueError, "b must be at least 20"):
        c.permuted_target_control(
            lambda f, t: 0,
            pd.DataFrame({"a": [1.0, 2.0]}),
            pd.Series([1.0, 2.0]),
            observed=0,
            b=5,
            seed=0,
            strata=None,
        )
    with raises(ValueError, "b must be at least 20"):
        c.resampling_control(
            lambda f, t: 0,
            pd.DataFrame({"a": [1.0, 2.0]}),
            pd.Series([1.0, 2.0]),
            observed=0,
            b=5,
            seed=0,
        )
    with raises(ValueError, "strata needs one label per row"):
        c.permute_within(
            np.arange(3.0), np.array([0, 1]), np.random.default_rng()
        )


def test_result_dictionary_keys_and_round_trip_with_exceeds_true():
    result = c.summarize("permuted_target", "strata", 4, (0,) * 20)
    record = result.to_dict()
    assert set(record) == {
        "kind",
        "scope",
        "observed",
        "permuted",
        "b",
        "p95",
        "p95_interval",
        "p95_interval_confidence",
        "exceedance_p",
        "exceeds",
        "message",
        "note",
    }
    assert record["b"] == 20 and record["p95_interval_confidence"] == 0.9
    assert record["note"].startswith("The 95th percentile is estimated from B")
    assert c.ControlResult.from_dict(record) == result
    assert c.ControlResult.from_dict(record).exceeds is True


def test_default_random_streams_are_the_documented_ones():
    features = pd.DataFrame({"a": np.arange(30.0)})
    target = pd.Series(np.arange(30.0))

    def count(frame, shuffled):
        return int(shuffled.iloc[0] % 5)

    default = c.permuted_target_control(
        count, features, target, observed=3, b=25, seed=4, strata=None
    )
    named = c.permuted_target_control(
        count,
        features,
        target,
        observed=3,
        b=25,
        seed=4,
        strata=None,
        component="ledger.control.permuted_target",
    )
    other = c.permuted_target_control(
        count,
        features,
        target,
        observed=3,
        b=25,
        seed=4,
        strata=None,
        component="somewhere.else",
    )
    assert default == named and default.permuted != other.permuted

    def count_columns(frame, y):
        return int(frame["a"].iloc[0] % 5)

    one = c.resampling_control(
        count_columns, features, target, observed=3, b=25, seed=4
    )
    two = c.resampling_control(
        count_columns,
        features,
        target,
        observed=3,
        b=25,
        seed=4,
        component="ledger.control.resampling",
    )
    assert one == two


def test_controls_keep_the_index_and_name_of_what_they_shuffle():
    index = pd.Index([f"r{i}" for i in range(30)])
    features = pd.DataFrame({"a": np.arange(30.0)}, index=index)
    target = pd.Series(np.arange(30.0), index=index, name="outcome")
    seen = []

    def spy(frame, shuffled):
        assert list(frame.index) == list(index)
        assert shuffled.name == "outcome" and list(shuffled.index) == list(
            index
        )
        seen.append(1)
        return 0

    c.permuted_target_control(
        spy, features, target, observed=0, b=20, seed=1, strata=None
    )
    c.resampling_control(spy, features, target, observed=0, b=20, seed=1)
    c.resampling_control(
        spy,
        features,
        target,
        observed=0,
        b=20,
        seed=1,
        mode="blocks",
        blocks=np.repeat([0, 1, 2], 10),
    )
    assert len(seen) == 60


def test_resampling_modes_hand_the_right_frames_to_the_count_function():
    features = pd.DataFrame({"a": np.arange(40.0), "b": np.arange(40.0) * 2})
    target = pd.Series(np.arange(40.0))
    blocks = np.repeat(np.arange(4), 10)
    frames = []

    def spy(frame, shuffled):
        assert isinstance(frame, pd.DataFrame)
        frames.append(frame.copy())
        return 0

    c.resampling_control(
        spy,
        features,
        target,
        observed=0,
        b=20,
        seed=2,
        mode="blocks",
        blocks=blocks,
    )
    for frame in frames:
        values = frame["a"].to_numpy()
        for start in range(0, 40, 10):
            piece = values[start : start + 10]
            assert list(piece) == list(range(int(piece[0]), int(piece[0]) + 10))
    frames.clear()
    result = c.resampling_control(
        spy,
        features,
        target,
        observed=0,
        b=20,
        seed=2,
        strata=np.repeat([0, 1], 20),
    )
    assert result.scope == "strata"
    for frame in frames:
        assert set(frame["a"][:20]) == set(range(20))  # shuffled within strata
    # With labels but the default mode, labels are ignored: columns mode runs.
    frames.clear()
    c.resampling_control(
        spy, features, target, observed=0, b=20, seed=2, blocks=blocks
    )
    assert any(not np.array_equal(f["a"][:10], np.arange(10.0)) for f in frames)
    assert any(set(f["a"][:10]) != set(range(10)) for f in frames)


# --- bivariate --------------------------------------------------------------


def test_untestable_pairs_return_a_complete_outcome():
    out = bv.spearman(np.array([1.0, 2.0]), np.array([1.0, 3.0]))
    assert out == bv.Outcome(
        1.0, 0.0, 0.0, 2, "fewer than three complete pairs"
    )
    constant_y = bv.spearman(np.arange(10.0), np.ones(10))
    assert constant_y == bv.Outcome(1.0, 0.0, 0.0, 10, "a variable is constant")
    constant_x = bv.kruskal(np.ones(8), np.array([0, 0, 0, 0, 1, 1, 1, 1]))
    assert constant_x == bv.Outcome(1.0, 0.0, 0.0, 8, "a variable is constant")
    single = bv.kruskal(np.arange(5.0), np.zeros(5))
    assert single == bv.Outcome(
        1.0, 0.0, 0.0, 5, "fewer than two classes or three rows"
    )


def test_a_perfect_relation_reports_its_rho_as_statistic_and_effect(
    monkeypatch,
):
    x = np.array([1.0, 2.0, 3.0])
    out = bv.spearman(x, x)
    assert (out.statistic, out.effect, out.n) == (1.0, 1.0, 3)
    # Rounding can push the computed correlation just past one; it is clipped.
    monkeypatch.setattr(
        bv.np,
        "corrcoef",
        lambda a, b: np.array([[1.0, 1.0000000000000002]] * 2),
    )
    over = bv.spearman(x, x)
    assert over.statistic == 1.0 and over.effect == 1.0
    monkeypatch.setattr(
        bv.np,
        "corrcoef",
        lambda a, b: np.array([[1.0, -1.0000000000000002]] * 2),
    )
    assert bv.spearman(x, -x).statistic == -1.0


def test_kruskal_edge_classes_match_scipy():
    x = np.array([1.0, 2.0, 3.0, 4.0, 10.0])
    labels = np.array([0, 0, 0, 0, 1])
    out = bv.kruskal(x, labels)
    reference = stats.kruskal(x[:4], x[4:])
    assert out.p == pytest.approx(reference.pvalue, rel=1e-12)
    assert out.statistic == pytest.approx(reference.statistic, rel=1e-12)
    three = bv.kruskal(np.array([1.0, 5.0, 3.0]), np.array([0, 0, 1]))
    ref3 = stats.kruskal([1.0, 5.0], [3.0])
    assert three.p == pytest.approx(ref3.pvalue, rel=1e-12) and three.n == 3
    two = bv.kruskal(np.arange(12.0), np.repeat([0, 1], 6))
    assert two.p == pytest.approx(
        stats.kruskal(np.arange(6.0), np.arange(6.0, 12.0)).pvalue, rel=1e-12
    )


def test_run_statistic_passes_every_argument_through():
    rng = np.random.default_rng(1)
    x = rng.normal(size=50)
    y = x + rng.normal(size=50)
    assert bv.run_statistic("spearman", x, y, bv.GREATER) == bv.spearman(
        x, y, bv.GREATER
    )
    assert (
        bv.run_statistic("spearman", x, y).p
        != bv.run_statistic("spearman", x, y, bv.LESS).p
    )
    labels = (y > np.median(y)).astype(float)
    assert bv.run_statistic("kruskal", x, labels) == bv.kruskal(x, labels)
    with raises(ValueError, "statistic must be one of ['kruskal', 'spearman']"):
        bv.run_statistic("t", x, y)
    with raises(
        ValueError, "direction must be one of ('two-sided', 'greater', 'less')"
    ):
        bv.spearman(x, y, "up")
    with raises(ValueError, "kruskal supports only the two-sided alternative"):
        bv.kruskal(x, labels, bv.LESS)
    with raises(ValueError, "x and y must be one-dimensional and equally long"):
        bv.spearman(x, y[:5])


# --- second round of survivors ----------------------------------------------


def test_p_values_are_calibrated_with_the_rules_own_kappa():
    rule = o.ELond(ALPHA, kappa=0.25)
    step = rule.step(rule.initial(), o.Evidence(p=0.01))
    assert step.e_value == o.calibrate(0.01, 0.25)
    assert step.e_value != o.calibrate(0.01)


def test_harmonic_numbers_are_exact_fractions():
    assert isinstance(m.harmonic_exact(0), Fraction)
    assert m.harmonic_exact(4) == F(25, 12)


def test_percentile_takes_the_requested_quantile():
    counts = tuple(range(20))
    assert c.percentile(counts, q=0.5) == 10  # ceil(0.5 * 21) = 11th
    assert c.percentile(counts, q=0.95) == 19


@pytest.mark.parametrize(
    ("counts", "expected"),
    [((5,), (5, 5)), ((5, 6), (5, 6)), ((5, 6, 7), (6, 7))],
)
def test_percentile_interval_for_tiny_samples(counts, expected):
    assert c.percentile_interval(counts) == expected


def test_permute_within_follows_the_stable_order_of_members():
    rng_seed = 7
    values = np.arange(400.0)
    strata = np.tile([3, 1, 2, 1, 3, 2, 2, 3], 50)
    out = c.permute_within(values, strata, np.random.default_rng(rng_seed))
    # Independent construction: members of each label in ascending position,
    # labels visited in ascending order, one permutation of positions each.
    rng = np.random.default_rng(rng_seed)
    expected = values.copy()
    for label in sorted(set(strata.tolist())):
        members = np.flatnonzero(strata == label)
        expected[members] = values[rng.permutation(members)]
    assert np.array_equal(out, expected)


def test_block_labels_use_a_stable_rank_for_ties():
    order = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0] * 40)
    labels = c.block_labels(order, 10)
    zeros = np.flatnonzero(order == 0.0)
    assert list(labels[zeros][:10]) == [0] * 10  # ties keep row order
    ones = np.flatnonzero(order == 1.0)
    assert list(labels[ones]) == sorted(labels[ones])


def test_shuffle_blocks_is_reproducible_from_a_hand_built_oracle():
    features = pd.DataFrame({"a": np.arange(300.0)})
    blocks = np.repeat(np.arange(30), 10)
    out = c.shuffle_blocks(features, blocks, np.random.default_rng(5))
    rng = np.random.default_rng(5)
    pieces = [np.flatnonzero(blocks == label) for label in range(30)]
    sequence = np.concatenate([pieces[i] for i in rng.permutation(30)])
    expected = np.arange(300.0)
    expected[np.concatenate(pieces)] = np.arange(300.0)[sequence]
    assert np.array_equal(out["a"].to_numpy(), expected)


def test_remaining_control_messages_and_streams_are_exact():
    with raises(ValueError, "strata needs one label per row"):
        c.permuted_target_control(
            lambda f, t: 0,
            pd.DataFrame({"a": np.arange(5.0)}),
            pd.Series(np.arange(5.0)),
            observed=0,
            b=20,
            seed=0,
            strata=np.array([0, 1]),
        )
    features = pd.DataFrame({"a": np.arange(30.0)})
    target = pd.Series(np.arange(30.0))
    with raises(ValueError, "mode must be 'columns' or 'blocks'"):
        c.resampling_control(
            lambda f, t: 0, features, target, observed=0, b=20, seed=0, mode="x"
        )
    with raises(ValueError, "blocks mode needs block labels"):
        c.resampling_control(
            lambda f, t: 0,
            features,
            target,
            observed=0,
            b=20,
            seed=0,
            mode="blocks",
        )
    firsts = []
    c.resampling_control(
        lambda f, t: firsts.append(float(f["a"].iloc[0])) or 0,
        features,
        target,
        observed=0,
        b=20,
        seed=0,
    )
    assert len(set(firsts)) > 5  # a stream per repeat, not one stream


def test_a_block_label_array_does_not_switch_the_default_mode():
    features = pd.DataFrame({"a": np.arange(60.0)})
    target = pd.Series(np.arange(60.0))
    blocks = np.repeat(np.arange(6), 10)
    broken = []

    def spy(frame, shuffled):
        values = frame["a"].to_numpy()
        runs = all(
            list(values[s : s + 10])
            == list(range(int(values[s]), int(values[s]) + 10))
            for s in range(0, 60, 10)
        )
        broken.append(not runs)
        return 0

    c.resampling_control(
        spy, features, target, observed=0, b=20, seed=3, blocks=blocks
    )
    assert all(broken)  # columns mode shuffles single rows, never whole blocks


def test_missing_values_given_as_none_are_dropped_not_fatal():
    x = [1.0, None, 3.0, 4.0, 5.0, 6.0]
    y = [2.0, 1.0, 5.0, 3.0, 8.0, 9.0]
    out = bv.spearman(x, y)
    assert out.n == 5 and 0.0 <= out.p <= 1.0
    out2 = bv.kruskal(x, [0, 0, 0, 1, 1, 1])
    assert out2.n == 5


def test_kruskal_with_exactly_three_rows_is_tested():
    x = np.array([1.0, 2.0, 9.0])
    out = bv.kruskal(x, np.array([0, 0, 1]))
    assert out.statistic == pytest.approx(1.5)
    assert out.p == pytest.approx(stats.chi2.sf(1.5, 1))
    assert out.n == 3 and out.effect == pytest.approx(1.5 / 2)


def test_code_hash_normalises_line_endings(monkeypatch):
    import hashlib

    monkeypatch.setattr(bv.inspect, "getsource", lambda f: "a\r\nb\r\n")
    assert bv.code_hash("spearman") == hashlib.sha256(b"a\nb\n").hexdigest()
