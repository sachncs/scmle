"""Behavioural tests of the leakage screen submodules, imported by full name."""

from fractions import Fraction
import math

import numpy
import pandas
import pytest

from scmle.assurance.admissibility import leakage
from scmle.assurance.admissibility.screen import association
from scmle.assurance.admissibility.screen import fitting
from scmle.assurance.admissibility.screen import hypotheses
from scmle.assurance.admissibility.screen import permutation
from scmle.assurance.admissibility.screen import planning
from scmle.assurance.admissibility.screen import report
from scmle.assurance.admissibility.screen import settings
from scmle.assurance.detectors import certificates
from scmle.contracts import errors


def config(**changes):
    values = {"skill_ceiling": 0.9, "association_ceiling": 0.5}
    values.update(changes)
    return leakage.Config(**values)


def binary_problem(n=40, folds=4):
    y = pandas.Series([0, 1] * (n // 2))
    return fitting.encode(y, "classification", config(folds=folds), None, None)


# ---- settings ---------------------------------------------------------------


def test_the_mechanisms_and_detectors_are_named():
    assert settings.DETECTOR_ID == "f4-leakage-screen"
    assert (
        settings.SKILL,
        settings.MISSING,
        settings.PURITY,
        settings.ECHO,
    ) == (
        "skill",
        "missingness",
        "record_purity",
        "echo",
    )
    assert settings.MECHANISM_DEFECTS[settings.ECHO] == ()
    assert settings.MECHANISM_DEFECTS[settings.PURITY] == ("leak_rare_record",)
    assert settings.DETECTORS == {
        "binary": "f4-leakage-screen",
        "multiclass": "f4-leakage-screen-multiclass",
        "regression": "f4-leakage-screen-regression",
    }
    assert settings.STORE_PATH.name == "screen_certificates.json"
    assert set(settings.WORDS) == set(settings.MECHANISM_DEFECTS)


def test_the_word_pattern_splits_on_underscores_and_punctuation():
    assert settings.WORD.findall("closed_the account, 42!") == [
        "closed",
        "the",
        "account",
        "42",
    ]


@pytest.mark.parametrize(
    ("task", "classes", "expected"),
    [
        ("regression", 0, "regression"),
        ("classification", 2, "binary"),
        ("classification", 3, "multiclass"),
        ("classification", 10, "multiclass"),
        ("classification", 11, None),
        ("classification", 1, None),
    ],
)
def test_family_names_the_certified_kind(task, classes, expected):
    assert settings.family(task, classes) == expected


def test_allowance_and_share_scale_the_budget():
    assert settings.allowance(None) is None
    assert settings.allowance(10.0) == 37_500
    assert settings.allowance(0.0001) == 0
    assert settings.share(None) is None
    assert settings.share(100.0) == pytest.approx(15.0)


def test_cost_adds_the_fixed_scan_to_the_per_fit_rate():
    assert settings.cost("classification", 1000, 1000) == pytest.approx(1.08)
    assert settings.cost("regression", 1000, 1000) == pytest.approx(1.012)
    assert settings.cost("regression", 0, 5) == 1.0


# ---- fitting ----------------------------------------------------------------


def test_a_numeric_column_with_few_values_gets_level_indicators():
    out = fitting.basis(pandas.Series([1.0, 2.0, numpy.nan, 1.0]))
    # levels: "1", "2", "<missing>"
    assert out.tolist() == [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]]


def test_a_rich_numeric_column_gets_hat_functions_summing_to_one():
    values = pandas.Series([*list(numpy.arange(20.0)), numpy.nan])
    out = fitting.basis(values, knots=4)
    assert out.shape == (21, 6)
    assert numpy.allclose(out.sum(axis=1), 1.0)
    assert out[-1].tolist() == [0, 0, 0, 0, 0, 1]
    assert out[0, 0] == 1.0 and out[19, 4] == 1.0


def test_a_rich_numeric_column_without_gaps_drops_the_missing_column():
    out = fitting.basis(pandas.Series(numpy.arange(20.0)), knots=4)
    assert out.shape == (20, 5)


def test_a_datetime_column_is_read_in_days():
    dates = pandas.Series(pandas.to_datetime(["2020-01-01", "2020-01-03"]))
    assert fitting.basis(dates).tolist() == [[1.0, 0.0], [0.0, 1.0]]


def test_free_text_adds_word_indicators():
    texts = [f"row {i} closed" if i % 2 else f"row {i} open" for i in range(60)]
    out = fitting.basis(pandas.Series(texts))
    assert out.shape[0] == 60
    assert out.shape[1] == leakage.MAX_LEVELS + 1 + leakage.MAX_LEVELS


def test_words_marks_the_commonest_case_folded_words():
    out = fitting.words(pandas.Series(["Aa bb", "aa", "cc aa"]))
    assert out.tolist() == [[1, 1, 0], [1, 0, 0], [1, 0, 1]]


def test_stratify_by_groups_blocks_or_nothing():
    assert fitting.stratify(3, numpy.array(["b", "a", "b"]), 2).tolist() == [
        0,
        1,
        0,
    ]
    assert fitting.stratify(5, None, 2).tolist() == [0, 0, 1, 1, 2]
    assert fitting.stratify(3, None, None).tolist() == [0, 0, 0]


def test_permute_keeps_rows_inside_their_stratum():
    strata = numpy.array([1, 0, 1, 0, 1])
    out = fitting.permute(strata, 30, numpy.random.default_rng(0))
    assert out.shape == (30, 5)
    for row in out:
        assert sorted(row.tolist()) == [0, 1, 2, 3, 4]
        assert (strata[row] == strata).all()


def test_crossfit_folds_cover_every_row_once():
    splits = fitting.crossfit(12, 3, None, None, 0)
    tests = numpy.concatenate([t for train, t in splits])
    assert sorted(tests.tolist()) == list(range(12))
    blocks = fitting.crossfit(6, 2, None, 3, 0)
    assert [t.tolist() for train, t in blocks] == [[0, 1, 2], [3, 4, 5]]


def test_oof_fit_matches_a_direct_ridge_fit():
    rng = numpy.random.default_rng(0)
    phi = rng.normal(size=(12, 2))
    values = rng.normal(size=(2, 12))
    splits = fitting.crossfit(12, 3, None, None, 0)
    out = fitting.oof_fit(values, phi, splits)
    for train, test in splits:
        for b in range(2):
            mean = values[b, train].mean()
            centred = values[b, train] - mean
            gram = phi[train].T @ phi[train] + settings.RIDGE * numpy.eye(2)
            cols = phi[train].sum(axis=0)
            cross = phi[train].T @ values[b, train] - mean * cols
            beta = numpy.linalg.solve(gram, cross)
            del centred
            assert numpy.allclose(out[b, test], mean + phi[test] @ beta)


def test_concordance_counts_pairs_within_folds_only():
    score = numpy.array([[0.1, 0.9, 0.5, 0.5]])
    positive = numpy.array([[False, True, False, True]])
    tests = [numpy.array([0, 1]), numpy.array([2, 3])]
    wins, pairs = fitting.concordance(score, positive, tests)
    assert wins.tolist() == [1.5]
    assert pairs.tolist() == [2.0]
    assert fitting.aucs(score, positive, tests).tolist() == [0.75]


def test_aucs_are_nan_where_no_fold_has_both_classes():
    out = fitting.aucs(
        numpy.array([[0.1, 0.2]]),
        numpy.array([[True, True]]),
        [numpy.arange(2)],
    )
    assert math.isnan(out[0])


def test_encode_codes_classes_and_refuses_a_constant_regression():
    problem = binary_problem()
    assert problem.classes == 2 and problem.total_squares == 0.0
    assert len(problem.splits) == 4
    regression = fitting.encode(
        pandas.Series([1.0, 2.0, 3.0, 4.0]),
        "regression",
        config(folds=2),
        None,
        None,
    )
    assert regression.classes == 0
    assert regression.total_squares == 5.0
    with pytest.raises(errors.InputError, match="constant target"):
        fitting.encode(
            pandas.Series([1.0] * 4), "regression", config(folds=2), None, None
        )


def test_a_copy_of_a_binary_target_has_perfect_skill():
    problem = binary_problem()
    phi = fitting.basis(pandas.Series(problem.y))
    statistic = fitting.skill_statistic(problem, phi)
    assert statistic(problem.y[None, :]).tolist() == [1.0]
    predictions = fitting.predict(problem, phi)
    assert len(predictions) == 1
    fold_of = numpy.empty(40, dtype="int64")
    for number, (train, test) in enumerate(problem.splits):
        assert len(train) == 30
        fold_of[test] = number
    assert (
        fitting.replicate(problem, predictions, numpy.arange(40), fold_of)
        == 1.0
    )
    bound = fitting.skill_bound(
        problem, phi, config(bootstrap=20), None, numpy.random.default_rng(0)
    )
    assert bound == 1.0


def test_a_multiclass_skill_averages_one_versus_rest():
    y = pandas.Series([0, 1, 2] * 10)
    problem = fitting.encode(y, "classification", config(folds=3), None, None)
    phi = fitting.basis(y)
    statistic = fitting.skill_statistic(problem, phi)
    assert statistic(problem.y[None, :])[0] == pytest.approx(1.0)
    assert len(fitting.predict(problem, phi)) == 3


def test_a_regression_skill_is_r_squared():
    y = pandas.Series(numpy.arange(30.0))
    problem = fitting.encode(y, "regression", config(folds=3), None, None)
    phi = fitting.basis(y)
    value = fitting.skill_statistic(problem, phi)(problem.y[None, :])[0]
    assert 0.9 < value <= 1.0
    predictions = fitting.predict(problem, phi)
    fold_of = numpy.zeros(30, dtype="int64")
    assert (
        fitting.replicate(
            problem, predictions, numpy.zeros(5, dtype=int), fold_of
        )
        is None
    )


def test_replicate_is_none_when_no_fold_has_both_classes():
    problem = binary_problem()
    predictions = [numpy.zeros(40)]
    fold_of = numpy.zeros(40, dtype="int64")
    assert (
        fitting.replicate(problem, predictions, numpy.array([0, 2, 4]), fold_of)
        is None
    )


def test_skill_bound_is_minus_infinity_without_a_computable_draw():
    problem = binary_problem()

    class Fixed:
        def integers(self, low, high, size):
            return numpy.zeros(size, dtype=int)

    phi = fitting.basis(pandas.Series(problem.y))
    assert (
        fitting.skill_bound(problem, phi, config(bootstrap=3), None, Fixed())
        == -math.inf
    )


def test_pairs_are_formed_of_two_level_columns_in_order():
    frame = pandas.DataFrame(
        {"a": [0, 1, 0], "b": ["x", "y", "y"], "c": [1, 2, 3], "d": [5, 5, 6]}
    )
    assert fitting.pairs(frame, ["a", "b", "c", "d"]) == [
        ("a", "b"),
        ("a", "d"),
        ("b", "d"),
    ]
    assert fitting.pair("a", "b") == "a & b"


# ---- association ------------------------------------------------------------


def test_chisquare_matches_the_contingency_formula():
    missing = numpy.array([True, True, False, False, False, False])
    labels = numpy.array([[0, 0, 1, 1, 1, 0], [0, 1, 0, 1, 0, 1]])
    totals = numpy.array([3.0, 3.0])
    out = association.chisquare(missing, labels, totals)
    assert out[0] == pytest.approx(3.0)
    assert out[1] == pytest.approx(0.0)
    v = association.cramer(missing, totals)(labels)
    assert v.tolist() == pytest.approx([math.sqrt(0.5), 0.0])


def test_purity_is_the_share_of_the_commonest_recorded_class():
    recorded = numpy.array([True, True, True, False])
    statistic = association.purity_statistic(recorded)
    out = statistic(numpy.array([[0, 0, 1, 1], [2, 2, 2, 0]]))
    assert out.tolist() == pytest.approx([2 / 3, 1.0])


def test_dominance_is_a_clopper_pearson_lower_bound():
    recorded = numpy.array([True] * 10)
    classes = numpy.array([1] * 10)
    cfg = config(confidence=0.95)
    assert association.dominance(recorded, classes, cfg, None) == pytest.approx(
        0.05 ** (1 / 10)
    )
    groups = numpy.array([0] * 5 + [1] * 5)
    assert association.dominance(
        recorded, classes, cfg, groups
    ) == pytest.approx(0.05 ** (1 / 2))


def test_resample_of_a_perfect_split_is_one_and_degenerate_is_minus_infinity():
    missing = numpy.array([True, False] * 10)
    classes = numpy.array([0, 1] * 10)
    bound = association.resample(
        missing,
        classes,
        2,
        config(bootstrap=20),
        None,
        numpy.random.default_rng(0),
    )
    assert bound == pytest.approx(1.0)
    groups = numpy.arange(20)
    assert association.resample(
        missing,
        classes,
        2,
        config(bootstrap=20),
        groups,
        numpy.random.default_rng(0),
    ) == pytest.approx(1.0)

    class Fixed:
        def integers(self, low, high, size):
            return numpy.zeros(size, dtype=int)

    assert (
        association.resample(
            missing, classes, 2, config(bootstrap=3), None, Fixed()
        )
        == -math.inf
    )


# ---- hypotheses -------------------------------------------------------------


def test_spell_drops_the_decimals_of_whole_numbers():
    out = hypotheses.spell(pandas.Series([1.0, 1.5, "a", 2], dtype=object))
    assert out.tolist() == ["1", "1.5", "a", "2"]


def test_forecast_returns_codes_for_an_echoing_label_column():
    problem = binary_problem()
    outcome = pandas.Series(problem.y)
    series = pandas.Series(problem.y.astype(float))
    codes = hypotheses.forecast(series, outcome, problem, 0.5)
    assert codes.tolist() == problem.y.tolist()
    constant = pandas.Series([0.0] * 40)
    assert hypotheses.forecast(constant, outcome, problem, 0.5) is None
    unknown = pandas.Series(["z"] * 40)
    assert hypotheses.forecast(unknown, outcome, problem, 0.5) is None


def test_forecast_of_a_number_checks_mean_and_spread():
    y = pandas.Series(numpy.arange(20.0))
    problem = fitting.encode(y, "regression", config(folds=2), None, None)
    assert (
        hypotheses.forecast(y + 0.1, y, problem, 0.9).tolist()
        == (y + 0.1).tolist()
    )
    assert hypotheses.forecast(y + 100.0, y, problem, 0.9) is None
    assert hypotheses.forecast(y * 3.0, y, problem, 0.9) is None
    assert hypotheses.forecast(y.astype(str), y, problem, 0.9) is None
    gappy = y.copy()
    gappy[0] = numpy.nan
    assert hypotheses.forecast(gappy, y, problem, 0.9) is None


def test_reflect_measures_agreement_or_unfitted_r_squared():
    problem = binary_problem(n=4, folds=2)
    statistic = hypotheses.reflect(problem, numpy.array([0, 1, 1, 1]))
    assert statistic(problem.y[None, :]).tolist() == [0.75]
    y = pandas.Series([0.0, 1.0, 2.0, 3.0])
    regression = fitting.encode(y, "regression", config(folds=2), None, None)
    statistic = hypotheses.reflect(
        regression, numpy.array([0.0, 1.0, 2.0, 4.0])
    )
    assert statistic(regression.y[None, :]).tolist() == [0.8]


def test_hedge_bounds_the_echo_and_skips_constant_draws():
    problem = binary_problem()
    bound = hypotheses.hedge(
        problem,
        problem.y.copy(),
        config(bootstrap=10),
        None,
        numpy.random.default_rng(0),
    )
    assert bound == 1.0
    y = pandas.Series([0.0, 1.0, 2.0, 3.0])
    regression = fitting.encode(y, "regression", config(folds=2), None, None)

    class Fixed:
        def integers(self, low, high, size):
            return numpy.zeros(size, dtype=int)

    assert (
        hypotheses.hedge(
            regression, regression.y, config(bootstrap=3), None, Fixed()
        )
        == -math.inf
    )


def test_gauge_skips_a_single_column_design():
    problem = binary_problem()
    assert hypotheses.gauge("c", numpy.ones((40, 1)), problem) == []
    [found] = hypotheses.gauge(
        "c", fitting.basis(pandas.Series(problem.y)), problem
    )
    assert (found.feature, found.mechanism, found.observed) == (
        "c",
        "skill",
        1.0,
    )
    assert found.uses_binned is False


def test_caps_share_the_association_ceiling():
    assert hypotheses.caps(config()) == {
        "skill": 0.9,
        "missingness": 0.5,
        "record_purity": 0.5,
        "echo": 0.5,
    }


def test_sift_stops_hypotheses_at_or_below_their_ceiling():
    cfg = config()
    pending = [
        hypotheses.Pending("a", "skill", 0.9, abs, False),
        hypotheses.Pending("b", "skill", 0.91, abs, False),
        hypotheses.Pending(
            "c", "missingness", 0.4, None, True, p_value=0.1, active=False
        ),
        hypotheses.Pending("d", "echo", 0.6, abs, False),
    ]
    assert hypotheses.sift(pending, cfg) == 2
    assert [h.active for h in pending] == [False, True, False, True]


def test_tested_lists_candidate_columns_once_in_order():
    pending = [
        hypotheses.Pending(name, "skill", 0.0, None, False)
        for name in ["b", "a & b", "b", "a"]
    ]
    assert hypotheses.tested(pending, ["a", "b"]) == ("b", "a")


def test_hypothesize_forms_purity_then_missingness():
    n = 40
    outcome = pandas.Series([0, 1] * (n // 2))
    series = pandas.Series(numpy.where(outcome == 1, numpy.nan, 1.0))
    codes, totals = report.tally(outcome, "classification")
    structured = hypotheses.hypothesize(
        "f",
        series,
        outcome,
        "classification",
        config(),
        codes,
        totals,
        structured=True,
    )
    assert [h.mechanism for h in structured] == ["record_purity", "missingness"]
    assert structured[0].observed == 1.0
    assert structured[1].observed == pytest.approx(1.0)
    analytic = hypotheses.hypothesize(
        "f",
        series,
        outcome,
        "classification",
        config(),
        codes,
        totals,
        structured=False,
    )
    assert analytic[1].statistic is None and analytic[1].active is False
    assert analytic[1].p_value < 1e-6
    full = pandas.Series(numpy.ones(n))
    assert (
        hypotheses.hypothesize(
            "f",
            full,
            outcome,
            "classification",
            config(),
            codes,
            totals,
            structured=False,
        )
        == []
    )


def test_propose_notes_too_many_pairs():
    rng = numpy.random.default_rng(0)
    frame = pandas.DataFrame(
        {f"c{i}": rng.integers(0, 2, 40) for i in range(4)}
    )
    frame["y"] = [0, 1] * 20
    problem = fitting.encode(
        frame["y"], "classification", config(folds=4), None, None
    )
    codes, totals = report.tally(frame["y"], "classification")
    pending, notes = hypotheses.propose(
        frame,
        "y",
        ["c0", "c1", "c2", "c3"],
        problem,
        config(),
        codes,
        totals,
        structured=False,
    )
    assert len(notes) == 1
    assert notes[0].startswith("6 pairs of two-level columns, more than the 4")
    pending, notes = hypotheses.propose(
        frame,
        "y",
        ["c0", "c1", "c2"],
        problem,
        config(),
        codes,
        totals,
        structured=False,
    )
    assert notes == ()
    assert {h.feature for h in pending if "&" in h.feature} == {
        "c0 & c1",
        "c0 & c2",
        "c1 & c2",
    }


def test_floor_routes_each_mechanism_to_its_bound():
    n = 40
    outcome = pandas.Series([0, 1] * (n // 2))
    frame = pandas.DataFrame({"f": numpy.where(outcome == 1, numpy.nan, 1.0)})
    problem = fitting.encode(
        outcome, "classification", config(folds=4), None, None
    )
    codes, totals = report.tally(outcome, "classification")
    cfg = config(bootstrap=20)
    purity = hypotheses.Pending("f", "record_purity", 1.0, None, True)
    assert hypotheses.floor(
        purity, problem, frame, codes, totals, cfg, None
    ) == pytest.approx(0.05 ** (1 / 20))
    missing = hypotheses.Pending("f", "missingness", 1.0, None, True)
    assert hypotheses.floor(
        missing, problem, frame, codes, totals, cfg, None
    ) == pytest.approx(1.0)
    echo = hypotheses.Pending(
        "f", "echo", 1.0, None, False, data=problem.y.copy()
    )
    assert (
        hypotheses.floor(echo, problem, frame, codes, totals, cfg, None) == 1.0
    )


# ---- permutation ------------------------------------------------------------


def test_resamples_is_the_smallest_count_that_reaches_the_level():
    assert permutation.resamples(0, Fraction(1, 20)) == 0
    assert permutation.resamples(-1, Fraction(1, 20)) == 0
    assert permutation.resamples(1, Fraction(1, 20)) == 20
    assert permutation.resamples(7, Fraction(1, 20)) == 363
    b = permutation.resamples(3, Fraction(1, 10))
    assert Fraction(1, b + 1) < Fraction(1, 10) / (3 * Fraction(11, 6))
    assert Fraction(1, b) >= Fraction(1, 10) / (3 * Fraction(11, 6))


def test_draw_permutations_counts_exceedances_and_stops_hopeless_ones():
    labels = numpy.array([0, 1] * 20)

    def agreement(versions):
        return (versions == labels[None, :]).mean(axis=1)

    def constant(versions):
        return numpy.ones(versions.shape[0])

    strong = hypotheses.Pending("s", "skill", 1.0, agreement, False)
    weak = hypotheses.Pending("w", "skill", 0.0, constant, True)
    fixed = hypotheses.Pending(
        "m", "missingness", 0.0, None, True, p_value=0.5, active=False
    )
    permutation.draw_permutations(
        [strong, weak, fixed],
        labels,
        labels,
        numpy.zeros(40, dtype=int),
        250,
        Fraction(1, 20),
        0,
    )
    assert strong.drawn == 250 and strong.exceed == 0 and strong.active
    assert weak.drawn == settings.BATCH and not weak.active
    assert weak.exceed == settings.BATCH
    assert (fixed.drawn, fixed.exceed) == (0, 0)


# ---- planning ---------------------------------------------------------------


def test_subsample_keeps_everything_for_a_small_table():
    rows, block = planning.subsample(settings.MAX_ROWS, None, 7, 0)
    assert rows.tolist() == list(range(settings.MAX_ROWS))
    assert block == 7


def test_subsample_strides_time_blocks():
    n = settings.MAX_ROWS * 2 + 1
    rows, block = planning.subsample(n, None, 9, 0)
    assert rows.tolist() == list(range(0, n, 3))
    assert block == 3
    assert planning.subsample(n, None, 3, 0)[1] == 2


def test_subsample_draws_whole_groups_until_enough_rows():
    groups = numpy.repeat(numpy.arange(300), 10)
    rows, block = planning.subsample(3000, groups, None, 0)
    assert len(rows) == settings.MAX_ROWS
    assert list(rows) == sorted(rows)
    chosen = set(groups[rows].tolist())
    assert len(chosen) == 200
    assert block is None


def test_subsample_draws_random_rows_otherwise():
    rows, block = planning.subsample(5000, None, None, 1)
    assert block is None
    assert len(rows) == settings.MAX_ROWS == len(set(rows.tolist()))
    assert rows.tolist() == planning.subsample(5000, None, None, 1)[0].tolist()


def test_load_store_returns_none_for_a_missing_or_broken_file(tmp_path):
    assert planning.load_store(tmp_path / "absent.json") is None
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    assert planning.load_store(broken) is None
    assert isinstance(planning.bundled(), certificates.Store)


def test_decline_lists_every_column_as_unchecked():
    demand = planning.Demand(100, 5, 40, 2, 80, ("a", "b"), 3.25)
    screen = planning.decline(config(), demand, 1.5, 10.0)
    assert screen.unchecked == ("a", "b")
    assert screen.rationale == "work limit for a 10 s time budget"
    assert screen.findings == ()
    assert screen.permutations == 0
    assert screen.notes[0].startswith(
        "5 hypotheses need about 80 cross-fitted fits (40 permutations for "
        "2 columns on 100 rows), predicted 3.2 s; the limit is 1.5 s."
    )
    assert set(screen.gates.values()) == {certificates.Authority.GATING}


def test_arrange_orders_by_time_or_keeps_groups():
    frame = pandas.DataFrame({"t": [3, 1, 2] * 20, "g": [1, 2, 3] * 20})
    ordered, groups, block = planning.arrange(frame, None, "t")
    assert ordered["t"].tolist() == [1] * 20 + [2] * 20 + [3] * 20
    assert groups is None and block == 3
    kept, groups, block = planning.arrange(frame, "g", "t")
    assert kept["t"].tolist() == frame["t"].tolist()
    assert groups.tolist() == [str(v) for v in frame["g"]]
    assert block is None
    small = frame.head(5)
    assert planning.arrange(small, "absent", "t")[2] == 2
    assert planning.arrange(small, None, "absent")[2] is None


def test_disclose_records_the_probe_and_missed_plants():
    demand = planning.Demand(10, 1, 0, 0, 0, (), 0.0)
    base = planning.decline(config(), demand, 1.0, 1.0)
    hit = planning.disclose(base, {"leak_copy": True}, "published")
    assert hit.notes[-1] == "published"
    assert dict(hit.probe) == {"leak_copy": True}
    missed = planning.disclose(
        base, {"leak_copy": False, "leak_threshold": False}, "published"
    )
    assert missed.notes[-1] == (
        "on these rows the screen did not find the planted leak_copy, "
        "leak_threshold: a leak of that kind may remain undetected"
    )
    skipped = planning.disclose(base, {}, "published", left=-1.0, limit=4.0)
    assert skipped.notes[-1] == (
        "probe not run: the planted copy did not fit the 0.0 s left of the "
        "screen's 4.0 s share of the time budget"
    )
    assert dict(skipped.probe) == {}
    assert (
        planning.disclose(base, {}, "p")
        .notes[-1]
        .endswith(
            "fit the 0.0 s left of the screen's 0.0 s share of the time budget"
        )
    )
