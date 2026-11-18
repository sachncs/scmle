"""Behavioural tests of the leakage submodules, imported by full name."""

import math

import numpy
import pandas
import pytest

from scmle.assurance.admissibility.leakage import assessment
from scmle.assurance.admissibility.leakage import conditional
from scmle.assurance.admissibility.leakage import fitting
from scmle.assurance.admissibility.leakage import missingness
from scmle.assurance.admissibility.leakage import settings
from scmle.assurance.admissibility.leakage import skill
from scmle.assurance.detectors import certificates
from scmle.contracts import errors
from scmle.contracts import qualification


def config(**changes):
    values = {"skill_ceiling": 0.9, "association_ceiling": 0.5}
    values.update(changes)
    return settings.Config(**values)


# ---- settings ---------------------------------------------------------------


def test_the_constants_hold_their_registered_values():
    assert settings.DETECTOR_ID == "f4-leakage-skill"
    assert settings.TASKS == ("classification", "regression")
    assert settings.MAX_LEVELS == 50
    assert settings.EXPECTED_FISHER == 5.0
    assert settings.TARGET_BINS == 4
    assert settings.MIN_STRATUM_ROWS == 2
    assert settings.SKILL == "leakage.skill"
    assert settings.CONDITIONAL == "leakage.conditional"
    assert settings.MISSING == "leakage.missingness"
    assert settings.MIN_CLASSES == 2
    assert settings.BINARY_CLASSES == 2
    assert len(settings.LEAK_DEFECTS) == 4


def test_a_default_config_has_the_documented_defaults():
    cfg = config()
    assert (cfg.alpha, cfg.permutations, cfg.folds) == (0.05, 199, 5)
    assert (cfg.confidence, cfg.bootstrap, cfg.min_rows) == (0.95, 200, 100)
    assert (cfg.max_leaf_nodes, cfg.min_samples_leaf) == (8, 5)
    assert (cfg.n_strata, cfg.run_seed) == (8, 0)


def test_the_certified_config_uses_the_certified_ceilings():
    assert settings.CERTIFIED.skill_ceiling == 0.95
    assert settings.CERTIFIED.association_ceiling == 0.9


def test_a_ceiling_of_exactly_one_is_accepted():
    assert config(skill_ceiling=1.0, association_ceiling=1.0).skill_ceiling == 1


@pytest.mark.parametrize("name", ["skill_ceiling", "association_ceiling"])
@pytest.mark.parametrize("value", [0.0, 1.0000001, -0.1])
def test_a_ceiling_outside_the_unit_interval_is_refused(name, value):
    with pytest.raises(errors.InputError, match=f"{name} must be within"):
        config(**{name: value})


@pytest.mark.parametrize("name", ["alpha", "confidence"])
@pytest.mark.parametrize("value", [0.0, 1.0])
def test_a_level_at_either_end_is_refused(name, value):
    with pytest.raises(
        errors.InputError, match=rf"{name} must be within \(0, 1\)"
    ):
        config(**{name: value})


def test_a_boolean_ceiling_is_refused():
    with pytest.raises(errors.InputError, match="finite number"):
        config(skill_ceiling=True)


def test_permutations_below_the_budget_minimum_are_refused():
    with pytest.raises(
        errors.InputError, match="permutations must be at least 20"
    ):
        config(permutations=19)


def test_permutations_that_cannot_reach_alpha_are_refused():
    with pytest.raises(errors.InputError, match="too few to reach alpha"):
        config(alpha=0.01, permutations=98)
    assert config(alpha=0.01, permutations=99).permutations == 99


@pytest.mark.parametrize(
    ("name", "low"),
    [
        ("folds", 1),
        ("bootstrap", 1),
        ("min_rows", 0),
        ("max_leaf_nodes", 1),
        ("min_samples_leaf", 0),
        ("n_strata", 1),
        ("run_seed", -1),
    ],
)
def test_a_count_below_its_minimum_is_refused(name, low):
    with pytest.raises(errors.InputError, match=name):
        config(**{name: low})
    assert getattr(config(**{name: low + 1}), name) == low + 1


# ---- fitting ----------------------------------------------------------------


def test_a_numeric_feature_is_one_column_keeping_nan():
    out = fitting.design(pandas.Series([1.0, numpy.nan, 3.0]))
    assert out.shape == (3, 1)
    assert out[0, 0] == 1.0 and math.isnan(out[1, 0]) and out[2, 0] == 3.0


def test_a_boolean_feature_is_one_hot_encoded():
    out = fitting.design(pandas.Series([True, False, True]))
    assert out.tolist() == [[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]


def test_a_categorical_feature_gets_a_missing_level():
    out = fitting.design(pandas.Series(["a", None, "b"], dtype="object"))
    # columns sorted: <missing>, a, b
    assert out.tolist() == [[0, 1, 0], [1, 0, 0], [0, 0, 1]]


def test_levels_beyond_the_most_frequent_are_pooled():
    values = [f"v{i}" for i in range(60)] + ["v0"] * 5
    out = fitting.design(pandas.Series(values))
    assert out.shape == (65, settings.MAX_LEVELS + 1)
    assert out.sum(axis=1).tolist() == [1.0] * 65


def test_vectorize_encodes_sorted_class_codes():
    out = fitting.vectorize(pandas.Series(["b", "a", "b"]), "classification")
    assert out.dtype == numpy.int64
    assert out.tolist() == [1, 0, 1]


def test_vectorize_keeps_a_regression_target_as_floats():
    out = fitting.vectorize(pandas.Series([1, 2, 3]), "regression")
    assert out.dtype == numpy.float64
    assert out.tolist() == [1.0, 2.0, 3.0]


def test_vectorize_refuses_an_unknown_task_missing_values_and_one_class():
    with pytest.raises(errors.InputError, match="task must be one of"):
        fitting.vectorize(pandas.Series([1, 2]), "ranking")
    with pytest.raises(errors.InputError, match="missing values"):
        fitting.vectorize(pandas.Series([1.0, None]), "regression")
    with pytest.raises(errors.InputError, match="two classes"):
        fitting.vectorize(pandas.Series([1, 1, 1]), "classification")


def test_permute_within_groups_moves_rows_only_inside_their_group():
    groups = numpy.array([0, 0, 0, 1, 1, 1, 2, 2])
    rng = numpy.random.default_rng(0)
    order = fitting.permute_within(numpy.zeros(8), rng, groups, block_size=3)
    assert sorted(order.tolist()) == list(range(8))
    assert (groups[order] == groups).all()


def test_permute_within_blocks_keeps_the_ragged_last_block():
    rng = numpy.random.default_rng(1)
    order = fitting.permute_within(numpy.zeros(7), rng, None, block_size=3)
    assert sorted(order[:3].tolist()) == [0, 1, 2]
    assert sorted(order[3:6].tolist()) == [3, 4, 5]
    assert order[6] == 6


def test_permute_without_structure_is_a_global_permutation():
    order = fitting.permute_within(
        numpy.zeros(50), numpy.random.default_rng(2), None, None
    )
    assert sorted(order.tolist()) == list(range(50))
    expected = numpy.random.default_rng(2).permutation(50)
    assert order.tolist() == expected.tolist()


def test_carve_caps_group_folds_at_the_group_count():
    y = numpy.array([0, 1] * 6)
    groups = numpy.repeat([0, 1, 2], 4)
    splits = fitting.carve(y, "classification", 5, groups, None, 0)
    assert len(splits) == 3
    for train, test in splits:
        assert set(groups[train]).isdisjoint(set(groups[test]))


def test_carve_with_blocks_is_contiguous_and_unshuffled():
    splits = fitting.carve(numpy.arange(10.0), "regression", 2, None, 5, 0)
    assert [t.tolist() for train, t in splits] == [
        [0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9],
    ]


def test_carve_stratifies_classification_folds():
    y = numpy.array([0] * 10 + [1] * 10)
    splits = fitting.carve(y, "classification", 5, None, None, 3)
    assert len(splits) == 5
    for train, test in splits:
        assert len(train) + len(test) == 20
        assert numpy.bincount(y[test]).tolist() == [2, 2]


def test_carve_shuffles_regression_folds_reproducibly():
    y = numpy.arange(20.0)
    first = fitting.carve(y, "regression", 4, None, None, 7)
    second = fitting.carve(y, "regression", 4, None, None, 7)
    assert [t.tolist() for train, t in first] == [
        t.tolist() for train, t in second
    ]
    assert next(t.tolist() for train, t in first) != [0, 1, 2, 3, 4]


def test_auc_counts_ties_as_half():
    truth = numpy.array([0, 1, 0, 1])
    assert fitting.auc(truth, numpy.array([0.0, 1.0, 0.0, 1.0])) == 1.0
    assert fitting.auc(truth, numpy.array([1.0, 0.0, 1.0, 0.0])) == 0.0
    assert fitting.auc(truth, numpy.ones(4)) == 0.5
    assert fitting.auc(
        numpy.array([0, 1, 1]), numpy.array([0.5, 0.5, 1.0])
    ) == (0.75)


def test_auc_needs_both_classes():
    with pytest.raises(errors.InputError, match="both classes are needed"):
        fitting.auc(numpy.ones(3, dtype=int), numpy.arange(3.0))


def test_predictiveness_of_a_regression_is_r_squared():
    y = numpy.array([1.0, 2.0, 3.0, 4.0])
    assert fitting.predictiveness(y.copy(), y, "regression") == 1.0
    assert fitting.predictiveness(numpy.full(4, 2.5), y, "regression") == 0.0
    pred = numpy.array([1.0, 2.0, 3.0, 5.0])
    assert fitting.predictiveness(pred, y, "regression") == pytest.approx(0.8)


def test_predictiveness_of_a_constant_target_is_refused():
    with pytest.raises(errors.InputError, match="constant target"):
        fitting.predictiveness(numpy.ones(3), numpy.ones(3), "regression")


def test_predictiveness_of_two_classes_uses_the_second_column():
    y = numpy.array([0, 1, 0, 1])
    pred = numpy.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.4, 0.6]])
    assert fitting.predictiveness(pred, y, "classification") == 1.0


def test_predictiveness_of_three_classes_is_the_macro_auc():
    y = numpy.array([0, 1, 2, 0, 1, 2])
    pred = numpy.eye(3)[y]
    assert fitting.predictiveness(pred, y, "classification") == 1.0


def test_out_of_fold_fills_absent_classes_with_zero():
    x = numpy.arange(6.0).reshape(-1, 1)
    y = numpy.array([0, 0, 0, 1, 1, 2])
    splits = [(numpy.arange(3), numpy.arange(3, 6))]
    out = fitting.out_of_fold(x, y, "classification", splits, config(), 0)
    assert out.shape == (6, 3)
    assert out[3:].tolist() == [[1.0, 0.0, 0.0]] * 3
    assert out[:3].tolist() == [[0.0, 0.0, 0.0]] * 3


def test_out_of_fold_regression_predicts_from_other_folds():
    x = numpy.arange(20.0).reshape(-1, 1)
    y = numpy.arange(20.0)
    splits = [(numpy.arange(10), numpy.arange(10, 20))]
    cfg = config(min_samples_leaf=10)
    out = fitting.out_of_fold(x, y, "regression", splits, cfg, 0)
    assert out[10:].tolist() == [4.5] * 10
    assert out[:10].tolist() == [0.0] * 10


def test_crossfit_finds_a_perfect_copy_of_the_target():
    y = numpy.array([0, 1] * 50)
    score, pred = fitting.crossfit(
        y.reshape(-1, 1).astype(float),
        y,
        "classification",
        config(),
        None,
        None,
        0,
    )
    assert score == 1.0
    assert pred.shape == (100, 2)


def test_rescore_returns_none_when_a_resample_misses_a_class():
    y = numpy.array([0] * 99 + [1])
    pred = numpy.column_stack([1 - y, y]).astype(float)

    class Fixed:
        def integers(self, low, high, size):
            return numpy.zeros(size, dtype=int)

    assert fitting.rescore(pred, y, "classification", None, Fixed()) is None


def test_rescore_resamples_whole_groups():
    y = numpy.array([0.0, 1.0, 2.0, 3.0])
    members = [numpy.array([0, 1]), numpy.array([2, 3])]

    class Fixed:
        def integers(self, low, high, size):
            assert (low, high, size) == (0, 2, 2)
            return numpy.array([0, 1])

    assert fitting.rescore(y.copy(), y, "regression", members, Fixed()) == 1.0


def test_lower_bound_is_minus_infinity_when_no_draw_is_computable():
    y = numpy.ones(10)
    bound = fitting.lower_bound(
        y,
        y,
        "regression",
        config(bootstrap=5),
        None,
        numpy.random.default_rng(0),
    )
    assert bound == -math.inf


def test_lower_bound_of_perfect_predictions_is_one():
    y = numpy.arange(30.0)
    groups = numpy.repeat(numpy.arange(10), 3)
    bound = fitting.lower_bound(
        y,
        y,
        "regression",
        config(bootstrap=10),
        groups,
        numpy.random.default_rng(0),
    )
    assert bound == 1.0


def test_prepare_refuses_groups_together_with_blocks():
    with pytest.raises(errors.InputError, match="groups or block_size"):
        fitting.prepare(
            pandas.Series([1.0]),
            pandas.Series([1.0]),
            "regression",
            pandas.Series([0]),
            2,
        )


def test_prepare_encodes_the_name_target_groups_and_design():
    name, y, groups, x = fitting.prepare(
        pandas.Series([1.0, 2.0], name=3),
        pandas.Series(["n", "y"]),
        "classification",
        pandas.Series(["g", "h"]),
        None,
    )
    assert name == "3"
    assert y.tolist() == [0, 1]
    assert groups.tolist() == ["g", "h"]
    assert x.tolist() == [[1.0], [2.0]]
    assert (
        fitting.prepare(
            pandas.Series([1.0, 2.0], name="f"),
            pandas.Series([1.0, 2.0]),
            "regression",
            None,
            2,
        )[2]
        is None
    )


def test_measure_returns_one_null_skill_per_permutation():
    y = numpy.array([0, 1] * 30)
    x = y.reshape(-1, 1).astype(float)
    cfg = config(permutations=20, min_samples_leaf=2)

    def draw(rng):
        return x, y[rng.permutation(len(y))]

    observed, pred, null = fitting.measure(
        x, y, "classification", cfg, None, None, "p", draw
    )
    assert observed == 1.0
    assert pred.shape == (60, 2)
    assert len(null) == 20
    assert max(null) < 1.0


def test_significance_counts_ties_as_exceeding():
    assert fitting.significance(0.5, [0.1, 0.5, 0.9]) == 0.75
    assert fitting.significance(1.0, [0.1, 0.2, 0.3]) == 0.25
    assert fitting.significance(0.0, []) == 1.0


def test_sufficient_needs_rows_and_enough_of_each_class():
    cfg = config(min_rows=10, folds=3)
    assert not fitting.sufficient(numpy.zeros(9), "regression", cfg)
    assert fitting.sufficient(numpy.zeros(10), "regression", cfg)
    assert fitting.sufficient(
        numpy.array([0] * 7 + [1] * 3), "classification", cfg
    )
    assert not fitting.sufficient(
        numpy.array([0] * 8 + [1] * 2), "classification", cfg
    )


# ---- conditional ------------------------------------------------------------


def test_strata_are_one_when_there_are_no_other_features():
    labels = conditional.strata_labels(
        pandas.Series([1.0, 2.0, 3.0]),
        pandas.DataFrame(index=range(3)),
        config(),
    )
    assert labels.dtype == numpy.int64
    assert labels.tolist() == [0, 0, 0]


def test_strata_follow_the_other_features_for_a_numeric_feature():
    others = pandas.DataFrame({"z": [0.0] * 10 + [1.0] * 10})
    feature = pandas.Series([0.0] * 10 + [5.0] * 9 + [numpy.nan])
    labels = conditional.strata_labels(feature, others, config(n_strata=2))
    assert len(set(labels[:10])) == 1 and len(set(labels[10:])) == 1
    assert labels[0] != labels[10]


def test_strata_of_a_categorical_feature_come_from_a_classifier():
    others = pandas.DataFrame({"z": [0.0] * 10 + [1.0] * 10})
    feature = pandas.Series(["a"] * 10 + ["b"] * 10)
    labels = conditional.strata_labels(feature, others, config(n_strata=4))
    assert len(set(labels.tolist())) == 2


def test_scramble_moves_rows_only_within_their_stratum():
    members = [numpy.array([0, 2, 4]), numpy.array([1, 3])]
    order = conditional.scramble(6, members, numpy.random.default_rng(0))
    assert sorted(order[[0, 2, 4]].tolist()) == [0, 2, 4]
    assert sorted(order[[1, 3]].tolist()) == [1, 3]
    assert order[5] == 5


# ---- skill ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("suspect", "low", "gate", "expected"),
    [
        (
            True,
            False,
            certificates.Authority.GATING,
            qualification.Decision.QUARANTINE,
        ),
        (
            True,
            True,
            certificates.Authority.GATING,
            qualification.Decision.QUARANTINE,
        ),
        (
            True,
            False,
            certificates.Authority.DESCRIPTIVE,
            qualification.Decision.WARN,
        ),
        (
            False,
            True,
            certificates.Authority.GATING,
            qualification.Decision.WARN,
        ),
        (
            False,
            False,
            certificates.Authority.GATING,
            qualification.Decision.PASS,
        ),
        (
            False,
            False,
            certificates.Authority.DESCRIPTIVE,
            qualification.Decision.PASS,
        ),
    ],
)
def test_decide_picks_and_demotes_the_decision(suspect, low, gate, expected):
    assert skill.decide(suspect, low, gate) is expected


def test_summarize_writes_the_skill_check():
    cfg = config(run_seed=3)
    check = skill.summarize(
        "leakage.skill.f",
        cfg,
        (0.8, [0.4, 0.6], 0.7, 0.05),
        40,
        qualification.Decision.WARN,
        qualification.Subset.CONFIRMATION,
    )
    assert check.identifier == "leakage.skill.f"
    assert check.statistic == 0.8
    assert check.p_value == 0.05
    assert check.interval == (0.7, 0.8)
    assert check.effect == pytest.approx(0.3)
    assert check.n == 40
    assert check.decision is qualification.Decision.WARN
    assert check.subset is qualification.Subset.CONFIRMATION
    assert check.seeds[0] == 3 and check.seeds[2] == 1
    assert check.tier is qualification.Tier.UNVERIFIED
    assert "permuted within its blocks" in check.null


def test_summarize_drops_the_interval_for_an_infinite_bound():
    check = skill.summarize(
        "p",
        config(),
        (0.5, [0.5], -math.inf, 1.0),
        4,
        qualification.Decision.PASS,
        qualification.Subset.EXPLORATION,
    )
    assert check.interval is None
    assert check.effect == 0.0


def test_summarize_caps_the_interval_below_at_the_bound():
    check = skill.summarize(
        "p",
        config(),
        (0.5, [0.0], 0.6, 0.5),
        4,
        qualification.Decision.PASS,
        qualification.Subset.EXPLORATION,
    )
    assert check.interval == (0.6, 0.6)


# ---- missingness ------------------------------------------------------------


def test_cramers_v_is_zero_for_empty_or_single_line_tables():
    assert missingness.cramers_v(numpy.zeros((2, 2))) == 0.0
    assert missingness.cramers_v(numpy.array([[3, 4]])) == 0.0
    assert missingness.cramers_v(numpy.array([[3], [4]])) == 0.0


def test_cramers_v_of_a_perfect_and_a_null_association():
    assert missingness.cramers_v(
        numpy.array([[5, 0], [0, 5]])
    ) == pytest.approx(1.0)
    assert missingness.cramers_v(numpy.array([[5, 5], [5, 5]])) == 0.0
    # chi2 = 400/21 on 40 rows of a 2x2 table
    v = missingness.cramers_v(numpy.array([[10, 10], [5, 15]]))
    assert v == pytest.approx(
        math.sqrt((40 * (150 - 50) ** 2 / (20 * 20 * 15 * 25)) / 40)
    )


def test_bucket_keeps_classes_and_bins_a_continuous_target():
    classes = missingness.bucket(pandas.Series([1, 2, 1]), "classification")
    assert classes.dtype == object and classes.tolist() == [1, 2, 1]
    bins = missingness.bucket(pandas.Series(numpy.arange(8.0)), "regression")
    assert bins.nunique() == 4
    constant = missingness.bucket(
        pandas.Series([1.0] * 3 + [2.0]), "regression"
    )
    assert constant.nunique() == 2


def test_missingness_refuses_mismatched_lengths():
    with pytest.raises(errors.InputError, match="same length"):
        missingness.missingness_test(
            pandas.Series([1.0]),
            pandas.Series([1, 2]),
            "classification",
            config(),
        )


def test_missingness_is_none_for_never_or_always_missing_or_one_class():
    target = pandas.Series([0, 1, 0, 1])
    full = pandas.Series([1.0, 2.0, 3.0, 4.0])
    empty = pandas.Series([numpy.nan] * 4)
    some = pandas.Series([numpy.nan, 1.0, 2.0, 3.0])
    assert (
        missingness.missingness_test(full, target, "classification", config())
        is None
    )
    assert (
        missingness.missingness_test(empty, target, "classification", config())
        is None
    )
    assert (
        missingness.missingness_test(
            some, pandas.Series([1] * 4), "classification", config()
        )
        is None
    )


def test_a_small_table_uses_fisher_and_reports_the_counts():
    feature = pandas.Series([numpy.nan, numpy.nan, 1.0, 2.0], name="f")
    target = pandas.Series([1, 1, 0, 0])
    out = missingness.missingness_test(
        feature, target, "classification", config()
    )
    assert out.method == "fisher"
    assert out.p_value == pytest.approx(1 / 3)
    assert out.n_missing == 2 and out.n == 4
    assert out.feature == "f"
    assert not out.suspect
    assert out.check.identifier == "leakage.missingness.f"
    assert out.check.decision is qualification.Decision.WARN
    assert out.check.seeds[2] == 0
    assert out.check.interval is None
    assert out.check.statistic == out.cramers_v == out.check.effect


def test_a_large_association_is_a_gated_suspect_by_chi_square():
    n = 200
    target = pandas.Series([0, 1] * (n // 2))
    feature = pandas.Series(numpy.where(target == 1, numpy.nan, 1.0), name="g")
    cfg = config(min_rows=n)
    out = missingness.missingness_test(
        feature, target, "classification", cfg, certificates.Authority.GATING
    )
    assert out.method == "chi2"
    assert out.cramers_v == pytest.approx(1.0)
    assert out.suspect
    assert out.check.decision is qualification.Decision.QUARANTINE
    demoted = missingness.missingness_test(
        feature, target, "classification", cfg
    )
    assert demoted.check.decision is qualification.Decision.WARN
    weak = missingness.missingness_test(
        feature,
        target,
        "classification",
        config(min_rows=n + 1),
        certificates.Authority.GATING,
    )
    assert not weak.suspect


def test_an_association_below_the_ceiling_is_not_a_suspect():
    rng = numpy.random.default_rng(0)
    target = pandas.Series(rng.integers(0, 2, 400))
    feature = pandas.Series(rng.normal(size=400))
    feature[rng.random(400) < 0.3] = numpy.nan
    out = missingness.missingness_test(
        feature,
        target,
        "classification",
        config(),
        certificates.Authority.GATING,
    )
    assert not out.suspect
    assert out.check.decision is qualification.Decision.PASS


# ---- assessment -------------------------------------------------------------


def make_check(name):
    return qualification.Check(
        identifier=name,
        null="n",
        statistic=0.0,
        p_value=1.0,
        interval=None,
        effect=0.0,
        n=1,
        decision=qualification.Decision.PASS,
        subset=qualification.Subset.EXPLORATION,
        seeds=(0, name, 0),
        tier=qualification.Tier.UNVERIFIED,
    )


def make_assessment(name, bound=0.5, with_missing=True, with_conditional=False):
    test = skill.Skill(
        name, 0.7, 0.5, bound, 0.01, 50, False, True, make_check(f"s.{name}")
    )
    cond = (
        conditional.Conditional(
            name, 0.7, 0.5, 0.02, 3, True, make_check(f"c.{name}")
        )
        if with_conditional
        else None
    )
    miss = (
        missingness.Missingness(
            name, "chi2", 0.03, 0.4, 5, 50, False, make_check(f"m.{name}")
        )
        if with_missing
        else None
    )
    return assessment.Assessment(name, test, cond, miss, True, with_conditional)


def test_assessment_checks_skip_the_tests_that_did_not_run():
    one = make_assessment("a", with_missing=False)
    assert [c.identifier for c in one.checks] == ["s.a"]
    full = make_assessment("b", with_conditional=True)
    assert [c.identifier for c in full.checks] == ["s.b", "c.b", "m.b"]


def test_assessments_report_suspects_checks_and_a_document():
    report = assessment.Assessments(
        (
            make_assessment("a", bound=-math.inf, with_missing=False),
            make_assessment("b", with_conditional=True),
        ),
        certificates.Authority.DESCRIPTIVE,
    )
    assert report.suspects == {"a": True, "b": True}
    assert [c.identifier for c in report.checks] == ["s.a", "s.b", "c.b", "m.b"]
    doc = report.to_dict()
    assert doc["note"] == "statistical suspicion only; not proof of leakage"
    assert doc["gate"] == "descriptive"
    first, second = doc["features"]
    assert first == {
        "feature": "a",
        "suspect": True,
        "escalated": False,
        "skill": 0.7,
        "skill_lower_bound": None,
        "skill_p_value": 0.01,
        "low_power": False,
        "conditional_p_value": None,
        "missingness_p_value": None,
        "missingness_cramers_v": None,
    }
    assert second["skill_lower_bound"] == 0.5
    assert second["conditional_p_value"] == 0.02
    assert second["missingness_p_value"] == 0.03
    assert second["missingness_cramers_v"] == 0.4
    assert second["escalated"] is True
