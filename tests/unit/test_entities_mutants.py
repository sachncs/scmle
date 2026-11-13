"""Targeted tests for entity-resolution behaviours the main suite left open."""

import math

import numpy
import pandas
import pytest

from benchmarks.qualification import ingest_worlds
from scmle.contracts import qualification
from scmle.data.ingest import entities

FIELDS = [
    entities.Field("first"),
    entities.Field("last"),
    entities.Field("dob", fuzzy=False),
    entities.Field("addr"),
]
PASSES = [
    entities.Pass("last", "soundex"),
    entities.Pass("dob"),
    entities.Pass("first", "prefix", 3),
]
SMALL = pandas.DataFrame(
    {
        "first": ["anna", "anna", "bob", "bobby", "carl", "carl", None, "dina"],
        "last": [
            "smith",
            "smyth",
            "jones",
            "jones",
            "brown",
            "brown",
            "brown",
            "white",
        ],
        "dob": [
            "1990",
            "1990",
            "1980",
            "1980",
            "1970",
            "1971",
            "1970",
            None,
        ],
    }
)
SMALL_FIELDS = [
    entities.Field("first"),
    entities.Field("last"),
    entities.Field("dob", fuzzy=False),
]
SMALL_PASSES = [
    entities.Pass("last", "soundex"),
    entities.Pass("dob", "prefix", 3),
]
SMALL_ENTITY = [0, 0, 1, 1, 2, 2, 2, 3]
LEVELS = numpy.array(
    [
        [2, 2, 2],
        [2, 1, 2],
        [0, 0, 0],
        [0, 0, 2],
        [1, 0, 0],
        [2, 2, -1],
        [-1, 0, 0],
        [0, 0, 0],
        [0, 1, 0],
        [2, 2, 2],
        [0, 0, 0],
        [0, 0, 0],
    ],
    dtype=numpy.int8,
)


def small_labeler(i, j):
    return SMALL_ENTITY[i] == SMALL_ENTITY[j]


def test_normalize_handles_missing_values_and_punctuation():
    assert entities.normalize(None) == ""
    assert entities.normalize(float("nan")) == ""
    assert entities.normalize(" A-b 1_") == "ab1"
    assert entities.normalize(12) == "12"
    assert entities.normalize(1.5) == "15"
    assert entities.normalize("O'Brien") == "obrien"


def test_jaro_matches_published_values():
    assert entities.jaro("dixon", "dicksonx") == pytest.approx(0.7667, abs=1e-4)
    assert entities.jaro("crate", "trace") == pytest.approx(0.7333, abs=1e-4)
    assert entities.jaro("jones", "johnson") == pytest.approx(0.7905, abs=1e-4)
    assert entities.jaro("dwayne", "duane") == pytest.approx(0.8222, abs=1e-4)
    assert entities.jaro("abcd", "abdc") == pytest.approx(0.9167, abs=1e-4)
    assert entities.jaro("abc", "xbc") == pytest.approx(0.7778, abs=1e-4)
    assert entities.jaro("abcde", "abcdef") == pytest.approx(0.9444, abs=1e-4)


def test_jaro_is_zero_without_a_match_and_for_one_empty_string():
    assert entities.jaro("a", "b") == 0.0
    assert entities.jaro("xy", "yx") == 0.0
    assert entities.jaro("xyz", "yxz") == pytest.approx((1 / 3 + 1 / 3 + 1) / 3)
    assert entities.jaro("abc", "") == 0.0
    assert entities.jaro("", "abc") == 0.0
    assert entities.jaro("", "") == 1.0
    assert entities.jaro("abcdef", "uvwxyz") == 0.0


def test_jaro_looks_only_within_the_match_window():
    assert entities.jaro("abcdefgh", "xbcdefgh") == pytest.approx(
        (7 / 8 + 7 / 8 + 1.0) / 3
    )
    assert entities.jaro("abcdefghij", "jbcdefghia") == pytest.approx(
        (8 / 10 + 8 / 10 + 1.0) / 3
    )
    assert entities.jaro("aaaaabbbbb", "bbbbbaaaaa") == pytest.approx(
        (8 / 10 + 8 / 10 + 4 / 8) / 3
    )
    assert entities.jaro("abcdef", "defabc") == 0.0


def test_jaro_winkler_rewards_a_common_prefix_of_up_to_four():
    assert entities.jaro_winkler("dixon", "dicksonx") == pytest.approx(
        0.8133, abs=1e-4
    )
    assert entities.jaro_winkler("dwayne", "duane") == pytest.approx(
        0.84, abs=1e-4
    )
    assert entities.jaro_winkler("crate", "trace") == pytest.approx(
        0.7333, abs=1e-4
    )
    assert entities.jaro_winkler("abcdef", "abcxyz") == pytest.approx(
        0.7667, abs=1e-4
    )
    assert entities.jaro_winkler("abcde", "abcdef") == pytest.approx(
        0.9667, abs=1e-4
    )
    assert entities.jaro_winkler("abcdxyz", "abcdxyw") == pytest.approx(
        entities.jaro("abcdxyz", "abcdxyw")
        + 4 * 0.1 * (1 - entities.jaro("abcdxyz", "abcdxyw"))
    )


def test_jaro_winkler_of_strings_shorter_than_the_prefix_window():
    assert entities.jaro_winkler("ab", "abc") == pytest.approx(
        entities.jaro("ab", "abc") + 2 * 0.1 * (1 - entities.jaro("ab", "abc"))
    )
    assert entities.jaro_winkler("a", "") == 0.0
    assert entities.jaro_winkler("", "") == 1.0


def test_soundex_known_codes():
    assert entities.soundex("tymczak") == "T522"
    assert entities.soundex("pfister") == "P236"
    assert entities.soundex("honeyman") == "H555"
    assert entities.soundex("rubin") == "R150"
    assert entities.soundex("jackson") == "J250"
    assert entities.soundex("a") == "A000"
    assert entities.soundex("lee") == "L000"
    assert entities.soundex("Ab1") == "A100"
    assert entities.soundex("ashcraft") == "A261"
    assert entities.soundex("bbb") == "B000"
    assert entities.soundex("bpb") == "B000"


def test_pass_keys_by_kind():
    assert entities.Pass("f").key("Ab-1") == "ab1"
    assert entities.Pass("f", "prefix").key("abcdef") == "abc"
    assert entities.Pass("f", "prefix", 2).key("abcdef") == "ab"
    assert entities.Pass("f", "soundex").key("Robert") == "R163"
    assert entities.Pass("f").size == 3
    assert entities.Pass("f").kind == "exact"


def test_pass_key_of_a_blank_value_is_empty_for_every_kind():
    for kind in ("exact", "prefix", "soundex"):
        rule = entities.Pass("f", kind)
        assert rule.key(None) == ""
        assert rule.key(float("nan")) == ""
        assert rule.key("--") == ""


def test_an_unknown_blocking_kind_is_refused():
    with pytest.raises(ValueError, match="unknown blocking kind: metaphone"):
        entities.Pass("f", "metaphone").key("abc")


def test_blocks_at_the_limit_are_compared_and_larger_ones_are_skipped():
    frame = pandas.DataFrame({"k": ["a", "a", "a", "b", "b"]})
    rule = [entities.Pass("k")]
    assert entities.candidate_pairs(frame, rule, max_block=3) == (
        [(0, 1), (0, 2), (1, 2), (3, 4)],
        0,
    )
    assert entities.candidate_pairs(frame, rule, max_block=2) == (
        [(3, 4)],
        1,
    )


def test_every_skipped_block_is_counted_and_later_blocks_still_run():
    frame = pandas.DataFrame(
        {"k": ["a", "a", "a", "b", "b", "b", "c", "c"], "j": list("xxyyzzzz")}
    )
    pairs, skipped = entities.candidate_pairs(
        frame, [entities.Pass("k")], max_block=2
    )
    assert pairs == [(6, 7)]
    assert skipped == 2
    both = entities.candidate_pairs(
        frame, [entities.Pass("k"), entities.Pass("j")], max_block=2
    )
    assert both == ([(0, 1), (2, 3), (6, 7)], 3)


def test_blank_keys_are_not_blocked_and_pairs_are_not_repeated():
    frame = pandas.DataFrame(
        {"k": ["a", None, "a", "", "a"], "j": [1, 1, 1, 1, 2]}
    )
    pairs, skipped = entities.candidate_pairs(
        frame, [entities.Pass("k"), entities.Pass("k", "prefix", 1)]
    )
    assert pairs == [(0, 2), (0, 4), (2, 4)]
    assert skipped == 0


def test_candidate_pairs_default_block_limit_is_two_hundred():
    frame = pandas.DataFrame({"k": ["a"] * 201 + ["b"] * 200})
    pairs, skipped = entities.candidate_pairs(frame, [entities.Pass("k")])
    assert skipped == 1
    assert len(pairs) == 200 * 199 // 2


def test_compare_levels_for_each_field_and_pair():
    pairs = entities.candidate_pairs(SMALL, SMALL_PASSES)[0]
    assert pairs == [(0, 1), (2, 3), (4, 5), (4, 6), (5, 6)]
    levels = entities.compare(SMALL, SMALL_FIELDS, pairs)
    assert levels.dtype == numpy.int8
    assert levels.tolist() == [
        [2, 1, 2],
        [1, 2, 2],
        [2, 2, 0],
        [-1, 2, 2],
        [-1, 2, 0],
    ]


def test_a_missing_value_on_either_side_gives_missing():
    frame = pandas.DataFrame(
        {"a": ["x", None, "x", None], "b": ["x", "x", None, None]}
    )
    levels = entities.compare(
        frame,
        [entities.Field("a"), entities.Field("b")],
        [(0, 1), (0, 2), (1, 3)],
    )
    assert levels.tolist() == [[-1, 2], [2, -1], [-1, -1]]


def test_exact_fields_never_report_a_similar_level():
    frame = pandas.DataFrame(
        {"a": ["martha", "marhta"], "b": ["martha", "marhta"]}
    )
    levels = entities.compare(
        frame, [entities.Field("a"), entities.Field("b", fuzzy=False)], [(0, 1)]
    )
    assert levels.tolist() == [[1, 0]]


def test_similarity_cut_is_inclusive_at_the_threshold(monkeypatch):
    frame = pandas.DataFrame({"a": ["xx", "yy"]})
    monkeypatch.setattr(
        entities, "jaro_winkler", lambda left, right: entities.SIMILAR_AT
    )
    assert entities.compare(
        frame, [entities.Field("a")], [(0, 1)]
    ).tolist() == [[1]]
    monkeypatch.setattr(
        entities, "jaro_winkler", lambda left, right: entities.SIMILAR_AT - 1e-9
    )
    assert entities.compare(
        frame, [entities.Field("a")], [(0, 1)]
    ).tolist() == [[0]]


def test_weights_add_zero_for_a_missing_field():
    model = entities.fit_em(LEVELS)
    weights = model.weights(LEVELS)
    assert weights.round(4).tolist() == [
        42.4481,
        23.9316,
        -59.187,
        -36.4481,
        -56.6021,
        39.4481,
        -39.4778,
        -59.187,
        -38.4481,
        42.4481,
        -59.187,
        -59.187,
    ]
    only = numpy.array([[-1, -1, -1]], dtype=numpy.int8)
    assert model.weights(only).tolist() == [0.0]
    one = numpy.array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]], dtype=numpy.int8)
    single = model.weights(one)
    expected = [
        math.log2(model.m[col][2] / model.u[col][2]) for col in range(3)
    ]
    assert single.tolist() == pytest.approx(expected)


def test_em_converges_to_a_known_fixed_point():
    model = entities.fit_em(LEVELS)
    assert model.iterations == 6
    assert model.prior == pytest.approx(1 / 3, abs=1e-6)
    assert model.m.round(6).tolist() == [
        [1e-6, 1e-6, 1.0],
        [1e-6, 0.25, 0.75],
        [1e-6, 1e-6, 1.0],
    ]
    assert model.u.round(6).tolist() == [
        [0.857143, 0.142857, 1e-6],
        [0.875, 0.125, 1e-6],
        [0.875, 1e-6, 0.125],
    ]


def test_em_stops_at_the_iteration_cap_and_reports_it():
    zero = entities.fit_em(LEVELS, iterations=0)
    assert zero.iterations == 0
    assert zero.prior == 0.05
    assert zero.m.tolist() == [[0.05, 0.15, 0.8]] * 3
    assert zero.u.tolist() == [[0.9, 0.07, 0.03]] * 3
    one = entities.fit_em(LEVELS, iterations=1)
    assert one.iterations == 1
    assert one.prior == pytest.approx(0.33040558, abs=1e-7)
    assert one.m.round(6).tolist() == [
        [0.001185, 8.8e-05, 0.998727],
        [0.001226, 0.249197, 0.749577],
        [0.000299, 1e-06, 0.999701],
    ]
    assert one.u.round(6).tolist() == [
        [0.852175, 0.142091, 0.005734],
        [0.870569, 0.125943, 0.003488],
        [0.873893, 1e-06, 0.126107],
    ]
    three = entities.fit_em(LEVELS, iterations=3)
    assert three.iterations == 3
    assert three.m.round(6).tolist() == [
        [1e-06, 1e-06, 1.0],
        [1e-06, 0.249997, 0.750003],
        [1e-06, 1e-06, 1.0],
    ]


def test_a_loose_tolerance_stops_earlier():
    assert entities.fit_em(LEVELS, tol=1e-3).iterations == 3
    assert entities.fit_em(LEVELS, tol=1e-8).iterations == 6
    assert entities.fit_em(LEVELS, iterations=400, tol=0.0).iterations == 400


def test_parameters_are_floored_and_the_prior_is_kept_inside_the_unit_interval():
    agree = entities.fit_em(numpy.full((6, 2), 2, dtype=numpy.int8))
    assert agree.prior <= 1 - 1e-6
    assert agree.m.min() == 1e-6
    assert agree.u.min() == 1e-6
    assert agree.m.max() <= 1.0
    disagree = entities.fit_em(numpy.zeros((6, 2), dtype=numpy.int8))
    assert disagree.prior >= 1e-6
    assert disagree.m.min() == 1e-6
    assert disagree.u.min() == 1e-6
    assert disagree.m[0].tolist() == [1.0, 1e-6, 1e-6]


def test_a_column_that_is_always_missing_keeps_the_starting_rates():
    levels = numpy.array([[2, -1], [0, -1], [2, -1], [0, -1]], dtype=numpy.int8)
    model = entities.fit_em(levels)
    assert model.m[1].tolist() == pytest.approx([0.05, 0.15, 0.8])
    assert model.u[1].tolist() == pytest.approx([0.9, 0.07, 0.03])


def test_rates_are_updated_from_even_a_single_seen_pair():
    levels = numpy.array([[2, -1], [0, -1], [2, -1], [0, 1]], dtype=numpy.int8)
    model = entities.fit_em(levels)
    assert model.iterations == 2
    assert model.prior == pytest.approx(0.29425633, abs=1e-6)
    assert model.m.round(6).tolist() == [
        [0.007767, 1e-06, 0.992233],
        [1e-06, 1.0, 1e-06],
    ]
    assert model.u.round(6).tolist() == [
        [0.705234, 1e-06, 0.294766],
        [1e-06, 1.0, 1e-06],
    ]


def test_em_on_separated_classes_is_deterministic_and_pins_the_prior():
    first = entities.fit_em(LEVELS)
    second = entities.fit_em(LEVELS)
    assert first.prior == second.prior
    assert first.m.tolist() == second.m.tolist()


def test_components_use_the_smallest_index_as_the_group_id():
    assert entities.components(6, [(5, 4), (4, 3), (1, 0)]) == [
        0,
        0,
        2,
        3,
        3,
        3,
    ]
    assert entities.components(4, []) == [0, 1, 2, 3]
    assert entities.components(0, []) == []
    assert entities.components(5, [(4, 0), (3, 4)]) == [0, 1, 2, 0, 0]
    assert entities.components(7, [(6, 5), (5, 4), (3, 2), (4, 3)]) == [
        0,
        1,
        2,
        2,
        2,
        2,
        2,
    ]


def test_geometric_bands_thin_out_at_the_top():
    def sizes(total, bands):
        return [
            len(b) for b in entities.geometric_bands(numpy.arange(total), bands)
        ]

    assert sizes(1024, 4) == [512, 256, 128, 128]
    assert sizes(100, 5) == [50, 25, 13, 6, 6]
    assert sizes(10, 10) == [5, 3, 1, 1]
    assert sizes(100, 1) == [100]
    assert sizes(100, 2) == [50, 50]
    assert sizes(100, 3) == [50, 25, 25]
    assert sizes(0, 4) == []
    assert sizes(1, 4) == [1]


def test_geometric_bands_cover_the_order_without_overlap():
    order = numpy.arange(37)[::-1]
    bands = entities.geometric_bands(order, 5)
    assert numpy.concatenate(bands).tolist() == order.tolist()
    assert bands[0][0] == order[0]


def test_band_and_estimate_serialize_their_fields():
    band = entities.BandEstimate(-1.0, 2.0, 10, 4, 3)
    assert band.to_dict() == {
        "low": -1.0,
        "high": 2.0,
        "pairs": 10,
        "sampled": 4,
        "matches": 3,
    }
    estimate = entities.Estimate(
        (0.5, 0.1, 0.9), (0.6, 0.2, 1.0), (band,), "rv"
    )
    assert estimate.to_dict() == {
        "precision": [0.5, 0.1, 0.9],
        "recall": [0.6, 0.2, 1.0],
        "bands": [band.to_dict()],
        "reviewer": "rv",
    }


def stratified_inputs():
    rng = numpy.random.default_rng(5)
    weights = rng.normal(size=300)
    pairs = [(i, i + 1) for i in range(300)]

    def labeler(i, j):
        return (i * 7 + 3) % 5 == 0 or bool(weights[i] > 1.0)

    return weights, pairs, labeler


def band_rows(estimate):
    return [
        (round(b.low, 4), round(b.high, 4), b.pairs, b.sampled, int(b.matches))
        for b in estimate.bands
    ]


def test_the_clerical_sample_and_intervals_are_pinned():
    weights, pairs, labeler = stratified_inputs()
    estimate = entities.stratified_estimate(
        weights, pairs, 0.5, labeler, "rev", 6, 3
    )
    assert estimate.reviewer == "rev"
    assert estimate.precision == pytest.approx(
        (0.5585714285714285, 0.43954091782346444, 0.7421331044644154)
    )
    assert estimate.recall == pytest.approx(
        (0.6099843993759749, 0.3697416238335117, 0.9229204680275614)
    )
    assert band_rows(estimate) == [
        (-2.3979, -0.07, 150, 6, 0),
        (-0.0631, 0.6308, 75, 6, 1),
        (0.6389, 1.0929, 38, 6, 1),
        (1.136, 1.5484, 19, 6, 6),
        (1.5908, 1.8533, 9, 6, 6),
        (1.8756, 1.9505, 5, 5, 5),
        (2.16, 2.4317, 2, 2, 2),
        (2.6926, 2.6926, 1, 1, 1),
        (2.7736, 2.7736, 1, 1, 1),
    ]


def test_a_smaller_sample_per_band_changes_the_labels_drawn():
    weights, pairs, labeler = stratified_inputs()
    estimate = entities.stratified_estimate(
        weights, pairs, 0.5, labeler, "rev", 4, 3
    )
    assert estimate.precision == pytest.approx(
        (0.6915555555555556, 0.4954203340508876, 0.8965612056857264)
    )
    assert [b.sampled for b in estimate.bands] == [4, 4, 4, 4, 4, 4, 2, 1, 1]


def test_fewer_bands_and_draws_are_honoured():
    weights, pairs, labeler = stratified_inputs()
    estimate = entities.stratified_estimate(
        weights, pairs, 0.5, labeler, "rev", 6, 4, bands=5, draws=50
    )
    assert len(estimate.bands) == 5
    assert estimate.precision == pytest.approx(
        (0.6390476190476191, 0.47991258476574156, 0.8442908626724664)
    )
    assert estimate.recall == pytest.approx(
        (0.49851411589895994, 0.35567337550585165, 0.8052533895434254)
    )


def test_a_threshold_equal_to_a_band_edge_counts_that_band_as_linked():
    weights, pairs, labeler = stratified_inputs()
    everything = entities.stratified_estimate(
        weights, pairs, -math.inf, labeler, "rev", 6, 3
    )
    edge = everything.bands[3].low
    at = entities.stratified_estimate(
        weights, pairs, edge, labeler, "rev", 6, 3
    )
    above = entities.stratified_estimate(
        weights, pairs, math.nextafter(edge, math.inf), labeler, "rev", 6, 3
    )
    assert at.precision[0] != above.precision[0]
    assert at.recall[0] > above.recall[0]


def test_nothing_linked_gives_zero_precision_and_recall():
    weights, pairs, labeler = stratified_inputs()
    estimate = entities.stratified_estimate(
        weights, pairs, math.inf, labeler, "r", 6, 1
    )
    assert estimate.precision == (0.0, 0.0, 0.0)
    assert estimate.recall == (0.0, 0.0, 0.0)


def test_a_single_pair_is_fully_linked():
    estimate = entities.stratified_estimate(
        numpy.array([1.0]),
        [(0, 1)],
        -math.inf,
        lambda i, j: True,
        "r",
        3,
        1,
    )
    assert estimate.precision[0] == 0.75
    assert estimate.recall == (1.0, 1.0, 1.0)
    assert estimate.precision[1] == pytest.approx(0.15280215212003773)
    assert estimate.precision[2] == pytest.approx(0.999696770194363)
    assert band_rows(estimate) == [(1.0, 1.0, 1, 1, 1)]


def test_tied_weights_are_banded_in_input_order():
    weights = numpy.zeros(200)
    pairs = [(i, i + 1) for i in range(200)]
    estimate = entities.stratified_estimate(
        weights, pairs, 0.0, lambda i, j: i % 3 == 0, "r", 5, 2
    )
    assert [(b.pairs, b.sampled, int(b.matches)) for b in estimate.bands] == [
        (100, 5, 1),
        (50, 5, 2),
        (25, 5, 2),
        (13, 5, 2),
        (6, 5, 1),
        (3, 3, 1),
        (2, 2, 1),
        (1, 1, 0),
    ]
    assert estimate.precision == pytest.approx(
        (0.3277083333333334, 0.16429622176081246, 0.5315210019721369)
    )


def test_the_clerical_sample_depends_on_the_seed():
    weights, pairs, labeler = stratified_inputs()
    first = entities.stratified_estimate(
        weights, pairs, 0.5, labeler, "r", 6, 3
    )
    again = entities.stratified_estimate(
        weights, pairs, 0.5, labeler, "r", 6, 3
    )
    other = entities.stratified_estimate(
        weights, pairs, 0.5, labeler, "r", 6, 4
    )
    assert first == again
    assert first.precision != other.precision


def test_thresholds_scan_the_bands_from_the_top():
    bands = (
        entities.BandEstimate(-5.0, -1.0, 100, 10, 0),
        entities.BandEstimate(0.0, 2.0, 20, 10, 3),
        entities.BandEstimate(3.0, 5.0, 10, 10, 6),
        entities.BandEstimate(6.0, 9.0, 4, 4, 4),
    )
    assert entities.thresholds_from_bands(bands, 0.95) == (6.0, 3.0)
    assert entities.thresholds_from_bands(bands, 0.8) == (6.0, 3.0)
    assert entities.thresholds_from_bands(bands, 0.6) == (3.0, 3.0)
    assert entities.thresholds_from_bands(bands, 0.0) == (-5.0, 3.0)
    assert entities.thresholds_from_bands(bands, 1.0) == (6.0, 3.0)


def test_cumulative_precision_is_counted_in_pairs_not_in_bands():
    bands = (
        entities.BandEstimate(0.0, 1.0, 10, 10, 5),
        entities.BandEstimate(2.0, 3.0, 2, 2, 2),
    )
    assert entities.thresholds_from_bands(bands, 0.5) == (0.0, 0.0)
    assert entities.thresholds_from_bands(bands, 0.6) == (2.0, 0.0)
    assert entities.thresholds_from_bands(bands, 0.9) == (2.0, 0.0)
    assert entities.thresholds_from_bands(bands, 1.0) == (2.0, 0.0)


def test_a_band_with_no_sampled_pairs_has_a_zero_rate():
    bands = (entities.BandEstimate(1.0, 2.0, 5, 0, 0),)
    assert entities.thresholds_from_bands(bands, 0.5) == (math.inf, math.inf)
    assert entities.thresholds_from_bands(bands, 0.0) == (1.0, math.inf)
    assert entities.thresholds_from_bands((), 0.5) == (math.inf, math.inf)


def test_a_match_rate_of_exactly_one_half_makes_the_group_threshold():
    bands = (entities.BandEstimate(4.0, 5.0, 8, 2, 1),)
    assert entities.thresholds_from_bands(bands, 0.99) == (math.inf, 4.0)
    below = (entities.BandEstimate(4.0, 5.0, 8, 3, 1),)
    assert entities.thresholds_from_bands(below, 0.99) == (math.inf, math.inf)


def test_precision_exactly_at_the_target_is_enough():
    bands = (entities.BandEstimate(4.0, 5.0, 4, 4, 3),)
    assert entities.thresholds_from_bands(bands, 0.75) == (4.0, 4.0)
    assert entities.thresholds_from_bands(bands, 0.76) == (math.inf, 4.0)


def test_lowest_weight_of_nothing_is_infinite():
    weights = numpy.array([3.0, 1.0, 2.0])
    assert entities.lowest_weight(weights, weights > 1.5) == 2.0
    assert entities.lowest_weight(weights, weights > 5) == math.inf
    assert entities.lowest_weight(weights, weights >= 1.0) == 1.0


def test_resolution_without_labels_on_a_tiny_table_is_pinned():
    result = entities.resolve(SMALL, SMALL_FIELDS, SMALL_PASSES)
    assert result.groups == [0, 0, 2, 2, 4, 5, 4, 7]
    assert result.links == ((0, 1), (2, 3), (4, 6))
    assert result.t_group == pytest.approx(19.346605467484476)
    assert result.t_match == pytest.approx(19.346605467484476)
    assert result.compared == 5
    assert result.reduction_ratio == pytest.approx(1 - 5 / 28)
    assert result.skipped_blocks == 0
    assert result.estimate is None
    assert result.tier is qualification.Tier.UNVERIFIED
    assert result.model.iterations == 23
    assert result.model.prior == pytest.approx(0.6, abs=1e-5)


def test_resolution_with_labels_on_a_tiny_table_is_pinned():
    result = entities.resolve(
        SMALL,
        SMALL_FIELDS,
        SMALL_PASSES,
        labeler=small_labeler,
        reviewer="x",
        per_band=2,
        seed=1,
    )
    assert result.groups == [0, 0, 2, 2, 4, 4, 4, 7]
    assert result.links == ((0, 1), (2, 3), (4, 5), (4, 6), (5, 6))
    assert result.t_group == pytest.approx(-21.516529867789284)
    assert result.t_match == pytest.approx(-21.516529867789284)
    assert result.tier is qualification.Tier.ATTESTED
    assert result.estimate.reviewer == "x"
    assert result.estimate.precision == pytest.approx(
        (0.8, 0.48450644485082595, 0.9774811741358)
    )
    assert result.estimate.recall == pytest.approx((1.0, 1.0, 1.0))
    assert band_rows(result.estimate) == [
        (-21.5165, 19.3466, 3, 2, 2),
        (37.2782, 37.2782, 1, 1, 1),
        (38.2782, 38.2782, 1, 1, 1),
    ]


def test_to_dict_of_a_labelled_resolution_has_every_field():
    result = entities.resolve(
        SMALL,
        SMALL_FIELDS,
        SMALL_PASSES,
        labeler=small_labeler,
        reviewer="x",
        per_band=2,
        seed=1,
    )
    document = result.to_dict()
    assert set(document) == {
        "tier",
        "t_group",
        "t_match",
        "compared_pairs",
        "reduction_ratio",
        "skipped_blocks",
        "links",
        "groups",
        "records",
        "model",
        "estimate",
    }
    assert document["tier"] == "attested"
    assert document["t_group"] == result.t_group
    assert document["t_match"] == result.t_match
    assert document["compared_pairs"] == 5
    assert document["reduction_ratio"] == result.reduction_ratio
    assert document["skipped_blocks"] == 0
    assert document["links"] == 5
    assert document["groups"] == 4
    assert document["records"] == 8
    assert set(document["model"]) == {"prior", "m", "u", "iterations"}
    assert document["model"]["prior"] == result.model.prior
    assert document["model"]["m"] == result.model.m.tolist()
    assert document["model"]["u"] == result.model.u.tolist()
    assert document["model"]["iterations"] == result.model.iterations
    assert document["estimate"] == result.estimate.to_dict()


def test_to_dict_of_an_unlabelled_resolution_has_no_estimate():
    result = entities.resolve(SMALL, SMALL_FIELDS, SMALL_PASSES)
    document = result.to_dict()
    assert document["estimate"] is None
    assert document["tier"] == "unverified"
    assert document["links"] == 3
    assert document["groups"] == 5
    assert document["records"] == 8


def test_a_labeler_without_a_reviewer_is_refused_with_a_message():
    with pytest.raises(ValueError, match=r"^a labeler needs a named reviewer$"):
        entities.resolve(
            SMALL, SMALL_FIELDS, SMALL_PASSES, labeler=small_labeler
        )
    with pytest.raises(ValueError, match=r"^a labeler needs a named reviewer$"):
        entities.resolve(
            SMALL,
            SMALL_FIELDS,
            SMALL_PASSES,
            labeler=small_labeler,
            reviewer="   ",
        )


def test_without_blocked_pairs_every_record_is_alone_with_an_empty_model():
    frame = SMALL.copy()
    frame["uid"] = range(len(frame))
    result = entities.resolve(frame, SMALL_FIELDS, [entities.Pass("uid")])
    assert result.groups == list(range(8))
    assert result.links == ()
    assert result.t_group == math.inf
    assert result.t_match == math.inf
    assert result.compared == 0
    assert result.reduction_ratio == 1.0
    assert result.skipped_blocks == 0
    assert result.estimate is None
    assert result.tier is qualification.Tier.UNVERIFIED
    assert result.model.prior == 0.0
    assert result.model.iterations == 0
    assert result.model.m.tolist() == [[1.0, 1.0, 1.0]] * 3
    assert result.model.u.tolist() == [[1.0, 1.0, 1.0]] * 3
    assert result.to_dict()["groups"] == 8


def test_one_block_compares_every_pair_inside_it():
    frame = pandas.DataFrame({"k": ["a"] * 5, "v": list("abcde")})
    result = entities.resolve(
        frame, [entities.Field("v")], [entities.Pass("k")]
    )
    assert result.compared == 10
    assert result.reduction_ratio == 0.0


def test_an_empty_record_set_resolves_to_nothing():
    frame = pandas.DataFrame({"k": [], "v": []})
    result = entities.resolve(
        frame, [entities.Field("v")], [entities.Pass("k")]
    )
    assert result.groups == []
    assert result.reduction_ratio == 1.0
    assert result.compared == 0
    one = pandas.DataFrame({"k": ["a"], "v": ["x"]})
    single = entities.resolve(one, [entities.Field("v")], [entities.Pass("k")])
    assert single.groups == [0]
    assert single.reduction_ratio == 1.0


def test_oversized_blocks_are_reported_by_the_resolution():
    frame = pandas.DataFrame({"k": ["a"] * 201 + ["b", "b"], "v": ["x"] * 203})
    result = entities.resolve(
        frame, [entities.Field("v")], [entities.Pass("k")]
    )
    assert result.skipped_blocks == 1
    assert result.compared == 1
    assert result.reduction_ratio == pytest.approx(1 - 1 / (203 * 202 / 2))


def test_the_reduction_ratio_counts_all_unordered_pairs():
    frame = pandas.DataFrame({"k": ["a", "a", "b", "c"], "v": list("wxyz")})
    result = entities.resolve(
        frame, [entities.Field("v")], [entities.Pass("k")]
    )
    assert result.compared == 1
    assert result.reduction_ratio == pytest.approx(1 - 1 / 6)


@pytest.fixture(scope="module")
def world():
    return ingest_worlds.er_world(3, 200)


def test_unlabelled_thresholds_come_from_the_posterior(world):
    result = entities.resolve(world, FIELDS, PASSES)
    document = result.to_dict()
    assert document["tier"] == "unverified"
    assert result.t_group == pytest.approx(8.609669212785187)
    assert result.t_match == pytest.approx(9.651041949752166)
    assert document["compared_pairs"] == 2241
    assert document["reduction_ratio"] == pytest.approx(0.9483223797993774)
    assert document["links"] == 95
    assert document["groups"] == 200
    assert document["records"] == 295
    assert document["model"]["iterations"] == 15
    assert document["model"]["prior"] == pytest.approx(0.04229401343261634)


def test_a_lower_precision_target_lowers_the_match_threshold(world):
    result = entities.resolve(world, FIELDS, PASSES, min_precision=0.6)
    assert result.t_match == pytest.approx(8.609669212785187)
    assert result.t_group == pytest.approx(8.609669212785187)
    assert len(result.links) == 95
    strict = entities.resolve(world, FIELDS, PASSES, min_precision=0.999)
    assert strict.t_match >= result.t_match
    assert strict.t_group == result.t_group


def test_the_group_threshold_never_exceeds_the_match_threshold(world):
    for target in (0.1, 0.5, 0.9, 0.99):
        result = entities.resolve(world, FIELDS, PASSES, min_precision=target)
        assert result.t_group <= result.t_match


def test_labelled_resolution_of_a_world_is_pinned(world):
    truth = world["entity"].to_numpy()
    result = entities.resolve(
        world,
        FIELDS,
        PASSES,
        labeler=lambda i, j: bool(truth[i] == truth[j]),
        reviewer="rv",
        seed=2,
        per_band=20,
    )
    document = result.to_dict()
    assert document["tier"] == "attested"
    assert result.t_group == pytest.approx(28.54123776491317)
    assert result.t_match == pytest.approx(28.54123776491317)
    assert document["links"] == 75
    assert document["groups"] == 220
    assert result.estimate.precision == pytest.approx(
        (0.963471177944862, 0.9073940542063113, 0.9937461081026095)
    )
    assert result.estimate.recall == pytest.approx(
        (0.5091312429937044, 0.27013406700861425, 0.8050849466367271)
    )
    assert result.estimate.reviewer == "rv"
    assert len(result.estimate.bands) == 10


def test_the_seed_and_the_sample_size_reach_the_clerical_sample(world):
    truth = world["entity"].to_numpy()

    def labeler(i, j):
        return bool(truth[i] == truth[j])

    base = entities.resolve(
        world,
        FIELDS,
        PASSES,
        labeler=labeler,
        reviewer="r",
        seed=2,
        per_band=20,
    )
    other_seed = entities.resolve(
        world,
        FIELDS,
        PASSES,
        labeler=labeler,
        reviewer="r",
        seed=9,
        per_band=20,
    )
    smaller = entities.resolve(
        world, FIELDS, PASSES, labeler=labeler, reviewer="r", seed=2, per_band=5
    )
    assert other_seed.estimate.precision != base.estimate.precision
    assert smaller.estimate.precision != base.estimate.precision
    assert max(b.sampled for b in smaller.estimate.bands) == 5


def test_measure_quality_counts_pairs_inside_entities_and_groups():
    resolution = entities.Resolution(
        groups=[0, 0, 0, 3, 3, 5],
        links=(),
        t_group=0.0,
        t_match=0.0,
        model=entities.Model(0.0, numpy.ones((1, 3)), numpy.ones((1, 3)), 0),
        compared=0,
        reduction_ratio=1.0,
        skipped_blocks=0,
        estimate=None,
        tier=qualification.Tier.UNVERIFIED,
    )
    entity = ["a", "a", "b", "b", "b", "c"]
    pairs = [(0, 1), (2, 3), (3, 4), (0, 5)]
    quality = entities.measure_quality(entity, resolution, pairs)
    truth = {(0, 1), (2, 3), (2, 4), (3, 4)}
    assert quality.true_pairs == len(truth) == 4
    assert quality.grouped_pairs == 2
    assert quality.pair_completeness == pytest.approx(3 / 4)
    assert quality.link_recall == pytest.approx(2 / 4)
    assert quality.false_group_pairs == 2
    low, high = quality.link_recall_interval
    assert 0.0 < low < 0.5 < high < 1.0


def test_measure_quality_of_three_member_groups_counts_every_pair():
    resolution = entities.Resolution(
        [0, 0, 0, 0],
        (),
        0.0,
        0.0,
        entities.Model(0.0, numpy.ones((1, 3)), numpy.ones((1, 3)), 0),
        0,
        1.0,
        0,
        None,
        qualification.Tier.UNVERIFIED,
    )
    quality = entities.measure_quality(["a", "b", "c", "d"], resolution, [])
    assert quality.true_pairs == 0
    assert quality.false_group_pairs == 6
    triple = entities.measure_quality(
        ["a", "a", "a", "d"], resolution, [(0, 1)]
    )
    assert triple.true_pairs == 3
    assert triple.grouped_pairs == 3
    assert triple.false_group_pairs == 3
    assert triple.pair_completeness == pytest.approx(1 / 3)


def test_measure_quality_without_true_matches_is_perfect_by_convention():
    resolution = entities.Resolution(
        [0, 1, 2],
        (),
        0.0,
        0.0,
        entities.Model(0.0, numpy.ones((1, 3)), numpy.ones((1, 3)), 0),
        0,
        1.0,
        0,
        None,
        qualification.Tier.UNVERIFIED,
    )
    quality = entities.measure_quality(["a", "b", "c"], resolution, [])
    assert quality == entities.Quality(1.0, 1.0, (0.0, 1.0), 0, 0, 0)


def test_measure_quality_matches_the_resolution_of_a_labelled_run():
    result = entities.resolve(
        SMALL,
        SMALL_FIELDS,
        SMALL_PASSES,
        labeler=small_labeler,
        reviewer="x",
        per_band=2,
        seed=1,
    )
    pairs = entities.candidate_pairs(SMALL, SMALL_PASSES)[0]
    quality = entities.measure_quality(SMALL_ENTITY, result, pairs)
    assert quality.pair_completeness == 1.0
    assert quality.link_recall == 1.0
    assert quality.true_pairs == 5
    assert quality.grouped_pairs == 5
    assert quality.false_group_pairs == 0
    assert quality.link_recall_interval == pytest.approx(
        (0.5655175352168251, 1.0)
    )


def test_the_prior_is_held_inside_the_floor_at_both_ends():
    agree = entities.fit_em(numpy.full((6, 6), 2, dtype=numpy.int8))
    assert agree.prior == 1 - 1e-6
    disagree = entities.fit_em(numpy.zeros((6, 5), dtype=numpy.int8))
    assert disagree.prior == pytest.approx(1e-6, rel=1e-6)


STOPPING_CASES = [
    (
        "prior",
        [
            [2, 1, 2],
            [2, 2, 2],
            [2, 2, 2],
            [2, 2, 1],
            [2, 2, 1],
            [2, 2, 2],
            [2, -1, 2],
        ],
        1e-6,
        44,
    ),
    (
        "match",
        [[0, 0]] * 10
        + [[-1, 0], [1, 1]]
        + [[0, 0]] * 6
        + [[-1, 0], [0, -1], [0, -1], [0, 0], [0, -1], [0, -1]]
        + [[0, 0]] * 9,
        1e-6,
        7,
    ),
    (
        "other",
        [
            [v]
            for v in (
                *(2, -1, 2, 2, 2, 1, -1, 2, -1, 1, 1, 2, 2, 2, 1),
                *(2, -1, -1, 2, 2, 0, 2, 2, 2, 1, 1, 2, 1, -1, 2),
            )
        ],
        1e-3,
        4,
    ),
]


@pytest.mark.parametrize(
    ("name", "levels", "tol", "iterations"),
    STOPPING_CASES,
    ids=[case[0] for case in STOPPING_CASES],
)
def test_every_kind_of_parameter_movement_holds_the_fit_open(
    name, levels, tol, iterations
):
    model = entities.fit_em(numpy.array(levels, dtype=numpy.int8), tol=tol)
    assert model.iterations == iterations
