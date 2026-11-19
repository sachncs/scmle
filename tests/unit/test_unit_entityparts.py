"""Behavioural tests of the entity-resolution blocking, similarity, fitting
and sampling modules."""

import math

import numpy
import pandas
import pytest

from scmle.assurance.ingest.entities import blocking
from scmle.assurance.ingest.entities import fitting
from scmle.assurance.ingest.entities import sampling
from scmle.assurance.ingest.entities import similarity
from scmle.contracts import errors
from scmle.contracts import qualification

# ---- similarity -------------------------------------------------------------


def test_normalize_keeps_lower_case_alphanumerics():
    assert similarity.normalize("  O'Brien-Smith 2 ") == "obriensmith2"
    assert similarity.normalize(None) == ""
    assert similarity.normalize(float("nan")) == ""
    assert similarity.normalize(12.5) == "125"


def test_jaro_matches_the_textbook_values():
    assert similarity.jaro("martha", "marhta") == pytest.approx(
        0.944444, abs=1e-6
    )
    assert similarity.jaro("dixon", "dicksonx") == pytest.approx(
        0.766667, abs=1e-6
    )
    assert similarity.jaro("abc", "abc") == 1.0
    assert similarity.jaro("", "abc") == 0.0
    assert similarity.jaro("abc", "") == 0.0
    assert similarity.jaro("abc", "xyz") == 0.0
    assert similarity.jaro("", "") == 1.0


def test_jaro_uses_the_matching_window():
    # "a" at position 0 and 3 are outside the window of length 4 strings.
    assert similarity.jaro("abcd", "bcda") == pytest.approx(
        (0.75 + 0.75 + 1) / 3
    )
    assert similarity.jaro("ab", "ba") == 0.0


def test_jarowinkler_raises_the_score_for_a_common_prefix():
    assert similarity.jarowinkler("martha", "marhta") == pytest.approx(
        0.961111, abs=1e-6
    )
    assert similarity.jarowinkler("dixon", "dicksonx") == pytest.approx(
        0.813333, abs=1e-6
    )
    assert similarity.jarowinkler("abcdefgh", "abcdefgx") == pytest.approx(
        similarity.jaro("abcdefgh", "abcdefgx")
        + 0.4 * (1 - similarity.jaro("abcdefgh", "abcdefgx"))
    )
    assert similarity.jarowinkler("xa", "ya") == similarity.jaro("xa", "ya")


@pytest.mark.parametrize(
    ("text", "code"),
    [
        ("robert", "R163"),
        ("rupert", "R163"),
        ("rubin", "R150"),
        ("ashcraft", "A261"),
        ("tymczak", "T522"),
        ("pfister", "P236"),
        ("a", "A000"),
        ("lee", "L000"),
        ("12", ""),
        ("", ""),
    ],
)
def test_soundex_codes_the_standard_examples(text, code):
    assert similarity.soundex(text) == code


# ---- blocking ---------------------------------------------------------------


def test_a_field_defaults_to_fuzzy():
    assert blocking.Field("name").fuzzy is True
    assert blocking.MAX_BLOCK == 200


def test_pass_keys_by_kind():
    assert blocking.Pass("n").key("Ann-Marie") == "annmarie"
    assert blocking.Pass("n", "prefix", 2).key("Ann") == "an"
    assert blocking.Pass("n", "prefix").key("Annabel") == "ann"
    assert blocking.Pass("n", "soundex").key("Robert") == "R163"
    assert blocking.Pass("n", "soundex").key(None) == ""
    with pytest.raises(errors.InputError, match="unknown blocking kind: fuzzy"):
        blocking.Pass("n", "fuzzy").key("x")


def test_pairing_unions_passes_and_skips_big_blocks():
    frame = pandas.DataFrame(
        {
            "name": ["ann", "Ann", "bob", "bobby", None],
            "zip": ["1", "2", "2", "2", "2"],
        }
    )
    pairs, skipped = blocking.pairing(
        frame, [blocking.Pass("name"), blocking.Pass("name", "prefix")]
    )
    assert pairs == [(0, 1), (2, 3)]
    assert skipped == 0
    pairs, skipped = blocking.pairing(
        frame, [blocking.Pass("zip")], max_block=3
    )
    assert (pairs, skipped) == ([], 1)
    pairs, skipped = blocking.pairing(
        frame, [blocking.Pass("zip")], max_block=4
    )
    assert pairs == [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]


# ---- fitting ----------------------------------------------------------------


def test_the_levels_are_numbered():
    assert (
        fitting.MISSING,
        fitting.DISAGREE,
        fitting.SIMILAR,
        fitting.AGREE,
    ) == (
        -1,
        0,
        1,
        2,
    )
    assert fitting.LEVELS == 3
    assert fitting.SIMILAR_AT == 0.88


def test_model_weights_sum_log_ratios_and_skip_missing_fields():
    m = numpy.array([[0.1, 0.2, 0.7], [0.25, 0.25, 0.5]])
    u = numpy.array([[0.8, 0.1, 0.1], [0.5, 0.25, 0.25]])
    model = fitting.Model(0.1, m, u, 3)
    levels = numpy.array([[2, 2], [0, -1], [-1, -1], [1, 0]])
    out = model.weights(levels)
    assert out.tolist() == pytest.approx(
        [math.log2(7) + 1.0, math.log2(0.125), 0.0, 1.0 - 1.0]
    )


def test_em_separates_matches_from_non_matches():
    agree = numpy.full((20, 3), 2)
    disagree = numpy.zeros((180, 3), dtype=int)
    levels = numpy.vstack([agree, disagree])
    model = fitting.em(levels)
    assert model.prior == pytest.approx(0.1, abs=1e-3)
    assert model.m[:, 2] == pytest.approx([1.0, 1.0, 1.0], abs=1e-3)
    assert model.u[:, 0] == pytest.approx([1.0, 1.0, 1.0], abs=1e-3)
    assert model.m.min() >= 1e-6 and model.u.min() >= 1e-6
    assert 1 < model.iterations < 100
    weights = model.weights(levels)
    assert weights[:20].min() > 0 > weights[20:].max()


def test_em_stops_at_the_iteration_cap():
    levels = numpy.array([[2], [0], [1], [-1]])
    assert fitting.em(levels, iterations=1).iterations == 1
    assert fitting.em(levels, iterations=2, tol=10.0).iterations == 1


def test_components_name_each_group_by_its_smallest_member():
    assert fitting.components(6, [(4, 1), (1, 3), (5, 2)]) == [0, 1, 2, 1, 1, 2]
    assert fitting.components(3, []) == [0, 1, 2]
    assert fitting.components(4, [(3, 2), (2, 1), (1, 0)]) == [0, 0, 0, 0]


# ---- sampling ---------------------------------------------------------------


def test_stratify_thins_bands_toward_the_top():
    out = sampling.stratify(numpy.arange(16), 4)
    assert [len(b) for b in out] == [8, 4, 2, 2]
    assert out[-1].tolist() == [14, 15]
    assert [len(b) for b in sampling.stratify(numpy.arange(3), 4)] == [2, 1]
    assert sampling.stratify(numpy.arange(0), 3) == []
    assert [len(b) for b in sampling.stratify(numpy.arange(5), 1)] == [5]


def test_band_and_estimate_serialise():
    band = sampling.Band(0.0, 1.0, 10, 5, 2)
    assert band.to_dict() == {
        "low": 0.0,
        "high": 1.0,
        "pairs": 10,
        "sampled": 5,
        "matches": 2,
    }
    estimate = sampling.Estimate(
        (0.9, 0.8, 1.0), (0.5, 0.4, 0.6), (band,), "ann"
    )
    assert estimate.to_dict() == {
        "precision": [0.9, 0.8, 1.0],
        "recall": [0.5, 0.4, 0.6],
        "bands": [band.to_dict()],
        "reviewer": "ann",
    }


def test_weakest_is_the_smallest_selected_weight():
    weights = numpy.array([3.0, 1.0, 2.0])
    assert sampling.weakest(weights, numpy.array([True, False, True])) == 2.0
    assert sampling.weakest(weights, numpy.zeros(3, dtype=bool)) == math.inf


def test_cutoffs_scan_from_the_top():
    bands = [
        sampling.Band(0.0, 1.0, 100, 10, 0),
        sampling.Band(1.0, 2.0, 10, 10, 5),
        sampling.Band(2.0, 3.0, 10, 10, 10),
    ]
    assert sampling.cutoffs(bands, 0.9) == (2.0, 1.0)
    assert sampling.cutoffs(bands, 0.75) == (1.0, 1.0)
    assert sampling.cutoffs(bands, 0.0) == (0.0, 1.0)
    empty = [sampling.Band(0.0, 1.0, 5, 0, 0)]
    assert sampling.cutoffs(empty, 0.5) == (math.inf, math.inf)


def test_infer_point_estimates_use_the_jeffreys_rate():
    rng = numpy.random.default_rng(0)
    bands = [sampling.Band(0, 1, 10, 4, 0), sampling.Band(1, 2, 10, 4, 4)]
    out = sampling.infer(
        ([10, 10], [0, 4], [4, 4]), [False, True], bands, "r", 500, rng
    )
    point_low = 0.5 / 5.0 * 10
    point_high = 4.5 / 5.0 * 10
    assert out.precision[0] == pytest.approx(point_high / 10)
    assert out.recall[0] == pytest.approx(point_high / (point_low + point_high))
    assert out.precision[1] <= out.precision[0] <= out.precision[2]
    assert out.bands == tuple(bands) and out.reviewer == "r"
    none = sampling.infer(([10], [0], [4]), [False], bands[:1], "r", 10, rng)
    assert none.precision == (0.0, 0.0, 0.0)


def test_stratified_estimate_labels_each_band():
    weights = numpy.arange(16.0)
    pairs = [(i, i + 1) for i in range(16)]
    calls = []

    def labeler(a, b):
        calls.append((a, b))
        return a >= 12

    out = sampling.appraise(weights, pairs, 12.0, labeler, "ann", 2, 0, bands=4)
    assert [b.pairs for b in out.bands] == [8, 4, 2, 2]
    assert [b.sampled for b in out.bands] == [2, 2, 2, 2]
    assert [b.matches for b in out.bands] == [0, 0, 2, 2]
    assert out.bands[2].low == 12.0 and out.bands[2].high == 13.0
    assert len(calls) == 8
    assert out.reviewer == "ann"


def test_calibrate_without_a_labeler_uses_the_posterior():
    weights = numpy.array([-10.0, 0.0, 10.0, 20.0])
    pairs = [(0, 1)] * 4
    t_match, t_group, estimate, tier = sampling.calibrate(
        weights, pairs, 0.5, None, "", 2, 0.99, 0
    )
    assert (t_match, t_group) == (10.0, 0.0)
    assert estimate is None
    assert tier is qualification.Tier.UNVERIFIED


def test_calibrate_with_a_labeler_is_attested():
    weights = numpy.arange(16.0)
    pairs = [(i, i) for i in range(16)]
    t_match, t_group, estimate, tier = sampling.calibrate(
        weights, pairs, 0.1, lambda a, b: a >= 14, "ann", 4, 0.9, 0
    )
    assert tier is qualification.Tier.ATTESTED
    assert t_match == 14.0 and t_group == 14.0
    # ten bands of 16 pairs collapse to sizes 8, 4, 2, 1, 1
    assert [b.pairs for b in estimate.bands] == [8, 4, 2, 1, 1]
    assert estimate.precision[0] == pytest.approx(0.75)
    assert estimate.recall[0] == pytest.approx(
        1.5 / (0.5 / 5 * 8 + 0.5 / 5 * 4 + 0.5 / 3 * 2 + 1.5)
    )
