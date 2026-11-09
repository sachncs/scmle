"""Entity resolution: blocking, EM, clerical sample, link-then-group."""

import numpy
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


def test_string_comparators_against_known_values():
    assert entities.jaro("martha", "marhta") == pytest.approx(0.9444, abs=1e-4)
    assert entities.jaro_winkler("martha", "marhta") == pytest.approx(
        0.9611, abs=1e-4
    )
    assert entities.jaro_winkler("abc", "abc") == 1.0
    assert entities.jaro("", "a") == 0.0
    assert entities.soundex("robert") == "R163"
    assert entities.soundex("rupert") == "R163"
    assert entities.soundex("ashcraft") == "A261"
    assert entities.soundex("123") == ""


def test_blocking_passes_union_and_skip_oversized_blocks():
    frame = ingest_worlds.er_world(1, 40)
    pairs, skipped = entities.candidate_pairs(frame, PASSES)
    assert pairs == sorted(set(pairs))
    assert skipped == 0
    coarse = entities.candidate_pairs(
        frame, [entities.Pass("dob", "prefix", 1)], max_block=3
    )
    assert coarse[1] > 0


def test_resolution_with_labels_reports_pair_completeness_and_recall():
    frame = ingest_worlds.er_world(2, 250)
    truth = frame["entity"].to_numpy()
    result = entities.resolve(
        frame,
        FIELDS,
        PASSES,
        labeler=lambda i, j: bool(truth[i] == truth[j]),
        reviewer="benchmark truth",
        seed=2,
    )
    pairs = entities.candidate_pairs(frame, PASSES)[0]
    quality = entities.measure_quality(truth, result, pairs)
    assert result.tier is qualification.Tier.ATTESTED
    assert quality.pair_completeness >= 0.95
    assert quality.link_recall >= 0.9
    assert result.reduction_ratio > 0.8
    assert result.estimate is not None
    low, high = result.estimate.precision[1:]
    assert 0.0 <= low <= high <= 1.0
    assert result.t_group <= result.t_match


def test_unlabelled_resolution_is_unverified_and_groups_more_not_less():
    frame = ingest_worlds.er_world(3, 200)
    truth = frame["entity"].to_numpy()
    result = entities.resolve(frame, FIELDS, PASSES)
    assert result.tier is qualification.Tier.UNVERIFIED
    assert result.estimate is None
    pairs = entities.candidate_pairs(frame, PASSES)[0]
    assert entities.measure_quality(truth, result, pairs).link_recall >= 0.8


def test_resolution_is_deterministic():
    frame = ingest_worlds.er_world(4, 120)
    first = entities.resolve(frame, FIELDS, PASSES)
    second = entities.resolve(frame, FIELDS, PASSES)
    assert first.groups == second.groups
    assert first.t_group == second.t_group


def test_labeler_needs_a_named_reviewer():
    frame = ingest_worlds.er_world(5, 30)
    with pytest.raises(ValueError):
        entities.resolve(frame, FIELDS, PASSES, labeler=lambda i, j: True)


def test_no_blocked_pairs_gives_singletons():
    frame = ingest_worlds.er_world(6, 20)
    frame["uid"] = range(len(frame))
    result = entities.resolve(frame, FIELDS, [entities.Pass("uid")])
    assert result.groups == list(range(len(frame)))
    assert result.tier is qualification.Tier.UNVERIFIED


def test_em_recovers_separated_classes():
    rng = numpy.random.default_rng(0)
    match = rng.random(4000) < 0.1
    levels = numpy.where(
        match[:, None],
        rng.choice([0, 1, 2], (4000, 3), p=[0.05, 0.15, 0.8]),
        rng.choice([0, 1, 2], (4000, 3), p=[0.9, 0.07, 0.03]),
    ).astype(numpy.int8)
    model = entities.fit_em(levels)
    weights = model.weights(levels)
    assert model.prior == pytest.approx(0.1, abs=0.03)
    assert weights[match].mean() > weights[~match].mean() + 5


def test_components_link_transitively():
    assert entities.components(5, [(0, 1), (1, 2), (3, 4)]) == [0, 0, 0, 3, 3]
