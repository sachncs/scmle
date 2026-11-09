"""Unequal levels and overlapping columns of the subgroup family."""

import json

import numpy

from benchmarks.qualification import subgroup_extended


def stored() -> dict:
    return json.loads(
        subgroup_extended.MEASUREMENTS_PATH.read_text(encoding="utf-8")
    )


def test_the_unequal_world_has_the_declared_level_sizes():
    frame = subgroup_extended.unequal_world(
        numpy.random.default_rng(0),
        subgroup_extended.UNEQUAL_SIZES,
        {"g0": 1.0},
    )
    counts = frame["g"].value_counts().sort_index()
    assert list(counts.to_numpy()) == list(subgroup_extended.UNEQUAL_SIZES)
    assert set(frame["t"]) == {0.0, 1.0}


def test_the_overlapping_world_crosses_two_declared_columns():
    frame = subgroup_extended.overlap_world(numpy.random.default_rng(0), 0.5)
    assert len(frame) == subgroup_extended.OVERLAP_ROWS
    assert set(frame["g"]) == set(subgroup_extended.overlap_names("g"))
    assert set(frame["h"]) == set(subgroup_extended.overlap_names("h"))
    # the columns vary independently: every cell of the cross is occupied
    assert frame.groupby(["g", "h"]).size().shape[0] == 16


def test_a_small_run_returns_every_filed_number():
    document = subgroup_extended.measure_all(
        subgroup_extended.subgroup_certify.DEV_SEED_BASE, 0.02
    )
    unequal = document["unequal_levels"]
    assert set(unequal["power_at_effect_half_sd"]) == {
        "50",
        "100",
        "200",
        "400",
    }
    assert 0.0 <= unequal["familywise_error_null"]["rate"] <= 1.0
    assert (
        "familywise_error_true_nulls_with_effect_in_g0"
        in (document["overlapping_columns"])
    )


def test_the_stored_measurements_are_for_the_current_code():
    assert subgroup_extended.problems() == []


def test_the_stored_measurements_use_certification_seeds_and_report_passes():
    document = stored()
    assert document["seeds"][0] == 0
    assert (
        document["seeds"][1] < subgroup_extended.subgroup_certify.DEV_SEED_BASE
    )
    assert document["registered"] == "2026-11-07"
    for name in ("unequal_levels", "overlapping_columns"):
        assert document[name]["replicates"] == subgroup_extended.NULL_REPLICATES
        assert "passed" in document[name]


def test_a_changed_hash_is_reported():
    document = stored()
    document["code_hash"] = "0" * 64
    assert subgroup_extended.problems(document) == [
        "subgroup_extended: measurements are for other code"
    ]
