"""Unequal levels and overlapping columns of the subgroup family."""

import json

import numpy as np

from benchmarks.qualification import subgroup_extended as ext


def stored() -> dict:
    return json.loads(ext.MEASUREMENTS_PATH.read_text(encoding="utf-8"))


def test_the_unequal_world_has_the_declared_level_sizes():
    frame = ext.unequal_world(
        np.random.default_rng(0), ext.UNEQUAL_SIZES, {"g0": 1.0}
    )
    counts = frame["g"].value_counts().sort_index()
    assert list(counts.to_numpy()) == list(ext.UNEQUAL_SIZES)
    assert set(frame["t"]) == {0.0, 1.0}


def test_the_overlapping_world_crosses_two_declared_columns():
    frame = ext.overlap_world(np.random.default_rng(0), 0.5)
    assert len(frame) == ext.OVERLAP_ROWS
    assert set(frame["g"]) == set(ext.overlap_names("g"))
    assert set(frame["h"]) == set(ext.overlap_names("h"))
    # the columns vary independently: every cell of the cross is occupied
    assert frame.groupby(["g", "h"]).size().shape[0] == 16


def test_a_small_run_returns_every_filed_number():
    document = ext.measure_all(ext.base_gate.DEV_SEED_BASE, 0.02)
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
    assert ext.problems() == []


def test_the_stored_measurements_use_certification_seeds_and_report_passes():
    document = stored()
    assert document["seeds"][0] == 0
    assert document["seeds"][1] < ext.base_gate.DEV_SEED_BASE
    assert document["registered"] == "2026-11-07"
    for name in ("unequal_levels", "overlapping_columns"):
        assert document[name]["replicates"] == ext.NULL_REPLICATES
        assert "passed" in document[name]


def test_a_changed_hash_is_reported():
    document = stored()
    document["code_hash"] = "0" * 64
    assert ext.problems(document) == [
        "subgroup_extended: measurements are for other code"
    ]
