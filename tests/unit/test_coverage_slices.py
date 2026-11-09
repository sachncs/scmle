"""Coverage and set size per declared slice, descriptive unless the flag is certified."""

import json

import numpy
import pandas

from scmle.evaluation import coverage_slices


def test_categorical_levels_are_the_labels_and_a_new_level_is_kept():
    reference = pandas.Series(["a", "b", "a", "b"])
    test = pandas.Series(["a", "c", "b"])
    labels = coverage_slices.labels_for(test, reference)
    assert list(labels) == ["a", "c", "b"]


def test_numeric_bands_come_from_the_reference_rows_only():
    reference = pandas.Series(numpy.arange(100.0))
    test = pandas.Series([0.0, 30.0, 60.0, 99.0, 500.0, -5.0])
    labels = coverage_slices.labels_for(test, reference)
    assert list(labels) == [
        "band 1",
        "band 2",
        "band 3",
        "band 4",
        "band 4",
        "band 1",
    ]


def test_too_many_levels_is_not_assessable():
    reference = pandas.Series([f"u{i}" for i in range(50)])
    assert coverage_slices.labels_for(reference, reference) is None


def descriptive_gate():
    return coverage_slices.FlagGate(coverage_slices.DESCRIPTIVE, "test")


def test_the_summary_gives_coverage_interval_size_and_a_descriptive_flag():
    rng = numpy.random.default_rng(0)
    labels = numpy.array(["x"] * 200 + ["y"] * 200 + ["z"] * 5)
    inside = rng.random(405) < 0.9
    sizes = numpy.where(labels == "x", 1.0, 2.0)

    found = coverage_slices.summary(
        inside, sizes, labels, 0.9, gate=descriptive_gate()
    )

    assert found["descriptive"] is True
    assert found["skipped_for_support"] == 1
    by_name = {s["slice"]: s for s in found["slices"]}
    assert set(by_name) == {"x", "y"}
    assert by_name["x"]["mean_size"] == 1.0 and by_name["y"]["mean_size"] == 2.0
    assert (
        by_name["x"]["lower"]
        <= by_name["x"]["coverage"]
        <= by_name["x"]["upper"]
    )
    assert found["flagged"] == []
    assert all(not s["flag_certified"] for s in found["slices"])
    assert found["flag_gate"]["status"] == "descriptive"


def test_a_badly_undercovered_slice_is_flagged_and_certified_only_when_gating():
    rng = numpy.random.default_rng(1)
    labels = numpy.array(["ok"] * 300 + ["bad"] * 300)
    inside = numpy.concatenate([rng.random(300) < 0.9, rng.random(300) < 0.5])
    sizes = numpy.ones(600)

    loose = coverage_slices.summary(
        inside, sizes, labels, 0.9, gate=descriptive_gate()
    )
    gated = coverage_slices.summary(
        inside,
        sizes,
        labels,
        0.9,
        gate=coverage_slices.FlagGate(coverage_slices.GATING, "test", 200),
    )

    assert loose["flagged"] == ["bad"] == gated["flagged"]
    assert loose["descriptive"] is True and gated["descriptive"] is False
    assert not any(s["flag_certified"] for s in loose["slices"])
    assert all(s["flag_certified"] for s in gated["slices"])


def test_a_slice_below_the_certified_size_is_not_certified():
    labels = numpy.array(["a"] * 60 + ["b"] * 300)
    inside = numpy.ones(360, dtype=bool)
    gate = coverage_slices.FlagGate(coverage_slices.GATING, "test", 100)
    found = coverage_slices.summary(
        inside, numpy.ones(360), labels, 0.9, gate=gate
    )
    by_name = {s["slice"]: s for s in found["slices"]}
    assert by_name["a"]["flag_certified"] is False
    assert by_name["b"]["flag_certified"] is True


def test_the_flag_gate_follows_the_stored_certificate(tmp_path):
    path = tmp_path / "c.json"
    assert coverage_slices.flag_gate(path).status == "descriptive"
    entry = {
        "method_hash": coverage_slices.flag_method_hash(),
        "passed": True,
        "min_certified_rows": 100,
    }
    path.write_text(json.dumps({"gates": {"G5b": entry}}))
    gate = coverage_slices.flag_gate(path)
    assert gate.status == "gating" and gate.min_rows == 100
    entry["passed"] = False
    path.write_text(json.dumps({"gates": {"G5b": entry}}))
    assert coverage_slices.flag_gate(path).status == "descriptive"
    entry.update(passed=True, method_hash="0" * 64)
    path.write_text(json.dumps({"gates": {"G5b": entry}}))
    assert "stale" in coverage_slices.flag_gate(path).reason


def test_infinite_sizes_give_no_mean():
    inside = numpy.ones(40, dtype=bool)
    sizes = numpy.full(40, numpy.inf)
    found = coverage_slices.summary(inside, sizes, numpy.array(["a"] * 40), 0.9)
    assert found["slices"][0]["mean_size"] is None
