"""The filed analysis of the CQR conditional-coverage miss: shape, seeds and what it does not do."""

import json

import numpy

from benchmarks.qualification import cqr_analysis
from benchmarks.qualification import uncertainty_certify


def stored() -> dict:
    return json.loads(cqr_analysis.OUTPUT_PATH.read_text(encoding="utf-8"))


def test_the_analysis_uses_development_seeds_and_every_variant():
    document = stored()
    assert document["seeds"][0] == uncertainty_certify.DEV_SEED_BASE
    assert set(document["variants"]) == {
        v["name"] for v in cqr_analysis.VARIANTS
    }
    assert document["replicates"] == cqr_analysis.REPLICATES
    assert document["registered"] == "2026-11-07"


def test_the_analysis_changes_no_certificate():
    certificate = json.loads(
        uncertainty_certify.STORE_PATH.read_text(encoding="utf-8")
    )
    entry = certificate["gates"]["G3"]
    assert entry["passed"] is False and entry["replicates"] == 150


def test_a_replicate_returns_three_coverages_in_the_unit_interval():
    values = cqr_analysis.replicate(0, uncertainty_certify.DEV_SEED_BASE, 20, 2)
    assert len(values) == 3
    assert all(0.0 <= v <= 1.0 for v in values)
    assert numpy.isclose(values[0], 0.9, atol=0.1)


def test_summaries_carry_a_standard_error():
    found = cqr_analysis.summarize(numpy.array([0.8, 0.9, 1.0]))
    assert found["mean"] == 0.9 and found["se"] > 0
