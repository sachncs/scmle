"""The filed analysis of the G3 miss: shape, seeds and what it does not do."""

import json

import numpy as np

from benchmarks.qualification import cqr_analysis as analysis
from benchmarks.qualification import uncertainty_certify as cert


def stored() -> dict:
    return json.loads(analysis.OUTPUT_PATH.read_text(encoding="utf-8"))


def test_the_analysis_uses_development_seeds_and_every_variant():
    document = stored()
    assert document["seeds"][0] == cert.DEV_SEED_BASE
    assert set(document["variants"]) == {v["name"] for v in analysis.VARIANTS}
    assert document["replicates"] == analysis.REPLICATES
    assert document["registered"] == "2026-11-07"


def test_the_analysis_changes_no_certificate():
    certificate = json.loads(cert.STORE_PATH.read_text(encoding="utf-8"))
    entry = certificate["gates"]["G3"]
    assert entry["passed"] is False and entry["replicates"] == 150


def test_a_replicate_returns_three_coverages_in_the_unit_interval():
    values = analysis.replicate(0, cert.DEV_SEED_BASE, 20, 2)
    assert len(values) == 3
    assert all(0.0 <= v <= 1.0 for v in values)
    assert np.isclose(values[0], 0.9, atol=0.1)


def test_summaries_carry_a_standard_error():
    found = analysis.summarize(np.array([0.8, 0.9, 1.0]))
    assert found["mean"] == 0.9 and found["se"] > 0
