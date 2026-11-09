"""Subgroup hypotheses: registered with the claim, adjusted as one family."""

import numpy
import pandas

from scmle.data.charter import causal
from scmle.data.identification import run
from tests.unit.test_identification_run import block
from tests.unit.test_identification_run import BUDGET
from tests.unit.test_identification_run import GATES
from tests.unit.test_identification_run import NAMED


def world(n, seed, effects):
    """Randomised-looking world whose effect differs by `group`."""
    rng = numpy.random.default_rng(seed)
    a, b = rng.standard_normal(n), rng.standard_normal(n)
    t = (rng.uniform(size=n) < 0.5).astype(float)
    labels = numpy.array(sorted(effects))
    group = labels[rng.integers(0, len(labels), n)]
    tau = numpy.array([effects[g] for g in group])
    y = tau * t + a - 0.5 * b + rng.standard_normal(n)
    region = numpy.where(rng.uniform(size=n) < 0.5, "n", "s")
    nco = a + rng.standard_normal(n)
    return pandas.DataFrame(
        {
            "a": a,
            "b": b,
            "t": t,
            "y": y,
            "nco": nco,
            "region": region,
            "group": group,
        }
    )


def go(spec, x, c):
    return run.identify(
        spec, x, c, BUDGET, attestors=NAMED, certified=GATES, seed=0
    )


def subgroup_spec(levels):
    return block(subgroups=(causal.Subgroup("group", tuple(levels)),))


def test_the_effective_subgroup_is_rejected_and_the_others_are_not():
    effects = {"g1": 2.0, "g2": 0.0, "g3": 0.0}
    spec = subgroup_spec(effects)
    out = go(spec, world(1500, 1, effects), world(1500, 2, effects))
    found = out.documents["identification_report.json"]["subgroups"]
    assert found["rule"] == "holm" and found["family_size"] == 3
    rows = {r["value"]: r for r in found["hypotheses"]}
    assert rows["g1"]["rejected"] is True
    assert rows["g2"]["rejected"] is False and rows["g3"]["rejected"] is False
    assert rows["g1"]["low"] > 0
    assert found["rejected"] == ["group=g1"]


def test_holm_controls_the_family_where_unadjusted_tests_would_not():
    levels = [f"g{i:02d}" for i in range(20)]
    effects = dict.fromkeys(levels, 0.0)
    unadjusted = 0
    flagged = 0
    for seed in range(5):
        spec = subgroup_spec(levels)
        out = go(
            spec, world(2000, seed, effects), world(4000, 50 + seed, effects)
        )
        rows = out.documents["identification_report.json"]["subgroups"]
        unadjusted += sum(r["p_value"] < 0.05 for r in rows["hypotheses"])
        flagged += len(rows["rejected"])
    assert unadjusted >= 3
    assert flagged == 0


def test_a_subgroup_too_small_to_estimate_counts_in_the_family_unrejected():
    effects = {"g1": 2.0, "g2": 0.0}
    spec = subgroup_spec(("g1", "g2", "missing"))
    out = go(spec, world(1500, 3, effects), world(1500, 4, effects))
    found = out.documents["identification_report.json"]["subgroups"]
    rows = {r["value"]: r for r in found["hypotheses"]}
    assert found["family_size"] == 3
    assert rows["missing"]["skipped"] and rows["missing"]["rejected"] is False
    assert rows["missing"]["adjusted_p"] == 1.0


def test_subgroups_do_not_change_the_primary_estimate_or_the_tier():
    effects = {"g1": 2.0, "g2": 0.0}
    x, c = world(1500, 5, effects), world(1500, 6, effects)
    plain = go(block(), x, c)
    split = go(subgroup_spec(effects), x, c)
    assert (
        plain.documents["identification_report.json"]["estimate"]
        == split.documents["identification_report.json"]["estimate"]
    )
    assert (
        plain.documents["causal_claims.json"]["tier"]
        == split.documents["causal_claims.json"]["tier"]
    )
    claims = split.documents["causal_claims.json"]
    assert claims["subgroups"]["rejected"] == ["group=g1"]
    assert "subgroups" not in plain.documents["identification_report.json"]
    assert "subgroups" not in plain.documents["causal_claims.json"]
