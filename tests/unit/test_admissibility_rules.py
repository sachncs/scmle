"""Admissibility by estimand type, tier and mode."""

import datetime

import pytest

from scmle.assurance.admissibility import dag
from scmle.assurance.admissibility import provenance
from scmle.assurance.admissibility import rules
from scmle.assurance.detectors import certificates
from scmle.contracts.qualification import Tier
from tests.unit.test_admissibility_provenance import attestation
from tests.unit.test_admissibility_provenance import make_field

DAY = datetime.date(2026, 6, 15)
A, Q, F = (
    rules.Admission.ADMIT,
    rules.Admission.QUARANTINE,
    rules.Admission.ADMIT_FLAGGED,
)


def unverified(name="u"):
    return provenance.undeclared(name)


def test_a_planted_mediator_known_before_the_moment_is_admitted():
    # T -> M -> Y: M is a mediator, known an hour before the moment.
    graph = dag.Dag.build([("T", "M"), ("M", "Y")], "expert")
    table = provenance.Dictionary(
        (make_field("M", lead=3600.0), make_field("T", lead=7200.0))
    )
    out = rules.build(table, "predictive", graph=graph)
    assert out.names(A) == ("M", "T")
    assert out.dag_hash is None  # a pure prediction task never reads roles


def test_a_planted_post_outcome_field_is_quarantined():
    table = provenance.Dictionary((make_field("after", lead=-60.0),))
    out = rules.build(table, "predictive", mode=rules.Mode.PERMISSIVE)
    assert out.names(Q) == ("after",)
    assert "less than the declared" in out.decisions[0].reasons[0]


def test_the_declared_lag_boundary_is_inclusive():
    on = rules.forecastable(make_field(lead=100.0), declared_lag_seconds=100.0)
    below = rules.forecastable(
        make_field(lead=99.999), declared_lag_seconds=100.0
    )
    assert on.admission is A
    assert below.admission is Q
    zero = rules.forecastable(make_field(lead=0.0))
    assert zero.admission is A  # known exactly at the moment, no lag declared
    assert rules.forecastable(make_field(lead=-1e-9)).admission is Q


def test_a_negative_or_nonfinite_lag_is_rejected():
    with pytest.raises(ValueError):
        rules.forecastable(make_field(), declared_lag_seconds=-1.0)
    with pytest.raises(ValueError):
        rules.forecastable(make_field(), declared_lag_seconds=float("inf"))


def test_not_available_at_serving_is_quarantined_in_every_mode():
    record = make_field(serving=provenance.Availability.NO)
    for mode in rules.Mode:
        d = rules.forecastable(record, mode)
        assert d.admission is Q and "serving" in d.reasons[0]


def test_unknown_serving_availability_follows_the_mode():
    record = make_field(serving=provenance.Availability.UNKNOWN)
    assert rules.forecastable(record, rules.Mode.STRICT).admission is Q
    assert rules.forecastable(record, rules.Mode.PERMISSIVE).admission is F


def test_strict_quarantines_every_unverified_field_permissive_flags_it():
    # Enforces docs/READING_RESULTS.md:374.
    # Enforces website/src/learn/reading-results.html:199.
    table = provenance.Dictionary.covering(["a", "b"], [make_field("b")])
    strict = rules.build(table, "predictive", rules.Mode.STRICT, today=DAY)
    loose = rules.build(table, "predictive", rules.Mode.PERMISSIVE, today=DAY)
    assert strict.names(Q) == ("a",) and strict.names(A) == ("b",)
    assert loose.names(F) == ("a",) and loose.names(A) == ("b",)
    for d in strict.decisions + loose.decisions:
        assert d.tier is (Tier.UNVERIFIED if d.name == "a" else Tier.ATTESTED)


def test_expired_attestation_is_treated_as_unverified():
    record = make_field(att=attestation(expires="2026-06-14"))
    assert rules.forecastable(record, today=DAY).admission is Q
    assert rules.forecastable(record, today=None).admission is A


def test_a_log_that_disagrees_quarantines_even_a_permissive_run():
    record = make_field(
        source=provenance.Provenance.LOG,
        kind=provenance.Cutoff.TIMESTAMP,
        log=provenance.Reconstruction(100, 3, 0.0),
        att=attestation(),
    )
    d = rules.forecastable(record, rules.Mode.PERMISSIVE, today=DAY)
    assert d.admission is Q and "change log disagrees" in d.reasons[0]


def test_suspicion_without_as_of_evidence_quarantines_only_when_gating():
    # Enforces docs/READING_RESULTS.md:374.
    # Enforces website/src/learn/reading-results.html:199.
    record = unverified()
    off = rules.forecastable(record, rules.Mode.PERMISSIVE, suspect=True)
    on = rules.forecastable(
        record,
        rules.Mode.PERMISSIVE,
        suspect=True,
        leakage_gate=certificates.Authority.GATING,
    )
    assert off.admission is F and on.admission is Q
    assert off.suspected and on.suspected


def test_a_suspect_with_log_evidence_is_admitted_and_the_suspected():
    record = make_field(
        source=provenance.Provenance.LOG, kind=provenance.Cutoff.TIMESTAMP
    )
    d = rules.forecastable(
        record, suspect=True, leakage_gate=certificates.Authority.GATING
    )
    assert d.admission is A and d.suspected and d.tier is Tier.CONFIRMED
    assert any("suspicion kept" in x for x in d.reasons)


def test_a_suspect_with_only_an_attestation_is_flagged():
    d = rules.forecastable(make_field(), today=DAY, suspect=True)
    assert d.admission is F and d.tier is Tier.ATTESTED


def test_no_suspicion_no_flag():
    d = rules.forecastable(make_field(), today=DAY)
    assert d.admission is A and not d.suspected


# T -> M -> Y mediator, C -> T and C -> Y confounder,
# U1 -> T, U1 -> K <- U2, U2 -> Y with K a collider (U latent), P -> Y a pure
# predictor, D descends from T.
GRAPH = dag.Dag.build(
    [
        ("C", "T"),
        ("C", "Y"),
        ("T", "M"),
        ("M", "Y"),
        ("T", "Y"),
        ("U1", "T"),
        ("U1", "K"),
        ("U2", "K"),
        ("U2", "Y"),
        ("P", "Y"),
        ("T", "D"),
    ],
    "expert",
    latent=["U1", "U2"],
)


def causal(name, graph=GRAPH, mode=rules.Mode.STRICT, lead=3600.0):
    return rules.adjudicate(
        make_field(name, lead=lead), graph, "T", "Y", mode, today=DAY
    )


def test_a_planted_collider_is_excluded_from_the_adjustment_set():
    d = causal("K")
    assert d.admission is Q and "collider" in d.reasons[-1]


def test_confounder_and_pure_predictor_are_admitted_mediator_and_descendant_not():
    assert causal("C").admission is A
    assert causal("P").admission is A
    assert causal("M").admission is Q
    assert causal("D").admission is Q
    assert "descendant" in causal("M").reasons[-1]


def test_treatment_and_outcome_are_decided_on_timing_alone():
    for name in ("T", "Y"):
        d = causal(name)
        assert d.admission is A and d.reasons[-1] == "treatment or outcome node"


def test_a_field_outside_the_dag_follows_the_mode():
    assert causal("Z").admission is Q
    assert causal("Z", mode=rules.Mode.PERMISSIVE).admission is F
    assert "not in the declared DAG" in causal("Z").reasons[-1]


def test_timing_quarantine_wins_before_the_graph_is_consulted():
    d = causal("C", lead=-5.0)
    assert d.admission is Q and "declared" in d.reasons[0]


def test_without_a_valid_adjustment_set_covariates_are_flagged():
    graph = dag.Dag.build(
        [("U", "T"), ("U", "Y"), ("T", "Y"), ("P", "Y")], "e", latent=["U"]
    )
    d = causal("P", graph)
    assert d.admission is F and "no valid adjustment set" in d.reasons[-1]


def test_causal_table_records_the_graph_hash_and_set():
    fields = [make_field(n) for n in ("C", "K", "M", "P", "T", "Y")]
    out = rules.build(
        provenance.Dictionary(tuple(fields)),
        "causal",
        graph=GRAPH,
        treatment="T",
        outcome="Y",
        today=DAY,
    )
    assert out.dag_hash == GRAPH.digest()
    assert out.adjustment == GRAPH.deconfound("T", "Y")
    assert out.names(Q) == ("K", "M")
    assert out.to_dict()["kind"] == "causal"


def test_arguments_of_the_table_are_validated():
    table = provenance.Dictionary((make_field("a"),))
    with pytest.raises(ValueError, match="kind"):
        rules.build(table, "descriptive")
    with pytest.raises(ValueError, match="graph, treatment, outcome"):
        rules.build(table, "causal")
    d = rules.forecastable(make_field("a"))
    with pytest.raises(ValueError, match="unique"):
        rules.Admissibility("predictive", rules.Mode.STRICT, (d, d), None, None)
    with pytest.raises(ValueError, match="kind"):
        rules.Admissibility("x", rules.Mode.STRICT, (), None, None)


def test_decisions_round_trip():
    d = rules.forecastable(make_field(), today=DAY, suspect=True)
    assert rules.Decision.from_dict(d.to_dict()) == d
    with pytest.raises(ValueError, match="unexpected"):
        rules.Decision.from_dict({**d.to_dict(), "x": 1})


def test_a_gate_per_field_blocks_only_the_field_it_names():
    table = provenance.Dictionary.covering(["a", "b"])
    out = rules.build(
        table,
        "predictive",
        rules.Mode.PERMISSIVE,
        suspects={"a": True, "b": True},
        leakage_gate={"a": certificates.Authority.GATING},
    )
    assert out.names(Q) == ("a",)
    assert out.names(F) == ("b",)
    assert all(d.suspected for d in out.decisions)
